"""Fail-closed canonical pilot runner for an approved positive-only manifest."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dataset_validation.qualification import qualify_sample, validate_review_decision
from pipeline.canonical_adapter import dataset_sha256, load_and_adapt_dataset
from pipeline.context_policy import FULL_BLOBS, ContextPolicy, prepare_context
from pipeline.evaluator import evaluate_positive_only_ablation
from pipeline.llm_client import StructuredLLMClient
from pipeline.offline_integration import prompt_hashes, render_static_prompt_plan, run_canonical_sample
from pipeline.run_state import RunWorkspace, build_run_spec, hash_files, hash_value


PILOT_RUNNER_VERSION = "phase3d-canonical-pilot-runner-1.1"
PIPELINES = ("P0", "P1", "P2")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


class PilotRunnerError(ValueError):
    pass


LEGACY_PACKAGE_FILENAMES = {
    "dataset": "canonical_dataset.json",
    "qualification": "qualification_record.json",
    "decision": "review_decision.json",
    "manifest": "official_pilot_manifest.json",
    "receipt": "package_receipt.json",
}

MULTI_PACKAGE_FILENAMES = {
    "dataset": "canonical_dataset.json",
    "qualification": "qualification_records.json",
    "decision": "review_decisions.json",
    "manifest": "official_pilot_manifest.json",
    "receipt": "package_receipt.json",
}

# Kept for callers that only need the common dataset filename.
PACKAGE_FILENAMES = LEGACY_PACKAGE_FILENAMES


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotRunnerError(f"{label}_INVALID:{exc}") from exc
    if not isinstance(value, dict):
        raise PilotRunnerError(f"{label}_NOT_OBJECT")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_manifest(
    manifest_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    """Validate the minimum official-pilot contract and exact dataset bytes."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotRunnerError(f"OFFICIAL_MANIFEST_INVALID:{exc}") from exc
    if not isinstance(manifest, dict):
        raise PilotRunnerError("OFFICIAL_MANIFEST_NOT_OBJECT")
    if manifest.get("official_dataset") is not True:
        raise PilotRunnerError("OFFICIAL_DATASET_APPROVAL_REQUIRED")
    if manifest.get("purpose") not in {"PILOT", "OFFICIAL_PILOT"}:
        raise PilotRunnerError("OFFICIAL_PILOT_PURPOSE_REQUIRED")
    ids = manifest.get("ordered_sample_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        raise PilotRunnerError("ORDERED_SAMPLE_LIST_INVALID")
    if len(ids) != len(set(ids)):
        raise PilotRunnerError("DUPLICATE_SAMPLE_ID")
    if manifest.get("sample_list_sha256") != hash_value(ids):
        raise PilotRunnerError("SAMPLE_LIST_HASH_MISMATCH")
    actual_dataset_hash = dataset_sha256(Path(dataset_path))
    if manifest.get("dataset_sha256") != actual_dataset_hash:
        raise PilotRunnerError("DATASET_HASH_MISMATCH")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise PilotRunnerError("MANIFEST_SAMPLE_RECORDS_MISSING")
    by_id = {item.get("sample_id"): item for item in samples if isinstance(item, dict)}
    if set(by_id) != set(ids) or len(by_id) != len(samples):
        raise PilotRunnerError("MANIFEST_SAMPLE_RECORDS_MISMATCH")
    not_ready = [sample_id for sample_id in ids if by_id[sample_id].get("qualification_status") != "READY"]
    if not_ready:
        raise PilotRunnerError(f"PILOT_HAS_NON_READY_SAMPLES:{not_ready}")
    return manifest


def load_official_package(
    package_dir: Path,
    schema_path: Path,
    *,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Validate all five review-gated package artifacts without model calls."""
    package_dir = Path(package_dir)
    legacy_present = all((package_dir / name).is_file() for name in LEGACY_PACKAGE_FILENAMES.values())
    multi_present = all((package_dir / name).is_file() for name in MULTI_PACKAGE_FILENAMES.values())
    if legacy_present and multi_present:
        raise PilotRunnerError("AMBIGUOUS_PACKAGE_FORMAT")
    if multi_present:
        filenames = MULTI_PACKAGE_FILENAMES
        package_format = "MULTI_SAMPLE_REVIEW_RECORDS_V1"
    else:
        filenames = LEGACY_PACKAGE_FILENAMES
        package_format = "LEGACY_ONE_SAMPLE_V1"
    paths = {key: package_dir / name for key, name in filenames.items()}
    missing = [name for key, name in filenames.items() if not paths[key].is_file()]
    if missing:
        raise PilotRunnerError(f"OFFICIAL_PACKAGE_FILES_MISSING:{missing}")
    if not SHA40.fullmatch(expected_git_commit):
        raise PilotRunnerError("FULL_GIT_COMMIT_REQUIRED")

    manifest = load_official_manifest(paths["manifest"], paths["dataset"])
    manifest_base = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != hash_value(manifest_base):
        raise PilotRunnerError("MANIFEST_SELF_HASH_MISMATCH")
    if manifest.get("dataset_file") != filenames["dataset"]:
        raise PilotRunnerError("MANIFEST_DATASET_FILENAME_MISMATCH")
    if manifest.get("git_commit") != expected_git_commit:
        raise PilotRunnerError("MANIFEST_GIT_COMMIT_MISMATCH")
    if manifest.get("schema_sha256") != _sha256_file(Path(schema_path)):
        raise PilotRunnerError("SCHEMA_HASH_MISMATCH")

    qualification_payload = _read_json_object(paths["qualification"], "QUALIFICATION_RECORD")
    decision_payload = _read_json_object(paths["decision"], "REVIEW_DECISION")
    receipt = _read_json_object(paths["receipt"], "PACKAGE_RECEIPT")
    dataset = _read_json_object(paths["dataset"], "CANONICAL_DATASET")
    raw_samples = dataset.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples or any(
        not isinstance(sample, dict) for sample in raw_samples
    ):
        raise PilotRunnerError("CANONICAL_SAMPLES_INVALID")
    sample_ids = [sample.get("sample_id") for sample in raw_samples]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise PilotRunnerError("CANONICAL_SAMPLE_ID_INVALID")
    if len(sample_ids) != len(set(sample_ids)):
        raise PilotRunnerError("DUPLICATE_CANONICAL_SAMPLE_ID")
    if manifest.get("ordered_sample_ids") != sample_ids:
        raise PilotRunnerError("PACKAGE_SAMPLE_ID_MISMATCH")

    if package_format == "MULTI_SAMPLE_REVIEW_RECORDS_V1":
        qualifications = qualification_payload.get("records")
        decisions = decision_payload.get("decisions")
        if not isinstance(qualifications, list) or not isinstance(decisions, list):
            raise PilotRunnerError("MULTI_SAMPLE_REVIEW_RECORDS_REQUIRED")
    else:
        if len(raw_samples) != 1:
            raise PilotRunnerError("ONE_SAMPLE_DATASET_REQUIRED")
        qualifications = [qualification_payload]
        decisions = [decision_payload]
    if len(qualifications) != len(raw_samples) or len(decisions) != len(raw_samples):
        raise PilotRunnerError("PACKAGE_REVIEW_RECORD_COUNT_MISMATCH")
    qualification_by_id = {
        item.get("sample_id"): item for item in qualifications if isinstance(item, dict)
    }
    decision_by_id = {item.get("sample_id"): item for item in decisions if isinstance(item, dict)}
    if set(qualification_by_id) != set(sample_ids) or len(qualification_by_id) != len(qualifications):
        raise PilotRunnerError("QUALIFICATION_SAMPLE_SET_MISMATCH")
    if set(decision_by_id) != set(sample_ids) or len(decision_by_id) != len(decisions):
        raise PilotRunnerError("DECISION_SAMPLE_SET_MISMATCH")

    entries_by_id = {item.get("sample_id"): item for item in manifest["samples"]}
    validated_decisions = []
    ordered_qualifications = []
    for sample in raw_samples:
        sample_id = sample["sample_id"]
        qualification = qualification_by_id[sample_id]
        decision = decision_by_id[sample_id]
        entry = entries_by_id[sample_id]
        if qualification.get("status") != "READY":
            raise PilotRunnerError("QUALIFICATION_READY_RECORD_REQUIRED")
        if qualification.get("input_sample_sha256") != hash_value(sample):
            raise PilotRunnerError("QUALIFICATION_INPUT_HASH_MISMATCH")
        if qualification.get("security_correctness_verified") is not False:
            raise PilotRunnerError("SECURITY_CORRECTNESS_MUST_REMAIN_FALSE")
        if qualification.get("ground_truth_review_status_unchanged") != "PENDING":
            raise PilotRunnerError("CANONICAL_GROUND_TRUTH_MUST_REMAIN_PENDING")
        if sample.get("ground_truth", {}).get("review_status") != "PENDING":
            raise PilotRunnerError("SOURCE_GROUND_TRUTH_MUST_REMAIN_PENDING")
        validated_decision = validate_review_decision(decision, sample_id)
        if validated_decision.get("decision") != "APPROVE":
            raise PilotRunnerError("ATTRIBUTABLE_APPROVE_DECISION_REQUIRED")
        recomputed_qualification = qualify_sample(sample, decision=validated_decision)
        if recomputed_qualification.get("status") != "READY":
            raise PilotRunnerError("RECOMPUTED_QUALIFICATION_NOT_READY")
        if recomputed_qualification != qualification:
            raise PilotRunnerError("QUALIFICATION_RECORD_RECOMPUTE_MISMATCH")
        if package_format == "MULTI_SAMPLE_REVIEW_RECORDS_V1":
            expected_qualification_hash = hash_value(qualification)
            expected_decision_hash = hash_value(validated_decision)
        else:
            expected_qualification_hash = _sha256_file(paths["qualification"])
            expected_decision_hash = _sha256_file(paths["decision"])
        if entry.get("qualification_record_sha256") != expected_qualification_hash:
            raise PilotRunnerError("QUALIFICATION_RECORD_HASH_MISMATCH")
        if entry.get("review_decision_sha256") != expected_decision_hash:
            raise PilotRunnerError("REVIEW_DECISION_HASH_MISMATCH")
        if entry.get("canonical_input_sha256") != qualification.get("input_sample_sha256"):
            raise PilotRunnerError("MANIFEST_CANONICAL_INPUT_HASH_MISMATCH")
        ordered_qualifications.append(qualification)
        validated_decisions.append(validated_decision)

    artifact_hashes = {key: _sha256_file(path) for key, path in paths.items()}
    if receipt.get("git_commit") != expected_git_commit:
        raise PilotRunnerError("PACKAGE_RECEIPT_IDENTITY_MISMATCH")
    if package_format == "MULTI_SAMPLE_REVIEW_RECORDS_V1":
        if receipt.get("package_format") != package_format or receipt.get("ordered_sample_ids") != sample_ids:
            raise PilotRunnerError("PACKAGE_RECEIPT_IDENTITY_MISMATCH")
    elif receipt.get("sample_id") != sample_ids[0]:
        raise PilotRunnerError("PACKAGE_RECEIPT_IDENTITY_MISMATCH")
    if receipt.get("source_dataset_sha256_before") != receipt.get("source_dataset_sha256_after"):
        raise PilotRunnerError("PACKAGE_SOURCE_DATASET_MUTATION_RECORDED")
    receipt_files = receipt.get("files")
    expected_receipt_files = {
        filenames["dataset"], filenames["qualification"],
        filenames["decision"], filenames["manifest"],
    }
    if not isinstance(receipt_files, dict) or set(receipt_files) != expected_receipt_files:
        raise PilotRunnerError("PACKAGE_RECEIPT_FILE_SET_MISMATCH")
    for filename, expected_hash in receipt_files.items():
        if _sha256_file(package_dir / filename) != expected_hash:
            raise PilotRunnerError(f"PACKAGE_RECEIPT_HASH_MISMATCH:{filename}")
    expected_builder_module = (
        "dataset_validation/pilot_package.py"
        if package_format == "MULTI_SAMPLE_REVIEW_RECORDS_V1"
        else "dataset_validation/smoke_package.py"
    )
    if manifest.get("builder_module", expected_builder_module) != expected_builder_module:
        raise PilotRunnerError("PACKAGE_BUILDER_MODULE_MISMATCH")
    if receipt.get("builder_module", expected_builder_module) != expected_builder_module:
        raise PilotRunnerError("PACKAGE_BUILDER_MODULE_MISMATCH")
    builder_hash = _sha256_file(Path(__file__).resolve().parents[1] / expected_builder_module)
    if (
        manifest.get("builder_code_sha256") != builder_hash
        or receipt.get("builder_code_sha256") != builder_hash
    ):
        raise PilotRunnerError("PACKAGE_BUILDER_CODE_HASH_MISMATCH")
    return {
        "package_dir": str(package_dir.resolve()),
        "package_format": package_format,
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "manifest": manifest,
        "qualification": qualification_payload,
        "decision": decision_payload,
        "qualifications": ordered_qualifications,
        "decisions": validated_decisions,
        "receipt": receipt,
        "artifact_sha256": artifact_hashes,
    }


def validate_pilot_inputs(
    *,
    manifest_path: Path,
    dataset_path: Path,
    schema_path: Path,
    context_policy: ContextPolicy | None = None,
    model_context_tokens: int | None = None,
    reserved_output_tokens: int = 0,
) -> dict[str, Any]:
    """No-LLM validation used by the Kaggle confirmation gate."""
    policy = context_policy or ContextPolicy()
    manifest = load_official_manifest(manifest_path, dataset_path)
    adapted = load_and_adapt_dataset(dataset_path, schema_path=schema_path)
    by_id = {item["sample_id"]: item for item in adapted}
    missing = sorted(set(manifest["ordered_sample_ids"]) - set(by_id))
    if missing:
        raise PilotRunnerError(f"CANONICAL_SAMPLE_MISSING:{missing}")
    if model_context_tokens is not None:
        if model_context_tokens <= 0 or reserved_output_tokens < 0 or reserved_output_tokens >= model_context_tokens:
            raise PilotRunnerError("MODEL_CONTEXT_BUDGET_INVALID")
    input_budget = model_context_tokens - reserved_output_tokens if model_context_tokens else None
    contexts = []
    overflow_likely = []
    for sample_id in manifest["ordered_sample_ids"]:
        item = by_id[sample_id]
        if item["integration_status"] == "BLOCKED":
            raise PilotRunnerError(f"ADAPTER_BLOCKED:{sample_id}:{item['reason_codes']}")
        prepared = prepare_context(item, policy)
        file_bytes = [len(part["code_block"].encode("utf-8")) + len(part["diff"].encode("utf-8")) for part in prepared["files"]]
        diff_bytes = len(prepared["diff_bundle"]["text"].encode("utf-8"))
        # Four UTF-8 bytes/token is deliberately optimistic for source code.
        # If this lower estimate exceeds the window, inference must not start.
        p0_bytes = sum(file_bytes) + diff_bytes + 4096
        local_max_bytes = max(file_bytes, default=0) + 4096
        dependency_bytes = diff_bytes + 4096
        estimates = {
            "p0_optimistic_input_tokens": math.ceil(p0_bytes / 4),
            "p1_p2_local_optimistic_input_tokens_max": math.ceil(local_max_bytes / 4),
            "p2_dependency_optimistic_input_tokens": math.ceil(dependency_bytes / 4),
        }
        risks = [stage for stage, value in estimates.items() if input_budget is not None and value > input_budget]
        if risks:
            overflow_likely.append({"sample_id": sample_id, "stages": risks})
        contexts.append({
            "sample_id": sample_id,
            "num_files": len(prepared["files"]),
            "file_context_utf8_bytes": [len(part["code_block"].encode("utf-8")) for part in prepared["files"]],
            "diff_bundle_utf8_bytes": diff_bytes,
            "model_diff_sha256": prepared["diff_bundle"]["sha256"],
            "context_estimates": estimates,
            "overflow_likely_stages": risks,
            "representation_provenance": [
                part["representation_provenance"] for part in prepared["files"]
                if "representation_provenance" in part
            ],
        })
    return {
        "status": "BLOCKED_CONTEXT_OVERFLOW_LIKELY" if overflow_likely else "PASS_NO_LLM",
        "dataset_sha256": manifest["dataset_sha256"],
        "sample_list_sha256": manifest["sample_list_sha256"],
        "ordered_sample_ids": list(manifest["ordered_sample_ids"]),
        "context_policy_sha256": policy.sha256(),
        "model_context_tokens": model_context_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "available_input_tokens": input_budget,
        "estimator": "optimistic_utf8_bytes_div_4_plus_4096_prompt_overhead",
        "overflow_likely": overflow_likely,
        "contexts": contexts,
    }


def validate_official_package_inputs(
    *,
    package_dir: Path,
    schema_path: Path,
    expected_git_commit: str,
    context_policy: ContextPolicy | None = None,
    model_context_tokens: int | None = None,
    reserved_output_tokens: int = 0,
    exact_token_counter: Callable[[str], int] | None = None,
    server_template_overhead_tokens: int = 0,
) -> dict[str, Any]:
    """Validate the complete five-file package, then perform no-LLM context checks."""
    package = load_official_package(
        package_dir,
        schema_path,
        expected_git_commit=expected_git_commit,
    )
    validation = validate_pilot_inputs(
        manifest_path=Path(package["paths"]["manifest"]),
        dataset_path=Path(package["paths"]["dataset"]),
        schema_path=schema_path,
        context_policy=context_policy,
        model_context_tokens=model_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    static_token_counts: list[dict[str, Any]] = []
    runtime_dependent_stages: list[dict[str, str]] = []
    exact_overflow: list[dict[str, Any]] = []
    if exact_token_counter is None:
        exact_status = "NOT_RUN"
    else:
        adapted = load_and_adapt_dataset(
            Path(package["paths"]["dataset"]), schema_path=schema_path
        )
        by_id = {item["sample_id"]: item for item in adapted}
        if type(server_template_overhead_tokens) is not int or server_template_overhead_tokens < 0:
            raise PilotRunnerError("SERVER_TEMPLATE_OVERHEAD_INVALID")
        exact_limit = (
            model_context_tokens - reserved_output_tokens - server_template_overhead_tokens
            if model_context_tokens is not None else None
        )
        for sample_id in validation["ordered_sample_ids"]:
            for pipeline in ("p0", "p1", "p2"):
                plan = render_static_prompt_plan(pipeline, by_id[sample_id], context_policy)
                for entry in plan["static_prompts"]:
                    count = exact_token_counter(entry["prompt"])
                    if type(count) is not int or count < 0:
                        raise PilotRunnerError("EXACT_TOKEN_COUNTER_INVALID")
                    record = {
                        "sample_id": sample_id,
                        "pipeline": pipeline.upper(),
                        "stage": entry["stage"],
                        "prompt_sha256": entry["prompt_sha256"],
                        "utf8_bytes": entry["utf8_bytes"],
                        "input_tokens": count,
                    }
                    static_token_counts.append(record)
                    if exact_limit is not None and count > exact_limit:
                        exact_overflow.append(record)
                runtime_dependent_stages.extend(
                    {"sample_id": sample_id, "pipeline": pipeline.upper(), "stage": stage}
                    for stage in plan["runtime_dependent_stages"]
                )
        exact_status = (
            "BLOCKED_STATIC_CONTEXT_OVERFLOW"
            if exact_overflow else "PASS_STATIC_PROMPTS_RUNTIME_GUARD_REQUIRED"
        )
    return {
        **validation,
        "package_dir": package["package_dir"],
        "official_manifest_sha256": package["artifact_sha256"]["manifest"],
        "qualification_record_sha256": package["artifact_sha256"]["qualification"],
        "review_decision_sha256": package["artifact_sha256"]["decision"],
        "package_receipt_sha256": package["artifact_sha256"]["receipt"],
        "package_artifact_sha256": {
            "canonical_dataset": package["artifact_sha256"]["dataset"],
            "qualification_record": package["artifact_sha256"]["qualification"],
            "review_decision": package["artifact_sha256"]["decision"],
            "official_manifest": package["artifact_sha256"]["manifest"],
            "package_receipt": package["artifact_sha256"]["receipt"],
        },
        "exact_token_preflight_status": exact_status,
        "exact_static_prompt_token_counts": static_token_counts,
        "runtime_dependent_prompt_stages": runtime_dependent_stages,
        "exact_static_overflow": exact_overflow,
        "server_template_overhead_tokens": server_template_overhead_tokens,
    }


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    return hash_files([
        root / "pipeline" / "canonical_adapter.py",
        root / "pipeline" / "context_policy.py",
        root / "pipeline" / "lossless_context.py",
        root / "pipeline" / "llm_client.py",
        root / "pipeline" / "offline_integration.py",
        root / "pipeline" / "ollama_transport.py",
        root / "pipeline" / "pilot_runner.py",
        root / "pipeline" / "run_state.py",
    ])


def run_approved_pilot(
    *,
    package_dir: Path,
    schema_path: Path,
    output_root: Path,
    run_prefix: str,
    git_commit: str,
    model_id: str,
    model_digest: str,
    model_quantization: str,
    tokenizer_metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    retry_policy: Mapping[str, Any],
    client_factory: Callable[[str], StructuredLLMClient],
    context_policy: ContextPolicy | None = None,
    model_context_tokens: int = 32768,
    reserved_output_tokens: int = 2048,
    max_new_samples_per_pipeline: int | None = None,
    exact_token_counter: Callable[[str], int] | None = None,
    package_git_commit: str | None = None,
    allow_candidate_context_policy: bool = False,
    server_template_overhead_tokens: int = 0,
) -> dict[str, Any]:
    """Run/resume P0/P1/P2 only after the official manifest passes."""
    if not SHA40.fullmatch(git_commit):
        raise PilotRunnerError("FULL_GIT_COMMIT_REQUIRED")
    if model_id != "qwen2.5-coder:7b":
        raise PilotRunnerError("MODEL_ID_MISMATCH")
    if not isinstance(model_digest, str) or not model_digest:
        raise PilotRunnerError("MODEL_DIGEST_REQUIRED")
    if not isinstance(model_quantization, str) or not model_quantization:
        raise PilotRunnerError("MODEL_QUANTIZATION_REQUIRED")
    if not isinstance(tokenizer_metadata, Mapping) or not all(
        tokenizer_metadata.get(key) for key in ("id", "library", "version", "class")
    ):
        raise PilotRunnerError("TOKENIZER_METADATA_REQUIRED")
    if exact_token_counter is None:
        raise PilotRunnerError("EXACT_TOKEN_COUNTER_REQUIRED")
    if type(server_template_overhead_tokens) is not int or server_template_overhead_tokens < 0:
        raise PilotRunnerError("SERVER_TEMPLATE_OVERHEAD_INVALID")
    policy = context_policy or ContextPolicy()
    if policy.code_representation != FULL_BLOBS and allow_candidate_context_policy is not True:
        raise PilotRunnerError("HUMAN_DECISION_CONTEXT_POLICY_REQUIRED")
    package_commit = package_git_commit if package_git_commit is not None else git_commit
    if not SHA40.fullmatch(package_commit):
        raise PilotRunnerError("FULL_PACKAGE_GIT_COMMIT_REQUIRED")
    required_parameters = {"temperature", "num_ctx", "num_predict", "seed", "stream"}
    if not required_parameters.issubset(parameters):
        raise PilotRunnerError("IMMUTABLE_MODEL_PARAMETERS_INCOMPLETE")
    if (
        parameters.get("num_ctx") != model_context_tokens
        or parameters.get("num_predict") != reserved_output_tokens
        or parameters.get("temperature") != 0.0
        or parameters.get("seed") != 25550029
        or parameters.get("stream") is not False
    ):
        raise PilotRunnerError("MODEL_PARAMETER_CONTRACT_MISMATCH")
    retry_attempts = retry_policy.get("max_attempts")
    retry_delays = retry_policy.get("retry_delays_sec")
    if (
        type(retry_attempts) is not int
        or retry_attempts < 1
        or not isinstance(retry_delays, list)
        or len(retry_delays) < max(0, retry_attempts - 1)
        or any(type(value) not in {int, float} or value < 0 for value in retry_delays)
    ):
        raise PilotRunnerError("SHARED_RETRY_POLICY_INVALID")
    validation = validate_official_package_inputs(
        package_dir=package_dir,
        schema_path=schema_path,
        expected_git_commit=package_commit,
        context_policy=context_policy,
        model_context_tokens=model_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
        exact_token_counter=exact_token_counter,
        server_template_overhead_tokens=server_template_overhead_tokens,
    )
    if validation["status"] != "PASS_NO_LLM":
        raise PilotRunnerError(
            f"CONTEXT_BUDGET_BLOCKED:{validation['overflow_likely']}"
        )
    if not validation["exact_token_preflight_status"].startswith("PASS_"):
        raise PilotRunnerError(
            f"EXACT_TOKEN_PREFLIGHT_BLOCKED:{validation['exact_token_preflight_status']}"
        )
    policy = context_policy or ContextPolicy()
    dataset_path = Path(package_dir) / PACKAGE_FILENAMES["dataset"]
    adapted = load_and_adapt_dataset(dataset_path, schema_path=schema_path)
    by_id = {item["sample_id"]: item for item in adapted}
    ids = validation["ordered_sample_ids"]
    results_by_pipeline: dict[str, list[dict[str, Any]]] = {}
    specs: dict[str, dict[str, Any]] = {}
    complete = True
    for pipeline in PIPELINES:
        lower = pipeline.lower()
        spec = build_run_spec(
            run_id=f"{run_prefix}-{lower}",
            dataset_sha256=validation["dataset_sha256"],
            sample_ids=ids,
            pipeline_id=pipeline,
            pipeline_version=PILOT_RUNNER_VERSION,
            model_id=model_id,
            model_version=model_digest,
            parameters=dict(parameters),
            retry_policy=dict(retry_policy),
            context_policy_sha256=policy.sha256(),
            prompt_hashes=prompt_hashes(lower),
            code_sha256=_code_hash(),
            git_commit=git_commit,
            artifact_hashes=validation["package_artifact_sha256"],
            model_metadata={
                "digest": model_digest,
                "quantization": model_quantization,
                "tokenizer": dict(tokenizer_metadata),
                "server_template_overhead_tokens": server_template_overhead_tokens,
            },
            output_directory=str((Path(output_root) / "runs" / f"{run_prefix}-{lower}").resolve()),
            resume_policy={
                "mode": "EXACT_SPEC_ONLY",
                "existing_results": "REUSE_IF_HASH_AND_SPEC_MATCH",
                "overwrite": False,
            },
            context_policy_configuration=(
                {**asdict(policy), "candidate_opt_in": allow_candidate_context_policy}
                if policy.code_representation != FULL_BLOBS else None
            ),
            source_package_git_commit=package_git_commit,
        )
        workspace = RunWorkspace(Path(output_root) / "runs", spec)
        workspace.initialize()
        existing = workspace.load_results()
        completed = {item["sample_id"] for item in existing}
        new_count = 0
        client = client_factory(pipeline)
        retry_delay_count = max(0, retry_attempts - 1)
        if (
            client.max_attempts != retry_attempts
            or list(client.retry_delays[:retry_delay_count])
            != list(retry_delays[:retry_delay_count])
        ):
            raise PilotRunnerError("CLIENT_RETRY_POLICY_MISMATCH")
        for sample_id in ids:
            if sample_id in completed:
                continue
            if max_new_samples_per_pipeline is not None and new_count >= max_new_samples_per_pipeline:
                break
            result = run_canonical_sample(
                lower,
                by_id[sample_id],
                client,
                policy,
                token_counter=exact_token_counter,
                model_input_token_limit=(
                    model_context_tokens - reserved_output_tokens - server_template_overhead_tokens
                ),
            )
            workspace.write_result(result)
            workspace.append_event({
                "event": "SAMPLE_RESULT_WRITTEN",
                "sample_id": sample_id,
                "execution_status": result["execution_status"],
                "actual_attempts": result.get("num_attempts"),
            })
            new_count += 1
        loaded = workspace.load_results()
        result_by_id = {item["sample_id"]: item for item in loaded}
        ordered = [result_by_id[item] for item in ids if item in result_by_id]
        if len(ordered) != len(ids):
            complete = False
        results_by_pipeline[pipeline] = ordered
        specs[pipeline] = spec
    evaluation = evaluate_positive_only_ablation(results_by_pipeline, ids) if complete else None
    return {
        "runner_version": PILOT_RUNNER_VERSION,
        "experiment_git_commit": git_commit,
        "source_package_git_commit": package_commit,
        "complete": complete,
        "ordered_sample_ids": ids,
        "dataset_sha256": validation["dataset_sha256"],
        "official_manifest_sha256": validation["official_manifest_sha256"],
        "review_decision_sha256": validation["review_decision_sha256"],
        "qualification_record_sha256": validation["qualification_record_sha256"],
        "package_receipt_sha256": validation["package_receipt_sha256"],
        "exact_token_preflight_status": validation["exact_token_preflight_status"],
        "sample_list_sha256": validation["sample_list_sha256"],
        "run_spec_sha256_by_pipeline": {key: value["spec_sha256"] for key, value in specs.items()},
        "result_count_by_pipeline": {key: len(value) for key, value in results_by_pipeline.items()},
        "evaluation": evaluation,
    }
