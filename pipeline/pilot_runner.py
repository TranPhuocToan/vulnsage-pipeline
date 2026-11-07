"""Fail-closed canonical pilot runner for an approved positive-only manifest."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline.canonical_adapter import dataset_sha256, load_and_adapt_dataset
from pipeline.context_policy import ContextPolicy, prepare_context
from pipeline.evaluator import evaluate_positive_only_ablation
from pipeline.llm_client import StructuredLLMClient
from pipeline.offline_integration import prompt_hashes, run_canonical_sample
from pipeline.run_state import RunWorkspace, build_run_spec, hash_files, hash_value


PILOT_RUNNER_VERSION = "phase3c3-canonical-pilot-runner-1.0"
PIPELINES = ("P0", "P1", "P2")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


class PilotRunnerError(ValueError):
    pass


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


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    return hash_files([
        root / "pipeline" / "canonical_adapter.py",
        root / "pipeline" / "context_policy.py",
        root / "pipeline" / "llm_client.py",
        root / "pipeline" / "offline_integration.py",
        root / "pipeline" / "ollama_transport.py",
        root / "pipeline" / "pilot_runner.py",
        root / "pipeline" / "run_state.py",
    ])


def run_approved_pilot(
    *,
    manifest_path: Path,
    dataset_path: Path,
    schema_path: Path,
    output_root: Path,
    run_prefix: str,
    git_commit: str,
    model_id: str,
    model_digest: str,
    parameters: Mapping[str, Any],
    retry_policy: Mapping[str, Any],
    client_factory: Callable[[str], StructuredLLMClient],
    context_policy: ContextPolicy | None = None,
    model_context_tokens: int = 32768,
    reserved_output_tokens: int = 2048,
    max_new_samples_per_pipeline: int | None = None,
) -> dict[str, Any]:
    """Run/resume P0/P1/P2 only after the official manifest passes."""
    if not SHA40.fullmatch(git_commit):
        raise PilotRunnerError("FULL_GIT_COMMIT_REQUIRED")
    if model_id != "qwen2.5-coder:7b":
        raise PilotRunnerError("MODEL_ID_MISMATCH")
    if not isinstance(model_digest, str) or not model_digest:
        raise PilotRunnerError("MODEL_DIGEST_REQUIRED")
    validation = validate_pilot_inputs(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        schema_path=schema_path,
        context_policy=context_policy,
        model_context_tokens=model_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    if validation["status"] != "PASS_NO_LLM":
        raise PilotRunnerError(
            f"CONTEXT_BUDGET_BLOCKED:{validation['overflow_likely']}"
        )
    policy = context_policy or ContextPolicy()
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
        )
        workspace = RunWorkspace(Path(output_root) / "runs", spec)
        workspace.initialize()
        existing = workspace.load_results()
        completed = {item["sample_id"] for item in existing}
        new_count = 0
        client = client_factory(pipeline)
        for sample_id in ids:
            if sample_id in completed:
                continue
            if max_new_samples_per_pipeline is not None and new_count >= max_new_samples_per_pipeline:
                break
            result = run_canonical_sample(lower, by_id[sample_id], client, policy)
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
        "complete": complete,
        "ordered_sample_ids": ids,
        "dataset_sha256": validation["dataset_sha256"],
        "sample_list_sha256": validation["sample_list_sha256"],
        "run_spec_sha256_by_pipeline": {key: value["spec_sha256"] for key, value in specs.items()},
        "result_count_by_pipeline": {key: len(value) for key, value in results_by_pipeline.items()},
        "evaluation": evaluation,
    }
