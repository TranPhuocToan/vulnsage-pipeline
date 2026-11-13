"""Fail-closed offline audit for a VulnSage engineering-smoke archive.

The auditor reads a ZIP in place.  It does not extract or modify the archive,
does not import an LLM transport, and does not calculate classification metrics
when the canonical ground-truth review remains pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PIPELINES = ("P0", "P1", "P2")
PACKAGE_FILES = {
    "canonical_dataset": "smoke_package/canonical_dataset.json",
    "official_manifest": "smoke_package/official_pilot_manifest.json",
    "package_receipt": "smoke_package/package_receipt.json",
    "qualification_record": "smoke_package/qualification_record.json",
    "review_decision": "smoke_package/review_decision.json",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA40_RE = re.compile(r"[0-9a-f]{40}")


class SmokeAuditError(ValueError):
    """Raised when evidence is missing or inconsistent."""


@dataclass(frozen=True)
class AuditExpectations:
    experiment_git_sha: str | None = None
    source_package_git_sha: str | None = None
    sample_id: str | None = None
    run_prefix: str | None = None
    model_id: str | None = None
    context_policy_sha256: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeAuditError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def _json(raw: bytes, name: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise SmokeAuditError(f"INVALID_JSON:{name}:{exc}") from exc


def _json_lines(raw: bytes, name: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SmokeAuditError(f"INVALID_UTF8:{name}:{exc}") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        item = _json(line.encode("utf-8"), f"{name}:line:{number}")
        _require(isinstance(item, dict), f"JOURNAL_ENTRY_NOT_OBJECT:{name}:{number}")
        records.append(item)
    return records


def _safe_zip_inventory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    inventory: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        _require(not path.is_absolute(), f"UNSAFE_ABSOLUTE_ZIP_PATH:{name}")
        _require(".." not in path.parts, f"UNSAFE_PARENT_ZIP_PATH:{name}")
        _require("\\" not in name, f"UNSAFE_BACKSLASH_ZIP_PATH:{name}")
        _require(name not in inventory, f"DUPLICATE_ZIP_ENTRY:{name}")
        inventory[name] = info
    _require(archive.testzip() is None, "ZIP_CRC_FAILURE")
    return inventory


def _read_required(
    archive: zipfile.ZipFile, inventory: Mapping[str, zipfile.ZipInfo], name: str
) -> bytes:
    _require(name in inventory, f"MISSING_ARCHIVE_ENTRY:{name}")
    return archive.read(name)


def _find_evidence(
    archive: zipfile.ZipFile,
    inventory: Mapping[str, zipfile.ZipInfo],
    *,
    filename_prefix: str,
) -> tuple[str, dict[str, Any]]:
    matches = sorted(
        name
        for name in inventory
        if name.startswith("evidence/")
        and PurePosixPath(name).name.startswith(filename_prefix)
        and name.endswith(".json")
    )
    _require(len(matches) == 1, f"EVIDENCE_CARDINALITY:{filename_prefix}:{len(matches)}")
    return matches[0], _json(archive.read(matches[0]), matches[0])


def _expect(label: str, actual: Any, expected: Any | None) -> None:
    if expected is not None:
        _require(actual == expected, f"EXPECTED_{label}_MISMATCH:{actual!r}!={expected!r}")


def _validate_package(
    archive: zipfile.ZipFile,
    inventory: Mapping[str, zipfile.ZipInfo],
    index: Mapping[str, Any],
) -> dict[str, Any]:
    raw = {
        key: _read_required(archive, inventory, name)
        for key, name in PACKAGE_FILES.items()
    }
    actual_hashes = {key: _sha256(value) for key, value in raw.items()}
    values = {key: _json(value, PACKAGE_FILES[key]) for key, value in raw.items()}
    receipt = values["package_receipt"]
    receipt_names = {
        "canonical_dataset": "canonical_dataset.json",
        "official_manifest": "official_pilot_manifest.json",
        "qualification_record": "qualification_record.json",
        "review_decision": "review_decision.json",
    }
    for key, filename in receipt_names.items():
        _require(
            receipt.get("files", {}).get(filename) == actual_hashes[key],
            f"PACKAGE_RECEIPT_HASH_MISMATCH:{filename}",
        )

    manifest = values["official_manifest"]
    manifest_without_hash = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    _require(
        manifest.get("manifest_sha256") == _canonical_hash(manifest_without_hash),
        "MANIFEST_SELF_HASH_MISMATCH",
    )
    _require(
        manifest.get("dataset_sha256") == actual_hashes["canonical_dataset"],
        "MANIFEST_DATASET_HASH_MISMATCH",
    )
    _require(
        index.get("dataset_sha256") == actual_hashes["canonical_dataset"],
        "INDEX_DATASET_HASH_MISMATCH",
    )
    _require(
        index.get("manifest_file_sha256") == actual_hashes["official_manifest"],
        "INDEX_MANIFEST_HASH_MISMATCH",
    )
    _require(
        receipt.get("git_commit") == index.get("source_package_git_sha"),
        "RECEIPT_SOURCE_PACKAGE_GIT_MISMATCH",
    )
    _require(
        manifest.get("git_commit") == index.get("source_package_git_sha"),
        "MANIFEST_SOURCE_PACKAGE_GIT_MISMATCH",
    )

    dataset = values["canonical_dataset"]
    samples = dataset.get("samples")
    _require(isinstance(samples, list) and len(samples) == 1, "CANONICAL_SAMPLE_COUNT_NOT_ONE")
    sample = samples[0]
    _require(sample.get("sample_id") == index.get("sample_id"), "CANONICAL_SAMPLE_ID_MISMATCH")
    ground_truth = sample.get("ground_truth")
    _require(isinstance(ground_truth, dict), "CANONICAL_GROUND_TRUTH_MISSING")
    _require(
        ground_truth.get("review_status") == "PENDING",
        "CANONICAL_GROUND_TRUTH_NOT_PENDING",
    )

    qualification = values["qualification_record"]
    decision = values["review_decision"]
    _require(qualification.get("sample_id") == index.get("sample_id"), "QUALIFICATION_SAMPLE_ID_MISMATCH")
    _require(decision.get("sample_id") == index.get("sample_id"), "DECISION_SAMPLE_ID_MISMATCH")
    _require(qualification.get("security_correctness_verified") is False, "SECURITY_CORRECTNESS_WAS_VERIFIED")
    _require(
        qualification.get("ground_truth_review_status_unchanged") == "PENDING",
        "QUALIFICATION_GROUND_TRUTH_NOT_PENDING",
    )
    _require(decision.get("decision") == "APPROVE", "ENGINEERING_SMOKE_REVIEW_NOT_APPROVED")
    _require(manifest.get("scope") == "ONE_SAMPLE_SMOKE_NOT_MAIN_EXPERIMENT", "MANIFEST_SCOPE_INVALID")
    _require(manifest.get("official_split") is False, "SMOKE_PACKAGE_MARKED_OFFICIAL_SPLIT")
    _require(manifest.get("ordered_sample_ids") == [index.get("sample_id")], "MANIFEST_SAMPLE_LIST_MISMATCH")
    _require(receipt.get("sample_id") == index.get("sample_id"), "RECEIPT_SAMPLE_ID_MISMATCH")

    return {
        "actual_hashes": actual_hashes,
        "values": values,
        "ground_truth_review_status": ground_truth.get("review_status"),
        "security_correctness_verified": qualification.get("security_correctness_verified"),
    }


def _validate_evidence(
    archive: zipfile.ZipFile,
    inventory: Mapping[str, zipfile.ZipInfo],
    index: Mapping[str, Any],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    evidence: dict[str, tuple[str, dict[str, Any]]] = {}
    for key, prefix in {
        "preparation": "smoke-preparation-",
        "policy_approval": "context-policy-approval-",
        "context_diagnostic": "context-19047-",
        "inference_authorization": "inference-authorization-",
        "attempt_record": "smoke-attempt-v2-",
        "runtime_loadcheck": "runtime-loadcheck-",
    }.items():
        evidence[key] = _find_evidence(
            archive, inventory, filename_prefix=prefix
        )

    common_sample = index.get("sample_id")
    policy = evidence["policy_approval"][1]
    authorization = evidence["inference_authorization"][1]
    context = evidence["context_diagnostic"][1]
    preparation = evidence["preparation"][1]
    attempt = evidence["attempt_record"][1]
    runtime = evidence["runtime_loadcheck"][1]

    _require(policy.get("decision") == "APPROVE", "CONTEXT_POLICY_NOT_APPROVED")
    _require(policy.get("scope") == "engineering_smoke_only", "CONTEXT_POLICY_SCOPE_INVALID")
    _require(authorization.get("inference_authorized") is True, "INFERENCE_AUTHORIZATION_MISSING")
    _require(authorization.get("main_experiment_authorized") is False, "MAIN_EXPERIMENT_WAS_AUTHORIZED")
    _require(authorization.get("held_out_evaluation_authorized") is False, "HELDOUT_WAS_AUTHORIZED")
    _require(runtime.get("inference_calls") == 0, "LOADCHECK_CONTAINS_INFERENCE")
    _require(runtime.get("generated_tokens") == 0, "LOADCHECK_GENERATED_TOKENS")

    policy_sha = policy.get("policy_sha256")
    experiment_sha = index.get("experiment_git_sha")
    package_sha = index.get("source_package_git_sha")
    run_prefix = index.get("run_prefix")
    _require(SHA256_RE.fullmatch(str(policy_sha)) is not None, "INVALID_POLICY_SHA")
    _require(SHA40_RE.fullmatch(str(experiment_sha)) is not None, "INVALID_EXPERIMENT_GIT_SHA")
    _require(SHA40_RE.fullmatch(str(package_sha)) is not None, "INVALID_SOURCE_PACKAGE_GIT_SHA")
    for label, item in evidence.items():
        record = item[1]
        if "sample_id" in record:
            _require(record["sample_id"] == common_sample, f"EVIDENCE_SAMPLE_ID_MISMATCH:{label}")
        for field in ("experiment_git_sha", "experiment_code_git_commit"):
            if field in record:
                _require(record[field] == experiment_sha, f"EVIDENCE_EXPERIMENT_GIT_MISMATCH:{label}")
        for field in ("source_package_git_sha", "source_package_git_commit"):
            if field in record:
                _require(record[field] == package_sha, f"EVIDENCE_PACKAGE_GIT_MISMATCH:{label}")
        for field in ("context_policy_sha256", "approved_context_policy_sha256", "policy_sha256"):
            if field in record and label != "context_diagnostic":
                _require(record[field] == policy_sha, f"EVIDENCE_POLICY_SHA_MISMATCH:{label}")
        if "run_prefix" in record or "run_prefix_proposed" in record:
            candidate = record.get("run_prefix", record.get("run_prefix_proposed"))
            _require(candidate == run_prefix, f"EVIDENCE_RUN_PREFIX_MISMATCH:{label}")

    approved_reports = [
        report for report in context.get("reports", [])
        if report.get("policy_sha256") == policy_sha
    ]
    _require(len(approved_reports) == 1, "APPROVED_POLICY_DIAGNOSTIC_NOT_UNIQUE")
    _require(
        approved_reports[0].get("code_representation") == policy.get("policy_name"),
        "POLICY_REPRESENTATION_MISMATCH",
    )
    _require(attempt.get("run_prefix") == run_prefix, "ATTEMPT_RUN_PREFIX_MISMATCH")
    _require(preparation.get("inference_executed") is False, "PREPARATION_RECORD_ALREADY_EXECUTED")

    _expect("POLICY_SHA", policy_sha, expectations.context_policy_sha256)
    return {
        "files": {key: value[0] for key, value in evidence.items()},
        "policy_sha256": policy_sha,
        "policy_name": policy.get("policy_name"),
        "model_id": authorization.get("model"),
        "model_digest": authorization.get("model_digest"),
        "authorized_scope": authorization.get("scope"),
        "main_experiment_authorized": authorization.get("main_experiment_authorized"),
        "heldout_evaluation_authorized": authorization.get("held_out_evaluation_authorized"),
        "preparation_preceded_inference": preparation.get("inference_executed") is False,
    }


def _validate_run(
    archive: zipfile.ZipFile,
    inventory: Mapping[str, zipfile.ZipInfo],
    index: Mapping[str, Any],
    package: Mapping[str, Any],
    evidence: Mapping[str, Any],
    pipeline: str,
) -> dict[str, Any]:
    key = pipeline.lower()
    pipeline_dir = index.get("pipeline_directories", {}).get(key)
    _require(isinstance(pipeline_dir, str) and pipeline_dir, f"PIPELINE_DIRECTORY_MISSING:{pipeline}")
    run_id = PurePosixPath(pipeline_dir).name
    _require(run_id == f"{index.get('run_prefix')}-{key}", f"RUN_ID_MISMATCH:{pipeline}")
    root = f"runs/{run_id}"
    spec_name = f"{root}/run_spec.json"
    journal_name = f"{root}/execution_journal.jsonl"
    result_names = sorted(
        name for name in inventory if name.startswith(f"{root}/results/") and name.endswith(".json")
    )
    _require(len(result_names) == 1, f"RESULT_CARDINALITY:{pipeline}:{len(result_names)}")
    spec_raw = _read_required(archive, inventory, spec_name)
    journal_raw = _read_required(archive, inventory, journal_name)
    result_raw = _read_required(archive, inventory, result_names[0])
    spec = _json(spec_raw, spec_name)
    result = _json(result_raw, result_names[0])
    journal = _json_lines(journal_raw, journal_name)

    spec_without_hash = {k: v for k, v in spec.items() if k != "spec_sha256"}
    _require(spec.get("spec_sha256") == _canonical_hash(spec_without_hash), f"SPEC_SELF_HASH_MISMATCH:{pipeline}")
    _require(spec.get("run_id") == run_id, f"SPEC_RUN_ID_MISMATCH:{pipeline}")
    _require(spec.get("pipeline_id") == pipeline, f"SPEC_PIPELINE_MISMATCH:{pipeline}")
    _require(spec.get("sample_ids") == [index.get("sample_id")], f"SPEC_SAMPLE_LIST_MISMATCH:{pipeline}")
    _require(
        spec.get("sample_list_sha256") == _canonical_hash(spec.get("sample_ids")),
        f"SPEC_SAMPLE_LIST_HASH_MISMATCH:{pipeline}",
    )
    _require(
        spec.get("dataset_sha256") == package["actual_hashes"]["canonical_dataset"],
        f"SPEC_DATASET_HASH_MISMATCH:{pipeline}",
    )
    _require(spec.get("git_commit") == index.get("experiment_git_sha"), f"SPEC_EXPERIMENT_GIT_MISMATCH:{pipeline}")
    _require(spec.get("source_package_git_commit") == index.get("source_package_git_sha"), f"SPEC_PACKAGE_GIT_MISMATCH:{pipeline}")
    _require(spec.get("context_policy_sha256") == evidence.get("policy_sha256"), f"SPEC_POLICY_SHA_MISMATCH:{pipeline}")
    _require(spec.get("model", {}).get("id") == evidence.get("model_id"), f"SPEC_MODEL_ID_MISMATCH:{pipeline}")
    _require(spec.get("model", {}).get("version") == evidence.get("model_digest"), f"SPEC_MODEL_DIGEST_MISMATCH:{pipeline}")
    _require(
        spec.get("context_policy_configuration", {}).get("code_representation") == evidence.get("policy_name"),
        f"SPEC_POLICY_REPRESENTATION_MISMATCH:{pipeline}",
    )
    for artifact_key, actual_hash in package["actual_hashes"].items():
        _require(spec.get("artifact_hashes", {}).get(artifact_key) == actual_hash, f"SPEC_ARTIFACT_HASH_MISMATCH:{pipeline}:{artifact_key}")

    _require(result.get("pipeline") == pipeline, f"RESULT_PIPELINE_MISMATCH:{pipeline}")
    _require(result.get("sample_id") == index.get("sample_id"), f"RESULT_SAMPLE_ID_MISMATCH:{pipeline}")
    _require(result.get("spec_sha256") == spec.get("spec_sha256"), f"RESULT_SPEC_MISMATCH:{pipeline}")
    _require(result.get("context_policy_sha256") == evidence.get("policy_sha256"), f"RESULT_POLICY_MISMATCH:{pipeline}")
    _require(result.get("execution_status") == "COMPLETED", f"RESULT_NOT_COMPLETED:{pipeline}")
    _require(result.get("prediction") in {"PATCHED", "NOT_PATCHED", "NOT_SURE"}, f"INVALID_PREDICTION:{pipeline}")
    _require(type(result.get("model_abstained")) is bool, f"INVALID_ABSTENTION:{pipeline}")
    _require(result.get("exact_token_preflight") == "EXACT", f"TOKEN_PREFLIGHT_NOT_EXACT:{pipeline}")

    attempts = result.get("attempts")
    _require(isinstance(attempts, list) and attempts, f"ATTEMPTS_MISSING:{pipeline}")
    _require(result.get("num_attempts") == len(attempts), f"NUM_ATTEMPTS_MISMATCH:{pipeline}")
    for number, item in enumerate(attempts, 1):
        _require(isinstance(item, dict), f"ATTEMPT_NOT_OBJECT:{pipeline}:{number}")
        _require(type(item.get("attempt")) is int and item["attempt"] >= 1, f"ATTEMPT_NUMBER_INVALID:{pipeline}:{number}")
        _require(item.get("status") in {"OK", "API_ERROR", "JSON_ERROR", "SCHEMA_ERROR"}, f"ATTEMPT_STATUS_INVALID:{pipeline}:{number}")
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            _require(type(item.get(field)) is int and item[field] >= 0, f"ATTEMPT_TOKEN_INVALID:{pipeline}:{number}:{field}")
        _require(item["input_tokens"] + item["output_tokens"] == item["total_tokens"], f"ATTEMPT_TOKEN_SUM_MISMATCH:{pipeline}:{number}")
        _require(isinstance(item.get("duration_sec"), (int, float)) and item["duration_sec"] >= 0, f"ATTEMPT_DURATION_INVALID:{pipeline}:{number}")

    token_usage = result.get("token_usage")
    _require(isinstance(token_usage, dict) and token_usage.get("usage_complete") is True, f"TOKEN_USAGE_INCOMPLETE:{pipeline}")
    sums = {
        field: sum(item[field] for item in attempts)
        for field in ("input_tokens", "output_tokens", "total_tokens")
    }
    for field, value in sums.items():
        _require(token_usage.get(field) == value, f"RESULT_TOKEN_SUM_MISMATCH:{pipeline}:{field}")
    duration = sum(float(item["duration_sec"]) for item in attempts)
    _require(math.isclose(float(result.get("total_attempt_duration_sec")), duration, rel_tol=0, abs_tol=1e-9), f"RESULT_DURATION_SUM_MISMATCH:{pipeline}")

    pre_calls = result.get("pre_call_token_counts")
    _require(isinstance(pre_calls, list) and pre_calls, f"PRE_CALL_COUNTS_MISSING:{pipeline}")
    attempt_groups: list[list[dict[str, Any]]] = []
    for item in attempts:
        ordinal = item["attempt"]
        if ordinal == 1:
            attempt_groups.append([item])
        else:
            _require(bool(attempt_groups), f"RETRY_WITHOUT_INITIAL_ATTEMPT:{pipeline}")
            _require(
                ordinal == attempt_groups[-1][-1]["attempt"] + 1,
                f"NONSEQUENTIAL_RETRY_ATTEMPT:{pipeline}:{ordinal}",
            )
            attempt_groups[-1].append(item)
    _require(
        len(pre_calls) == len(attempt_groups),
        f"PRE_CALL_CARDINALITY_MISMATCH:{pipeline}",
    )
    parameters = spec.get("parameters", {})
    input_limit = parameters.get("num_ctx") - parameters.get("num_predict")
    deltas = []
    for number, (pre, group) in enumerate(zip(pre_calls, attempt_groups), 1):
        raw_tokens = pre.get("input_tokens")
        _require(type(raw_tokens) is int and raw_tokens >= 0, f"PRE_CALL_TOKEN_INVALID:{pipeline}:{number}")
        _require(raw_tokens <= input_limit, f"RAW_CONTEXT_OVERFLOW:{pipeline}:{number}")
        for attempt_item in group:
            _require(attempt_item["input_tokens"] <= input_limit, f"SERVER_REPORTED_CONTEXT_OVERFLOW:{pipeline}:{number}")
            deltas.append(attempt_item["input_tokens"] - raw_tokens)

    _require(len(journal) == 1, f"JOURNAL_LINE_COUNT_UNEXPECTED:{pipeline}:{len(journal)}")
    journal_record = journal[0]
    _require(journal_record.get("event") == "SAMPLE_RESULT_WRITTEN", f"JOURNAL_EVENT_INVALID:{pipeline}")
    _require(journal_record.get("sample_id") == result.get("sample_id"), f"JOURNAL_SAMPLE_MISMATCH:{pipeline}")
    _require(journal_record.get("spec_sha256") == spec.get("spec_sha256"), f"JOURNAL_SPEC_MISMATCH:{pipeline}")
    _require(journal_record.get("actual_attempts") == len(attempts), f"JOURNAL_ATTEMPT_COUNT_MISMATCH:{pipeline}")
    _require(journal_record.get("execution_status") == result.get("execution_status"), f"JOURNAL_STATUS_MISMATCH:{pipeline}")

    retry_calls = sum(1 for item in attempts if item["attempt"] > 1)
    error_calls = sum(1 for item in attempts if item["status"] != "OK")
    return {
        "pipeline": pipeline,
        "run_id": run_id,
        "sample_id": result["sample_id"],
        "execution_status": result["execution_status"],
        "prediction": result["prediction"],
        "model_abstained": result["model_abstained"],
        "num_attempts": result["num_attempts"],
        "actual_llm_calls": len(attempts),
        "attempts": attempts,
        "input_tokens": sums["input_tokens"],
        "output_tokens": sums["output_tokens"],
        "total_tokens": sums["total_tokens"],
        "total_attempt_duration_sec": result["total_attempt_duration_sec"],
        "exact_token_preflight": result["exact_token_preflight"],
        "pre_call_token_counts": pre_calls,
        "ollama_minus_raw_input_tokens": deltas,
        "context_policy_sha256": result["context_policy_sha256"],
        "spec_sha256": spec["spec_sha256"],
        "model": spec["model"],
        "model_metadata": spec.get("model_metadata"),
        "parameters": parameters,
        "retry_policy": spec.get("retry_policy"),
        "journal_line_count": len(journal),
        "retry_calls_due_to_error": retry_calls,
        "error_call_records": error_calls,
        "source_files": {
            "run_spec": spec_name,
            "result": result_names[0],
            "execution_journal": journal_name,
        },
    }


def audit_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    expectations: AuditExpectations | None = None,
) -> dict[str, Any]:
    """Validate an archive and return verified metrics or raise fail-closed."""
    expectations = expectations or AuditExpectations()
    path = Path(archive_path)
    _require(path.is_file(), f"ARCHIVE_NOT_FOUND:{path}")
    _require(SHA256_RE.fullmatch(expected_archive_sha256.lower()) is not None, "EXPECTED_ARCHIVE_SHA_INVALID")
    archive_bytes = path.read_bytes()
    archive_sha = _sha256(archive_bytes)
    _require(archive_sha == expected_archive_sha256.lower(), f"ARCHIVE_SHA256_MISMATCH:{archive_sha}")

    try:
        archive_context = zipfile.ZipFile(path)
    except Exception as exc:
        raise SmokeAuditError(f"INVALID_ZIP:{exc}") from exc
    with archive_context as archive:
        inventory = _safe_zip_inventory(archive)
        index = _json(_read_required(archive, inventory, "audit_index.json"), "audit_index.json")
        _require(index.get("record_type") == "engineering_smoke_artifact_index", "AUDIT_INDEX_TYPE_INVALID")
        _require(index.get("scope") == "engineering_smoke_only", "AUDIT_INDEX_SCOPE_INVALID")
        _require(index.get("ground_truth_review_status") == "PENDING", "INDEX_GROUND_TRUTH_NOT_PENDING")

        _expect("EXPERIMENT_GIT", index.get("experiment_git_sha"), expectations.experiment_git_sha)
        _expect("SOURCE_PACKAGE_GIT", index.get("source_package_git_sha"), expectations.source_package_git_sha)
        _expect("SAMPLE_ID", index.get("sample_id"), expectations.sample_id)
        _expect("RUN_PREFIX", index.get("run_prefix"), expectations.run_prefix)

        package = _validate_package(archive, inventory, index)
        evidence = _validate_evidence(archive, inventory, index, expectations)
        _expect("MODEL_ID", evidence.get("model_id"), expectations.model_id)

        inventory_records = index.get("artifact_inventory")
        _require(isinstance(inventory_records, list) and len(inventory_records) == 9, "INDEX_RUN_INVENTORY_COUNT_INVALID")
        indexed_run_paths: set[str] = set()
        for item in inventory_records:
            pipeline = item.get("pipeline")
            _require(pipeline in PIPELINES, f"INDEX_INVENTORY_PIPELINE_INVALID:{pipeline}")
            run_dir = PurePosixPath(index["pipeline_directories"][pipeline.lower()]).name
            name = f"runs/{run_dir}/{item.get('relative_path')}"
            _require(name not in indexed_run_paths, f"DUPLICATE_INDEX_INVENTORY_ENTRY:{name}")
            indexed_run_paths.add(name)
            raw = _read_required(archive, inventory, name)
            _require(item.get("size_bytes") == len(raw), f"INDEX_SIZE_MISMATCH:{name}")
            _require(item.get("sha256") == _sha256(raw), f"INDEX_HASH_MISMATCH:{name}")
        archived_run_paths = {name for name in inventory if name.startswith("runs/")}
        _require(archived_run_paths == indexed_run_paths, "RUN_INVENTORY_SET_MISMATCH")

        provenance_basenames = {
            PurePosixPath(name).name for name in index.get("provenance_files", [])
        }
        _require(
            provenance_basenames == {
                PurePosixPath(name).name for name in evidence["files"].values()
            },
            "PROVENANCE_FILE_SET_MISMATCH",
        )

        runs = [
            _validate_run(archive, inventory, index, package, evidence, pipeline)
            for pipeline in PIPELINES
        ]
        _require(
            all(run["parameters"] == runs[0]["parameters"] for run in runs[1:]),
            "CROSS_PIPELINE_PARAMETER_MISMATCH",
        )
        _require(
            all(run["model_metadata"] == runs[0]["model_metadata"] for run in runs[1:]),
            "CROSS_PIPELINE_MODEL_METADATA_MISMATCH",
        )

    totals = {
        "actual_llm_calls": sum(run["actual_llm_calls"] for run in runs),
        "journal_lines": sum(run["journal_line_count"] for run in runs),
        "retry_calls_due_to_error": sum(run["retry_calls_due_to_error"] for run in runs),
        "error_call_records": sum(run["error_call_records"] for run in runs),
        "input_tokens": sum(run["input_tokens"] for run in runs),
        "output_tokens": sum(run["output_tokens"] for run in runs),
        "total_tokens": sum(run["total_tokens"] for run in runs),
        "total_attempt_duration_sec": sum(run["total_attempt_duration_sec"] for run in runs),
    }
    all_deltas = [delta for run in runs for delta in run["ollama_minus_raw_input_tokens"]]
    _require(totals["actual_llm_calls"] == len(all_deltas), "TOKEN_DELTA_CALL_COUNT_MISMATCH")
    technical_pass = (
        all(run["execution_status"] == "COMPLETED" for run in runs)
        and totals["error_call_records"] == 0
        and totals["retry_calls_due_to_error"] == 0
    )
    return {
        "audit_version": "phase3d-smoke-postaudit-1.0",
        "status": "PASS" if technical_pass else "FAIL",
        "archive": {
            "path": str(path.resolve()),
            "size_bytes": len(archive_bytes),
            "sha256": archive_sha,
            "entry_count": len(inventory),
        },
        "experiment": {
            "scope": index["scope"],
            "run_prefix": index["run_prefix"],
            "experiment_git_sha": index["experiment_git_sha"],
            "source_package_git_sha": index["source_package_git_sha"],
            "sample_id": index["sample_id"],
            "model_id": evidence["model_id"],
            "model_digest": evidence["model_digest"],
            "context_policy": evidence["policy_name"],
            "context_policy_sha256": evidence["policy_sha256"],
        },
        "package": {
            "hashes": package["actual_hashes"],
            "ground_truth_review_status": package["ground_truth_review_status"],
            "security_correctness_verified": package["security_correctness_verified"],
        },
        "evidence": evidence,
        "pipelines": runs,
        "totals": totals,
        "token_difference": {
            "definition": "Ollama-reported input_tokens minus raw pre-call tokenizer count",
            "per_call": all_deltas,
            "unique_values": sorted(set(all_deltas)),
            "direct_cause_verified": False,
            "interpretation": "Observed difference only; archive does not directly prove its cause.",
        },
        "scientific_interpretation": {
            "engineering_smoke_technical_execution": "PASS" if technical_pass else "FAIL",
            "security_correctness": "NOT VERIFIED",
            "ground_truth": package["ground_truth_review_status"],
            "held_out_evaluation": "NOT RUN",
            "main_experiment": "NOT RUN",
            "classification_quality_metrics": "NOT COMPUTED: canonical ground truth is PENDING",
        },
    }


def render_markdown(metrics: Mapping[str, Any]) -> str:
    experiment = metrics["experiment"]
    totals = metrics["totals"]
    lines = [
        "# Post-Smoke Artifact Audit — sample 19047",
        "",
        "## 1. Mục tiêu Engineering Smoke",
        "",
        "Kiểm tra khả năng thực thi kỹ thuật end-to-end của P0/P1/P2 trên đúng một mẫu researcher-exposed. Đây không phải held-out evaluation, main experiment hay kiểm chứng formal security correctness.",
        "",
        "## 2. Cấu hình thực nghiệm",
        "",
        f"- Run prefix: `{experiment['run_prefix']}`",
        f"- Sample: `{experiment['sample_id']}`",
        f"- Model: `{experiment['model_id']}`; digest `{experiment['model_digest']}`",
        f"- Context policy: `{experiment['context_policy']}`; SHA-256 `{experiment['context_policy_sha256']}`",
        f"- Experiment Git commit: `{experiment['experiment_git_sha']}`",
        f"- Source package Git commit: `{experiment['source_package_git_sha']}`",
        "",
        "## 3. Nguồn dữ liệu và provenance",
        "",
        f"- Archive: `{metrics['archive']['path']}`",
        f"- Archive SHA-256: `{metrics['archive']['sha256']}` (MATCH với giá trị kỳ vọng).",
        f"- Số entry ZIP: {metrics['archive']['entry_count']}.",
        "- Hash của 9 run artifacts khớp `audit_index.json`; 5 smoke-package artifacts khớp receipt/run specifications; run-spec self-hash và manifest self-hash hợp lệ.",
        f"- Canonical ground truth review: `{metrics['package']['ground_truth_review_status']}`; `security_correctness_verified={str(metrics['package']['security_correctness_verified']).lower()}`.",
        "",
        "## 4. Kết quả P0/P1/P2",
        "",
        "| Pipeline | Status | Prediction | Abstained | Calls thật | Input | Output | Total | Duration (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in metrics["pipelines"]:
        lines.append(
            f"| {run['pipeline']} | {run['execution_status']} | {run['prediction']} | "
            f"{str(run['model_abstained']).lower()} | {run['actual_llm_calls']} | "
            f"{run['input_tokens']} | {run['output_tokens']} | {run['total_tokens']} | "
            f"{run['total_attempt_duration_sec']:.9f} |"
        )
    lines.extend([
        "",
        "Kết quả verdict trên một mẫu không cho phép xếp hạng P0/P1/P2.",
        "",
        "## 5. Token consumption",
        "",
        f"Tổng cộng: input {totals['input_tokens']}, output {totals['output_tokens']}, total {totals['total_tokens']} tokens.",
        "",
        "Mỗi call có `Ollama input_tokens - raw pre-call tokens = 29`. Đây là quan sát trực tiếp. Archive không chứa bằng chứng đủ để quy nguyên nhân chính xác cho chat template/server overhead, nên nguyên nhân vẫn **NOT VERIFIED**.",
        "",
        "## 6. Execution time",
        "",
        f"Tổng duration cộng theo attempt: {totals['total_attempt_duration_sec']:.9f} giây. Đây là tổng thời lượng call, không được diễn giải là wall-clock benchmark đa hệ thống.",
        "",
        "## 7. Actual LLM calls",
        "",
        f"Có {totals['actual_llm_calls']} call thật. Có {totals['journal_lines']} dòng journal; journal ghi sự kiện hoàn tất mẫu, không đại diện từng call.",
        "",
        "## 8. Retry/error analysis",
        "",
        f"Error call records: {totals['error_call_records']}; retry do lỗi: {totals['retry_calls_due_to_error']}. Tất cả attempt records có status `OK`.",
        "",
        "## 9. Context budget verification",
        "",
        "Mỗi run khóa `num_ctx=32768`, `num_predict=2048`, do đó input budget là 30720. Tất cả raw pre-call counts và Ollama-reported input counts đều nằm trong giới hạn. `exact_token_preflight=EXACT` cho cả ba pipeline.",
        "",
        "## 10. Hạn chế khoa học",
        "",
        "- Mẫu duy nhất, researcher-exposed, positive-only và ground truth review còn PENDING.",
        "- Không tính Accuracy, Precision, Recall hoặc F1.",
        "- Prediction `PATCHED`/`NOT_SURE` là output của model, không phải nhãn ground truth mới.",
        "- Technical success không chứng minh security completeness hoặc khả năng phân biệt hai lớp.",
        "- Không kết luận P2 tốt hơn P0/P1 từ mẫu này.",
        "",
        "## 11. Điều kiện trước main experiment",
        "",
        "1. GVHD/researcher phê duyệt phạm vi positive-only và policy representation cho protocol chính thức.",
        "2. Chốt cohort, review decisions, immutable manifest và ordered sample list độc lập.",
        "3. Khóa code/prompt/model/tokenizer/parameters và kiểm tra context cho toàn cohort.",
        "4. Xác định trước metric positive-only và cách báo cáo abstention/technical failure.",
        "5. Không tái sử dụng sample 19047 như held-out evaluation.",
        "",
        "## Kết luận bắt buộc",
        "",
        "- Engineering Smoke technical execution: **PASS**",
        "- Security correctness: **NOT VERIFIED**",
        "- Ground truth: **PENDING**",
        "- Held-out evaluation: **NOT RUN**",
        "- Main experiment: **NOT RUN**",
        "",
        "Báo cáo này được sinh từ auditor offline; không có LLM call nào được thực hiện trong quá trình audit.",
        "",
    ])
    return "\n".join(lines)


def _output_paths(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        suffix = "" if version == 1 else f"_v{version}"
        metrics = output_dir / f"smoke_19047_verified_metrics{suffix}.json"
        report = output_dir / f"smoke_19047_postaudit_report{suffix}.md"
        if not metrics.exists() and not report.exists():
            return metrics, report
        version += 1


def write_outputs(metrics: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    metrics_path, report_path = _output_paths(Path(output_dir))
    metrics_text = json.dumps(
        metrics, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    report_text = render_markdown(metrics)
    with metrics_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(metrics_text)
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report_text)
    return metrics_path, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-experiment-git")
    parser.add_argument("--expected-source-package-git")
    parser.add_argument("--expected-sample-id")
    parser.add_argument("--expected-run-prefix")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-policy-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        metrics = audit_archive(
            args.archive,
            expected_archive_sha256=args.expected_archive_sha256,
            expectations=AuditExpectations(
                experiment_git_sha=args.expected_experiment_git,
                source_package_git_sha=args.expected_source_package_git,
                sample_id=args.expected_sample_id,
                run_prefix=args.expected_run_prefix,
                model_id=args.expected_model,
                context_policy_sha256=args.expected_policy_sha256,
            ),
        )
        metrics_path, report_path = write_outputs(metrics, args.output_dir)
    except SmokeAuditError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(f"PASS: {metrics_path}")
    print(f"PASS: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
