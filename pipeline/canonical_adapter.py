"""Validated adapter from normalized schema 1.1 to the offline pipeline DTO.

The returned object deliberately separates model-facing data, evaluation-only
ground truth, and audit provenance.  Callers must pass only ``model_input`` to
prompt renderers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "vulnsage-kltndata-1.1"
SUPPORTED_CHANGE_TYPES = {"MODIFY", "ADD", "DELETE", "RENAME"}
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class CanonicalAdapterError(ValueError):
    """Raised when canonical input fails closed."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_sample_key(sample_id: str) -> str:
    """Return a deterministic Windows-safe intermediate filename stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip(" ._")
    if not stem:
        stem = "sample"
    if stem.upper().split(".")[0] in _WINDOWS_RESERVED:
        stem = f"sample_{stem}"
    stem = stem[:80]
    return f"{stem}-{sha256_text(sample_id)[:12]}"


def _default_schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "dataset_validation" / "schemas" / "normalized_sample.schema.json"
    )


def _load_schema(path: Path | None = None) -> Dict[str, Any]:
    with (path or _default_schema_path()).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _verify_text_hash(value: Any, expected: Any, field: str) -> None:
    if value is None:
        if expected is not None:
            raise CanonicalAdapterError(f"{field}: hash present while value is null")
        return
    if not isinstance(value, str) or not isinstance(expected, str):
        raise CanonicalAdapterError(f"{field}: text/hash must both be strings")
    if sha256_text(value) != expected:
        raise CanonicalAdapterError(f"{field}: sha256 mismatch")


def _validate_change_semantics(file_data: Mapping[str, Any], index: int) -> None:
    change_type = file_data.get("change_type")
    before_path = file_data.get("path_before")
    after_path = file_data.get("path_after")
    before_code = file_data.get("vulnerable_code")
    after_code = file_data.get("patched_code")
    before_status = file_data.get("before_provenance", {}).get("status")
    after_status = file_data.get("after_provenance", {}).get("status")
    prefix = f"files[{index}]"

    if change_type not in SUPPORTED_CHANGE_TYPES:
        raise CanonicalAdapterError(f"{prefix}: unsupported change_type={change_type!r}")
    if change_type == "MODIFY" and (not before_path or not after_path):
        raise CanonicalAdapterError(f"{prefix}: MODIFY requires both paths")
    if change_type == "ADD":
        if before_path is not None or not after_path:
            raise CanonicalAdapterError(f"{prefix}: ADD requires null source and target path")
        if before_code is not None or before_status != "MISSING_BY_CHANGE_TYPE":
            raise CanonicalAdapterError(f"{prefix}: ADD source must be missing by change type")
    if change_type == "DELETE":
        if not before_path or after_path is not None:
            raise CanonicalAdapterError(f"{prefix}: DELETE requires source and null target path")
        if after_code is not None or after_status != "MISSING_BY_CHANGE_TYPE":
            raise CanonicalAdapterError(f"{prefix}: DELETE target must be missing by change type")
    if change_type == "RENAME":
        if not before_path or not after_path or before_path == after_path:
            raise CanonicalAdapterError(f"{prefix}: RENAME requires two distinct paths")

    if before_code is None and before_status == "VERIFIED":
        raise CanonicalAdapterError(f"{prefix}: VERIFIED source code is null")
    if after_code is None and after_status == "VERIFIED":
        raise CanonicalAdapterError(f"{prefix}: VERIFIED target code is null")

    _verify_text_hash(
        file_data.get("original_git_diff"),
        file_data.get("original_git_diff_sha256"),
        f"{prefix}.original_git_diff",
    )
    _verify_text_hash(
        file_data.get("reconstructed_git_diff"),
        file_data.get("reconstructed_git_diff_sha256"),
        f"{prefix}.reconstructed_git_diff",
    )


def adapt_sample(sample: Mapping[str, Any], schema_path: Path | None = None) -> Dict[str, Any]:
    """Validate and adapt one schema-1.1 sample without mutating it."""
    original = copy.deepcopy(dict(sample))
    validator = Draft202012Validator(_load_schema(schema_path))
    errors = sorted(validator.iter_errors(original), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "<root>"
        raise CanonicalAdapterError(f"schema validation failed at {location}: {first.message}")
    if original.get("schema_version") != SCHEMA_VERSION:
        raise CanonicalAdapterError("unsupported normalized schema version")

    files: List[Dict[str, Any]] = []
    blocked_reasons: List[str] = []
    approval_reasons: List[str] = []
    seen_path_pairs = set()
    for index, item in enumerate(original["files"]):
        _validate_change_semantics(item, index)
        pair = (item.get("path_before"), item.get("path_after"), item["change_type"])
        if pair in seen_path_pairs:
            raise CanonicalAdapterError(f"files[{index}]: duplicate path/change identity")
        seen_path_pairs.add(pair)
        if item.get("original_git_diff") is None:
            blocked_reasons.append(f"ORIGINAL_DIFF_MISSING:{index}")
        for side in ("before", "after"):
            provenance = item[f"{side}_provenance"]
            if provenance["status"] == "UNVERIFIABLE":
                blocked_reasons.append(f"{side.upper()}_CODE_UNVERIFIABLE:{index}")

        display_path = item.get("path_after") or item.get("path_before") or f"file-{index}"
        files.append({
            "index": index,
            "change_type": item["change_type"],
            "path_before": item.get("path_before"),
            "path_after": item.get("path_after"),
            "display_path": display_path,
            "before_code": item.get("vulnerable_code"),
            "before_code_sha256": item.get("vulnerable_code_sha256"),
            "before_provenance": copy.deepcopy(item["before_provenance"]),
            "after_code": item.get("patched_code"),
            "after_code_sha256": item.get("patched_code_sha256"),
            "after_provenance": copy.deepcopy(item["after_provenance"]),
            "original_diff": item.get("original_git_diff"),
            "original_diff_sha256": item.get("original_git_diff_sha256"),
            "reconstructed_diff": item.get("reconstructed_git_diff"),
            "reconstructed_diff_sha256": item.get("reconstructed_git_diff_sha256"),
            "diff_origin": item.get("diff_origin"),
            "coverage": copy.deepcopy(item.get("coverage")),
        })

    full_patch = original["full_patch"]
    _verify_text_hash(
        full_patch.get("original_git_diff"),
        full_patch.get("original_git_diff_sha256"),
        "full_patch.original_git_diff",
    )
    _verify_text_hash(
        full_patch.get("reconstructed_git_diff"),
        full_patch.get("reconstructed_git_diff_sha256"),
        "full_patch.reconstructed_git_diff",
    )
    if full_patch.get("original_git_diff") is None:
        blocked_reasons.append("FULL_ORIGINAL_DIFF_MISSING")

    ground_truth = original["ground_truth"]
    if ground_truth.get("review_status") != "VERIFIED":
        approval_reasons.append(f"GROUND_TRUTH_{ground_truth.get('review_status', 'UNKNOWN')}")
    quality = original["quality"].get("validation_status")
    verification = original["commit_verification"].get("status")
    if quality in {"FAIL", "UNRESOLVED"}:
        approval_reasons.append(f"QUALITY_{quality}")
    if verification != "VERIFIED":
        approval_reasons.append(f"COMMIT_{verification}")

    status = "BLOCKED" if blocked_reasons else ("NEEDS_APPROVAL" if approval_reasons else "READY")
    identity = original["identity"]
    return {
        "adapter_version": "phase3b-canonical-adapter-1.0",
        "sample_id": original["sample_id"],
        "safe_sample_key": safe_sample_key(original["sample_id"]),
        "family_id": original["family_id"],
        "dataset_version": original["dataset_version"],
        "integration_status": status,
        "reason_codes": sorted(set(blocked_reasons + approval_reasons)),
        "model_input": {
            "cve_id": identity["cve_id"],
            "cwe_ids": copy.deepcopy(identity["cwe_ids"]),
            "project": identity["project"],
            "files": files,
        },
        "evaluation_only": {"ground_truth": copy.deepcopy(ground_truth)},
        "audit_only": {
            "source": copy.deepcopy(original["source"]),
            "identity": copy.deepcopy(identity),
            "granularity": copy.deepcopy(original["granularity"]),
            "quality": copy.deepcopy(original["quality"]),
            "commit_verification": copy.deepcopy(original["commit_verification"]),
            "full_patch": copy.deepcopy(full_patch),
        },
    }


def load_and_adapt_dataset(path: Path, schema_path: Path | None = None) -> List[Dict[str, Any]]:
    """Load a sample array or Phase-2E wrapper and reject duplicate identities."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        samples = payload.get("samples")
    else:
        samples = payload
    if not isinstance(samples, list):
        raise CanonicalAdapterError("dataset must be an array or an object with a samples array")
    seen = set()
    adapted = []
    for raw in samples:
        item = adapt_sample(raw, schema_path=schema_path)
        if item["sample_id"] in seen:
            raise CanonicalAdapterError(f"duplicate sample_id: {item['sample_id']}")
        seen.add(item["sample_id"])
        adapted.append(item)
    return adapted


def dataset_sha256(path: Path) -> str:
    """Hash exact dataset bytes for immutable run specifications."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
