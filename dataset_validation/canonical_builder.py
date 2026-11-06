"""Build canonical normalized samples from CSV, commit evidence, and targeted blobs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .commit_verification import (
    evidence_from_github_api,
    measure_file_coverage,
    verify_commit_evidence,
)
from .validator import reconstructed_diff, split_marked_code


CANONICAL_BUILDER_VERSION = "1.0.0"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_verified_resource(
    manifest: Mapping[str, Any], manifest_dir: Path, *, kind: str, side: str | None, path: str | None
) -> tuple[str | None, dict[str, Any]]:
    match = next(
        (
            item
            for item in manifest.get("resources", [])
            if item.get("kind") == kind and item.get("side") == side and item.get("path") == path
        ),
        None,
    )
    if match is None:
        return None, {
            "status": "UNVERIFIABLE",
            "source": "UNAVAILABLE",
            "commit_sha": None,
            "path": path,
            "blob_sha256": None,
        }
    target = manifest_dir / str(match["local_path"])
    data = target.read_bytes()
    digest = _sha256_bytes(data)
    if digest != match.get("sha256"):
        raise ValueError(f"Targeted artifact hash mismatch: {target}")
    return data.decode("utf-8", errors="replace"), {
        "status": "VERIFIED",
        "source": "TARGETED_COMMIT_BLOB",
        "commit_sha": match.get("commit_sha"),
        "path": path,
        "blob_sha256": digest,
    }


def _missing_by_change_type() -> dict[str, Any]:
    return {
        "status": "MISSING_BY_CHANGE_TYPE",
        "source": "NOT_APPLICABLE",
        "commit_sha": None,
        "path": None,
        "blob_sha256": None,
    }


def _hunk_headers(patch: str | None) -> list[dict[str, str]]:
    if not patch:
        return []
    return [{"header": line} for line in patch.splitlines() if line.startswith("@@")]


def _noise(value: Any) -> float | None:
    try:
        return float(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def build_canonical_sample(
    row: Mapping[str, Any],
    validation_record: Mapping[str, Any],
    github_evidence: Mapping[str, Any],
    retrieval_manifest: Mapping[str, Any],
    *,
    retrieval_manifest_dir: Path,
    source_file: str,
    source_file_sha256: str,
    source_repository_url: str,
    dataset_version: str,
    ground_truth: Mapping[str, Any],
    evidence_snapshot_sha256: str,
    retrieval_manifest_sha256: str,
) -> dict[str, Any]:
    """Build one normalized sample. Missing required blobs remain null and flagged."""

    evidence = evidence_from_github_api(github_evidence)
    verification = verify_commit_evidence(row, evidence)
    commit_sha = str(github_evidence.get("sha") or "")
    parents = github_evidence.get("parents") or []
    parent_sha = str(parents[0].get("sha") or "") if parents else ""
    if commit_sha != str(row.get("COMMIT_HASH") or ""):
        raise ValueError("Commit evidence does not match source record")
    if retrieval_manifest.get("commit_sha") != commit_sha:
        raise ValueError("Retrieval manifest commit does not match evidence")
    if retrieval_manifest.get("parent_commit_sha") != parent_sha:
        raise ValueError("Retrieval manifest parent does not match evidence")

    original_full, _ = _read_verified_resource(
        retrieval_manifest,
        retrieval_manifest_dir,
        kind="ORIGINAL_DIFF",
        side=None,
        path=None,
    )
    vulnerable = split_marked_code(row.get("VULNERABLE_CODE_BLOCK"))
    patched = split_marked_code(row.get("PATCHED_CODE_BLOCK"))
    normalized_files = []
    reconstructed_chunks = []

    for item in github_evidence.get("files", []):
        raw_status = str(item.get("status") or "").lower()
        change_type = {
            "modified": "MODIFY",
            "added": "ADD",
            "removed": "DELETE",
            "renamed": "RENAME",
        }.get(raw_status, "UNRESOLVED")
        path_after = None if raw_status == "removed" else str(item.get("filename") or "")
        path_before = None if raw_status == "added" else str(
            item.get("previous_filename") or item.get("filename") or ""
        )

        if path_before is None:
            before_code, before_provenance = None, _missing_by_change_type()
        else:
            before_code, before_provenance = _read_verified_resource(
                retrieval_manifest,
                retrieval_manifest_dir,
                kind="BLOB",
                side="BEFORE",
                path=path_before,
            )
        if path_after is None:
            after_code, after_provenance = None, _missing_by_change_type()
        else:
            after_code, after_provenance = _read_verified_resource(
                retrieval_manifest,
                retrieval_manifest_dir,
                kind="BLOB",
                side="AFTER",
                path=path_after,
            )

        fragment_before = vulnerable.files.get(path_before or path_after or "")
        fragment_after = patched.files.get(path_after or path_before or "")
        diff_path = path_after or path_before or "unknown"
        reconstructed = None
        if fragment_before is not None or fragment_after is not None:
            reconstructed = reconstructed_diff(
                fragment_before or "", fragment_after or "", diff_path
            )
            if reconstructed:
                reconstructed_chunks.append(reconstructed)
        original_patch = item.get("patch")
        coverage = measure_file_coverage(
            original_patch, fragment_before, fragment_after, diff_path
        )
        flags = []
        if coverage["path"]["ratio"] == 0:
            flags.append("CSV_REPRESENTATION_MISSING")
        if before_provenance["status"] == "UNVERIFIABLE":
            flags.append("BEFORE_BLOB_UNVERIFIABLE")
        if after_provenance["status"] == "UNVERIFIABLE":
            flags.append("AFTER_BLOB_UNVERIFIABLE")
        if coverage["changed_lines"]["ratio"] not in (1.0, None):
            flags.append("PARTIAL_CHANGED_LINE_COVERAGE")
        normalized_files.append(
            {
                "path_before": path_before,
                "path_after": path_after,
                "change_type": change_type,
                "vulnerable_code": before_code,
                "patched_code": after_code,
                "vulnerable_code_sha256": _sha256_text(before_code) if before_code is not None else None,
                "patched_code_sha256": _sha256_text(after_code) if after_code is not None else None,
                "before_provenance": before_provenance,
                "after_provenance": after_provenance,
                "original_git_diff": original_patch,
                "original_git_diff_sha256": _sha256_text(original_patch) if original_patch else None,
                "reconstructed_git_diff": reconstructed,
                "reconstructed_git_diff_sha256": _sha256_text(reconstructed) if reconstructed is not None else None,
                "diff_origin": "BOTH" if reconstructed is not None and original_patch else (
                    "SOURCE_COMMIT" if original_patch else "MISSING"
                ),
                "changed_hunks": _hunk_headers(original_patch),
                "coverage": coverage,
                "validation_flags": sorted(flags),
            }
        )

    combined_reconstructed = "\n".join(reconstructed_chunks) or None
    reasons = sorted(
        set(validation_record.get("reason_codes", [])) | set(verification.get("reason_codes", []))
    )
    missing_blob = any(
        flag.endswith("BLOB_UNVERIFIABLE")
        for item in normalized_files
        for flag in item["validation_flags"]
    )
    if missing_blob or verification["verification_status"] in {"FAILED", "UNVERIFIABLE"}:
        quality_status = "FAIL"
    elif reasons:
        quality_status = "UNRESOLVED"
    else:
        quality_status = validation_record.get("status", "PASS")

    commit_verification = {
        "status": verification["verification_status"],
        "commit_sha": commit_sha,
        "parent_commit_sha": parent_sha,
        "source_url": evidence["source_url"],
        "reason_codes": list(verification["reason_codes"]),
        "coverage": deepcopy(verification["coverage"]),
        "security_semantics_verified": False,
    }
    input_fingerprint = {
        "builder_version": CANONICAL_BUILDER_VERSION,
        "row": dict(row),
        "validation_record": dict(validation_record),
        "evidence_sha256": evidence_snapshot_sha256,
        "retrieval_manifest_sha256": retrieval_manifest_sha256,
        "ground_truth": dict(ground_truth),
        "dataset_version": dataset_version,
    }
    parsed_num_files = len(set(vulnerable.files) | set(patched.files))
    declared_files = int(row["NUM_FILES_CHANGED"]) if str(row.get("NUM_FILES_CHANGED", "")).isdigit() else None
    declared_functions = int(row["NUM_FUNCTIONS_CHANGED"]) if str(row.get("NUM_FUNCTIONS_CHANGED", "")).isdigit() else None
    granularity_label = "G3" if declared_files is not None and declared_files > 1 else "UNRESOLVED"
    return {
        "schema_version": "vulnsage-kltndata-1.1",
        "sample_id": f"vs:{source_file_sha256[:12]}:{row.get('id')}:canonical",
        "family_id": str(row.get("VULNERABILITY_CVE") or row.get("id")),
        "dataset_version": dataset_version,
        "source": {
            "dataset_name": "VulnSage",
            "repository_url": source_repository_url,
            "source_file": source_file,
            "source_file_sha256": source_file_sha256,
            "source_row_id": str(row.get("id")),
            "source_commit_or_release": None,
        },
        "identity": {
            "cve_id": row.get("VULNERABILITY_CVE"),
            "cwe_ids": [row.get("VULNERABILITY_CWE")],
            "project": row.get("PROJECT"),
            "fix_commit_hash": commit_sha,
            "parent_commit_hash": parent_sha,
        },
        "granularity": {
            "label": granularity_label,
            "rule_id": "operational-num-files-gt-1-not-final",
            "declared_num_files": declared_files,
            "declared_num_functions": declared_functions,
            "parsed_num_files": parsed_num_files,
            "derivation_status": "UNRESOLVED",
        },
        "files": normalized_files,
        "full_patch": {
            "original_git_diff": original_full,
            "original_git_diff_sha256": _sha256_text(original_full) if original_full is not None else None,
            "original_git_diff_hash_scope": "GITHUB_COMMIT_PATCH_UTF8",
            "reconstructed_git_diff": combined_reconstructed,
            "reconstructed_git_diff_sha256": _sha256_text(combined_reconstructed) if combined_reconstructed is not None else None,
            "canonical_source": "ORIGINAL_COMMIT_DIFF",
        },
        "ground_truth": deepcopy(dict(ground_truth)),
        "variant": {
            "is_variant": False,
            "parent_sample_id": None,
            "transformation_type": None,
            "transformation_spec_sha256": None,
            "removed_hunks": [],
        },
        "quality": {
            "validation_status": quality_status,
            "reason_codes": reasons,
            "noise_score": _noise(row.get("NOISE_AMOUNT")),
            "file_marker_status": "PASS" if vulnerable.marker_count and patched.marker_count else "FAIL",
            "metadata_consistency": "PASS" if not reasons else "UNRESOLVED",
            "diff_round_trip": "NOT_CHECKED",
            "function_boundary_check": "HEURISTIC_ONLY",
            "validation_flags": sorted(set(flag for item in normalized_files for flag in item["validation_flags"])),
        },
        "commit_verification": commit_verification,
        "build": {
            "builder_version": CANONICAL_BUILDER_VERSION,
            "evidence_snapshot_sha256": evidence_snapshot_sha256,
            "retrieval_manifest_sha256": retrieval_manifest_sha256,
            "deterministic_input_sha256": _canonical_hash(input_fingerprint),
            "prototype_only": True,
        },
    }


def ensure_unique_sample_ids(samples: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(sample.get("sample_id")) for sample in samples]
    duplicates = sorted(value for value in set(ids) if ids.count(value) > 1)
    if duplicates:
        raise ValueError(f"Duplicate canonical sample identity: {duplicates}")
