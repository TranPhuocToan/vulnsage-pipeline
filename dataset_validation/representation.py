"""Build provenance-preserving derived representations without altering source rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .validator import reconstructed_diff, split_marked_code


DERIVED_REPRESENTATION_VERSION = "vulnsage-derived-1.0"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nullable_hash(value: str | None) -> str | None:
    return _hash_text(value) if value is not None else None


def build_derived_representation(
    row: Mapping[str, Any],
    validation_record: Mapping[str, Any],
    commit_verification: Mapping[str, Any],
    *,
    source_file: str,
    source_file_sha256: str,
) -> dict[str, Any]:
    """Create a derived artifact; missing code remains null and is never fabricated."""

    vulnerable = split_marked_code(row.get("VULNERABLE_CODE_BLOCK"))
    patched = split_marked_code(row.get("PATCHED_CODE_BLOCK"))
    paths = sorted(set(vulnerable.files) | set(patched.files))
    extracted_files = []
    reconstructed_files = []
    for path in paths:
        before_present = path in vulnerable.files
        after_present = path in patched.files
        before = vulnerable.files.get(path) if before_present else None
        after = patched.files.get(path) if after_present else None
        extracted_files.append(
            {
                "path": path,
                "before_present": before_present,
                "after_present": after_present,
                "vulnerable_code": before,
                "patched_code": after,
                "vulnerable_code_sha256": _nullable_hash(before),
                "patched_code_sha256": _nullable_hash(after),
                "provenance": "VULNSAGE_CSV_CODE_BLOCK",
            }
        )
        diff = reconstructed_diff(before or "", after or "", path)
        reconstructed_files.append(
            {
                "path": path,
                "diff": diff,
                "diff_sha256": _hash_text(diff),
                "provenance": "RECONSTRUCTED_FROM_CSV_BLOCKS",
            }
        )

    missing_paths = list(commit_verification.get("missing_commit_paths", []))
    recovery_candidates = [
        {
            "path": path,
            "available_from": "ORIGINAL_COMMIT_DIFF",
            "recovered_code": None,
            "recovery_status": "NOT_PERFORMED",
        }
        for path in missing_paths
    ]
    unverified = []
    for name, coverage in commit_verification.get("coverage", {}).items():
        if coverage.get("status") in {"UNVERIFIED", "NOT_APPLICABLE"}:
            unverified.append(f"coverage.{name}:{coverage.get('status')}")
    unverified.extend(commit_verification.get("reason_codes", []))

    canonical_row = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "schema_version": DERIVED_REPRESENTATION_VERSION,
        "sample_id": f"vulnsage:{source_file_sha256[:12]}:{row.get('id')}:derived",
        "source": {
            "source_file": source_file,
            "source_file_sha256": source_file_sha256,
            "source_row_id": str(row.get("id", "")),
            "source_row_sha256": _hash_text(canonical_row),
            "source_data_modified": False,
        },
        "identity": {
            "cve_id": row.get("VULNERABILITY_CVE"),
            "cwe_id": row.get("VULNERABILITY_CWE"),
            "project": row.get("PROJECT"),
            "commit_sha": row.get("COMMIT_HASH"),
        },
        "csv_extraction": {
            "files": extracted_files,
            "validation_status": validation_record.get("status"),
            "validation_reason_codes": list(validation_record.get("reason_codes", [])),
        },
        "original_commit": {
            "verification": dict(commit_verification),
            "provenance": "EXTERNAL_COMMIT_EVIDENCE",
        },
        "reconstructed_diff": {
            "files": reconstructed_files,
            "provenance": "DERIVED_NOT_ORIGINAL",
        },
        "recovery": {
            "performed": False,
            "candidates": recovery_candidates,
            "policy": "NO_CODE_IS_FILLED_WITHOUT_EXPLICIT_COMMIT_PROVENANCE",
        },
        "unverified": sorted(set(unverified)),
    }
