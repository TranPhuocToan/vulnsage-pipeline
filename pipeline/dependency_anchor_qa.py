"""Offline textual QA for LLM-generated P2 dependency anchors.

An anchor match only proves that the cited text occurs in the authoritative
diff.  It does not prove that the dependency type, description, or security
interpretation is correct.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


EXACT_MATCH = "EXACT_MATCH"
WHITESPACE_NORMALIZED_MATCH = "WHITESPACE_NORMALIZED_MATCH"
UNSUPPORTED = "UNSUPPORTED"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_whitespace(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def validate_dependency_anchors(
    *,
    original_diff_bytes: bytes | None,
    expected_diff_sha256: str | None,
    dependency_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate anchor presence against exact original-diff bytes, fail closed."""
    base = {
        "validation_status": "BLOCKED",
        "semantic_dependency_correctness": "NOT_ASSESSED",
        "warning": "Text presence is not proof that a claimed dependency is semantically correct.",
        "reason_codes": [],
        "expected_diff_sha256": expected_diff_sha256,
        "actual_diff_sha256": None,
        "anchors": [],
        "summary": {EXACT_MATCH: 0, WHITESPACE_NORMALIZED_MATCH: 0, UNSUPPORTED: 0},
    }
    if not isinstance(original_diff_bytes, bytes) or not original_diff_bytes:
        base["reason_codes"].append("ORIGINAL_DIFF_MISSING")
        return base
    if not isinstance(expected_diff_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_diff_sha256):
        base["reason_codes"].append("EXPECTED_DIFF_SHA256_INVALID")
        return base
    actual_hash = _sha256(original_diff_bytes)
    base["actual_diff_sha256"] = actual_hash
    if actual_hash != expected_diff_sha256:
        base["reason_codes"].append("ORIGINAL_DIFF_HASH_MISMATCH")
        return base
    if not isinstance(dependency_evidence, Mapping):
        base["reason_codes"].append("DEPENDENCY_EVIDENCE_MISSING")
        return base
    dependencies = dependency_evidence.get("dependencies")
    if not isinstance(dependencies, list):
        base["reason_codes"].append("DEPENDENCIES_NOT_LIST")
        return base

    try:
        diff_text = original_diff_bytes.decode("utf-8")
    except UnicodeDecodeError:
        base["reason_codes"].append("ORIGINAL_DIFF_NOT_UTF8")
        return base
    diff_lines = diff_text.splitlines()
    exact_index: dict[str, list[int]] = {}
    normalised_index: dict[str, list[int]] = {}
    for number, line in enumerate(diff_lines, 1):
        exact_index.setdefault(line, []).append(number)
        normalised_index.setdefault(_normalise_whitespace(line), []).append(number)

    for dep_index, dependency in enumerate(dependencies):
        if not isinstance(dependency, Mapping):
            base["reason_codes"].append(f"DEPENDENCY_INVALID:{dep_index}")
            return base
        anchor_lines = dependency.get("anchor_lines")
        if not isinstance(anchor_lines, list) or any(
            not isinstance(anchor, str) or not anchor for anchor in anchor_lines
        ):
            base["reason_codes"].append(f"ANCHOR_LINES_INVALID:{dep_index}")
            return base
        for anchor_index, anchor in enumerate(anchor_lines):
            if anchor in exact_index:
                status = EXACT_MATCH
                matched_lines = exact_index[anchor]
            else:
                matched_lines = normalised_index.get(_normalise_whitespace(anchor), [])
                status = WHITESPACE_NORMALIZED_MATCH if matched_lines else UNSUPPORTED
            base["summary"][status] += 1
            base["anchors"].append({
                "dependency_index": dep_index,
                "anchor_index": anchor_index,
                "source_file": dependency.get("source_file"),
                "target_file": dependency.get("target_file"),
                "anchor": anchor,
                "anchor_sha256": hashlib.sha256(anchor.encode("utf-8")).hexdigest(),
                "status": status,
                "matched_diff_line_numbers": matched_lines,
            })

    base["validation_status"] = "PASS"
    return base
