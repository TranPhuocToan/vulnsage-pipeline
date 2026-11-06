"""Offline commit-evidence verification for VulnSage CSV representations.

This module never fetches repositories or mutates source data.  Callers provide a
commit evidence snapshot obtained from a trusted source, then the verifier compares
that snapshot with the code blocks already present in a CSV row.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

from .validator import reconstructed_diff, split_marked_code


COMMIT_VERIFIER_VERSION = "1.0.0"
COMMIT_REASON_CATALOG = {
    "SOURCE_COMMIT_FILE_MISSING": "A changed file in the source commit is absent from the CSV representation.",
    "SOURCE_COMMIT_FILE_EXTRA": "A represented path is not present in the source commit file operations.",
    "FILE_STATUS_MISMATCH": "CSV side presence contradicts the original file-level status.",
    "RECONSTRUCTED_DIFF_EMPTY_BUT_SOURCE_DIFF_NONEMPTY": "The source commit changes lines but the reconstructed diff is empty.",
    "RECONSTRUCTED_CHANGE_COUNT_MISMATCH": "Reconstructed changed-line counts or changed-line content coverage differ from the source commit.",
    "ADDED_DELETED_PATH_CONFLATION": "A deleted path and an added path appear to have been collapsed into one represented path.",
}
VALID_EVIDENCE_STATUSES = {"VERIFIED", "UNVERIFIABLE", "FAILED"}
VALID_FILE_STATUSES = {"MODIFIED", "ADDED", "DELETED", "RENAMED"}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_status(value: Any) -> str:
    aliases = {
        "modified": "MODIFIED",
        "added": "ADDED",
        "removed": "DELETED",
        "deleted": "DELETED",
        "renamed": "RENAMED",
    }
    raw = str(value or "").strip()
    return aliases.get(raw.lower(), raw.upper())


def evidence_from_github_api(
    payload: Mapping[str, Any], source_url: str | None = None
) -> dict[str, Any]:
    """Convert a saved GitHub commit API response into provider-neutral evidence."""

    files = []
    for item in payload.get("files", []):
        patch = item.get("patch")
        files.append(
            {
                "path": item.get("filename"),
                "previous_path": item.get("previous_filename"),
                "status": _normalise_status(item.get("status")),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
                "patch": patch,
                "patch_complete": patch is not None,
            }
        )
    return {
        "verification_status": "VERIFIED",
        "commit_sha": payload.get("sha"),
        "parent_commit_sha": (payload.get("parents") or [{}])[0].get("sha"),
        "source_url": source_url or payload.get("html_url"),
        "files": files,
        "original_diff": None,
        "failure_reason": None,
    }


def _canonical_diff_hash(evidence: Mapping[str, Any]) -> tuple[str | None, str]:
    original = evidence.get("original_diff")
    if isinstance(original, str):
        return _sha256_text(original), "RAW_ORIGINAL_DIFF_UTF8"
    files = []
    for item in evidence.get("files", []):
        files.append(
            {
                "path": item.get("path"),
                "previous_path": item.get("previous_path"),
                "status": _normalise_status(item.get("status")),
                "patch": item.get("patch"),
                "patch_complete": bool(item.get("patch_complete", item.get("patch") is not None)),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
            }
        )
    if not files:
        return None, "UNAVAILABLE"
    canonical = json.dumps(
        sorted(files, key=lambda item: (str(item["path"]), str(item["previous_path"]))),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(canonical), "CANONICAL_FILE_PATCHES_V1"


def _changed_lines(diff: str) -> tuple[Counter[str], Counter[str]]:
    deleted: Counter[str] = Counter()
    added: Counter[str] = Counter()
    for line in diff.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            deleted[line[1:]] += 1
        elif line.startswith("+"):
            added[line[1:]] += 1
    return deleted, added


def _patch_hunks(patch: str) -> list[tuple[Counter[str], Counter[str]]]:
    hunks: list[tuple[Counter[str], Counter[str]]] = []
    current_deleted: Counter[str] | None = None
    current_added: Counter[str] | None = None
    for line in patch.splitlines():
        if line.startswith("@@"):
            current_deleted, current_added = Counter(), Counter()
            hunks.append((current_deleted, current_added))
        elif current_deleted is not None and line.startswith("-") and not line.startswith("---"):
            current_deleted[line[1:]] += 1
        elif current_added is not None and line.startswith("+") and not line.startswith("+++"):
            current_added[line[1:]] += 1
    return hunks


def _patch_context(patch: str) -> Counter[str]:
    context: Counter[str] = Counter()
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith(" "):
            context[line[1:]] += 1
    return context


def _counter_overlap(expected: Counter[str], actual: Counter[str]) -> int:
    return sum((expected & actual).values())


def _counter_contains(actual: Counter[str], expected: Counter[str]) -> bool:
    return all(actual[value] >= count for value, count in expected.items())


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def measure_file_coverage(
    patch: str | None, before_fragment: str | None, after_fragment: str | None, path: str
) -> dict[str, Any]:
    """Measure one original file patch against its CSV-derived fragments."""

    if patch is None:
        unavailable = {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None}
        return {
            "path": dict(unavailable),
            "changed_lines": dict(unavailable),
            "hunks": dict(unavailable),
            "code_context": dict(unavailable),
        }
    represented = before_fragment is not None or after_fragment is not None
    reconstructed = reconstructed_diff(before_fragment or "", after_fragment or "", path)
    actual_deleted, actual_added = _changed_lines(reconstructed)
    expected_deleted, expected_added = _changed_lines(patch)
    changed_total = sum(expected_deleted.values()) + sum(expected_added.values())
    changed_matched = _counter_overlap(expected_deleted, actual_deleted) + _counter_overlap(
        expected_added, actual_added
    )
    hunks = _patch_hunks(patch)
    hunk_matched = sum(
        _counter_contains(actual_deleted, deleted)
        and _counter_contains(actual_added, added)
        for deleted, added in hunks
    )
    context = _patch_context(patch)
    fragment_context = Counter((before_fragment or "").splitlines())
    fragment_context.update((after_fragment or "").splitlines())
    context_total = sum(context.values())
    context_matched = _counter_overlap(context, fragment_context)
    return {
        "path": {
            "status": "MEASURED",
            "matched": int(represented),
            "total": 1,
            "ratio": float(int(represented)),
        },
        "changed_lines": {
            "status": "MEASURED_FROM_PATCH",
            "matched": changed_matched,
            "total": changed_total,
            "ratio": _ratio(changed_matched, changed_total),
        },
        "hunks": {
            "status": "MEASURED_FROM_PATCH",
            "matched": hunk_matched,
            "total": len(hunks),
            "ratio": _ratio(hunk_matched, len(hunks)),
        },
        "code_context": {
            "status": "MEASURED_FROM_PATCH" if context_total else "NOT_APPLICABLE",
            "matched": context_matched,
            "total": context_total,
            "ratio": _ratio(context_matched, context_total),
        },
    }


def verify_commit_evidence(
    row: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare one CSV row with an already acquired commit evidence snapshot."""

    evidence_status = str(evidence.get("verification_status", "FAILED")).upper()
    if evidence_status not in VALID_EVIDENCE_STATUSES:
        evidence_status = "FAILED"
    commit_sha = str(evidence.get("commit_sha") or "")
    expected_sha = str(row.get("COMMIT_HASH") or "")
    diff_hash, diff_hash_scope = _canonical_diff_hash(evidence)

    base = {
        "verifier_version": COMMIT_VERIFIER_VERSION,
        "verification_status": evidence_status,
        "security_semantics_verified": False,
        "commit_sha": commit_sha or None,
        "parent_commit_sha": evidence.get("parent_commit_sha"),
        "verification_source_url": evidence.get("source_url"),
        "original_diff_sha256": diff_hash,
        "original_diff_hash_scope": diff_hash_scope,
        "failure_reason": evidence.get("failure_reason"),
        "reason_codes": [],
        "original_files": [],
        "missing_commit_paths": [],
        "extra_represented_paths": [],
        "coverage": {
            "path": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
            "changed_lines": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
            "hunks": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
            "code_context": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
        },
    }
    if evidence_status != "VERIFIED":
        return base
    if not commit_sha or (expected_sha and commit_sha.lower() != expected_sha.lower()):
        base["verification_status"] = "FAILED"
        base["failure_reason"] = "Evidence commit SHA does not match the CSV row"
        return base

    vulnerable = split_marked_code(row.get("VULNERABLE_CODE_BLOCK"))
    patched = split_marked_code(row.get("PATCHED_CODE_BLOCK"))
    vulnerable_paths, patched_paths = set(vulnerable.files), set(patched.files)
    represented_paths = vulnerable_paths | patched_paths
    reconstructed = {
        path: reconstructed_diff(
            vulnerable.files.get(path, ""), patched.files.get(path, ""), path
        )
        for path in represented_paths
    }

    reasons: set[str] = set()
    expected_path_universe: set[str] = set()
    matched_operations = 0
    missing_paths: list[str] = []
    deleted_files: list[Mapping[str, Any]] = []
    added_files: list[Mapping[str, Any]] = []
    original_files: list[dict[str, Any]] = []

    for raw in evidence.get("files", []):
        status = _normalise_status(raw.get("status"))
        path = str(raw.get("path") or "")
        previous = str(raw.get("previous_path") or "") or None
        if status not in VALID_FILE_STATUSES or not path:
            base["verification_status"] = "FAILED"
            base["failure_reason"] = "Commit evidence contains an invalid file operation"
            return base
        if status == "RENAMED" and previous:
            expected_path_universe.update((previous, path))
            matched = previous in vulnerable_paths and path in patched_paths
        elif status == "ADDED":
            expected_path_universe.add(path)
            matched = path in patched_paths
            added_files.append(raw)
            if path in vulnerable_paths and vulnerable.files.get(path, "").strip():
                reasons.add("FILE_STATUS_MISMATCH")
        elif status == "DELETED":
            expected_path_universe.add(path)
            matched = path in vulnerable_paths
            deleted_files.append(raw)
            if path in patched_paths and patched.files.get(path, "").strip():
                reasons.add("FILE_STATUS_MISMATCH")
        else:
            expected_path_universe.add(path)
            matched = path in represented_paths
        if matched:
            matched_operations += 1
        else:
            missing_paths.append(previous or path if status == "RENAMED" else path)

        patch = raw.get("patch")
        original_files.append(
            {
                "path": path,
                "previous_path": previous,
                "status": status,
                "additions": raw.get("additions"),
                "deletions": raw.get("deletions"),
                "patch_sha256": _sha256_text(patch) if isinstance(patch, str) else None,
                "patch_complete": bool(raw.get("patch_complete", patch is not None)),
            }
        )

    extra_paths = sorted(represented_paths - expected_path_universe)
    if missing_paths:
        reasons.add("SOURCE_COMMIT_FILE_MISSING")
    if extra_paths:
        reasons.add("SOURCE_COMMIT_FILE_EXTRA")

    for added in added_files:
        added_path = str(added.get("path") or "")
        if added_path in vulnerable_paths and added_path in patched_paths:
            if any(str(item.get("path") or "") not in represented_paths for item in deleted_files):
                reasons.add("ADDED_DELETED_PATH_CONFLATION")

    original_added = sum(
        int(item.get("additions")) for item in evidence.get("files", []) if item.get("additions") is not None
    )
    original_deleted = sum(
        int(item.get("deletions")) for item in evidence.get("files", []) if item.get("deletions") is not None
    )
    count_complete = all(
        item.get("additions") is not None and item.get("deletions") is not None
        for item in evidence.get("files", [])
    )
    reconstructed_deleted: Counter[str] = Counter()
    reconstructed_added: Counter[str] = Counter()
    per_path_changed: dict[str, tuple[Counter[str], Counter[str]]] = {}
    for path, diff in reconstructed.items():
        deleted, added = _changed_lines(diff)
        per_path_changed[path] = (deleted, added)
        reconstructed_deleted.update(deleted)
        reconstructed_added.update(added)
    reconstructed_total = sum(reconstructed_deleted.values()) + sum(reconstructed_added.values())
    original_total = original_added + original_deleted
    if original_total > 0 and reconstructed_total == 0:
        reasons.add("RECONSTRUCTED_DIFF_EMPTY_BUT_SOURCE_DIFF_NONEMPTY")
    if count_complete and (
        original_added != sum(reconstructed_added.values())
        or original_deleted != sum(reconstructed_deleted.values())
    ):
        reasons.add("RECONSTRUCTED_CHANGE_COUNT_MISMATCH")

    patch_complete = all(
        isinstance(item.get("patch"), str)
        and bool(item.get("patch_complete", True))
        for item in evidence.get("files", [])
    )
    matched_lines = total_patch_lines = matched_hunks = total_hunks = 0
    matched_context = total_context = 0
    if patch_complete:
        for item in evidence.get("files", []):
            path = str(item.get("path") or "")
            previous = str(item.get("previous_path") or "")
            candidates = [path]
            if previous:
                candidates.append(previous)
            actual_deleted: Counter[str] = Counter()
            actual_added: Counter[str] = Counter()
            context_code: Counter[str] = Counter()
            for candidate in candidates:
                deleted, added = per_path_changed.get(candidate, (Counter(), Counter()))
                actual_deleted.update(deleted)
                actual_added.update(added)
                context_code.update(vulnerable.files.get(candidate, "").splitlines())
                context_code.update(patched.files.get(candidate, "").splitlines())
            patch = str(item.get("patch"))
            expected_deleted, expected_added = _changed_lines(patch)
            total_patch_lines += sum(expected_deleted.values()) + sum(expected_added.values())
            matched_lines += _counter_overlap(expected_deleted, actual_deleted)
            matched_lines += _counter_overlap(expected_added, actual_added)
            hunks = _patch_hunks(patch)
            total_hunks += len(hunks)
            matched_hunks += sum(
                _counter_contains(actual_deleted, deleted)
                and _counter_contains(actual_added, added)
                for deleted, added in hunks
            )
            context = _patch_context(patch)
            total_context += sum(context.values())
            matched_context += _counter_overlap(context, context_code)
        if matched_lines < total_patch_lines:
            reasons.add("RECONSTRUCTED_CHANGE_COUNT_MISMATCH")

    total_operations = len(evidence.get("files", []))
    base["coverage"]["path"] = {
        "status": "MEASURED",
        "matched": matched_operations,
        "total": total_operations,
        "ratio": _ratio(matched_operations, total_operations),
    }
    if patch_complete:
        base["coverage"]["changed_lines"] = {
            "status": "MEASURED_FROM_PATCH",
            "matched": matched_lines,
            "total": total_patch_lines,
            "ratio": _ratio(matched_lines, total_patch_lines),
        }
        base["coverage"]["hunks"] = {
            "status": "MEASURED_FROM_PATCH",
            "matched": matched_hunks,
            "total": total_hunks,
            "ratio": _ratio(matched_hunks, total_hunks),
        }
        base["coverage"]["code_context"] = {
            "status": "MEASURED_FROM_PATCH" if total_context else "NOT_APPLICABLE",
            "matched": matched_context,
            "total": total_context,
            "ratio": _ratio(matched_context, total_context),
        }
    base["verification_status"] = "VERIFIED_WITH_GAPS" if reasons else "VERIFIED"
    base["reason_codes"] = sorted(reasons)
    base["original_files"] = original_files
    base["missing_commit_paths"] = sorted(set(missing_paths))
    base["extra_represented_paths"] = extra_paths
    base["reconstructed_change_counts"] = {
        "additions": sum(reconstructed_added.values()),
        "deletions": sum(reconstructed_deleted.values()),
    }
    base["original_change_counts"] = {
        "additions": original_added if count_complete else None,
        "deletions": original_deleted if count_complete else None,
    }
    return base
