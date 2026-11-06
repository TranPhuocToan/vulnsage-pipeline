"""Phase 3C.2 offline qualification for canonical positive-only samples.

Qualification never mutates the canonical sample or its ground-truth object.
Human decisions and waivers are separate, attributable artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


QUALIFIER_VERSION = "phase3c2-qualifier-1.0"
STATUSES = {"READY", "NEEDS_APPROVAL", "BLOCKED", "OUT_OF_SCOPE", "NOT_ASSESSED"}
DECISIONS = {"APPROVE", "REJECT", "DEFER"}
SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_OPERATIONS = {"MODIFY", "ADD", "DELETE", "RENAME"}
DEFAULT_CODE_EXTENSIONS = frozenset({
    ".c", ".h", ".cc", ".hh", ".cpp", ".hpp", ".cxx", ".hxx", ".s", ".asm", ".inc"
})

# These describe loss/mismatch in the CSV-derived representation.  They are
# reviewable only when the canonical original diff, operations and blobs pass.
DERIVED_GAP_CODES = frozenset({
    "ADDED_DELETED_PATH_CONFLATION",
    "ALL_RECONSTRUCTED_DIFFS_EMPTY",
    "EMPTY_PATCHED_CONTENT",
    "EMPTY_VULNERABLE_CONTENT",
    "FILE_COUNT_MISMATCH",
    "FILE_STATUS_MISMATCH",
    "PATH_SET_MISMATCH_REVIEW",
    "RECONSTRUCTED_CHANGE_COUNT_MISMATCH",
    "RECONSTRUCTED_DIFF_EMPTY_BUT_SOURCE_DIFF_NONEMPTY",
    "SOURCE_COMMIT_FILE_EXTRA",
    "SOURCE_COMMIT_FILE_MISSING",
    "UNCHANGED_FILE_BLOCK",
})

# This is a cohort/family review signal, not evidence that canonical commit
# content is missing.  It is handled explicitly by ``family_requires_review``.
CONTROL_REASON_CODES = frozenset({"DUPLICATE_CVE_FAMILY"})


class QualificationError(ValueError):
    pass


@dataclass(frozen=True)
class QualificationPolicy:
    policy_version: str = "phase3c2-positive-only-multifile-1.0"
    minimum_code_files: int = 2
    code_extensions: tuple[str, ...] = tuple(sorted(DEFAULT_CODE_EXTENSIONS))
    require_human_approval: bool = True

    def sha256(self) -> str:
        return _hash_json(asdict(self))


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise QualificationError("decided_at_utc must be an RFC3339 UTC string ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise QualificationError("invalid decided_at_utc") from exc


def validate_review_decision(decision: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    """Validate an attributable decision without applying it to ground truth."""
    required = {
        "decision_id", "sample_id", "decision", "reviewer_id", "reviewer_name",
        "reason", "decided_at_utc", "evidence_refs", "waivers",
    }
    missing = sorted(required - set(decision))
    if missing:
        raise QualificationError(f"review decision missing fields: {missing}")
    if decision["sample_id"] != sample_id:
        raise QualificationError("review decision sample_id mismatch")
    if decision["decision"] not in DECISIONS:
        raise QualificationError("invalid review decision enum")
    for field in ("decision_id", "reviewer_id", "reviewer_name", "reason"):
        if not isinstance(decision[field], str) or not decision[field].strip():
            raise QualificationError(f"{field} must be a non-empty string")
    _parse_time(decision["decided_at_utc"])
    if not isinstance(decision["evidence_refs"], list) or not all(
        isinstance(item, str) and item for item in decision["evidence_refs"]
    ):
        raise QualificationError("evidence_refs must be a list of non-empty strings")
    if not isinstance(decision["waivers"], list):
        raise QualificationError("waivers must be an array")
    seen = set()
    for index, waiver in enumerate(decision["waivers"]):
        if not isinstance(waiver, Mapping):
            raise QualificationError(f"waivers[{index}] must be an object")
        if set(waiver) != {"reason_code", "scope", "reason", "evidence_refs"}:
            raise QualificationError(f"waivers[{index}] has invalid fields")
        code = waiver["reason_code"]
        if code not in DERIVED_GAP_CODES:
            raise QualificationError(f"waiver not allowed for canonical/non-derived gap: {code}")
        if waiver["scope"] != "DERIVED_REPRESENTATION_ONLY":
            raise QualificationError("waiver scope must be DERIVED_REPRESENTATION_ONLY")
        if not isinstance(waiver["reason"], str) or not waiver["reason"].strip():
            raise QualificationError("waiver reason must be non-empty")
        if not isinstance(waiver["evidence_refs"], list) or not waiver["evidence_refs"]:
            raise QualificationError("waiver evidence_refs must be non-empty")
        if code in seen:
            raise QualificationError(f"duplicate waiver: {code}")
        seen.add(code)
    return json.loads(json.dumps(decision, ensure_ascii=False))


def _is_code_path(path: str | None, policy: QualificationPolicy) -> bool:
    if not path:
        return False
    return PurePosixPath(path).suffix.lower() in set(policy.code_extensions)


def _canonical_blockers(sample: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    identity = sample.get("identity", {})
    if not SHA40.fullmatch(str(identity.get("fix_commit_hash") or "")):
        blockers.append("FIX_COMMIT_INVALID")
    if not SHA40.fullmatch(str(identity.get("parent_commit_hash") or "")):
        blockers.append("PARENT_COMMIT_INVALID")
    verification = sample.get("commit_verification", {})
    if verification.get("status") in {"FAILED", "UNVERIFIABLE"}:
        blockers.append(f"COMMIT_{verification.get('status')}")
    full_patch = sample.get("full_patch", {})
    original = full_patch.get("original_git_diff")
    original_hash = full_patch.get("original_git_diff_sha256")
    if not isinstance(original, str) or not original:
        blockers.append("ORIGINAL_COMMIT_DIFF_MISSING")
    elif not isinstance(original_hash, str) or _hash_text(original) != original_hash:
        blockers.append("ORIGINAL_COMMIT_DIFF_HASH_MISMATCH")

    for index, file_data in enumerate(sample.get("files", [])):
        operation = file_data.get("change_type")
        prefix = f"FILE_{index}"
        if operation not in SUPPORTED_OPERATIONS:
            blockers.append(f"{prefix}_OPERATION_INVALID")
            continue
        patch = file_data.get("original_git_diff")
        patch_hash = file_data.get("original_git_diff_sha256")
        if not isinstance(patch, str) or not patch:
            blockers.append(f"{prefix}_ORIGINAL_DIFF_MISSING")
        elif not isinstance(patch_hash, str) or _hash_text(patch) != patch_hash:
            blockers.append(f"{prefix}_ORIGINAL_DIFF_HASH_MISMATCH")
        for side in ("before", "after"):
            provenance = file_data.get(f"{side}_provenance", {})
            status = provenance.get("status")
            structurally_missing = (
                operation == "ADD" and side == "before"
            ) or (operation == "DELETE" and side == "after")
            expected = "MISSING_BY_CHANGE_TYPE" if structurally_missing else "VERIFIED"
            if status != expected:
                blockers.append(f"{prefix}_{side.upper()}_PROVENANCE_{status or 'MISSING'}")
            if status == "VERIFIED" and not SHA256.fullmatch(str(provenance.get("blob_sha256") or "")):
                blockers.append(f"{prefix}_{side.upper()}_BLOB_HASH_INVALID")
    return sorted(set(blockers))


def _derived_gaps(sample: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    known: set[str] = set()
    unknown: set[str] = set()
    reasons = list(sample.get("quality", {}).get("reason_codes", []))
    reasons += list(sample.get("commit_verification", {}).get("reason_codes", []))
    for reason in reasons:
        if reason in CONTROL_REASON_CODES:
            continue
        (known if reason in DERIVED_GAP_CODES else unknown).add(str(reason))
    return sorted(known), sorted(unknown)


def qualify_sample(
    sample: Mapping[str, Any],
    *,
    decision: Mapping[str, Any] | None = None,
    policy: QualificationPolicy | None = None,
    family_requires_review: bool = False,
) -> dict[str, Any]:
    """Return a qualification record; never modify ``sample``."""
    policy = policy or QualificationPolicy()
    sample_id = str(sample.get("sample_id") or "")
    if not sample_id:
        raise QualificationError("sample_id is required")
    files = list(sample.get("files", []))
    code_paths = {
        path
        for item in files
        for path in (item.get("path_before"), item.get("path_after"))
        if _is_code_path(path, policy)
    }
    declared = sample.get("granularity", {}).get("declared_num_files")
    original_operation_count = len(files)
    in_scope = (
        isinstance(declared, int)
        and declared > 1
        and original_operation_count >= policy.minimum_code_files
        and len(code_paths) >= policy.minimum_code_files
    )
    blockers = _canonical_blockers(sample)
    derived_gaps, unknown_gaps = _derived_gaps(sample)
    blockers.extend(f"UNCLASSIFIED_GAP:{item}" for item in unknown_gaps)
    review = validate_review_decision(decision, sample_id) if decision is not None else None
    waived = {item["reason_code"] for item in review["waivers"]} if review else set()
    unwaived = sorted(set(derived_gaps) - waived)
    reason_codes: list[str] = []

    if not in_scope:
        status = "OUT_OF_SCOPE"
        reason_codes.append("OPERATIONAL_MULTIFILE_SCOPE_NOT_MET")
    elif blockers:
        status = "BLOCKED"
        reason_codes.extend(blockers)
    elif review and review["decision"] == "REJECT":
        status = "BLOCKED"
        reason_codes.append("HUMAN_REVIEW_REJECTED")
    else:
        approval_reasons = []
        if family_requires_review:
            approval_reasons.append("MULTI_COMMIT_FAMILY_REVIEW_REQUIRED")
        if unwaived:
            approval_reasons.extend(f"DERIVED_GAP_UNWAIVED:{item}" for item in unwaived)
        if not review or review["decision"] != "APPROVE":
            approval_reasons.append("HUMAN_REVIEW_PENDING")
        if policy.require_human_approval and sample.get("ground_truth", {}).get("review_status") == "PENDING":
            # The canonical label remains PENDING even when an external decision
            # is supplied; the decision is carried separately and auditable.
            if not review or review["decision"] != "APPROVE":
                approval_reasons.append("GROUND_TRUTH_PENDING")
        status = "NEEDS_APPROVAL" if approval_reasons else "READY"
        reason_codes.extend(approval_reasons)

    return {
        "qualification_version": QUALIFIER_VERSION,
        "policy_version": policy.policy_version,
        "policy_sha256": policy.sha256(),
        "sample_id": sample_id,
        "source_row_id": str(sample.get("source", {}).get("source_row_id", "")),
        "family_id": sample.get("family_id"),
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        "scope_evidence": {
            "declared_num_files": declared,
            "original_operation_count": original_operation_count,
            "code_file_count": len(code_paths),
            "code_paths": sorted(code_paths),
        },
        "canonical_blockers": sorted(set(blockers)),
        "derived_representation_gaps": derived_gaps,
        "waived_derived_gaps": sorted(waived),
        "unwaived_derived_gaps": unwaived,
        "commit_verification_status": sample.get("commit_verification", {}).get("status"),
        "source_quality_status": sample.get("quality", {}).get("validation_status"),
        "ground_truth_review_status_unchanged": sample.get("ground_truth", {}).get("review_status"),
        "security_correctness_verified": False,
        "review_decision": review,
        "input_sample_sha256": _hash_json(sample),
    }


def not_assessed_record(source_row_id: str, reason: str = "CANONICAL_SAMPLE_NOT_AVAILABLE") -> dict[str, Any]:
    return {
        "qualification_version": QUALIFIER_VERSION,
        "sample_id": None,
        "source_row_id": str(source_row_id),
        "family_id": None,
        "status": "NOT_ASSESSED",
        "reason_codes": [reason],
        "security_correctness_verified": False,
    }
