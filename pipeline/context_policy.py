"""Shared, auditable model-context policy for P0/P1/P2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping

from pipeline.lossless_context import (
    LosslessContextError, original_modify_section, replay_original_modify_diff,
)


FULL_BLOBS = "full_before_after_v1"
BEFORE_PLUS_DIFF = "before_plus_original_diff_v1"


class ContextPolicyError(ValueError):
    pass


class ContextOverflowError(ContextPolicyError):
    def __init__(self, stage: str, measured: int, limit: int) -> None:
        super().__init__(f"CONTEXT_OVERFLOW:{stage}:{measured}>{limit}")
        self.stage = stage
        self.measured = measured
        self.limit = limit


@dataclass(frozen=True)
class ContextPolicy:
    include_cve_id: bool = True
    include_cwe_ids: bool = True
    include_project: bool = False
    include_commit_message: bool = False
    max_utf8_bytes_per_call: int = 2_000_000
    policy_version: str = "phase3b-context-policy-1.0"
    code_representation: str = FULL_BLOBS

    def __post_init__(self) -> None:
        if self.code_representation not in {FULL_BLOBS, BEFORE_PLUS_DIFF}:
            raise ContextPolicyError("UNSUPPORTED_CODE_REPRESENTATION")

    def sha256(self) -> str:
        values = asdict(self)
        # Retain the historical hash for the unchanged default representation.
        if self.code_representation == FULL_BLOBS:
            del values["code_representation"]
        raw = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


FORBIDDEN_MODEL_KEYS = {
    "ground_truth", "ground_truth_label", "label", "label_scope",
    "review_status", "verification_methods", "security_correctness",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_no_ground_truth_leakage(value: Any, path: str = "model_input") -> None:
    """Reject evaluation fields in any object passed to prompt rendering."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_MODEL_KEYS:
                raise ContextPolicyError(f"GROUND_TRUTH_LEAKAGE:{path}.{key}")
            assert_no_ground_truth_leakage(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_ground_truth_leakage(nested, f"{path}[{index}]")


def _missing_code_marker(status: str, side: str) -> str:
    return f"<CODE_{side.upper()}_{status}>"


def _code_or_marker(file_data: Mapping[str, Any], side: str) -> str:
    value = file_data[f"{side}_code"]
    if value is not None:
        return value
    status = file_data[f"{side}_provenance"]["status"]
    return _missing_code_marker(status, side)


def render_model_diff_bundle(model_input: Mapping[str, Any]) -> Dict[str, str]:
    """Build a metadata-free bundle from per-file original diffs.

    The canonical full patch remains available under ``audit_only``.  This
    representation excludes commit subject/author/message while retaining file
    operation and path semantics.
    """
    assert_no_ground_truth_leakage(model_input)
    sections: List[str] = []
    for file_data in model_input["files"]:
        sections.extend([
            f"=== FILE CHANGE {file_data['index'] + 1} ===",
            f"change_type: {file_data['change_type']}",
            f"path_before: {file_data['path_before'] if file_data['path_before'] is not None else '<NONE>'}",
            f"path_after: {file_data['path_after'] if file_data['path_after'] is not None else '<NONE>'}",
            "original_diff:",
            file_data.get("original_diff") or "<ORIGINAL_DIFF_MISSING>",
        ])
    text = "\n".join(sections)
    return {"text": text, "sha256": sha256_text(text)}


def render_file_context(
    file_data: Mapping[str, Any], policy: ContextPolicy | None = None,
    full_original_patch: str | None = None,
) -> Dict[str, Any]:
    policy = policy or ContextPolicy()
    before = _code_or_marker(file_data, "before")
    after = _code_or_marker(file_data, "after")
    provenance = None
    if policy.code_representation == BEFORE_PLUS_DIFF:
        provenance = {
            "representation": BEFORE_PLUS_DIFF,
            "path_before": file_data["path_before"],
            "path_after": file_data["path_after"],
            "change_type": file_data["change_type"],
            "before_sha256": file_data.get("before_code_sha256"),
            "after_sha256": file_data.get("after_code_sha256"),
            "original_diff_sha256": file_data.get("original_diff_sha256"),
            "round_trip": "NOT_APPLICABLE_FULL_BLOBS_FALLBACK",
        }
        if file_data["change_type"] == "MODIFY":
            if file_data["path_before"] != file_data["path_after"]:
                raise ContextPolicyError("DELTA_MODIFY_PATH_MISMATCH")
            patch = file_data.get("original_diff")
            for field, expected in (
                (file_data.get("before_code"), file_data.get("before_code_sha256")),
                (file_data.get("after_code"), file_data.get("after_code_sha256")),
                (patch, file_data.get("original_diff_sha256")),
            ):
                if not isinstance(field, str) or sha256_text(field) != expected:
                    raise ContextPolicyError("DELTA_SOURCE_TEXT_HASH_MISMATCH")
            try:
                if full_original_patch is None:
                    raise LosslessContextError("FULL_ORIGINAL_PATCH_REQUIRED")
                replay_patch = original_modify_section(full_original_patch, file_data["path_before"])
                # Canonical per-file diffs are hunk text with the final newline
                # stripped by the builder; replay only the exact full-patch section.
                replay_hunks = replay_patch[replay_patch.index("@@ "):].rstrip("\n")
                if patch != replay_hunks:
                    raise LosslessContextError("CANONICAL_HUNK_ORIGINAL_SECTION_MISMATCH")
                restored = replay_original_modify_diff(before, replay_patch, file_data["path_before"])
            except LosslessContextError as exc:
                raise ContextPolicyError(f"DELTA_ROUND_TRIP_BLOCKED:{exc}") from exc
            if restored != after:
                raise ContextPolicyError("DELTA_AFTER_ROUND_TRIP_MISMATCH")
            provenance["round_trip"] = "EXACT_AFTER_MATCH"
            provenance["replay_original_section_sha256"] = sha256_text(replay_patch)
            provenance["full_original_patch_sha256"] = sha256_text(full_original_patch)
            after = (
                "<AFTER_REPRESENTATION: before_plus_original_diff_v1>\n"
                "The complete AFTER file is the complete BEFORE file above with this file's "
                "original Git diff below applied. All unchanged lines remain identical. "
                "Use that diff for every addition/deletion; no file content is truncated.\n"
                f"AFTER_UTF8_SHA256: {file_data['after_code_sha256']}\n"
                "ROUND_TRIP: exact match to the canonical AFTER blob.\n"
                "</AFTER_REPRESENTATION>"
            )
    text = "\n".join([
        f"CHANGE TYPE: {file_data['change_type']}",
        f"SOURCE PATH: {file_data['path_before'] if file_data['path_before'] is not None else '<NONE>'}",
        f"TARGET PATH: {file_data['path_after'] if file_data['path_after'] is not None else '<NONE>'}",
        "\n--- BEFORE ---", before,
        "\n--- AFTER ---", after,
    ])
    result = {
        "code_block": text,
        "before_code": before,
        "after_code": after,
        "diff": file_data.get("original_diff") or "<ORIGINAL_DIFF_MISSING>",
        "sha256": sha256_text(text),
    }
    if provenance is not None:
        result["representation_provenance"] = provenance
    return result


def policy_metadata(model_input: Mapping[str, Any], policy: ContextPolicy) -> Dict[str, str]:
    return {
        "cve_id": model_input["cve_id"] if policy.include_cve_id else "<WITHHELD_BY_POLICY>",
        "cwe_id": ", ".join(model_input["cwe_ids"]) if policy.include_cwe_ids else "<WITHHELD_BY_POLICY>",
    }


def enforce_context_limit(text: str, stage: str, policy: ContextPolicy) -> None:
    measured = len(text.encode("utf-8"))
    if measured > policy.max_utf8_bytes_per_call:
        raise ContextOverflowError(stage, measured, policy.max_utf8_bytes_per_call)


def prepare_context(adapted: Mapping[str, Any], policy: ContextPolicy) -> Dict[str, Any]:
    """Prepare the common source pool; methods choose different views of it."""
    model_input = adapted["model_input"]
    assert_no_ground_truth_leakage(model_input)
    full_original = None
    if policy.code_representation == BEFORE_PLUS_DIFF:
        full_patch = adapted["audit_only"]["full_patch"]
        full_original = full_patch.get("original_git_diff")
        if not isinstance(full_original, str) or sha256_text(full_original) != full_patch.get("original_git_diff_sha256"):
            raise ContextPolicyError("DELTA_FULL_ORIGINAL_PATCH_HASH_MISMATCH")
    files = [render_file_context(item, policy, full_original) for item in model_input["files"]]
    diff_bundle = render_model_diff_bundle(model_input)
    return {
        "metadata": policy_metadata(model_input, policy),
        "files": files,
        "diff_bundle": diff_bundle,
        "policy_sha256": policy.sha256(),
    }
