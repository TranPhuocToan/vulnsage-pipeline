"""Shared, auditable model-context policy for P0/P1/P2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping


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

    def sha256(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
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


def render_file_context(file_data: Mapping[str, Any]) -> Dict[str, str]:
    before = _code_or_marker(file_data, "before")
    after = _code_or_marker(file_data, "after")
    text = "\n".join([
        f"CHANGE TYPE: {file_data['change_type']}",
        f"SOURCE PATH: {file_data['path_before'] if file_data['path_before'] is not None else '<NONE>'}",
        f"TARGET PATH: {file_data['path_after'] if file_data['path_after'] is not None else '<NONE>'}",
        "\n--- BEFORE ---", before,
        "\n--- AFTER ---", after,
    ])
    return {
        "code_block": text,
        "before_code": before,
        "after_code": after,
        "diff": file_data.get("original_diff") or "<ORIGINAL_DIFF_MISSING>",
        "sha256": sha256_text(text),
    }


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
    files = [render_file_context(item) for item in model_input["files"]]
    diff_bundle = render_model_diff_bundle(model_input)
    return {
        "metadata": policy_metadata(model_input, policy),
        "files": files,
        "diff_bundle": diff_bundle,
        "policy_sha256": policy.sha256(),
    }
