"""Review-gated builder for a multi-sample canonical pilot package.

This module is deliberately separate from the historical one-sample smoke
builder.  It never promotes canonical ground truth, creates labels, or selects
samples.  Every ordered sample must already have an attributable APPROVE
decision and enough derived-only waivers for ``qualify_sample`` to return
READY.  The output remains a pilot package, never a held-out/main split.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_validation.qualification import qualify_sample
from pipeline.canonical_adapter import adapt_sample
from pipeline.run_state import hash_value


PACKAGE_VERSION = "phase3d-multi-sample-pilot-package-1.0"
BUILDER_MODULE = "dataset_validation/pilot_package.py"
SHA40 = re.compile(r"^[0-9a-f]{40}$")

PACKAGE_FILENAMES = {
    "dataset": "canonical_dataset.json",
    "qualification": "qualification_records.json",
    "decision": "review_decisions.json",
    "manifest": "official_pilot_manifest.json",
    "receipt": "package_receipt.json",
}


class PilotPackageError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_samples(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotPackageError(f"SOURCE_DATASET_INVALID:{exc}") from exc
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list) or not all(isinstance(item, dict) for item in samples):
        raise PilotPackageError("SOURCE_DATASET_SAMPLES_INVALID")
    return samples


def _load_decision(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotPackageError(f"REVIEW_DECISION_INVALID:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise PilotPackageError(f"REVIEW_DECISION_NOT_OBJECT:{path}")
    return value


def prepare_multi_sample_pilot_package(
    *,
    source_dataset_path: Path,
    ordered_source_row_ids: Sequence[str],
    review_decision_paths: Mapping[str, Path],
    schema_path: Path,
    output_dir: Path,
    git_commit: str,
) -> dict[str, Any]:
    """Create a deterministic non-overwriting package after human review.

    ``review_decision_paths`` is keyed by source row ID.  Fixture decisions may
    be used in tests, but production callers must supply real reviewer
    provenance.  This function does not decide which samples should be used.
    """
    source_dataset_path = Path(source_dataset_path)
    schema_path = Path(schema_path)
    output_dir = Path(output_dir)
    source_ids = [str(item) for item in ordered_source_row_ids]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise PilotPackageError("ORDERED_SOURCE_IDS_MUST_BE_NONEMPTY_UNIQUE")
    if output_dir.exists():
        raise PilotPackageError(f"OUTPUT_ALREADY_EXISTS:{output_dir}")
    if not SHA40.fullmatch(git_commit):
        raise PilotPackageError("FULL_GIT_COMMIT_REQUIRED")
    if set(str(key) for key in review_decision_paths) != set(source_ids):
        raise PilotPackageError("REVIEW_DECISION_SOURCE_ID_SET_MISMATCH")

    builder_path = Path(__file__).resolve()
    builder_hash = _sha256_file(builder_path)
    source_hash_before = _sha256_file(source_dataset_path)
    all_samples = _load_samples(source_dataset_path)
    by_source: dict[str, dict[str, Any]] = {}
    for sample in all_samples:
        source_id = str(sample.get("source", {}).get("source_row_id") or "")
        if source_id in by_source:
            raise PilotPackageError(f"DUPLICATE_SOURCE_ROW_ID:{source_id}")
        by_source[source_id] = sample
    missing = [source_id for source_id in source_ids if source_id not in by_source]
    if missing:
        raise PilotPackageError(f"SOURCE_IDS_NOT_FOUND:{missing}")

    selected = [by_source[source_id] for source_id in source_ids]
    adapted = [adapt_sample(sample, schema_path=schema_path) for sample in selected]
    sample_ids = [item["sample_id"] for item in adapted]
    family_ids = [item["family_id"] for item in adapted]
    if len(sample_ids) != len(set(sample_ids)):
        raise PilotPackageError("DUPLICATE_CANONICAL_SAMPLE_ID")
    if len(family_ids) != len(set(family_ids)):
        raise PilotPackageError("DUPLICATE_CVE_FAMILY_IN_PILOT")

    qualifications: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for source_id, sample, adapted_sample in zip(source_ids, selected, adapted):
        if sample.get("ground_truth", {}).get("review_status") != "PENDING":
            raise PilotPackageError(
                f"CANONICAL_REVIEW_STATUS_MUST_REMAIN_PENDING:{adapted_sample['sample_id']}"
            )
        decision = _load_decision(Path(review_decision_paths[source_id]))
        if decision.get("decision") != "APPROVE":
            raise PilotPackageError(
                f"ATTRIBUTABLE_APPROVE_DECISION_REQUIRED:{adapted_sample['sample_id']}"
            )
        qualification = qualify_sample(sample, decision=decision)
        if qualification["status"] != "READY":
            raise PilotPackageError(
                f"QUALIFICATION_NOT_READY:{adapted_sample['sample_id']}:{qualification['reason_codes']}"
            )
        gap_set = set(qualification["derived_representation_gaps"])
        waiver_set = {item.get("reason_code") for item in decision.get("waivers", [])}
        if waiver_set != gap_set:
            raise PilotPackageError(
                f"WAIVER_SET_MISMATCH:{adapted_sample['sample_id']}:"
                f"expected={sorted(gap_set)}:actual={sorted(waiver_set)}"
            )
        if qualification["security_correctness_verified"] is not False:
            raise PilotPackageError("SECURITY_CORRECTNESS_MUST_NOT_BE_AUTO_VERIFIED")
        if qualification["ground_truth_review_status_unchanged"] != "PENDING":
            raise PilotPackageError("GROUND_TRUTH_REVIEW_STATUS_CHANGED")
        qualifications.append(qualification)
        decisions.append(decision)

    dataset_payload = {
        "package_version": PACKAGE_VERSION,
        "purpose": "MULTI_SAMPLE_CANONICAL_PILOT_NOT_MAIN_EXPERIMENT",
        "official_dataset": True,
        "official_split": False,
        "source_dataset_sha256": source_hash_before,
        "builder_version": PACKAGE_VERSION,
        "builder_module": BUILDER_MODULE,
        "builder_code_sha256": builder_hash,
        "samples": selected,
    }
    qualification_payload = {
        "package_version": PACKAGE_VERSION,
        "records": qualifications,
    }
    decision_payload = {
        "package_version": PACKAGE_VERSION,
        "decisions": decisions,
    }
    dataset_bytes = _json_bytes(dataset_payload)
    qualification_bytes = _json_bytes(qualification_payload)
    decision_bytes = _json_bytes(decision_payload)
    dataset_hash = _sha256_bytes(dataset_bytes)
    manifest_entries = []
    for source_id, adapted_sample, qualification, decision in zip(
        source_ids, adapted, qualifications, decisions
    ):
        manifest_entries.append(
            {
                "sample_id": adapted_sample["sample_id"],
                "source_row_id": source_id,
                "family_id": adapted_sample["family_id"],
                "qualification_status": "READY",
                "canonical_input_sha256": qualification["input_sample_sha256"],
                "qualification_record_sha256": hash_value(qualification),
                "review_decision_sha256": hash_value(decision),
            }
        )
    manifest_base = {
        "manifest_version": PACKAGE_VERSION,
        "purpose": "PILOT",
        "scope": "MULTI_SAMPLE_CANONICAL_PILOT_NOT_MAIN_EXPERIMENT",
        "official_dataset": True,
        "official_split": False,
        "dataset_file": PACKAGE_FILENAMES["dataset"],
        "dataset_sha256": dataset_hash,
        "schema_version": selected[0]["schema_version"],
        "schema_sha256": _sha256_file(schema_path),
        "ordered_sample_ids": sample_ids,
        "sample_list_sha256": hash_value(sample_ids),
        "git_commit": git_commit,
        "builder_version": PACKAGE_VERSION,
        "builder_module": BUILDER_MODULE,
        "builder_code_sha256": builder_hash,
        "samples": manifest_entries,
    }
    manifest = {**manifest_base, "manifest_sha256": hash_value(manifest_base)}
    manifest_bytes = _json_bytes(manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        files = {
            PACKAGE_FILENAMES["dataset"]: dataset_bytes,
            PACKAGE_FILENAMES["qualification"]: qualification_bytes,
            PACKAGE_FILENAMES["decision"]: decision_bytes,
            PACKAGE_FILENAMES["manifest"]: manifest_bytes,
        }
        for name, content in files.items():
            with (temporary / name).open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        receipt = {
            "package_version": PACKAGE_VERSION,
            "package_format": "MULTI_SAMPLE_REVIEW_RECORDS_V1",
            "source_dataset_path": str(source_dataset_path.resolve()),
            "source_dataset_sha256_before": source_hash_before,
            "source_dataset_sha256_after": _sha256_file(source_dataset_path),
            "ordered_source_row_ids": source_ids,
            "ordered_sample_ids": sample_ids,
            "family_ids": family_ids,
            "git_commit": git_commit,
            "builder_module": BUILDER_MODULE,
            "builder_code_sha256": builder_hash,
            "files": {name: _sha256_file(temporary / name) for name in sorted(files)},
        }
        if receipt["source_dataset_sha256_before"] != receipt["source_dataset_sha256_after"]:
            raise PilotPackageError("SOURCE_DATASET_CHANGED_DURING_BUILD")
        with (temporary / PACKAGE_FILENAMES["receipt"]).open("xb") as handle:
            handle.write(_json_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists():
            raise PilotPackageError(f"OUTPUT_ALREADY_EXISTS:{output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()

    return {
        "status": "READY_PACKAGE_CREATED",
        "output_dir": str(output_dir),
        "ordered_sample_ids": sample_ids,
        "sample_list_sha256": manifest["sample_list_sha256"],
        "dataset_sha256": dataset_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "ground_truth_review_status_unchanged": "PENDING",
        "security_correctness_verified": False,
    }
