"""Review-gated builder for a one-sample canonical Kaggle smoke package.

The builder never changes the source dataset or canonical ground-truth object.
It requires an attributable external APPROVE decision and exact derived-gap
waivers before it can emit a READY qualification record and official manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from dataset_validation.qualification import qualify_sample
from pipeline.canonical_adapter import adapt_sample
from pipeline.run_state import hash_value


PACKAGE_VERSION = "phase3c4-one-sample-smoke-package-1.0"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


class SmokePackageError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_samples(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SmokePackageError(f"SOURCE_DATASET_INVALID:{exc}") from exc
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise SmokePackageError("SOURCE_DATASET_SAMPLES_MISSING")
    return samples


def _load_decision(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SmokePackageError(f"REVIEW_DECISION_INVALID:{exc}") from exc
    if not isinstance(value, dict):
        raise SmokePackageError("REVIEW_DECISION_NOT_OBJECT")
    return value


def prepare_one_sample_smoke_package(
    *,
    source_dataset_path: Path,
    source_row_id: str,
    review_decision_path: Path,
    schema_path: Path,
    output_dir: Path,
    git_commit: str,
) -> dict[str, Any]:
    """Create a deterministic, non-overwriting package only after approval."""
    source_dataset_path = Path(source_dataset_path)
    review_decision_path = Path(review_decision_path)
    schema_path = Path(schema_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise SmokePackageError(f"OUTPUT_ALREADY_EXISTS:{output_dir}")
    if not SHA40.fullmatch(git_commit):
        raise SmokePackageError("FULL_GIT_COMMIT_REQUIRED")
    builder_code_sha256 = _sha256_file(Path(__file__).resolve())
    source_hash_before = _sha256_file(source_dataset_path)
    samples = _load_samples(source_dataset_path)
    matches = [
        sample for sample in samples
        if str(sample.get("source", {}).get("source_row_id")) == str(source_row_id)
    ]
    if len(matches) != 1:
        raise SmokePackageError(f"SOURCE_ID_MATCH_COUNT:{len(matches)}")
    sample = matches[0]
    adapted = adapt_sample(sample, schema_path=schema_path)
    if sample.get("ground_truth", {}).get("review_status") != "PENDING":
        raise SmokePackageError("CANONICAL_REVIEW_STATUS_MUST_REMAIN_PENDING")
    decision = _load_decision(review_decision_path)
    if decision.get("decision") != "APPROVE":
        raise SmokePackageError("ATTRIBUTABLE_APPROVE_DECISION_REQUIRED")
    qualification = qualify_sample(sample, decision=decision)
    if qualification["status"] != "READY":
        raise SmokePackageError(
            f"QUALIFICATION_NOT_READY:{qualification['reason_codes']}"
        )
    gap_set = set(qualification["derived_representation_gaps"])
    waiver_set = {item.get("reason_code") for item in decision.get("waivers", [])}
    if waiver_set != gap_set:
        raise SmokePackageError(
            f"WAIVER_SET_MISMATCH:expected={sorted(gap_set)}:actual={sorted(waiver_set)}"
        )
    if qualification["security_correctness_verified"] is not False:
        raise SmokePackageError("SECURITY_CORRECTNESS_MUST_NOT_BE_AUTO_VERIFIED")
    if qualification["ground_truth_review_status_unchanged"] != "PENDING":
        raise SmokePackageError("GROUND_TRUTH_REVIEW_STATUS_CHANGED")

    dataset_payload = {
        "package_version": PACKAGE_VERSION,
        "purpose": "ONE_SAMPLE_KAGGLE_SMOKE_NOT_MAIN_EXPERIMENT",
        "official_dataset": True,
        "official_split": False,
        "source_dataset_sha256": source_hash_before,
        "builder_version": PACKAGE_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "samples": [sample],
    }
    dataset_bytes = _json_bytes(dataset_payload)
    dataset_hash = _sha256_bytes(dataset_bytes)
    qualification_bytes = _json_bytes(qualification)
    decision_bytes = _json_bytes(decision)
    ordered_ids = [adapted["sample_id"]]
    manifest_base = {
        "manifest_version": PACKAGE_VERSION,
        "purpose": "PILOT",
        "scope": "ONE_SAMPLE_SMOKE_NOT_MAIN_EXPERIMENT",
        "official_dataset": True,
        "official_split": False,
        "dataset_file": "canonical_dataset.json",
        "dataset_sha256": dataset_hash,
        "schema_version": sample["schema_version"],
        "schema_sha256": _sha256_file(schema_path),
        "ordered_sample_ids": ordered_ids,
        "sample_list_sha256": hash_value(ordered_ids),
        "git_commit": git_commit,
        "builder_version": PACKAGE_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "samples": [{
            "sample_id": adapted["sample_id"],
            "source_row_id": str(source_row_id),
            "family_id": adapted["family_id"],
            "qualification_status": "READY",
            "canonical_input_sha256": qualification["input_sample_sha256"],
            "qualification_record_sha256": _sha256_bytes(qualification_bytes),
            "review_decision_sha256": _sha256_bytes(decision_bytes),
        }],
    }
    manifest = {**manifest_base, "manifest_sha256": hash_value(manifest_base)}
    manifest_bytes = _json_bytes(manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)))
    try:
        files = {
            "canonical_dataset.json": dataset_bytes,
            "qualification_record.json": qualification_bytes,
            "review_decision.json": decision_bytes,
            "official_pilot_manifest.json": manifest_bytes,
        }
        for name, content in files.items():
            with (temporary / name).open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        receipt = {
            "package_version": PACKAGE_VERSION,
            "source_dataset_path": str(source_dataset_path.resolve()),
            "source_dataset_sha256_before": source_hash_before,
            "source_dataset_sha256_after": _sha256_file(source_dataset_path),
            "source_row_id": str(source_row_id),
            "sample_id": adapted["sample_id"],
            "git_commit": git_commit,
            "builder_code_sha256": builder_code_sha256,
            "files": {
                name: _sha256_file(temporary / name) for name in sorted(files)
            },
        }
        if receipt["source_dataset_sha256_before"] != receipt["source_dataset_sha256_after"]:
            raise SmokePackageError("SOURCE_DATASET_CHANGED_DURING_BUILD")
        with (temporary / "package_receipt.json").open("xb") as handle:
            handle.write(_json_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists():
            raise SmokePackageError(f"OUTPUT_ALREADY_EXISTS:{output_dir}")
        os.replace(temporary, output_dir)
    finally:
        # Only clean an uncommitted private temp directory created above.
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return {
        "status": "READY_PACKAGE_CREATED",
        "output_dir": str(output_dir),
        "sample_id": adapted["sample_id"],
        "dataset_sha256": dataset_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "ground_truth_review_status_unchanged": sample["ground_truth"]["review_status"],
        "security_correctness_verified": False,
    }
