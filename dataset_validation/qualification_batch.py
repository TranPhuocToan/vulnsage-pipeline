"""Resumable, non-overwriting batch qualification for canonical samples."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .qualification import QualificationPolicy, not_assessed_record, qualify_sample


BATCH_VERSION = "phase3c2-qualification-batch-1.0"


class QualificationBatchError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_key(value: str) -> str:
    prefix = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._")
    return f"{(prefix or 'sample')[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def _atomic_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise QualificationBatchError(f"refusing to overwrite artifact: {path}")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise QualificationBatchError(f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise QualificationBatchError("dataset must be a sample array or wrapper with samples")
    ids = [item.get("sample_id") for item in samples]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise QualificationBatchError("dataset has missing/duplicate sample_id")
    return samples


def _load_decisions(path: Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions") if isinstance(payload, dict) else payload
    if not isinstance(decisions, list):
        raise QualificationBatchError("review decision file must contain an array")
    by_id: dict[str, dict[str, Any]] = {}
    for item in decisions:
        sample_id = item.get("sample_id") if isinstance(item, dict) else None
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise QualificationBatchError("review decisions have missing/duplicate sample_id")
        by_id[sample_id] = item
    return by_id, hash_file(path)


def run_qualification_batch(
    *,
    dataset_path: Path,
    output_dir: Path,
    candidate_source_ids: Sequence[str] | None = None,
    review_decisions_path: Path | None = None,
    policy: QualificationPolicy | None = None,
    multi_commit_family_ids: Sequence[str] = (),
    max_new_samples: int | None = None,
) -> dict[str, Any]:
    """Run or resume qualification. Existing per-sample records are validated and reused."""
    policy = policy or QualificationPolicy()
    samples = _load_dataset(dataset_path)
    decisions, decisions_hash = _load_decisions(review_decisions_path)
    ordered_ids = [item["sample_id"] for item in samples]
    candidate_ids = [str(item) for item in (candidate_source_ids or [])]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise QualificationBatchError("duplicate candidate source ID")
    spec_base = {
        "batch_version": BATCH_VERSION,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": hash_file(dataset_path),
        "ordered_sample_ids": ordered_ids,
        "sample_id_list_sha256": hash_value(ordered_ids),
        "candidate_source_ids": candidate_ids,
        "candidate_source_id_list_sha256": hash_value(candidate_ids),
        "review_decisions_sha256": decisions_hash,
        # Round-trip tuples to JSON arrays so an on-disk spec compares exactly
        # with the same in-memory policy on resume.
        "policy": json.loads(json.dumps(policy.__dict__, sort_keys=True)),
        "policy_sha256": policy.sha256(),
        "multi_commit_family_ids": sorted(set(multi_commit_family_ids)),
    }
    spec = {**spec_base, "spec_sha256": hash_value(spec_base)}
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "qualification_spec.json"
    if spec_path.exists():
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        if stored != spec:
            raise QualificationBatchError("QUALIFICATION_RESUME_SPEC_MISMATCH")
    else:
        _atomic_create(spec_path, spec)

    record_dir = output_dir / "records"
    completed: list[dict[str, Any]] = []
    new_count = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        record_path = record_dir / f"{_safe_key(sample_id)}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("batch_spec_sha256") != spec["spec_sha256"]:
                raise QualificationBatchError(f"record/spec mismatch: {sample_id}")
            if record.get("input_sample_sha256") != hash_value(sample):
                raise QualificationBatchError(f"record/input mismatch: {sample_id}")
        else:
            if max_new_samples is not None and new_count >= max_new_samples:
                break
            record = qualify_sample(
                sample,
                decision=decisions.get(sample_id),
                policy=policy,
                family_requires_review=sample.get("family_id") in set(multi_commit_family_ids),
            )
            record["batch_spec_sha256"] = spec["spec_sha256"]
            _atomic_create(record_path, record)
            with (output_dir / "checkpoint.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({
                    "sample_id": sample_id,
                    "record_sha256": hash_file(record_path),
                    "batch_spec_sha256": spec["spec_sha256"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            new_count += 1
        completed.append(record)

    all_complete = len(completed) == len(samples)
    status_counts = Counter(item["status"] for item in completed)
    canonical_source_ids = {item["source_row_id"] for item in completed if item.get("source_row_id")}
    candidate_records = []
    if candidate_ids:
        by_source = {item["source_row_id"]: item for item in completed}
        candidate_records = [
            by_source.get(source_id) or not_assessed_record(source_id)
            for source_id in candidate_ids
        ]
    summary = {
        "batch_version": BATCH_VERSION,
        "batch_spec_sha256": spec["spec_sha256"],
        "complete": all_complete,
        "canonical_samples_total": len(samples),
        "canonical_samples_completed": len(completed),
        "new_samples_processed": new_count,
        "canonical_status_counts": dict(sorted(status_counts.items())),
        "candidate_frame_total": len(candidate_ids),
        "candidate_frame_status_counts": dict(sorted(Counter(item["status"] for item in candidate_records).items())),
        "candidate_frame_records": candidate_records,
        "out_of_frame_canonical_source_ids": sorted(canonical_source_ids - set(candidate_ids)),
    }
    summary_path = output_dir / "qualification_summary.json"
    if all_complete:
        if summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            # Reruns may process zero new samples; compare stable content only.
            existing_stable = {k: v for k, v in existing.items() if k != "new_samples_processed"}
            current_stable = {k: v for k, v in summary.items() if k != "new_samples_processed"}
            if existing_stable != current_stable:
                raise QualificationBatchError("existing summary differs from resumed result")
        else:
            _atomic_create(summary_path, summary)
    return summary
