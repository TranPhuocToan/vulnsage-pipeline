"""Build the controlled Phase 2E normalized prototype from saved evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from .canonical_builder import build_canonical_sample, ensure_unique_sample_ids
from .validator import sha256_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    validation_by_id = {str(item["source_id"]): item for item in validation["records"]}
    old_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(sys.maxsize)
        with args.csv.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = {str(row["id"]): row for row in csv.DictReader(stream)}
    finally:
        csv.field_size_limit(old_limit)
    source_hash = sha256_file(args.csv)
    base = args.config.parent
    samples = []
    for entry in config["samples"]:
        source_id = str(entry["source_id"])
        evidence_path = (base / entry["evidence_path"]).resolve()
        manifest_path = (base / entry["retrieval_manifest_path"]).resolve()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples.append(
            build_canonical_sample(
                rows[source_id],
                validation_by_id[source_id],
                evidence,
                manifest,
                retrieval_manifest_dir=manifest_path.parent,
                source_file=str(args.csv.resolve()),
                source_file_sha256=source_hash,
                source_repository_url=config["source_repository_url"],
                dataset_version=config["dataset_version"],
                ground_truth=config["ground_truth"],
                evidence_snapshot_sha256=sha256_file(evidence_path),
                retrieval_manifest_sha256=sha256_file(manifest_path),
            )
        )
    ensure_unique_sample_ids(samples)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validation_errors = []
    for sample in samples:
        errors = sorted(validator.iter_errors(sample), key=lambda error: list(error.path))
        if errors:
            validation_errors.extend(
                {"sample_id": sample["sample_id"], "path": list(error.path), "message": error.message}
                for error in errors
            )
    if validation_errors:
        raise ValueError(json.dumps(validation_errors, ensure_ascii=False))
    samples.sort(key=lambda sample: int(sample["source"]["source_row_id"]))
    sample_hash = hashlib_sha256_json(samples)
    output = {
        "prototype_version": config["dataset_version"],
        "prototype_only": True,
        "canonical_diff_source": "ORIGINAL_COMMIT_DIFF",
        "source_csv_sha256": source_hash,
        "schema_sha256": sha256_file(args.schema),
        "sample_count": len(samples),
        "sample_list_sha256": sample_hash,
        "schema_validation": {"status": "PASS", "errors": 0},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: output[key] for key in ("sample_count", "sample_list_sha256", "schema_validation")}, sort_keys=True))
    return 0


def hashlib_sha256_json(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
