"""Run a deterministic commit-verification audit from saved API evidence files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .audit_sampling import stratified_sample
from .commit_verification import evidence_from_github_api, verify_commit_evidence
from .validator import sha256_file


def _parse_quota(value: str) -> tuple[tuple[str, str], int]:
    try:
        stratum, raw_count = value.rsplit("=", 1)
        status, project = stratum.split(":", 1)
        count = int(raw_count)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("quota must be STATUS:project=count") from error
    return (status, project), count


def _coverage_summary(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    measured = [
        result["verification"]["coverage"][metric]
        for result in results
        if result["verification"]["coverage"][metric]["status"].startswith("MEASURED")
    ]
    ratios = [item["ratio"] for item in measured if item["ratio"] is not None]
    matched = sum(item["matched"] for item in measured)
    total = sum(item["total"] for item in measured)
    return {
        "measured_samples": len(measured),
        "macro_mean_ratio": round(sum(ratios) / len(ratios), 6) if ratios else None,
        "micro_ratio": round(matched / total, 6) if total else None,
        "matched": matched,
        "total": total,
    }


def run_batch(
    csv_path: Path,
    validation_report_path: Path,
    evidence_dir: Path,
    quotas: dict[tuple[str, str], int],
    seed: int,
    excluded_ids: set[str],
) -> dict[str, Any]:
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    selected = stratified_sample(
        validation_report["records"],
        quotas,
        seed=seed,
        excluded_source_ids=excluded_ids,
    )
    old_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(sys.maxsize)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = {str(row["id"]): row for row in csv.DictReader(stream)}
    finally:
        csv.field_size_limit(old_limit)

    results = []
    for record in selected:
        source_id = str(record["source_id"])
        evidence_path = evidence_dir / f"{source_id}.json"
        if evidence_path.exists():
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence = evidence_from_github_api(payload)
        else:
            evidence = {
                "verification_status": "UNVERIFIABLE",
                "commit_sha": None,
                "parent_commit_sha": None,
                "source_url": None,
                "files": [],
                "original_diff": None,
                "failure_reason": "Evidence file is missing",
            }
        verification = verify_commit_evidence(rows[source_id], evidence)
        results.append(
            {
                "source_id": source_id,
                "validator_status": record["status"],
                "validator_reason_codes": record["reason_codes"],
                "project": record["project"],
                "cve_id": record["cve_id"],
                "cwe_id": record["cwe_id"],
                "commit_sha": record["commit_hash"],
                "verification": verification,
            }
        )

    status_counts = Counter(
        result["verification"]["verification_status"] for result in results
    )
    reason_counts = Counter(
        reason
        for result in results
        for reason in result["verification"]["reason_codes"]
    )
    frame_counts = Counter(
        (record["status"], record["project"])
        for record in validation_report["records"]
        if str(record["source_id"]) not in excluded_ids
    )
    return {
        "audit_version": "phase2d-random-commit-audit-1.0",
        "sampling": {
            "seed": seed,
            "frame": "Phase 2B records excluding Phase 2C diagnostic source IDs",
            "frame_size": sum(frame_counts.values()),
            "frame_counts": {
                f"{status}:{project}": count
                for (status, project), count in sorted(frame_counts.items())
            },
            "quotas": {
                f"{status}:{project}": count
                for (status, project), count in sorted(quotas.items())
            },
            "excluded_source_ids": sorted(excluded_ids, key=int),
            "selected_source_ids": [result["source_id"] for result in results],
        },
        "sources": {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
            "validation_report_path": str(validation_report_path.resolve()),
            "validation_report_sha256": sha256_file(validation_report_path),
            "evidence_format": "Saved GitHub commit API responses; no repository clone",
        },
        "summary": {
            "sample_count": len(results),
            "verification_status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "coverage": {
                metric: _coverage_summary(results, metric)
                for metric in ("path", "changed_lines", "hunks", "code_context")
            },
        },
        "results": results,
        "interpretation_limit": (
            "Diagnostic stratified audit only; do not infer population error rates or security labels."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--quota", action="append", type=_parse_quota, required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    args = parser.parse_args(argv)
    quotas = dict(args.quota)
    report = run_batch(
        args.csv,
        args.validation_report,
        args.evidence_dir,
        quotas,
        args.seed,
        set(args.exclude_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
