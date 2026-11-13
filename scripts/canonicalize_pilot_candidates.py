"""Targeted, resumable canonicalization of an explicit source-only shortlist.

The script retrieves only the designated commit metadata, patch, and changed
file blobs.  It does not infer CVE mappings, approve security correctness,
change review status, call an LLM, or create an official pilot manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_validation.canonical_builder import build_canonical_sample, ensure_unique_sample_ids
from dataset_validation.targeted_retrieval import create_retrieval_plan, retrieve_plan
from dataset_validation.validator import DatasetValidator, sha256_file


REPOSITORIES = {
    "linux": "torvalds/linux",
    "mozilla": "mozilla/gecko-dev",
    "xen": "xen-project/xen",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_once(path: Path, value: Any) -> None:
    encoded = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    path.write_bytes(encoded)


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "VulnSage-canonical-pilot20/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GitHub commit response is not an object")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    previous = csv.field_size_limit()
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    finally:
        csv.field_size_limit(previous)


def canonicalize(
    *, shortlist_path: Path, source_csv: Path, schema_path: Path, output_root: Path
) -> dict[str, Any]:
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    source_ids = [str(item) for item in shortlist["ordered_source_ids"]]
    rows = _load_csv(source_csv)
    rows_by_id = {str(row.get("id")): row for row in rows}
    validation = DatasetValidator().validate_csv(source_csv)
    validation_by_id = {str(item["source_id"]): item for item in validation["records"]}
    source_hash = sha256_file(source_csv)
    if source_hash != shortlist.get("source_csv_sha256"):
        raise ValueError("Shortlist source CSV hash mismatch")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    import jsonschema

    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for source_id in source_ids:
        row = rows_by_id[source_id]
        project = str(row.get("PROJECT") or "").strip().lower()
        repository = REPOSITORIES.get(project)
        candidate_dir = output_root / "evidence" / source_id
        record: dict[str, Any] = {
            "source_id": source_id,
            "project": project,
            "cve_id": row.get("VULNERABILITY_CVE"),
            "fix_commit": str(row.get("COMMIT_HASH") or "").lower(),
            "repository": repository,
            "status": "BLOCKED",
            "reason_codes": [],
            "experimental_llm_calls": 0,
        }
        try:
            if repository is None:
                raise ValueError("UPSTREAM_REPOSITORY_UNKNOWN")
            evidence_path = candidate_dir / "github_commit_evidence.json"
            if evidence_path.exists():
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            else:
                evidence = _fetch_json(
                    f"https://api.github.com/repos/{repository}/commits/{record['fix_commit']}"
                )
                _write_once(evidence_path, evidence)
            if str(evidence.get("sha") or "").lower() != record["fix_commit"]:
                raise ValueError("FIX_COMMIT_IDENTITY_MISMATCH")
            parents = evidence.get("parents") or []
            if len(parents) != 1 or len(str(parents[0].get("sha") or "")) != 40:
                raise ValueError("EXACTLY_ONE_PARENT_REQUIRED")
            parent_sha = str(parents[0]["sha"])
            plan = create_retrieval_plan(source_id, repository, evidence)
            manifest = retrieve_plan(plan, candidate_dir / "retrieval")
            manifest_path = candidate_dir / "retrieval" / "retrieval_manifest.json"
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            sample = build_canonical_sample(
                row,
                validation_by_id[source_id],
                evidence,
                manifest,
                retrieval_manifest_dir=manifest_path.parent,
                source_file=str(source_csv.resolve()),
                source_file_sha256=source_hash,
                source_repository_url=f"https://github.com/{repository}",
                dataset_version="canonical-pilot20-prototype-20261111",
                ground_truth={
                    "task": "PATCH_VERIFICATION",
                    "label": "PATCHED",
                    "source": "VULNSAGE_FIX_COMMIT",
                    "label_scope": (
                        "VulnSage-designated fix-commit post-state; security correctness "
                        "is not independently approved."
                    ),
                    "review_status": "PENDING",
                    "verification_methods": [
                        "SOURCE_DATASET", "COMMIT_IDENTITY_VERIFIED", "ORIGINAL_DIFF_RETRIEVED"
                    ],
                    "evidence_artifacts": [str(evidence_path.resolve()), str(manifest_path.resolve())],
                    "reviewers": [],
                    "verified_at_utc": None,
                },
                evidence_snapshot_sha256=evidence_hash,
                retrieval_manifest_sha256=manifest_hash,
            )
            jsonschema.validate(sample, schema)
            if sample["ground_truth"]["review_status"] != "PENDING":
                raise ValueError("GROUND_TRUTH_REVIEW_STATUS_CHANGED")
            samples.append(sample)
            record.update(
                {
                    "status": "TECHNICALLY_PREPARED",
                    "sample_id": sample["sample_id"],
                    "parent_commit": parent_sha,
                    "commit_url": evidence.get("html_url"),
                    "commit_subject": str(evidence.get("commit", {}).get("message") or "").splitlines()[0],
                    "changed_file_count": len(evidence.get("files") or []),
                    "canonical_file_count": len(sample["files"]),
                    "operations": [item["change_type"] for item in sample["files"]],
                    "canonical_sha256": hashlib.sha256(_json_bytes(sample)).hexdigest(),
                    "evidence_snapshot_sha256": evidence_hash,
                    "retrieval_manifest_sha256": manifest_hash,
                    "commit_verification_status": sample["commit_verification"]["status"],
                    "quality_status": sample["quality"]["validation_status"],
                    "reason_codes": sorted(set(sample["quality"]["reason_codes"])),
                    "security_review": "HUMAN_REVIEW_REQUIRED",
                    "ground_truth_review": "PENDING",
                }
            )
        except Exception as exc:
            record["reason_codes"] = [f"{type(exc).__name__}:{exc}"]
        records.append(record)

    ensure_unique_sample_ids(samples)
    dataset = {
        "artifact_type": "canonical_pilot20_prototype_not_official",
        "dataset_version": "canonical-pilot20-prototype-20261111",
        "official_pilot": False,
        "inference_authorized": False,
        "security_review_complete": False,
        "ground_truth_mutated": False,
        "experimental_llm_calls": 0,
        "source_csv_sha256": source_hash,
        "ordered_source_ids": source_ids,
        "canonicalized_source_ids": [record["source_id"] for record in records if record["status"] == "TECHNICALLY_PREPARED"],
        "samples": samples,
    }
    _write_once(output_root / "canonical_dataset.json", dataset)
    summary = {
        "source_only_candidates": len(source_ids),
        "canonicalized": len(samples),
        "blocked": len(source_ids) - len(samples),
        "records": records,
        "canonical_dataset_sha256": hashlib.sha256((output_root / "canonical_dataset.json").read_bytes()).hexdigest(),
        "experimental_llm_calls": 0,
    }
    _write_once(output_root / "canonicalization_run.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = canonicalize(
        shortlist_path=args.shortlist,
        source_csv=args.source_csv,
        schema_path=args.schema,
        output_root=args.output_root,
    )
    print(json.dumps({
        "source_only_candidates": result["source_only_candidates"],
        "canonicalized": result["canonicalized"],
        "blocked": result["blocked"],
        "experimental_llm_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
