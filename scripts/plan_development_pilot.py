"""Select a small, source-only VulnSage development pilot without LLM calls.

This plan is NOT a canonical dataset, official pilot manifest, ground-truth
decision, token preflight, or inference authorization.  Only the source CSV is
read; no source record is mutated.  The plan intentionally excludes every
researcher-exposed CVE family, not just the exposed individual rows.

Usage:
    python -m scripts.plan_development_pilot \
        --source-csv vulnerability_dataset/database/vulnerabilities.csv \
        --output pilot_candidates_20.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.november_feasibility import RESEARCHER_EXPOSED_IDS


FILE_HEADER = re.compile(r"^//[ \t]*File(?:[ \t]+path)?[ \t]*:[ \t]*(.+?)\r?$", re.MULTILINE)
CVE_ID = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")

REQUIRED_COLUMNS = {
    "id", "COMMIT_HASH", "VULNERABILITY_CVE", "VULNERABILITY_CWE",
    "NUM_FILES_CHANGED", "VULNERABLE_CODE_BLOCK", "PATCHED_CODE_BLOCK", "PROJECT",
}


class PilotPlanningError(ValueError):
    pass


def load_source_rows(source_csv: Path) -> list[dict[str, str]]:
    """Parse multiline code blocks, including CSV fields larger than 128 KiB."""
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:  # Some C CSV builds accept a narrower limit.
        csv.field_size_limit(2**31 - 1)
    with source_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise PilotPlanningError(f"SOURCE_COLUMNS_MISSING:{sorted(missing)}")
        rows = list(reader)
    seen: set[str] = set()
    for row in rows:
        source_id = row["id"].strip()
        if not source_id.isdecimal() or source_id in seen:
            raise PilotPlanningError(f"SOURCE_ID_INVALID_OR_DUPLICATE:{source_id}")
        seen.add(source_id)
    return rows


def split_source_blocks(text: str) -> dict[str, str] | None:
    """Return fragments by unique original path, or None for ambiguous headers."""
    headers = list(FILE_HEADER.finditer(text))
    if not headers:
        return None
    files: dict[str, str] = {}
    for i, header in enumerate(headers):
        path = header.group(1).strip()
        if not path or path in files:
            return None
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        files[path] = text[header.end():end].strip()
    return files


def cves_of(row: Mapping[str, str]) -> frozenset[str]:
    return frozenset(match.upper() for match in CVE_ID.findall(row["VULNERABILITY_CVE"]))


def source_structure(row: Mapping[str, str]) -> dict[str, Any]:
    """Cheap structural gates only; never infer patch correctness or G3 gold."""
    try:
        declared = int(row["NUM_FILES_CHANGED"])
    except (TypeError, ValueError):
        declared = -1
    before = split_source_blocks(row["VULNERABLE_CODE_BLOCK"])
    after = split_source_blocks(row["PATCHED_CODE_BLOCK"])
    same_paths = before is not None and after is not None and set(before) == set(after)
    two_paths = bool(same_paths and len(before) >= 2)
    changed = sum(before[path] != after[path] for path in before) if same_paths else 0
    nonempty = bool(same_paths and all(before.values()) and all(after.values()))
    exact_file_count = bool(two_paths and len(before) == declared)
    raw_chars = len(row["VULNERABLE_CODE_BLOCK"]) + len(row["PATCHED_CODE_BLOCK"])
    return {
        "declared_num_files": declared,
        "parsed_num_files": len(before) if before is not None else 0,
        "matched_paths": bool(same_paths),
        "two_paths": two_paths,
        "changed_path_count": changed,
        "all_fragments_nonempty": nonempty,
        "exact_file_count": exact_file_count,
        "combined_source_characters": raw_chars,
        "source_structurally_strict": bool(
            declared >= 2 and two_paths and changed >= 2 and nonempty and exact_file_count
        ),
    }


def plan_pilot(
    rows: Iterable[Mapping[str, str]],
    *,
    pilot_size: int = 20,
    exposed_ids: Iterable[str] = RESEARCHER_EXPOSED_IDS,
    max_source_characters: int = 50_000,
) -> dict[str, Any]:
    """Select diverse, short-context DEVELOPMENT candidates, CVE-family disjoint.

    ``max_source_characters`` is a prioritization heuristic, NEVER a token
    budget or substitute for the exact Qwen preflight on P0/P1/P2 prompts.
    """
    if pilot_size < 1 or max_source_characters < 1:
        raise PilotPlanningError("PILOT_SIZE_AND_SOURCE_CHARACTER_LIMIT_MUST_BE_POSITIVE")
    source_rows = list(rows)
    ids_seen = {str(row["id"]) for row in source_rows}
    if len(ids_seen) != len(source_rows):
        raise PilotPlanningError("DUPLICATE_SOURCE_ID")
    exposed = set(str(value) for value in exposed_ids)
    exposed_cves = set().union(*(cves_of(row) for row in source_rows if row["id"] in exposed))

    counts = Counter()
    candidates: list[dict[str, Any]] = []
    for row in source_rows:
        counts["source_rows"] += 1
        structure = source_structure(row)
        if structure["declared_num_files"] >= 2:
            counts["declared_multifile"] += 1
            if structure["two_paths"]:
                counts["declared_multifile_with_two_parsed_paths"] += 1
            if structure["matched_paths"]:
                counts["declared_multifile_with_matched_path_sets"] += 1
        for flag in ("source_structurally_strict",):
            if structure[flag]:
                counts[flag] += 1
        if not structure["source_structurally_strict"]:
            continue
        row_cves = cves_of(row)
        if row["id"] in exposed or (row_cves and row_cves & exposed_cves):
            counts["excluded_exposed_id_or_cve_family"] += 1
            continue
        if not row_cves or not COMMIT_SHA.fullmatch(row["COMMIT_HASH"]):
            counts["excluded_missing_cve_or_commit_identity"] += 1
            continue
        counts["strict_after_exposure_and_identity"] += 1
        if structure["combined_source_characters"] > max_source_characters:
            counts["excluded_source_character_heuristic"] += 1
            continue
        candidates.append({
            "source_row_id": row["id"],
            "commit_hash": row["COMMIT_HASH"].lower(),
            "cve_ids": sorted(row_cves),
            "cwe_id": row["VULNERABILITY_CWE"].strip() or "UNKNOWN",
            "project": row["PROJECT"].strip() or "UNKNOWN",
            "declared_num_files": structure["declared_num_files"],
            "changed_path_count": structure["changed_path_count"],
            "combined_source_characters": structure["combined_source_characters"],
            "source_structure_status": "PASS_SOURCE_ONLY_NOT_CANONICAL",
            "gold_label_status": "NOT_VERIFIED",
            "exact_token_status": "NOT_RUN",
        })
    counts["shortlisted_after_character_heuristic"] = len(candidates)
    counts["shortlisted_distinct_cve_ids"] = len({cve for candidate in candidates for cve in candidate["cve_ids"]})

    selected: list[dict[str, Any]] = []
    used_cves: set[str] = set(exposed_cves)
    project_counts: Counter[str] = Counter()
    cwe_counts: Counter[str] = Counter()
    remaining = candidates.copy()
    while len(selected) < pilot_size:
        eligible = [c for c in remaining if not used_cves.intersection(c["cve_ids"])]
        if not eligible:
            raise PilotPlanningError(
                f"INSUFFICIENT_CVE_DISJOINT_DEVELOPMENT_CANDIDATES:{len(selected)}/{pilot_size}"
            )
        chosen = min(eligible, key=lambda c: (
            project_counts[c["project"]], cwe_counts[c["cwe_id"]],
            c["combined_source_characters"], int(c["source_row_id"]),
        ))
        selected.append(chosen)
        remaining.remove(chosen)
        used_cves.update(chosen["cve_ids"])
        project_counts[chosen["project"]] += 1
        cwe_counts[chosen["cwe_id"]] += 1

    selected_ids = [item["source_row_id"] for item in selected]
    return {
        "artifact_type": "vulnsage_source_only_development_pilot_candidates",
        "artifact_version": "2026-11-11.1",
        "official_pilot": False,
        "held_out_evaluation": False,
        "inference_authorized": False,
        "experimental_llm_calls": 0,
        "ground_truth_mutated": False,
        "scope": "DEVELOPMENT_SHORTLIST_ONLY_NO_INFERENCE",
        "counts": dict(sorted(counts.items())),
        "selection_contract": {
            "declared_multifile_required": True,
            "exact_path_set_and_declared_count_required": True,
            "at_least_two_changed_nonempty_files_required": True,
            "exclude_researcher_exposed_ids_and_entire_cve_families": True,
            "no_cve_overlap_within_selected_pilot": True,
            "max_combined_source_characters_heuristic": max_source_characters,
            "selection_order": "PROJECT_BALANCE_THEN_CWE_DIVERSITY_THEN_SHORTER_SOURCE_THEN_ID",
            "exact_model_context_fit": "NOT_CHECKED_MUST_RUN_EXACT_QWEN_TOKENIZER",
            "upstream_security_and_ground_truth_review": "NOT_CHECKED",
            "source_label_not_promoted_to_gold": True,
        },
        "ordered_source_ids": selected_ids,
        "sample_list_sha256": hashlib.sha256(json.dumps(selected_ids, separators=(",", ":")).encode()).hexdigest(),
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--max-source-characters", type=int, default=50_000)
    args = parser.parse_args()
    rows = load_source_rows(args.source_csv)
    plan = plan_pilot(rows, pilot_size=args.pilot_size, max_source_characters=args.max_source_characters)
    plan["source_csv_sha256"] = hashlib.sha256(args.source_csv.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "scope": plan["scope"],
        "sample_count": len(plan["selected"]),
        "source_csv_sha256": plan["source_csv_sha256"],
        "output": str(args.output.resolve()),
        "real_llm_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
