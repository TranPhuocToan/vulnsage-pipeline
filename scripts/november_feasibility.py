"""Deterministic, offline cohort-feasibility audit for the November fast track.

This tool deliberately fails closed.  It does not retrieve evidence, mutate a
canonical sample, approve a review, or infer that a missing gate passed.  The
two outputs are diagnostic research artifacts, not an official manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_VERSION = "november-fast-track-feasibility-1.0"
PASS = "PASS"

DERIVED_GAP_CODES = {
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
}

# These IDs have already been inspected, shortlisted, canonicalized, used for
# an engineering smoke, or discussed in candidate-review artifacts.  They may
# be used in a declared development cohort after approval, but not silently in
# held-out evaluation.
RESEARCHER_EXPOSED_IDS = {
    "8245", "12858", "14373", "15301", "16614", "16627", "16944",
    "17169", "17186", "17283", "17383", "17415", "17460", "18579",
    "18685", "19047",
}

EVIDENCE_REGISTRY: dict[str, dict[str, Any]] = {
    "17169": {
        "upstream_mapping_gate": PASS,
        "security_review_gate": "PENDING_HUMAN_DECISION",
        "evidence_refs": [
            "audits/artifacts/phase3d/pilot_upstream_review_20261109/upstream_evidence_17169.md"
        ],
    },
    "17415": {
        "upstream_mapping_gate": PASS,
        "security_review_gate": "PENDING_HUMAN_DECISION_AND_WAIVERS",
        "evidence_refs": [
            "audits/artifacts/phase3d/pilot_upstream_review_20261109/upstream_evidence_17415.md"
        ],
    },
    "17186": {
        "upstream_mapping_gate": "FAIL_SECURITY_MAPPING_MISMATCH",
        "security_review_gate": "BLOCKED_SECURITY_MAPPING_MISMATCH",
        "evidence_refs": [
            "audits/artifacts/phase3d/pilot_upstream_review_20261109/upstream_evidence_17186.md"
        ],
    },
    "19047": {
        "upstream_mapping_gate": PASS,
        "security_review_gate": "BLOCKED_ENGINEERING_SMOKE_SCOPE_ONLY",
        "evidence_refs": [
            "reviews/19047_independent_verification.md",
            "handoffs/phase3d_19047_independent_review.md",
        ],
    },
    "18579": {
        "upstream_mapping_gate": PASS,
        "security_review_gate": "PENDING_HUMAN_DECISION_AND_WAIVER",
        "evidence_refs": [
            "audits/artifacts/phase3d/security_review_preparation_20261109/18579_security_review_draft.md",
            "audits/artifacts/phase3d/security_review_preparation_20261109/18579_derived_gap_waiver_draft.md",
        ],
    },
    "16944": {
        "upstream_mapping_gate": "PARTIAL_ORDERED_TWO_COMMIT_SECURITY_UNIT",
        "security_review_gate": "BLOCKED_RESEARCH_UNIT_DECISION_PENDING",
        "evidence_refs": [
            "audits/artifacts/phase3d/security_review_preparation_20261109/16944_security_unit_analysis.md"
        ],
    },
    "14373": {
        "upstream_mapping_gate": PASS,
        "security_review_gate": "BLOCKED_AGGREGATE_CVE_SEMANTICS_PENDING",
        "evidence_refs": [
            "audits/artifacts/phase3d/security_review_preparation_20261109/14373_mapping_status.md"
        ],
    },
    "16627": {
        "upstream_mapping_gate": "NOT_ASSESSED_MULTI_COMMIT_FAMILY",
        "security_review_gate": "BLOCKED_MULTI_COMMIT_FAMILY",
        "evidence_refs": [
            "audits/artifacts/phase3d/pilot_cohort_readiness_20261109/candidate_readiness_matrix.md"
        ],
    },
}

MANDATORY_GATES = (
    "source_row_gate",
    "commit_uniqueness_gate",
    "family_scope_gate",
    "multi_file_gate",
    "phase2b_strict_gate",
    "canonical_evidence_gate",
    "upstream_mapping_gate",
    "derived_gap_gate",
    "context_p0_gate",
    "context_p1_gate",
    "context_p2_gate",
    "context_policy_authorization_gate",
    "security_review_gate",
    "ground_truth_review_gate",
    "smoke_overlap_gate",
)

CSV_FIELDS = (
    "source_id", "cve_id", "cwe_id", "project", "commit_hash",
    "declared_num_files", "parsed_num_files", "source_row_gate",
    "commit_uniqueness_gate", "family_scope_gate", "multi_file_gate",
    "phase2b_status", "phase2b_strict_gate", "canonical_sample_id",
    "canonical_evidence_gate", "upstream_mapping_gate", "derived_gap_gate",
    "commit_unit_gate", "context_p0_gate", "context_p1_gate",
    "context_p2_gate", "context_policy_authorization_gate",
    "security_review_gate", "ground_truth_label",
    "ground_truth_review_gate", "label_class", "smoke_overlap_gate",
    "researcher_exposure", "held_out_gate", "pilot_eligible",
    "main_eligible", "reason_codes", "evidence_refs",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _source_id(sample: Mapping[str, Any]) -> str | None:
    source = sample.get("source")
    if isinstance(source, Mapping) and source.get("source_row_id") is not None:
        return str(source["source_row_id"])
    value = sample.get("source_row_id") or sample.get("source_id")
    return str(value) if value is not None else None


def _canonical_evidence(sample: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    identity = sample.get("identity")
    patch = sample.get("full_patch")
    files = sample.get("files")
    if not isinstance(identity, Mapping):
        reasons.append("CANONICAL_IDENTITY_MISSING")
    else:
        for key in ("fix_commit_hash", "parent_commit_hash"):
            value = identity.get(key)
            if not isinstance(value, str) or len(value) != 40:
                reasons.append(f"{key.upper()}_INVALID")
    if not isinstance(patch, Mapping):
        reasons.append("ORIGINAL_PATCH_MISSING")
    else:
        text = patch.get("original_git_diff")
        expected = patch.get("original_git_diff_sha256")
        if not isinstance(text, str) or not text:
            reasons.append("ORIGINAL_PATCH_EMPTY")
        elif expected != sha256_bytes(text.encode("utf-8")):
            reasons.append("ORIGINAL_PATCH_HASH_MISMATCH")
    if not isinstance(files, list) or not files:
        reasons.append("CANONICAL_FILES_MISSING")
    else:
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                reasons.append(f"FILE_{index}_INVALID")
                continue
            operation = item.get("change_type")
            if operation not in {"MODIFY", "ADD", "DELETE", "RENAME"}:
                reasons.append(f"FILE_{index}_OPERATION_INVALID")
            before_path, after_path = item.get("path_before"), item.get("path_after")
            if operation in {"MODIFY", "DELETE", "RENAME"} and not before_path:
                reasons.append(f"FILE_{index}_BEFORE_PATH_MISSING")
            if operation in {"MODIFY", "ADD", "RENAME"} and not after_path:
                reasons.append(f"FILE_{index}_AFTER_PATH_MISSING")
            before, after = item.get("vulnerable_code"), item.get("patched_code")
            before_hash, after_hash = item.get("vulnerable_code_sha256"), item.get("patched_code_sha256")
            if operation in {"MODIFY", "DELETE", "RENAME"}:
                if not isinstance(before, str) or before_hash != sha256_bytes(before.encode("utf-8")):
                    reasons.append(f"FILE_{index}_BEFORE_BLOB_OR_HASH_INVALID")
            if operation in {"MODIFY", "ADD", "RENAME"}:
                if not isinstance(after, str) or after_hash != sha256_bytes(after.encode("utf-8")):
                    reasons.append(f"FILE_{index}_AFTER_BLOB_OR_HASH_INVALID")
    return (PASS if not reasons else "FAIL"), reasons


def load_canonical_samples(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = read_json(path)
        values = payload.get("samples") if isinstance(payload, Mapping) else None
        for sample in _as_list(values):
            if not isinstance(sample, dict):
                continue
            source_id = _source_id(sample)
            if source_id is None:
                continue
            if source_id in samples:
                raise ValueError(f"duplicate canonical source ID across inputs: {source_id}")
            samples[source_id] = sample
    return samples


def _context_from_token_record(candidate: Mapping[str, Any]) -> dict[str, str]:
    gates: dict[str, str] = {}
    pipelines = candidate.get("pipelines")
    if not isinstance(pipelines, Mapping):
        return {"P0": "NOT_ASSESSED", "P1": "NOT_ASSESSED", "P2": "NOT_ASSESSED"}
    for pipeline in ("P0", "P1", "P2"):
        detail = pipelines.get(pipeline)
        if not isinstance(detail, Mapping):
            gates[pipeline] = "NOT_ASSESSED"
            continue
        static_calls = _as_list(detail.get("static_calls"))
        if any(item.get("status") == "CONTEXT_OVERFLOW" for item in static_calls if isinstance(item, Mapping)):
            gates[pipeline] = "FAIL_CONTEXT_OVERFLOW"
        elif _as_list(detail.get("runtime_dependent_stages")):
            gates[pipeline] = "PARTIAL_RUNTIME_SYNTHESIS_GUARD_REQUIRED"
        elif static_calls and all(item.get("status") == "FIT_RAW_PROMPT" for item in static_calls if isinstance(item, Mapping)):
            gates[pipeline] = PASS
        else:
            gates[pipeline] = "NOT_ASSESSED"
    return gates


def load_context_gates(
    token_reports: Iterable[Path],
    candidate_inventory_path: Path,
    expansion_path: Path,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in token_reports:
        payload = read_json(path)
        for candidate in _as_list(payload.get("candidates") if isinstance(payload, Mapping) else None):
            if isinstance(candidate, Mapping) and candidate.get("source_id") is not None:
                result[str(candidate["source_id"])] = _context_from_token_record(candidate)

    inventory = read_json(candidate_inventory_path)
    for candidate in _as_list(inventory.get("candidates")):
        if not isinstance(candidate, Mapping):
            continue
        source_id = str(candidate.get("source_row_id"))
        exact = candidate.get("candidate_context_exact_tokens")
        if source_id == "19047" and isinstance(exact, Mapping):
            result[source_id] = {"P0": PASS, "P1": PASS, "P2": PASS}

    expansion = read_json(expansion_path)
    for candidate in _as_list(expansion.get("additional_canonical_candidates")):
        if not isinstance(candidate, Mapping):
            continue
        source_id = str(candidate.get("source_id"))
        exact = candidate.get("exact_static_context")
        if not isinstance(exact, Mapping):
            continue
        gates: dict[str, str] = {}
        p0 = exact.get("P0") if isinstance(exact.get("P0"), Mapping) else {}
        gates["P0"] = "FAIL_CONTEXT_OVERFLOW" if p0.get("status") == "CONTEXT_OVERFLOW" else PASS
        for pipeline in ("P1", "P2"):
            detail = exact.get(pipeline) if isinstance(exact.get(pipeline), Mapping) else {}
            text = json.dumps(detail, sort_keys=True)
            if "CONTEXT_OVERFLOW" in text or "local_overflow_indices" in detail:
                gates[pipeline] = "FAIL_CONTEXT_OVERFLOW"
            else:
                gates[pipeline] = "PARTIAL_RUNTIME_SYNTHESIS_GUARD_REQUIRED"
        result[source_id] = gates
    return result


def _derived_gaps(sample: Mapping[str, Any]) -> list[str]:
    values: set[str] = set()
    for container_name, key in (
        ("commit_verification", "reason_codes"),
        ("quality", "reason_codes"),
    ):
        container = sample.get(container_name)
        if isinstance(container, Mapping):
            values.update(str(item) for item in _as_list(container.get(key)))
    return sorted(values & DERIVED_GAP_CODES)


def row_is_qualified(row: Mapping[str, Any]) -> bool:
    return all(row.get(gate) == PASS for gate in MANDATORY_GATES)


def build_rows(
    validation: Mapping[str, Any],
    canonical_samples: Mapping[str, Mapping[str, Any]],
    context_gates: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    records = validation.get("records")
    if not isinstance(records, list):
        raise ValueError("validation records missing")
    cve_counts = Counter(str(item.get("cve_id")) for item in records)
    commit_counts = Counter(str(item.get("commit_hash")) for item in records)
    rows: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id"))
        canonical = canonical_samples.get(source_id)
        reasons = set(str(item) for item in _as_list(record.get("reason_codes")))
        evidence_refs: list[str] = ["audits/artifacts/phase2b/vulnerabilities_validation.json"]
        declared = record.get("declared_num_files")
        parsed = record.get("parsed_num_files")
        strict = (
            record.get("status") == PASS
            and isinstance(declared, int) and declared > 1
            and parsed == declared
            and record.get("nonempty_reconstructed_diff_count") == parsed
        )
        canonical_gate = "NOT_ASSESSED"
        canonical_reasons: list[str] = []
        gaps: list[str] = []
        ground_truth_label = "UNKNOWN"
        ground_truth_review = "NOT_ASSESSED"
        sample_id = ""
        if canonical is not None:
            canonical_gate, canonical_reasons = _canonical_evidence(canonical)
            gaps = _derived_gaps(canonical)
            sample_id = str(canonical.get("sample_id") or "")
            ground_truth = canonical.get("ground_truth")
            if isinstance(ground_truth, Mapping):
                ground_truth_label = str(ground_truth.get("label") or "UNKNOWN")
                review = ground_truth.get("review_status")
                ground_truth_review = PASS if review == "VERIFIED" else f"PENDING:{review or 'UNKNOWN'}"
            evidence_refs.append("canonical sample artifact")

        registry = EVIDENCE_REGISTRY.get(source_id, {})
        evidence_refs.extend(str(item) for item in registry.get("evidence_refs", []))
        upstream = str(registry.get("upstream_mapping_gate", "NOT_ASSESSED"))
        security = str(registry.get("security_review_gate", "NOT_ASSESSED"))

        if not gaps:
            derived_gate = PASS if canonical is not None else "NOT_ASSESSED"
        elif source_id == "19047":
            derived_gate = "BLOCKED_WAIVERS_ENGINEERING_SMOKE_ONLY"
        else:
            derived_gate = "PENDING_UNWAIVED_DERIVED_GAPS"

        family_gate = PASS if cve_counts[str(record.get("cve_id"))] == 1 else "BLOCKED_MULTI_RECORD_CVE_FAMILY"
        commit_gate = PASS if commit_counts[str(record.get("commit_hash"))] == 1 else "FAIL_DUPLICATE_COMMIT"
        context = context_gates.get(source_id, {})
        if source_id == "19047":
            policy_gate = "BLOCKED_ENGINEERING_SMOKE_SCOPE_ONLY"
        elif canonical is not None:
            policy_gate = "BLOCKED_NO_PILOT_MAIN_POLICY_AUTHORIZATION"
        else:
            policy_gate = "NOT_ASSESSED"
        multi_file_gate = PASS if isinstance(declared, int) and declared > 1 else "OUT_OF_SCOPE_NOT_OPERATIONAL_MULTIFILE"
        smoke_gate = "FAIL_ENGINEERING_SMOKE_OVERLAP" if source_id == "19047" else PASS
        exposed = source_id in RESEARCHER_EXPOSED_IDS
        heldout_gate = "EXCLUDE_RESEARCHER_EXPOSED" if exposed else PASS
        commit_unit_gate = family_gate
        if source_id == "16944":
            commit_unit_gate = "BLOCKED_TWO_COMMIT_UNIT_UNDECIDED"
        if source_id == "14373":
            commit_unit_gate = "BLOCKED_AGGREGATE_CVE_UNIT_UNDECIDED"

        row: dict[str, Any] = {
            "source_id": source_id,
            "cve_id": record.get("cve_id") or "",
            "cwe_id": record.get("cwe_id") or "",
            "project": record.get("project") or "",
            "commit_hash": record.get("commit_hash") or "",
            "declared_num_files": declared,
            "parsed_num_files": parsed,
            "source_row_gate": PASS,
            "commit_uniqueness_gate": commit_gate,
            "family_scope_gate": family_gate,
            "multi_file_gate": multi_file_gate,
            "phase2b_status": record.get("status") or "UNKNOWN",
            "phase2b_strict_gate": PASS if strict else "FAIL_NOT_IN_61_STRICT_PREFILTER",
            "canonical_sample_id": sample_id,
            "canonical_evidence_gate": canonical_gate,
            "upstream_mapping_gate": upstream,
            "derived_gap_gate": derived_gate,
            "commit_unit_gate": commit_unit_gate,
            "context_p0_gate": context.get("P0", "NOT_ASSESSED"),
            "context_p1_gate": context.get("P1", "NOT_ASSESSED"),
            "context_p2_gate": context.get("P2", "NOT_ASSESSED"),
            "context_policy_authorization_gate": policy_gate,
            "security_review_gate": security,
            "ground_truth_label": ground_truth_label,
            "ground_truth_review_gate": ground_truth_review,
            "label_class": "POSITIVE_SOURCE_DESIGNATED" if ground_truth_label == "PATCHED" else "UNKNOWN",
            "smoke_overlap_gate": smoke_gate,
            "researcher_exposure": "YES" if exposed else "NO",
            "held_out_gate": heldout_gate,
            "pilot_eligible": False,
            "main_eligible": False,
            "reason_codes": "",
            "evidence_refs": "",
        }
        reasons.update(canonical_reasons)
        reasons.update(gaps)
        for gate in MANDATORY_GATES:
            if row.get(gate) != PASS:
                reasons.add(f"{gate}={row.get(gate)}")
        qualified = row_is_qualified(row)
        row["pilot_eligible"] = qualified
        row["main_eligible"] = qualified and row["held_out_gate"] == PASS
        row["reason_codes"] = ";".join(sorted(reasons))
        row["evidence_refs"] = ";".join(dict.fromkeys(evidence_refs))
        rows.append(row)
    return sorted(rows, key=lambda item: int(item["source_id"]))


def build_summary(
    *, rows: Sequence[Mapping[str, Any]], validation: Mapping[str, Any],
    input_hashes: Mapping[str, str], audit_date: str,
) -> dict[str, Any]:
    count = lambda predicate: sum(1 for row in rows if predicate(row))
    canonical_rows = [row for row in rows if row["canonical_sample_id"]]
    reviewed_positive = count(
        lambda row: row["ground_truth_label"] == "PATCHED" and row["ground_truth_review_gate"] == PASS
    )
    reviewed_negative = count(
        lambda row: row["ground_truth_label"] == "NOT_PATCHED" and row["ground_truth_review_gate"] == PASS
    )
    gate_counts = {
        gate: dict(sorted(Counter(str(row[gate]) for row in rows).items()))
        for gate in MANDATORY_GATES
    }
    return {
        "artifact_type": "november_fast_track_cohort_feasibility",
        "artifact_version": AUDIT_VERSION,
        "audit_date_asia_saigon": audit_date,
        "official_dataset": False,
        "official_split": False,
        "inference_authorized": False,
        "experimental_llm_calls": 0,
        "inputs_sha256": dict(sorted(input_hashes.items())),
        "definitions": {
            "source_record": "One row in VulnSage vulnerabilities.csv.",
            "strict_prefilter_61": "Phase2B PASS, operational multi-file, exact parsed count, and all reconstructed diffs non-empty.",
            "canonical_sample": "A prototype normalized sample backed by original commit evidence; not automatically approved.",
            "qualified": "Every mandatory gate equals PASS; no inference or pending decision can substitute for PASS.",
        },
        "counts": {
            "source_records": len(rows),
            "unique_cves": len({row["cve_id"] for row in rows}),
            "unique_commits": len({row["commit_hash"] for row in rows}),
            "operational_multifile_records": count(lambda row: row["multi_file_gate"] == PASS),
            "phase2b_strict_prefilter_records": count(lambda row: row["phase2b_strict_gate"] == PASS),
            "canonical_prototype_records": len(canonical_rows),
            "canonical_multifile_records": sum(row["multi_file_gate"] == PASS for row in canonical_rows),
            "source_designated_positive_canonical_records": sum(row["ground_truth_label"] == "PATCHED" for row in canonical_rows),
            "human_reviewed_positive_gold": reviewed_positive,
            "human_reviewed_negative_gold": reviewed_negative,
            "pilot_eligible": count(lambda row: bool(row["pilot_eligible"])),
            "main_eligible": count(lambda row: bool(row["main_eligible"])),
            "researcher_exposed_records": count(lambda row: row["researcher_exposure"] == "YES"),
        },
        "upper_bounds": {
            "source_operational_multifile": count(lambda row: row["multi_file_gate"] == PASS),
            "phase2b_strict_prefilter_only_not_qualified": count(lambda row: row["phase2b_strict_gate"] == PASS),
            "canonical_multifile_with_commit_evidence": count(
                lambda row: row["multi_file_gate"] == PASS and row["canonical_evidence_gate"] == PASS
            ),
            "all_static_or_historical_exact_context_fit": count(
                lambda row: row["context_p0_gate"] == PASS
                and row["context_p1_gate"] in {PASS, "PARTIAL_RUNTIME_SYNTHESIS_GUARD_REQUIRED"}
                and row["context_p2_gate"] in {PASS, "PARTIAL_RUNTIME_SYNTHESIS_GUARD_REQUIRED"}
            ),
            "fully_qualified_now": count(lambda row: bool(row["pilot_eligible"])),
            "warning": "Every value except fully_qualified_now is an upper bound or prefilter count, not a runnable cohort.",
        },
        "gate_counts": gate_counts,
        "cohort_decision": {
            "status": "NO_GO",
            "pilot_target": "20-40",
            "main_target": "100-150",
            "reason": "No record passes every provenance, security, context-policy, human-review, and label gate.",
            "main_locked_until_pilot_passes": True,
            "engineering_smoke_19047_is_main": False,
        },
        "positive_only_metric_policy": {
            "allowed": [
                "Positive Acceptance Rate", "False Rejection Rate", "Not Sure Rate",
                "Technical Failure Rate", "Coverage", "Selective Accuracy",
                "tokens", "latency",
            ],
            "binary_accuracy_precision_recall_f1_allowed": False,
            "reason": "There is no human-reviewed negative gold class or valid two-class confusion matrix.",
        },
        "validation_summary_cross_check": validation.get("summary"),
    }


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write(path, buffer.getvalue().encode("utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--canonical-json", type=Path, action="append", required=True)
    parser.add_argument("--token-report", type=Path, action="append", required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--candidate-expansion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-date", default="2026-11-09")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [
        args.source_csv, args.validation_json, *args.canonical_json,
        *args.token_report, args.candidate_inventory, args.candidate_expansion,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input artifacts: {missing}")
    validation = read_json(args.validation_json)
    source_contract = validation.get("source")
    actual_source_hash = sha256_file(args.source_csv)
    if not isinstance(source_contract, Mapping) or source_contract.get("sha256_before") != actual_source_hash:
        raise ValueError("source CSV hash does not match Phase2B validation provenance")
    canonical = load_canonical_samples(args.canonical_json)
    context = load_context_gates(
        args.token_report, args.candidate_inventory, args.candidate_expansion
    )
    rows = build_rows(validation, canonical, context)
    input_hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    summary = build_summary(
        rows=rows, validation=validation, input_hashes=input_hashes,
        audit_date=args.audit_date,
    )
    if summary["counts"]["source_records"] != 593:
        raise ValueError("unexpected source-record count")
    if summary["counts"]["phase2b_strict_prefilter_records"] != 61:
        raise ValueError("strict Phase2B prefilter no longer equals the evidenced 61")
    write_csv(args.output_dir / "candidate_gate_matrix.csv", rows)
    atomic_write(
        args.output_dir / "cohort_feasibility.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    print(json.dumps(summary["cohort_decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
