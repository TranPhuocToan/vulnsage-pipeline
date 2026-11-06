"""Deterministic, read-only validation for VulnSage source CSV files.

The validator never rewrites its input.  It emits a derived JSON report containing
one status and a list of reason codes for every source row.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


VALIDATOR_VERSION = "1.0.0"
FILE_MARKER_RE = re.compile(r"(?m)^//\s*File(?:\s+path)?:\s*(.+)$")


class ValidationStatus(IntEnum):
    """Ordered severity used to derive a record's final status."""

    PASS = 0
    WARNING = 1
    UNRESOLVED = 2
    FAIL = 3


@dataclass(frozen=True)
class ReasonDefinition:
    severity: ValidationStatus
    description: str
    commit_level_review: bool = False


REASON_CATALOG: dict[str, ReasonDefinition] = {
    "MISSING_REQUIRED_FIELD": ReasonDefinition(
        ValidationStatus.FAIL, "A required source field is missing or blank."
    ),
    "INVALID_NUM_FILES_CHANGED": ReasonDefinition(
        ValidationStatus.FAIL, "NUM_FILES_CHANGED is not a non-negative integer."
    ),
    "INVALID_NUM_FUNCTIONS_CHANGED": ReasonDefinition(
        ValidationStatus.FAIL, "NUM_FUNCTIONS_CHANGED is not a non-negative integer."
    ),
    "DUPLICATE_SOURCE_ID": ReasonDefinition(
        ValidationStatus.FAIL, "The source row ID is not unique."
    ),
    "DUPLICATE_COMMIT_HASH": ReasonDefinition(
        ValidationStatus.FAIL, "The commit hash is not unique in the source dataset."
    ),
    "DUPLICATE_CVE_FAMILY": ReasonDefinition(
        ValidationStatus.WARNING,
        "The CVE occurs in more than one source row; group it during splitting.",
    ),
    "MISSING_VULNERABLE_FILE_MARKER": ReasonDefinition(
        ValidationStatus.FAIL, "The vulnerable block has no parseable file marker."
    ),
    "MISSING_PATCHED_FILE_MARKER": ReasonDefinition(
        ValidationStatus.FAIL, "The patched block has no parseable file marker."
    ),
    "EMPTY_FILE_PATH": ReasonDefinition(
        ValidationStatus.FAIL, "A file marker contains an empty path."
    ),
    "DUPLICATE_VULNERABLE_PATH": ReasonDefinition(
        ValidationStatus.FAIL,
        "A vulnerable-side path occurs more than once and would be overwritten by dict parsing.",
    ),
    "DUPLICATE_PATCHED_PATH": ReasonDefinition(
        ValidationStatus.FAIL,
        "A patched-side path occurs more than once and would be overwritten by dict parsing.",
    ),
    "FILE_COUNT_MISMATCH": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "Declared file count differs from the union of parsed paths.",
        True,
    ),
    "OPERATIONAL_G3_PARSE_LT_2": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "The row declares multiple files but fewer than two paths were parsed.",
        True,
    ),
    "PATH_SET_MISMATCH_REVIEW": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "Before/after path sets differ; add, delete, rename, or extraction failure must be reviewed.",
        True,
    ),
    "EMPTY_VULNERABLE_CONTENT": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "At least one parsed path has empty vulnerable-side content.",
        True,
    ),
    "EMPTY_PATCHED_CONTENT": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "At least one parsed path has empty patched-side content.",
        True,
    ),
    "ALL_RECONSTRUCTED_DIFFS_EMPTY": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "No changed line can be reconstructed from any parsed file pair.",
        True,
    ),
    "UNCHANGED_FILE_BLOCK": ReasonDefinition(
        ValidationStatus.WARNING,
        "At least one file pair is identical although another file in the row changes.",
    ),
    "MISSING_NOISE_SCORE": ReasonDefinition(
        ValidationStatus.WARNING,
        "Noise score is missing; it must remain null rather than default to zero.",
    ),
    "INVALID_NOISE_SCORE": ReasonDefinition(
        ValidationStatus.WARNING, "Noise score is not numeric or is outside [0, 100]."
    ),
    "BRACE_IMBALANCE_VULNERABLE": ReasonDefinition(
        ValidationStatus.WARNING,
        "A vulnerable block fails the existing brace-balance heuristic.",
    ),
    "BRACE_IMBALANCE_PATCHED": ReasonDefinition(
        ValidationStatus.WARNING,
        "A patched block fails the existing brace-balance heuristic.",
    ),
    "MISSING_LABEL": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "A configured patch-verification label column is present but this row is blank.",
        True,
    ),
    "INVALID_LABEL": ReasonDefinition(
        ValidationStatus.UNRESOLVED,
        "Patch-verification label is outside PATCHED, NOT_PATCHED, or UNKNOWN.",
        True,
    ),
}


REQUIRED_COLUMNS = (
    "id",
    "COMMIT_HASH",
    "VULNERABILITY_CVE",
    "VULNERABILITY_CWE",
    "PROJECT",
    "VULNERABLE_CODE_BLOCK",
    "PATCHED_CODE_BLOCK",
    "NUM_FILES_CHANGED",
    "NUM_FUNCTIONS_CHANGED",
)
LABEL_COLUMN_CANDIDATES = ("label", "ground_truth", "PATCH_LABEL")
VALID_LABELS = {"PATCHED", "NOT_PATCHED", "UNKNOWN"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _parse_nonnegative_int(value: Any) -> int | None:
    if _blank(value):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _parse_noise(value: Any) -> tuple[float | None, bool]:
    if _blank(value):
        return None, True
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None, False
    return (parsed, True) if 0.0 <= parsed <= 100.0 else (parsed, False)


def _remove_strings_and_comments(code: str) -> str:
    pattern = re.compile(
        r"//.*?$|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub("", code)


def _brace_balanced(code: str) -> bool:
    cleaned = _remove_strings_and_comments(code)
    return cleaned.count("{") == cleaned.count("}")


@dataclass
class ParsedBlock:
    files: dict[str, str]
    ordered_paths: list[str]
    duplicate_paths: list[str]
    empty_paths: int
    marker_count: int


def split_marked_code(code_block: Any) -> ParsedBlock:
    """Split a source block without using the guide's synthetic main.c fallback."""

    if not isinstance(code_block, str) or not code_block.strip():
        return ParsedBlock({}, [], [], 0, 0)

    matches = list(FILE_MARKER_RE.finditer(code_block))
    files: dict[str, str] = {}
    ordered_paths: list[str] = []
    duplicate_paths: list[str] = []
    empty_paths = 0

    for index, match in enumerate(matches):
        path = match.group(1).strip()
        if not path:
            empty_paths += 1
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(code_block)
        content = code_block[match.end() : end].strip()
        if path in files:
            duplicate_paths.append(path)
        else:
            files[path] = content
            ordered_paths.append(path)

    return ParsedBlock(files, ordered_paths, duplicate_paths, empty_paths, len(matches))


def reconstructed_diff(before: str, after: str, path: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    return "\n".join(lines) + ("\n" if lines else "")


class DatasetValidator:
    """Validate VulnSage rows and produce a deterministic report."""

    def validate_csv(self, csv_path: str | Path) -> dict[str, Any]:
        path = Path(csv_path).resolve()
        before_hash = sha256_file(path)
        previous_field_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(sys.maxsize)
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        finally:
            csv.field_size_limit(previous_field_limit)
        after_hash = sha256_file(path)
        if before_hash != after_hash:
            raise RuntimeError("Source CSV changed while it was being validated")
        report = self.validate_rows(rows, fieldnames=fieldnames)
        report["source"] = {
            "path": str(path),
            "sha256_before": before_hash,
            "sha256_after": after_hash,
            "unchanged": before_hash == after_hash,
            "size_bytes": path.stat().st_size,
        }
        return report

    def validate_rows(
        self, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None
    ) -> dict[str, Any]:
        fieldnames = list(fieldnames or (rows[0].keys() if rows else []))
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
        label_column = next((name for name in LABEL_COLUMN_CANDIDATES if name in fieldnames), None)

        id_counts = Counter(str(row.get("id", "")).strip() for row in rows if not _blank(row.get("id")))
        commit_counts = Counter(
            str(row.get("COMMIT_HASH", "")).strip()
            for row in rows
            if not _blank(row.get("COMMIT_HASH"))
        )
        cve_counts = Counter(
            str(row.get("VULNERABILITY_CVE", "")).strip()
            for row in rows
            if not _blank(row.get("VULNERABILITY_CVE"))
        )

        records = [
            self._validate_row(row, id_counts, commit_counts, cve_counts, label_column)
            for row in rows
        ]
        records.sort(key=lambda record: self._identity_sort_key(record["source_id"]))

        status_counts = Counter(record["status"] for record in records)
        reason_counts = Counter(reason for record in records for reason in record["reason_codes"])
        review_records = [
            record["source_id"]
            for record in records
            if any(REASON_CATALOG[reason].commit_level_review for reason in record["reason_codes"])
        ]

        return {
            "validator_version": VALIDATOR_VERSION,
            "input_contract": {
                "required_columns": list(REQUIRED_COLUMNS),
                "missing_required_columns": missing_columns,
                "label_column_present": label_column is not None,
                "label_column": label_column,
                "label_note": (
                    "The raw VulnSage CSV is label-free by design; no per-row label error is emitted."
                    if label_column is None
                    else "The configured label column is validated per row."
                ),
            },
            "status_definitions": {
                "PASS": "No validation reason was found.",
                "WARNING": "Parseable and identifiable, but contains a non-blocking caveat.",
                "UNRESOLVED": "Requires commit-level or label-provenance review before inclusion.",
                "FAIL": "Cannot be parsed or identified safely under the source validation contract.",
            },
            "reason_catalog": {
                code: {
                    "severity": definition.severity.name,
                    "description": definition.description,
                    "commit_level_review": definition.commit_level_review,
                }
                for code, definition in sorted(REASON_CATALOG.items())
            },
            "summary": {
                "total_records": len(records),
                "status_counts": {
                    status.name: status_counts.get(status.name, 0)
                    for status in ValidationStatus
                },
                "reason_counts": dict(sorted(reason_counts.items())),
                "unique_source_ids": len(id_counts),
                "unique_commits": len(commit_counts),
                "unique_cves": len(cve_counts),
                "duplicate_source_id_values": sum(count > 1 for count in id_counts.values()),
                "duplicate_commit_values": sum(count > 1 for count in commit_counts.values()),
                "duplicate_cve_families": sum(count > 1 for count in cve_counts.values()),
                "commit_level_review_records": len(review_records),
                "operational_g3_declared_candidates": sum(
                    record["declared_num_files"] is not None
                    and record["declared_num_files"] > 1
                    for record in records
                ),
            },
            "commit_level_review_source_ids": review_records,
            "records": records,
        }

    @staticmethod
    def _identity_sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, value)

    def _validate_row(
        self,
        row: Mapping[str, Any],
        id_counts: Counter,
        commit_counts: Counter,
        cve_counts: Counter,
        label_column: str | None,
    ) -> dict[str, Any]:
        reasons: set[str] = set()
        source_id = str(row.get("id", "")).strip()
        commit_hash = str(row.get("COMMIT_HASH", "")).strip()
        cve_id = str(row.get("VULNERABILITY_CVE", "")).strip()
        cwe_id = str(row.get("VULNERABILITY_CWE", "")).strip()
        project = str(row.get("PROJECT", "")).strip()

        for field in REQUIRED_COLUMNS:
            if field not in row or _blank(row.get(field)):
                reasons.add("MISSING_REQUIRED_FIELD")

        if source_id and id_counts[source_id] > 1:
            reasons.add("DUPLICATE_SOURCE_ID")
        if commit_hash and commit_counts[commit_hash] > 1:
            reasons.add("DUPLICATE_COMMIT_HASH")
        if cve_id and cve_counts[cve_id] > 1:
            reasons.add("DUPLICATE_CVE_FAMILY")

        declared_files = _parse_nonnegative_int(row.get("NUM_FILES_CHANGED"))
        declared_functions = _parse_nonnegative_int(row.get("NUM_FUNCTIONS_CHANGED"))
        if declared_files is None:
            reasons.add("INVALID_NUM_FILES_CHANGED")
        if declared_functions is None:
            reasons.add("INVALID_NUM_FUNCTIONS_CHANGED")

        vulnerable = split_marked_code(row.get("VULNERABLE_CODE_BLOCK"))
        patched = split_marked_code(row.get("PATCHED_CODE_BLOCK"))
        if vulnerable.marker_count == 0:
            reasons.add("MISSING_VULNERABLE_FILE_MARKER")
        if patched.marker_count == 0:
            reasons.add("MISSING_PATCHED_FILE_MARKER")
        if vulnerable.empty_paths or patched.empty_paths:
            reasons.add("EMPTY_FILE_PATH")
        if vulnerable.duplicate_paths:
            reasons.add("DUPLICATE_VULNERABLE_PATH")
        if patched.duplicate_paths:
            reasons.add("DUPLICATE_PATCHED_PATH")

        vulnerable_paths = set(vulnerable.files)
        patched_paths = set(patched.files)
        union_paths = sorted(vulnerable_paths | patched_paths)
        before_only = sorted(vulnerable_paths - patched_paths)
        after_only = sorted(patched_paths - vulnerable_paths)

        if declared_files is not None and declared_files != len(union_paths):
            reasons.add("FILE_COUNT_MISMATCH")
        if declared_files is not None and declared_files > 1 and len(union_paths) < 2:
            reasons.add("OPERATIONAL_G3_PARSE_LT_2")
        if vulnerable_paths != patched_paths:
            reasons.add("PATH_SET_MISMATCH_REVIEW")

        empty_vulnerable_paths = sorted(
            path for path in union_paths if not vulnerable.files.get(path, "").strip()
        )
        empty_patched_paths = sorted(
            path for path in union_paths if not patched.files.get(path, "").strip()
        )
        if empty_vulnerable_paths:
            reasons.add("EMPTY_VULNERABLE_CONTENT")
        if empty_patched_paths:
            reasons.add("EMPTY_PATCHED_CONTENT")

        diffs = {
            path: reconstructed_diff(
                vulnerable.files.get(path, ""), patched.files.get(path, ""), path
            )
            for path in union_paths
        }
        nonempty_diff_count = sum(bool(diff) for diff in diffs.values())
        if union_paths and nonempty_diff_count == 0:
            reasons.add("ALL_RECONSTRUCTED_DIFFS_EMPTY")
        elif any(not diff for diff in diffs.values()):
            reasons.add("UNCHANGED_FILE_BLOCK")

        if any(not _brace_balanced(code) for code in vulnerable.files.values() if code):
            reasons.add("BRACE_IMBALANCE_VULNERABLE")
        if any(not _brace_balanced(code) for code in patched.files.values() if code):
            reasons.add("BRACE_IMBALANCE_PATCHED")

        noise_score, noise_valid = _parse_noise(row.get("NOISE_AMOUNT"))
        if _blank(row.get("NOISE_AMOUNT")):
            reasons.add("MISSING_NOISE_SCORE")
        elif not noise_valid:
            reasons.add("INVALID_NOISE_SCORE")

        label = None
        if label_column is not None:
            label = str(row.get(label_column, "")).strip().upper()
            if not label:
                reasons.add("MISSING_LABEL")
            elif label not in VALID_LABELS:
                reasons.add("INVALID_LABEL")

        severity = max(
            (REASON_CATALOG[reason].severity for reason in reasons),
            default=ValidationStatus.PASS,
        )
        ordered_reasons = sorted(
            reasons, key=lambda reason: (-int(REASON_CATALOG[reason].severity), reason)
        )

        return {
            "source_id": source_id,
            "commit_hash": commit_hash,
            "cve_id": cve_id,
            "cwe_id": cwe_id,
            "project": project,
            "status": severity.name,
            "reason_codes": ordered_reasons,
            "declared_num_files": declared_files,
            "declared_num_functions": declared_functions,
            "parsed_num_files": len(union_paths),
            "vulnerable_marker_count": vulnerable.marker_count,
            "patched_marker_count": patched.marker_count,
            "parsed_paths": union_paths,
            "before_only_paths": before_only,
            "after_only_paths": after_only,
            "duplicate_vulnerable_paths": sorted(set(vulnerable.duplicate_paths)),
            "duplicate_patched_paths": sorted(set(patched.duplicate_paths)),
            "empty_vulnerable_paths": empty_vulnerable_paths,
            "empty_patched_paths": empty_patched_paths,
            "nonempty_reconstructed_diff_count": nonempty_diff_count,
            "noise_score": noise_score,
            "label": label,
            "needs_commit_level_review": any(
                REASON_CATALOG[reason].commit_level_review for reason in reasons
            ),
        }


def write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only VulnSage CSV validator")
    parser.add_argument("--input", required=True, help="Path to source vulnerabilities.csv")
    parser.add_argument("--output", required=True, help="Path to derived JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = DatasetValidator().validate_csv(args.input)
    write_report(report, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
