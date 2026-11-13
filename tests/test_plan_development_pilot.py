"""No-network regression tests for source-only development candidate planning."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plan_development_pilot import (
    PilotPlanningError, load_source_rows, plan_pilot, source_structure, split_source_blocks,
)


def row(
    source_id: int, cve: str, *, project: str = "linux", cwe: str = "CWE-20",
    declared: int = 2, before: str | None = None, after: str | None = None,
) -> dict[str, str]:
    return {
        "id": str(source_id),
        "COMMIT_HASH": f"{source_id:040x}",
        "VULNERABILITY_CVE": cve,
        "VULNERABILITY_CWE": cwe,
        "NUM_FILES_CHANGED": str(declared),
        "VULNERABLE_CODE_BLOCK": before if before is not None else (
            "// File: a.c\nold_a();\n// File path: b.c\nold_b();\n"
        ),
        "PATCHED_CODE_BLOCK": after if after is not None else (
            "// File: a.c\nnew_a();\n// File path: b.c\nnew_b();\n"
        ),
        "PROJECT": project,
    }


class SourceOnlyPilotPlannerTests(unittest.TestCase):
    def test_parses_both_header_formats_and_exact_changed_files(self):
        structure = source_structure(row(1, "CVE-2020-0001"))
        self.assertTrue(structure["source_structurally_strict"])
        self.assertEqual(structure["parsed_num_files"], 2)
        self.assertEqual(structure["changed_path_count"], 2)

    def test_rejects_missing_extra_duplicate_or_empty_file_fragments(self):
        base = row(2, "CVE-2020-0002")
        self.assertFalse(source_structure({**base, "NUM_FILES_CHANGED": "3"})["source_structurally_strict"])
        self.assertFalse(source_structure({**base, "PATCHED_CODE_BLOCK": "// File: a.c\nx\n// File: c.c\ny"})["source_structurally_strict"])
        self.assertIsNone(split_source_blocks("// File: a.c\nx\n// File: a.c\ny"))
        empty = row(3, "CVE-2020-0003", before="// File: a.c\n// File path: b.c\nold_b()")
        self.assertFalse(source_structure(empty)["source_structurally_strict"])

    def test_excludes_exposed_cve_family_and_prevents_pilot_leakage(self):
        records = [
            row(1, "CVE-2020-0001", project="linux"),
            row(2, "CVE-2020-0001", project="xen"),
            row(3, "CVE-2020-0002", project="mozilla"),
            row(4, "CVE-2020-0003", project="linux"),
            row(5, "CVE-2020-0004", project="xen"),
        ]
        plan = plan_pilot(records, pilot_size=2, exposed_ids={"1"})
        self.assertNotIn("1", plan["ordered_source_ids"])
        self.assertNotIn("2", plan["ordered_source_ids"])
        self.assertEqual(len(plan["ordered_source_ids"]), 2)
        self.assertFalse(plan["official_pilot"])
        self.assertFalse(plan["inference_authorized"])
        self.assertEqual(plan["experimental_llm_calls"], 0)
        self.assertEqual(plan["selection_contract"]["exact_model_context_fit"], "NOT_CHECKED_MUST_RUN_EXACT_QWEN_TOKENIZER")
        self.assertTrue(all(item["gold_label_status"] == "NOT_VERIFIED" for item in plan["selected"]))
        reversed_plan = plan_pilot(list(reversed(records)), pilot_size=2, exposed_ids={"1"})
        self.assertEqual(plan["ordered_source_ids"], reversed_plan["ordered_source_ids"])
        self.assertEqual(plan["sample_list_sha256"], reversed_plan["sample_list_sha256"])

    def test_cve_overlap_within_selection_is_forbidden(self):
        records = [
            row(1, "CVE-2020-0001 and CVE-2020-0002"),
            row(2, "CVE-2020-0002"),
            row(3, "CVE-2020-0003"),
        ]
        plan = plan_pilot(records, pilot_size=2, exposed_ids=set())
        selected_cves = [set(item["cve_ids"]) for item in plan["selected"]]
        self.assertFalse(selected_cves[0] & selected_cves[1])
        with self.assertRaisesRegex(PilotPlanningError, "INSUFFICIENT_CVE_DISJOINT"):
            plan_pilot(records[:2], pilot_size=2, exposed_ids=set())

    def test_large_multiline_csv_is_read_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            entry = row(11, "CVE-2020-0011")
            entry["VULNERABLE_CODE_BLOCK"] += "x" * 140_000
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(entry))
                writer.writeheader()
                writer.writerow(entry)
            parsed = load_source_rows(path)
            self.assertEqual(parsed[0]["VULNERABLE_CODE_BLOCK"], entry["VULNERABLE_CODE_BLOCK"])
            with path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=list(entry)).writerow(entry)
            with self.assertRaisesRegex(PilotPlanningError, "SOURCE_ID_INVALID_OR_DUPLICATE"):
                load_source_rows(path)


if __name__ == "__main__":
    unittest.main()
