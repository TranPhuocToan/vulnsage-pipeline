import csv
import tempfile
import unittest
from pathlib import Path

from dataset_validation.validator import DatasetValidator, reconstructed_diff


BASE_FIELDS = [
    "id",
    "COMMIT_HASH",
    "VULNERABILITY_CVE",
    "VULNERABILITY_CWE",
    "PROJECT",
    "VULNERABLE_CODE_BLOCK",
    "PATCHED_CODE_BLOCK",
    "NUM_FILES_CHANGED",
    "NUM_FUNCTIONS_CHANGED",
    "NOISE_AMOUNT",
]


def make_row(**overrides):
    row = {
        "id": "1",
        "COMMIT_HASH": "a" * 40,
        "VULNERABILITY_CVE": "CVE-2020-0001",
        "VULNERABILITY_CWE": "CWE-119",
        "PROJECT": "example",
        "VULNERABLE_CODE_BLOCK": (
            "// File: src/a.c\nint a(void) { return 0; }\n"
            "// File: src/b.c\nint b(void) { return 0; }"
        ),
        "PATCHED_CODE_BLOCK": (
            "// File: src/a.c\nint a(void) { return 1; }\n"
            "// File: src/b.c\nint b(void) { return 2; }"
        ),
        "NUM_FILES_CHANGED": "2",
        "NUM_FUNCTIONS_CHANGED": "2",
        "NOISE_AMOUNT": "0",
    }
    row.update(overrides)
    return row


class DatasetValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = DatasetValidator()

    def validate(self, rows, fieldnames=None):
        return self.validator.validate_rows(rows, fieldnames or list(rows[0]))

    def test_valid_multifile_sample_passes(self):
        report = self.validate([make_row()])
        record = report["records"][0]
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["reason_codes"], [])
        self.assertEqual(record["parsed_num_files"], 2)

    def test_missing_file_marker_fails(self):
        row = make_row(VULNERABLE_CODE_BLOCK="int a(void) { return 0; }")
        record = self.validate([row])["records"][0]
        self.assertEqual(record["status"], "FAIL")
        self.assertIn("MISSING_VULNERABLE_FILE_MARKER", record["reason_codes"])

    def test_duplicate_path_fails_without_overwriting_silently(self):
        row = make_row(
            VULNERABLE_CODE_BLOCK=(
                "// File: src/a.c\nint a = 0;\n"
                "// File: src/a.c\nint a = 1;"
            )
        )
        record = self.validate([row])["records"][0]
        self.assertEqual(record["status"], "FAIL")
        self.assertIn("DUPLICATE_VULNERABLE_PATH", record["reason_codes"])
        self.assertEqual(record["duplicate_vulnerable_paths"], ["src/a.c"])

    def test_empty_diff_is_unresolved(self):
        row = make_row()
        row["PATCHED_CODE_BLOCK"] = row["VULNERABLE_CODE_BLOCK"]
        record = self.validate([row])["records"][0]
        self.assertEqual(record["status"], "UNRESOLVED")
        self.assertIn("ALL_RECONSTRUCTED_DIFFS_EMPTY", record["reason_codes"])

    def test_reconstructed_diff_keeps_single_line_changes_separate(self):
        diff = reconstructed_diff("old", "new", "a.c")
        self.assertIn("\n-old\n+new\n", diff)

    def test_declared_and_parsed_file_count_mismatch_is_unresolved(self):
        row = make_row(NUM_FILES_CHANGED="3")
        record = self.validate([row])["records"][0]
        self.assertEqual(record["status"], "UNRESOLVED")
        self.assertIn("FILE_COUNT_MISMATCH", record["reason_codes"])

    def test_add_file_is_flagged_for_review_not_automatically_failed(self):
        row = make_row(
            VULNERABLE_CODE_BLOCK="// File: src/a.c\nint a(void) { return 0; }",
            PATCHED_CODE_BLOCK=(
                "// File: src/a.c\nint a(void) { return 1; }\n"
                "// File: src/new.c\nint new_file(void) { return 1; }"
            ),
        )
        record = self.validate([row])["records"][0]
        self.assertEqual(record["status"], "UNRESOLVED")
        self.assertNotEqual(record["status"], "FAIL")
        self.assertIn("PATH_SET_MISMATCH_REVIEW", record["reason_codes"])
        self.assertEqual(record["after_only_paths"], ["src/new.c"])
        self.assertTrue(record["needs_commit_level_review"])

    def test_delete_file_is_flagged_for_review_not_automatically_failed(self):
        row = make_row(
            PATCHED_CODE_BLOCK="// File: src/a.c\nint a(void) { return 1; }"
        )
        record = self.validate([row])["records"][0]
        self.assertEqual(record["status"], "UNRESOLVED")
        self.assertIn("PATH_SET_MISMATCH_REVIEW", record["reason_codes"])
        self.assertEqual(record["before_only_paths"], ["src/b.c"])

    def test_missing_noise_is_distinct_from_zero(self):
        missing = self.validate([make_row(NOISE_AMOUNT="")])["records"][0]
        zero = self.validate([make_row(NOISE_AMOUNT="0")])["records"][0]
        self.assertEqual(missing["status"], "WARNING")
        self.assertIn("MISSING_NOISE_SCORE", missing["reason_codes"])
        self.assertIsNone(missing["noise_score"])
        self.assertEqual(zero["status"], "PASS")
        self.assertEqual(zero["noise_score"], 0.0)

    def test_missing_and_invalid_labels_are_unresolved_when_column_exists(self):
        fields = BASE_FIELDS + ["label"]
        missing_row = make_row(label="")
        invalid_row = make_row(id="2", COMMIT_HASH="b" * 40, label="MAYBE")
        report = self.validate([missing_row, invalid_row], fields)
        by_id = {record["source_id"]: record for record in report["records"]}
        self.assertIn("MISSING_LABEL", by_id["1"]["reason_codes"])
        self.assertIn("INVALID_LABEL", by_id["2"]["reason_codes"])
        self.assertEqual(by_id["1"]["status"], "UNRESOLVED")
        self.assertEqual(by_id["2"]["status"], "UNRESOLVED")

    def test_duplicate_source_identity_fails_each_affected_row(self):
        first = make_row()
        second = make_row(VULNERABILITY_CVE="CVE-2020-0002")
        report = self.validate([first, second])
        for record in report["records"]:
            self.assertEqual(record["status"], "FAIL")
            self.assertIn("DUPLICATE_SOURCE_ID", record["reason_codes"])
            self.assertIn("DUPLICATE_COMMIT_HASH", record["reason_codes"])

    def test_csv_validation_leaves_source_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=BASE_FIELDS)
                writer.writeheader()
                writer.writerow(make_row())
            original = path.read_bytes()
            report = self.validator.validate_csv(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(report["source"]["unchanged"])
            self.assertEqual(
                report["source"]["sha256_before"], report["source"]["sha256_after"]
            )

    def test_csv_reader_accepts_source_blocks_larger_than_default_field_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.csv"
            row = make_row(
                VULNERABLE_CODE_BLOCK="// File: src/a.c\n" + ("int x;\n" * 20000),
                PATCHED_CODE_BLOCK="// File: src/a.c\n" + ("int y;\n" * 20000),
                NUM_FILES_CHANGED="1",
            )
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=BASE_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            report = self.validator.validate_csv(path)
            self.assertEqual(report["summary"]["total_records"], 1)
            self.assertTrue(report["source"]["unchanged"])


if __name__ == "__main__":
    unittest.main()
