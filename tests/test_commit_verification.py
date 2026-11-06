import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dataset_validation.commit_verification import verify_commit_evidence
from dataset_validation.representation import build_derived_representation


def row(before, after, commit="a" * 40, source_id="1"):
    return {
        "id": source_id,
        "COMMIT_HASH": commit,
        "VULNERABILITY_CVE": "CVE-2020-0001",
        "VULNERABILITY_CWE": "CWE-119",
        "PROJECT": "example",
        "VULNERABLE_CODE_BLOCK": before,
        "PATCHED_CODE_BLOCK": after,
        "NUM_FILES_CHANGED": "1",
        "NUM_FUNCTIONS_CHANGED": "1",
    }


def file_evidence(path, status, patch, additions, deletions, previous_path=None):
    return {
        "path": path,
        "previous_path": previous_path,
        "status": status,
        "patch": patch,
        "patch_complete": True,
        "additions": additions,
        "deletions": deletions,
    }


def evidence(files, status="VERIFIED", commit="a" * 40):
    return {
        "verification_status": status,
        "commit_sha": commit if status == "VERIFIED" else None,
        "parent_commit_sha": "b" * 40 if status == "VERIFIED" else None,
        "source_url": "https://example.test/commit",
        "files": files,
        "original_diff": None,
        "failure_reason": None if status == "VERIFIED" else "not found",
    }


class CommitVerificationTests(unittest.TestCase):
    def test_modified_add_only_fragment_is_not_file_status_mismatch(self):
        sample = row("// File: a.c", "// File: a.c\nint added;")
        source = evidence(
            [file_evidence("a.c", "MODIFIED", "@@ -0,0 +1 @@\n+int added;", 1, 0)]
        )
        result = verify_commit_evidence(sample, source)
        self.assertEqual(result["coverage"]["path"]["ratio"], 1.0)
        self.assertNotIn("FILE_STATUS_MISMATCH", result["reason_codes"])

    def test_modified_delete_only_fragment_is_not_file_status_mismatch(self):
        sample = row("// File: a.c\nint removed;", "// File: a.c")
        source = evidence(
            [file_evidence("a.c", "MODIFIED", "@@ -1 +0,0 @@\n-int removed;", 0, 1)]
        )
        result = verify_commit_evidence(sample, source)
        self.assertEqual(result["coverage"]["path"]["ratio"], 1.0)
        self.assertNotIn("FILE_STATUS_MISMATCH", result["reason_codes"])

    def test_true_add_and_delete_are_preserved(self):
        sample = row("// File: old.c\nint old;", "// File: new.c\nint new;")
        source = evidence(
            [
                file_evidence("old.c", "DELETED", "@@ -1 +0,0 @@\n-int old;", 0, 1),
                file_evidence("new.c", "ADDED", "@@ -0,0 +1 @@\n+int new;", 1, 0),
            ]
        )
        result = verify_commit_evidence(sample, source)
        self.assertEqual(result["coverage"]["path"]["ratio"], 1.0)
        self.assertNotIn("ADDED_DELETED_PATH_CONFLATION", result["reason_codes"])
        self.assertNotIn("FILE_STATUS_MISMATCH", result["reason_codes"])

    def test_extra_represented_path_is_reported(self):
        sample = row(
            "// File: a.c\nold\n// File: extra.c\ncontext",
            "// File: a.c\nnew\n// File: extra.c\ncontext",
        )
        source = evidence(
            [file_evidence("a.c", "MODIFIED", "@@ -1 +1 @@\n-old\n+new", 1, 1)]
        )
        result = verify_commit_evidence(sample, source)
        self.assertIn("SOURCE_COMMIT_FILE_EXTRA", result["reason_codes"])
        self.assertEqual(result["extra_represented_paths"], ["extra.c"])

    def test_id_17283_style_add_delete_conflation_is_detected(self):
        sample = row(
            "// File: README\nold doc\n// File: bitops.c\nold asm",
            "// File: bitops.c\nnew c",
            source_id="17283",
        )
        source = evidence(
            [
                file_evidence("README", "MODIFIED", "@@ -1 +0,0 @@\n-old doc", 0, 1),
                file_evidence("bitops.S", "DELETED", "@@ -1 +0,0 @@\n-old asm", 0, 1),
                file_evidence("bitops.c", "ADDED", "@@ -0,0 +1 @@\n+new c", 1, 0),
            ]
        )
        result = verify_commit_evidence(sample, source)
        self.assertIn("SOURCE_COMMIT_FILE_MISSING", result["reason_codes"])
        self.assertIn("FILE_STATUS_MISMATCH", result["reason_codes"])
        self.assertIn("ADDED_DELETED_PATH_CONFLATION", result["reason_codes"])
        self.assertEqual(result["missing_commit_paths"], ["bitops.S"])

    def test_false_empty_diff_is_detected(self):
        sample = row("// File: a.c\nint same;", "// File: a.c\nint same;")
        source = evidence(
            [file_evidence("a.c", "MODIFIED", "@@ -1 +1 @@\n-int old;\n+int new;", 1, 1)]
        )
        result = verify_commit_evidence(sample, source)
        self.assertIn(
            "RECONSTRUCTED_DIFF_EMPTY_BUT_SOURCE_DIFF_NONEMPTY", result["reason_codes"]
        )
        self.assertIn("RECONSTRUCTED_CHANGE_COUNT_MISMATCH", result["reason_codes"])

    def test_path_count_can_match_while_hunk_coverage_is_incomplete(self):
        sample = row("// File: a.c\nold1", "// File: a.c\nnew1")
        source = evidence(
            [
                file_evidence(
                    "a.c",
                    "MODIFIED",
                    "@@ -1 +1 @@\n-old1\n+new1\n@@ -10 +10 @@\n-old2\n+new2",
                    2,
                    2,
                )
            ]
        )
        result = verify_commit_evidence(sample, source)
        self.assertEqual(result["coverage"]["path"]["ratio"], 1.0)
        self.assertEqual(result["coverage"]["hunks"]["ratio"], 0.5)
        self.assertIn("RECONSTRUCTED_CHANGE_COUNT_MISMATCH", result["reason_codes"])

    def test_missing_commit_is_unverifiable_not_invented(self):
        sample = row("// File: a.c\nold", "// File: a.c\nnew")
        result = verify_commit_evidence(sample, evidence([], status="UNVERIFIABLE"))
        self.assertEqual(result["verification_status"], "UNVERIFIABLE")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["coverage"]["path"]["status"], "UNVERIFIED")

    def test_wrong_commit_evidence_fails_closed(self):
        sample = row("// File: a.c\nold", "// File: a.c\nnew")
        result = verify_commit_evidence(sample, evidence([], commit="c" * 40))
        self.assertEqual(result["verification_status"], "FAILED")
        self.assertIn("does not match", result["failure_reason"])

    def test_derived_representation_preserves_provenance_and_never_fills_code(self):
        sample = row("// File: a.c\nold", "// File: a.c\nnew")
        verification = verify_commit_evidence(
            sample,
            evidence(
                [
                    file_evidence("a.c", "MODIFIED", "@@ -1 +1 @@\n-old\n+new", 1, 1),
                    file_evidence("missing.h", "MODIFIED", "@@ -1 +1 @@\n-x\n+y", 1, 1),
                ]
            ),
        )
        derived = build_derived_representation(
            sample,
            {"status": "UNRESOLVED", "reason_codes": ["FILE_COUNT_MISMATCH"]},
            verification,
            source_file="vulnerabilities.csv",
            source_file_sha256="d" * 64,
        )
        self.assertFalse(derived["source"]["source_data_modified"])
        self.assertFalse(derived["recovery"]["performed"])
        self.assertEqual(derived["recovery"]["candidates"][0]["recovered_code"], None)
        self.assertEqual(
            derived["reconstructed_diff"]["provenance"], "DERIVED_NOT_ORIGINAL"
        )
        schema_path = (
            Path(__file__).parents[1]
            / "dataset_validation"
            / "schemas"
            / "derived_representation.schema.json"
        )
        Draft202012Validator(
            json.loads(schema_path.read_text(encoding="utf-8"))
        ).validate(derived)


if __name__ == "__main__":
    unittest.main()
