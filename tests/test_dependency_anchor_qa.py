import hashlib
import unittest

from pipeline.dependency_anchor_qa import (
    EXACT_MATCH,
    UNSUPPORTED,
    WHITESPACE_NORMALIZED_MATCH,
    validate_dependency_anchors,
)


class DependencyAnchorQATests(unittest.TestCase):
    def setUp(self):
        self.diff = (
            "diff --git a/a.c b/a.c\n"
            "--- a/a.c\n+++ b/a.c\n"
            "@@ -1 +1 @@\n"
            "-\told_call();\n"
            "+\tif (req->buf != NULL)\n"
        ).encode()
        self.sha = hashlib.sha256(self.diff).hexdigest()

    def validate(self, anchors):
        return validate_dependency_anchors(
            original_diff_bytes=self.diff,
            expected_diff_sha256=self.sha,
            dependency_evidence={"dependencies": [{
                "source_file": "a.c", "target_file": "b.h", "anchor_lines": anchors,
            }]},
        )

    def test_literal_tab_indentation_is_exact(self):
        result = self.validate(["+\tif (req->buf != NULL)"])
        self.assertEqual(result["anchors"][0]["status"], EXACT_MATCH)

    def test_existing_line_with_whitespace_change_is_not_exact(self):
        result = self.validate(["+ if (req->buf != NULL)"])
        self.assertEqual(result["anchors"][0]["status"], WHITESPACE_NORMALIZED_MATCH)

    def test_hallucinated_line_is_unsupported(self):
        result = self.validate(["+ invented_security_check();"])
        self.assertEqual(result["anchors"][0]["status"], UNSUPPORTED)

    def test_missing_or_wrong_hash_blocks_without_anchor_classification(self):
        missing = validate_dependency_anchors(
            original_diff_bytes=None,
            expected_diff_sha256=self.sha,
            dependency_evidence={"dependencies": []},
        )
        self.assertEqual(missing["validation_status"], "BLOCKED")
        self.assertIn("ORIGINAL_DIFF_MISSING", missing["reason_codes"])
        wrong = validate_dependency_anchors(
            original_diff_bytes=self.diff,
            expected_diff_sha256="0" * 64,
            dependency_evidence={"dependencies": []},
        )
        self.assertEqual(wrong["validation_status"], "BLOCKED")
        self.assertIn("ORIGINAL_DIFF_HASH_MISMATCH", wrong["reason_codes"])
        self.assertEqual(wrong["anchors"], [])


if __name__ == "__main__":
    unittest.main()
