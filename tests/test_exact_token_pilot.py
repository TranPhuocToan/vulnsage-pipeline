import tempfile
import unittest
from pathlib import Path

from scripts.exact_token_pilot import (
    ExactTokenPreflightError,
    TOKENIZER_ASSET_SHA256,
    classify_count,
    verify_tokenizer_assets,
)


class ExactTokenPilotTests(unittest.TestCase):
    def test_observed_overhead_changes_exact_boundary(self):
        self.assertEqual(classify_count(30_691, 30_720, 29), "FIT_WITH_OBSERVED_OVERHEAD")
        self.assertEqual(classify_count(30_692, 30_720, 29), "OVERFLOW")

    def test_tokenizer_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ExactTokenPreflightError, "TOKENIZER_ASSET_INTEGRITY_FAILED"):
                verify_tokenizer_assets(root)
            for name in TOKENIZER_ASSET_SHA256:
                (root / name).write_text("wrong", encoding="utf-8")
            with self.assertRaisesRegex(ExactTokenPreflightError, "TOKENIZER_ASSET_INTEGRITY_FAILED"):
                verify_tokenizer_assets(root)


if __name__ == "__main__":
    unittest.main()
