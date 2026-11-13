import json
import tempfile
import unittest
from pathlib import Path

from scripts.canonicalize_pilot_candidates import REPOSITORIES, _write_once
from scripts.report_canonical_pilot import build_reports


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "data" / "pilot" / "canonical_pilot20_20261111_retry3"
SCHEMA = ROOT / "dataset_validation" / "schemas" / "normalized_sample.schema.json"


class CanonicalPilotToolTests(unittest.TestCase):
    def test_write_once_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            _write_once(path, {"status": "PENDING", "experimental_llm_calls": 0})
            original = path.read_bytes()
            _write_once(path, {"status": "PENDING", "experimental_llm_calls": 0})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _write_once(path, {"status": "VERIFIED"})

    def test_real_canonical_prototype_reports_blockers_without_promotion(self):
        self.assertEqual(
            REPOSITORIES,
            {"linux": "torvalds/linux", "mozilla": "mozilla/gecko-dev", "xen": "xen-project/xen"},
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = build_reports(
                FINAL_ROOT / "canonical_dataset.json",
                FINAL_ROOT / "canonicalization_run.json",
                SCHEMA,
                Path(directory) / "reports",
            )
            self.assertEqual(summary["canonicalized_samples"], 20)
            self.assertEqual(summary["ground_truth_pending"], 20)
            self.assertEqual(summary["qualified_ready"], 0)
            self.assertEqual(summary["context_fit_exact"], 0)
            self.assertEqual(summary["experimental_llm_calls"], 0)
            token = json.loads(
                (Path(directory) / "reports" / "exact_token_preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(token["status"], "BLOCKED_EXACT_TOKENIZER_UNAVAILABLE")
            self.assertFalse(token["exact"])
            self.assertFalse(token["heuristic_substituted"])


if __name__ == "__main__":
    unittest.main()
