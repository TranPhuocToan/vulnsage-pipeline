import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_critical_path_preflight import (
    CriticalPathPreflightError,
    MODEL_CONTRACT,
    build_summary_artifact,
    ensure_contract,
    validate_main_gate,
)


class CriticalPathCommandGuardsTests(unittest.TestCase):
    def test_main_is_locked_without_attributable_pilot_pass(self):
        with self.assertRaisesRegex(CriticalPathPreflightError, "PILOT_PASS_GATE_REQUIRED"):
            validate_main_gate(None)

    def test_main_gate_requires_hash_and_positive_only_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps({
                "status": "PASS",
                "evaluation_scope": "POSITIVE_ONLY_KNOWN_FIX_COMMIT_POST_STATES",
                "model_id": MODEL_CONTRACT["model_id"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(CriticalPathPreflightError, "HASH_PROVENANCE_REQUIRED"):
                validate_main_gate(path)

    def test_contract_resume_is_exact_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract = {"purpose": "OFFLINE_PREFLIGHT_ONLY", "hash": "a" * 64}
            ensure_contract(path, contract)
            ensure_contract(path, contract)
            with self.assertRaisesRegex(CriticalPathPreflightError, "CONTRACT_MISMATCH"):
                ensure_contract(path, {**contract, "hash": "b" * 64})

    def test_offline_summary_has_evaluation_provenance_and_zero_real_calls(self):
        summary = {
            "complete": True,
            "ordered_sample_ids": ["sample-1"],
            "sample_list_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "run_spec_sha256_by_pipeline": {"P0": "c" * 64},
            "result_count_by_pipeline": {"P0": 1},
            "evaluation": {"P0": {"N": 1}},
        }
        artifact = build_summary_artifact(
            summary, {"contract_sha256": "d" * 64}
        )
        self.assertEqual(artifact["experimental_llm_calls"], 0)
        self.assertFalse(artifact["official_pilot"])
        self.assertFalse(
            artifact["evaluation_provenance"]["research_metric_claim_authorized"]
        )
        self.assertEqual(
            artifact["evaluation_provenance"]["ground_truth_review"],
            "PENDING_NOT_PROMOTED",
        )


if __name__ == "__main__":
    unittest.main()
