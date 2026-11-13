import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.canonical_adapter import (
    CanonicalAdapterError,
    adapt_sample,
    load_and_adapt_dataset,
    safe_sample_key,
)
from pipeline.context_policy import ContextPolicy, ContextPolicyError, assert_no_ground_truth_leakage
from pipeline.evaluator import (
    PositiveOnlyEvaluationError,
    evaluate_positive_only,
    evaluate_positive_only_ablation,
)
from pipeline.llm_client import StructuredLLMClient
from pipeline.pipeline_p0 import process_canonical_sample_p0
from pipeline.pipeline_p1 import process_canonical_sample_p1
from pipeline.pipeline_p2 import process_canonical_sample_p2
from pipeline.run_state import RunStateError, RunWorkspace, build_run_spec


WORKSPACE = Path(__file__).resolve().parents[3]
PROTOTYPE = WORKSPACE / "docs" / "audits" / "artifacts" / "phase2e" / "prototype_normalized_dataset.json"


def prototype_sample(source_id="17283"):
    payload = json.loads(PROTOTYPE.read_text(encoding="utf-8"))
    return next(item for item in payload["samples"] if str(item["source"]["source_row_id"]) == source_id)


class RecordingTransport:
    def __init__(self, verdict="PATCHED", usage=True):
        self.prompts = []
        self.verdict = verdict
        self.usage = usage

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if '"evidence_integration"' in prompt or '"global_synthesis"' in prompt:
            payload = {"verdict": self.verdict, "confidence": "HIGH", "reasoning": "offline fixture"}
        elif '"dependency_analysis"' in prompt and '"num_cross_file_dependencies"' in prompt:
            payload = {
                "dependency_analysis": "offline fixture",
                "dependencies": [],
                "num_cross_file_dependencies": 0,
                "overall_coupling": "LOOSE",
            }
        elif '"function_summary"' in prompt and '"local_verdict"' in prompt:
            payload = {
                "file_name": "fixture.c", "function_summary": "fixture",
                "changes_summary": "fixture", "security_purpose": "fixture",
                "local_verdict": "CONTRIBUTES_TO_FIX", "confidence": "HIGH",
                "key_observations": ["offline fixture"],
            }
        else:
            payload = {"verdict": self.verdict, "confidence": "HIGH", "reasoning": "offline fixture"}
        tokens = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12} if self.usage else None
        return json.dumps(payload), tokens


class CanonicalAdapterTests(unittest.TestCase):
    def test_real_prototype_preserves_add_delete_rename_and_pending(self):
        adapted = adapt_sample(prototype_sample())
        self.assertEqual([item["change_type"] for item in adapted["model_input"]["files"]], ["MODIFY", "DELETE", "ADD"])
        self.assertEqual(adapted["integration_status"], "NEEDS_APPROVAL")
        self.assertIn("GROUND_TRUTH_PENDING", adapted["reason_codes"])
        self.assertEqual(adapted["model_input"]["files"][1]["after_provenance"]["status"], "MISSING_BY_CHANGE_TYPE")
        self.assertEqual(adapted["model_input"]["files"][2]["before_provenance"]["status"], "MISSING_BY_CHANGE_TYPE")
        self.assertNotIn(":", adapted["safe_sample_key"])

        renamed = prototype_sample()
        renamed["files"][0]["change_type"] = "RENAME"
        renamed["files"][0]["path_before"] = "old-" + renamed["files"][0]["path_before"]
        self.assertEqual(adapt_sample(renamed)["model_input"]["files"][0]["change_type"], "RENAME")

    def test_hash_mismatch_fails_closed(self):
        sample = prototype_sample()
        sample["files"][0]["original_git_diff_sha256"] = "0" * 64
        with self.assertRaisesRegex(CanonicalAdapterError, "sha256 mismatch"):
            adapt_sample(sample)

    def test_missing_required_data_fails_schema_validation(self):
        sample = prototype_sample()
        del sample["identity"]
        with self.assertRaisesRegex(CanonicalAdapterError, "schema validation failed"):
            adapt_sample(sample)

    def test_duplicate_ids_and_windows_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            sample = prototype_sample()
            path.write_text(json.dumps({"samples": [sample, sample]}), encoding="utf-8")
            with self.assertRaisesRegex(CanonicalAdapterError, "duplicate sample_id"):
                load_and_adapt_dataset(path)
        self.assertTrue(safe_sample_key("CON").startswith("sample_CON-"))

    def test_loading_is_read_only(self):
        before = PROTOTYPE.read_bytes()
        load_and_adapt_dataset(PROTOTYPE)
        self.assertEqual(PROTOTYPE.read_bytes(), before)


class ContextAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.adapted = adapt_sample(prototype_sample())

    def test_leakage_guard_rejects_ground_truth_key(self):
        with self.assertRaisesRegex(ContextPolicyError, "GROUND_TRUTH_LEAKAGE"):
            assert_no_ground_truth_leakage({"ground_truth": "PATCHED"})

    def test_mock_p0_p1_p2_use_same_sample_without_label_leakage(self):
        results = {}
        for pipeline, function in {
            "P0": process_canonical_sample_p0,
            "P1": process_canonical_sample_p1,
            "P2": process_canonical_sample_p2,
        }.items():
            transport = RecordingTransport()
            client = StructuredLLMClient(transport, max_attempts=1)
            result = function(self.adapted, client)
            self.assertEqual(result["execution_status"], "COMPLETED")
            self.assertEqual(result["prediction"], "PATCHED")
            self.assertEqual(result["sample_id"], self.adapted["sample_id"])
            all_prompts = "\n".join(transport.prompts)
            self.assertNotIn("VULNSAGE_FIX_COMMIT", all_prompts)
            self.assertNotIn("review_status", all_prompts)
            self.assertNotIn("Subject: [PATCH]", all_prompts)
            results[pipeline] = [result]
        report = evaluate_positive_only_ablation(results, [self.adapted["sample_id"]])
        self.assertEqual(set(report), {"P0", "P1", "P2"})

    def test_context_overflow_is_a_technical_error(self):
        transport = RecordingTransport()
        client = StructuredLLMClient(transport, max_attempts=1)
        result = process_canonical_sample_p0(
            self.adapted, client, ContextPolicy(max_utf8_bytes_per_call=8)
        )
        self.assertEqual(result["execution_status"], "TECHNICAL_ERROR")
        self.assertIn("CONTEXT_OVERFLOW", result["technical_error"])
        self.assertEqual(transport.prompts, [])


class StrictClientTests(unittest.TestCase):
    def test_not_sure_is_a_model_outcome_and_missing_usage_stays_unknown(self):
        client = StructuredLLMClient(RecordingTransport(verdict="NOT_SURE", usage=False), max_attempts=1)
        result = client.invoke("prompt", "verdict")
        self.assertEqual(result.status, "OK")
        self.assertTrue(result.model_abstained)
        self.assertIsNone(result.attempts[0]["total_tokens"])

    def test_invalid_json_and_schema_are_technical_errors_not_not_sure(self):
        responses = iter([("not json", None), (json.dumps({"verdict": 1}), None)])
        client = StructuredLLMClient(lambda prompt: next(responses), max_attempts=2)
        result = client.invoke("prompt", "verdict")
        self.assertEqual(result.status, "TECHNICAL_ERROR")
        self.assertFalse(result.model_abstained)
        self.assertEqual(len(result.attempts), 2)

    def test_retry_records_failed_and_successful_attempts(self):
        responses = iter([
            ("bad", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            (json.dumps({"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "ok"}), {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}),
        ])
        result = StructuredLLMClient(lambda prompt: next(responses), max_attempts=2).invoke("x", "verdict")
        self.assertEqual(result.status, "OK")
        self.assertEqual([item["status"] for item in result.attempts], ["ERROR", "OK"])


class RunStateTests(unittest.TestCase):
    def make_spec(self, run_id="run-001", dataset_hash="a" * 64):
        return build_run_spec(
            run_id=run_id,
            dataset_sha256=dataset_hash,
            sample_ids=["sample:1"],
            pipeline_id="P0",
            pipeline_version="phase3b",
            model_id="mock",
            model_version="1",
            parameters={"temperature": 0.0},
            retry_policy={"max_attempts": 1},
            context_policy_sha256="b" * 64,
            prompt_hashes={"p0": "c" * 64},
            code_sha256="d" * 64,
            git_commit=None,
        )

    def test_resume_accepts_identical_spec_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = RunWorkspace(root, self.make_spec())
            first.initialize()
            RunWorkspace(root, self.make_spec()).initialize()
            changed = RunWorkspace(root, self.make_spec(dataset_hash="e" * 64))
            with self.assertRaisesRegex(RunStateError, "RESUME_SPEC_MISMATCH"):
                changed.initialize()

    def test_results_are_scoped_validated_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = RunWorkspace(Path(directory), self.make_spec())
            workspace.initialize()
            result = {"sample_id": "sample:1", "pipeline": "P0", "prediction": "PATCHED"}
            path = workspace.write_result(result)
            self.assertNotIn(":", path.name)
            with self.assertRaisesRegex(RunStateError, "refusing to overwrite"):
                workspace.write_result(result)
            self.assertEqual(workspace.load_results()[0]["prediction"], "PATCHED")

    def test_duplicate_sample_ids_are_rejected(self):
        kwargs = dict(
            run_id="x", dataset_sha256="a" * 64, sample_ids=["x", "x"], pipeline_id="P0",
            pipeline_version="1", model_id="m", model_version="1", parameters={}, retry_policy={},
            context_policy_sha256="b" * 64, prompt_hashes={}, code_sha256="c" * 64, git_commit=None,
        )
        with self.assertRaisesRegex(RunStateError, "duplicate sample_id"):
            build_run_spec(**kwargs)

    def test_corrupt_output_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = RunWorkspace(Path(directory), self.make_spec())
            workspace.initialize()
            result_dir = workspace.run_dir / "results"
            result_dir.mkdir()
            (result_dir / "broken.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RunStateError, "corrupt result"):
                workspace.load_results()


class PositiveOnlyEvaluatorTests(unittest.TestCase):
    def test_edge_cases_and_denominators(self):
        ids = ["c", "r", "u", "e", "missing"]
        results = [
            {"sample_id": "c", "execution_status": "COMPLETED", "prediction": "PATCHED"},
            {"sample_id": "r", "execution_status": "COMPLETED", "prediction": "NOT_PATCHED"},
            {"sample_id": "u", "execution_status": "COMPLETED", "prediction": "NOT_SURE"},
            {"sample_id": "e", "execution_status": "TECHNICAL_ERROR", "prediction": None},
        ]
        report = evaluate_positive_only(results, ids)
        self.assertEqual((report["C"], report["R"], report["U"], report["E"]), (1, 1, 1, 2))
        self.assertEqual(report["positive_acceptance_rate"], 0.2)
        self.assertEqual(report["coverage"], 0.4)
        self.assertEqual(report["selective_accuracy"], 0.5)
        self.assertFalse(report["binary_accuracy_f1_reported"])

    def test_zero_denominator_and_pipeline_mismatch(self):
        report = evaluate_positive_only([], [])
        self.assertIsNone(report["coverage"])
        self.assertIsNone(report["selective_accuracy"])
        with self.assertRaisesRegex(PositiveOnlyEvaluationError, "sample list mismatch"):
            evaluate_positive_only_ablation({"P0": [], "P1": [], "P2": []}, ["x"])

    def test_telemetry_is_aggregated_without_coercing_missing_usage_to_zero(self):
        complete = [
            {
                "sample_id": "a", "execution_status": "COMPLETED", "prediction": "PATCHED",
                "attempts": [{"attempt": 1}, {"attempt": 2}],
                "token_usage": {"usage_complete": True, "input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "total_attempt_duration_sec": 1.25,
            },
            {
                "sample_id": "b", "execution_status": "COMPLETED", "prediction": "NOT_SURE",
                "attempts": [{"attempt": 1}],
                "token_usage": {"usage_complete": True, "input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
                "total_attempt_duration_sec": 2.5,
            },
        ]
        telemetry = evaluate_positive_only(complete, ["a", "b"])["telemetry"]
        self.assertEqual(telemetry["actual_llm_calls"], 3)
        self.assertEqual(telemetry["retry_count"], 1)
        self.assertEqual(telemetry["input_tokens"], 30)
        self.assertEqual(telemetry["total_tokens"], 35)
        self.assertEqual(telemetry["total_attempt_duration_sec"], 3.75)

        incomplete = json.loads(json.dumps(complete))
        incomplete[1]["token_usage"] = {
            "usage_complete": False, "input_tokens": None,
            "output_tokens": None, "total_tokens": None,
        }
        telemetry = evaluate_positive_only(incomplete, ["a", "b"])["telemetry"]
        self.assertFalse(telemetry["token_usage_complete"])
        self.assertIsNone(telemetry["input_tokens"])
        self.assertIsNone(telemetry["total_tokens"])
        self.assertFalse(telemetry["missing_usage_is_zero"])


if __name__ == "__main__":
    unittest.main()
