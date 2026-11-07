import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from pipeline.canonical_adapter import load_and_adapt_dataset
from pipeline.context_diagnostics import candidate_policy, compare_context_representations
from pipeline.context_policy import ContextPolicy, ContextPolicyError, prepare_context, render_file_context
from pipeline.evaluator import evaluate_positive_only
from pipeline.llm_client import StructuredLLMClient
from pipeline.lossless_context import LosslessContextError, original_modify_section, replay_original_modify_diff
from pipeline.offline_integration import render_static_prompt_plan, run_canonical_sample
from pipeline.ollama_transport import OllamaConfig
from pipeline.pilot_runner import PilotRunnerError, run_approved_pilot
from pipeline.run_state import RunStateError
from tests.test_phase3c3_kaggle_integration import SCHEMA, build_smoke_package


PROTOTYPE = Path(__file__).resolve().parents[3] / "docs/audits/artifacts/phase2e/prototype_normalized_dataset.json"


def mock_payload(prompt):
    if '"function_summary"' in prompt:
        value = {
            "file_name": "fixture.c", "function_summary": "fixture", "changes_summary": "fixture",
            "security_purpose": "fixture", "local_verdict": "CONTRIBUTES_TO_FIX",
            "confidence": "HIGH", "key_observations": ["fixture"],
        }
    elif '"num_cross_file_dependencies"' in prompt:
        value = {
            "dependency_analysis": "fixture", "dependencies": [],
            "num_cross_file_dependencies": 0, "overall_coupling": "LOOSE",
        }
    else:
        value = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "fixture"}
    return json.dumps(value), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class LosslessContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapted = next(
            sample for sample in load_and_adapt_dataset(PROTOTYPE)
            if sample["audit_only"]["source"]["source_row_id"] == "19047"
        )

    def test_real_three_file_round_trip_deterministic_read_only_and_smaller(self):
        source_hash = hashlib.sha256(PROTOTYPE.read_bytes()).hexdigest()
        original = copy.deepcopy(self.adapted)
        policy = candidate_policy()
        first = prepare_context(self.adapted, policy)
        self.assertEqual(first, prepare_context(self.adapted, policy))
        self.assertEqual(self.adapted, original)
        self.assertEqual(hashlib.sha256(PROTOTYPE.read_bytes()).hexdigest(), source_hash)
        self.assertEqual(self.adapted["evaluation_only"]["ground_truth"]["review_status"], "PENDING")
        self.assertEqual(len(first["files"]), 3)
        full = self.adapted["audit_only"]["full_patch"]["original_git_diff"]
        for raw, rendered in zip(self.adapted["model_input"]["files"], first["files"]):
            section = original_modify_section(full, raw["display_path"])
            self.assertEqual(replay_original_modify_diff(raw["before_code"], section, raw["display_path"]), raw["after_code"])
            self.assertEqual(rendered["before_code"], raw["before_code"])
            self.assertEqual(rendered["diff"], raw["original_diff"])
            self.assertEqual(rendered["representation_provenance"]["round_trip"], "EXACT_AFTER_MATCH")
        old = render_static_prompt_plan("p0", self.adapted)["static_prompts"][0]
        new = render_static_prompt_plan("p0", self.adapted, policy)["static_prompts"][0]
        self.assertLess(new["utf8_bytes"], old["utf8_bytes"] * 0.6)
        self.assertNotEqual(old["prompt_sha256"], new["prompt_sha256"])

    def test_historical_policy_hash_preserved_and_candidate_hash_distinct(self):
        values = asdict(ContextPolicy())
        del values["code_representation"]
        expected = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(ContextPolicy().sha256(), expected)
        self.assertNotEqual(ContextPolicy().sha256(), candidate_policy().sha256())
        with self.assertRaises(ContextPolicyError):
            ContextPolicy(code_representation="silently_truncate")

    def test_add_delete_rename_preserve_full_blob_and_missing_side_semantics(self):
        for operation in ("ADD", "DELETE", "RENAME"):
            raw = copy.deepcopy(self.adapted["model_input"]["files"][0])
            raw["change_type"] = operation
            if operation == "ADD":
                raw.update(path_before=None, before_code=None, before_code_sha256=None)
                raw["before_provenance"]["status"] = "MISSING_BY_CHANGE_TYPE"
            elif operation == "DELETE":
                raw.update(path_after=None, after_code=None, after_code_sha256=None)
                raw["after_provenance"]["status"] = "MISSING_BY_CHANGE_TYPE"
            else:
                raw["path_after"] = "renamed.h"
            rendered = render_file_context(raw, candidate_policy())
            self.assertEqual(rendered["representation_provenance"]["round_trip"], "NOT_APPLICABLE_FULL_BLOBS_FALLBACK")
            self.assertEqual(rendered["diff"], raw["original_diff"])
            self.assertEqual(rendered["before_code"], raw["before_code"] if raw["before_code"] is not None else "<CODE_BEFORE_MISSING_BY_CHANGE_TYPE>")
            self.assertEqual(rendered["after_code"], raw["after_code"] if raw["after_code"] is not None else "<CODE_AFTER_MISSING_BY_CHANGE_TYPE>")

    def test_mismatch_does_not_fall_back_or_invent_after(self):
        raw = copy.deepcopy(self.adapted)
        raw["model_input"]["files"][0]["after_code"] += "fake\n"
        with self.assertRaisesRegex(ContextPolicyError, "TEXT_HASH_MISMATCH"):
            prepare_context(raw, candidate_policy())
        raw["model_input"]["files"][0]["after_code_sha256"] = hashlib.sha256(
            raw["model_input"]["files"][0]["after_code"].encode()
        ).hexdigest()
        with self.assertRaisesRegex(ContextPolicyError, "ROUND_TRIP_MISMATCH"):
            prepare_context(raw, candidate_policy())

    def test_strict_patch_eof_context_offsets_and_insert_delete_hunks(self):
        patch = "diff --git a/a.c b/a.c\n--- a/a.c\n+++ b/a.c\n@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
        self.assertEqual(replay_original_modify_diff("old", patch, "a.c"), "new")
        with self.assertRaisesRegex(LosslessContextError, "CONTEXT_MISMATCH"):
            replay_original_modify_diff("wrong", patch, "a.c")
        with self.assertRaisesRegex(LosslessContextError, "PATH_MISMATCH"):
            replay_original_modify_diff("old", patch, "other.c")
        add = "diff --git a/a.c b/a.c\n--- a/a.c\n+++ b/a.c\n@@ -0,0 +1 @@\n+x\n"
        delete = "diff --git a/a.c b/a.c\n--- a/a.c\n+++ b/a.c\n@@ -1 +0,0 @@\n-x\n"
        self.assertEqual(replay_original_modify_diff("", add, "a.c"), "x\n")
        self.assertEqual(replay_original_modify_diff("x\n", delete, "a.c"), "")
        with self.assertRaises(LosslessContextError):
            replay_original_modify_diff("x\n", delete.replace("@@ -1 +0,0", "@@ -99 +0,0"), "a.c")

    def test_same_candidate_input_and_diff_for_all_methods(self):
        p0, p1, p2 = [render_static_prompt_plan(method, self.adapted, candidate_policy()) for method in ("p0", "p1", "p2")]
        self.assertEqual(p1["representation_provenance"], p0["representation_provenance"])
        self.assertEqual(p2["representation_provenance"], p0["representation_provenance"])
        self.assertEqual(
            [call["prompt"] for call in p1["static_prompts"]],
            [call["prompt"] for call in p2["static_prompts"][:3]],
        )
        old_dep = render_static_prompt_plan("p2", self.adapted)["static_prompts"][-1]
        self.assertEqual(old_dep, p2["static_prompts"][-1])

    def test_exact_boundary_and_observed_overflow_stop_before_transport(self):
        for count, expected, calls_expected in (
            (30_720, "COMPLETED", 1), (30_721, "TECHNICAL_ERROR", 0),
            (32_042, "TECHNICAL_ERROR", 0),
        ):
            with self.subTest(count=count):
                calls = []
                def transport(prompt):
                    calls.append(prompt)
                    return mock_payload(prompt)
                result = run_canonical_sample(
                    "p0", self.adapted, StructuredLLMClient(transport, max_attempts=1),
                    candidate_policy(), token_counter=lambda text: count, model_input_token_limit=30_720,
                )
                self.assertEqual(result["execution_status"], expected)
                self.assertEqual(len(calls), calls_expected)
                self.assertEqual(evaluate_positive_only([result], [self.adapted["sample_id"]])["N"], 1)

    def test_runtime_dependent_synthesis_overflow_for_p1_and_p2(self):
        for method, static_calls in (("p1", 3), ("p2", 4)):
            with self.subTest(method=method):
                counts = iter([10] * static_calls + [30_721])
                sent = []
                def transport(prompt):
                    sent.append(prompt)
                    return mock_payload(prompt)
                result = run_canonical_sample(
                    method, self.adapted, StructuredLLMClient(transport, max_attempts=1),
                    candidate_policy(), token_counter=lambda text: next(counts),
                    model_input_token_limit=30_720,
                )
                self.assertEqual(result["execution_status"], "TECHNICAL_ERROR")
                self.assertIn(f"tokens:{method}.synthesis", result["technical_error"])
                self.assertEqual(len(sent), static_calls)

    def test_diagnostic_counts_remain_unknown_without_actual_tokenizer(self):
        report = compare_context_representations(self.adapted)
        self.assertEqual(report["exact_token_preflight"], "NOT_RUN")
        for representation in report["reports"]:
            for plan in representation["plans"]:
                self.assertTrue(all(call["raw_input_tokens"] is None for call in plan["calls"]))

    def test_opt_in_package_pin_and_policy_resume_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = build_smoke_package(root, git_commit="a" * 40)
            hashes_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()}
            kwargs = dict(
                package_dir=package, schema_path=SCHEMA, output_root=root / "output",
                run_prefix="delta-fixture", git_commit="b" * 40, package_git_commit="a" * 40,
                model_id="qwen2.5-coder:7b", model_digest="fixture-digest", model_quantization="fixture-Q4",
                tokenizer_metadata={"id": "fixture", "library": "fixture", "version": "1", "class": "FixtureCounter"},
                parameters=OllamaConfig().run_parameters(), retry_policy={"max_attempts": 1},
                context_policy=candidate_policy(), exact_token_counter=lambda text: 10,
                client_factory=lambda pipeline: StructuredLLMClient(mock_payload, max_attempts=1),
            )
            with self.assertRaisesRegex(PilotRunnerError, "HUMAN_DECISION_CONTEXT_POLICY_REQUIRED"):
                run_approved_pilot(**kwargs)
            self.assertFalse((root / "output").exists())
            kwargs["allow_candidate_context_policy"] = True  # test fixture opt-in only
            first = run_approved_pilot(**kwargs)
            self.assertEqual(first, run_approved_pilot(**kwargs))
            self.assertEqual(first["source_package_git_commit"], "a" * 40)
            self.assertEqual(first["experiment_git_commit"], "b" * 40)
            spec = json.loads((root / "output/runs/delta-fixture-p0/run_spec.json").read_text())
            self.assertEqual(spec["context_policy_configuration"]["code_representation"], "before_plus_original_diff_v1")
            self.assertEqual(spec["source_package_git_commit"], "a" * 40)
            kwargs["context_policy"] = ContextPolicy()
            with self.assertRaisesRegex(RunStateError, "RESUME_SPEC_MISMATCH"):
                run_approved_pilot(**kwargs)
            self.assertEqual(hashes_before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()})


if __name__ == "__main__":
    unittest.main()
