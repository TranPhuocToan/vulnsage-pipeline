import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dataset_validation.pilot_package import (
    PilotPackageError,
    prepare_multi_sample_pilot_package,
)
from dataset_validation.qualification import qualify_sample
from pipeline.context_policy import BEFORE_PLUS_DIFF, ContextPolicy
from pipeline.llm_client import StructuredLLMClient
from pipeline.ollama_transport import OllamaConfig
from pipeline.pilot_runner import (
    PilotRunnerError,
    load_official_package,
    run_approved_pilot,
    validate_official_package_inputs,
)


WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE = (
    WORKSPACE / "docs" / "audits" / "artifacts" / "phase3d"
    / "targeted_three_candidate_qualification_20261109" / "canonical_candidates.json"
)
SCHEMA = (
    Path(__file__).resolve().parents[1] / "dataset_validation" / "schemas"
    / "normalized_sample.schema.json"
)
GIT_COMMIT = "a" * 40
SOURCE_IDS = ("16944", "18579")


def fixture_decision(sample):
    qualification = qualify_sample(sample)
    return {
        "decision_id": f"fixture-review-{sample['source']['source_row_id']}",
        "sample_id": sample["sample_id"],
        "decision": "APPROVE",
        "reviewer_id": "fixture-reviewer",
        "reviewer_name": "Fixture Reviewer",
        "reason": "Test-only attributable decision; not a research approval.",
        "decided_at_utc": "2026-11-11T00:00:00Z",
        "evidence_refs": ["fixture://canonical-evidence"],
        "waivers": [
            {
                "reason_code": code,
                "scope": "DERIVED_REPRESENTATION_ONLY",
                "reason": "Test-only derived representation waiver.",
                "evidence_refs": ["fixture://canonical-evidence"],
            }
            for code in qualification["derived_representation_gaps"]
        ],
    }


def build_package(root):
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {str(item["source"]["source_row_id"]): item for item in source["samples"]}
    decisions = {}
    for source_id in SOURCE_IDS:
        path = root / f"decision-{source_id}.json"
        path.write_text(json.dumps(fixture_decision(by_id[source_id])), encoding="utf-8")
        decisions[source_id] = path
    output = root / "package"
    result = prepare_multi_sample_pilot_package(
        source_dataset_path=SOURCE,
        ordered_source_row_ids=SOURCE_IDS,
        review_decision_paths=decisions,
        schema_path=SCHEMA,
        output_dir=output,
        git_commit=GIT_COMMIT,
    )
    return output, result


class MultiSamplePilotPackageTests(unittest.TestCase):
    def test_documented_cli_loads_repository_modules(self):
        completed = subprocess.run(
            [sys.executable, "-B", "scripts/prepare_pilot_package.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("SOURCE_ROW_ID=PATH", completed.stdout)

    def test_builder_and_loader_preserve_pending_ground_truth(self):
        before = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            package, result = build_package(Path(directory))
            loaded = load_official_package(package, SCHEMA, expected_git_commit=GIT_COMMIT)
            self.assertEqual(loaded["package_format"], "MULTI_SAMPLE_REVIEW_RECORDS_V1")
            self.assertEqual(len(loaded["qualifications"]), 2)
            self.assertEqual(len(loaded["decisions"]), 2)
            dataset = json.loads((package / "canonical_dataset.json").read_text(encoding="utf-8"))
            self.assertTrue(all(
                sample["ground_truth"]["review_status"] == "PENDING"
                for sample in dataset["samples"]
            ))
            self.assertTrue(all(
                sample["ground_truth"]["label"] == "PATCHED"
                for sample in dataset["samples"]
            ))
            self.assertEqual(result["ground_truth_review_status_unchanged"], "PENDING")
            with self.assertRaisesRegex(PilotPackageError, "OUTPUT_ALREADY_EXISTS"):
                prepare_multi_sample_pilot_package(
                    source_dataset_path=SOURCE,
                    ordered_source_row_ids=SOURCE_IDS,
                    review_decision_paths={source_id: Path(directory) / f"decision-{source_id}.json" for source_id in SOURCE_IDS},
                    schema_path=SCHEMA,
                    output_dir=package,
                    git_commit=GIT_COMMIT,
                )
        self.assertEqual(SOURCE.read_bytes(), before)

    def test_tamper_and_duplicate_family_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, _ = build_package(root)
            decisions = package / "review_decisions.json"
            payload = json.loads(decisions.read_text(encoding="utf-8"))
            payload["decisions"][0]["reviewer_id"] = "tampered"
            decisions.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PilotRunnerError):
                load_official_package(package, SCHEMA, expected_git_commit=GIT_COMMIT)

        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        duplicate = json.loads(json.dumps(source["samples"][0]))
        duplicate["source"]["source_row_id"] = "duplicate-family"
        duplicate["sample_id"] = "vs:fixture:duplicate-family:canonical"
        source["samples"].append(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            decisions = {}
            for sample in (source["samples"][0], duplicate):
                source_id = str(sample["source"]["source_row_id"])
                path = root / f"{source_id}.json"
                path.write_text(json.dumps(fixture_decision(sample)), encoding="utf-8")
                decisions[source_id] = path
            with self.assertRaisesRegex(PilotPackageError, "DUPLICATE_CVE_FAMILY"):
                prepare_multi_sample_pilot_package(
                    source_dataset_path=source_path,
                    ordered_source_row_ids=[str(source["samples"][0]["source"]["source_row_id"]), "duplicate-family"],
                    review_decision_paths=decisions,
                    schema_path=SCHEMA,
                    output_dir=root / "output",
                    git_commit=GIT_COMMIT,
                )

    def test_non_approve_and_missing_waiver_fail_closed(self):
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        by_id = {str(item["source"]["source_row_id"]): item for item in source["samples"]}
        for case, mutate, expected in (
            ("defer", lambda value: value.update(decision="DEFER"), "APPROVE_DECISION_REQUIRED"),
            ("missing-waiver", lambda value: value.update(waivers=[]), "QUALIFICATION_NOT_READY"),
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                decisions = {}
                for source_id in SOURCE_IDS:
                    decision = fixture_decision(by_id[source_id])
                    if source_id == "18579":
                        mutate(decision)
                    path = root / f"decision-{source_id}.json"
                    path.write_text(json.dumps(decision), encoding="utf-8")
                    decisions[source_id] = path
                with self.assertRaisesRegex(PilotPackageError, expected):
                    prepare_multi_sample_pilot_package(
                        source_dataset_path=SOURCE,
                        ordered_source_row_ids=SOURCE_IDS,
                        review_decision_paths=decisions,
                        schema_path=SCHEMA,
                        output_dir=root / "package",
                        git_commit=GIT_COMMIT,
                    )

    def test_static_preflight_and_mock_resume_use_same_ordered_list(self):
        class PipelineMock:
            def __call__(self, prompt):
                if '"dependency_analysis"' in prompt and '"num_cross_file_dependencies"' in prompt:
                    value = {"dependency_analysis": "fixture", "dependencies": [], "num_cross_file_dependencies": 0, "overall_coupling": "LOOSE"}
                elif '"function_summary"' in prompt and '"local_verdict"' in prompt:
                    value = {"file_name": "fixture.c", "function_summary": "fixture", "changes_summary": "fixture", "security_purpose": "fixture", "local_verdict": "CONTRIBUTES_TO_FIX", "confidence": "HIGH", "key_observations": ["fixture"]}
                else:
                    value = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "fixture"}
                return json.dumps(value), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, build_result = build_package(root)
            policy = ContextPolicy(code_representation=BEFORE_PLUS_DIFF)
            preflight = validate_official_package_inputs(
                package_dir=package,
                schema_path=SCHEMA,
                expected_git_commit=GIT_COMMIT,
                context_policy=policy,
                model_context_tokens=32768,
                reserved_output_tokens=2048,
                exact_token_counter=lambda text: len(text.split()),
            )
            self.assertTrue(preflight["exact_token_preflight_status"].startswith("PASS_"))
            self.assertEqual(preflight["ordered_sample_ids"], build_result["ordered_sample_ids"])
            parameters = OllamaConfig().run_parameters()
            kwargs = dict(
                package_dir=package,
                schema_path=SCHEMA,
                output_root=root / "output",
                run_prefix="multi-fixture",
                git_commit=GIT_COMMIT,
                model_id="qwen2.5-coder:7b",
                model_digest="sha256:model-fixture",
                model_quantization="Q4_K_M",
                tokenizer_metadata={"id": "fixture", "library": "fixture", "version": "1", "class": "WhitespaceTokenizer"},
                parameters=parameters,
                retry_policy={"max_attempts": 1, "retry_delays_sec": [0.0]},
                client_factory=lambda pipeline: StructuredLLMClient(PipelineMock(), max_attempts=1, retry_delays=[0.0]),
                context_policy=policy,
                model_context_tokens=32768,
                reserved_output_tokens=2048,
                exact_token_counter=lambda text: len(text.split()),
                allow_candidate_context_policy=True,
            )
            first = run_approved_pilot(**kwargs)
            second = run_approved_pilot(**kwargs)
            self.assertTrue(first["complete"])
            self.assertEqual(first, second)
            self.assertEqual(first["ordered_sample_ids"], build_result["ordered_sample_ids"])
            self.assertEqual(first["result_count_by_pipeline"], {"P0": 2, "P1": 2, "P2": 2})


if __name__ == "__main__":
    unittest.main()
