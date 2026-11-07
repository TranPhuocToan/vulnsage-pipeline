import json
import ast
import tempfile
import unittest
from pathlib import Path

from pipeline.canonical_adapter import dataset_sha256, load_and_adapt_dataset
from pipeline.llm_client import StructuredLLMClient
from pipeline.ollama_transport import OllamaConfig, OllamaGenerateTransport, OllamaTransportError
from pipeline.pilot_runner import (
    PilotRunnerError,
    load_official_manifest,
    run_approved_pilot,
    validate_pilot_inputs,
)
from pipeline.run_state import hash_value


WORKSPACE = Path(__file__).resolve().parents[3]
PROTOTYPE = WORKSPACE / "docs" / "audits" / "artifacts" / "phase2e" / "prototype_normalized_dataset.json"
SCHEMA = Path(__file__).resolve().parents[1] / "dataset_validation" / "schemas" / "normalized_sample.schema.json"
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "phase3d_kaggle_qwen25coder7b_v2.ipynb"


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.posts = []

    def get(self, url, timeout):
        return FakeResponse({
            "models": [{
                "name": "qwen2.5-coder:7b",
                "digest": "sha256:model-fixture",
                "size": 4_700_000_000,
                "modified_at": "2026-01-01T00:00:00Z",
                "details": {"quantization_level": "Q4_K_M"},
            }]
        })

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        if url.endswith("/api/show"):
            return FakeResponse({"model_info": {"general.architecture": "qwen2"}, "parameters": "fixture", "template": "{{ .Prompt }}"})
        return FakeResponse({
            "response": '{"verdict":"PATCHED","confidence":"HIGH","reasoning":"fixture"}',
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 11,
            "eval_count": 5,
            "total_duration": 100,
        })


def write_manifest(path, sample_ids, *, ready=True, official=True):
    manifest = {
        "manifest_version": "fixture-official-pilot-1.0",
        "purpose": "PILOT",
        "official_dataset": official,
        "dataset_sha256": dataset_sha256(PROTOTYPE),
        "ordered_sample_ids": list(sample_ids),
        "sample_list_sha256": hash_value(list(sample_ids)),
        "samples": [
            {"sample_id": sample_id, "qualification_status": "READY" if ready else "NEEDS_APPROVAL"}
            for sample_id in sample_ids
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class OllamaTransportTests(unittest.TestCase):
    def test_probe_and_generate_capture_exact_model_tokens_and_calls(self):
        session = FakeSession()
        transport = OllamaGenerateTransport(OllamaConfig(), session=session, clock=lambda: 1.0)
        metadata = transport.probe()
        self.assertTrue(metadata["available"])
        self.assertEqual(metadata["digest"], "sha256:model-fixture")
        text, usage = transport("prompt")
        self.assertIn('"PATCHED"', text)
        self.assertEqual(usage, {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16})
        generate_request = session.posts[-1][1]
        self.assertEqual(generate_request["model"], "qwen2.5-coder:7b")
        self.assertEqual(generate_request["options"]["temperature"], 0.0)
        self.assertFalse(generate_request["stream"])
        self.assertEqual(len(transport.call_log), 1)

    def test_configuration_rejects_remote_service_and_wrong_model(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            OllamaConfig(model="other:7b")
        with self.assertRaisesRegex(ValueError, "local"):
            OllamaConfig(base_url="https://example.com")


class KaggleNotebookStaticTests(unittest.TestCase):
    def test_notebook_is_valid_has_gates_and_contains_no_secret_literal(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])))
        self.assertIn("APPROVED_GIT_COMMIT", code)
        self.assertIn("CONFIRM_PAID_OR_GPU_INFERENCE is True", code)
        self.assertIn("validate_pilot_inputs", code)
        self.assertIn("run_approved_pilot", code)
        self.assertIn("qwen2.5-coder:7b", code)
        self.assertNotIn("sk-", code)
        self.assertNotIn("hf_", code)
        self.assertNotIn("API_KEY =", code)


class OfficialPilotContractTests(unittest.TestCase):
    def setUp(self):
        self.adapted = load_and_adapt_dataset(PROTOTYPE)
        self.sample_id = next(
            item["sample_id"] for item in self.adapted
            if str(item["audit_only"]["source"]["source_row_id"]) == "17169"
        )

    def test_no_llm_validation_requires_official_ready_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [self.sample_id])
            report = validate_pilot_inputs(
                manifest_path=path,
                dataset_path=PROTOTYPE,
                schema_path=SCHEMA,
            )
            self.assertEqual(report["status"], "PASS_NO_LLM")
            self.assertEqual(report["ordered_sample_ids"], [self.sample_id])
            write_manifest(path, [self.sample_id], ready=False)
            with self.assertRaisesRegex(PilotRunnerError, "NON_READY"):
                load_official_manifest(path, PROTOTYPE)

    def test_manifest_rejects_approval_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [self.sample_id], official=False)
            with self.assertRaisesRegex(PilotRunnerError, "APPROVAL_REQUIRED"):
                load_official_manifest(path, PROTOTYPE)
            payload = write_manifest(path, [self.sample_id])
            payload["sample_list_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PilotRunnerError, "LIST_HASH_MISMATCH"):
                load_official_manifest(path, PROTOTYPE)

    def test_mocked_real_runner_uses_same_list_and_resumes(self):
        class PipelineMock:
            def __call__(self, prompt):
                if '"evidence_integration"' in prompt or '"global_synthesis"' in prompt:
                    payload = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "fixture"}
                elif '"dependency_analysis"' in prompt and '"num_cross_file_dependencies"' in prompt:
                    payload = {"dependency_analysis": "fixture", "dependencies": [], "num_cross_file_dependencies": 0, "overall_coupling": "LOOSE"}
                elif '"function_summary"' in prompt and '"local_verdict"' in prompt:
                    payload = {"file_name": "fixture.c", "function_summary": "fixture", "changes_summary": "fixture", "security_purpose": "fixture", "local_verdict": "CONTRIBUTES_TO_FIX", "confidence": "HIGH", "key_observations": ["fixture"]}
                else:
                    payload = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "fixture"}
                return json.dumps(payload), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            write_manifest(manifest, [self.sample_id])
            kwargs = dict(
                manifest_path=manifest,
                dataset_path=PROTOTYPE,
                schema_path=SCHEMA,
                output_root=root / "output",
                run_prefix="pilot-fixture",
                git_commit="a" * 40,
                model_id="qwen2.5-coder:7b",
                model_digest="sha256:model-fixture",
                parameters=OllamaConfig().run_parameters(),
                retry_policy={"max_attempts": 1},
                client_factory=lambda pipeline: StructuredLLMClient(PipelineMock(), max_attempts=1),
                model_context_tokens=1_000_000,
                reserved_output_tokens=2048,
            )
            first = run_approved_pilot(**kwargs)
            second = run_approved_pilot(**kwargs)
            self.assertTrue(first["complete"])
            self.assertEqual(first, second)
            self.assertEqual(first["result_count_by_pipeline"], {"P0": 1, "P1": 1, "P2": 1})

    def test_qwen_32k_preflight_blocks_oversized_canonical_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [self.sample_id])
            report = validate_pilot_inputs(
                manifest_path=path,
                dataset_path=PROTOTYPE,
                schema_path=SCHEMA,
                model_context_tokens=32768,
                reserved_output_tokens=2048,
            )
            self.assertEqual(report["status"], "BLOCKED_CONTEXT_OVERFLOW_LIKELY")
            self.assertTrue(report["overflow_likely"])


if __name__ == "__main__":
    unittest.main()
