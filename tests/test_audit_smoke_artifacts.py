import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.audit_smoke_artifacts import (
    AuditExpectations,
    SmokeAuditError,
    audit_archive,
    write_outputs,
)


SAMPLE = "vs:e51ac3a94239:19047:canonical"
EXPERIMENT_SHA = "2" * 40
PACKAGE_SHA = "4" * 40
POLICY_SHA = "9" * 64
MODEL_DIGEST = "d" * 64
RUN_PREFIX = "smoke-19047-test"


def canonical_hash(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def build_archive(
    path: Path,
    *,
    omit: str | None = None,
    result_sample: str = SAMPLE,
    result_pipeline_override: str | None = None,
    result_policy: str = POLICY_SHA,
    ground_truth_status: str = "PENDING",
):
    entries = {}
    dataset = {
        "samples": [
            {
                "sample_id": SAMPLE,
                "ground_truth": {
                    "label": "PATCHED",
                    "review_status": ground_truth_status,
                },
            }
        ]
    }
    qualification = {
        "sample_id": SAMPLE,
        "ground_truth_review_status_unchanged": ground_truth_status,
        "security_correctness_verified": False,
    }
    decision = {
        "sample_id": SAMPLE,
        "decision": "APPROVE",
        "reviewer_id": "fixture-reviewer",
        "reviewer_name": "Fixture Reviewer",
    }
    dataset_raw = json_bytes(dataset)
    qualification_raw = json_bytes(qualification)
    decision_raw = json_bytes(decision)
    manifest = {
        "git_commit": PACKAGE_SHA,
        "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "ordered_sample_ids": [SAMPLE],
        "official_split": False,
        "scope": "ONE_SAMPLE_SMOKE_NOT_MAIN_EXPERIMENT",
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    manifest_raw = json_bytes(manifest)
    receipt = {
        "git_commit": PACKAGE_SHA,
        "sample_id": SAMPLE,
        "files": {
            "canonical_dataset.json": hashlib.sha256(dataset_raw).hexdigest(),
            "official_pilot_manifest.json": hashlib.sha256(manifest_raw).hexdigest(),
            "qualification_record.json": hashlib.sha256(qualification_raw).hexdigest(),
            "review_decision.json": hashlib.sha256(decision_raw).hexdigest(),
        },
    }
    receipt_raw = json_bytes(receipt)
    package_raw = {
        "canonical_dataset": dataset_raw,
        "official_manifest": manifest_raw,
        "package_receipt": receipt_raw,
        "qualification_record": qualification_raw,
        "review_decision": decision_raw,
    }
    entries.update(
        {
            "smoke_package/canonical_dataset.json": dataset_raw,
            "smoke_package/official_pilot_manifest.json": manifest_raw,
            "smoke_package/package_receipt.json": receipt_raw,
            "smoke_package/qualification_record.json": qualification_raw,
            "smoke_package/review_decision.json": decision_raw,
        }
    )

    preparation = {
        "sample_id": SAMPLE,
        "experiment_git_sha": EXPERIMENT_SHA,
        "source_package_git_sha": PACKAGE_SHA,
        "run_prefix_proposed": RUN_PREFIX,
        "inference_executed": False,
    }
    policy = {
        "sample_id": SAMPLE,
        "experiment_code_git_commit": EXPERIMENT_SHA,
        "source_package_git_commit": PACKAGE_SHA,
        "policy_sha256": POLICY_SHA,
        "policy_name": "before_plus_original_diff_v1",
        "decision": "APPROVE",
        "scope": "engineering_smoke_only",
    }
    context = {
        "sample_id": SAMPLE,
        "experiment_code_git_commit": EXPERIMENT_SHA,
        "source_package_git_commit": PACKAGE_SHA,
        "reports": [
            {
                "policy_sha256": POLICY_SHA,
                "code_representation": "before_plus_original_diff_v1",
            }
        ],
    }
    authorization = {
        "sample_id": SAMPLE,
        "experiment_git_sha": EXPERIMENT_SHA,
        "source_package_git_sha": PACKAGE_SHA,
        "context_policy_sha256": POLICY_SHA,
        "run_prefix": RUN_PREFIX,
        "model": "qwen2.5-coder:7b",
        "model_digest": MODEL_DIGEST,
        "scope": "engineering_smoke_only",
        "inference_authorized": True,
        "main_experiment_authorized": False,
        "held_out_evaluation_authorized": False,
    }
    attempt_record = {
        "sample_id": SAMPLE,
        "experiment_git_sha": EXPERIMENT_SHA,
        "source_package_git_sha": PACKAGE_SHA,
        "context_policy_sha256": POLICY_SHA,
        "run_prefix": RUN_PREFIX,
    }
    runtime = {
        "experiment_git_sha": EXPERIMENT_SHA,
        "source_package_git_sha": PACKAGE_SHA,
        "approved_context_policy_sha256": POLICY_SHA,
        "inference_calls": 0,
        "generated_tokens": 0,
    }
    for name, value in {
        "evidence/smoke-preparation-fixture.json": preparation,
        "evidence/context-policy-approval-fixture.json": policy,
        "evidence/context-19047-fixture.json": context,
        "evidence/inference-authorization-fixture.json": authorization,
        "evidence/smoke-attempt-v2-fixture.json": attempt_record,
        "evidence/runtime-loadcheck-fixture.json": runtime,
    }.items():
        entries[name] = json_bytes(value)

    inventory = []
    directories = {}
    artifact_hashes = {
        key: hashlib.sha256(raw).hexdigest() for key, raw in package_raw.items()
    }
    for pipeline, call_count in (("P0", 1), ("P1", 2), ("P2", 3)):
        lower = pipeline.lower()
        run_id = f"{RUN_PREFIX}-{lower}"
        directories[lower] = f"/kaggle/working/runs/{run_id}"
        spec = {
            "run_id": run_id,
            "pipeline_id": pipeline,
            "sample_ids": [SAMPLE],
            "sample_list_sha256": canonical_hash([SAMPLE]),
            "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
            "git_commit": EXPERIMENT_SHA,
            "source_package_git_commit": PACKAGE_SHA,
            "context_policy_sha256": POLICY_SHA,
            "context_policy_configuration": {
                "code_representation": "before_plus_original_diff_v1"
            },
            "model": {"id": "qwen2.5-coder:7b", "version": MODEL_DIGEST},
            "parameters": {"num_ctx": 32768, "num_predict": 2048},
            "artifact_hashes": artifact_hashes,
        }
        spec["spec_sha256"] = canonical_hash(spec)
        attempts = [
            {
                "attempt": 1,
                "status": "OK",
                "error_message": None,
                "error_type": None,
                "input_tokens": 100 + index,
                "output_tokens": 10,
                "total_tokens": 110 + index,
                "duration_sec": 1.0 + index,
            }
            for index in range(call_count)
        ]
        result = {
            "sample_id": result_sample,
            "pipeline": result_pipeline_override or pipeline,
            "execution_status": "COMPLETED",
            "prediction": "NOT_SURE" if pipeline == "P0" else "PATCHED",
            "model_abstained": pipeline == "P0",
            "num_attempts": call_count,
            "attempts": attempts,
            "token_usage": {
                "input_tokens": sum(x["input_tokens"] for x in attempts),
                "output_tokens": sum(x["output_tokens"] for x in attempts),
                "total_tokens": sum(x["total_tokens"] for x in attempts),
                "usage_complete": True,
            },
            "total_attempt_duration_sec": sum(x["duration_sec"] for x in attempts),
            "exact_token_preflight": "EXACT",
            "pre_call_token_counts": [
                {"stage": f"{lower}.{index}", "input_tokens": item["input_tokens"] - 29}
                for index, item in enumerate(attempts)
            ],
            "context_policy_sha256": result_policy,
            "spec_sha256": spec["spec_sha256"],
        }
        journal = {
            "actual_attempts": call_count,
            "event": "SAMPLE_RESULT_WRITTEN",
            "execution_status": "COMPLETED",
            "sample_id": result_sample,
            "spec_sha256": spec["spec_sha256"],
        }
        run_entries = {
            "run_spec.json": json_bytes(spec),
            "results/result.json": json_bytes(result),
            "execution_journal.jsonl": (
                json.dumps(journal, sort_keys=True) + "\n"
            ).encode("utf-8"),
        }
        for relative, raw in run_entries.items():
            full = f"runs/{run_id}/{relative}"
            entries[full] = raw
            inventory.append(
                {
                    "pipeline": pipeline,
                    "relative_path": relative,
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )

    audit_index = {
        "record_type": "engineering_smoke_artifact_index",
        "scope": "engineering_smoke_only",
        "run_prefix": RUN_PREFIX,
        "experiment_git_sha": EXPERIMENT_SHA,
        "source_package_git_sha": PACKAGE_SHA,
        "sample_id": SAMPLE,
        "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "pipeline_directories": directories,
        "artifact_inventory": inventory,
        "provenance_files": [
            "/kaggle/working/smoke-preparation-fixture.json",
            "/kaggle/working/context-policy-approval-fixture.json",
            "/kaggle/working/context-19047-fixture.json",
            "/kaggle/working/inference-authorization-fixture.json",
            "/kaggle/working/smoke-attempt-v2-fixture.json",
            "/kaggle/working/runtime-loadcheck-fixture.json",
        ],
        "ground_truth_review_status": ground_truth_status,
    }
    entries["audit_index.json"] = json_bytes(audit_index)
    if omit is not None:
        entries.pop(omit)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SmokeArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "audit.zip"

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self, sha):
        return audit_archive(
            self.archive,
            expected_archive_sha256=sha,
            expectations=AuditExpectations(
                experiment_git_sha=EXPERIMENT_SHA,
                source_package_git_sha=PACKAGE_SHA,
                sample_id=SAMPLE,
                run_prefix=RUN_PREFIX,
                model_id="qwen2.5-coder:7b",
                context_policy_sha256=POLICY_SHA,
            ),
        )

    def test_valid_archive(self):
        metrics = self.audit(build_archive(self.archive))
        self.assertEqual(metrics["status"], "PASS")
        self.assertEqual(metrics["scientific_interpretation"]["ground_truth"], "PENDING")

    def test_sha256_mismatch_fails_closed(self):
        build_archive(self.archive)
        with self.assertRaisesRegex(SmokeAuditError, "ARCHIVE_SHA256_MISMATCH"):
            self.audit("0" * 64)

    def test_missing_run_spec_fails_closed(self):
        sha = build_archive(
            self.archive,
            omit=f"runs/{RUN_PREFIX}-p0/run_spec.json",
        )
        with self.assertRaisesRegex(SmokeAuditError, "MISSING_ARCHIVE_ENTRY"):
            self.audit(sha)

    def test_missing_result_fails_closed(self):
        sha = build_archive(
            self.archive,
            omit=f"runs/{RUN_PREFIX}-p1/results/result.json",
        )
        with self.assertRaisesRegex(SmokeAuditError, "MISSING_ARCHIVE_ENTRY"):
            self.audit(sha)

    def test_wrong_sample_id_fails_closed(self):
        sha = build_archive(self.archive, result_sample="wrong-sample")
        with self.assertRaisesRegex(SmokeAuditError, "RESULT_SAMPLE_ID_MISMATCH"):
            self.audit(sha)

    def test_wrong_pipeline_fails_closed(self):
        sha = build_archive(self.archive, result_pipeline_override="PX")
        with self.assertRaisesRegex(SmokeAuditError, "RESULT_PIPELINE_MISMATCH"):
            self.audit(sha)

    def test_wrong_policy_sha_fails_closed(self):
        sha = build_archive(self.archive, result_policy="8" * 64)
        with self.assertRaisesRegex(SmokeAuditError, "RESULT_POLICY_MISMATCH"):
            self.audit(sha)

    def test_actual_call_count_is_not_journal_line_count(self):
        metrics = self.audit(build_archive(self.archive))
        self.assertEqual(metrics["totals"]["actual_llm_calls"], 6)
        self.assertEqual(metrics["totals"]["journal_lines"], 3)
        self.assertEqual(metrics["totals"]["retry_calls_due_to_error"], 0)

    def test_token_aggregation_uses_attempt_records(self):
        metrics = self.audit(build_archive(self.archive))
        expected_input = 100 + (100 + 101) + (100 + 101 + 102)
        self.assertEqual(metrics["totals"]["input_tokens"], expected_input)
        self.assertEqual(metrics["totals"]["output_tokens"], 60)
        self.assertEqual(metrics["totals"]["total_tokens"], expected_input + 60)

    def test_ground_truth_must_remain_pending(self):
        sha = build_archive(self.archive, ground_truth_status="VERIFIED")
        with self.assertRaisesRegex(SmokeAuditError, "INDEX_GROUND_TRUTH_NOT_PENDING"):
            self.audit(sha)

    def test_no_quality_metrics_when_label_review_is_pending(self):
        metrics = self.audit(build_archive(self.archive))
        scientific = metrics["scientific_interpretation"]
        self.assertIn("NOT COMPUTED", scientific["classification_quality_metrics"])
        for forbidden in ("accuracy", "precision", "recall", "f1"):
            self.assertNotIn(forbidden, scientific)

    def test_output_writer_versions_instead_of_overwriting(self):
        metrics = self.audit(build_archive(self.archive))
        first = write_outputs(metrics, self.root / "reports")
        second = write_outputs(metrics, self.root / "reports")
        self.assertNotEqual(first, second)
        self.assertEqual(first[0].name, "smoke_19047_verified_metrics.json")
        self.assertEqual(second[0].name, "smoke_19047_verified_metrics_v2.json")


if __name__ == "__main__":
    unittest.main()
