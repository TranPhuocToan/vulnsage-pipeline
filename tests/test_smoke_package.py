import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dataset_validation.smoke_package import SmokePackageError, prepare_one_sample_smoke_package
from pipeline.pilot_runner import load_official_manifest


WORKSPACE = Path(__file__).resolve().parents[3]
PROTOTYPE = WORKSPACE / "docs" / "audits" / "artifacts" / "phase2e" / "prototype_normalized_dataset.json"
SCHEMA = Path(__file__).resolve().parents[1] / "dataset_validation" / "schemas" / "normalized_sample.schema.json"
SAMPLE_ID = "vs:e51ac3a94239:19047:canonical"
GAPS = (
    "FILE_COUNT_MISMATCH",
    "RECONSTRUCTED_CHANGE_COUNT_MISMATCH",
    "SOURCE_COMMIT_FILE_MISSING",
)


def signed_fixture_decision():
    return {
        "decision_id": "fixture-review-19047-v1",
        "sample_id": SAMPLE_ID,
        "decision": "APPROVE",
        "reviewer_id": "fixture-reviewer",
        "reviewer_name": "Fixture Reviewer",
        "reason": "Test-only attributable decision; not a research approval.",
        "decided_at_utc": "2026-11-07T05:00:00Z",
        "evidence_refs": ["fixture://original-patch", "fixture://advisory"],
        "waivers": [
            {
                "reason_code": code,
                "scope": "DERIVED_REPRESENTATION_ONLY",
                "reason": "Test-only canonical evidence finding.",
                "evidence_refs": ["fixture://canonical-evidence"],
            }
            for code in GAPS
        ],
    }


class SmokePackageTests(unittest.TestCase):
    def test_documented_direct_cli_invocation_loads_repository_modules(self):
        repository = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "-B", "scripts/prepare_smoke_package.py", "--help"],
            cwd=repository,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--review-decision", completed.stdout)

    def test_ready_package_is_deterministic_valid_and_preserves_pending_ground_truth(self):
        before = hashlib.sha256(PROTOTYPE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = root / "decision.json"
            decision.write_text(json.dumps(signed_fixture_decision()), encoding="utf-8")
            output = root / "package"
            result = prepare_one_sample_smoke_package(
                source_dataset_path=PROTOTYPE,
                source_row_id="19047",
                review_decision_path=decision,
                schema_path=SCHEMA,
                output_dir=output,
                git_commit="1b564e510e6deafc93719913a1618799a65e3c7a",
            )
            self.assertEqual(result["status"], "READY_PACKAGE_CREATED")
            self.assertEqual(result["ground_truth_review_status_unchanged"], "PENDING")
            self.assertFalse(result["security_correctness_verified"])
            dataset = json.loads((output / "canonical_dataset.json").read_text(encoding="utf-8"))
            qualification = json.loads((output / "qualification_record.json").read_text(encoding="utf-8"))
            receipt = json.loads((output / "package_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset["samples"][0]["ground_truth"]["review_status"], "PENDING")
            self.assertEqual(dataset["samples"][0]["ground_truth"]["label"], "PATCHED")
            self.assertEqual(qualification["status"], "READY")
            manifest = load_official_manifest(output / "official_pilot_manifest.json", output / "canonical_dataset.json")
            self.assertEqual(manifest["ordered_sample_ids"], [SAMPLE_ID])
            self.assertEqual(manifest["git_commit"], "1b564e510e6deafc93719913a1618799a65e3c7a")
            for name, expected_hash in receipt["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected_hash)
            with self.assertRaisesRegex(SmokePackageError, "OUTPUT_ALREADY_EXISTS"):
                prepare_one_sample_smoke_package(
                    source_dataset_path=PROTOTYPE,
                    source_row_id="19047",
                    review_decision_path=decision,
                    schema_path=SCHEMA,
                    output_dir=output,
                    git_commit="1b564e510e6deafc93719913a1618799a65e3c7a",
                )
        self.assertEqual(hashlib.sha256(PROTOTYPE.read_bytes()).hexdigest(), before)

    def test_reviewer_provenance_and_sample_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            missing_reviewer = signed_fixture_decision()
            del missing_reviewer["reviewer_id"]
            cases.append((missing_reviewer, "missing fields"))
            placeholder_reviewer = signed_fixture_decision()
            placeholder_reviewer["reviewer_name"] = "REPLACE_WITH_REVIEWER_NAME"
            cases.append((placeholder_reviewer, "template placeholder"))
            no_evidence = signed_fixture_decision()
            no_evidence["evidence_refs"] = []
            cases.append((no_evidence, "non-empty list"))
            wrong_sample = signed_fixture_decision()
            wrong_sample["sample_id"] = "vs:wrong:sample"
            cases.append((wrong_sample, "sample_id mismatch"))
            for index, (decision, message) in enumerate(cases):
                with self.subTest(case=index):
                    path = root / f"decision-{index}.json"
                    path.write_text(json.dumps(decision), encoding="utf-8")
                    with self.assertRaisesRegex((SmokePackageError, ValueError), message):
                        prepare_one_sample_smoke_package(
                            source_dataset_path=PROTOTYPE,
                            source_row_id="19047",
                            review_decision_path=path,
                            schema_path=SCHEMA,
                            output_dir=root / f"output-{index}",
                            git_commit="1b564e510e6deafc93719913a1618799a65e3c7a",
                        )
    def test_non_approve_and_incomplete_waivers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = signed_fixture_decision()
            decision["decision"] = "DEFER"
            path = root / "defer.json"
            path.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(SmokePackageError, "APPROVE_DECISION_REQUIRED"):
                prepare_one_sample_smoke_package(
                    source_dataset_path=PROTOTYPE,
                    source_row_id="19047",
                    review_decision_path=path,
                    schema_path=SCHEMA,
                    output_dir=root / "deferred",
                    git_commit="1b564e510e6deafc93719913a1618799a65e3c7a",
                )
            decision = signed_fixture_decision()
            decision["waivers"] = decision["waivers"][:-1]
            path = root / "incomplete.json"
            path.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(SmokePackageError, "QUALIFICATION_NOT_READY"):
                prepare_one_sample_smoke_package(
                    source_dataset_path=PROTOTYPE,
                    source_row_id="19047",
                    review_decision_path=path,
                    schema_path=SCHEMA,
                    output_dir=root / "incomplete",
                    git_commit="1b564e510e6deafc93719913a1618799a65e3c7a",
                )


if __name__ == "__main__":
    unittest.main()
