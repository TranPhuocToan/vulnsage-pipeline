import copy
import json
import tempfile
import unittest
from pathlib import Path

from dataset_validation.qualification import (
    QualificationError,
    QualificationPolicy,
    qualify_sample,
)
from dataset_validation.qualification_batch import (
    QualificationBatchError,
    run_qualification_batch,
)
from pipeline.canonical_adapter import adapt_sample, load_and_adapt_dataset
from pipeline.llm_client import StructuredLLMClient
from pipeline.preflight import PreflightError, run_offline_preflight


WORKSPACE = Path(__file__).resolve().parents[3]
PROTOTYPE = WORKSPACE / "docs" / "audits" / "artifacts" / "phase2e" / "prototype_normalized_dataset.json"
SCHEMA = Path(__file__).resolve().parents[1] / "dataset_validation" / "schemas" / "normalized_sample.schema.json"
MULTI_COMMIT_FAMILIES = ("CVE-2014-2915", "CVE-2019-17350")


def raw_samples():
    return json.loads(PROTOTYPE.read_text(encoding="utf-8"))["samples"]


def sample_by_source(source_id):
    return next(item for item in raw_samples() if str(item["source"]["source_row_id"]) == str(source_id))


def approval(sample, waivers=()):
    return {
        "decision_id": f"review-{sample['source']['source_row_id']}",
        "sample_id": sample["sample_id"],
        "decision": "APPROVE",
        "reviewer_id": "reviewer-fixture",
        "reviewer_name": "Offline Test Reviewer",
        "reason": "Fixture decision tests external review provenance.",
        "decided_at_utc": "2026-11-07T00:00:00Z",
        "evidence_refs": ["fixture://review/evidence"],
        "waivers": [
            {
                "reason_code": code,
                "scope": "DERIVED_REPRESENTATION_ONLY",
                "reason": "Canonical commit evidence is complete in this fixture.",
                "evidence_refs": ["fixture://canonical/evidence"],
            }
            for code in waivers
        ],
    }


class DeterministicMockTransport:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if '"evidence_integration"' in prompt or '"global_synthesis"' in prompt:
            payload = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "offline fixture"}
        elif '"dependency_analysis"' in prompt and '"num_cross_file_dependencies"' in prompt:
            payload = {
                "dependency_analysis": "offline fixture",
                "dependencies": [],
                "num_cross_file_dependencies": 0,
                "overall_coupling": "LOOSE",
            }
        elif '"function_summary"' in prompt and '"local_verdict"' in prompt:
            payload = {
                "file_name": "fixture.c",
                "function_summary": "offline fixture",
                "changes_summary": "offline fixture",
                "security_purpose": "offline fixture",
                "local_verdict": "CONTRIBUTES_TO_FIX",
                "confidence": "HIGH",
                "key_observations": ["offline fixture"],
            }
        else:
            payload = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "offline fixture"}
        return json.dumps(payload), {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


class QualificationProtocolTests(unittest.TestCase):
    def test_prototype_statuses_and_ground_truth_are_fail_closed(self):
        before = PROTOTYPE.read_bytes()
        records = {
            str(sample["source"]["source_row_id"]): qualify_sample(
                sample,
                family_requires_review=sample["family_id"] in MULTI_COMMIT_FAMILIES,
            )
            for sample in raw_samples()
        }
        self.assertEqual(sum(item["status"] == "OUT_OF_SCOPE" for item in records.values()), 2)
        self.assertEqual(sum(item["status"] == "NEEDS_APPROVAL" for item in records.values()), 8)
        self.assertTrue(all(not item["security_correctness_verified"] for item in records.values()))
        self.assertTrue(all(item["ground_truth_review_status_unchanged"] == "PENDING" for item in records.values()))
        self.assertIn("MULTI_COMMIT_FAMILY_REVIEW_REQUIRED", records["16627"]["reason_codes"])
        self.assertEqual(PROTOTYPE.read_bytes(), before)

    def test_canonical_gap_blocks_and_cannot_be_waived(self):
        sample = copy.deepcopy(sample_by_source(17169))
        sample["files"][0]["before_provenance"]["status"] = "UNVERIFIABLE"
        record = qualify_sample(sample, decision=approval(sample))
        self.assertEqual(record["status"], "BLOCKED")
        self.assertIn("FILE_0_BEFORE_PROVENANCE_UNVERIFIABLE", record["canonical_blockers"])
        bad = approval(sample, ["FILE_0_BEFORE_PROVENANCE_UNVERIFIABLE"])
        with self.assertRaisesRegex(QualificationError, "waiver not allowed"):
            qualify_sample(sample, decision=bad)

    def test_attributable_review_can_approve_only_after_derived_gaps_are_addressed(self):
        clean = sample_by_source(17169)
        clean_record = qualify_sample(clean, decision=approval(clean))
        self.assertEqual(clean_record["status"], "READY")
        self.assertEqual(clean_record["ground_truth_review_status_unchanged"], "PENDING")
        self.assertFalse(clean_record["security_correctness_verified"])

        gap = sample_by_source(17415)
        pending = qualify_sample(gap, decision=approval(gap))
        self.assertEqual(pending["status"], "NEEDS_APPROVAL")
        approved = qualify_sample(
            gap,
            decision=approval(gap, pending["derived_representation_gaps"]),
        )
        self.assertEqual(approved["status"], "READY")


class QualificationBatchTests(unittest.TestCase):
    def test_checkpoint_resume_and_spec_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qualification"
            first = run_qualification_batch(
                dataset_path=PROTOTYPE,
                output_dir=output,
                candidate_source_ids=["16627", "17169", "99999"],
                multi_commit_family_ids=MULTI_COMMIT_FAMILIES,
                max_new_samples=3,
            )
            self.assertFalse(first["complete"])
            self.assertEqual(first["canonical_samples_completed"], 3)
            resumed = run_qualification_batch(
                dataset_path=PROTOTYPE,
                output_dir=output,
                candidate_source_ids=["16627", "17169", "99999"],
                multi_commit_family_ids=MULTI_COMMIT_FAMILIES,
            )
            self.assertTrue(resumed["complete"])
            self.assertEqual(resumed["canonical_status_counts"], {"NEEDS_APPROVAL": 8, "OUT_OF_SCOPE": 2})
            self.assertEqual(resumed["candidate_frame_status_counts"]["NOT_ASSESSED"], 1)
            with self.assertRaisesRegex(QualificationBatchError, "RESUME_SPEC_MISMATCH"):
                run_qualification_batch(
                    dataset_path=PROTOTYPE,
                    output_dir=output,
                    candidate_source_ids=["17169"],
                    multi_commit_family_ids=MULTI_COMMIT_FAMILIES,
                )


class OfflinePreflightTests(unittest.TestCase):
    def setUp(self):
        self.adapted = load_and_adapt_dataset(PROTOTYPE)
        selected_sources = {"17169", "17415"}
        selected_raw = [
            item for item in raw_samples()
            if str(item["source"]["source_row_id"]) in selected_sources
        ]
        self.ids = [item["sample_id"] for item in selected_raw]
        self.qualifications = {
            item["sample_id"]: qualify_sample(item) for item in selected_raw
        }

    @staticmethod
    def client_factory(_pipeline):
        return StructuredLLMClient(DeterministicMockTransport(), max_attempts=1)

    def test_same_ordered_list_interrupt_resume_and_positive_only_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                dataset_path=PROTOTYPE,
                adapted_samples=self.adapted,
                qualifications=self.qualifications,
                ordered_sample_ids=self.ids,
                schema_path=SCHEMA,
                output_root=Path(directory),
                run_prefix="fixture",
                client_factory=self.client_factory,
                git_commit=None,
            )
            interrupted = run_offline_preflight(**kwargs, max_new_samples_per_pipeline=1)
            self.assertFalse(interrupted["complete"])
            self.assertIsNone(interrupted["evaluation"])
            resumed = run_offline_preflight(**kwargs)
            self.assertTrue(resumed["complete"])
            self.assertEqual(resumed["result_count_by_pipeline"], {"P0": 2, "P1": 2, "P2": 2})
            self.assertEqual(
                {item["N"] for item in resumed["evaluation"].values()}, {2}
            )
            self.assertTrue(all(item["C"] == 2 for item in resumed["evaluation"].values()))

    def test_official_gate_rejects_pending_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                dataset_path=PROTOTYPE,
                adapted_samples=self.adapted,
                qualifications=self.qualifications,
                ordered_sample_ids=self.ids,
                schema_path=SCHEMA,
                output_root=Path(directory),
                run_prefix="fixture",
                client_factory=self.client_factory,
                git_commit=None,
            )
            with self.assertRaisesRegex(PreflightError, "qualification gate rejected"):
                run_offline_preflight(**kwargs, allow_needs_approval_for_preflight=False)
            run_offline_preflight(**kwargs)
            with self.assertRaisesRegex(PreflightError, "MANIFEST_MISMATCH"):
                run_offline_preflight(**{**kwargs, "ordered_sample_ids": list(reversed(self.ids))})


if __name__ == "__main__":
    unittest.main()
