import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


SCHEMA_DIR = Path(__file__).parents[1] / "dataset_validation" / "schemas"
SHA = "a" * 64


def load_schema(name):
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def normalized_sample():
    return {
        "schema_version": "vulnsage-kltndata-1.1",
        "sample_id": "vs-1",
        "family_id": "CVE-2020-0001",
        "dataset_version": "phase2b-proposal",
        "source": {
            "dataset_name": "VulnSage",
            "repository_url": "https://example.org/VulnSage",
            "source_file": "vulnerabilities.csv",
            "source_file_sha256": SHA,
            "source_row_id": "1",
            "source_commit_or_release": None,
        },
        "identity": {
            "cve_id": "CVE-2020-0001",
            "cwe_ids": ["CWE-119"],
            "project": "example",
            "fix_commit_hash": "b" * 40,
            "parent_commit_hash": None,
        },
        "granularity": {
            "label": "G3",
            "rule_id": "operational-num-files-gt-1",
            "declared_num_files": 2,
            "declared_num_functions": 2,
            "parsed_num_files": 2,
            "derivation_status": "UNRESOLVED",
        },
        "files": [
            {
                "path_before": "src/a.c",
                "path_after": "src/a.c",
                "change_type": "MODIFY",
                "vulnerable_code": "int a = 0;",
                "patched_code": "int a = 1;",
                "vulnerable_code_sha256": SHA,
                "patched_code_sha256": SHA,
                "before_provenance": {
                    "status": "VERIFIED",
                    "source": "TARGETED_COMMIT_BLOB",
                    "commit_sha": "c" * 40,
                    "path": "src/a.c",
                    "blob_sha256": SHA,
                },
                "after_provenance": {
                    "status": "VERIFIED",
                    "source": "TARGETED_COMMIT_BLOB",
                    "commit_sha": "b" * 40,
                    "path": "src/a.c",
                    "blob_sha256": SHA,
                },
                "original_git_diff": None,
                "original_git_diff_sha256": None,
                "reconstructed_git_diff": "--- a/src/a.c\n+++ b/src/a.c\n",
                "reconstructed_git_diff_sha256": SHA,
                "diff_origin": "RECONSTRUCTED",
                "changed_hunks": [],
                "coverage": {
                    "path": {"status": "MEASURED", "matched": 1, "total": 1, "ratio": 1},
                    "changed_lines": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
                    "hunks": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
                    "code_context": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
                },
                "validation_flags": [],
            }
        ],
        "full_patch": {
            "original_git_diff": None,
            "original_git_diff_sha256": None,
            "original_git_diff_hash_scope": "UNAVAILABLE",
            "reconstructed_git_diff": "--- a/src/a.c\n+++ b/src/a.c\n",
            "reconstructed_git_diff_sha256": SHA,
            "canonical_source": "ORIGINAL_COMMIT_DIFF",
        },
        "ground_truth": {
            "task": "PATCH_VERIFICATION",
            "label": "PATCHED",
            "label_scope": "fix commit represented by source before/after pair",
            "source": "VULNSAGE_FIX_COMMIT",
            "verification_methods": ["SOURCE_DATASET"],
            "evidence_artifacts": ["vulnerabilities.csv#1"],
            "review_status": "PENDING",
            "reviewers": [],
            "verified_at_utc": None,
        },
        "variant": {
            "is_variant": False,
            "parent_sample_id": None,
            "transformation_type": None,
            "transformation_spec_sha256": None,
            "removed_hunks": [],
        },
        "quality": {
            "validation_status": "PASS",
            "reason_codes": [],
            "noise_score": 0,
            "file_marker_status": "PASS",
            "metadata_consistency": "PASS",
            "diff_round_trip": "NOT_CHECKED",
            "function_boundary_check": "HEURISTIC_ONLY",
            "validation_flags": [],
        },
        "commit_verification": {
            "status": "VERIFIED",
            "commit_sha": "b" * 40,
            "parent_commit_sha": "c" * 40,
            "source_url": "https://example.org/commit/1",
            "reason_codes": [],
            "coverage": {
                "path": {"status": "MEASURED", "matched": 1, "total": 1, "ratio": 1},
                "changed_lines": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
                "hunks": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
                "code_context": {"status": "UNVERIFIED", "matched": 0, "total": 0, "ratio": None},
            },
            "security_semantics_verified": False,
        },
        "build": {
            "builder_version": "1.0.0",
            "evidence_snapshot_sha256": SHA,
            "retrieval_manifest_sha256": SHA,
            "deterministic_input_sha256": SHA,
            "prototype_only": True,
        },
    }


def experiment_manifest():
    return {
        "manifest_schema_version": "experiment-manifest-1.0",
        "run_id": "run-001",
        "created_at_utc": "2026-11-06T00:00:00Z",
        "dataset": {
            "name": "VulnSage normalized evaluation set",
            "version": "not-finalized",
            "canonical_sha256": SHA,
            "schema_version": "vulnsage-kltndata-1.1",
            "source_files": [{"path": "vulnerabilities.csv", "sha256": SHA}],
            "filter_rule_id": "rule-001",
            "filter_rule_sha256": SHA,
            "validation_rule_version": "1.0.0",
        },
        "sample_selection": {
            "split": "PILOT",
            "seed": 20261106,
            "sample_ids": ["vs-1"],
            "sample_list_sha256": SHA,
            "family_ids": ["CVE-2020-0001"],
        },
        "model": {"provider": "local", "model_id": "model", "model_version": "v1"},
        "prompts": [{"prompt_id": "p0", "path": "prompt.txt", "sha256": SHA}],
        "code": {
            "repository": "VulnSage",
            "git_commit": "b" * 40,
            "dirty_worktree": False,
        },
        "pipeline": {
            "pipeline_id": "P0",
            "pipeline_version": "1.0.0",
            "entrypoint_sha256": SHA,
        },
        "parameters": {"temperature": 0, "max_output_tokens": 1024, "extra": {}},
        "retry_policy": {
            "max_attempts": 1,
            "backoff": "none",
            "retryable_errors": [],
            "fallback_label": "ERROR",
        },
        "execution": {
            "status": "PLANNED",
            "started_at_utc": None,
            "completed_at_utc": None,
            "result_files": [],
        },
    }


class DatasetSchemaTests(unittest.TestCase):
    def test_schemas_are_valid_draft_2020_12(self):
        for name in (
            "normalized_sample.schema.json",
            "experiment_manifest.schema.json",
            "derived_representation.schema.json",
        ):
            Draft202012Validator.check_schema(load_schema(name))

    def test_valid_normalized_sample(self):
        validator = Draft202012Validator(
            load_schema("normalized_sample.schema.json"), format_checker=FormatChecker()
        )
        validator.validate(normalized_sample())

    def test_normalized_sample_requires_ground_truth_provenance(self):
        sample = normalized_sample()
        del sample["ground_truth"]
        validator = Draft202012Validator(load_schema("normalized_sample.schema.json"))
        with self.assertRaises(ValidationError):
            validator.validate(sample)

    def test_valid_experiment_manifest(self):
        validator = Draft202012Validator(
            load_schema("experiment_manifest.schema.json"), format_checker=FormatChecker()
        )
        validator.validate(experiment_manifest())

    def test_manifest_rejects_missing_model_version_and_bad_hash(self):
        schema = load_schema("experiment_manifest.schema.json")
        validator = Draft202012Validator(schema)
        missing_version = experiment_manifest()
        del missing_version["model"]["model_version"]
        with self.assertRaises(ValidationError):
            validator.validate(missing_version)

        bad_hash = copy.deepcopy(experiment_manifest())
        bad_hash["dataset"]["canonical_sha256"] = "not-a-sha256"
        with self.assertRaises(ValidationError):
            validator.validate(bad_hash)


if __name__ == "__main__":
    unittest.main()
