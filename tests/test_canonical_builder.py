import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dataset_validation.canonical_builder import (
    build_canonical_sample,
    ensure_unique_sample_ids,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CanonicalBuilderTests(unittest.TestCase):
    def setUp(self):
        self.commit = "a" * 40
        self.parent = "b" * 40
        self.row = {
            "id": "17283",
            "COMMIT_HASH": self.commit,
            "VULNERABILITY_CVE": "CVE-2019-17350",
            "VULNERABILITY_CWE": "CWE-835",
            "PROJECT": "xen",
            "VULNERABLE_CODE_BLOCK": "// File: old.c\nold\n// File: renamed.c\nold name",
            "PATCHED_CODE_BLOCK": "// File: new.c\nnew\n// File: final.c\nnew name",
            "NUM_FILES_CHANGED": "3",
            "NUM_FUNCTIONS_CHANGED": "2",
            "NOISE_AMOUNT": "0",
        }
        self.evidence = {
            "sha": self.commit,
            "parents": [{"sha": self.parent}],
            "html_url": "https://example.org/commit",
            "files": [
                {
                    "filename": "old.c",
                    "status": "removed",
                    "additions": 0,
                    "deletions": 1,
                    "patch": "@@ -1 +0,0 @@\n-old",
                },
                {
                    "filename": "new.c",
                    "status": "added",
                    "additions": 1,
                    "deletions": 0,
                    "patch": "@@ -0,0 +1 @@\n+new",
                },
                {
                    "filename": "final.c",
                    "previous_filename": "renamed.c",
                    "status": "renamed",
                    "additions": 1,
                    "deletions": 1,
                    "patch": "@@ -1 +1 @@\n-old name\n+new name",
                },
            ],
        }

    def build_fixture(self, directory):
        root = Path(directory)
        resources = []
        contents = {
            "original.patch": b"original commit patch\n",
            "blobs/before/old.c": b"old\n",
            "blobs/after/new.c": b"new\n",
            "blobs/before/renamed.c": b"old name\n",
            "blobs/after/final.c": b"new name\n",
        }
        metadata = {
            "original.patch": ("ORIGINAL_DIFF", None, None, self.commit),
            "blobs/before/old.c": ("BLOB", "BEFORE", "old.c", self.parent),
            "blobs/after/new.c": ("BLOB", "AFTER", "new.c", self.commit),
            "blobs/before/renamed.c": ("BLOB", "BEFORE", "renamed.c", self.parent),
            "blobs/after/final.c": ("BLOB", "AFTER", "final.c", self.commit),
        }
        for relative, data in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            kind, side, source_path, commit = metadata[relative]
            resources.append(
                {
                    "kind": kind,
                    "side": side,
                    "path": source_path,
                    "commit_sha": commit,
                    "local_path": relative,
                    "sha256": sha(data),
                    "status": "VERIFIED",
                    "url": "https://example.org/resource",
                    "size_bytes": len(data),
                }
            )
        return {
            "retriever_version": "1.0.0",
            "source_id": "17283",
            "repository": "owner/repo",
            "commit_sha": self.commit,
            "parent_commit_sha": self.parent,
            "operations": [],
            "resources": resources,
            "total_bytes": sum(len(value) for value in contents.values()),
        }

    def test_builder_preserves_add_delete_rename_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.build_fixture(directory)
            ground_truth = {
                "task": "PATCH_VERIFICATION",
                "label": "UNKNOWN",
                "label_scope": "caller supplied prototype label",
                "source": "UNVERIFIED",
                "verification_methods": [],
                "evidence_artifacts": [],
                "review_status": "PENDING",
                "reviewers": [],
                "verified_at_utc": None,
            }
            sample = build_canonical_sample(
                self.row,
                {"status": "UNRESOLVED", "reason_codes": []},
                self.evidence,
                manifest,
                retrieval_manifest_dir=Path(directory),
                source_file="vulnerabilities.csv",
                source_file_sha256="c" * 64,
                source_repository_url="https://example.org/source",
                dataset_version="prototype",
                ground_truth=ground_truth,
                evidence_snapshot_sha256="d" * 64,
                retrieval_manifest_sha256="e" * 64,
            )
            self.assertEqual([item["change_type"] for item in sample["files"]], ["DELETE", "ADD", "RENAME"])
            self.assertIsNone(sample["files"][0]["path_after"])
            self.assertIsNone(sample["files"][1]["path_before"])
            self.assertEqual(sample["files"][2]["path_before"], "renamed.c")
            self.assertEqual(sample["files"][2]["path_after"], "final.c")
            self.assertEqual(sample["ground_truth"], ground_truth)
            self.assertFalse(sample["commit_verification"]["security_semantics_verified"])

            schema_path = Path(__file__).parents[1] / "dataset_validation" / "schemas" / "normalized_sample.schema.json"
            Draft202012Validator(json.loads(schema_path.read_text())).validate(sample)

    def test_builder_is_deterministic_and_does_not_modify_source_record(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.build_fixture(directory)
            original = json.dumps(self.row, sort_keys=True)
            kwargs = dict(
                row=self.row,
                validation_record={"status": "UNRESOLVED", "reason_codes": []},
                github_evidence=self.evidence,
                retrieval_manifest=manifest,
                retrieval_manifest_dir=Path(directory),
                source_file="vulnerabilities.csv",
                source_file_sha256="c" * 64,
                source_repository_url="https://example.org/source",
                dataset_version="prototype",
                ground_truth={
                    "task": "PATCH_VERIFICATION",
                    "label": "PATCHED",
                    "label_scope": "source fix commit",
                    "source": "VULNSAGE_FIX_COMMIT",
                    "verification_methods": ["SOURCE_DATASET"],
                    "evidence_artifacts": [],
                    "review_status": "PENDING",
                    "reviewers": [],
                    "verified_at_utc": None,
                },
                evidence_snapshot_sha256="d" * 64,
                retrieval_manifest_sha256="e" * 64,
            )
            first = build_canonical_sample(**kwargs)
            second = build_canonical_sample(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(json.dumps(self.row, sort_keys=True), original)

    def test_missing_blob_remains_null_and_is_never_filled_from_csv_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.build_fixture(directory)
            manifest["resources"] = [
                item
                for item in manifest["resources"]
                if not (item["side"] == "AFTER" and item["path"] == "new.c")
            ]
            sample = build_canonical_sample(
                self.row,
                {"status": "UNRESOLVED", "reason_codes": []},
                self.evidence,
                manifest,
                retrieval_manifest_dir=Path(directory),
                source_file="vulnerabilities.csv",
                source_file_sha256="c" * 64,
                source_repository_url="https://example.org/source",
                dataset_version="prototype",
                ground_truth={
                    "task": "PATCH_VERIFICATION",
                    "label": "PATCHED",
                    "label_scope": "source fix commit",
                    "source": "VULNSAGE_FIX_COMMIT",
                    "verification_methods": [],
                    "evidence_artifacts": [],
                    "review_status": "PENDING",
                    "reviewers": [],
                    "verified_at_utc": None,
                },
                evidence_snapshot_sha256="d" * 64,
                retrieval_manifest_sha256="e" * 64,
            )
            added = next(item for item in sample["files"] if item["change_type"] == "ADD")
            self.assertIsNone(added["patched_code"])
            self.assertEqual(added["after_provenance"]["status"], "UNVERIFIABLE")
            self.assertEqual(sample["quality"]["validation_status"], "FAIL")

    def test_duplicate_sample_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            ensure_unique_sample_ids([{"sample_id": "x"}, {"sample_id": "x"}])


if __name__ == "__main__":
    unittest.main()
