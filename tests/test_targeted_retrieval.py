import json
import tempfile
import unittest
from pathlib import Path

from dataset_validation.targeted_retrieval import create_retrieval_plan, retrieve_plan


class TargetedRetrievalTests(unittest.TestCase):
    def evidence(self):
        return {
            "sha": "a" * 40,
            "parents": [{"sha": "b" * 40}],
            "files": [
                {"filename": "mod.c", "status": "modified"},
                {"filename": "new.c", "status": "added"},
                {"filename": "old.c", "status": "removed"},
                {
                    "filename": "renamed.c",
                    "previous_filename": "before.c",
                    "status": "renamed",
                },
            ],
        }

    def test_plan_fetches_only_patch_and_required_targeted_blobs(self):
        plan = create_retrieval_plan("1", "owner/repo", self.evidence())
        self.assertEqual(len(plan["resources"]), 7)
        urls = [resource["url"] for resource in plan["resources"]]
        self.assertTrue(all("git clone" not in url for url in urls))
        self.assertFalse(
            any(resource["side"] == "BEFORE" and resource["path"] == "new.c" for resource in plan["resources"])
        )
        self.assertFalse(
            any(resource["side"] == "AFTER" and resource["path"] == "old.c" for resource in plan["resources"])
        )

    def test_retrieval_is_deterministic_and_never_overwrites_different_bytes(self):
        plan = create_retrieval_plan("1", "owner/repo", self.evidence())
        payloads = {resource["url"]: resource["url"].encode() for resource in plan["resources"]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = retrieve_plan(plan, output, fetch=payloads.__getitem__)
            second = retrieve_plan(plan, output, fetch=payloads.__getitem__)
            self.assertEqual(first, second)
            manifest = json.loads((output / "retrieval_manifest.json").read_text())
            self.assertEqual(manifest["total_bytes"], first["total_bytes"])

            changed = dict(payloads)
            changed[plan["resources"][0]["url"]] = b"different"
            third = retrieve_plan(plan, output, fetch=changed.__getitem__)
            self.assertEqual(first, third)
            self.assertNotEqual(
                (output / plan["resources"][0]["local_path"]).read_bytes(), b"different"
            )

    def test_resource_limit_fails_closed(self):
        plan = create_retrieval_plan("1", "owner/repo", self.evidence())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                retrieve_plan(plan, Path(directory), fetch=lambda _: b"x", max_resources=1)


if __name__ == "__main__":
    unittest.main()
