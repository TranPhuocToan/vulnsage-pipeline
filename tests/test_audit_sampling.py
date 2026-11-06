import unittest

from dataset_validation.audit_sampling import stratified_sample


class AuditSamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_stratified_and_excludes_diagnostic_ids(self):
        records = [
            {"source_id": str(index), "status": status, "project": project}
            for index, (status, project) in enumerate(
                [
                    ("PASS", "linux"),
                    ("PASS", "linux"),
                    ("PASS", "linux"),
                    ("WARNING", "mozilla"),
                    ("WARNING", "mozilla"),
                    ("WARNING", "mozilla"),
                ],
                start=1,
            )
        ]
        quotas = {("PASS", "linux"): 2, ("WARNING", "mozilla"): 2}
        first = stratified_sample(records, quotas, seed=123, excluded_source_ids={"1"})
        second = stratified_sample(records, quotas, seed=123, excluded_source_ids={"1"})
        self.assertEqual(first, second)
        self.assertNotIn("1", {record["source_id"] for record in first})
        self.assertEqual(len(first), 4)

    def test_invalid_quota_fails_closed(self):
        with self.assertRaises(ValueError):
            stratified_sample([], {("PASS", "linux"): 1}, seed=1)


if __name__ == "__main__":
    unittest.main()
