import unittest

from scripts.build_pilot_unblock_reports import MISMATCHES, mismatch_category


class PilotUnblockReportTests(unittest.TestCase):
    def test_all_eighteen_mismatches_have_deterministic_explanation(self):
        self.assertEqual(len(MISMATCHES), 18)
        categories = {source_id: mismatch_category(values) for source_id, values in MISMATCHES.items()}
        self.assertEqual(categories["17623"], "COUNTS_EQUAL_BUT_CHANGED_LINE_OR_HUNK_COVERAGE_DIFFERS")
        self.assertEqual(categories["18602"], "ADD_DELETE_DISTRIBUTION_DIFFERS")
        self.assertEqual(categories["17080"], "SNIPPET_BOUNDARY_DIFFLIB_ADDS_DERIVED_CHANGES")
        self.assertEqual(categories["16892"], "CSV_SNIPPET_OMITS_COMMIT_CHANGED_LINES")


if __name__ == "__main__":
    unittest.main()
