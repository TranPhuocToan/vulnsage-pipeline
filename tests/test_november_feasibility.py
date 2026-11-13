import unittest

from scripts.november_feasibility import MANDATORY_GATES, PASS, row_is_qualified


class NovemberFeasibilityBoundaryTests(unittest.TestCase):
    def passing_row(self):
        return {gate: PASS for gate in MANDATORY_GATES}

    def test_all_mandatory_gates_are_required(self):
        row = self.passing_row()
        self.assertTrue(row_is_qualified(row))
        for gate in MANDATORY_GATES:
            changed = dict(row)
            changed[gate] = "PENDING"
            self.assertFalse(row_is_qualified(changed), gate)

    def test_pending_ground_truth_cannot_be_promoted(self):
        row = self.passing_row()
        row["ground_truth_review_gate"] = "PENDING:PENDING"
        self.assertFalse(row_is_qualified(row))

    def test_engineering_smoke_scope_cannot_authorize_pilot(self):
        row = self.passing_row()
        row["context_policy_authorization_gate"] = "BLOCKED_ENGINEERING_SMOKE_SCOPE_ONLY"
        self.assertFalse(row_is_qualified(row))

    def test_static_fit_does_not_prove_runtime_synthesis_fit(self):
        row = self.passing_row()
        row["context_p2_gate"] = "PARTIAL_RUNTIME_SYNTHESIS_GUARD_REQUIRED"
        self.assertFalse(row_is_qualified(row))

    def test_missing_gate_fails_closed(self):
        row = self.passing_row()
        del row["upstream_mapping_gate"]
        self.assertFalse(row_is_qualified(row))


if __name__ == "__main__":
    unittest.main()
