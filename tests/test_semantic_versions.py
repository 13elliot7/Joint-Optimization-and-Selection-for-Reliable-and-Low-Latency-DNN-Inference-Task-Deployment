from __future__ import annotations

import unittest

from semantics import LEGACY_SEMANTIC_VERSIONS, PERIODIC_SEMANTIC_VERSIONS


class SemanticVersionTests(unittest.TestCase):
    def test_semantic_pairs_are_stable_and_complete(self) -> None:
        pairs = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        self.assertEqual(len(pairs), len(dict(pairs)))
        self.assertIn(("planning_semantics_version", "periodic_repository_v1"), pairs)
        self.assertIn(("joint_budget_control_version", "discrete_mpc_v1"), pairs)
        self.assertIn(("joint_action_space_version", "grid_tkb_v1"), pairs)
        self.assertIn(
            ("online_anytime_revalidation_version", "budgeted_fixed_point_v1"),
            pairs,
        )
        self.assertIn(("control_objective_version", "goodput_cost_v1"), pairs)
        self.assertIn(("candidate_prediction_version", "unified_fixed_point_v3"), pairs)
        self.assertIn(("result_schema_version", "periodic_three_objective_v3"), pairs)

    def test_legacy_and_periodic_results_cannot_share_objective_version(self) -> None:
        self.assertNotEqual(
            LEGACY_SEMANTIC_VERSIONS.objective,
            PERIODIC_SEMANTIC_VERSIONS.objective,
        )


if __name__ == "__main__":
    unittest.main()
