from __future__ import annotations

import random
import unittest

from core.environment import Environment


class StabilityModelTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260722)
        self.environment = Environment(
            t_max=1,
            alpha_r=0.0,
            beta_r=0.0,
            alpha_l=0.0,
            beta_l=0.0,
            verbose=False,
        )

    def _assignment(self) -> list[int]:
        edges = [
            index for index, node in enumerate(self.environment.nodes)
            if node.level == 2
        ]
        return [edges[index % 3] for index in range(len(self.environment.ds[0].tasks))]

    def test_topology_stability_matches_prior_before_load(self) -> None:
        self.assertTrue(all(
            node.operational_stability == node.base_operational_stability
            for node in self.environment.nodes
        ))

    def test_operational_stability_is_monotonic(self) -> None:
        assignment = self._assignment()
        before = self.environment.count_operational_stability_by_assignment(0, assignment)
        node = self.environment.collect_used_nodes(0, assignment)[0]
        self.environment.nodes[node].operational_stability *= 0.8
        after = self.environment.count_operational_stability_by_assignment(0, assignment)
        self.assertLess(after, before)

    def test_candidate_prediction_is_side_effect_free(self) -> None:
        assignment = self._assignment()
        before = [
            (node.cpu, node.load_ratio, node.heat, node.operational_stability)
            for node in self.environment.nodes
        ]
        scores = self.environment.predict_candidate_scores_by_assignment(0, assignment)
        self.assertGreater(scores.operational_stability, 0.0)
        self.assertEqual(before, [
            (node.cpu, node.load_ratio, node.heat, node.operational_stability)
            for node in self.environment.nodes
        ])

    def test_post_admission_metrics_have_three_physical_outputs(self) -> None:
        metrics = self.environment.evaluate_post_admission_metrics(0, self._assignment())
        self.assertGreater(metrics.operational_stability, 0.0)
        self.assertGreater(metrics.delay, 0.0)
        self.assertGreater(metrics.total_energy, 0.0)

    def test_active_model_exposes_no_removed_quality_proxy(self) -> None:
        snapshot = self.environment.capture_observation_snapshot("planning")
        candidate = self.environment.predict_candidate_state(0, self._assignment(), snapshot)
        for value in (self.environment.nodes[0], snapshot, candidate):
            self.assertFalse(hasattr(value, "inference_fidelity"))
        self.assertFalse(hasattr(self.environment, "set_profile_node_fidelity"))


if __name__ == "__main__":
    unittest.main()
