from __future__ import annotations

import random
import unittest

from core.environment import Environment


class CandidateStateSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260804)
        self.environment = Environment(t_max=1, verbose=False)

    def _assignment(self) -> list[int]:
        dnn = self.environment.ds[0]
        edges = [
            index
            for index, node in enumerate(self.environment.nodes)
            if node.level == 2
        ]
        return [edges[index % 3] for index in range(len(dnn.tasks))]

    def _environment_state(self) -> tuple:
        return (
            self.environment.current_slot,
            self.environment.environment_state_version,
            tuple(
                (
                    node.cpu,
                    node.load_ratio,
                    node.heat,
                    node.operational_stability,
                )
                for node in self.environment.nodes
            ),
            tuple(
                (
                    link.load_ratio,
                    link.effective_band_width,
                )
                for link in self.environment.link_nodes
            ),
            tuple(
                (
                    link.load_ratio,
                    link.heat,
                    link.transmission_stability,
                )
                for link in self.environment.physical_links
            ),
            tuple(self.environment.running_dnns),
        )

    def test_candidate_prediction_is_side_effect_free(self) -> None:
        before = self._environment_state()
        observation = self.environment.capture_observation_snapshot("planning")
        state_after_observation = self._environment_state()
        candidate = self.environment.predict_candidate_state(
            0,
            self._assignment(),
            observation,
        )
        self.assertEqual(self._environment_state(), state_after_observation)
        self.assertEqual(before, state_after_observation)
        self.assertGreater(candidate.estimated_delay_ms, 0.0)
        self.assertGreater(candidate.total_energy, 0.0)

    def test_candidate_contains_aligned_link_vectors(self) -> None:
        candidate = self.environment.predict_candidate_state(0, self._assignment())
        self.assertEqual(
            len(candidate.directional_offered_rates),
            len(self.environment.link_nodes),
        )
        self.assertEqual(
            len(candidate.directional_link_loads),
            len(self.environment.link_nodes),
        )
        self.assertEqual(
            len(candidate.predicted_effective_bandwidths),
            len(self.environment.link_nodes),
        )
        self.assertEqual(
            len(candidate.physical_link_loads),
            len(self.environment.physical_links),
        )

    def test_explicit_snapshot_is_stable_after_environment_changes(self) -> None:
        assignment = self._assignment()
        observation = self.environment.capture_observation_snapshot("planning")
        before = self.environment.predict_candidate_state(0, assignment, observation)
        self.environment.add_running_dnn(
            0,
            assignment,
            estimated_runtime=self.environment.slot_length,
            remaining_slots=1,
        )
        after = self.environment.predict_candidate_state(0, assignment, observation)
        self.assertEqual(before, after)

    def test_fixed_point_converges_or_reports_conservative_fallback(self) -> None:
        candidate = self.environment.predict_candidate_state(0, self._assignment())
        self.assertGreaterEqual(candidate.fixed_point_iterations, 1)
        self.assertLessEqual(
            candidate.fixed_point_iterations,
            self.environment.candidate_fixed_point_max_iterations,
        )
        self.assertGreaterEqual(candidate.fixed_point_relative_residual, 0.0)
        if candidate.fixed_point_converged:
            self.assertLessEqual(
                candidate.fixed_point_relative_residual,
                self.environment.candidate_fixed_point_relative_tolerance,
            )
        if not candidate.fixed_point_converged:
            current_delay = self.environment.estimate_delay_from_assignment(0, self._assignment())
            self.assertGreaterEqual(candidate.estimated_delay_ms, current_delay)

    def test_one_iteration_forces_explicit_nonconvergence(self) -> None:
        environment = Environment(
            t_max=1,
            candidate_fixed_point_max_iterations=1,
            candidate_fixed_point_relative_tolerance=1e-15,
            verbose=False,
        )
        dnn = environment.ds[0]
        edges = [
            index for index, node in enumerate(environment.nodes) if node.level == 2
        ]
        assignment = [edges[index % 2] for index in range(len(dnn.tasks))]
        candidate = environment.predict_candidate_state(0, assignment)
        self.assertFalse(candidate.fixed_point_converged)
        self.assertEqual(candidate.fixed_point_iterations, 1)
        self.assertGreater(
            candidate.fixed_point_relative_residual,
            environment.candidate_fixed_point_relative_tolerance,
        )
        self.assertGreater(candidate.estimated_delay_ms, 0.0)

    def test_offline_node_is_a_hard_constraint(self) -> None:
        assignment = self._assignment()
        used = assignment[0]
        if used >= len(self.environment.up):
            self.skipTest("selected edge is configured as always-on")
        self.environment.up[used] = False
        candidate = self.environment.predict_candidate_state(0, assignment)
        self.assertIn("node_offline", candidate.constraint_violations)
        self.assertFalse(candidate.is_strictly_feasible)

    def test_delay_with_explicit_current_bandwidth_matches_legacy_delay(self) -> None:
        assignment = self._assignment()
        bandwidths = {
            link: self.environment.get_effective_bandwidth(link)
            for link in self.environment.link_nodes
        }
        explicit = self.environment.estimate_delay_from_assignment_with_bandwidths(
            0,
            assignment,
            bandwidths,
        )
        legacy = self.environment.estimate_delay_from_assignment(0, assignment)
        self.assertAlmostEqual(explicit, legacy)


if __name__ == "__main__":
    unittest.main()
