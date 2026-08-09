from __future__ import annotations

import dataclasses
import math
import random
import unittest

from core.environment import Environment
from semantics import PERIODIC_SEMANTIC_VERSIONS


class ProfiledAvailabilityModelTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260805)
        self.environment = Environment(t_max=1, verbose=False)

    def _assignment(self, environment: Environment | None = None) -> list[int]:
        environment = environment or self.environment
        dnn = environment.ds[0]
        edge = next(
            index for index, node in enumerate(environment.nodes) if node.level == 2
        )
        return [edge for _ in dnn.tasks]

    @staticmethod
    def _replace_used_histories(snapshot, used_nodes: set[int], values: tuple[bool, ...]):
        histories = list(snapshot.node_availability_history)
        for node_index in used_nodes:
            histories[node_index] = values
        return dataclasses.replace(snapshot, node_availability_history=tuple(histories))

    def test_every_node_has_an_explicit_availability_mode(self) -> None:
        snapshot = self.environment.capture_observation_snapshot()
        self.assertEqual(len(snapshot.node_availability_modes), len(self.environment.nodes))
        self.assertTrue(
            all(mode in {"stochastic", "always_on"} for mode in snapshot.node_availability_modes)
        )
        self.assertTrue(
            all(
                mode == "always_on"
                for mode in snapshot.node_availability_modes[
                    self.environment.availability_node_count :
                ]
            )
        )

    def test_availability_advances_only_at_slot_boundary(self) -> None:
        node_index = 0
        self.environment.up[node_index] = True
        self.environment.remaining_time[node_index] = 0
        history_length = len(self.environment.rj_history[node_index])
        first = self.environment.generate_availability()
        second = self.environment.generate_availability()
        self.assertEqual(first, second)
        self.assertTrue(self.environment.up[node_index])
        self.assertEqual(len(self.environment.rj_history[node_index]), history_length)

        self.environment.advance_time_slot()
        self.assertFalse(self.environment.up[node_index])
        self.assertEqual(len(self.environment.rj_history[node_index]), history_length + 1)

    def test_beta_window_probability_is_nonincreasing_with_horizon(self) -> None:
        node_index = self._assignment()[0]
        snapshot = self.environment.capture_observation_snapshot()
        snapshot = self._replace_used_histories(
            snapshot,
            {node_index},
            (True, True, False, True, True, True, False, True),
        )
        short, _ = self.environment.predict_node_continuous_availability(
            node_index, 1, snapshot
        )
        long, confidence = self.environment.predict_node_continuous_availability(
            node_index, 4, snapshot
        )
        self.assertLessEqual(long, short)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_always_on_node_has_unit_continuous_availability(self) -> None:
        environment = Environment(t_max=1, availability_node_count=0, verbose=False)
        snapshot = environment.capture_observation_snapshot()
        probability, confidence = environment.predict_node_continuous_availability(
            0, 100, snapshot
        )
        self.assertEqual((probability, confidence), (1.0, 1.0))

    def test_aggregate_confidence_is_clamped_at_floating_point_boundary(self) -> None:
        environment = Environment(t_max=1, availability_node_count=0, verbose=False)
        dnn = environment.ds[0]
        float_nums = (0.2, 82.1749838769326, 82.1749838769326)
        for task_index, task in enumerate(dnn.tasks):
            task.float_num = float_nums[task_index] if task_index < 3 else 0.0
        assignment = [0, 1, 2] + [0] * (len(dnn.tasks) - 3)

        probability, confidence = environment.aggregate_candidate_availability(
            0,
            assignment,
            1,
            environment.capture_observation_snapshot(),
        )

        self.assertEqual(probability, 1.0)
        self.assertEqual(confidence, 1.0)

    def test_periodic_oss_uses_observed_availability_history(self) -> None:
        assignment = self._assignment()
        used_nodes = set(assignment)
        base = self.environment.capture_observation_snapshot()
        healthy = self._replace_used_histories(base, used_nodes, (True,) * 100)
        degraded = self._replace_used_histories(
            base, used_nodes, (True, False) * 50
        )
        periodic_healthy = self.environment.predict_candidate_state(0, assignment, healthy)
        periodic_degraded = self.environment.predict_candidate_state(0, assignment, degraded)
        self.assertGreater(
            periodic_healthy.node_availability_score,
            periodic_degraded.node_availability_score,
        )
        self.assertGreater(
            periodic_healthy.operational_stability,
            periodic_degraded.operational_stability,
        )

    def test_candidate_availability_horizon_matches_runtime_slots(self) -> None:
        assignment = self._assignment()
        candidate = self.environment.predict_candidate_state(0, assignment)
        self.assertEqual(
            candidate.availability_horizon_slots,
            max(1, math.ceil(candidate.estimated_delay_ms / self.environment.slot_length)),
        )


if __name__ == "__main__":
    unittest.main()
