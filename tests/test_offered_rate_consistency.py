from __future__ import annotations

import random
import unittest

from core.environment import Environment


class OfferedRateConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260803)
        self.environment = Environment(t_max=1, verbose=False)

    def _cross_node_assignment(self) -> list[int]:
        dnn = self.environment.ds[0]
        edge_nodes = [
            index
            for index, node in enumerate(self.environment.nodes)
            if node.level == 2
        ]
        return [edge_nodes[index % 2] for index in range(len(dnn.tasks))]

    def test_prediction_and_post_admission_actual_load_share_kernel(self) -> None:
        assignment = self._cross_node_assignment()
        runtime = max(
            self.environment.estimate_delay_from_assignment(0, assignment),
            self.environment.slot_length,
        )
        predicted = self.environment.predict_link_load_ratios(0, assignment, runtime)

        self.environment.add_running_dnn(
            0,
            assignment,
            estimated_runtime=runtime,
            remaining_slots=2,
        )

        for link in self.environment.link_nodes:
            self.assertAlmostEqual(
                predicted.get(link, 0.0),
                link.load_ratio,
                places=12,
            )

    def test_offered_rate_is_data_divided_by_active_duration(self) -> None:
        link = self.environment.link_nodes[0]
        rates = self.environment.directional_offered_rates_from_data(
            {link: 1000.0},
            active_duration_ms=100.0,
        )
        self.assertEqual(rates[link], 10.0)

    def test_zero_traffic_does_not_create_link_load(self) -> None:
        self.assertEqual(self.environment.aggregate_directional_offered_rates(), {})
        self.assertEqual(self.environment.directional_load_ratios_from_offered_rates({}), {})
        stability, feasibility = self.environment.predict_paths_state([], 100.0)
        self.assertEqual(stability, 1.0)
        self.assertEqual(feasibility, 1.0)

    def test_snapshot_offered_rates_equal_running_rate_kernel(self) -> None:
        assignment = self._cross_node_assignment()
        runtime = self.environment.slot_length
        self.environment.add_running_dnn(0, assignment, runtime, remaining_slots=1)
        expected = self.environment.aggregate_directional_offered_rates()
        snapshot = self.environment.capture_observation_snapshot()
        for index, link in enumerate(self.environment.link_nodes):
            self.assertAlmostEqual(
                snapshot.directional_offered_rates[index],
                expected.get(link, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
