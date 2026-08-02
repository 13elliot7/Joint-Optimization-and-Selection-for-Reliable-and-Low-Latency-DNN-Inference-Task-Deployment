from __future__ import annotations

import math
import random
import unittest

from core.environment import Environment
from core.topology import TopologyConfig, _add_bidirectional_link
from models import Node


class PhysicalLinkStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260719)
        self.environment = Environment(t_max=1, verbose=False)

    def _single_hop_round_trip_assignment(self) -> tuple[int, list[int]]:
        dnn = self.environment.ds[0]
        origin = dnn.initiateNode
        outgoing = [
            link
            for link in self.environment.link_nodes
            if link.s_node is self.environment.nodes[origin]
        ]
        self.assertTrue(outgoing)
        target = self.environment.nodes.index(outgoing[0].e_node)
        return target, [target] * len(dnn.tasks)

    def test_bidirectional_arcs_share_one_physical_state(self) -> None:
        for physical_link in self.environment.physical_links:
            directed = self.environment._directed_links_by_physical_id[
                physical_link.physical_link_id
            ]
            self.assertEqual(len(directed), 2)
            self.assertIs(directed[0].physical_state, physical_link)
            self.assertIs(directed[1].physical_state, physical_link)
            self.assertIs(directed[0].physical_state, directed[1].physical_state)

    def test_round_trip_counts_physical_link_stability_once(self) -> None:
        _, assignment = self._single_hop_round_trip_assignment()
        directed = self.environment.collect_used_directed_links(0, assignment)
        physical = self.environment.collect_used_physical_links(0, assignment)

        self.assertEqual(len(directed), 2)
        self.assertEqual(len(physical), 1)
        physical[0].transmission_stability = 0.98
        self.assertAlmostEqual(
            self.environment.count_raw_link_product_by_assignment(
                0,
                assignment,
            ),
            0.98,
        )

    def test_candidate_prediction_uses_one_physical_failure_domain(self) -> None:
        target, assignment = self._single_hop_round_trip_assignment()
        dnn = self.environment.ds[0]
        dnn.startFloat = 1.0
        dnn.backFloat = 1.0
        physical = self.environment.collect_used_physical_links(0, assignment)
        self.assertEqual(len(physical), 1)
        physical[0].base_transmission_stability = 0.98
        physical[0].transmission_stability = 0.98

        runtime = 100.0
        scores = self.environment.predict_candidate_scores_by_assignment(
            0,
            assignment,
            estimated_runtime=runtime,
        )
        node = self.environment.nodes[target]
        added_cpu = sum(task.cpu_need for task in dnn.tasks)
        node_load = added_cpu / node.max_cpu
        node_heat = (1.0 - self.environment.lambda_h) * node_load
        expected_node = node.base_operational_stability * math.exp(
            -self.environment.alpha_r * node_load
            - self.environment.beta_r * node_heat,
        )
        expected_node = min(
            node.base_operational_stability,
            max(self.environment.o_min, expected_node),
        )
        directed = self.environment.collect_used_directed_links(0, assignment)
        directional_loads = self.environment.predict_link_load_ratios(
            0,
            assignment,
            runtime,
        )
        expected_load = max(directional_loads[link] for link in directed)
        expected_heat = (1.0 - self.environment.lambda_g) * expected_load
        expected_link = 0.98 * math.exp(
            -self.environment.alpha_l * expected_load
            - self.environment.beta_l * expected_heat
        )
        weight_sum = (
            self.environment.node_stability_weight
            + self.environment.link_stability_weight
        )
        expected_oss = (
            expected_node ** (self.environment.node_stability_weight / weight_sum)
            * expected_link ** (self.environment.link_stability_weight / weight_sum)
        )
        self.assertAlmostEqual(scores.operational_stability, expected_oss)

    def test_multi_path_prediction_counts_shared_physical_link_once(self) -> None:
        target, _ = self._single_hop_round_trip_assignment()
        origin = self.environment.ds[0].initiateNode
        path = self.environment._path_links[(origin, target)]
        self.assertEqual(len(path), 1)
        physical = path[0].physical_state
        assert physical is not None
        physical.base_transmission_stability = 0.98
        physical.transmission_stability = 0.98

        stability, _ = self.environment.predict_paths_state(
            [(origin, target, 0.0), (target, origin, 0.0)],
            estimated_runtime=100.0,
        )
        self.assertEqual(stability, 1.0)

        stability, _ = self.environment.predict_paths_state(
            [(origin, target, 1.0), (target, origin, 1.0)],
            estimated_runtime=100.0,
        )
        directional_load = 1.0 / 100.0 / path[0].band_width
        predicted_heat = (1.0 - self.environment.lambda_g) * directional_load
        expected = 0.98 * math.exp(
            -self.environment.alpha_l * directional_load
            - self.environment.beta_l * predicted_heat
        )
        self.assertAlmostEqual(stability, expected)

    def test_physical_load_uses_maximum_directional_load(self) -> None:
        physical = self.environment.physical_links[0]
        directed = self.environment._directed_links_by_physical_id[
            physical.physical_link_id
        ]
        directed[0].load_ratio = 0.8
        directed[1].load_ratio = 0.2
        loads = self.environment._physical_load_ratios_from_directional(
            physical_links=[physical]
        )
        self.assertEqual(loads[physical], 0.8)

    def test_parallel_connections_keep_distinct_physical_ids(self) -> None:
        node_a = Node(1, 1, 2, 0.99, 0.99, 1.0)
        node_b = Node(1, 1, 2, 0.99, 0.99, 1.0)
        links = []
        first = _add_bidirectional_link(links, node_a, node_b, 100)
        second = _add_bidirectional_link(links, node_a, node_b, 100)

        self.assertEqual(first[0].physical_link_id, first[1].physical_link_id)
        self.assertEqual(second[0].physical_link_id, second[1].physical_link_id)
        self.assertNotEqual(first[0].physical_link_id, second[0].physical_link_id)

    def test_two_edge_parametric_ring_has_no_duplicate_connection(self) -> None:
        random.seed(20260719)
        environment = Environment(
            t_max=1,
            topology_config=TopologyConfig(
                cloud_count=1,
                edge_count=2,
                user_count=2,
                edge_link_factor=1.0,
            ),
            verbose=False,
        )
        edge_states = [
            physical
            for physical in environment.physical_links
            if physical.node_a.level == physical.node_b.level == 2
        ]
        self.assertEqual(len(edge_states), 1)


if __name__ == "__main__":
    unittest.main()
