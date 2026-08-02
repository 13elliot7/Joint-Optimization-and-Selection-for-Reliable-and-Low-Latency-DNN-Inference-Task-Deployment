from __future__ import annotations

import random
import unittest

from core.environment import Environment


class StabilityFidelityModelTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260722)
        self.environment = Environment(
            t_max=1,
            alpha_r=0.0,
            beta_r=0.0,
            alpha_a=0.0,
            beta_a=0.0,
            alpha_l=0.0,
            beta_l=0.0,
            verbose=False,
        )

    def _multi_node_assignment(self) -> list[int]:
        task_count = len(self.environment.ds[0].tasks)
        edge_indices = [
            index
            for index, node in enumerate(self.environment.nodes)
            if node.level == 2
        ]
        return [edge_indices[index % 3] for index in range(task_count)]

    def test_default_topology_current_scores_match_priors(self) -> None:
        for node in self.environment.nodes:
            self.assertEqual(
                node.operational_stability,
                node.base_operational_stability,
            )
            self.assertEqual(
                node.inference_fidelity,
                node.base_inference_fidelity,
            )

    def test_semantic_scores_are_mutable(self) -> None:
        node = self.environment.nodes[0]
        node.operational_stability = 0.91
        node.inference_fidelity = 0.92
        self.assertEqual(node.operational_stability, 0.91)
        self.assertEqual(node.inference_fidelity, 0.92)

    def test_hierarchical_geometric_score_is_size_invariant(self) -> None:
        for node in self.environment.nodes:
            node.operational_stability = 0.9
        for physical_link in self.environment.physical_links:
            physical_link.transmission_stability = 0.9

        dnn = self.environment.ds[0]
        single_node = [dnn.initiateNode] * len(dnn.tasks)
        multi_node = self._multi_node_assignment()
        single_oss = self.environment.count_operational_stability_by_assignment(
            0,
            single_node,
        )
        multi_oss = self.environment.count_operational_stability_by_assignment(
            0,
            multi_node,
        )
        self.assertAlmostEqual(single_oss, 0.9)
        self.assertAlmostEqual(multi_oss, 0.9)
        self.assertLess(
            self.environment.count_raw_joint_product_by_assignment(0, multi_node),
            multi_oss,
        )

    def test_operational_stability_is_monotonic(self) -> None:
        assignment = self._multi_node_assignment()
        before = self.environment.count_operational_stability_by_assignment(0, assignment)
        degraded_node = self.environment.collect_used_nodes(0, assignment)[0]
        self.environment.nodes[degraded_node].operational_stability *= 0.8
        after = self.environment.count_operational_stability_by_assignment(0, assignment)
        self.assertLess(after, before)

    def test_inference_fidelity_uses_flops_weights(self) -> None:
        dnn = self.environment.ds[0]
        first_node, second_node = self._multi_node_assignment()[:2]
        assignment = [first_node] * len(dnn.tasks)
        assignment[-1] = second_node
        self.environment.nodes[first_node].inference_fidelity = 0.8
        self.environment.nodes[second_node].inference_fidelity = 1.0
        total_flops = sum(task.float_num for task in dnn.tasks)
        second_weight = dnn.tasks[-1].float_num / total_flops
        expected = 0.8 ** (1.0 - second_weight)
        self.assertAlmostEqual(
            self.environment.count_inference_fidelity_by_assignment(0, assignment),
            expected,
        )

    def test_candidate_prediction_does_not_mutate_environment(self) -> None:
        assignment = self._multi_node_assignment()
        node_state = [
            (
                node.cpu,
                node.load_ratio,
                node.heat,
                node.operational_stability,
                node.inference_fidelity,
            )
            for node in self.environment.nodes
        ]
        link_state = [
            (link.load_ratio, link.heat, link.transmission_stability)
            for link in self.environment.physical_links
        ]
        scores = self.environment.predict_candidate_scores_by_assignment(0, assignment)
        self.assertGreater(scores.operational_stability, 0.0)
        self.assertGreater(scores.inference_fidelity, 0.0)
        self.assertEqual(
            node_state,
            [
                (
                    node.cpu,
                    node.load_ratio,
                    node.heat,
                    node.operational_stability,
                    node.inference_fidelity,
                )
                for node in self.environment.nodes
            ],
        )
        self.assertEqual(
            link_state,
            [
                (link.load_ratio, link.heat, link.transmission_stability)
                for link in self.environment.physical_links
            ],
        )

    def test_post_admission_metrics_use_named_fields(self) -> None:
        assignment = self._multi_node_assignment()
        metrics = self.environment.evaluate_post_admission_metrics(0, assignment)
        self.assertGreater(metrics.operational_stability, 0.0)
        self.assertGreater(metrics.inference_fidelity, 0.0)
        self.assertGreater(metrics.delay, 0.0)
        self.assertGreater(metrics.total_energy, 0.0)

    def test_link_energy_includes_result_return(self) -> None:
        dnn = self.environment.ds[0]
        origin = dnn.initiateNode
        outgoing = [
            link
            for link in self.environment.link_nodes
            if link.s_node is self.environment.nodes[origin]
        ]
        self.assertTrue(outgoing)
        target = self.environment.nodes.index(outgoing[0].e_node)
        assignment = [target] * len(dnn.tasks)
        dnn.startFloat = 0.0
        for dnn_link in dnn.links:
            dnn_link.float_tran = 0
        dnn.backFloat = 128.0

        return_path = self.environment._path_links[(target, origin)]
        expected_return_energy = sum(
            link.energy_per_mb * dnn.backFloat
            for link in return_path
        )
        self.assertGreater(expected_return_energy, 0.0)
        self.assertAlmostEqual(
            self.environment.count_link_energy_by_assignment(0, assignment),
            expected_return_energy,
        )

    def test_energy_reference_upper_bound_includes_result_return(self) -> None:
        dnn = self.environment.ds[0]
        dnn.backFloat = 0.0
        _, upper_without_return = self.environment.energy_reference_bounds(0)
        dnn.backFloat = 1024.0
        _, upper_with_return = self.environment.energy_reference_bounds(0)
        self.assertGreater(upper_with_return, upper_without_return)

    def test_invalid_quality_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Environment(t_max=1, node_stability_weight=0.0, link_stability_weight=0.0)

    def test_log_domain_geometric_mean_does_not_underflow(self) -> None:
        score = self.environment._geometric_mean([0.9] * 10000)
        self.assertAlmostEqual(score, 0.9)


if __name__ == "__main__":
    unittest.main()
