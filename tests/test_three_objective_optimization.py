from __future__ import annotations

import random
import unittest

from core.environment import Environment
from experiment_runner import _normalize_pareto_points, _pareto_hypervolume
from proposed import AllDNNRefactor, CustomizedSearchConfig


class ThreeObjectiveOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260809)
        self.environment = Environment(t_max=1, verbose=False)
        self.algorithm = AllDNNRefactor(self.environment)

    def test_objective_vector_is_oss_delay_energy(self) -> None:
        assignment = self.algorithm._heuristic_assignment(
            0, self.algorithm._build_dnn_context(0), "cloud"
        )
        values = self.algorithm._evaluate_assignment(0, assignment)
        self.assertEqual(len(values.vector), 3)
        self.assertEqual(
            values.vector,
            (
                values.operational_stability,
                values.delay_satisfaction,
                values.energy_satisfaction,
            ),
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values.vector))

    def test_three_objective_dominance(self) -> None:
        better = {
            "operational_stability_satisfaction": 0.9,
            "delay_satisfaction": 0.8,
            "energy_satisfaction": 0.7,
        }
        worse = dict(better, energy_satisfaction=0.6)
        self.assertTrue(self.algorithm._dominates_point(better, worse))
        self.assertFalse(self.algorithm._dominates_point(worse, better))

    def test_three_dimensional_single_point_hypervolume(self) -> None:
        point = {
            "operational_stability_norm": 0.5,
            "delay_norm": 0.5,
            "energy_norm": 0.5,
        }
        self.assertAlmostEqual(_pareto_hypervolume([point]), 0.125)

    def test_normalization_reuses_search_satisfactions(self) -> None:
        point = {
            "operational_stability_satisfaction": 0.82,
            "delay_satisfaction": 0.73,
            "energy_satisfaction": 0.64,
        }
        normalized = _normalize_pareto_points([point])[0]
        self.assertEqual(normalized["operational_stability_norm"], 0.82)
        self.assertEqual(normalized["delay_norm"], 0.73)
        self.assertEqual(normalized["energy_norm"], 0.64)

    def test_customized_archive_exposes_three_objectives(self) -> None:
        result = self.algorithm.generate_customized_candidates(
            dnn_index=0,
            snapshot=self.environment.capture_observation_snapshot("planning"),
            search_config=CustomizedSearchConfig(
                population_size=4,
                generations=1,
                archive_capacity=8,
                random_seed=7,
            ),
        )
        self.assertTrue(all(len(candidate.objectives) == 3 for candidate in result.candidates))


if __name__ == "__main__":
    unittest.main()
