from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.environment import Environment
from experiment_runner import (
    RunOutput,
    Scenario,
    _configure_search,
    _normalize_pareto_points,
    _pareto_hypervolume,
    _pareto_hypervolume_by_dnn,
    _result_row,
    _validate_existing_pareto_points_schema,
    _validate_existing_result_schema,
)
from metrics import ExperimentMetrics
from proposed import AllDNNRefactor


class FourObjectiveOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260613)
        self.scenario = Scenario(
            suite="ablation",
            name="full",
            dnn_count=2,
            population_size=20,
            iteration_limit=2,
        )
        self.environment = Environment(**self.scenario.environment_kwargs())
        self.algorithm = AllDNNRefactor(self.environment)
        _configure_search(self.algorithm, self.scenario)

    def test_all_pareto_objectives_use_maximized_satisfaction(self) -> None:
        lower_energy = {
            "inference_fidelity": 0.9,
            "operational_stability": 0.8,
            "delay_utility": 0.7,
            "total_energy": 100.0,
            "inference_fidelity_satisfaction": 0.9,
            "operational_stability_satisfaction": 0.8,
            "delay_satisfaction": 0.7,
            "energy_satisfaction": 0.8,
        }
        higher_energy = dict(
            lower_energy,
            total_energy=120.0,
            energy_satisfaction=0.6,
        )
        self.assertTrue(self.algorithm._dominates_point(lower_energy, higher_energy))
        self.assertFalse(self.algorithm._dominates_point(higher_energy, lower_energy))

    def test_assignment_uses_unified_normalized_objective_vector(self) -> None:
        assignment = self.algorithm._heuristic_assignment(
            0,
            self.algorithm._build_dnn_context(0),
            "cloud",
        )
        values = self.algorithm._evaluate_assignment(0, assignment)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values.vector))
        self.assertEqual(
            values.energy_satisfaction,
            self.environment.energy_satisfaction(0, values.total_energy),
        )
        expected_delay = max(
            0.0,
            min(
                1.0,
                (self.environment.ds[0].delay - values.estimated_delay)
                / self.environment.ds[0].delay,
            ),
        )
        self.assertEqual(values.delay_satisfaction, expected_delay)

    def test_deadline_feasibility_remains_separate_from_delay_objective(self) -> None:
        assignment = self.algorithm._heuristic_assignment(
            0,
            self.algorithm._build_dnn_context(0),
            "cloud",
        )
        deadline = float(self.environment.ds[0].delay)
        boundary = self.algorithm._evaluate_assignment(0, assignment, known_delay=deadline)
        overdue = self.algorithm._evaluate_assignment(0, assignment, known_delay=deadline + 1.0)
        self.assertTrue(boundary.deadline_feasible)
        self.assertFalse(overdue.deadline_feasible)
        self.assertEqual(boundary.delay_satisfaction, 0.0)
        self.assertEqual(overdue.delay_satisfaction, 0.0)

    def test_final_score_reuses_normalized_objectives_without_second_scaling(self) -> None:
        score = self.algorithm._score(
            0.9,
            0.8,
            0.7,
            0.6,
            0.10,
            0.20,
            0.30,
            0.40,
        )
        self.assertAlmostEqual(score, 0.9 * 0.10 + 0.8 * 0.20 + 0.7 * 0.30 + 0.6 * 0.40)

    def test_pareto_postprocessing_reuses_search_satisfaction_values(self) -> None:
        point = {
            "inference_fidelity": 0.91,
            "operational_stability": 0.82,
            "delay_utility": 0.73,
            "total_energy": 100.0,
            "inference_fidelity_satisfaction": 0.91,
            "operational_stability_satisfaction": 0.82,
            "delay_satisfaction": 0.73,
            "energy_satisfaction": 0.64,
        }
        normalized = _normalize_pareto_points([point])[0]
        self.assertEqual(normalized["inference_fidelity_norm"], 0.91)
        self.assertEqual(normalized["operational_stability_norm"], 0.82)
        self.assertEqual(normalized["delay_norm"], 0.73)
        self.assertEqual(normalized["energy_norm"], 0.64)

    def test_four_dimensional_single_point_hv(self) -> None:
        point = {
            "inference_fidelity_norm": 0.5,
            "operational_stability_norm": 0.5,
            "delay_norm": 0.5,
            "energy_norm": 0.5,
        }
        self.assertAlmostEqual(_pareto_hypervolume([point]), 0.0625)

    def test_hv_is_rejected_when_disabled(self) -> None:
        with self.assertRaises(RuntimeError):
            _pareto_hypervolume_by_dnn([], enabled=False)

    def test_non_pareto_result_has_no_pareto_fields(self) -> None:
        output = RunOutput(
            metrics=ExperimentMetrics(1.0, 0.9, 0.8, 10.0, 0, 5),
            pareto_points=[],
        )
        row = _result_row("customized", self.scenario, 0, 1, output)
        self.assertEqual(
            row["objective_semantics_version"],
            "stability_fidelity_v2_return_energy",
        )
        self.assertEqual(row["result_schema_version"], "semantic_names_v4_global")
        self.assertIn("avg_operational_stability_score", row)
        self.assertIn("avg_inference_fidelity_score", row)
        self.assertNotIn("pareto_hypervolume", row)
        self.assertNotIn("pareto_point_count", row)

    def test_legacy_three_dimensional_pareto_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw_results.csv"
            with raw_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["task_id", "suite", "pareto_hypervolume"],
                )
                writer.writeheader()
            with self.assertRaisesRegex(RuntimeError, "legacy 3D-HV"):
                _validate_existing_result_schema(raw_path, "pareto")

    def test_non_pareto_schema_rejects_hv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw_results.csv"
            with raw_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["task_id", "suite", "pareto_hypervolume"],
                )
                writer.writeheader()
            with self.assertRaisesRegex(RuntimeError, "HV-free"):
                _validate_existing_result_schema(raw_path, "ablation")

    def test_legacy_pareto_point_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            points_path = Path(directory) / "pareto_points.csv"
            with points_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["task_id", "accuracy_norm", "operation_norm", "delay_norm"],
                )
                writer.writeheader()
            with self.assertRaisesRegex(RuntimeError, "legacy 3D-HV"):
                _validate_existing_pareto_points_schema(points_path)

    def test_energy_reference_bounds_are_stable(self) -> None:
        lower, upper = self.environment.energy_reference_bounds(0)
        self.assertLess(lower, upper)
        self.assertEqual(self.environment.energy_satisfaction(0, lower), 1.0)
        self.assertEqual(self.environment.energy_satisfaction(0, upper), 0.0)

    def test_dag_initialization_reaches_diversity_target(self) -> None:
        self.algorithm.res = self.algorithm._new_population(self.algorithm.pop_size)
        context = self.algorithm._build_dnn_context(0)
        self.assertTrue(self.algorithm._initialize_customized_population(0, context))
        unique_count = self.algorithm.operator_stats["initial_unique_count"]
        self.assertGreaterEqual(unique_count, 15)
        self.assertEqual(sum(
            self.algorithm.population_roles.count(role)
            for role in self.algorithm.customized_membrane_roles
        ), self.algorithm.pop_size)

    def test_dag_crossover_block_is_successor_closed(self) -> None:
        context = self.algorithm._build_dnn_context(0)
        block = set(self.algorithm._dag_crossover_block(
            context,
            len(self.environment.ds[0].tasks),
        ))
        for task_idx in block:
            self.assertTrue(set(context.successors[task_idx]).issubset(block))

    def test_zero_crossover_probability_skips_crossover(self) -> None:
        self.algorithm.res = self.algorithm._new_population(self.algorithm.pop_size)
        context = self.algorithm._build_dnn_context(0)
        self.assertTrue(self.algorithm._initialize_customized_population(0, context))
        self.algorithm.res_pq = self.algorithm._new_population(2 * self.algorithm.pop_size)
        self.algorithm.cross_over_pm = 0.0
        self.algorithm.mutate_pm = 0.0
        self.algorithm._write_customized_child(0, 0, 1, self.algorithm.pop_size, context)
        self.assertEqual(self.algorithm.operator_stats.get("crossover_attempts", 0), 0)

    def test_failed_repair_returns_original_assignment(self) -> None:
        context = self.algorithm._build_dnn_context(0)
        assignment = self.algorithm._heuristic_assignment(0, context, "resource")
        with patch.object(self.algorithm, "_is_hierarchy_valid", return_value=False):
            result = self.algorithm._repair_assignment_result(0, assignment, context)
        self.assertFalse(result.success)
        self.assertEqual(result.assignment, assignment)
        self.assertEqual(result.reason, "no_structurally_feasible_repair")


if __name__ == "__main__":
    unittest.main()
