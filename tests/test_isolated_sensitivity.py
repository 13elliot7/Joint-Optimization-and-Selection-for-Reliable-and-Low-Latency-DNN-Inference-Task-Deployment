from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

from core.environment import Environment
from experiment_runner import Scenario, build_parser, build_scenarios, run_once


class IsolatedSensitivityTests(unittest.TestCase):
    def _scenario(self, **changes) -> Scenario:
        return dataclasses.replace(
            Scenario(
                suite="sensitivity",
                name="smoke",
                dnn_count=1,
                population_size=4,
                iteration_limit=2,
                archive_capacity=8,
                sensitivity_instance_count=2,
            ),
            **changes,
        )

    def test_search_never_commits_or_advances_environment(self) -> None:
        with patch.object(
            Environment,
            "advance_time_slot",
            side_effect=AssertionError("sensitivity must not advance time"),
        ), patch.object(
            Environment,
            "commit_assignment",
            side_effect=AssertionError("sensitivity must not commit resources"),
        ):
            output = run_once("customized", self._scenario(), 20260719)
        self.assertEqual(
            output.diagnostics["workload_mode"],
            "isolated_single_dnn_instance_bank",
        )
        self.assertEqual(output.diagnostics["quality_instance_count"], 2)

    def test_fixed_instance_bank_is_reproducible_except_wall_clock(self) -> None:
        first = run_once("customized", self._scenario(), 20260719)
        second = run_once("customized", self._scenario(), 20260719)
        first_metrics = dataclasses.asdict(first.metrics)
        second_metrics = dataclasses.asdict(second.metrics)
        first_metrics.pop("runtime_ms")
        second_metrics.pop("runtime_ms")
        self.assertEqual(first_metrics, second_metrics)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_sensitivity_scenarios_are_single_request_instance_banks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--suite", "sensitivity"])
        args.population_sizes = [20]
        args.iteration_limits = [80]
        args.mutation_probabilities = [0.25]
        args.crossover_probabilities = [0.5]
        args.archive_capacities = [64]
        args.sensitivity_instances = 7
        scenarios = build_scenarios(args)
        self.assertEqual(len(scenarios), 5)
        self.assertEqual({scenario.dnn_count for scenario in scenarios}, {1})
        self.assertEqual(
            {scenario.sensitivity_instance_count for scenario in scenarios},
            {7},
        )


if __name__ == "__main__":
    unittest.main()
