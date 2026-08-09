from __future__ import annotations

import argparse
import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.environment import Environment
from core.topology import TopologyConfig
from metrics import PeriodicExperimentMetrics
from periodic_experiment_runner import (
    PERIODIC_SUITES,
    PeriodicScenario,
    build_parser,
    build_scenarios,
    run_scenarios,
)
from planning.periodic_planner import PublicationCandidateAudit
from semantics import PERIODIC_SEMANTIC_VERSIONS


def metrics(seed: int = 7) -> PeriodicExperimentMetrics:
    return PeriodicExperimentMetrics(
        arrival_slots=2,
        elapsed_slots=2,
        arrived_requests=1,
        accepted_requests=1,
        rejected_requests=0,
        completed_requests=1,
        runtime_failed_requests=0,
        deadline_missed_requests=0,
        unfinished_requests=0,
        goodput_utility=1.0,
        avg_estimated_delay_ms=10.0,
        avg_operational_stability_score=0.9,
        avg_total_energy=2.0,
        plan_direct_count=1,
        plan_repaired_count=0,
        fast_fallback_count=0,
        planning_jobs_started=1,
        planning_jobs_published=1,
        planning_jobs_retained=0,
        total_planning_runtime_ms=1.0,
        total_online_runtime_ms=0.5,
        avg_online_runtime_ms=0.5,
        p95_online_runtime_ms=0.5,
        control_decision_count=1,
        control_outcome_count=0,
        control_fallback_count=0,
        trace_seed=seed,
        bootstrap_mode="prewarm",
        semantic_versions=PERIODIC_SEMANTIC_VERSIONS.as_pairs(),
    )


class PeriodicExperimentRunnerTests(unittest.TestCase):
    def test_parser_exposes_all_periodic_suites(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--suite", "all"])
        self.assertEqual(args.suite, "all")
        self.assertEqual(args.candidate_fixed_point_max_iterations, 20)
        self.assertEqual(args.candidate_fixed_point_relative_tolerance, 1e-4)
        self.assertEqual(args.candidate_fixed_point_damping, 0.5)
        self.assertEqual(
            PERIODIC_SUITES,
            (
                "load",
                "algorithm_baseline",
                "robustness",
                "scale",
            ),
        )

    def test_algorithm_baselines_share_poisson_environment_identity(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--suite", "algorithm_baseline", "--quick"])
        from periodic_experiment_runner import _normalize_args

        _normalize_args(parser, args)
        scenarios = build_scenarios(args, "algorithm_baseline")
        self.assertEqual(
            {scenario.algorithm for scenario in scenarios},
            {
                "periodic_customized",
                "periodic_lightweight",
                "per_request_customized",
                "random",
                "sa",
                "localfirst",
                "max_resource_fast",
                "rtbl",
            },
        )
        self.assertEqual({scenario.workload_mode for scenario in scenarios}, {"poisson_profile_catalog"})
        self.assertEqual({scenario.lambda_per_slot for scenario in scenarios}, {1.0})
        self.assertIn(
            "periodic_customized_cold_start",
            {scenario.comparison_method for scenario in scenarios},
        )

    def test_load_compares_only_three_selected_methods(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--suite", "load", "--lambda-values", "0.5,1"])
        from periodic_experiment_runner import _normalize_args

        _normalize_args(parser, args)
        scenarios = build_scenarios(args, "load")
        self.assertEqual(
            {scenario.comparison_method for scenario in scenarios},
            {"periodic_customized", "periodic_lightweight", "per_request_customized"},
        )
        self.assertEqual(len(scenarios), 6)

    def test_robustness_has_three_environment_groups(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--suite", "robustness"])
        from periodic_experiment_runner import _normalize_args

        _normalize_args(parser, args)
        scenarios = build_scenarios(args, "robustness")
        self.assertEqual(
            {scenario.comparison_group for scenario in scenarios},
            {"stable", "failure_prone", "fast_drift"},
        )

    def test_batch_output_is_resumable_and_writes_summary(self) -> None:
        scenario = PeriodicScenario(
            suite="load",
            name="lambda_1",
            lambda_per_slot=1.0,
            slot_count=2,
            profile_count=1,
            origin_group_count=1,
            fixed_total_pool_budget=2,
            total_pool_budgets=(2,),
            online_timing_mode="deterministic",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def fake_run_once(scenario, seed, audit_sink=None, event_sink=None):
                if audit_sink is not None:
                    audit_sink(
                        (
                            PublicationCandidateAudit(
                                job_id=1,
                                plan_id="plan-1",
                                trigger_reason="prewarm",
                                profile_id="profile-1",
                                origin_group=0,
                                generator_mode="customized",
                                generator_version="customized_periodic_pool_v1",
                                search_seed=17,
                                source_generation=1,
                                publication_slot=0,
                                source_snapshot_version=1,
                                publication_snapshot_version=2,
                                source_environment_state_version=0,
                                publication_environment_state_version=0,
                                accepted=False,
                                primary_reason="node_offline",
                                reasons=("node_offline",),
                                offline_nodes=(3,),
                            ),
                        )
                    )
                if event_sink is not None:
                    event_sink(
                        {
                            "request_events": ({"request_id": 1, "path": "plan_direct"},),
                            "planning_jobs": (),
                            "control_actions": (),
                        }
                    )
                return metrics()

            with patch(
                "periodic_experiment_runner.run_once",
                side_effect=fake_run_once,
            ) as mocked:
                run_scenarios((scenario,), 1, 7, output)
                run_scenarios((scenario,), 1, 7, output)
            self.assertEqual(mocked.call_count, 1)
            with (output / "raw_results.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["objective_semantics_version"],
                PERIODIC_SEMANTIC_VERSIONS.objective,
            )
            self.assertTrue((output / "summary.csv").exists())
            with (output / "planning_candidate_audits.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                audit_rows = list(csv.DictReader(stream))
            self.assertEqual(len(audit_rows), 1)
            self.assertEqual(audit_rows[0]["primary_reason"], "node_offline")
            self.assertEqual(audit_rows[0]["offline_nodes"], "[3]")
            self.assertTrue((output / "request_events.jsonl").exists())

    def test_availability_seed_is_independent_from_global_algorithm_rng(self) -> None:
        config = TopologyConfig(cloud_count=1, edge_count=1, user_count=1)
        random.seed(9)
        first = Environment(
            t_max=1,
            topology_config=config,
            availability_node_count=3,
            availability_seed=77,
            shape=1.0,
            scale=1.0,
            mean=0.0,
            sigma=0.1,
            verbose=False,
        )
        random.seed(9)
        second = Environment(
            t_max=1,
            topology_config=config,
            availability_node_count=3,
            availability_seed=77,
            shape=1.0,
            scale=1.0,
            mean=0.0,
            sigma=0.1,
            verbose=False,
        )
        first_trace = []
        second_trace = []
        for _ in range(12):
            first.update_node_availability()
            first_trace.append(tuple(first.up))
            for _ in range(50):
                random.random()
            second.update_node_availability()
            second_trace.append(tuple(second.up))
        self.assertEqual(first_trace, second_trace)


if __name__ == "__main__":
    unittest.main()
