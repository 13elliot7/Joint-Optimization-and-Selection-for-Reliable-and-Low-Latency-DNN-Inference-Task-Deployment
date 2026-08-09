from __future__ import annotations

import dataclasses
import random
import time
import unittest

from core.arrival import ArrivalBatch, ArrivalTrace
from core.environment import Environment
from core.topology import TopologyConfig
from models import (
    DNN,
    DNNProfileCatalog,
    InferenceRequest,
    Task,
    profile_from_dnn,
)
from periodic_experiment import PeriodicExperimentConfig, PeriodicExperimentRunner
from semantics import PERIODIC_SEMANTIC_VERSIONS


def catalog() -> DNNProfileCatalog:
    return DNNProfileCatalog(
        profiles=(
            profile_from_dnn(DNN([Task(1, 10.0)], [], 100_000, 0), "first"),
            profile_from_dnn(DNN([Task(1, 15.0)], [], 100_000, 0), "second"),
        ),
        sampling_weights=(0.5, 0.5),
    )


def request(request_id: int, profile_id: str, slot: int, origin: int) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        profile_id=profile_id,
        arrival_slot=slot,
        initiate_node=origin,
        deadline_ms=100_000.0,
        preference_stability=0.4,
        preference_delay=0.3,
        preference_energy=0.3,
    )


class PeriodicExperimentTests(unittest.TestCase):
    def test_periodic_config_defaults_to_prewarm(self) -> None:
        self.assertEqual(
            PeriodicExperimentConfig(origin_nodes=(0,)).bootstrap_mode,
            "prewarm",
        )

    def _environment_and_trace(self):
        random.seed(20260811)
        environment = Environment(
            t_max=1,
            availability_node_count=0,
            topology_config=TopologyConfig(
                cloud_count=1,
                edge_count=2,
                user_count=2,
                edge_link_factor=1.0,
            ),
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        users = tuple(
            index for index, node in enumerate(environment.nodes) if node.level == 1
        )
        trace = ArrivalTrace(
            batches=(
                ArrivalBatch(0, ()),
                ArrivalBatch(
                    1,
                    (
                        request(0, "first", 1, users[0]),
                        request(1, "second", 1, users[1]),
                    ),
                ),
                ArrivalBatch(2, (request(2, "first", 2, users[0]),)),
                ArrivalBatch(3, ()),
            ),
            seed=20260811,
        )
        return environment, users, trace

    def _run(
        self,
        mode: str = "fixed",
        bootstrap: str = "prewarm",
        algorithm: str = "periodic_customized",
    ):
        environment, users, trace = self._environment_and_trace()
        config = PeriodicExperimentConfig(
            origin_nodes=users,
            algorithm=algorithm,
            bootstrap_mode=bootstrap,
            control_mode=mode,
            planning_intervals=(2, 5, 10),
            total_pool_budgets=(8, 20, 40),
            online_budgets_ms=(5.0, 10.0, 20.0),
            fixed_planning_interval_slots=5,
            fixed_total_pool_budget=8,
            fixed_online_budget_ms=10.0,
            planning_runtime_ms=100.0,
            planning_sample_count=5,
            planning_customized_population_size=4,
            planning_customized_generations=2,
            plan_ttl_slots=10,
            minimum_replanning_interval_slots=1,
            minimum_control_window_slots=2,
            online_timing_mode="deterministic",
            deterministic_clock_step_ms=0.01,
        )
        metrics = PeriodicExperimentRunner(
            environment,
            catalog(),
            trace,
            config,
        ).run()
        return metrics, environment

    def test_batch_event_loop_conserves_zero_one_and_multiple_arrivals(self) -> None:
        metrics, environment = self._run()
        self.assertEqual(metrics.arrived_requests, 3)
        self.assertTrue(metrics.request_conservation_valid)
        self.assertEqual(
            metrics.arrived_requests,
            metrics.accepted_requests + metrics.rejected_requests,
        )
        self.assertEqual(environment.current_slot, metrics.elapsed_slots)
        self.assertGreaterEqual(metrics.elapsed_slots, metrics.arrival_slots)
        self.assertEqual(metrics.unfinished_requests, 0)

    def test_fixed_seed_and_deterministic_clock_reproduce_complete_result(self) -> None:
        first, _ = self._run()
        second, _ = self._run()
        self.assertEqual(dataclasses.asdict(first), dataclasses.asdict(second))

    def test_cold_start_and_all_controller_modes_complete(self) -> None:
        cold, _ = self._run("fixed", "cold_start")
        self.assertEqual(cold.bootstrap_mode, "cold_start")
        for mode in ("fixed", "rule_based", "discrete_mpc"):
            metrics, _ = self._run(mode, "prewarm")
            self.assertTrue(metrics.request_conservation_valid)
            self.assertGreaterEqual(metrics.control_decision_count, 1)

    def test_periodic_result_schema_contains_frozen_semantic_versions(self) -> None:
        metrics, _ = self._run()
        rows = dict(metrics.to_report_rows())
        self.assertEqual(
            rows["objective_semantics_version"],
            PERIODIC_SEMANTIC_VERSIONS.objective,
        )
        self.assertNotIn("request_conservation_valid", rows)
        self.assertNotIn("search_assignments_generated", rows)
        self.assertGreater(metrics.search_assignments_generated, 0)
        self.assertGreater(rows["publication_valid_candidates"], 0)

    def test_per_request_customized_uses_same_event_loop_without_planning(self) -> None:
        metrics, _ = self._run(
            bootstrap="cold_start",
            algorithm="per_request_customized",
        )
        self.assertTrue(metrics.request_conservation_valid)
        self.assertEqual(metrics.planning_jobs_started, 0)
        self.assertEqual(metrics.control_decision_count, 0)
        self.assertGreater(metrics.search_assignments_generated, 0)
        self.assertEqual(
            metrics.per_request_customized_count,
            metrics.accepted_requests,
        )

    def test_online_baselines_use_same_event_loop_without_planning(self) -> None:
        for algorithm in (
            "random",
            "sa",
            "localfirst",
            "max_resource_fast",
            "rtbl",
        ):
            with self.subTest(algorithm=algorithm):
                metrics, _ = self._run(
                    bootstrap="cold_start",
                    algorithm=algorithm,
                )
                self.assertTrue(metrics.request_conservation_valid)
                self.assertEqual(metrics.planning_jobs_started, 0)
                self.assertEqual(metrics.control_decision_count, 0)
                self.assertEqual(
                    metrics.online_baseline_count,
                    metrics.accepted_requests,
                )

    def test_small_periodic_simulation_performance_smoke(self) -> None:
        started = time.monotonic()
        self._run()
        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
