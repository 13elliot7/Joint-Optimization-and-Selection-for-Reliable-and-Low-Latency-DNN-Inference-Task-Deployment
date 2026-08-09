from __future__ import annotations

import random
import unittest

from core.environment import Environment
from models import DNN, Task
from proposed import AllDNNRefactor, CustomizedSearchConfig
from semantics import PERIODIC_SEMANTIC_VERSIONS


class CustomizedPeriodicGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260803)
        self.environment = Environment(
            t_max=1,
            availability_node_count=0,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        user = next(
            index for index, node in enumerate(self.environment.nodes) if node.level == 1
        )
        self.environment.ds[0] = DNN(
            tasks=[Task(1, 10.0), Task(2, 10.0)],
            links=[],
            delay=100_000.0,
            initiateNode=user,
            profile_id="profile-customized",
        )
        self.algorithm = AllDNNRefactor(self.environment)
        self.config = CustomizedSearchConfig(
            population_size=6,
            generations=2,
            archive_capacity=8,
            random_seed=17,
        )

    def _environment_state(self) -> tuple:
        return (
            self.environment.current_slot,
            self.environment.environment_state_version,
            tuple(node.cpu for node in self.environment.nodes),
            tuple(self.environment.running_dnns),
            random.getstate(),
        )

    def test_search_is_side_effect_free_and_restores_global_rng(self) -> None:
        snapshot = self.environment.capture_observation_snapshot("planning")
        before = self._environment_state()
        result = self.algorithm.generate_customized_candidates(
            dnn_index=0,
            snapshot=snapshot,
            search_config=self.config,
        )
        self.assertEqual(before, self._environment_state())
        self.assertTrue(result.candidates)
        self.assertTrue(all(candidate.constraint_violation <= 1e-12 for candidate in result.candidates))

    def test_fixed_snapshot_and_seed_are_deterministic_after_live_state_changes(self) -> None:
        snapshot = self.environment.capture_observation_snapshot("planning")
        first = self.algorithm.generate_customized_candidates(
            dnn_index=0,
            snapshot=snapshot,
            search_config=self.config,
        )
        assignment = list(first.candidates[0].assignment)
        self.environment.add_running_dnn(
            0,
            assignment,
            estimated_runtime=self.environment.slot_length,
            remaining_slots=1,
        )
        second = self.algorithm.generate_customized_candidates(
            dnn_index=0,
            snapshot=snapshot,
            search_config=self.config,
        )
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(first.seed, second.seed)

    def test_plan_pool_contains_customized_provenance(self) -> None:
        snapshot = self.environment.capture_observation_snapshot("planning")
        plans = self.algorithm.search_customized_plan_pool(
            0,
            snapshot,
            "profile-customized",
            self.environment.origin_group_for_node(self.environment.ds[0].initiateNode),
            max_pool_size=4,
            ttl_slots=20,
            semantic_versions=PERIODIC_SEMANTIC_VERSIONS.as_pairs(),
            search_config=self.config,
        )
        self.assertTrue(plans)
        self.assertTrue(all(plan.generator_mode == "customized" for plan in plans))
        self.assertTrue(all(plan.search_seed is not None for plan in plans))


if __name__ == "__main__":
    unittest.main()
