from __future__ import annotations

import dataclasses
import random
import unittest

from core.environment import Environment
from models import DNN, Task
from planning.repository import (
    AtomicPlanRepository,
    PlanRepository,
    build_plan_repository,
    make_deployment_plan,
    select_diverse_plans,
)
from proposed import AllDNNRefactor
from semantics import PERIODIC_SEMANTIC_VERSIONS


class PlanRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260806)
        self.environment = Environment(
            t_max=1,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        user = next(
            index for index, node in enumerate(self.environment.nodes) if node.level == 1
        )
        self.environment.ds[0] = DNN(
            tasks=[Task(1, 10.0)],
            links=[],
            delay=100_000,
            initiateNode=user,
            profile_id="profile-a",
        )
        self.edges = [
            index for index, node in enumerate(self.environment.nodes) if node.level == 2
        ]
        self.semantic_versions = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        self.origin_group = self.environment.origin_group_for_node(user)

    def _plan(self, node_index: int, plan_id: str, ttl_slots: int = 3):
        snapshot = self.environment.capture_observation_snapshot("planning")
        candidate = self.environment.predict_candidate_state(0, [node_index], snapshot)
        return make_deployment_plan(
            self.environment,
            0,
            candidate,
            profile_id="profile-a",
            origin_group=self.origin_group,
            created_slot=0,
            ttl_slots=ttl_slots,
            semantic_versions=self.semantic_versions,
            plan_id=plan_id,
        )

    def test_lookup_enforces_ttl_topology_and_semantic_versions(self) -> None:
        plan = self._plan(self.edges[0], "first", ttl_slots=2)
        repository = build_plan_repository(
            1,
            self.environment.topology_version,
            self.semantic_versions,
            0,
            [plan],
        )
        valid = repository.lookup(
            "profile-a",
            self.origin_group,
            1,
            self.environment.topology_version,
            self.semantic_versions,
            len(self.environment.nodes),
        )
        self.assertEqual(valid.candidates, (plan,))
        expired = repository.lookup(
            "profile-a",
            self.origin_group,
            2,
            self.environment.topology_version,
            self.semantic_versions,
            len(self.environment.nodes),
        )
        self.assertEqual(expired.rejection_reason, "plan_expired")
        mismatched = repository.lookup(
            "profile-a",
            self.origin_group,
            1,
            self.environment.topology_version,
            (("objective_semantics_version", "wrong"),),
            len(self.environment.nodes),
        )
        self.assertEqual(mismatched.rejection_reason, "semantic_mismatch")

    def test_diversity_filter_deduplicates_assignments_and_node_sets(self) -> None:
        first = self._plan(self.edges[0], "first")
        duplicate = dataclasses.replace(first, plan_id="duplicate")
        different = self._plan(self.edges[1], "different")
        selected = select_diverse_plans(
            [first, duplicate, different],
            max_size=3,
            max_node_overlap=0.5,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({plan.assignment for plan in selected}), 2)
        self.assertEqual(len({plan.used_nodes for plan in selected}), 2)

    def test_atomic_publish_rejects_a_stale_writer(self) -> None:
        empty = PlanRepository(
            repository_version=0,
            topology_version=self.environment.topology_version,
            semantic_versions=self.semantic_versions,
            published_slot=0,
            plans=(),
        )
        store = AtomicPlanRepository(empty)
        first = build_plan_repository(
            1,
            self.environment.topology_version,
            self.semantic_versions,
            0,
            [self._plan(self.edges[0], "first")],
        )
        second = dataclasses.replace(first, repository_version=2)
        self.assertTrue(store.publish(first, expected_current_version=0))
        self.assertFalse(store.publish(second, expected_current_version=0))
        self.assertIs(store.snapshot(), first)

    def test_plan_pool_search_is_side_effect_free_and_structurally_unique(self) -> None:
        snapshot = self.environment.capture_observation_snapshot("planning")
        before = (
            self.environment.environment_state_version,
            self.environment.current_slot,
            tuple(node.cpu for node in self.environment.nodes),
            tuple(self.environment.running_dnns),
            random.getstate(),
        )
        algorithm = AllDNNRefactor(self.environment)
        plans = algorithm.search_plan_pool(
            0,
            snapshot,
            profile_id="profile-a",
            origin_group=self.origin_group,
            max_pool_size=4,
            sample_count=20,
            ttl_slots=5,
        )
        after = (
            self.environment.environment_state_version,
            self.environment.current_slot,
            tuple(node.cpu for node in self.environment.nodes),
            tuple(self.environment.running_dnns),
            random.getstate(),
        )
        self.assertEqual(before, after)
        self.assertGreater(len(plans), 0)
        self.assertLessEqual(len(plans), 4)
        self.assertEqual(len(plans), len({plan.assignment for plan in plans}))
        stats = algorithm.last_plan_pool_search_statistics
        self.assertEqual(
            stats.generated_assignments,
            stats.constraint_rejected_candidates
            + stats.nonconverged_candidates
            + stats.feasible_candidates,
        )
        self.assertEqual(stats.selected_candidates, len(plans))
        for plan in plans:
            candidate = self.environment.predict_candidate_state(
                0,
                list(plan.assignment),
                snapshot,
            )
            self.assertTrue(candidate.is_strictly_feasible)
            self.assertTrue(candidate.fixed_point_converged)


if __name__ == "__main__":
    unittest.main()
