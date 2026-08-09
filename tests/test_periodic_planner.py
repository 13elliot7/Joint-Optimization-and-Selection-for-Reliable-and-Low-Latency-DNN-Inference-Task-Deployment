from __future__ import annotations

import random
import unittest

from core.environment import Environment
from models import DNN, Task
from planning.periodic_planner import (
    PeriodicPlanner,
    PeriodicPlannerConfig,
    PlanningTarget,
)
from planning.repository import (
    AtomicPlanRepository,
    PlanRepository,
    build_plan_repository,
    make_deployment_plan,
)
from semantics import PERIODIC_SEMANTIC_VERSIONS


class PeriodicPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260808)
        self.environment = Environment(
            t_max=1,
            availability_node_count=0,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        self.user = next(
            index for index, node in enumerate(self.environment.nodes) if node.level == 1
        )
        self.edge = next(
            index for index, node in enumerate(self.environment.nodes) if node.level == 2
        )
        self.environment.ds[0] = DNN(
            tasks=[Task(1, 10.0)],
            links=[],
            delay=100_000,
            initiateNode=self.user,
            profile_id="profile-a",
        )
        self.versions = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        self.origin_group = self.environment.origin_group_for_node(self.user)
        self.target = PlanningTarget(0, "profile-a", self.origin_group)

    def _old_repository(self):
        snapshot = self.environment.capture_observation_snapshot("planning")
        candidate = self.environment.predict_candidate_state(0, [self.edge], snapshot)
        plan = make_deployment_plan(
            self.environment,
            0,
            candidate,
            "profile-a",
            self.origin_group,
            created_slot=0,
            ttl_slots=100,
            semantic_versions=self.versions,
            plan_id="old-plan",
        )
        return build_plan_repository(
            1,
            self.environment.topology_version,
            self.versions,
            0,
            [plan],
        )

    def _empty_repository(self):
        return PlanRepository(
            repository_version=0,
            topology_version=self.environment.topology_version,
            semantic_versions=self.versions,
            published_slot=0,
            plans=(),
        )

    def _config(self, **overrides):
        values = dict(
            planning_interval_slots=50,
            minimum_replanning_interval_slots=5,
            plan_ttl_slots=100,
            max_pool_size=4,
            sample_count=20,
            min_publish_valid_ratio=0.1,
        )
        values.update(overrides)
        return PeriodicPlannerConfig(**values)

    def test_old_repository_serves_until_simulated_planning_ready_slot(self) -> None:
        old_repository = self._old_repository()
        store = AtomicPlanRepository(old_repository)
        planner = PeriodicPlanner(
            self.environment,
            store,
            (self.target,),
            config=self._config(),
            planning_runtime_provider=lambda snapshot, plans: 250.0,
        )
        started = planner.start()
        self.assertEqual(started.status, "started")
        assert planner.active_job is not None
        self.assertEqual(planner.active_job.ready_slot, 3)
        self.assertIs(store.snapshot(), old_repository)

        for _ in range(2):
            self.environment.advance_time_slot()
            self.assertEqual(planner.poll().status, "not_ready")
            self.assertIs(store.snapshot(), old_repository)

        self.environment.advance_time_slot()
        before_environment_version = self.environment.environment_state_version
        published = planner.poll()
        self.assertEqual(published.status, "published")
        self.assertEqual(store.snapshot().repository_version, 2)
        self.assertEqual(
            self.environment.environment_state_version,
            before_environment_version,
        )

    def test_start_search_does_not_advance_or_mutate_online_environment(self) -> None:
        store = AtomicPlanRepository(self._empty_repository())
        planner = PeriodicPlanner(
            self.environment,
            store,
            (self.target,),
            config=self._config(),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        before = (
            self.environment.current_slot,
            self.environment.environment_state_version,
            tuple(node.cpu for node in self.environment.nodes),
            tuple(self.environment.running_dnns),
            random.getstate(),
        )
        planner.start()
        after = (
            self.environment.current_slot,
            self.environment.environment_state_version,
            tuple(node.cpu for node in self.environment.nodes),
            tuple(self.environment.running_dnns),
            random.getstate(),
        )
        self.assertEqual(before, after)

    def test_publication_failure_retains_old_repository(self) -> None:
        environment = Environment(
            t_max=1,
            availability_node_count=10_000,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        user = next(index for index, node in enumerate(environment.nodes) if node.level == 1)
        edge = next(index for index, node in enumerate(environment.nodes) if node.level == 2)
        environment.ds[0] = DNN(
            [Task(1, 10.0)], [], 100_000, user, profile_id="profile-a"
        )
        origin = environment.origin_group_for_node(user)
        planning_snapshot = environment.capture_observation_snapshot("planning")
        candidate = environment.predict_candidate_state(0, [edge], planning_snapshot)
        old_plan = make_deployment_plan(
            environment,
            0,
            candidate,
            "profile-a",
            origin,
            0,
            100,
            self.versions,
            "old-plan",
        )
        old_repository = build_plan_repository(
            1, environment.topology_version, self.versions, 0, [old_plan]
        )
        store = AtomicPlanRepository(old_repository)
        planner = PeriodicPlanner(
            environment,
            store,
            (PlanningTarget(0, "profile-a", origin),),
            config=self._config(min_publish_valid_ratio=0.5),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        planner.start()
        environment.advance_time_slot()
        environment.up = [False for _ in environment.nodes]
        result = planner.poll()
        self.assertEqual(result.status, "retained_old_repository")
        self.assertIs(store.snapshot(), old_repository)
        self.assertEqual(result.valid_count, 0)
        self.assertTrue(planner.publication_audits)
        self.assertTrue(
            all(
                audit.primary_reason == "node_offline"
                and not audit.accepted
                for audit in planner.publication_audits
            )
        )

    def test_previous_repository_structures_are_used_as_warm_start(self) -> None:
        old_repository = self._old_repository()
        planner = PeriodicPlanner(
            self.environment,
            AtomicPlanRepository(old_repository),
            (self.target,),
            config=self._config(),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        planner.algorithm.search_customized_plan_pool = lambda *args, **kwargs: ()  # type: ignore[method-assign]
        result = planner.start()
        self.assertEqual(result.status, "started")
        assert planner.active_job is not None
        self.assertEqual(len(planner.active_job.proposed_plans), 1)
        self.assertEqual(planner.active_job.proposed_plans[0].plan_id, "old-plan")

    def test_default_periodic_generator_is_customized_and_traceable(self) -> None:
        planner = PeriodicPlanner(
            self.environment,
            AtomicPlanRepository(self._empty_repository()),
            (self.target,),
            config=self._config(),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        self.assertEqual(planner.config.candidate_generator_mode, "customized")
        planner.start()
        assert planner.active_job is not None
        self.assertTrue(planner.active_job.proposed_plans)
        self.assertTrue(
            all(
                plan.generator_mode == "customized"
                and plan.generator_version == "customized_periodic_pool_v1"
                and plan.search_seed is not None
                for plan in planner.active_job.proposed_plans
            )
        )

    def test_minimum_interval_suppresses_replanning_storm(self) -> None:
        planner = PeriodicPlanner(
            self.environment,
            AtomicPlanRepository(self._empty_repository()),
            (self.target,),
            config=self._config(minimum_replanning_interval_slots=5),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        planner.start("first")
        self.environment.advance_time_slot()
        planner.poll()
        suppressed = planner.start("load_drift")
        self.assertEqual(suppressed.status, "suppressed_minimum_interval")
        self.assertEqual(planner.statistics.suppressed_triggers, 1)

    def test_prewarm_publishes_without_advancing_formal_simulation_slot(self) -> None:
        store = AtomicPlanRepository(self._empty_repository())
        planner = PeriodicPlanner(
            self.environment,
            store,
            (self.target,),
            config=self._config(),
            planning_runtime_provider=lambda snapshot, plans: 250.0,
        )
        event = planner.prewarm()
        self.assertEqual(event.status, "published")
        self.assertGreater(event.proposed_count, 0)
        self.assertEqual(event.valid_count, event.proposed_count)
        self.assertEqual(event.valid_ratio, 1.0)
        self.assertEqual(len(planner.publication_audits), event.proposed_count)
        self.assertTrue(all(audit.accepted for audit in planner.publication_audits))
        self.assertEqual(self.environment.current_slot, 0)
        self.assertEqual(store.snapshot().repository_version, 1)
        self.assertEqual(planner.statistics.total_planning_runtime_ms, 250.0)
        self.assertEqual(
            planner.statistics.search_assignments_generated,
            planner.statistics.search_constraint_rejected_candidates
            + planner.statistics.search_nonconverged_candidates
            + planner.statistics.search_feasible_candidates,
        )

    def test_dispatch_feedback_can_trigger_on_excessive_fallback_rate(self) -> None:
        planner = PeriodicPlanner(
            self.environment,
            AtomicPlanRepository(self._empty_repository()),
            (self.target,),
            config=self._config(
                dispatch_history_window=2,
                min_direct_hit_rate=0.0,
                max_fallback_rate=0.4,
            ),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        self.assertEqual(planner.prewarm().status, "published")
        for _ in range(5):
            self.environment.advance_time_slot()
        planner.record_dispatch("fast_fallback", "profile-a")
        planner.record_dispatch("fast_fallback", "profile-a")
        events = planner.tick()
        self.assertEqual(events[-1].status, "started")
        self.assertEqual(events[-1].trigger_reason, "fallback_rate_high")


if __name__ == "__main__":
    unittest.main()
