from __future__ import annotations

import random
import unittest

from core.environment import Environment
from models import DNN, InferenceRequest, Task
from planning.control_models import (
    JointActionPrediction,
    JointBudgetAction,
    JointBudgetDecision,
)
from planning.online_dispatcher import OnlineDispatcher, OnlineDispatcherConfig
from planning.joint_budget_runtime import JointBudgetRuntimeCoordinator
from planning.control_loop import JointBudgetControlLoop
from planning.action_predictor import AnalyticActionPredictor
from planning.joint_budget_controller import FixedJointBudgetController
from planning.periodic_planner import PeriodicPlanner, PeriodicPlannerConfig, PlanningTarget
from planning.repository import (
    AtomicPlanRepository,
    PlanRepository,
    build_plan_repository,
    make_deployment_plan,
)
from semantics import PERIODIC_SEMANTIC_VERSIONS


def prediction() -> JointActionPrediction:
    return JointActionPrediction(
        predicted_goodput_utility=1.0,
        predicted_rejection_rate=0.0,
        predicted_runtime_failure_rate=0.0,
        predicted_deadline_violation_rate=0.0,
        predicted_planner_utilization=0.1,
        predicted_dispatcher_utilization=0.1,
        predicted_stale_plan_rate=0.0,
        predicted_planner_runtime_ms=10.0,
        predicted_online_p95_ms=5.0,
        prediction_confidence=0.8,
    )


class IncrementClock:
    def __init__(self, step_ms: float) -> None:
        self.value = 0.0
        self.step = step_ms / 1000.0

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class JointBudgetIntegrationTests(unittest.TestCase):
    def _environment_and_repository(self, plan_count: int = 6):
        random.seed(20260809)
        environment = Environment(
            t_max=1,
            availability_node_count=0,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        user = next(index for index, node in enumerate(environment.nodes) if node.level == 1)
        environment.ds[0] = DNN(
            [Task(1, 10.0)], [], 100_000, user, profile_id="profile-a"
        )
        origin = environment.origin_group_for_node(user)
        snapshot = environment.capture_observation_snapshot("planning")
        compute_nodes = [
            index for index, node in enumerate(environment.nodes) if node.level >= 2
        ][:plan_count]
        plans = []
        for offset, node_index in enumerate(compute_nodes):
            candidate = environment.predict_candidate_state(0, [node_index], snapshot)
            plans.append(
                make_deployment_plan(
                    environment,
                    0,
                    candidate,
                    "profile-a",
                    origin,
                    0,
                    100,
                    PERIODIC_SEMANTIC_VERSIONS.as_pairs(),
                    f"plan-{offset}",
                )
            )
        repository = build_plan_repository(
            1,
            environment.topology_version,
            PERIODIC_SEMANTIC_VERSIONS.as_pairs(),
            0,
            plans,
            max_pool_size=plan_count,
        )
        request = InferenceRequest(
            0,
            "profile-a",
            0,
            user,
            100_000.0,
            1 / 3,
            1 / 3,
            1 / 3,
        )
        return environment, repository, request, origin

    def test_joint_decision_updates_planner_interval_pool_quota_and_dispatch_budget(self) -> None:
        environment, _, request, origin = self._environment_and_repository(2)
        versions = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        empty = PlanRepository(0, environment.topology_version, versions, 0, ())
        store = AtomicPlanRepository(empty)
        planner = PeriodicPlanner(
            environment,
            store,
            (PlanningTarget(0, "profile-a", origin),),
            PeriodicPlannerConfig(
                planning_interval_slots=50,
                minimum_replanning_interval_slots=1,
                max_pool_size=20,
                sample_count=20,
                min_publish_valid_ratio=0.1,
            ),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        dispatcher = OnlineDispatcher(environment)
        action = JointBudgetAction(20, 2, 5.0)
        decision = JointBudgetDecision(
            "joint-1",
            1,
            0,
            "fixed_joint_v1",
            action,
            (("profile-a", origin, 2),),
            1.0,
            prediction(),
            False,
            None,
        )
        coordinator = JointBudgetRuntimeCoordinator(planner, dispatcher)
        self.assertTrue(coordinator.apply(decision))
        self.assertIs(coordinator.current_decision, decision)
        planner.start()
        self.assertFalse(coordinator.apply(decision))
        environment.advance_time_slot()
        self.assertEqual(planner.poll().status, "published")
        self.assertLessEqual(len(store.snapshot().plans), 2)

        dispatch = dispatcher.dispatch(0, request, store.snapshot())
        self.assertEqual(dispatch.configured_online_budget_ms, 5.0)
        self.assertEqual(dispatch.control_decision_id, "joint-1")

        while environment.current_slot < 20:
            environment.advance_time_slot()
        events = planner.tick()
        self.assertEqual(events[-1].status, "started")
        self.assertEqual(events[-1].trigger_reason, "fixed_interval")

    def test_larger_anytime_budget_never_reduces_completed_evaluations(self) -> None:
        low_env, low_repo, low_request, _ = self._environment_and_repository()
        high_env, high_repo, high_request, _ = self._environment_and_repository()
        low = OnlineDispatcher(
            low_env,
            OnlineDispatcherConfig(online_budget_ms=2.0, hard_candidate_limit=20),
            clock=IncrementClock(0.4),
        ).dispatch(0, low_request, low_repo)
        high = OnlineDispatcher(
            high_env,
            OnlineDispatcherConfig(online_budget_ms=10.0, hard_candidate_limit=20),
            clock=IncrementClock(0.4),
        ).dispatch(0, high_request, high_repo)
        self.assertGreaterEqual(
            high.full_revalidation_count,
            low.full_revalidation_count,
        )
        self.assertLessEqual(low.request_online_budget_ms, 2.0)
        self.assertLessEqual(high.request_online_budget_ms, 10.0)
        self.assertTrue(low.budget_exhausted or low.early_stop_reason is not None)

    def test_fixed_top_k_compatibility_mode_limits_full_evaluations(self) -> None:
        environment, repository, request, _ = self._environment_and_repository()
        decision = OnlineDispatcher(
            environment,
            OnlineDispatcherConfig(
                full_revalidation_top_k=2,
                online_budget_ms=100.0,
                revalidation_mode="fixed_top_k",
            ),
            clock=IncrementClock(0.0),
        ).dispatch(0, request, repository)
        self.assertLessEqual(decision.full_revalidation_count, 2)

    def test_control_loop_preserves_cold_active_key_and_links_outcome(self) -> None:
        environment, _, _, origin = self._environment_and_repository(1)
        versions = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        store = AtomicPlanRepository(
            PlanRepository(0, environment.topology_version, versions, 0, ())
        )
        planner = PeriodicPlanner(
            environment,
            store,
            (PlanningTarget(0, "profile-a", origin),),
            PeriodicPlannerConfig(
                minimum_replanning_interval_slots=1,
                sample_count=5,
            ),
            planning_runtime_provider=lambda snapshot, plans: 100.0,
        )
        dispatcher = OnlineDispatcher(environment)
        action = JointBudgetAction(20, 2, 5.0)
        loop = JointBudgetControlLoop(
            FixedJointBudgetController(action, AnalyticActionPredictor()),
            JointBudgetRuntimeCoordinator(planner, dispatcher),
            minimum_control_window_slots=20,
        )
        first = loop.step(
            environment.capture_observation_snapshot("planning"), (), ()
        )
        assert first is not None and first.decision is not None
        self.assertTrue(first.applied)
        self.assertEqual(
            first.decision.pool_budget_by_key,
            (("profile-a", origin, 2),),
        )
        for _ in range(20):
            environment.advance_time_slot()
        second = loop.step(
            environment.capture_observation_snapshot("planning"), (), ()
        )
        assert second is not None
        self.assertIsNotNone(second.completed_outcome)
        self.assertEqual(len(loop.metrics.decisions), 2)
        self.assertEqual(len(loop.metrics.outcomes), 1)


if __name__ == "__main__":
    unittest.main()
