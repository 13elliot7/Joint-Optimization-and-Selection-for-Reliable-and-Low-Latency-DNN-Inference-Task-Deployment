from __future__ import annotations

import dataclasses
import unittest

from planning.action_predictor import AnalyticActionPredictor
from planning.control_models import (
    JointActionPrediction,
    JointBudgetAction,
    JointBudgetOutcome,
    PlanningControlSnapshot,
    enumerate_joint_actions,
)
from planning.joint_budget_controller import (
    DiscreteMPCJointBudgetController,
    FixedJointBudgetController,
    JointControlSafetyConfig,
    RuleBasedJointBudgetController,
)
from planning.pool_allocator import PoolBudgetAllocator, PoolDemand


def control_snapshot(**overrides) -> PlanningControlSnapshot:
    values = dict(
        control_epoch=1,
        slot=100,
        observation_snapshot_version=3,
        environment_state_version=4,
        window_start_slot=50,
        window_end_slot=100,
        arrival_rate_total=1.0,
        arrival_rate_by_key=(("a", 1, 0.8), ("b", 2, 0.2)),
        direct_hit_rate=0.8,
        valid_plan_ratio=0.8,
        pool_diversity_score=0.8,
    )
    values.update(overrides)
    return PlanningControlSnapshot(**values)


class JointBudgetControlTests(unittest.TestCase):
    def test_action_space_is_deterministic_unique_and_validated(self) -> None:
        actions = enumerate_joint_actions((20, 100), (20, 40), (5.0, 10.0))
        self.assertEqual(len(actions), 8)
        self.assertEqual(actions, enumerate_joint_actions((100, 20), (40, 20), (10.0, 5.0)))
        self.assertEqual(len(actions), len(set(actions)))
        with self.assertRaises(ValueError):
            JointBudgetAction(0, 20, 10.0)

    def test_pool_allocator_preserves_total_bounds_and_demand_order(self) -> None:
        allocator = PoolBudgetAllocator(minimum_per_key=2, maximum_per_key=8)
        allocation = allocator.allocate(
            10,
            (
                PoolDemand("high", 1, 10.0),
                PoolDemand("low", 2, 1.0),
            ),
        )
        by_key = {(profile, origin): budget for profile, origin, budget in allocation}
        self.assertEqual(sum(by_key.values()), 10)
        self.assertGreaterEqual(by_key[("high", 1)], by_key[("low", 2)])
        self.assertGreaterEqual(min(by_key.values()), 2)
        self.assertEqual(
            allocation,
            allocator.allocate(
                10,
                (PoolDemand("low", 2, 1.0), PoolDemand("high", 1, 10.0)),
            ),
        )
        smoothed = allocator.allocate_smoothed(
            10,
            (PoolDemand("high", 1, 10.0), PoolDemand("low", 2, 1.0)),
            previous=(("high", 1, 5), ("low", 2, 5)),
            smoothing_alpha=0.5,
            max_change_ratio=0.5,
        )
        self.assertEqual(sum(value for _, _, value in smoothed), 10)
        self.assertTrue(all(2 <= value <= 8 for _, _, value in smoothed))

    def test_fixed_controller_returns_configured_atomic_tkb_action(self) -> None:
        action = JointBudgetAction(100, 20, 10.0)
        controller = FixedJointBudgetController(action, AnalyticActionPredictor())
        decision = controller.decide(control_snapshot())
        self.assertEqual(decision.action, action)
        self.assertEqual(sum(value for _, _, value in decision.pool_budget_by_key), 20)
        self.assertIn("t100-k20-b10", decision.decision_id)

    def test_rule_controller_changes_each_dimension_and_respects_hold_interval(self) -> None:
        actions = enumerate_joint_actions((20, 50, 100), (20, 40, 80), (5.0, 10.0, 20.0))
        initial = JointBudgetAction(50, 40, 10.0)
        controller = RuleBasedJointBudgetController(
            actions,
            initial,
            AnalyticActionPredictor(),
            minimum_control_interval_slots=20,
        )
        drifted = control_snapshot(
            node_load_drift=0.4,
            valid_plan_ratio=0.4,
            direct_hit_rate=0.3,
            fallback_rate=0.4,
            deadline_violation_rate=0.0,
        )
        first = controller.decide(drifted)
        self.assertLess(first.action.planning_interval_slots, initial.planning_interval_slots)
        self.assertGreater(first.action.total_pool_budget, initial.total_pool_budget)
        self.assertGreater(first.action.base_online_budget_ms, initial.base_online_budget_ms)
        held = controller.decide(dataclasses.replace(drifted, control_epoch=2, slot=110))
        self.assertEqual(held.action, first.action)

    def test_mpc_filters_unsafe_action_with_deterministic_selection(self) -> None:
        unsafe = JointBudgetAction(20, 20, 10.0)
        safe = JointBudgetAction(100, 20, 10.0)

        class Predictor(AnalyticActionPredictor):
            def predict(self, snapshot, action):
                del snapshot
                return JointActionPrediction(
                    predicted_goodput_utility=10.0 if action == unsafe else 5.0,
                    predicted_rejection_rate=0.0,
                    predicted_runtime_failure_rate=0.0,
                    predicted_deadline_violation_rate=0.0,
                    predicted_planner_utilization=0.9 if action == unsafe else 0.1,
                    predicted_dispatcher_utilization=0.1,
                    predicted_stale_plan_rate=0.0,
                    predicted_planner_runtime_ms=100.0,
                    predicted_online_p95_ms=10.0,
                    prediction_confidence=0.8,
                )

        controller = DiscreteMPCJointBudgetController(
            (unsafe, safe), Predictor(), conservative_action=safe
        )
        first = controller.decide(control_snapshot())
        second = controller.decide(dataclasses.replace(control_snapshot(), control_epoch=2))
        self.assertEqual(first.action, safe)
        self.assertEqual(second.action, safe)

    def test_mpc_uses_conservative_fallback_when_every_action_is_unsafe(self) -> None:
        action = JointBudgetAction(100, 20, 10.0)
        controller = DiscreteMPCJointBudgetController(
            (action,),
            AnalyticActionPredictor(),
            conservative_action=action,
            safety=JointControlSafetyConfig(
                planner_utilization_limit=0.0,
                dispatcher_utilization_limit=0.0,
                online_p95_limit_ms=0.0,
            ),
        )
        decision = controller.decide(control_snapshot())
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.fallback_reason, "no_safe_action")

    def test_predictor_rejects_future_outcome_for_online_update(self) -> None:
        predictor = AnalyticActionPredictor()
        action = JointBudgetAction(100, 20, 10.0)
        outcome = JointBudgetOutcome(
            decision_id="d",
            observation_start_slot=0,
            observation_end_slot=10,
            available_after_slot=12,
            actual_goodput_utility=1.0,
            actual_rejection_rate=0.0,
            actual_runtime_failure_rate=0.0,
            actual_deadline_violation_rate=0.0,
            actual_planner_utilization=0.1,
            actual_dispatcher_utilization=0.1,
            actual_stale_plan_rate=0.0,
            realized_objective=1.0,
        )
        with self.assertRaises(ValueError):
            predictor.observe(action, outcome, current_slot=11)


if __name__ == "__main__":
    unittest.main()
