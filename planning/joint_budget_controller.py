from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from planning.action_predictor import AnalyticActionPredictor
from planning.control_models import (
    JointActionPrediction,
    JointBudgetAction,
    JointBudgetDecision,
    JointBudgetOutcome,
    PlanningControlSnapshot,
)
from planning.pool_allocator import PoolBudgetAllocator, PoolDemand


class JointBudgetController(Protocol):
    def decide(self, snapshot: PlanningControlSnapshot) -> JointBudgetDecision: ...
    def observe(self, outcome: JointBudgetOutcome) -> None: ...


@dataclass(frozen=True)
class JointControlSafetyConfig:
    planner_utilization_limit: float = 0.5
    dispatcher_utilization_limit: float = 0.7
    online_p95_limit_ms: float = 50.0
    uncertainty_penalty: float = 0.25
    rejection_penalty: float = 1.0
    runtime_failure_penalty: float = 2.0
    deadline_violation_penalty: float = 2.0
    planner_utilization_penalty: float = 0.2
    dispatcher_utilization_penalty: float = 0.5
    staleness_penalty: float = 0.5
    action_change_penalty: float = 0.1


class _ControllerBase:
    def __init__(
        self,
        controller_version: str,
        predictor: AnalyticActionPredictor,
        allocator: PoolBudgetAllocator | None = None,
        safety: JointControlSafetyConfig | None = None,
    ) -> None:
        self.controller_version = controller_version
        self.predictor = predictor
        self.allocator = allocator or PoolBudgetAllocator()
        self.safety = safety or JointControlSafetyConfig()
        self._decisions: dict[str, JointBudgetDecision] = {}
        self._last_safe_action: JointBudgetAction | None = None
        self._last_allocation: tuple[tuple[str, int, int], ...] = ()

    @staticmethod
    def _demands(snapshot: PlanningControlSnapshot) -> tuple[PoolDemand, ...]:
        return tuple(
            PoolDemand(
                profile,
                origin,
                rate,
                hit_failure_risk=1.0 - snapshot.direct_hit_rate,
                availability_risk=snapshot.availability_drift,
                diversity_need=1.0 - snapshot.pool_diversity_score,
            )
            for profile, origin, rate in snapshot.arrival_rate_by_key
        )

    def _allocate(
        self,
        action: JointBudgetAction,
        snapshot: PlanningControlSnapshot,
    ) -> tuple[tuple[str, int, int], ...]:
        return self.allocator.allocate(
            action.total_pool_budget,
            self._demands(snapshot),
            action.pool_allocation_mode,
        )

    def _is_safe(
        self,
        action: JointBudgetAction,
        prediction: JointActionPrediction,
        snapshot: PlanningControlSnapshot,
    ) -> bool:
        if prediction.predicted_planner_runtime_ms > (
            action.planning_interval_slots * self.predictor.slot_length_ms
        ):
            return False
        if prediction.predicted_planner_utilization > self.safety.planner_utilization_limit:
            return False
        if prediction.predicted_dispatcher_utilization > self.safety.dispatcher_utilization_limit:
            return False
        if prediction.predicted_online_p95_ms > self.safety.online_p95_limit_ms:
            return False
        try:
            self._allocate(action, snapshot)
        except ValueError:
            return False
        return True

    def _score(
        self,
        prediction: JointActionPrediction,
        action: JointBudgetAction,
        previous: JointBudgetAction | None,
    ) -> float:
        change = 0.0
        if previous is not None:
            change = (
                abs(action.planning_interval_slots - previous.planning_interval_slots)
                / max(action.planning_interval_slots, previous.planning_interval_slots)
                + abs(action.total_pool_budget - previous.total_pool_budget)
                / max(action.total_pool_budget, previous.total_pool_budget)
                + abs(action.base_online_budget_ms - previous.base_online_budget_ms)
                / max(action.base_online_budget_ms, previous.base_online_budget_ms)
            ) / 3.0
        return (
            prediction.predicted_goodput_utility
            - self.safety.rejection_penalty * prediction.predicted_rejection_rate
            - self.safety.runtime_failure_penalty
            * prediction.predicted_runtime_failure_rate
            - self.safety.deadline_violation_penalty
            * prediction.predicted_deadline_violation_rate
            - self.safety.planner_utilization_penalty
            * prediction.predicted_planner_utilization
            - self.safety.dispatcher_utilization_penalty
            * prediction.predicted_dispatcher_utilization
            - self.safety.staleness_penalty * prediction.predicted_stale_plan_rate
            - self.safety.action_change_penalty * change
            - self.safety.uncertainty_penalty
            * (1.0 - prediction.prediction_confidence)
        )

    def _decision(
        self,
        snapshot: PlanningControlSnapshot,
        action: JointBudgetAction,
        prediction: JointActionPrediction,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> JointBudgetDecision:
        try:
            allocation = self.allocator.allocate_smoothed(
                action.total_pool_budget,
                self._demands(snapshot),
                self._last_allocation,
                action.pool_allocation_mode,
            )
        except ValueError:
            if not fallback_used:
                raise
            allocation = ()
        decision_id = (
            f"{self.controller_version}-e{snapshot.control_epoch}-s{snapshot.slot}-"
            f"t{action.planning_interval_slots}-k{action.total_pool_budget}-"
            f"b{action.base_online_budget_ms:g}"
        )
        decision = JointBudgetDecision(
            decision_id=decision_id,
            control_epoch=snapshot.control_epoch,
            created_slot=snapshot.slot,
            controller_version=self.controller_version,
            action=action,
            pool_budget_by_key=allocation,
            predicted_objective=self._score(
                prediction,
                action,
                self._last_safe_action,
            ),
            predicted_metrics=prediction,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        self._decisions[decision_id] = decision
        self._last_allocation = allocation
        if self._is_safe(action, prediction, snapshot):
            self._last_safe_action = action
        return decision

    def observe(self, outcome: JointBudgetOutcome) -> None:
        decision = self._decisions.get(outcome.decision_id)
        if decision is None:
            raise ValueError("outcome does not match a known decision")
        self.predictor.observe(
            decision.action,
            outcome,
            current_slot=outcome.available_after_slot,
        )


class FixedJointBudgetController(_ControllerBase):
    def __init__(
        self,
        action: JointBudgetAction,
        predictor: AnalyticActionPredictor,
        allocator: PoolBudgetAllocator | None = None,
    ) -> None:
        super().__init__("fixed_joint_v1", predictor, allocator)
        self.action = action

    def decide(self, snapshot: PlanningControlSnapshot) -> JointBudgetDecision:
        prediction = self.predictor.predict(snapshot, self.action)
        return self._decision(snapshot, self.action, prediction)


class RuleBasedJointBudgetController(_ControllerBase):
    def __init__(
        self,
        actions: tuple[JointBudgetAction, ...],
        initial_action: JointBudgetAction,
        predictor: AnalyticActionPredictor,
        allocator: PoolBudgetAllocator | None = None,
        minimum_control_interval_slots: int = 20,
    ) -> None:
        super().__init__("rule_based_joint_v1", predictor, allocator)
        if initial_action not in actions or minimum_control_interval_slots <= 0:
            raise ValueError("invalid rule controller configuration")
        self.actions = actions
        self.current_action = initial_action
        self.minimum_control_interval_slots = minimum_control_interval_slots
        self._last_decision_slot: int | None = None
        self._drift_latched = False
        self._online_latched = False

    @staticmethod
    def _nearest_action(
        actions: tuple[JointBudgetAction, ...],
        interval: int,
        pool: int,
        budget: float,
        mode: str,
    ) -> JointBudgetAction:
        return min(
            actions,
            key=lambda action: (
                abs(action.planning_interval_slots - interval),
                abs(action.total_pool_budget - pool),
                abs(action.base_online_budget_ms - budget),
                action.pool_allocation_mode != mode,
                action,
            ),
        )

    def decide(self, snapshot: PlanningControlSnapshot) -> JointBudgetDecision:
        if (
            self._last_decision_slot is not None
            and snapshot.slot - self._last_decision_slot
            < self.minimum_control_interval_slots
        ):
            prediction = self.predictor.predict(snapshot, self.current_action)
            return self._decision(snapshot, self.current_action, prediction)

        intervals = sorted({action.planning_interval_slots for action in self.actions})
        pools = sorted({action.total_pool_budget for action in self.actions})
        budgets = sorted({action.base_online_budget_ms for action in self.actions})
        current = self.current_action
        interval_index = intervals.index(current.planning_interval_slots)
        pool_index = pools.index(current.total_pool_budget)
        budget_index = budgets.index(current.base_online_budget_ms)
        drift = max(
            snapshot.node_load_drift,
            snapshot.link_load_drift,
            snapshot.availability_drift,
            snapshot.profile_distribution_drift,
        )
        if drift >= 0.25 or snapshot.valid_plan_ratio <= 0.5:
            self._drift_latched = True
        elif drift <= 0.10 and snapshot.valid_plan_ratio >= 0.7:
            self._drift_latched = False
        if snapshot.online_decision_p95_ms >= 40.0 or snapshot.dispatcher_utilization >= 0.65:
            self._online_latched = True
        elif snapshot.online_decision_p95_ms <= 25.0 and snapshot.dispatcher_utilization <= 0.45:
            self._online_latched = False

        if self._drift_latched:
            interval_index = max(0, interval_index - 1)
        elif snapshot.planner_utilization >= 0.45 and drift <= 0.10:
            interval_index = min(len(intervals) - 1, interval_index + 1)
        if snapshot.direct_hit_rate <= 0.5:
            pool_index = min(len(pools) - 1, pool_index + 1)
        elif snapshot.valid_plan_ratio >= 0.9 and snapshot.pool_diversity_score <= 0.2:
            pool_index = max(0, pool_index - 1)
        if self._online_latched:
            budget_index = max(0, budget_index - 1)
        elif snapshot.fallback_rate >= 0.3 and snapshot.deadline_violation_rate <= 0.1:
            budget_index = min(len(budgets) - 1, budget_index + 1)

        proposed = self._nearest_action(
            self.actions,
            intervals[interval_index],
            pools[pool_index],
            budgets[budget_index],
            current.pool_allocation_mode,
        )
        prediction = self.predictor.predict(snapshot, proposed)
        if not self._is_safe(proposed, prediction, snapshot):
            proposed = self._last_safe_action or current
            prediction = self.predictor.predict(snapshot, proposed)
            decision = self._decision(
                snapshot,
                proposed,
                prediction,
                fallback_used=True,
                fallback_reason="rule_action_unsafe",
            )
        else:
            decision = self._decision(snapshot, proposed, prediction)
        self.current_action = proposed
        self._last_decision_slot = snapshot.slot
        return decision


class DiscreteMPCJointBudgetController(_ControllerBase):
    def __init__(
        self,
        actions: tuple[JointBudgetAction, ...],
        predictor: AnalyticActionPredictor,
        conservative_action: JointBudgetAction,
        allocator: PoolBudgetAllocator | None = None,
        safety: JointControlSafetyConfig | None = None,
    ) -> None:
        super().__init__("discrete_mpc_v1", predictor, allocator, safety)
        if not actions or conservative_action not in actions:
            raise ValueError("invalid MPC action space")
        self.actions = tuple(sorted(set(actions)))
        self.conservative_action = conservative_action

    def decide(self, snapshot: PlanningControlSnapshot) -> JointBudgetDecision:
        feasible: list[tuple[float, JointBudgetAction, JointActionPrediction]] = []
        for action in self.actions:
            prediction = self.predictor.predict(snapshot, action)
            if self._is_safe(action, prediction, snapshot):
                feasible.append(
                    (
                        self._score(prediction, action, self._last_safe_action),
                        action,
                        prediction,
                    )
                )
        if not feasible:
            fallback = self._last_safe_action or self.conservative_action
            prediction = self.predictor.predict(snapshot, fallback)
            return self._decision(
                snapshot,
                fallback,
                prediction,
                fallback_used=True,
                fallback_reason="no_safe_action",
            )
        score, action, prediction = max(
            feasible,
            key=lambda item: (
                item[0],
                -item[1].planning_interval_slots,
                -item[1].total_pool_budget,
                -item[1].base_online_budget_ms,
            ),
        )
        del score
        return self._decision(snapshot, action, prediction)
