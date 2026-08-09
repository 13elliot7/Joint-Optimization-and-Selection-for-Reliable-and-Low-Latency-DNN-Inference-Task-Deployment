from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PoolAllocationMode = Literal["demand_weighted", "uniform"]


@dataclass(frozen=True, order=True)
class JointBudgetAction:
    planning_interval_slots: int
    total_pool_budget: int
    base_online_budget_ms: float
    pool_allocation_mode: PoolAllocationMode = "demand_weighted"
    semantic_version: str = "grid_tkb_v1"

    def __post_init__(self) -> None:
        if self.planning_interval_slots <= 0 or self.total_pool_budget <= 0:
            raise ValueError("planning interval and pool budget must be positive")
        if self.base_online_budget_ms <= 0.0:
            raise ValueError("online budget must be positive")
        if self.pool_allocation_mode not in {"demand_weighted", "uniform"}:
            raise ValueError("unsupported pool allocation mode")


@dataclass(frozen=True)
class PlanningControlSnapshot:
    control_epoch: int
    slot: int
    observation_snapshot_version: int
    environment_state_version: int
    window_start_slot: int
    window_end_slot: int
    arrival_rate_total: float = 0.0
    arrival_rate_by_key: tuple[tuple[str, int, float], ...] = ()
    arrival_burstiness: float = 0.0
    profile_distribution_drift: float = 0.0
    node_load_drift: float = 0.0
    link_load_drift: float = 0.0
    availability_drift: float = 0.0
    topology_changed: bool = False
    direct_hit_rate: float = 0.0
    full_revalidation_pass_rate: float = 0.0
    repair_rate: float = 0.0
    fallback_rate: float = 0.0
    rejection_rate: float = 0.0
    runtime_failure_rate: float = 0.0
    deadline_violation_rate: float = 0.0
    valid_plan_ratio: float = 0.0
    expired_plan_ratio: float = 0.0
    pool_diversity_score: float = 0.0
    mean_plan_age_slots: float = 0.0
    online_decision_mean_ms: float = 0.0
    online_decision_p95_ms: float = 0.0
    mean_full_evaluation_ms: float = 0.0
    planner_runtime_mean_ms: float = 0.0
    planner_runtime_p95_ms: float = 0.0
    planner_utilization: float = 0.0
    dispatcher_utilization: float = 0.0
    prediction_error_mean: float = 0.0
    availability_calibration_error: float = 0.0
    semantic_version: str = "planning_control_snapshot_v1"

    def __post_init__(self) -> None:
        if min(
            self.control_epoch,
            self.slot,
            self.observation_snapshot_version,
            self.environment_state_version,
            self.window_start_slot,
            self.window_end_slot,
        ) < 0:
            raise ValueError("control snapshot indices must be non-negative")
        if self.window_end_slot < self.window_start_slot or self.window_end_slot > self.slot:
            raise ValueError("control window must be causal and end by the current slot")
        nonnegative = (
            self.arrival_rate_total,
            self.arrival_burstiness,
            self.online_decision_mean_ms,
            self.online_decision_p95_ms,
            self.mean_full_evaluation_ms,
            self.planner_runtime_mean_ms,
            self.planner_runtime_p95_ms,
            self.mean_plan_age_slots,
            self.prediction_error_mean,
            self.availability_calibration_error,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("control timing and rate metrics must be non-negative")
        bounded = (
            self.profile_distribution_drift,
            self.node_load_drift,
            self.link_load_drift,
            self.availability_drift,
            self.direct_hit_rate,
            self.full_revalidation_pass_rate,
            self.repair_rate,
            self.fallback_rate,
            self.rejection_rate,
            self.runtime_failure_rate,
            self.deadline_violation_rate,
            self.valid_plan_ratio,
            self.expired_plan_ratio,
            self.pool_diversity_score,
            self.planner_utilization,
            self.dispatcher_utilization,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("normalized control metrics must be in [0, 1]")
        if any(not profile or origin < 0 or rate < 0.0 for profile, origin, rate in self.arrival_rate_by_key):
            raise ValueError("invalid causal arrival-rate entry")


@dataclass(frozen=True)
class JointActionPrediction:
    predicted_goodput_utility: float
    predicted_rejection_rate: float
    predicted_runtime_failure_rate: float
    predicted_deadline_violation_rate: float
    predicted_planner_utilization: float
    predicted_dispatcher_utilization: float
    predicted_stale_plan_rate: float
    predicted_planner_runtime_ms: float
    predicted_online_p95_ms: float
    prediction_confidence: float
    semantic_version: str = "regularized_surrogate_v1"

    def __post_init__(self) -> None:
        for value in (
            self.predicted_rejection_rate,
            self.predicted_runtime_failure_rate,
            self.predicted_deadline_violation_rate,
            self.predicted_planner_utilization,
            self.predicted_dispatcher_utilization,
            self.predicted_stale_plan_rate,
            self.prediction_confidence,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("predicted rates and confidence must be in [0, 1]")
        if self.predicted_planner_runtime_ms < 0.0 or self.predicted_online_p95_ms < 0.0:
            raise ValueError("predicted runtimes must be non-negative")


@dataclass(frozen=True)
class JointBudgetDecision:
    decision_id: str
    control_epoch: int
    created_slot: int
    controller_version: str
    action: JointBudgetAction
    pool_budget_by_key: tuple[tuple[str, int, int], ...]
    predicted_objective: float
    predicted_metrics: JointActionPrediction
    fallback_used: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if not self.decision_id or self.control_epoch < 0 or self.created_slot < 0:
            raise ValueError("invalid joint budget decision identity")
        if sum(budget for _, _, budget in self.pool_budget_by_key) > self.action.total_pool_budget:
            raise ValueError("per-key pool budgets exceed total action budget")


@dataclass(frozen=True)
class JointBudgetOutcome:
    decision_id: str
    observation_start_slot: int
    observation_end_slot: int
    available_after_slot: int
    actual_goodput_utility: float
    actual_rejection_rate: float
    actual_runtime_failure_rate: float
    actual_deadline_violation_rate: float
    actual_planner_utilization: float
    actual_dispatcher_utilization: float
    actual_stale_plan_rate: float
    realized_objective: float

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("decision_id must not be empty")
        if not (
            0 <= self.observation_start_slot
            <= self.observation_end_slot
            <= self.available_after_slot
        ):
            raise ValueError("outcome availability must be causal")
        for value in (
            self.actual_rejection_rate,
            self.actual_runtime_failure_rate,
            self.actual_deadline_violation_rate,
            self.actual_planner_utilization,
            self.actual_dispatcher_utilization,
            self.actual_stale_plan_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("actual outcome rates must be in [0, 1]")


def enumerate_joint_actions(
    planning_intervals: tuple[int, ...],
    total_pool_budgets: tuple[int, ...],
    online_budgets_ms: tuple[float, ...],
    allocation_modes: tuple[PoolAllocationMode, ...] = ("demand_weighted",),
) -> tuple[JointBudgetAction, ...]:
    """稳定枚举无重复的有限联合动作空间。"""
    actions = {
        JointBudgetAction(interval, pool, budget, mode)
        for interval in planning_intervals
        for pool in total_pool_budgets
        for budget in online_budgets_ms
        for mode in allocation_modes
    }
    return tuple(
        sorted(
            actions,
            key=lambda action: (
                action.planning_interval_slots,
                action.total_pool_budget,
                action.base_online_budget_ms,
                action.pool_allocation_mode,
            ),
        )
    )
