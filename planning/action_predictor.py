from __future__ import annotations

import math
from dataclasses import dataclass

from planning.control_models import (
    JointActionPrediction,
    JointBudgetAction,
    JointBudgetOutcome,
    PlanningControlSnapshot,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class CausalActionSample:
    action: JointBudgetAction
    outcome: JointBudgetOutcome


class AnalyticActionPredictor:
    """解析开销先验与已完成因果样本均值组成的轻量代理模型。"""

    def __init__(self, slot_length_ms: float = 100.0) -> None:
        if slot_length_ms <= 0.0:
            raise ValueError("slot_length_ms must be positive")
        self.slot_length_ms = slot_length_ms
        self._samples: list[CausalActionSample] = []

    def observe(
        self,
        action: JointBudgetAction,
        outcome: JointBudgetOutcome,
        current_slot: int,
    ) -> None:
        if outcome.available_after_slot > current_slot:
            raise ValueError("cannot learn from an outcome that is not causally available")
        self._samples.append(CausalActionSample(action, outcome))

    def fit_offline(
        self,
        samples: tuple[CausalActionSample, ...],
        dataset_role: str,
    ) -> None:
        """只接受显式标记的训练场景样本，阻止测试轨迹进入离线拟合。"""
        if dataset_role != "training":
            raise ValueError("offline predictor fitting only accepts training data")
        self._samples.extend(samples)

    def predict(
        self,
        snapshot: PlanningControlSnapshot,
        action: JointBudgetAction,
    ) -> JointActionPrediction:
        pool_factor = min(1.0, math.log1p(action.total_pool_budget) / math.log(161.0))
        budget_factor = min(1.0, action.base_online_budget_ms / 50.0)
        drift = max(
            snapshot.node_load_drift,
            snapshot.link_load_drift,
            snapshot.availability_drift,
            snapshot.profile_distribution_drift,
        )
        age_risk = min(1.0, action.planning_interval_slots / 200.0) * drift
        rejection = _clamp(
            0.45 * snapshot.rejection_rate
            + 0.35 * (1.0 - snapshot.valid_plan_ratio)
            + 0.20 * age_risk
            - 0.12 * pool_factor
            - 0.08 * budget_factor
        )
        runtime_failure = _clamp(
            0.7 * snapshot.runtime_failure_rate
            + 0.2 * snapshot.availability_drift
            + 0.1 * age_risk
        )
        deadline_violation = _clamp(
            0.6 * snapshot.deadline_violation_rate
            + 0.25 * snapshot.dispatcher_utilization
            + 0.15 * max(0.0, action.base_online_budget_ms / 50.0 - 0.5)
        )
        base_planning_ms = max(
            snapshot.planner_runtime_mean_ms,
            action.total_pool_budget * 2.0,
        )
        planner_runtime_ms = base_planning_ms * (
            0.5 + action.total_pool_budget / 80.0
        )
        planner_utilization = _clamp(
            planner_runtime_ms
            / (action.planning_interval_slots * self.slot_length_ms)
        )
        evaluation_ms = max(snapshot.mean_full_evaluation_ms, 0.25)
        expected_evaluations = max(1.0, min(action.total_pool_budget, action.base_online_budget_ms / evaluation_ms))
        online_p95_ms = min(
            action.base_online_budget_ms,
            max(snapshot.online_decision_p95_ms, evaluation_ms * expected_evaluations),
        )
        dispatcher_utilization = _clamp(
            snapshot.arrival_rate_total * online_p95_ms / 1000.0
        )
        stale_rate = _clamp(
            0.5 * (1.0 - snapshot.valid_plan_ratio)
            + 0.3 * snapshot.expired_plan_ratio
            + 0.2 * age_risk
        )
        goodput = max(
            0.0,
            snapshot.arrival_rate_total
            * (1.0 - rejection)
            * (1.0 - runtime_failure)
            * (1.0 - deadline_violation)
            * (0.6 + 0.2 * pool_factor + 0.2 * budget_factor),
        )

        matching = [sample for sample in self._samples if sample.action == action]
        confidence = min(0.95, 0.25 + 0.15 * len(matching))
        if matching:
            weight = min(0.7, len(matching) / (len(matching) + 2.0))
            mean_goodput = sum(
                sample.outcome.actual_goodput_utility for sample in matching
            ) / len(matching)
            mean_rejection = sum(
                sample.outcome.actual_rejection_rate for sample in matching
            ) / len(matching)
            mean_failure = sum(
                sample.outcome.actual_runtime_failure_rate for sample in matching
            ) / len(matching)
            mean_deadline = sum(
                sample.outcome.actual_deadline_violation_rate for sample in matching
            ) / len(matching)
            mean_planner = sum(
                sample.outcome.actual_planner_utilization for sample in matching
            ) / len(matching)
            mean_dispatcher = sum(
                sample.outcome.actual_dispatcher_utilization for sample in matching
            ) / len(matching)
            mean_stale = sum(
                sample.outcome.actual_stale_plan_rate for sample in matching
            ) / len(matching)
            goodput = (1.0 - weight) * goodput + weight * mean_goodput
            rejection = (1.0 - weight) * rejection + weight * mean_rejection
            runtime_failure = (1.0 - weight) * runtime_failure + weight * mean_failure
            deadline_violation = (1.0 - weight) * deadline_violation + weight * mean_deadline
            planner_utilization = (1.0 - weight) * planner_utilization + weight * mean_planner
            dispatcher_utilization = (
                (1.0 - weight) * dispatcher_utilization + weight * mean_dispatcher
            )
            stale_rate = (1.0 - weight) * stale_rate + weight * mean_stale

        return JointActionPrediction(
            predicted_goodput_utility=goodput,
            predicted_rejection_rate=_clamp(rejection),
            predicted_runtime_failure_rate=_clamp(runtime_failure),
            predicted_deadline_violation_rate=_clamp(deadline_violation),
            predicted_planner_utilization=_clamp(planner_utilization),
            predicted_dispatcher_utilization=_clamp(dispatcher_utilization),
            predicted_stale_plan_rate=_clamp(stale_rate),
            predicted_planner_runtime_ms=planner_runtime_ms,
            predicted_online_p95_ms=online_p95_ms,
            prediction_confidence=confidence,
        )
