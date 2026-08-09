from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from metrics import JointControlMetrics
from models import ObservationSnapshot
from planning.control_models import JointBudgetDecision, JointBudgetOutcome
from planning.control_state import (
    CompletedDispatchRecord,
    CompletedPlanningRecord,
    JointBudgetOutcomeBuilder,
    PlanningControlSnapshotBuilder,
)
from planning.joint_budget_controller import JointBudgetController
from planning.joint_budget_runtime import JointBudgetRuntimeCoordinator


@dataclass(frozen=True)
class ControlCycleEvent:
    applied: bool
    decision: JointBudgetDecision | None
    completed_outcome: JointBudgetOutcome | None
    reason: str | None = None


class JointBudgetControlLoop:
    """在完整历史窗口边界因果更新控制器并应用下一联合动作。"""

    def __init__(
        self,
        controller: JointBudgetController,
        coordinator: JointBudgetRuntimeCoordinator,
        minimum_control_window_slots: int = 20,
    ) -> None:
        if minimum_control_window_slots <= 0:
            raise ValueError("minimum_control_window_slots must be positive")
        self.controller = controller
        self.coordinator = coordinator
        self.minimum_control_window_slots = minimum_control_window_slots
        slot_length = coordinator.planner.environment.slot_length
        self.snapshot_builder = PlanningControlSnapshotBuilder(slot_length)
        self.outcome_builder = JointBudgetOutcomeBuilder(slot_length)
        self.metrics = JointControlMetrics([], [])
        self.current_decision: JointBudgetDecision | None = None
        self._previous_observation: ObservationSnapshot | None = None
        self._control_epoch = 0

    def step(
        self,
        observation: ObservationSnapshot,
        dispatch_records: tuple[CompletedDispatchRecord, ...],
        planning_records: tuple[CompletedPlanningRecord, ...],
    ) -> ControlCycleEvent | None:
        if self.coordinator.planner.active_job is not None:
            return None
        if self.current_decision is not None:
            required_window = max(
                self.minimum_control_window_slots,
                self.current_decision.action.planning_interval_slots,
            )
            if observation.slot - self.current_decision.created_slot < required_window:
                return None

        completed_outcome: JointBudgetOutcome | None = None
        if self.current_decision is not None:
            completed_outcome = self.outcome_builder.build(
                self.current_decision,
                observation.slot,
                dispatch_records,
                planning_records,
            )
            self.controller.observe(completed_outcome)
            self.metrics.add_outcome(completed_outcome)

        previous_interval = (
            self.current_decision.action.planning_interval_slots
            if self.current_decision is not None
            else self.minimum_control_window_slots
        )
        window_slots = max(self.minimum_control_window_slots, previous_interval)
        window_start = max(0, observation.slot - window_slots + 1)
        self._control_epoch += 1
        control_snapshot = self.snapshot_builder.build(
            self._control_epoch,
            observation,
            self.coordinator.planner.repository_store.snapshot(),
            window_start,
            dispatch_records,
            planning_records,
            self._previous_observation,
        )
        # 活动规划键即使尚无请求，也以零到达率进入配额分配，避免冷启动饿死。
        existing_keys = {
            (profile, origin)
            for profile, origin, _ in control_snapshot.arrival_rate_by_key
        }
        missing = sorted(
            {
                (target.profile_id, target.origin_group)
                for target in self.coordinator.planner.targets
            }
            - existing_keys
        )
        if missing:
            control_snapshot = dataclasses.replace(
                control_snapshot,
                arrival_rate_by_key=control_snapshot.arrival_rate_by_key
                + tuple((profile, origin, 0.0) for profile, origin in missing),
            )
        decision = self.controller.decide(control_snapshot)
        if not self.coordinator.apply(decision):
            return ControlCycleEvent(
                False,
                decision,
                completed_outcome,
                "runtime_rejected_joint_action",
            )
        self.current_decision = decision
        self._previous_observation = observation
        self.metrics.add_decision(decision)
        return ControlCycleEvent(True, decision, completed_outcome)
