from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from models import ObservationSnapshot
from planning.control_models import (
    JointBudgetDecision,
    JointBudgetOutcome,
    PlanningControlSnapshot,
)
from planning.repository import PlanRepository


@dataclass(frozen=True)
class CompletedDispatchRecord:
    slot: int
    available_after_slot: int
    profile_id: str
    origin_group: int
    path: str
    online_runtime_ms: float
    full_revalidation_count: int
    feasible_candidate_count: int
    rejected: bool
    runtime_failed: bool | None = None
    deadline_violated: bool | None = None
    request_utility: float | None = None
    outcome_available_after_slot: int | None = None
    request_id: int = -1

    def __post_init__(self) -> None:
        if self.slot < 0 or self.available_after_slot < self.slot:
            raise ValueError("dispatch record must become available causally")
        if (
            self.outcome_available_after_slot is not None
            and self.outcome_available_after_slot < self.available_after_slot
        ):
            raise ValueError("runtime outcome cannot precede the dispatch decision")


@dataclass(frozen=True)
class CompletedPlanningRecord:
    start_slot: int
    end_slot: int
    available_after_slot: int
    runtime_ms: float
    proposed_candidates: int
    valid_candidates: int
    published: bool

    def __post_init__(self) -> None:
        if not 0 <= self.start_slot <= self.end_slot <= self.available_after_slot:
            raise ValueError("planning record must become available causally")


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


class PlanningControlSnapshotBuilder:
    """只聚合当前槽已经可见的历史记录，构造因果控制快照。"""

    def __init__(self, slot_length_ms: float = 100.0) -> None:
        if slot_length_ms <= 0.0:
            raise ValueError("slot_length_ms must be positive")
        self.slot_length_ms = slot_length_ms

    def build(
        self,
        control_epoch: int,
        observation: ObservationSnapshot,
        repository: PlanRepository,
        window_start_slot: int,
        dispatch_records: tuple[CompletedDispatchRecord, ...],
        planning_records: tuple[CompletedPlanningRecord, ...],
        previous_observation: ObservationSnapshot | None = None,
    ) -> PlanningControlSnapshot:
        slot = observation.slot
        if not 0 <= window_start_slot <= slot:
            raise ValueError("invalid control window start")
        dispatch = [
            record
            for record in dispatch_records
            if window_start_slot <= record.slot <= slot
            and record.available_after_slot <= slot
        ]
        planning = [
            record
            for record in planning_records
            if record.end_slot >= window_start_slot
            and record.available_after_slot <= slot
        ]
        window_slots = max(1, slot - window_start_slot + 1)
        window_seconds = window_slots * self.slot_length_ms / 1000.0
        arrival_rate_total = len(dispatch) / window_seconds
        key_counts = Counter((record.profile_id, record.origin_group) for record in dispatch)
        arrival_by_key = tuple(
            (profile, origin, count / window_seconds)
            for (profile, origin), count in sorted(key_counts.items())
        )
        per_slot = Counter(record.slot for record in dispatch)
        counts = [per_slot.get(current, 0) for current in range(window_start_slot, slot + 1)]
        mean_count = sum(counts) / len(counts)
        variance = sum((value - mean_count) ** 2 for value in counts) / len(counts)
        burstiness = variance / mean_count if mean_count > 0.0 else 0.0

        def rate(predicate) -> float:
            return sum(1 for record in dispatch if predicate(record)) / len(dispatch) if dispatch else 0.0

        full_count = sum(record.full_revalidation_count for record in dispatch)
        feasible_count = sum(record.feasible_candidate_count for record in dispatch)
        online_times = [record.online_runtime_ms for record in dispatch]
        evaluation_times = [
            record.online_runtime_ms / record.full_revalidation_count
            for record in dispatch
            if record.full_revalidation_count > 0
        ]
        planner_times = [record.runtime_ms for record in planning]
        proposed = sum(record.proposed_candidates for record in planning)
        valid = sum(record.valid_candidates for record in planning)

        plans = repository.plans
        expired = sum(1 for plan in plans if slot > plan.expires_slot)
        ages = [max(0, slot - plan.created_slot) for plan in plans]
        diversity_pairs: list[float] = []
        for index, first in enumerate(plans):
            for second in plans[index + 1 :]:
                union = first.used_nodes | second.used_nodes
                overlap = len(first.used_nodes & second.used_nodes) / len(union) if union else 1.0
                diversity_pairs.append(1.0 - overlap)

        if previous_observation is None:
            node_drift = link_drift = availability_drift = 0.0
            topology_changed = False
        else:
            node_drift = min(
                1.0,
                max(
                    (abs(a - b) for a, b in zip(observation.node_loads, previous_observation.node_loads)),
                    default=0.0,
                ),
            )
            link_drift = min(
                1.0,
                max(
                    (
                        abs(a - b)
                        for a, b in zip(
                            observation.directional_link_loads,
                            previous_observation.directional_link_loads,
                        )
                    ),
                    default=0.0,
                ),
            )
            availability_drift = sum(
                current != previous
                for current, previous in zip(
                    observation.node_online,
                    previous_observation.node_online,
                )
            ) / max(len(observation.node_online), 1)
            topology_changed = observation.topology_version != previous_observation.topology_version

        window_runtime_ms = window_slots * self.slot_length_ms
        return PlanningControlSnapshot(
            control_epoch=control_epoch,
            slot=slot,
            observation_snapshot_version=observation.snapshot_version,
            environment_state_version=observation.environment_state_version,
            window_start_slot=window_start_slot,
            window_end_slot=slot,
            arrival_rate_total=arrival_rate_total,
            arrival_rate_by_key=arrival_by_key,
            arrival_burstiness=burstiness,
            node_load_drift=node_drift,
            link_load_drift=link_drift,
            availability_drift=availability_drift,
            topology_changed=topology_changed,
            direct_hit_rate=rate(lambda record: record.path == "plan_direct"),
            full_revalidation_pass_rate=(feasible_count / full_count if full_count else 0.0),
            repair_rate=rate(lambda record: record.path == "plan_repaired"),
            fallback_rate=rate(lambda record: record.path == "fast_fallback"),
            rejection_rate=rate(lambda record: record.rejected),
            runtime_failure_rate=(
                sum(
                    record.runtime_failed is True
                    and record.outcome_available_after_slot is not None
                    and record.outcome_available_after_slot <= slot
                    for record in dispatch
                )
                / sum(
                    record.runtime_failed is not None
                    and record.outcome_available_after_slot is not None
                    and record.outcome_available_after_slot <= slot
                    for record in dispatch
                )
                if any(
                    record.runtime_failed is not None
                    and record.outcome_available_after_slot is not None
                    and record.outcome_available_after_slot <= slot
                    for record in dispatch
                )
                else 0.0
            ),
            deadline_violation_rate=(
                sum(
                    record.deadline_violated is True
                    and record.outcome_available_after_slot is not None
                    and record.outcome_available_after_slot <= slot
                    for record in dispatch
                )
                / sum(
                    record.deadline_violated is not None
                    and record.outcome_available_after_slot is not None
                    and record.outcome_available_after_slot <= slot
                    for record in dispatch
                )
                if any(
                    record.deadline_violated is not None
                    and record.outcome_available_after_slot is not None
                    and record.outcome_available_after_slot <= slot
                    for record in dispatch
                )
                else 0.0
            ),
            valid_plan_ratio=(valid / proposed if proposed else 0.0),
            expired_plan_ratio=(expired / len(plans) if plans else 0.0),
            pool_diversity_score=(sum(diversity_pairs) / len(diversity_pairs) if diversity_pairs else 0.0),
            mean_plan_age_slots=(sum(ages) / len(ages) if ages else 0.0),
            online_decision_mean_ms=(sum(online_times) / len(online_times) if online_times else 0.0),
            online_decision_p95_ms=_percentile95(online_times),
            mean_full_evaluation_ms=(sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0.0),
            planner_runtime_mean_ms=(sum(planner_times) / len(planner_times) if planner_times else 0.0),
            planner_runtime_p95_ms=_percentile95(planner_times),
            planner_utilization=min(1.0, sum(planner_times) / window_runtime_ms),
            dispatcher_utilization=min(1.0, sum(online_times) / window_runtime_ms),
        )


@dataclass(frozen=True)
class ControlObjectivePenalties:
    rejection: float = 1.0
    runtime_failure: float = 2.0
    deadline_violation: float = 2.0
    planner_utilization: float = 0.2
    dispatcher_utilization: float = 0.5
    staleness: float = 0.5


class JointBudgetOutcomeBuilder:
    """把已完成窗口的 goodput 与实际成本归属到对应 decision_id。"""

    def __init__(
        self,
        slot_length_ms: float = 100.0,
        penalties: ControlObjectivePenalties | None = None,
    ) -> None:
        if slot_length_ms <= 0.0:
            raise ValueError("slot_length_ms must be positive")
        self.slot_length_ms = slot_length_ms
        self.penalties = penalties or ControlObjectivePenalties()

    def build(
        self,
        decision: JointBudgetDecision,
        observation_end_slot: int,
        dispatch_records: tuple[CompletedDispatchRecord, ...],
        planning_records: tuple[CompletedPlanningRecord, ...],
    ) -> JointBudgetOutcome:
        start = decision.created_slot
        if observation_end_slot < start:
            raise ValueError("outcome window cannot end before its decision")
        dispatch = [
            record
            for record in dispatch_records
            if start <= record.slot <= observation_end_slot
            and record.available_after_slot <= observation_end_slot
        ]
        planning = [
            record
            for record in planning_records
            if record.end_slot >= start
            and record.available_after_slot <= observation_end_slot
        ]
        window_slots = max(1, observation_end_slot - start + 1)
        window_seconds = window_slots * self.slot_length_ms / 1000.0
        window_runtime_ms = window_slots * self.slot_length_ms

        def rate(predicate) -> float:
            return sum(1 for record in dispatch if predicate(record)) / len(dispatch) if dispatch else 0.0

        rejection = rate(lambda record: record.rejected)
        outcome_visible = [
            record
            for record in dispatch
            if record.outcome_available_after_slot is not None
            and record.outcome_available_after_slot <= observation_end_slot
        ]
        runtime_known = [
            record for record in outcome_visible if record.runtime_failed is not None
        ]
        deadline_known = [
            record for record in outcome_visible if record.deadline_violated is not None
        ]
        runtime_failure = (
            sum(record.runtime_failed is True for record in runtime_known)
            / len(runtime_known)
            if runtime_known
            else 0.0
        )
        deadline = (
            sum(record.deadline_violated is True for record in deadline_known)
            / len(deadline_known)
            if deadline_known
            else 0.0
        )
        planner_utilization = min(
            1.0,
            sum(record.runtime_ms for record in planning) / window_runtime_ms,
        )
        dispatcher_utilization = min(
            1.0,
            sum(record.online_runtime_ms for record in dispatch) / window_runtime_ms,
        )
        proposed = sum(record.proposed_candidates for record in planning)
        valid = sum(record.valid_candidates for record in planning)
        stale = max(0.0, min(1.0, (proposed - valid) / proposed)) if proposed else 0.0
        goodput = sum(
            record.request_utility or 0.0 for record in outcome_visible
        ) / window_seconds
        realized = (
            goodput
            - self.penalties.rejection * rejection
            - self.penalties.runtime_failure * runtime_failure
            - self.penalties.deadline_violation * deadline
            - self.penalties.planner_utilization * planner_utilization
            - self.penalties.dispatcher_utilization * dispatcher_utilization
            - self.penalties.staleness * stale
        )
        return JointBudgetOutcome(
            decision_id=decision.decision_id,
            observation_start_slot=start,
            observation_end_slot=observation_end_slot,
            available_after_slot=observation_end_slot,
            actual_goodput_utility=goodput,
            actual_rejection_rate=rejection,
            actual_runtime_failure_rate=runtime_failure,
            actual_deadline_violation_rate=deadline,
            actual_planner_utilization=planner_utilization,
            actual_dispatcher_utilization=dispatcher_utilization,
            actual_stale_plan_rate=stale,
            realized_objective=realized,
        )
