from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from core.environment import (
    Environment,
    InfeasibleAssignmentError,
    StaleEnvironmentStateError,
)
from models import CandidateStateSnapshot, InferenceRequest, ObservationSnapshot
from planning.repository import DeploymentPlan, PlanRepository
from planning.control_models import JointBudgetDecision
from semantics import PERIODIC_SEMANTIC_VERSIONS


@dataclass(frozen=True)
class OnlineDispatcherConfig:
    full_revalidation_top_k: int = 5
    online_budget_ms: float = 50.0
    minimum_availability: float = 0.0
    revalidation_mode: str = "anytime"
    deadline_safety_ratio: float = 0.25
    hard_candidate_limit: int = 20

    def __post_init__(self) -> None:
        if self.full_revalidation_top_k <= 0:
            raise ValueError("full_revalidation_top_k must be positive")
        if self.online_budget_ms <= 0.0:
            raise ValueError("online_budget_ms must be positive")
        if not 0.0 <= self.minimum_availability <= 1.0:
            raise ValueError("minimum_availability must be in [0, 1]")
        if self.revalidation_mode not in {"anytime", "fixed_top_k"}:
            raise ValueError("unsupported revalidation_mode")
        if not 0.0 <= self.deadline_safety_ratio <= 1.0:
            raise ValueError("deadline_safety_ratio must be in [0, 1]")
        if self.hard_candidate_limit <= 0:
            raise ValueError("hard_candidate_limit must be positive")


@dataclass(frozen=True)
class DispatchDecision:
    accepted: bool
    path: str
    rejection_reason: str | None
    plan_id: str | None
    assignment: tuple[int, ...] | None
    candidate: CandidateStateSnapshot | None
    repository_version: int
    evaluated_environment_state_version: int
    committed_environment_state_version: int | None
    full_revalidation_count: int
    online_runtime_ms: float
    configured_online_budget_ms: float
    request_online_budget_ms: float
    feasible_candidate_count: int
    budget_exhausted: bool
    early_stop_reason: str | None
    best_utility_after_each_evaluation: tuple[float, ...]
    control_decision_id: str | None


class OnlineDispatcher:
    """对周期方案执行快速过滤、完整复验和乐观并发提交。"""

    def __init__(
        self,
        environment: Environment,
        config: OnlineDispatcherConfig | None = None,
        semantic_versions: tuple[tuple[str, str], ...] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.environment = environment
        self.config = config or OnlineDispatcherConfig()
        self.semantic_versions = (
            semantic_versions or PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        )
        semantic_map = dict(self.semantic_versions)
        if semantic_map.get("objective_semantics_version") != (
            self.environment.objective_semantics_version
        ):
            raise ValueError(
                "dispatcher semantic versions do not match environment objective semantics"
            )
        self._clock = clock
        self._base_online_budget_ms = self.config.online_budget_ms
        self._control_decision_id: str | None = None

    def _elapsed_ms(self, started_at: float) -> float:
        return (self._clock() - started_at) * 1000.0

    def _within_budget(self, started_at: float, request_budget_ms: float) -> bool:
        return self._elapsed_ms(started_at) < request_budget_ms

    def apply_joint_budget_decision(self, decision: JointBudgetDecision) -> None:
        """原子更新后续请求使用的基础在线复验预算。"""
        self._base_online_budget_ms = decision.action.base_online_budget_ms
        self._control_decision_id = decision.decision_id

    @staticmethod
    def _plan_reference_score(plan: DeploymentPlan, request: InferenceRequest) -> float:
        return (
            request.preference_stability * plan.planning_reference_oss
            + request.preference_delay * plan.planning_reference_delay_satisfaction
            + request.preference_energy * plan.planning_reference_energy_satisfaction
        ) / (
            request.preference_stability
            + request.preference_delay
            + request.preference_energy
        )

    @staticmethod
    def _candidate_score(
        candidate: CandidateStateSnapshot,
        request: InferenceRequest,
    ) -> float:
        return (
            request.preference_stability * candidate.operational_stability
            + request.preference_delay * candidate.delay_satisfaction
            + request.preference_energy * candidate.energy_satisfaction
        ) / (
            request.preference_stability
            + request.preference_delay
            + request.preference_energy
        )

    def _quick_filter(
        self,
        plan: DeploymentPlan,
        snapshot: ObservationSnapshot,
    ) -> str | None:
        for node_index, required_cpu in plan.required_cpu:
            if not snapshot.node_online[node_index]:
                return "node_offline"
            if snapshot.node_available_cpu[node_index] + 1e-12 < required_cpu:
                return "insufficient_cpu"
        return None

    def _repair_assignment(
        self,
        dnn_index: int,
        plan: DeploymentPlan,
        snapshot: ObservationSnapshot,
    ) -> tuple[int, ...] | None:
        assignment = list(plan.assignment)
        dnn = self.environment.ds[dnn_index]
        cpu_by_node = dict(plan.required_cpu)
        bad_nodes = {
            node_index
            for node_index, required_cpu in plan.required_cpu
            if not snapshot.node_online[node_index]
            or snapshot.node_available_cpu[node_index] + 1e-12 < required_cpu
        }
        if not bad_nodes:
            return tuple(assignment)
        reserved: dict[int, float] = {
            node_index: required
            for node_index, required in cpu_by_node.items()
            if node_index not in bad_nodes
        }
        replacement_by_node: dict[int, int] = {}
        for bad_node in sorted(bad_nodes):
            required = cpu_by_node[bad_node]
            level = self.environment.nodes[bad_node].level
            alternatives = [
                node_index
                for node_index, node in enumerate(self.environment.nodes)
                if node.level == level
                and snapshot.node_online[node_index]
                and snapshot.node_available_cpu[node_index]
                - reserved.get(node_index, 0.0)
                + 1e-12
                >= required
            ]
            if not alternatives:
                return None
            replacement = max(
                alternatives,
                key=lambda node_index: (
                    snapshot.node_available_cpu[node_index]
                    - reserved.get(node_index, 0.0),
                    -node_index,
                ),
            )
            replacement_by_node[bad_node] = replacement
            reserved[replacement] = reserved.get(replacement, 0.0) + required
        for task_index in range(len(dnn.tasks)):
            assignment[task_index] = replacement_by_node.get(
                assignment[task_index], assignment[task_index]
            )
        return tuple(assignment)

    def _fallback_assignment(
        self,
        dnn_index: int,
        snapshot: ObservationSnapshot,
    ) -> tuple[int, ...] | None:
        dnn = self.environment.ds[dnn_index]
        remaining = list(snapshot.node_available_cpu)
        eligible = [
            node_index
            for node_index, node in enumerate(self.environment.nodes)
            if node.level >= 2 and snapshot.node_online[node_index]
        ]
        assignment: list[int] = []
        for task in dnn.tasks:
            choices = [
                node_index
                for node_index in eligible
                if remaining[node_index] + 1e-12 >= task.cpu_need
            ]
            if not choices:
                return None
            node_index = max(
                choices,
                key=lambda index: (
                    remaining[index],
                    self.environment.predict_node_continuous_availability(
                        index, 1, snapshot
                    )[0],
                    -index,
                ),
            )
            assignment.append(node_index)
            remaining[node_index] -= task.cpu_need
        return tuple(assignment)

    @staticmethod
    def _candidate_rejection_reason(candidate: CandidateStateSnapshot) -> str:
        if candidate.node_availability_score <= 0.0:
            return "node_offline"
        if candidate.constraint_violations:
            return candidate.constraint_violations[0]
        return "fallback_failed"

    def _decision(
        self,
        *,
        accepted: bool,
        path: str,
        reason: str | None,
        plan_id: str | None,
        assignment: tuple[int, ...] | None,
        candidate: CandidateStateSnapshot | None,
        repository: PlanRepository,
        evaluated_version: int,
        revalidation_count: int,
        started_at: float,
        request_budget_ms: float,
        feasible_count: int,
        budget_exhausted: bool,
        early_stop_reason: str | None,
        utility_trace: tuple[float, ...],
    ) -> DispatchDecision:
        return DispatchDecision(
            accepted=accepted,
            path=path,
            rejection_reason=reason,
            plan_id=plan_id,
            assignment=assignment,
            candidate=candidate,
            repository_version=repository.repository_version,
            evaluated_environment_state_version=evaluated_version,
            committed_environment_state_version=(
                self.environment.environment_state_version if accepted else None
            ),
            full_revalidation_count=revalidation_count,
            online_runtime_ms=self._elapsed_ms(started_at),
            configured_online_budget_ms=self._base_online_budget_ms,
            request_online_budget_ms=request_budget_ms,
            feasible_candidate_count=feasible_count,
            budget_exhausted=budget_exhausted,
            early_stop_reason=early_stop_reason,
            best_utility_after_each_evaluation=utility_trace,
            control_decision_id=self._control_decision_id,
        )

    def dispatch(
        self,
        dnn_index: int,
        request: InferenceRequest,
        repository: PlanRepository,
    ) -> DispatchDecision:
        """顺序处理一个请求；成功返回前已立即提交资源占用。"""
        started_at = self._clock()
        profile_id = self.environment.profile_id_for_dnn(dnn_index)
        if profile_id != request.profile_id:
            raise ValueError("request profile does not match environment DNN")
        if self.environment.ds[dnn_index].initiateNode != request.initiate_node:
            raise ValueError("request origin does not match environment DNN")
        snapshot = self.environment.capture_observation_snapshot("online_admission")
        evaluated_version = snapshot.environment_state_version
        origin_group = self.environment.origin_group_for_node(request.initiate_node)
        lookup = repository.lookup(
            profile_id=profile_id,
            origin_group=origin_group,
            current_slot=snapshot.slot,
            topology_version=snapshot.topology_version,
            semantic_versions=self.semantic_versions,
            node_count=len(self.environment.nodes),
        )
        cheap_delay_ms = min(
            (plan.planning_reference_delay_ms for plan in lookup.candidates),
            default=0.0,
        )
        deadline_limited_budget = max(
            0.0,
            self.config.deadline_safety_ratio
            * (request.deadline_ms - cheap_delay_ms),
        )
        request_budget_ms = min(
            self._base_online_budget_ms,
            deadline_limited_budget,
        )

        options: list[tuple[DeploymentPlan, tuple[int, ...], str]] = []
        quick_reasons: list[str] = []
        for plan in sorted(
            lookup.candidates,
            key=lambda item: (-self._plan_reference_score(item, request), item.plan_id),
        ):
            reason = self._quick_filter(plan, snapshot)
            if reason is None:
                options.append((plan, plan.assignment, "plan_direct"))
                continue
            quick_reasons.append(reason)
            repaired = self._repair_assignment(dnn_index, plan, snapshot)
            if repaired is not None and repaired != plan.assignment:
                options.append((plan, repaired, "plan_repaired"))

        feasible: list[
            tuple[float, DeploymentPlan | None, tuple[int, ...], str, CandidateStateSnapshot]
        ] = []
        revalidation_count = 0
        feasible_count = 0
        utility_trace: list[float] = []
        best_utility = float("-inf")
        evaluation_durations: list[float] = []
        early_stop_reason: str | None = None
        rejection_reasons = list(quick_reasons)
        evaluation_limit = (
            self.config.full_revalidation_top_k
            if self.config.revalidation_mode == "fixed_top_k"
            else self.config.hard_candidate_limit
        )
        for plan, assignment, path in options[:evaluation_limit]:
            if not self._within_budget(started_at, request_budget_ms):
                rejection_reasons.append("online_budget_exceeded")
                early_stop_reason = "budget_exhausted"
                break
            evaluation_started = self._clock()
            candidate = self.environment.predict_candidate_state(
                dnn_index,
                list(assignment),
                snapshot,
            )
            evaluation_durations.append((self._clock() - evaluation_started) * 1000.0)
            revalidation_count += 1
            if (
                candidate.is_strictly_feasible
                and candidate.node_availability_score
                >= self.config.minimum_availability
            ):
                feasible.append(
                    (
                        self._candidate_score(candidate, request),
                        plan,
                        assignment,
                        path,
                        candidate,
                    )
                )
                feasible_count += 1
                best_utility = max(best_utility, self._candidate_score(candidate, request))
            else:
                rejection_reasons.append(
                    "availability_below_threshold"
                    if candidate.is_strictly_feasible
                    else self._candidate_rejection_reason(candidate)
                )
            utility_trace.append(max(0.0, best_utility))
            if self.config.revalidation_mode == "anytime" and evaluation_durations:
                remaining = request_budget_ms - self._elapsed_ms(started_at)
                mean_evaluation_ms = sum(evaluation_durations) / len(evaluation_durations)
                if remaining < mean_evaluation_ms:
                    early_stop_reason = "insufficient_budget_for_next_evaluation"
                    break
        if (
            early_stop_reason is None
            and len(options) > evaluation_limit
            and revalidation_count >= evaluation_limit
        ):
            early_stop_reason = (
                "fixed_top_k_limit"
                if self.config.revalidation_mode == "fixed_top_k"
                else "hard_candidate_limit"
            )

        fallback_assignment: tuple[int, ...] | None = None
        if not feasible and self._within_budget(started_at, request_budget_ms):
            fallback_assignment = self._fallback_assignment(dnn_index, snapshot)
            if fallback_assignment is not None:
                fallback = self.environment.predict_candidate_state(
                    dnn_index,
                    list(fallback_assignment),
                    snapshot,
                )
                revalidation_count += 1
                if (
                    fallback.is_strictly_feasible
                    and fallback.node_availability_score
                    >= self.config.minimum_availability
                ):
                    feasible.append(
                        (
                            self._candidate_score(fallback, request),
                            None,
                            fallback_assignment,
                            "fast_fallback",
                            fallback,
                        )
                    )
                    feasible_count += 1
                    best_utility = max(best_utility, self._candidate_score(fallback, request))
                else:
                    rejection_reasons.append(
                        "availability_below_threshold"
                        if fallback.is_strictly_feasible
                        else self._candidate_rejection_reason(fallback)
                    )
                utility_trace.append(max(0.0, best_utility))

        for _, plan, assignment, path, candidate in sorted(
            feasible,
            key=lambda item: (-item[0], item[1].plan_id if item[1] else ""),
        ):
            if not self._within_budget(started_at, request_budget_ms):
                rejection_reasons.append("online_budget_exceeded")
                break
            try:
                committed = self.environment.commit_assignment(
                    dnn_index,
                    list(assignment),
                    expected_environment_state_version=evaluated_version,
                )
            except StaleEnvironmentStateError:
                if not self._within_budget(started_at, request_budget_ms):
                    rejection_reasons.append("online_budget_exceeded")
                    break
                refreshed = self.environment.capture_observation_snapshot(
                    "online_admission"
                )
                evaluated_version = refreshed.environment_state_version
                candidate = self.environment.predict_candidate_state(
                    dnn_index,
                    list(assignment),
                    refreshed,
                )
                revalidation_count += 1
                if (
                    not candidate.is_strictly_feasible
                    or candidate.node_availability_score
                    < self.config.minimum_availability
                ):
                    rejection_reasons.append(
                        "availability_below_threshold"
                        if candidate.is_strictly_feasible
                        else self._candidate_rejection_reason(candidate)
                    )
                    continue
                try:
                    committed = self.environment.commit_assignment(
                        dnn_index,
                        list(assignment),
                        expected_environment_state_version=evaluated_version,
                    )
                except StaleEnvironmentStateError:
                    rejection_reasons.append("stale_environment_state")
                    break
                except InfeasibleAssignmentError as error:
                    rejection_reasons.extend(str(error).split(","))
                    continue
                return self._decision(
                    accepted=True,
                    path=path,
                    reason=None,
                    plan_id=plan.plan_id if plan else None,
                    assignment=assignment,
                    candidate=committed,
                    repository=repository,
                    evaluated_version=evaluated_version,
                    revalidation_count=revalidation_count,
                    started_at=started_at,
                    request_budget_ms=request_budget_ms,
                    feasible_count=feasible_count,
                    budget_exhausted=not self._within_budget(started_at, request_budget_ms),
                    early_stop_reason=early_stop_reason,
                    utility_trace=tuple(utility_trace),
                )
            except InfeasibleAssignmentError as error:
                rejection_reasons.extend(str(error).split(","))
                continue
            return self._decision(
                accepted=True,
                path=path,
                reason=None,
                plan_id=plan.plan_id if plan else None,
                assignment=assignment,
                candidate=committed,
                repository=repository,
                evaluated_version=evaluated_version,
                revalidation_count=revalidation_count,
                started_at=started_at,
                request_budget_ms=request_budget_ms,
                feasible_count=feasible_count,
                budget_exhausted=not self._within_budget(started_at, request_budget_ms),
                early_stop_reason=early_stop_reason,
                utility_trace=tuple(utility_trace),
            )

        reason = (
            "online_budget_exceeded"
            if not self._within_budget(started_at, request_budget_ms)
            else next(
                (value for value in rejection_reasons if value),
                lookup.rejection_reason or "fallback_failed",
            )
        )
        return self._decision(
            accepted=False,
            path="rejected",
            reason=reason,
            plan_id=None,
            assignment=None,
            candidate=None,
            repository=repository,
            evaluated_version=evaluated_version,
            revalidation_count=revalidation_count,
            started_at=started_at,
            request_budget_ms=request_budget_ms,
            feasible_count=feasible_count,
            budget_exhausted=not self._within_budget(started_at, request_budget_ms),
            early_stop_reason=early_stop_reason,
            utility_trace=tuple(utility_trace),
        )
