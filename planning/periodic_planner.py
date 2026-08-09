from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

from core.environment import Environment
from models import ObservationSnapshot
from planning.repository import (
    AtomicPlanRepository,
    DeploymentPlan,
    PlanRepository,
    build_plan_repository,
    make_deployment_plan,
)
from planning.control_models import JointBudgetDecision
from proposed import AllDNNRefactor, CustomizedSearchConfig
from semantics import PERIODIC_SEMANTIC_VERSIONS


@dataclass(frozen=True)
class PlanningTarget:
    dnn_index: int
    profile_id: str
    origin_group: int

    def __post_init__(self) -> None:
        if self.dnn_index < 0 or not self.profile_id or self.origin_group < 0:
            raise ValueError("invalid planning target")


@dataclass(frozen=True)
class PeriodicPlannerConfig:
    planning_interval_slots: int = 50
    minimum_replanning_interval_slots: int = 10
    plan_ttl_slots: int = 100
    max_pool_size: int = 20
    sample_count: int = 80
    candidate_generator_mode: str = "customized"
    customized_population_size: int = 12
    customized_generations: int = 4
    customized_archive_capacity: int = 64
    customized_random_seed: int = 0
    customized_max_evaluations: int | None = None
    customized_wall_time_budget_ms: float | None = None
    min_publish_valid_ratio: float = 0.5
    max_node_load_drift: float = 0.25
    max_link_load_drift: float = 0.25
    max_plan_overlap: float = 0.85
    dispatch_history_window: int = 20
    min_direct_hit_rate: float = 0.4
    max_fallback_rate: float = 0.3
    profile_distribution_drift_threshold: float = 0.25

    def __post_init__(self) -> None:
        for name, value in (
            ("planning_interval_slots", self.planning_interval_slots),
            (
                "minimum_replanning_interval_slots",
                self.minimum_replanning_interval_slots,
            ),
            ("plan_ttl_slots", self.plan_ttl_slots),
            ("max_pool_size", self.max_pool_size),
            ("sample_count", self.sample_count),
            ("customized_population_size", self.customized_population_size),
            ("customized_generations", self.customized_generations),
            ("customized_archive_capacity", self.customized_archive_capacity),
            ("dispatch_history_window", self.dispatch_history_window),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.customized_population_size % 2 != 0:
            raise ValueError("customized_population_size must be even")
        if self.candidate_generator_mode not in {
            "customized",
            "lightweight",
            "customized_with_fallback",
        }:
            raise ValueError("unsupported candidate_generator_mode")
        if self.customized_max_evaluations is not None and self.customized_max_evaluations <= 0:
            raise ValueError("customized_max_evaluations must be positive")
        if (
            self.customized_wall_time_budget_ms is not None
            and self.customized_wall_time_budget_ms <= 0.0
        ):
            raise ValueError("customized_wall_time_budget_ms must be positive")
        for name, value in (
            ("min_publish_valid_ratio", self.min_publish_valid_ratio),
            ("max_node_load_drift", self.max_node_load_drift),
            ("max_link_load_drift", self.max_link_load_drift),
            ("max_plan_overlap", self.max_plan_overlap),
            ("min_direct_hit_rate", self.min_direct_hit_rate),
            ("max_fallback_rate", self.max_fallback_rate),
            (
                "profile_distribution_drift_threshold",
                self.profile_distribution_drift_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PlanningJob:
    job_id: int
    trigger_reason: str
    start_slot: int
    ready_slot: int
    planning_runtime_ms: float
    source_snapshot_version: int
    source_environment_state_version: int
    source_topology_version: int
    base_repository_version: int
    proposed_plans: tuple[DeploymentPlan, ...]
    control_decision_id: str | None = None


@dataclass(frozen=True)
class PlanningEvent:
    status: str
    job_id: int | None
    repository_version: int
    trigger_reason: str | None = None
    proposed_count: int = 0
    valid_count: int = 0
    valid_ratio: float = 0.0
    planning_runtime_ms: float = 0.0


@dataclass(frozen=True)
class PublicationCandidateAudit:
    """记录一个规划候选在发布快照上的复验结果。"""

    job_id: int
    plan_id: str
    trigger_reason: str
    profile_id: str
    origin_group: int
    generator_mode: str
    generator_version: str
    search_seed: int | None
    source_generation: int | None
    publication_slot: int
    source_snapshot_version: int
    publication_snapshot_version: int
    source_environment_state_version: int
    publication_environment_state_version: int
    accepted: bool
    primary_reason: str
    reasons: tuple[str, ...]
    offline_nodes: tuple[int, ...] = ()
    constraint_violations: tuple[str, ...] = ()
    constraint_violation: float | None = None
    fixed_point_converged: bool | None = None
    fixed_point_iterations: int | None = None
    fixed_point_relative_residual: float | None = None
    estimated_delay_ms: float | None = None


@dataclass
class PlanningStatistics:
    started_jobs: int = 0
    published_jobs: int = 0
    retained_old_repository_jobs: int = 0
    suppressed_triggers: int = 0
    proposed_candidates: int = 0
    publication_valid_candidates: int = 0
    publication_rejected_candidates: int = 0
    search_assignments_generated: int = 0
    search_constraint_rejected_candidates: int = 0
    search_nonconverged_candidates: int = 0
    search_feasible_candidates: int = 0
    search_pareto_candidates: int = 0
    search_selected_candidates: int = 0
    total_planning_runtime_ms: float = 0.0


PlanningRuntimeProvider = Callable[[ObservationSnapshot, tuple[DeploymentPlan, ...]], float]


class PeriodicPlanner:
    """以仿真时隙表达后台规划完成时间，并在完成时复验后发布。"""

    def __init__(
        self,
        environment: Environment,
        repository_store: AtomicPlanRepository,
        targets: tuple[PlanningTarget, ...],
        config: PeriodicPlannerConfig | None = None,
        semantic_versions: tuple[tuple[str, str], ...] | None = None,
        planning_runtime_provider: PlanningRuntimeProvider | None = None,
    ) -> None:
        if not targets:
            raise ValueError("periodic planner requires at least one target")
        self.environment = environment
        self.repository_store = repository_store
        self.targets = targets
        self.config = config or PeriodicPlannerConfig()
        self.semantic_versions = (
            semantic_versions or PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        )
        if dict(self.semantic_versions).get("objective_semantics_version") != (
            environment.objective_semantics_version
        ):
            raise ValueError("planner semantics do not match environment semantics")
        self.planning_runtime_provider = planning_runtime_provider
        self.algorithm = AllDNNRefactor(environment)
        self.active_job: PlanningJob | None = None
        self.statistics = PlanningStatistics()
        self.publication_audits: list[PublicationCandidateAudit] = []
        self._job_sequence = 0
        self._last_start_slot: int | None = None
        self._last_monitor_snapshot: ObservationSnapshot | None = None
        self._dispatch_history: deque[tuple[str, str]] = deque(
            maxlen=self.config.dispatch_history_window
        )
        self._baseline_profile_distribution: dict[str, float] = {}
        self._current_control_decision: JointBudgetDecision | None = None
        self._dynamic_planning_interval_slots = self.config.planning_interval_slots
        self._pool_budget_by_key: tuple[tuple[str, int, int], ...] = ()
        for target in targets:
            if environment.profile_id_for_dnn(target.dnn_index) != target.profile_id:
                raise ValueError("planning target profile does not match environment DNN")

    def _warm_start_plans(
        self,
        snapshot: ObservationSnapshot,
        repository: PlanRepository,
    ) -> list[DeploymentPlan]:
        """把旧仓库结构在启动快照上重新评价，不复用旧性能结论。"""
        target_by_key = {
            (target.profile_id, target.origin_group): target for target in self.targets
        }
        warmed: list[DeploymentPlan] = []
        for old_plan in repository.plans:
            target = target_by_key.get((old_plan.profile_id, old_plan.origin_group))
            if target is None or old_plan.topology_version != snapshot.topology_version:
                continue
            if any(
                node_index < 0 or node_index >= len(self.environment.nodes)
                for node_index in old_plan.assignment
            ):
                continue
            candidate = self.environment.predict_candidate_state(
                target.dnn_index,
                list(old_plan.assignment),
                snapshot,
            )
            if (
                not candidate.is_strictly_feasible
                or not candidate.fixed_point_converged
            ):
                continue
            warmed.append(
                make_deployment_plan(
                    self.environment,
                    target.dnn_index,
                    candidate,
                    target.profile_id,
                    target.origin_group,
                    created_slot=snapshot.slot,
                    ttl_slots=self.config.plan_ttl_slots,
                    semantic_versions=self.semantic_versions,
                    plan_id=old_plan.plan_id,
                    generator_mode=old_plan.generator_mode,
                    generator_version=old_plan.generator_version,
                    search_seed=old_plan.search_seed,
                    source_generation=old_plan.source_generation,
                )
            )
        return warmed

    def _search_at_snapshot(
        self,
        snapshot: ObservationSnapshot,
        repository: PlanRepository,
    ) -> tuple[DeploymentPlan, ...]:
        proposed = self._warm_start_plans(snapshot, repository)
        quotas = {
            (profile_id, origin_group): budget
            for profile_id, origin_group, budget in self._pool_budget_by_key
        }
        for target in self.targets:
            target_pool_size = quotas.get(
                (target.profile_id, target.origin_group),
                self.config.max_pool_size,
            )
            if self.config.candidate_generator_mode == "lightweight":
                target_plans = self.algorithm.search_plan_pool(
                    target.dnn_index,
                    snapshot,
                    target.profile_id,
                    target.origin_group,
                    max_pool_size=target_pool_size,
                    sample_count=self.config.sample_count,
                    ttl_slots=self.config.plan_ttl_slots,
                    semantic_versions=self.semantic_versions,
                )
            else:
                target_plans = self.algorithm.search_customized_plan_pool(
                    target.dnn_index,
                    snapshot,
                    target.profile_id,
                    target.origin_group,
                    max_pool_size=target_pool_size,
                    ttl_slots=self.config.plan_ttl_slots,
                    semantic_versions=self.semantic_versions,
                    search_config=CustomizedSearchConfig(
                        population_size=self.config.customized_population_size,
                        generations=self.config.customized_generations,
                        archive_capacity=self.config.customized_archive_capacity,
                        random_seed=self.config.customized_random_seed,
                        max_evaluations=self.config.customized_max_evaluations,
                        wall_time_budget_ms=self.config.customized_wall_time_budget_ms,
                    ),
                )
                if (
                    not target_plans
                    and self.config.candidate_generator_mode
                    == "customized_with_fallback"
                ):
                    target_plans = self.algorithm.search_plan_pool(
                        target.dnn_index,
                        snapshot,
                        target.profile_id,
                        target.origin_group,
                        max_pool_size=target_pool_size,
                        sample_count=self.config.sample_count,
                        ttl_slots=self.config.plan_ttl_slots,
                        semantic_versions=self.semantic_versions,
                    )
            proposed.extend(target_plans)
            search = self.algorithm.last_plan_pool_search_statistics
            self.statistics.search_assignments_generated += search.generated_assignments
            self.statistics.search_constraint_rejected_candidates += (
                search.constraint_rejected_candidates
            )
            self.statistics.search_nonconverged_candidates += (
                search.nonconverged_candidates
            )
            self.statistics.search_feasible_candidates += search.feasible_candidates
            self.statistics.search_pareto_candidates += search.pareto_candidates
            self.statistics.search_selected_candidates += search.selected_candidates
        staged = build_plan_repository(
            repository_version=repository.repository_version + 1,
            topology_version=snapshot.topology_version,
            semantic_versions=self.semantic_versions,
            published_slot=snapshot.slot,
            plans=proposed,
            max_pool_size=self.config.max_pool_size,
            max_node_overlap=self.config.max_plan_overlap,
            pool_budget_by_key=self._pool_budget_by_key or None,
        )
        return staged.plans

    def start(self, trigger_reason: str = "fixed_interval", force: bool = False) -> PlanningEvent:
        """在当前槽启动规划计算，但不立即发布其结果。"""
        current_slot = self.environment.current_slot
        current_repository = self.repository_store.snapshot()
        if self.active_job is not None:
            self.statistics.suppressed_triggers += 1
            return PlanningEvent(
                "already_running",
                self.active_job.job_id,
                current_repository.repository_version,
                trigger_reason,
            )
        if (
            not force
            and self._last_start_slot is not None
            and current_slot - self._last_start_slot
            < self.config.minimum_replanning_interval_slots
        ):
            self.statistics.suppressed_triggers += 1
            return PlanningEvent(
                "suppressed_minimum_interval",
                None,
                current_repository.repository_version,
                trigger_reason,
            )

        snapshot = self.environment.capture_observation_snapshot("planning")
        wall_started = time.monotonic()
        proposed = self._search_at_snapshot(snapshot, current_repository)
        measured_runtime_ms = (time.monotonic() - wall_started) * 1000.0
        runtime_ms = (
            self.planning_runtime_provider(snapshot, proposed)
            if self.planning_runtime_provider is not None
            else measured_runtime_ms
        )
        if runtime_ms < 0.0:
            raise ValueError("planning runtime must be non-negative")
        runtime_slots = max(1, math.ceil(runtime_ms / self.environment.slot_length))
        self._job_sequence += 1
        self.active_job = PlanningJob(
            job_id=self._job_sequence,
            trigger_reason=trigger_reason,
            start_slot=current_slot,
            ready_slot=current_slot + runtime_slots,
            planning_runtime_ms=runtime_ms,
            source_snapshot_version=snapshot.snapshot_version,
            source_environment_state_version=snapshot.environment_state_version,
            source_topology_version=snapshot.topology_version,
            base_repository_version=current_repository.repository_version,
            proposed_plans=proposed,
            control_decision_id=(
                self._current_control_decision.decision_id
                if self._current_control_decision is not None
                else None
            ),
        )
        self._last_start_slot = current_slot
        self._last_monitor_snapshot = snapshot
        if self._dispatch_history:
            counts = Counter(profile for _, profile in self._dispatch_history)
            total = sum(counts.values())
            self._baseline_profile_distribution = {
                profile: count / total for profile, count in counts.items()
            }
            self._dispatch_history.clear()
        self.statistics.started_jobs += 1
        self.statistics.proposed_candidates += len(proposed)
        self.statistics.total_planning_runtime_ms += runtime_ms
        return PlanningEvent(
            "started",
            self.active_job.job_id,
            current_repository.repository_version,
            trigger_reason,
            proposed_count=len(proposed),
            planning_runtime_ms=runtime_ms,
        )

    def _publication_candidates(
        self,
        job: PlanningJob,
        snapshot: ObservationSnapshot,
    ) -> list[DeploymentPlan]:
        target_by_key = {
            (target.profile_id, target.origin_group): target for target in self.targets
        }
        valid: list[DeploymentPlan] = []
        for plan in job.proposed_plans:
            reasons: list[str] = []
            target = target_by_key.get((plan.profile_id, plan.origin_group))
            if target is None:
                reasons.append("target_missing")
            if plan.topology_version != snapshot.topology_version:
                reasons.append("topology_mismatch")
            out_of_range = tuple(
                sorted(
                    {
                        node_index
                        for node_index in plan.assignment
                        if node_index < 0 or node_index >= len(self.environment.nodes)
                    }
                )
            )
            if out_of_range:
                reasons.append("assignment_out_of_range")
            offline_nodes = tuple(
                sorted(
                    {
                        node_index
                        for node_index in plan.assignment
                        if 0 <= node_index < len(self.environment.nodes)
                        and not snapshot.node_online[node_index]
                    }
                )
            )
            if offline_nodes:
                reasons.append("node_offline")

            candidate = None
            if target is not None and not out_of_range and not reasons:
                candidate = self.environment.predict_candidate_state(
                    target.dnn_index,
                    list(plan.assignment),
                    snapshot,
                )
                reasons.extend(candidate.constraint_violations)
                if (
                    candidate.constraint_violation > 1e-12
                    and not candidate.constraint_violations
                ):
                    reasons.append("constraint_violation_other")
                if not candidate.fixed_point_converged:
                    reasons.append("fixed_point_not_converged")

            if reasons:
                self.publication_audits.append(
                    PublicationCandidateAudit(
                        job_id=job.job_id,
                        plan_id=plan.plan_id,
                        trigger_reason=job.trigger_reason,
                        profile_id=plan.profile_id,
                        origin_group=plan.origin_group,
                        generator_mode=plan.generator_mode,
                        generator_version=plan.generator_version,
                        search_seed=plan.search_seed,
                        source_generation=plan.source_generation,
                        publication_slot=snapshot.slot,
                        source_snapshot_version=job.source_snapshot_version,
                        publication_snapshot_version=snapshot.snapshot_version,
                        source_environment_state_version=(
                            job.source_environment_state_version
                        ),
                        publication_environment_state_version=(
                            snapshot.environment_state_version
                        ),
                        accepted=False,
                        primary_reason=reasons[0],
                        reasons=tuple(dict.fromkeys(reasons)),
                        offline_nodes=offline_nodes,
                        constraint_violations=(
                            candidate.constraint_violations if candidate else ()
                        ),
                        constraint_violation=(
                            candidate.constraint_violation if candidate else None
                        ),
                        fixed_point_converged=(
                            candidate.fixed_point_converged if candidate else None
                        ),
                        fixed_point_iterations=(
                            candidate.fixed_point_iterations if candidate else None
                        ),
                        fixed_point_relative_residual=(
                            candidate.fixed_point_relative_residual
                            if candidate
                            else None
                        ),
                        estimated_delay_ms=(
                            candidate.estimated_delay_ms if candidate else None
                        ),
                    )
                )
                continue
            assert candidate is not None
            valid.append(
                make_deployment_plan(
                    self.environment,
                    target.dnn_index,
                    candidate,
                    target.profile_id,
                    target.origin_group,
                    created_slot=snapshot.slot,
                    ttl_slots=self.config.plan_ttl_slots,
                    semantic_versions=self.semantic_versions,
                    plan_id=plan.plan_id,
                    generator_mode=plan.generator_mode,
                    generator_version=plan.generator_version,
                    search_seed=plan.search_seed,
                    source_generation=plan.source_generation,
                )
            )
            self.publication_audits.append(
                PublicationCandidateAudit(
                    job_id=job.job_id,
                    plan_id=plan.plan_id,
                    trigger_reason=job.trigger_reason,
                    profile_id=plan.profile_id,
                    origin_group=plan.origin_group,
                    generator_mode=plan.generator_mode,
                    generator_version=plan.generator_version,
                    search_seed=plan.search_seed,
                    source_generation=plan.source_generation,
                    publication_slot=snapshot.slot,
                    source_snapshot_version=job.source_snapshot_version,
                    publication_snapshot_version=snapshot.snapshot_version,
                    source_environment_state_version=(
                        job.source_environment_state_version
                    ),
                    publication_environment_state_version=(
                        snapshot.environment_state_version
                    ),
                    accepted=True,
                    primary_reason="valid",
                    reasons=("valid",),
                    constraint_violation=candidate.constraint_violation,
                    fixed_point_converged=candidate.fixed_point_converged,
                    fixed_point_iterations=candidate.fixed_point_iterations,
                    fixed_point_relative_residual=(
                        candidate.fixed_point_relative_residual
                    ),
                    estimated_delay_ms=candidate.estimated_delay_ms,
                )
            )
        return valid

    def poll(self, force_ready: bool = False) -> PlanningEvent:
        """完成已到期作业的发布复验；未到期时旧仓库保持不变。"""
        current_repository = self.repository_store.snapshot()
        job = self.active_job
        if job is None:
            return PlanningEvent(
                "idle", None, current_repository.repository_version
            )
        if not force_ready and self.environment.current_slot < job.ready_slot:
            return PlanningEvent(
                "not_ready",
                job.job_id,
                current_repository.repository_version,
                job.trigger_reason,
                proposed_count=len(job.proposed_plans),
                planning_runtime_ms=job.planning_runtime_ms,
            )

        publication_snapshot = self.environment.capture_observation_snapshot(
            "publication_revalidation"
        )
        valid = self._publication_candidates(job, publication_snapshot)
        proposed_count = len(job.proposed_plans)
        valid_ratio = len(valid) / proposed_count if proposed_count else 0.0
        self.statistics.publication_valid_candidates += len(valid)
        self.statistics.publication_rejected_candidates += proposed_count - len(valid)
        self._last_monitor_snapshot = publication_snapshot
        self.active_job = None

        if valid_ratio < self.config.min_publish_valid_ratio or not valid:
            self.statistics.retained_old_repository_jobs += 1
            return PlanningEvent(
                "retained_old_repository",
                job.job_id,
                current_repository.repository_version,
                job.trigger_reason,
                proposed_count,
                len(valid),
                valid_ratio,
                job.planning_runtime_ms,
            )

        new_repository = build_plan_repository(
            repository_version=current_repository.repository_version + 1,
            topology_version=publication_snapshot.topology_version,
            semantic_versions=self.semantic_versions,
            published_slot=publication_snapshot.slot,
            plans=valid,
            max_pool_size=self.config.max_pool_size,
            max_node_overlap=self.config.max_plan_overlap,
            pool_budget_by_key=self._pool_budget_by_key or None,
        )
        if not self.repository_store.publish(
            new_repository,
            expected_current_version=current_repository.repository_version,
        ):
            self.statistics.retained_old_repository_jobs += 1
            return PlanningEvent(
                "publication_conflict",
                job.job_id,
                self.repository_store.snapshot().repository_version,
                job.trigger_reason,
                proposed_count,
                len(valid),
                valid_ratio,
                job.planning_runtime_ms,
            )
        self.statistics.published_jobs += 1
        return PlanningEvent(
            "published",
            job.job_id,
            new_repository.repository_version,
            job.trigger_reason,
            proposed_count,
            len(valid),
            valid_ratio,
            job.planning_runtime_ms,
        )

    def _trigger_reason(self, snapshot: ObservationSnapshot) -> str | None:
        baseline = self._last_monitor_snapshot
        repository = self.repository_store.snapshot()
        if repository.plans and all(
            snapshot.slot > plan.expires_slot for plan in repository.plans
        ):
            return "repository_expired"
        if baseline is None:
            return "cold_start"
        if snapshot.topology_version != baseline.topology_version:
            return "topology_changed"
        if snapshot.node_online != baseline.node_online:
            return "availability_changed"
        node_drift = max(
            (
                abs(current - previous)
                for current, previous in zip(snapshot.node_loads, baseline.node_loads)
            ),
            default=0.0,
        )
        if node_drift >= self.config.max_node_load_drift:
            return "node_load_drift"
        link_drift = max(
            (
                abs(current - previous)
                for current, previous in zip(
                    snapshot.directional_link_loads,
                    baseline.directional_link_loads,
                )
            ),
            default=0.0,
        )
        if link_drift >= self.config.max_link_load_drift:
            return "link_load_drift"
        if len(self._dispatch_history) >= self.config.dispatch_history_window:
            direct_rate = sum(
                1 for path, _ in self._dispatch_history if path == "plan_direct"
            ) / len(self._dispatch_history)
            fallback_rate = sum(
                1 for path, _ in self._dispatch_history if path == "fast_fallback"
            ) / len(self._dispatch_history)
            if direct_rate < self.config.min_direct_hit_rate:
                return "direct_hit_rate_low"
            if fallback_rate > self.config.max_fallback_rate:
                return "fallback_rate_high"
            if self._baseline_profile_distribution:
                counts = Counter(profile for _, profile in self._dispatch_history)
                total = sum(counts.values())
                current_distribution = {
                    profile: count / total for profile, count in counts.items()
                }
                profiles = set(current_distribution) | set(
                    self._baseline_profile_distribution
                )
                total_variation = 0.5 * sum(
                    abs(
                        current_distribution.get(profile, 0.0)
                        - self._baseline_profile_distribution.get(profile, 0.0)
                    )
                    for profile in profiles
                )
                if total_variation >= self.config.profile_distribution_drift_threshold:
                    return "profile_distribution_drift"
        if (
            self._last_start_slot is None
            or snapshot.slot - self._last_start_slot
            >= self._dynamic_planning_interval_slots
        ):
            return "fixed_interval"
        return None

    def apply_joint_budget_decision(self, decision: JointBudgetDecision) -> None:
        """在无规划作业运行时原子更新下一周期的 T 与分键 K。"""
        if self.active_job is not None:
            raise RuntimeError("cannot change joint planning budget while a job is active")
        target_keys = {(target.profile_id, target.origin_group) for target in self.targets}
        decision_keys = {
            (profile_id, origin_group)
            for profile_id, origin_group, _ in decision.pool_budget_by_key
        }
        if target_keys - decision_keys:
            raise ValueError("joint decision is missing a planning target pool quota")
        self._current_control_decision = decision
        self._dynamic_planning_interval_slots = decision.action.planning_interval_slots
        self._pool_budget_by_key = decision.pool_budget_by_key

    def record_dispatch(self, path: str, profile_id: str) -> None:
        """记录已发生的在线决策结果，供后续时隙的重规划触发判断。"""
        if path not in {"plan_direct", "plan_repaired", "fast_fallback", "rejected"}:
            raise ValueError("unsupported dispatch path")
        if not profile_id:
            raise ValueError("profile_id must not be empty")
        self._dispatch_history.append((path, profile_id))

    def tick(self) -> tuple[PlanningEvent, ...]:
        """供时隙事件循环调用：先发布已完成作业，再检查新触发。"""
        events: list[PlanningEvent] = []
        publication = self.poll()
        if publication.status != "idle":
            events.append(publication)
        if self.active_job is None:
            monitor = self.environment.capture_observation_snapshot("planning")
            reason = self._trigger_reason(monitor)
            if reason is not None:
                events.append(self.start(reason))
        return tuple(events)

    def prewarm(self) -> PlanningEvent:
        """在正式观察窗口前完成第一版规划，同时保留规划开销统计。"""
        started = self.start("prewarm", force=True)
        if started.status != "started":
            return started
        return self.poll(force_ready=True)
