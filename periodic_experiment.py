from __future__ import annotations

import dataclasses
import math
import statistics
import time
from dataclasses import dataclass

from core.arrival import ArrivalTrace
from core.environment import Environment
from metrics import PeriodicExperimentMetrics
from models import DNNProfileCatalog, InferenceRequest, materialize_dnn
from planning.action_predictor import AnalyticActionPredictor
from planning.control_loop import JointBudgetControlLoop
from planning.control_models import JointBudgetAction, enumerate_joint_actions
from planning.control_state import CompletedDispatchRecord, CompletedPlanningRecord
from planning.joint_budget_controller import (
    DiscreteMPCJointBudgetController,
    FixedJointBudgetController,
    RuleBasedJointBudgetController,
)
from planning.joint_budget_runtime import JointBudgetRuntimeCoordinator
from planning.online_dispatcher import OnlineDispatcher, OnlineDispatcherConfig
from planning.request_dispatcher import (
    OnlineBaselineConfig,
    OnlineBaselineDispatcher,
    PerRequestCustomizedConfig,
    PerRequestCustomizedDispatcher,
)
from planning.periodic_planner import (
    PeriodicPlanner,
    PeriodicPlannerConfig,
    PlanningEvent,
    PlanningTarget,
)
from planning.repository import AtomicPlanRepository, PlanRepository
from semantics import PERIODIC_SEMANTIC_VERSIONS


@dataclass(frozen=True)
class PeriodicExperimentConfig:
    origin_nodes: tuple[int, ...]
    algorithm: str = "periodic_customized"
    bootstrap_mode: str = "prewarm"
    control_mode: str = "fixed"
    planning_intervals: tuple[int, ...] = (5, 10, 20)
    total_pool_budgets: tuple[int, ...] = (8, 20, 40)
    online_budgets_ms: tuple[float, ...] = (5.0, 10.0, 20.0)
    fixed_planning_interval_slots: int = 5
    fixed_total_pool_budget: int = 20
    fixed_online_budget_ms: float = 10.0
    planning_runtime_ms: float = 100.0
    planning_timing_mode: str = "configured"
    planning_sample_count: int = 20
    planning_candidate_generator_mode: str = "customized"
    planning_customized_population_size: int = 12
    planning_customized_generations: int = 4
    planning_customized_archive_capacity: int = 64
    planning_customized_random_seed: int = 0
    plan_ttl_slots: int = 20
    minimum_replanning_interval_slots: int = 2
    minimum_control_window_slots: int = 5
    planning_deadline_ms: float = 2_000.0
    drain_max_slots: int = 200
    online_timing_mode: str = "measured"
    deterministic_clock_step_ms: float = 0.02

    def __post_init__(self) -> None:
        if not self.origin_nodes or any(node < 0 for node in self.origin_nodes):
            raise ValueError("origin_nodes must contain valid node indices")
        if self.bootstrap_mode not in {"cold_start", "prewarm"}:
            raise ValueError("unsupported bootstrap_mode")
        if self.control_mode not in {"fixed", "rule_based", "discrete_mpc"}:
            raise ValueError("unsupported control_mode")
        if self.algorithm not in {
            "periodic_customized",
            "periodic_lightweight",
            "online_fast_fallback",
            "per_request_customized",
            "random",
            "sa",
            "localfirst",
            "max_resource_fast",
            "rtbl",
        }:
            raise ValueError("unsupported scheduling algorithm")
        if self.online_timing_mode not in {"measured", "deterministic"}:
            raise ValueError("unsupported online_timing_mode")
        if self.planning_timing_mode not in {"measured", "configured"}:
            raise ValueError("unsupported planning_timing_mode")
        if self.planning_candidate_generator_mode not in {
            "customized",
            "lightweight",
            "customized_with_fallback",
        }:
            raise ValueError("unsupported planning candidate generator")
        if self.planning_runtime_ms < 0.0 or self.drain_max_slots < 0:
            raise ValueError("planning runtime and drain bound must be non-negative")


@dataclass
class _ActiveExecution:
    request: InferenceRequest
    dnn_index: int
    assignment: tuple[int, ...]
    completion_slot: int
    record_index: int
    candidate_delay_ms: float
    operational_stability: float
    total_energy: float
    delay_satisfaction: float
    energy_satisfaction: float


class _DeterministicClock:
    def __init__(self, step_ms: float) -> None:
        if step_ms < 0.0:
            raise ValueError("clock step must be non-negative")
        self._value = 0.0
        self._step = step_ms / 1000.0

    def __call__(self) -> float:
        value = self._value
        self._value += self._step
        return value


class PeriodicExperimentRunner:
    """统一执行规划、控制、泊松批量接纳、运行结算和单次时隙推进。"""

    def __init__(
        self,
        environment: Environment,
        catalog: DNNProfileCatalog,
        arrival_trace: ArrivalTrace,
        config: PeriodicExperimentConfig,
    ) -> None:
        if environment.objective_semantics_version != PERIODIC_SEMANTIC_VERSIONS.objective:
            raise ValueError("periodic runner requires profiled periodic objective semantics")
        if any(node >= len(environment.nodes) for node in config.origin_nodes):
            raise ValueError("periodic origin node is outside the topology")
        self.environment = environment
        self.catalog = catalog
        self.arrival_trace = arrival_trace
        self.config = config
        self.semantic_versions = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        self.dispatch_records: list[CompletedDispatchRecord] = []
        self.planning_records: list[CompletedPlanningRecord] = []
        self._active: dict[int, _ActiveExecution] = {}
        self._job_start_slots: dict[int, int] = {}
        self._accepted_candidates: list[_ActiveExecution] = []
        self._completed_candidates: list[_ActiveExecution] = []
        self._runtime_failed = 0
        self._deadline_missed = 0
        self._goodput_sum = 0.0

        targets = self._install_planning_representatives()
        empty_repository = PlanRepository(
            repository_version=0,
            topology_version=environment.topology_version,
            semantic_versions=self.semantic_versions,
            published_slot=environment.current_slot,
            plans=(),
        )
        self.repository_store = AtomicPlanRepository(empty_repository)
        maximum_per_key = 40
        key_count = len(targets)
        if not 2 * key_count <= config.fixed_total_pool_budget <= maximum_per_key * key_count:
            raise ValueError("fixed pool budget cannot satisfy active planning keys")
        planner_config = PeriodicPlannerConfig(
            planning_interval_slots=config.fixed_planning_interval_slots,
            minimum_replanning_interval_slots=config.minimum_replanning_interval_slots,
            plan_ttl_slots=config.plan_ttl_slots,
            max_pool_size=maximum_per_key,
            sample_count=config.planning_sample_count,
            candidate_generator_mode=config.planning_candidate_generator_mode,
            customized_population_size=config.planning_customized_population_size,
            customized_generations=config.planning_customized_generations,
            customized_archive_capacity=config.planning_customized_archive_capacity,
            customized_random_seed=config.planning_customized_random_seed,
            min_publish_valid_ratio=0.1,
        )
        self.planner = PeriodicPlanner(
            environment,
            self.repository_store,
            targets,
            planner_config,
            planning_runtime_provider=(
                None
                if config.planning_timing_mode == "measured"
                else lambda snapshot, plans: config.planning_runtime_ms
            ),
        )
        clock = (
            _DeterministicClock(config.deterministic_clock_step_ms)
            if config.online_timing_mode == "deterministic"
            else time.monotonic
        )
        self.dispatcher = OnlineDispatcher(
            environment,
            OnlineDispatcherConfig(
                online_budget_ms=config.fixed_online_budget_ms,
                revalidation_mode="anytime",
            ),
            clock=clock,
        )
        self.request_dispatcher = PerRequestCustomizedDispatcher(
            environment,
            PerRequestCustomizedConfig(
                population_size=config.planning_customized_population_size,
                generations=config.planning_customized_generations,
                archive_capacity=config.planning_customized_archive_capacity,
                random_seed=config.planning_customized_random_seed,
            ),
            clock=clock,
        )
        baseline_algorithms = {
            "random",
            "sa",
            "localfirst",
            "max_resource_fast",
            "rtbl",
        }
        self.baseline_dispatcher = (
            OnlineBaselineDispatcher(
                environment,
                OnlineBaselineConfig(
                    algorithm=config.algorithm,
                    evaluation_budget=(
                        config.planning_customized_population_size
                        * config.planning_customized_generations
                    ),
                    random_seed=config.planning_customized_random_seed,
                ),
                clock=clock,
            )
            if config.algorithm in baseline_algorithms
            else None
        )
        self.controller = self._create_controller(key_count)
        self.control_loop = JointBudgetControlLoop(
            self.controller,
            JointBudgetRuntimeCoordinator(self.planner, self.dispatcher),
            minimum_control_window_slots=config.minimum_control_window_slots,
        )

    def _install_planning_representatives(self) -> tuple[PlanningTarget, ...]:
        origin_by_group: dict[int, int] = {}
        for origin in self.config.origin_nodes:
            origin_by_group.setdefault(
                self.environment.origin_group_for_node(origin), origin
            )
        targets: list[PlanningTarget] = []
        for profile in self.catalog.profiles:
            for origin_group, origin in sorted(origin_by_group.items()):
                representative = InferenceRequest(
                    request_id=0,
                    profile_id=profile.profile_id,
                    arrival_slot=0,
                    initiate_node=origin,
                    deadline_ms=self.config.planning_deadline_ms,
                    preference_stability=1 / 3,
                    preference_delay=1 / 3,
                    preference_energy=1 / 3,
                )
                dnn_index = len(self.environment.ds)
                self.environment.ds.append(materialize_dnn(profile, representative))
                targets.append(
                    PlanningTarget(dnn_index, profile.profile_id, origin_group)
                )
        return tuple(targets)

    def _create_controller(self, key_count: int):
        fixed_action = JointBudgetAction(
            self.config.fixed_planning_interval_slots,
            self.config.fixed_total_pool_budget,
            self.config.fixed_online_budget_ms,
        )
        predictor = AnalyticActionPredictor(self.environment.slot_length)
        if self.config.control_mode == "fixed":
            return FixedJointBudgetController(fixed_action, predictor)
        actions = enumerate_joint_actions(
            self.config.planning_intervals,
            self.config.total_pool_budgets,
            self.config.online_budgets_ms,
        )
        feasible_by_key_bounds = tuple(
            action
            for action in actions
            if 2 * key_count <= action.total_pool_budget <= 40 * key_count
        )
        if fixed_action not in feasible_by_key_bounds:
            feasible_by_key_bounds = tuple(sorted(set((*feasible_by_key_bounds, fixed_action))))
        if self.config.control_mode == "rule_based":
            return RuleBasedJointBudgetController(
                feasible_by_key_bounds,
                fixed_action,
                predictor,
                minimum_control_interval_slots=self.config.minimum_control_window_slots,
            )
        return DiscreteMPCJointBudgetController(
            feasible_by_key_bounds,
            predictor,
            conservative_action=fixed_action,
        )

    def _record_planning_events(self, events: tuple[PlanningEvent, ...]) -> None:
        for event in events:
            if event.job_id is None:
                continue
            if event.status == "started":
                self._job_start_slots[event.job_id] = self.environment.current_slot
            elif event.status in {
                "published",
                "retained_old_repository",
                "publication_conflict",
            }:
                start = self._job_start_slots.pop(
                    event.job_id,
                    max(0, self.environment.current_slot - 1),
                )
                self.planning_records.append(
                    CompletedPlanningRecord(
                        start_slot=start,
                        end_slot=self.environment.current_slot,
                        available_after_slot=self.environment.current_slot,
                        runtime_ms=event.planning_runtime_ms,
                        proposed_candidates=event.proposed_count,
                        valid_candidates=event.valid_count,
                        published=event.status == "published",
                    )
                )

    def _quality_utility(self, execution: _ActiveExecution) -> float:
        request = execution.request
        weighted = (
            request.preference_stability * execution.operational_stability
            + request.preference_delay * execution.delay_satisfaction
            + request.preference_energy * execution.energy_satisfaction
        )
        denominator = (
            request.preference_stability
            + request.preference_delay
            + request.preference_energy
        )
        return weighted / max(denominator, 1e-12)

    def _resolve_executions_after_advance(self) -> None:
        for request_id, execution in list(self._active.items()):
            if self.environment.current_slot >= execution.completion_slot:
                actual_latency_ms = (
                    self.environment.current_slot - execution.request.arrival_slot
                ) * self.environment.slot_length
                deadline_missed = actual_latency_ms > execution.request.deadline_ms
                utility = 0.0 if deadline_missed else self._quality_utility(execution)
                if deadline_missed:
                    self._deadline_missed += 1
                self._goodput_sum += utility
                self._completed_candidates.append(execution)
                self.dispatch_records[execution.record_index] = dataclasses.replace(
                    self.dispatch_records[execution.record_index],
                    runtime_failed=False,
                    deadline_violated=deadline_missed,
                    request_utility=utility,
                    outcome_available_after_slot=self.environment.current_slot,
                )
                del self._active[request_id]
                continue
            if any(
                not self.environment.up[node_index]
                for node_index in set(execution.assignment)
            ):
                self.environment.abort_running_dnn(execution.dnn_index)
                self._runtime_failed += 1
                self.dispatch_records[execution.record_index] = dataclasses.replace(
                    self.dispatch_records[execution.record_index],
                    runtime_failed=True,
                    deadline_violated=False,
                    request_utility=0.0,
                    outcome_available_after_slot=self.environment.current_slot,
                )
                del self._active[request_id]

    def _dispatch_request(self, request: InferenceRequest) -> None:
        profile = self.catalog.get(request.profile_id)
        dnn_index = len(self.environment.ds)
        self.environment.ds.append(materialize_dnn(profile, request))
        if self.config.algorithm == "per_request_customized":
            dispatcher = self.request_dispatcher
        elif self.baseline_dispatcher is not None:
            dispatcher = self.baseline_dispatcher
        else:
            dispatcher = self.dispatcher
        decision = dispatcher.dispatch(
            dnn_index,
            request,
            self.repository_store.snapshot(),
        )
        origin_group = self.environment.origin_group_for_node(request.initiate_node)
        record_index = len(self.dispatch_records)
        record = CompletedDispatchRecord(
            slot=self.environment.current_slot,
            available_after_slot=self.environment.current_slot,
            profile_id=request.profile_id,
            origin_group=origin_group,
            path=decision.path,
            online_runtime_ms=decision.online_runtime_ms,
            full_revalidation_count=decision.full_revalidation_count,
            feasible_candidate_count=decision.feasible_candidate_count,
            rejected=not decision.accepted,
            request_id=request.request_id,
            outcome_available_after_slot=(
                self.environment.current_slot if not decision.accepted else None
            ),
            request_utility=0.0 if not decision.accepted else None,
        )
        self.dispatch_records.append(record)
        if self.config.algorithm.startswith("periodic_"):
            self.planner.record_dispatch(decision.path, request.profile_id)
        if not decision.accepted:
            return
        assert decision.candidate is not None and decision.assignment is not None
        execution = _ActiveExecution(
            request=request,
            dnn_index=dnn_index,
            assignment=decision.assignment,
            completion_slot=(
                self.environment.current_slot
                + decision.candidate.service_exposure_slots
            ),
            record_index=record_index,
            candidate_delay_ms=decision.candidate.estimated_delay_ms,
            operational_stability=decision.candidate.operational_stability,
            total_energy=decision.candidate.total_energy,
            delay_satisfaction=decision.candidate.delay_satisfaction,
            energy_satisfaction=decision.candidate.energy_satisfaction,
        )
        self._active[request.request_id] = execution
        self._accepted_candidates.append(execution)

    def _initialize_control_and_bootstrap(self) -> None:
        if not self.config.algorithm.startswith("periodic_"):
            return
        observation = self.environment.capture_observation_snapshot("planning")
        event = self.control_loop.step(observation, (), ())
        if event is None or not event.applied:
            raise RuntimeError("initial joint budget decision was not applied")
        if self.config.bootstrap_mode == "prewarm":
            publication = self.planner.prewarm()
            self._record_planning_events((publication,))

    def run(self) -> PeriodicExperimentMetrics:
        self._initialize_control_and_bootstrap()
        for slot in range(len(self.arrival_trace.batches)):
            if self.environment.current_slot != slot:
                raise RuntimeError("environment and arrival trace slots diverged")
            if self.config.algorithm.startswith("periodic_"):
                planning_events = self.planner.tick()
                self._record_planning_events(planning_events)
                control_observation = self.environment.capture_observation_snapshot("planning")
                self.control_loop.step(
                    control_observation,
                    tuple(self.dispatch_records),
                    tuple(self.planning_records),
                )
            for request in self.arrival_trace.environment_requests_at(slot):
                self._dispatch_request(request)
            self.environment.advance_time_slot()
            self._resolve_executions_after_advance()

        drain_slots = 0
        while self._active and drain_slots < self.config.drain_max_slots:
            self.environment.advance_time_slot()
            self._resolve_executions_after_advance()
            drain_slots += 1

        arrived = len(self.dispatch_records)
        rejected = sum(record.rejected for record in self.dispatch_records)
        accepted = arrived - rejected
        direct = sum(record.path == "plan_direct" for record in self.dispatch_records)
        repaired = sum(record.path == "plan_repaired" for record in self.dispatch_records)
        fallback = sum(record.path == "fast_fallback" for record in self.dispatch_records)
        per_request = sum(
            record.path == "per_request_customized" for record in self.dispatch_records
        )
        online_baseline = sum(
            record.path.startswith("baseline_") for record in self.dispatch_records
        )
        online_times = [record.online_runtime_ms for record in self.dispatch_records]
        completed = len(self._completed_candidates)
        unfinished = len(self._active)
        accepted_values = self._accepted_candidates

        def mean(attribute: str) -> float:
            return (
                statistics.fmean(getattr(item, attribute) for item in accepted_values)
                if accepted_values
                else 0.0
            )

        p95 = 0.0
        if online_times:
            ordered = sorted(online_times)
            p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        control_metrics = self.control_loop.metrics
        return PeriodicExperimentMetrics(
            arrival_slots=len(self.arrival_trace.batches),
            elapsed_slots=self.environment.current_slot,
            arrived_requests=arrived,
            accepted_requests=accepted,
            rejected_requests=rejected,
            completed_requests=completed,
            runtime_failed_requests=self._runtime_failed,
            deadline_missed_requests=self._deadline_missed,
            unfinished_requests=unfinished,
            goodput_utility=(
                self._goodput_sum
                / max(self.environment.current_slot * self.environment.slot_length / 1000.0, 1e-12)
            ),
            avg_estimated_delay_ms=mean("candidate_delay_ms"),
            avg_operational_stability_score=mean("operational_stability"),
            avg_total_energy=mean("total_energy"),
            plan_direct_count=direct,
            plan_repaired_count=repaired,
            fast_fallback_count=fallback,
            per_request_customized_count=per_request,
            online_baseline_count=online_baseline,
            planning_jobs_started=self.planner.statistics.started_jobs,
            planning_jobs_published=self.planner.statistics.published_jobs,
            planning_jobs_retained=self.planner.statistics.retained_old_repository_jobs,
            total_planning_runtime_ms=self.planner.statistics.total_planning_runtime_ms,
            total_online_runtime_ms=sum(online_times),
            avg_online_runtime_ms=(statistics.fmean(online_times) if online_times else 0.0),
            p95_online_runtime_ms=p95,
            control_decision_count=len(control_metrics.decisions),
            control_outcome_count=len(control_metrics.outcomes),
            control_fallback_count=control_metrics.fallback_count,
            search_assignments_generated=(
                self.planner.statistics.search_assignments_generated
                + self.request_dispatcher.statistics.assignments_generated
                + (self.baseline_dispatcher.evaluations if self.baseline_dispatcher else 0)
            ),
            search_constraint_rejected_candidates=(
                self.planner.statistics.search_constraint_rejected_candidates
                + self.request_dispatcher.statistics.constraint_rejected_candidates
            ),
            search_nonconverged_candidates=(
                self.planner.statistics.search_nonconverged_candidates
                + self.request_dispatcher.statistics.nonconverged_candidates
            ),
            search_feasible_candidates=(
                self.planner.statistics.search_feasible_candidates
                + self.request_dispatcher.statistics.feasible_candidates
            ),
            search_pareto_candidates=(
                self.planner.statistics.search_pareto_candidates
            ),
            search_selected_candidates=(
                self.planner.statistics.search_selected_candidates
                + self.request_dispatcher.statistics.selected_candidates
                + (self.baseline_dispatcher.selected if self.baseline_dispatcher else 0)
            ),
            publication_valid_candidates=(
                self.planner.statistics.publication_valid_candidates
            ),
            publication_rejected_candidates=(
                self.planner.statistics.publication_rejected_candidates
            ),
            trace_seed=self.arrival_trace.seed,
            bootstrap_mode=self.config.bootstrap_mode,
            semantic_versions=self.semantic_versions,
        )
