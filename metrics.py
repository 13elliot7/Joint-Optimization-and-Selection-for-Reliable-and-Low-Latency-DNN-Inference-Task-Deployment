from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from semantics import ACTIVE_OBJECTIVE_SEMANTICS_VERSION

if TYPE_CHECKING:
    from planning.control_models import JointBudgetDecision, JointBudgetOutcome


@dataclass
class ExperimentMetrics:
    """保存一次实验运行后的聚合指标。"""

    avg_delay: float
    avg_operational_stability_score: float
    avg_energy: float
    failure_count: int
    runtime_ms: int
    objective_semantics_version: str = ACTIVE_OBJECTIVE_SEMANTICS_VERSION

    @property
    def avg_estimated_delay(self) -> float:
        """返回动态模型口径下的平均估计完成时延。"""
        return self.avg_delay

    @property
    def rejected_or_failed_count(self) -> int:
        """返回接纳失败或部署失败的任务数量。"""
        return self.failure_count

    @property
    def avg_total_energy(self) -> float:
        """返回平均总能耗。"""
        return self.avg_energy

    def to_report_rows(self) -> List[Tuple[str, float | int | str]]:
        """按动态模型语义生成统一输出项。"""
        return [
            ("avg_estimated_delay", self.avg_estimated_delay),
            ("avg_operational_stability_score", self.avg_operational_stability_score),
            ("objective_semantics_version", self.objective_semantics_version),
            ("avg_total_energy", self.avg_total_energy),
            ("rejected_or_failed_count", self.rejected_or_failed_count),
            ("runtime_ms", self.runtime_ms),
        ]


@dataclass
class JointControlMetrics:
    """保存控制周期预测与因果可见实际结果的对应关系。"""

    decisions: List["JointBudgetDecision"]
    outcomes: List["JointBudgetOutcome"]
    joint_budget_control_version: str = "discrete_mpc_v1"

    def add_decision(self, decision: "JointBudgetDecision") -> None:
        self.decisions.append(decision)

    def add_outcome(self, outcome: "JointBudgetOutcome") -> None:
        if outcome.decision_id not in {decision.decision_id for decision in self.decisions}:
            raise ValueError("joint control outcome has no matching decision")
        self.outcomes.append(outcome)

    @property
    def mean_absolute_prediction_error(self) -> float:
        decisions = {decision.decision_id: decision for decision in self.decisions}
        errors = [
            abs(
                outcome.realized_objective
                - decisions[outcome.decision_id].predicted_objective
            )
            for outcome in self.outcomes
            if outcome.decision_id in decisions
        ]
        return sum(errors) / len(errors) if errors else 0.0

    @property
    def fallback_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.fallback_used)

    def to_report_rows(self) -> List[Tuple[str, float | int | str]]:
        return [
            ("joint_budget_control_version", self.joint_budget_control_version),
            ("joint_control_decision_count", len(self.decisions)),
            ("joint_control_outcome_count", len(self.outcomes)),
            ("joint_control_fallback_count", self.fallback_count),
            (
                "joint_control_mean_absolute_prediction_error",
                self.mean_absolute_prediction_error,
            ),
        ]


@dataclass(frozen=True)
class PeriodicExperimentMetrics:
    """周期规划事件循环的请求守恒、开销和语义冻结结果。"""

    arrival_slots: int
    elapsed_slots: int
    arrived_requests: int
    accepted_requests: int
    rejected_requests: int
    completed_requests: int
    runtime_failed_requests: int
    deadline_missed_requests: int
    unfinished_requests: int
    goodput_utility: float
    avg_estimated_delay_ms: float
    avg_operational_stability_score: float
    avg_total_energy: float
    plan_direct_count: int
    plan_repaired_count: int
    fast_fallback_count: int
    planning_jobs_started: int
    planning_jobs_published: int
    planning_jobs_retained: int
    total_planning_runtime_ms: float
    total_online_runtime_ms: float
    avg_online_runtime_ms: float
    p95_online_runtime_ms: float
    control_decision_count: int
    control_outcome_count: int
    control_fallback_count: int
    trace_seed: int
    bootstrap_mode: str
    semantic_versions: Tuple[Tuple[str, str], ...]
    search_assignments_generated: int = 0
    search_constraint_rejected_candidates: int = 0
    search_nonconverged_candidates: int = 0
    search_feasible_candidates: int = 0
    search_pareto_candidates: int = 0
    search_selected_candidates: int = 0
    publication_valid_candidates: int = 0
    publication_rejected_candidates: int = 0
    per_request_customized_count: int = 0
    online_baseline_count: int = 0

    @property
    def request_conservation_valid(self) -> bool:
        return (
            self.arrived_requests == self.accepted_requests + self.rejected_requests
            and self.accepted_requests
            == self.completed_requests
            + self.runtime_failed_requests
            + self.unfinished_requests
        )

    def __post_init__(self) -> None:
        if not self.request_conservation_valid:
            raise ValueError("periodic experiment request conservation failed")
        if self.bootstrap_mode not in {"cold_start", "prewarm"}:
            raise ValueError("unsupported periodic bootstrap mode")

    def to_report_rows(self) -> List[Tuple[str, float | int | str]]:
        rows: List[Tuple[str, float | int | str]] = [
            ("arrival_slots", self.arrival_slots),
            ("elapsed_slots", self.elapsed_slots),
            ("arrived_requests", self.arrived_requests),
            ("accepted_requests", self.accepted_requests),
            ("rejected_requests", self.rejected_requests),
            ("completed_requests", self.completed_requests),
            ("runtime_failed_requests", self.runtime_failed_requests),
            ("deadline_missed_requests", self.deadline_missed_requests),
            ("unfinished_requests", self.unfinished_requests),
            ("goodput_utility", self.goodput_utility),
            ("avg_estimated_delay_ms", self.avg_estimated_delay_ms),
            ("avg_operational_stability_score", self.avg_operational_stability_score),
            ("avg_total_energy", self.avg_total_energy),
            ("plan_direct_count", self.plan_direct_count),
            ("plan_repaired_count", self.plan_repaired_count),
            ("fast_fallback_count", self.fast_fallback_count),
            ("per_request_customized_count", self.per_request_customized_count),
            ("online_baseline_count", self.online_baseline_count),
            ("planning_jobs_started", self.planning_jobs_started),
            ("planning_jobs_published", self.planning_jobs_published),
            ("planning_jobs_retained", self.planning_jobs_retained),
            ("total_planning_runtime_ms", self.total_planning_runtime_ms),
            ("total_online_runtime_ms", self.total_online_runtime_ms),
            ("avg_online_runtime_ms", self.avg_online_runtime_ms),
            ("p95_online_runtime_ms", self.p95_online_runtime_ms),
            ("publication_valid_candidates", self.publication_valid_candidates),
            ("publication_rejected_candidates", self.publication_rejected_candidates),
        ]
        semantic_version_map = dict(self.semantic_versions)
        rows.append(
            (
                "objective_semantics_version",
                semantic_version_map["objective_semantics_version"],
            )
        )
        return rows
