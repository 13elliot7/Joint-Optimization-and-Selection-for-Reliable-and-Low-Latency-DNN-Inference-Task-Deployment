from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

from models import CandidateStateSnapshot

if TYPE_CHECKING:
    from core.environment import Environment


@dataclass(frozen=True)
class DeploymentPlan:
    """跨时隙复用的部署结构；规划指标只作为参考值。"""

    plan_id: str
    profile_id: str
    origin_group: int
    assignment: tuple[int, ...]
    required_cpu: tuple[tuple[int, float], ...]
    used_physical_link_ids: tuple[int, ...]
    planning_reference_delay_ms: float
    planning_reference_oss: float
    planning_reference_availability: float
    planning_reference_energy: float
    planning_reference_delay_satisfaction: float
    planning_reference_energy_satisfaction: float
    planning_snapshot_version: int
    topology_version: int
    created_slot: int
    expires_slot: int
    semantic_versions: tuple[tuple[str, str], ...]
    generator_mode: str = "legacy"
    generator_version: str = "legacy_v1"
    search_seed: int | None = None
    source_generation: int | None = None

    def __post_init__(self) -> None:
        if not self.plan_id or not self.profile_id:
            raise ValueError("plan_id and profile_id must not be empty")
        if not self.generator_mode or not self.generator_version:
            raise ValueError("plan generator metadata must not be empty")
        if not self.assignment:
            raise ValueError("assignment must not be empty")
        if self.origin_group < 0 or self.topology_version < 0:
            raise ValueError("origin_group and topology_version must be non-negative")
        if self.created_slot < 0 or self.expires_slot < self.created_slot:
            raise ValueError("invalid plan lifetime")
        if self.planning_reference_delay_ms < 0.0 or self.planning_reference_energy < 0.0:
            raise ValueError("planning delay and energy must be non-negative")
        if any(node_index < 0 or cpu < 0.0 for node_index, cpu in self.required_cpu):
            raise ValueError("required_cpu entries must be non-negative")

    @property
    def used_nodes(self) -> frozenset[int]:
        return frozenset(self.assignment)

    def is_compatible(
        self,
        current_slot: int,
        topology_version: int,
        semantic_versions: tuple[tuple[str, str], ...],
        node_count: int,
    ) -> bool:
        return (
            self.created_slot <= current_slot <= self.expires_slot
            and self.topology_version == topology_version
            and self.semantic_versions == semantic_versions
            and all(0 <= node_index < node_count for node_index in self.assignment)
        )


@dataclass(frozen=True)
class RepositoryLookup:
    candidates: tuple[DeploymentPlan, ...]
    rejection_reason: str | None = None


@dataclass(frozen=True)
class PlanRepository:
    """一个可原子替换的不可变方案仓库版本。"""

    repository_version: int
    topology_version: int
    semantic_versions: tuple[tuple[str, str], ...]
    published_slot: int
    plans: tuple[DeploymentPlan, ...]

    def __post_init__(self) -> None:
        if self.repository_version < 0 or self.topology_version < 0 or self.published_slot < 0:
            raise ValueError("repository versions and slot must be non-negative")
        plan_ids = [plan.plan_id for plan in self.plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("plan_id values must be unique within a repository")

    def lookup(
        self,
        profile_id: str,
        origin_group: int,
        current_slot: int,
        topology_version: int,
        semantic_versions: tuple[tuple[str, str], ...],
        node_count: int,
    ) -> RepositoryLookup:
        keyed = tuple(
            plan
            for plan in self.plans
            if plan.profile_id == profile_id and plan.origin_group == origin_group
        )
        if not keyed:
            return RepositoryLookup((), "no_profile_plan")
        if topology_version != self.topology_version:
            return RepositoryLookup((), "topology_mismatch")
        if semantic_versions != self.semantic_versions:
            return RepositoryLookup((), "semantic_mismatch")
        valid = tuple(
            plan
            for plan in keyed
            if plan.is_compatible(
                current_slot,
                topology_version,
                semantic_versions,
                node_count,
            )
        )
        if valid:
            return RepositoryLookup(valid)
        if all(current_slot > plan.expires_slot for plan in keyed):
            return RepositoryLookup((), "plan_expired")
        return RepositoryLookup((), "plan_incompatible")


class AtomicPlanRepository:
    """用短临界区维护当前不可变仓库引用。"""

    def __init__(self, initial: PlanRepository) -> None:
        self._current = initial
        self._lock = threading.Lock()

    def snapshot(self) -> PlanRepository:
        with self._lock:
            return self._current

    def publish(
        self,
        repository: PlanRepository,
        expected_current_version: int,
    ) -> bool:
        with self._lock:
            if self._current.repository_version != expected_current_version:
                return False
            if repository.repository_version <= self._current.repository_version:
                raise ValueError("new repository version must increase")
            self._current = repository
            return True


def _reference_score(plan: DeploymentPlan) -> float:
    return (
        plan.planning_reference_oss
        + plan.planning_reference_delay_satisfaction
        + plan.planning_reference_energy_satisfaction
    )


def _node_set_jaccard(first: DeploymentPlan, second: DeploymentPlan) -> float:
    union = first.used_nodes | second.used_nodes
    if not union:
        return 1.0
    return len(first.used_nodes & second.used_nodes) / len(union)


def select_diverse_plans(
    plans: Iterable[DeploymentPlan],
    max_size: int,
    max_node_overlap: float = 0.85,
) -> tuple[DeploymentPlan, ...]:
    """去重并贪心保留质量较高且节点集合不同的候选。"""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if not 0.0 <= max_node_overlap <= 1.0:
        raise ValueError("max_node_overlap must be in [0, 1]")
    deduplicated: dict[tuple[int, ...], DeploymentPlan] = {}
    for plan in plans:
        current = deduplicated.get(plan.assignment)
        if current is None or _reference_score(plan) > _reference_score(current):
            deduplicated[plan.assignment] = plan
    ranked = sorted(
        deduplicated.values(),
        key=lambda plan: (-_reference_score(plan), plan.plan_id),
    )
    selected: list[DeploymentPlan] = []
    for plan in ranked:
        if all(
            _node_set_jaccard(plan, existing) <= max_node_overlap
            for existing in selected
        ):
            selected.append(plan)
        if len(selected) >= max_size:
            break
    return tuple(selected)


def make_deployment_plan(
    environment: Environment,
    dnn_index: int,
    candidate: CandidateStateSnapshot,
    profile_id: str,
    origin_group: int,
    created_slot: int,
    ttl_slots: int,
    semantic_versions: tuple[tuple[str, str], ...],
    plan_id: str | None = None,
    generator_mode: str = "legacy",
    generator_version: str = "legacy_v1",
    search_seed: int | None = None,
    source_generation: int | None = None,
) -> DeploymentPlan:
    """从规划候选提取结构和仅供参考的指标。"""
    if ttl_slots <= 0:
        raise ValueError("ttl_slots must be positive")
    dnn = environment.ds[dnn_index]
    cpu_by_node: dict[int, float] = {}
    for task_index, node_index in enumerate(candidate.assignment):
        cpu_by_node[node_index] = (
            cpu_by_node.get(node_index, 0.0) + dnn.tasks[task_index].cpu_need
        )
    physical_ids = tuple(
        sorted(
            link.physical_link_id
            for link in environment.collect_used_physical_links(
                dnn_index,
                list(candidate.assignment),
            )
        )
    )
    if plan_id is None:
        digest = hashlib.sha256(
            repr((profile_id, origin_group, candidate.assignment, created_slot)).encode()
        ).hexdigest()[:16]
        plan_id = f"plan-{digest}"
    return DeploymentPlan(
        plan_id=plan_id,
        profile_id=profile_id,
        origin_group=origin_group,
        assignment=candidate.assignment,
        required_cpu=tuple(sorted(cpu_by_node.items())),
        used_physical_link_ids=physical_ids,
        planning_reference_delay_ms=candidate.estimated_delay_ms,
        planning_reference_oss=candidate.operational_stability,
        planning_reference_availability=candidate.node_availability_score,
        planning_reference_energy=candidate.total_energy,
        planning_reference_delay_satisfaction=candidate.delay_satisfaction,
        planning_reference_energy_satisfaction=candidate.energy_satisfaction,
        planning_snapshot_version=candidate.observation_snapshot_version,
        topology_version=environment.topology_version,
        created_slot=created_slot,
        expires_slot=created_slot + ttl_slots - 1,
        semantic_versions=semantic_versions,
        generator_mode=generator_mode,
        generator_version=generator_version,
        search_seed=search_seed,
        source_generation=source_generation,
    )


def build_plan_repository(
    repository_version: int,
    topology_version: int,
    semantic_versions: tuple[tuple[str, str], ...],
    published_slot: int,
    plans: Iterable[DeploymentPlan],
    max_pool_size: int = 20,
    max_node_overlap: float = 0.85,
    pool_budget_by_key: tuple[tuple[str, int, int], ...] | None = None,
) -> PlanRepository:
    """按画像和接入组分别执行多样性筛选后构造仓库。"""
    grouped: dict[tuple[str, int], list[DeploymentPlan]] = {}
    for plan in plans:
        if plan.topology_version != topology_version:
            raise ValueError("plan topology_version does not match repository")
        if plan.semantic_versions != semantic_versions:
            raise ValueError("plan semantic_versions do not match repository")
        grouped.setdefault((plan.profile_id, plan.origin_group), []).append(plan)
    selected: list[DeploymentPlan] = []
    quotas = {
        (profile_id, origin_group): budget
        for profile_id, origin_group, budget in (pool_budget_by_key or ())
    }
    for key in sorted(grouped):
        selected.extend(
            select_diverse_plans(
                grouped[key],
                max_size=quotas.get(key, max_pool_size),
                max_node_overlap=max_node_overlap,
            )
        )
    return PlanRepository(
        repository_version=repository_version,
        topology_version=topology_version,
        semantic_versions=semantic_versions,
        published_slot=published_slot,
        plans=tuple(selected),
    )
