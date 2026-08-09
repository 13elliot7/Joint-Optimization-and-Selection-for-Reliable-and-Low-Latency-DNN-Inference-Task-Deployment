from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set

from core.environment import Environment
from metrics import ExperimentMetrics
from models import CandidateStateSnapshot, DNN, LinkNode, Node, ObservationSnapshot
from semantics import PERIODIC_SEMANTIC_VERSIONS


@dataclass
class DNNContext:
    """保存定制化进化算子需要复用的 DNN 拓扑信息。"""

    topological_order: List[int]
    predecessors: Dict[int, List[int]]
    successors: Dict[int, List[int]]
    edge_data: Dict[tuple[int, int], float]
    critical_path_tasks: Set[int]
    candidate_nodes: Dict[int, List[int]]


@dataclass(frozen=True)
class RepairResult:
    """记录一次结构修复的结果，避免非法个体被静默写回种群。"""

    assignment: List[int]
    success: bool
    changed_gene_count: int
    reason: str = ""


@dataclass(frozen=True)
class PlanPoolSearchStatistics:
    """记录一次周期候选池搜索各阶段的候选数量。"""

    generated_assignments: int = 0
    constraint_rejected_candidates: int = 0
    nonconverged_candidates: int = 0
    feasible_candidates: int = 0
    pareto_candidates: int = 0
    selected_candidates: int = 0


@dataclass(frozen=True)
class CustomizedSearchConfig:
    """后台与逐请求入口共享的 customized 搜索预算。"""

    population_size: int = 12
    generations: int = 4
    archive_capacity: int = 64
    random_seed: int = 0
    max_evaluations: int | None = None
    wall_time_budget_ms: float | None = None

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.population_size % 2 != 0:
            raise ValueError("population_size must be an even integer >= 2")
        if self.generations <= 0 or self.archive_capacity <= 0:
            raise ValueError("generations and archive_capacity must be positive")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when provided")
        if self.wall_time_budget_ms is not None and self.wall_time_budget_ms <= 0.0:
            raise ValueError("wall_time_budget_ms must be positive when provided")


@dataclass(frozen=True)
class CustomizedCandidate:
    """customized 搜索档案中的一个严格可行候选。"""

    assignment: tuple[int, ...]
    objectives: tuple[float, float, float]
    constraint_violation: float
    generation: int
    source: str


@dataclass(frozen=True)
class CustomizedSearchResult:
    """无环境提交副作用的单 DNN customized 搜索结果。"""

    candidates: tuple[CustomizedCandidate, ...]
    evaluations: int
    generations_completed: int
    feasible_evaluations: int
    constraint_rejected_evaluations: int
    nonconverged_evaluations: int
    archive_size_before_selection: int
    termination_reason: str
    wall_time_ms: float
    seed: int


@dataclass(frozen=True)
class ObjectiveValues:
    """保存 OSS、时延和能耗三个最大化满意度及对应原值。"""

    operational_stability: float
    delay_satisfaction: float
    energy_satisfaction: float
    estimated_delay: float
    total_energy: float
    deadline_feasible: bool

    @property
    def vector(self) -> tuple[float, float, float]:
        return (
            self.operational_stability,
            self.delay_satisfaction,
            self.energy_satisfaction,
        )


def _java_div(numerator: float, denominator: float) -> float:
    """按 Java 风格处理零除场景的除法。"""
    if denominator == 0:
        if numerator == 0:
            return math.nan
        return math.inf
    return numerator / denominator


class AllDNNRefactor:
    def __init__(self, env: Environment | None = None) -> None:
        """初始化主算法与三个基线共享的运行状态。"""
        self.env = env or Environment()
        self.pop_size = 60
        self.iteration_limit = 200
        self.max_dnn_num = 20
        self.mutate_pm = 0.25
        self.cross_over_pm = 0.5
        self.customized_membrane_roles = ["stability", "latency", "energy", "feasibility"]
        self.cv_initial_epsilon = 0.35
        self.local_search_elite_ratio = 0.05
        self.local_search_top_k = 3
        self.preference = "adaptive"
        self.use_dag_aware_initialization = True
        self.use_dag_block_crossover = True
        self.use_skew_mutation = True
        self.use_elite_local_search = True
        self.use_predicted_quality_scores = True
        self.collect_pareto_points = False
        self.pareto_points: List[Dict[str, float | int | str]] = []
        self.population_roles: List[str] = []
        self._joint_population_roles: List[str] = []
        self.operator_stats: Dict[str, int] = {}
        self.sa_steps_multiplier = 2
        self.sa_initial_temperature = 1.0
        self.sa_final_temperature = 0.01
        self.sa_weight_profiles = [
            "adaptive",
            "delay_sensitive",
            "stability_sensitive",
            "energy_sensitive",
            "balanced",
        ]
        self.start_time = 0.0
        self.res: List[List[List[int]]] = []
        self.res_pq: List[List[List[int]]] = []
        self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
        self.estimated_delay_values = [math.inf for _ in range(2 * self.pop_size)]
        self.total_energy_values = [math.inf for _ in range(2 * self.pop_size)]
        self.deadline_feasible_values = [False for _ in range(2 * self.pop_size)]
        self.constraint_violations = [0.0 for _ in range(2 * self.pop_size)]
        self.pareto_level_sort: List[List[int]] = []
        self.distance = [[0 for _ in range(2 * self.pop_size)] for _ in range(1000)]
        self._path_transfer_cost_cache: Dict[tuple[int, int, float], float] = {}
        self._assignment_evaluation_cache: Dict[
            tuple[int, tuple[int, ...], int, str],
            ObjectiveValues,
        ] = {}
        self._candidate_state_cache: Dict[
            tuple[int, tuple[int, ...], int, str, float | None],
            CandidateStateSnapshot,
        ] = {}
        self._constraint_violation_cache: Dict[
            tuple[int, tuple[int, ...], int, str],
            float,
        ] = {}
        self._max_resource_delay_cache: Dict[tuple[int, ...], float] = {}
        self._max_resource_last_delay: float | None = None
        self._cache_state_version = self.env.environment_state_version
        self.max_resource_fast_top_k = 8
        self.max_resource_fast_beam_width = 30
        self.max_resource_fast_budget_ms = 3000
        self.last_plan_pool_search_statistics = PlanPoolSearchStatistics()
        self.last_customized_search_result: CustomizedSearchResult | None = None
        self._active_search_snapshot: ObservationSnapshot | None = None

    def _resolve_preference_weights(
        self,
        dnn_index: int,
        min_a: float,
        max_a: float,
        min_r: float,
        max_r: float,
        min_t: float,
        max_t: float,
    ) -> tuple[float, float, float, float]:
        """返回 OSS、时延、能耗权重；末位零仅兼容旧搜索缓存接口。"""
        explicit_weights = {
            "delay_sensitive": (0.15, 0.65, 0.20, 0.0),
            "stability_sensitive": (0.60, 0.20, 0.20, 0.0),
            "energy_sensitive": (0.15, 0.15, 0.70, 0.0),
            "balanced": (1 / 3, 1 / 3, 1 / 3, 0.0),
        }
        if self.preference in explicit_weights:
            return explicit_weights[self.preference]

        dnn = self.env.ds[dnn_index]
        weights = (
            max(0.0, dnn.preference_stability),
            max(0.0, dnn.preference_delay),
            max(0.0, dnn.preference_energy),
        )
        total = sum(weights)
        if total <= 0.0:
            return 1 / 3, 1 / 3, 1 / 3, 0.0
        return weights[0] / total, weights[1] / total, weights[2] / total, 0.0

    @staticmethod
    def _clamp_satisfaction(value: float) -> float:
        """把统一目标限制到稳定的满意度区间。"""
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _dominates_objectives(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> bool:
        """对三个“越大越好”的统一满意度目标执行支配判断。"""
        not_worse = all(left_value >= right_value for left_value, right_value in zip(left, right))
        strictly_better = any(left_value > right_value for left_value, right_value in zip(left, right))
        return not_worse and strictly_better

    @classmethod
    def _point_objective_vector(
        cls,
        point: Dict[str, float | int | str],
    ) -> tuple[float, float, float]:
        """从导出点读取与搜索阶段完全一致的统一目标向量。"""
        return (
            cls._clamp_satisfaction(
                float(point.get("operational_stability_satisfaction", 0.0))
            ),
            cls._clamp_satisfaction(float(point["delay_satisfaction"])),
            cls._clamp_satisfaction(float(point["energy_satisfaction"])),
        )

    @classmethod
    def _dominates_point(
        cls,
        left: Dict[str, float | int | str],
        right: Dict[str, float | int | str],
    ) -> bool:
        """按统一三目标满意度向量判断导出点支配关系。"""
        return cls._dominates_objectives(
            cls._point_objective_vector(left),
            cls._point_objective_vector(right),
        )

    def _pareto_point_from_values(
        self,
        algorithm: str,
        dnn_index: int,
        individual_index: int,
        values: ObjectiveValues,
    ) -> Dict[str, float | int | str] | None:
        """把已评价目标值转换成 Pareto 导出点。"""
        if not values.deadline_feasible:
            return None
        deadline = float(self.env.ds[dnn_index].delay)
        return {
            "algorithm": algorithm,
            "dnn_index": dnn_index,
            "individual_index": individual_index,
            "operational_stability": values.operational_stability,
            "delay_utility": values.delay_satisfaction,
            "total_energy": values.total_energy,
            "operational_stability_satisfaction": values.operational_stability,
            "objective_semantics_version": self.env.objective_semantics_version,
            "energy_satisfaction": values.energy_satisfaction,
            "estimated_delay": values.estimated_delay,
            "deadline": deadline,
            "delay_satisfaction": values.delay_satisfaction,
        }

    def _append_current_pareto_points(self, algorithm: str, dnn_index: int, population_size: int) -> None:
        """从当前已评价种群中导出可行非支配点。"""
        if not self.collect_pareto_points:
            return
        points: List[Dict[str, float | int | str]] = []
        for idx in range(population_size):
            if not self.deadline_feasible_values[idx] or self.constraint_violations[idx] > 1e-12:
                continue
            assignment = self.res_pq[idx][dnn_index]
            point = self._pareto_point_from_values(
                algorithm,
                dnn_index,
                idx,
                self._evaluate_assignment(dnn_index, assignment),
            )
            if point is not None:
                points.append(point)
        for point in points:
            if not any(
                other is not point and self._dominates_point(other, point)
                for other in points
            ):
                self.pareto_points.append(point)

    def _assignment_pareto_point(
        self,
        algorithm: str,
        dnn_index: int,
        assignment: List[int],
        individual_index: int,
    ) -> Dict[str, float | int | str] | None:
        """把一个可行部署向量转换成四目标 Pareto 候选点。"""
        if not self.collect_pareto_points:
            return None
        values = self._evaluate_assignment(dnn_index, assignment)
        return self._pareto_point_from_values(
            algorithm,
            dnn_index,
            individual_index,
            values,
        )

    def _filter_pareto_points(
        self,
        points: List[Dict[str, float | int | str]],
    ) -> List[Dict[str, float | int | str]]:
        """按当前四目标口径过滤非支配点。"""
        if not self.collect_pareto_points:
            return []
        filtered: List[Dict[str, float | int | str]] = []
        seen: set[tuple[float, float, float, float]] = set()
        for point in points:
            key = self._point_objective_vector(point)
            if key in seen:
                continue
            seen.add(key)
            if not any(other is not point and self._dominates_point(other, point) for other in points):
                filtered.append(point)
        return filtered

    def _new_population(self, population_size: int) -> List[List[List[int]]]:
        """创建指定规模的三维种群容器。"""
        return [
            [[0 for _ in range(self.max_dnn_num)] for _ in range(len(self.env.ds))]
            for _ in range(population_size)
        ]

    def _build_assignment_matrix(self, dnn_index: int, assignment: List[int]) -> List[List[int]]:
        """把一维部署向量展开成任务-节点二维矩阵。"""
        task_count = len(self.env.ds[dnn_index].tasks)
        x = [[0 for _ in range(len(self.env.nodes))] for _ in range(task_count)]
        for task_idx in range(task_count):
            node_idx = assignment[task_idx]
            if node_idx < 0:
                raise IndexError("assignment index -1")
            x[task_idx][node_idx] = 1
        return x

    def _assignment_slice(self, dnn_index: int, assignment: List[int]) -> List[int]:
        """截取当前 DNN 实际使用长度的部署向量。"""
        return list(assignment[: len(self.env.ds[dnn_index].tasks)])

    def _evaluate_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
        initial_delay_hint: float | None = None,
        snapshot: ObservationSnapshot | None = None,
    ) -> ObjectiveValues:
        """只消费统一候选快照并返回三目标最大化满意度表示。"""
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        candidate = self._candidate_state_for_assignment(
            dnn_index,
            assignment_slice,
            initial_delay_hint,
            snapshot,
        )
        return ObjectiveValues(
            operational_stability=self._clamp_satisfaction(candidate.operational_stability),
            delay_satisfaction=candidate.delay_satisfaction,
            energy_satisfaction=candidate.energy_satisfaction,
            estimated_delay=candidate.estimated_delay_ms,
            total_energy=candidate.total_energy,
            deadline_feasible=candidate.deadline_feasible,
        )

    def _candidate_state_for_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
        initial_delay_hint: float | None = None,
        snapshot: ObservationSnapshot | None = None,
    ) -> CandidateStateSnapshot:
        """按环境状态和语义版本缓存统一候选预测。"""
        observation = snapshot or self._active_search_snapshot
        if observation is None and self._cache_state_version != self.env.environment_state_version:
            self._candidate_state_cache.clear()
            self._assignment_evaluation_cache.clear()
            self._constraint_violation_cache.clear()
            self._cache_state_version = self.env.environment_state_version
        if observation is None:
            observation = self.env.capture_observation_snapshot("online_admission")
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        normalized_hint = (
            round(float(initial_delay_hint), 12)
            if initial_delay_hint is not None
            else None
        )
        cache_key = (
            dnn_index,
            tuple(assignment_slice),
            observation.environment_state_version,
            self.env.objective_semantics_version,
            normalized_hint,
            observation.snapshot_version,
        )
        cached = self._candidate_state_cache.get(cache_key)
        if cached is not None:
            return cached
        candidate = self.env.predict_candidate_state(
            dnn_index,
            assignment_slice,
            observation,
            initial_delay_hint,
        )
        self._candidate_state_cache[cache_key] = candidate
        return candidate

    def _evaluate_customized_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> ObjectiveValues:
        """在当前 DNN 搜索期间缓存定制化算法的候选评价。"""
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        snapshot = self._active_search_snapshot
        cache_key = (
            dnn_index,
            tuple(assignment_slice),
            (
                snapshot.environment_state_version
                if snapshot is not None
                else self.env.environment_state_version
            ),
            self.env.objective_semantics_version,
            snapshot.snapshot_version if snapshot is not None else -1,
        )
        cached = self._assignment_evaluation_cache.get(cache_key)
        if cached is not None:
            return cached
        values = self._evaluate_assignment(
            dnn_index,
            assignment_slice,
            snapshot=snapshot,
        )
        self._assignment_evaluation_cache[cache_key] = values
        return values

    def _count_customized_values(self, dnn_index: int, individual_idx: int) -> float:
        """评价定制化联合种群个体并返回其估计时延。"""
        values = self._evaluate_customized_assignment(
            dnn_index,
            self.res_pq[individual_idx][dnn_index],
        )
        self._write_objective_values(individual_idx, values)
        return values.estimated_delay

    def _write_objective_values(self, individual_idx: int, values: ObjectiveValues) -> None:
        """把统一目标和对应原始指标写入并行种群缓存。"""
        (
            self.function1_values[individual_idx],
            self.function2_values[individual_idx],
            self.function3_values[individual_idx],
        ) = values.vector
        self.function4_values[individual_idx] = 0.0
        self.estimated_delay_values[individual_idx] = values.estimated_delay
        self.total_energy_values[individual_idx] = values.total_energy
        self.deadline_feasible_values[individual_idx] = values.deadline_feasible

    def _register_running_dnn(self, dnn_index: int, assignment: List[int], delay: float) -> None:
        """把已接纳的 DNN 注册成时隙级运行实体。"""
        # DNN 被接纳后不再是“部署即结束”，而是转成一个运行块，
        # 在若干个时隙内持续占用资源。
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        remaining_slots = self.env.service_exposure_slots(delay)
        self.env.add_running_dnn(
            dnn_index=dnn_index,
            assignment=assignment_slice,
            estimated_runtime=delay,
            remaining_slots=remaining_slots,
        )

    def _advance_until_drained(self) -> None:
        """在没有新任务时推进系统直到运行队列清空。"""
        while self.env.running_dnns:
            self.env.advance_time_slot()

    def _log_proposed_stage(self, dnn_index: int, phase: str, **kwargs: object) -> None:
        """打印 proposed 算法的阶段性日志。"""
        if not self.env.verbose:
            return
        details = " ".join(f"{key}={value}" for key, value in kwargs.items())
        if details:
            print(f"\t[PROPOSED] dnn={dnn_index} phase={phase} {details}")
        else:
            print(f"\t[PROPOSED] dnn={dnn_index} phase={phase}")

    def _is_assignment_valid(
        self,
        dnn_index: int,
        assignment: List[int],
        snapshot: ObservationSnapshot | None = None,
    ) -> bool:
        """检查一维部署向量是否满足资源和节点层级约束。"""
        observation = snapshot or self._active_search_snapshot
        dnn = self.env.ds[dnn_index]
        task_count = len(dnn.tasks)
        if len(assignment) < task_count:
            return False
        if any(node_idx < 0 or node_idx >= len(self.env.nodes) for node_idx in assignment[:task_count]):
            return False
        if observation is None:
            xs = self._build_assignment_matrix(dnn_index, assignment[:task_count])
            if not self.env.check_resource(dnn, xs):
                return False
        else:
            required_cpu = [0.0 for _ in self.env.nodes]
            for task_idx, node_idx in enumerate(assignment[:task_count]):
                if not observation.node_online[node_idx]:
                    return False
                required_cpu[node_idx] += dnn.tasks[task_idx].cpu_need
            if any(
                required > observation.node_available_cpu[node_idx] + 1e-12
                for node_idx, required in enumerate(required_cpu)
            ):
                return False
        for task_idx in range(task_count):
            node_idx = assignment[task_idx]
            if self.env.nodes[node_idx].level == 1 and node_idx != dnn.initiateNode:
                return False
        return True

    def _is_hierarchy_valid(
        self,
        dnn_index: int,
        assignment: List[int],
        context: DNNContext,
    ) -> bool:
        """检查 DAG 内部部署是否遵守端到边到云的非回退层级约束。"""
        if not self._is_assignment_valid(dnn_index, assignment):
            return False
        for task_idx in context.topological_order:
            node_level = self.env.nodes[assignment[task_idx]].level
            if any(
                node_level < self.env.nodes[assignment[pred]].level
                for pred in context.predecessors[task_idx]
            ):
                return False
        return True

    def _build_dnn_context(
        self,
        dnn_index: int,
        snapshot: ObservationSnapshot | None = None,
    ) -> DNNContext:
        """构建定制化 RDODA 使用的 DAG 拓扑上下文。"""
        observation = snapshot or self._active_search_snapshot
        dnn = self.env.ds[dnn_index]
        task_count = len(dnn.tasks)
        predecessors: Dict[int, List[int]] = {idx: [] for idx in range(task_count)}
        successors: Dict[int, List[int]] = {idx: [] for idx in range(task_count)}
        edge_data: Dict[tuple[int, int], float] = {}
        indegree = [0 for _ in range(task_count)]
        for link in dnn.links:
            start_idx = dnn.tasks.index(link.s_task)
            end_idx = dnn.tasks.index(link.e_task)
            successors[start_idx].append(end_idx)
            predecessors[end_idx].append(start_idx)
            edge_data[(start_idx, end_idx)] = float(link.float_tran)
            indegree[end_idx] += 1

        ready = [idx for idx, degree in enumerate(indegree) if degree == 0]
        topological_order: List[int] = []
        while ready:
            current = ready.pop(0)
            topological_order.append(current)
            for succ in successors[current]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
        if len(topological_order) != task_count:
            topological_order = list(range(task_count))

        longest = [0.0 for _ in range(task_count)]
        parent = [-1 for _ in range(task_count)]
        for task_idx in topological_order:
            task_weight = dnn.tasks[task_idx].float_num
            best_prefix = 0.0
            best_parent = -1
            for pred in predecessors[task_idx]:
                if longest[pred] > best_prefix:
                    best_prefix = longest[pred]
                    best_parent = pred
            longest[task_idx] = best_prefix + task_weight
            parent[task_idx] = best_parent
        critical_path_tasks: Set[int] = set()
        if longest:
            cursor = max(range(task_count), key=lambda idx: longest[idx])
            while cursor != -1:
                critical_path_tasks.add(cursor)
                cursor = parent[cursor]

        candidate_nodes: Dict[int, List[int]] = {}
        for task_idx, task in enumerate(dnn.tasks):
            candidates = [
                node_idx
                for node_idx, node in enumerate(self.env.nodes)
                if (
                    (
                        observation is None
                        and node.cpu >= task.cpu_need
                    )
                    or (
                        observation is not None
                        and observation.node_online[node_idx]
                        and observation.node_available_cpu[node_idx] >= task.cpu_need
                    )
                )
                and (node.level != 1 or node_idx == dnn.initiateNode)
            ]
            if not candidates:
                candidates = [
                    node_idx
                    for node_idx, node in enumerate(self.env.nodes)
                    if (observation is None or observation.node_online[node_idx])
                    and (node.level != 1 or node_idx == dnn.initiateNode)
                ]
            candidate_nodes[task_idx] = candidates

        return DNNContext(
            topological_order=topological_order,
            predecessors=predecessors,
            successors=successors,
            edge_data=edge_data,
            critical_path_tasks=critical_path_tasks,
            candidate_nodes=candidate_nodes,
        )

    def _path_transfer_cost(self, start_node_idx: int, end_node_idx: int, data_amount: float) -> float:
        """估计两个节点之间的数据传输时延代价。"""
        if start_node_idx == end_node_idx:
            return 0.0
        cache_key = (start_node_idx, end_node_idx, data_amount)
        if cache_key in self._path_transfer_cost_cache:
            return self._path_transfer_cost_cache[cache_key]
        snapshot = self._active_search_snapshot
        if snapshot is None:
            cost = self.env.path_transfer_delay(start_node_idx, end_node_idx, data_amount)
        else:
            bandwidth_by_link = {
                link: snapshot.effective_bandwidths[index]
                for index, link in enumerate(self.env.link_nodes)
            }
            cost = sum(
                data_amount / max(bandwidth_by_link.get(link, 1.0), 1.0)
                for link in self.env._path_links[(start_node_idx, end_node_idx)]
            )
        self._path_transfer_cost_cache[cache_key] = cost
        return cost

    def _candidate_score(
        self,
        dnn_index: int,
        task_idx: int,
        node_idx: int,
        assignment: List[int],
        context: DNNContext,
        role: str = "balanced",
    ) -> float:
        """按边缘资源、链路和能耗信息给候选节点打分。"""
        dnn = self.env.ds[dnn_index]
        task = dnn.tasks[task_idx]
        node = self.env.nodes[node_idx]
        snapshot = self._active_search_snapshot
        available_cpu = (
            snapshot.node_available_cpu[node_idx]
            if snapshot is not None
            else node.cpu
        )
        node_load = snapshot.node_loads[node_idx] if snapshot is not None else node.load_ratio
        node_heat = snapshot.node_heats[node_idx] if snapshot is not None else node.heat
        cpu_score = available_cpu / node.max_cpu if node.max_cpu else 0.0
        exec_time = task.float_num / node.float_rate if node.float_rate else float("inf")
        exec_time_score = 1.0 / (1.0 + exec_time)
        if snapshot is None:
            quality_score = node.operational_stability
        else:
            history = snapshot.node_availability_history[node_idx]
            availability = sum(history) / len(history) if history else 1.0
            quality_score = availability
        heat_score = 1.0 / (1.0 + node_heat + node_load)
        energy = node.comp_power * exec_time if math.isfinite(exec_time) else float("inf")
        energy_score = 1.0 / (1.0 + energy)

        link_cost = 0.0
        path_transfers: List[tuple[int, int, float]] = []
        predicted_runtime = max(float(dnn.delay), self.env.slot_length, 1.0)
        for pred in context.predecessors[task_idx]:
            pred_node_idx = assignment[pred]
            if pred_node_idx >= 0:
                data_amount = context.edge_data.get((pred, task_idx), 0.0)
                link_cost += self._path_transfer_cost(
                    pred_node_idx,
                    node_idx,
                    data_amount,
                )
                path_transfers.append((pred_node_idx, node_idx, data_amount))
        for succ in context.successors[task_idx]:
            succ_node_idx = assignment[succ]
            if succ_node_idx >= 0:
                data_amount = context.edge_data.get((task_idx, succ), 0.0)
                link_cost += self._path_transfer_cost(
                    node_idx,
                    succ_node_idx,
                    data_amount,
                )
                path_transfers.append((node_idx, succ_node_idx, data_amount))
        if snapshot is None:
            path_stability_score, path_feasibility_score = self.env.predict_paths_state(
                path_transfers,
                predicted_runtime,
            )
        else:
            # 路径状态只用于搜索引导；最终目标和硬约束仍由统一快照预测内核计算。
            path_stability_score = 1.0
            path_feasibility_score = 1.0
        link_score = 1.0 / (1.0 + link_cost) if math.isfinite(link_cost) else 0.0
        predicted_node_load = node_load + task.cpu_need / max(float(node.max_cpu), 1.0)
        node_feasibility_score = 1.0 / (1.0 + max(0.0, predicted_node_load - 1.0))
        feasibility_score = path_feasibility_score * node_feasibility_score

        weights_by_role = {
            "balanced": (0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10),
            "stability": (0.10, 0.10, 0.10, 0.20, 0.25, 0.05, 0.05, 0.15),
            "latency": (0.10, 0.25, 0.25, 0.10, 0.05, 0.05, 0.05, 0.15),
            "energy": (0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.30, 0.15),
            "feasibility": (0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.25),
        }
        w_cpu, w_time, w_link, w_node_quality, w_path_stability, w_heat, w_e, w_feasibility = weights_by_role.get(
            role,
            weights_by_role["balanced"],
        )
        return (
            w_cpu * cpu_score
            + w_time * exec_time_score
            + w_link * link_score
            + w_node_quality * quality_score
            + w_path_stability * path_stability_score
            + w_heat * heat_score
            + w_e * energy_score
            + w_feasibility * feasibility_score
        )

    def _best_candidate_node(
        self,
        dnn_index: int,
        task_idx: int,
        assignment: List[int],
        context: DNNContext,
        min_level: int = 1,
    ) -> int:
        """在当前任务候选集合中选择综合评分最高的节点。"""
        candidates = [
            node_idx
            for node_idx in context.candidate_nodes[task_idx]
            if self.env.nodes[node_idx].level >= min_level
        ]
        if not candidates:
            candidates = list(context.candidate_nodes[task_idx])
        if not candidates:
            return 0
        return max(
            candidates,
            key=lambda node_idx: self._candidate_score(dnn_index, task_idx, node_idx, assignment, context),
        )

    def _repair_assignment_result(
        self,
        dnn_index: int,
        assignment: List[int],
        context: DNNContext,
    ) -> RepairResult:
        """修复结构约束并显式返回成功状态和修改规模。"""
        original_assignment = self._assignment_slice(dnn_index, assignment)
        repaired = self._assignment_slice(dnn_index, assignment)
        dnn = self.env.ds[dnn_index]
        for task_idx in context.topological_order:
            if repaired[task_idx] < 0 or repaired[task_idx] >= len(self.env.nodes):
                repaired[task_idx] = self._best_candidate_node(dnn_index, task_idx, repaired, context)
            required_level = 1
            for pred in context.predecessors[task_idx]:
                required_level = max(required_level, self.env.nodes[repaired[pred]].level)
            current_node = repaired[task_idx]
            current_valid = (
                self.env.nodes[current_node].level >= required_level
                and (self.env.nodes[current_node].level != 1 or current_node == dnn.initiateNode)
            )
            if not current_valid:
                repaired[task_idx] = self._best_candidate_node(
                    dnn_index,
                    task_idx,
                    repaired,
                    context,
                    min_level=required_level,
                )

        if self._is_hierarchy_valid(dnn_index, repaired, context):
            return RepairResult(
                repaired,
                True,
                sum(left != right for left, right in zip(original_assignment, repaired)),
            )

        # 资源仍冲突时，优先迁移非关键路径任务，降低关键路径扰动。
        task_order = sorted(
            range(len(dnn.tasks)),
            key=lambda idx: (idx in context.critical_path_tasks, -len(context.successors[idx])),
        )
        for task_idx in task_order:
            original_node = repaired[task_idx]
            required_level = 1
            for pred in context.predecessors[task_idx]:
                required_level = max(required_level, self.env.nodes[repaired[pred]].level)
            ranked_candidates = sorted(
                [
                    node_idx
                    for node_idx in context.candidate_nodes[task_idx]
                    if self.env.nodes[node_idx].level >= required_level
                ],
                key=lambda node_idx: self._candidate_score(dnn_index, task_idx, node_idx, repaired, context),
                reverse=True,
            )
            for node_idx in ranked_candidates[:5]:
                repaired[task_idx] = node_idx
                if self._is_hierarchy_valid(dnn_index, repaired, context):
                    return RepairResult(
                        list(repaired),
                        True,
                        sum(left != right for left, right in zip(original_assignment, repaired)),
                    )
            repaired[task_idx] = original_node
        return RepairResult(
            original_assignment,
            False,
            0,
            "no_structurally_feasible_repair",
        )

    def _repair_assignment(self, dnn_index: int, assignment: List[int], context: DNNContext) -> List[int]:
        """兼容旧调用方；修复失败时安全回退到输入部署。"""
        result = self._repair_assignment_result(dnn_index, assignment, context)
        return result.assignment

    def _weighted_choice(
        self,
        weighted_candidates: List[tuple[int, float]],
        temperature: float = 0.35,
    ) -> int:
        """使用数值稳定的 softmax 按候选评分采样节点。"""
        if not weighted_candidates:
            raise ValueError("weighted_candidates must not be empty")
        max_score = max(score for _, score in weighted_candidates)
        safe_temperature = max(temperature, 1e-6)
        probabilities = [
            (node_idx, math.exp((score - max_score) / safe_temperature))
            for node_idx, score in weighted_candidates
        ]
        total = sum(weight for _, weight in probabilities)
        if total <= 0 or not math.isfinite(total):
            return weighted_candidates[int(random.random() * len(weighted_candidates))][0]
        cursor = random.random() * total
        acc = 0.0
        for node_idx, weight in probabilities:
            acc += weight
            if acc >= cursor:
                return node_idx
        return probabilities[-1][0]

    def _skew_mutate_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
        context: DNNContext,
        role: str = "balanced",
    ) -> List[int]:
        """基于边缘资源状态执行偏斜变异。"""
        mutated = self._assignment_slice(dnn_index, assignment)
        if random.random() >= self.mutate_pm:
            return mutated
        task_pool = sorted(context.critical_path_tasks) or list(range(len(mutated)))
        if random.random() < 0.5:
            task_idx = task_pool[int(random.random() * len(task_pool))]
        else:
            task_idx = int(random.random() * len(mutated))
        required_level = 1
        for pred in context.predecessors[task_idx]:
            required_level = max(required_level, self.env.nodes[mutated[pred]].level)
        weighted = [
            (
                node_idx,
                self._candidate_score(dnn_index, task_idx, node_idx, mutated, context, role),
            )
            for node_idx in context.candidate_nodes[task_idx]
            if self.env.nodes[node_idx].level >= required_level
            and node_idx != mutated[task_idx]
        ]
        if not weighted:
            return mutated
        min_score = min(score for _, score in weighted)
        max_score = max(score for _, score in weighted)
        if max_score > min_score:
            weighted = [
                (node_idx, (score - min_score) / (max_score - min_score))
                for node_idx, score in weighted
            ]
        mutated[task_idx] = self._weighted_choice(weighted)
        repair = self._repair_assignment_result(dnn_index, mutated, context)
        if repair.success:
            self.operator_stats["effective_mutations"] = self.operator_stats.get("effective_mutations", 0) + 1
            return repair.assignment
        self.operator_stats["failed_mutations"] = self.operator_stats.get("failed_mutations", 0) + 1
        return self._assignment_slice(dnn_index, assignment)

    def _dag_crossover_block(self, context: DNNContext, task_count: int) -> List[int]:
        """选择一个大小受控的后继闭包作为 DAG 交叉块。"""
        if task_count <= 1:
            return list(range(task_count))
        max_block_size = max(2, math.ceil(task_count * 0.4))
        blocks: List[List[int]] = []
        fallback_blocks: List[List[int]] = []
        for anchor in context.topological_order:
            descendants: Set[int] = {anchor}
            pending = [anchor]
            while pending:
                current = pending.pop()
                for successor in context.successors[current]:
                    if successor not in descendants:
                        descendants.add(successor)
                        pending.append(successor)
            block = sorted(descendants)
            if len(block) < task_count:
                fallback_blocks.append(block)
            if 2 <= len(block) <= max_block_size:
                blocks.append(block)
        candidates = blocks or fallback_blocks
        if not candidates:
            return [context.topological_order[-1]]
        if not blocks:
            min_size = min(len(block) for block in candidates)
            candidates = [block for block in candidates if len(block) == min_size]
        return candidates[int(random.random() * len(candidates))]

    def _dag_block_crossover(
        self,
        dnn_index: int,
        parent_a: List[int],
        parent_b: List[int],
        context: DNNContext,
    ) -> tuple[List[int], List[int]]:
        """交换大小受控的 DAG 后继闭包，并限制 repair 对遗传信息的覆盖。"""
        task_count = len(self.env.ds[dnn_index].tasks)
        child_a = self._assignment_slice(dnn_index, parent_a)
        child_b = self._assignment_slice(dnn_index, parent_b)
        if task_count <= 1:
            return child_a, child_b
        swapped_tasks = self._dag_crossover_block(context, task_count)
        for task_idx in swapped_tasks:
            child_a[task_idx] = parent_b[task_idx]
            child_b[task_idx] = parent_a[task_idx]
        repair_a = self._repair_assignment_result(dnn_index, child_a, context)
        repair_b = self._repair_assignment_result(dnn_index, child_b, context)
        repair_limit = max(1, len(swapped_tasks) // 2)
        if repair_a.success and repair_a.changed_gene_count <= repair_limit:
            child_a = repair_a.assignment
        else:
            child_a = self._assignment_slice(dnn_index, parent_a)
        if repair_b.success and repair_b.changed_gene_count <= repair_limit:
            child_b = repair_b.assignment
        else:
            child_b = self._assignment_slice(dnn_index, parent_b)
        if child_a != self._assignment_slice(dnn_index, parent_a):
            self.operator_stats["effective_crossovers"] = self.operator_stats.get("effective_crossovers", 0) + 1
        else:
            self.operator_stats["no_change_crossovers"] = self.operator_stats.get("no_change_crossovers", 0) + 1
        return child_a, child_b

    def _write_customized_child(
        self,
        dnn_index: int,
        parent_a_idx: int,
        parent_b_idx: int,
        target_index: int,
        context: DNNContext,
        role: str = "balanced",
    ) -> None:
        """生成并写入定制化 RDODA 子代。"""
        parent_a = self._assignment_slice(dnn_index, self.res[parent_a_idx][dnn_index])
        parent_b = self._assignment_slice(dnn_index, self.res[parent_b_idx][dnn_index])
        if random.random() < self.cross_over_pm:
            self.operator_stats["crossover_attempts"] = self.operator_stats.get("crossover_attempts", 0) + 1
            if self.use_dag_block_crossover:
                child_a, child_b = self._dag_block_crossover(
                    dnn_index,
                    parent_a,
                    parent_b,
                    context,
                )
            else:
                child_a = list(parent_a)
                child_b = list(parent_b)
                if len(child_a) > 1:
                    boundary = 1 + int(random.random() * (len(child_a) - 1))
                    for task_idx in range(boundary, len(child_a)):
                        child_a[task_idx], child_b[task_idx] = child_b[task_idx], child_a[task_idx]
                repair_a = self._repair_assignment_result(dnn_index, child_a, context)
                repair_b = self._repair_assignment_result(dnn_index, child_b, context)
                child_a = repair_a.assignment if repair_a.success else parent_a
                child_b = repair_b.assignment if repair_b.success else parent_b
        else:
            child_a = parent_a
            child_b = parent_b
        if self.use_skew_mutation:
            child_a = self._skew_mutate_assignment(dnn_index, child_a, context, role)
            child_b = self._skew_mutate_assignment(dnn_index, child_b, context, role)
        else:
            child_a = self._mutate_assignment(dnn_index, child_a)
            child_b = self._mutate_assignment(dnn_index, child_b)
        self._write_child(dnn_index, child_a, target_index)
        self._write_child(dnn_index, child_b, target_index + 1)

    def _heuristic_assignment(self, dnn_index: int, context: DNNContext, mode: str) -> List[int]:
        """按指定启发式构造一个定制化初始个体。"""
        dnn = self.env.ds[dnn_index]
        assignment = [-1 for _ in range(len(dnn.tasks))]
        cloud_candidates = [idx for idx, node in enumerate(self.env.nodes) if node.level == 3]
        cloud_index = cloud_candidates[0] if cloud_candidates else 0
        for task_idx in context.topological_order:
            if mode == "cloud":
                assignment[task_idx] = cloud_index
            elif mode == "local":
                assignment[task_idx] = self._best_candidate_node(dnn_index, task_idx, assignment, context, min_level=2)
            elif mode == "resource":
                candidates = context.candidate_nodes[task_idx]
                assignment[task_idx] = max(candidates, key=lambda idx: self.env.nodes[idx].cpu)
            elif mode == "stability":
                candidates = context.candidate_nodes[task_idx]
                assignment[task_idx] = max(
                    candidates,
                    key=lambda idx: self.env.nodes[idx].operational_stability,
                )
            elif mode == "topology":
                assignment[task_idx] = self._best_candidate_node(dnn_index, task_idx, assignment, context)
            else:
                candidates = context.candidate_nodes[task_idx]
                assignment[task_idx] = candidates[int(random.random() * len(candidates))]
        return self._repair_assignment(dnn_index, assignment, context)

    def _initialize_customized_population(self, dnn_index: int, context: DNNContext) -> bool:
        """用去重模板、模板扰动和随机可行解构造多样化种群。"""
        if not self.use_dag_aware_initialization:
            return self._initialize_random_customized_population(dnn_index, context)
        task_count = len(self.env.ds[dnn_index].tasks)
        modes = ["cloud", "local", "resource", "stability", "topology"]
        assignments: List[List[int]] = []
        seen: Set[tuple[int, ...]] = set()

        def add_assignment(assignment: List[int]) -> bool:
            key = tuple(self._assignment_slice(dnn_index, assignment))
            if key in seen or not self._is_hierarchy_valid(dnn_index, assignment, context):
                return False
            seen.add(key)
            assignments.append(list(key))
            return True

        for mode in modes:
            assignment = self._heuristic_assignment(dnn_index, context, mode)
            add_assignment(assignment)

        target_unique = min(self.pop_size, max(len(assignments), math.ceil(self.pop_size * 0.75)))
        attempts = 0
        max_attempts = max(300, self.pop_size * 40)
        while len(assignments) < target_unique and attempts < max_attempts:
            attempts += 1
            if assignments and random.random() < 0.45:
                candidate = list(assignments[int(random.random() * len(assignments))])
                task_idx = int(random.random() * task_count)
                nodes = [
                    node_idx
                    for node_idx in context.candidate_nodes[task_idx]
                    if node_idx != candidate[task_idx]
                ]
                if not nodes:
                    continue
                candidate[task_idx] = nodes[int(random.random() * len(nodes))]
            else:
                candidate = [
                    context.candidate_nodes[task_idx][
                        int(random.random() * len(context.candidate_nodes[task_idx]))
                    ]
                    for task_idx in range(task_count)
                ]
            repair = self._repair_assignment_result(dnn_index, candidate, context)
            if repair.success:
                add_assignment(repair.assignment)

        if not assignments:
            return False
        while len(assignments) < self.pop_size:
            assignments.append(list(assignments[int(random.random() * len(assignments))]))

        random.shuffle(assignments)
        for individual_idx, assignment in enumerate(assignments[: self.pop_size]):
            for task_idx, node_idx in enumerate(assignment):
                self.res[individual_idx][dnn_index][task_idx] = node_idx
        self.population_roles = [
            self.customized_membrane_roles[
                individual_idx % len(self.customized_membrane_roles)
            ]
            for individual_idx in range(self.pop_size)
        ]
        self.operator_stats["initial_unique_count"] = len(seen)
        self.operator_stats["initial_duplicate_count"] = self.pop_size - len(seen)
        self.operator_stats["initialization_retry_count"] = attempts
        return True

    def _initialize_random_customized_population(self, dnn_index: int, context: DNNContext) -> bool:
        """不使用 DAG 启发式时的随机可行初始化。"""
        task_count = len(self.env.ds[dnn_index].tasks)
        seen: Set[tuple[int, ...]] = set()
        for individual_idx in range(self.pop_size):
            for _ in range(300):
                assignment = [
                    context.candidate_nodes[task_idx][
                        int(random.random() * len(context.candidate_nodes[task_idx]))
                    ]
                    for task_idx in range(task_count)
                ]
                repair = self._repair_assignment_result(dnn_index, assignment, context)
                key = tuple(repair.assignment)
                if repair.success and (key not in seen or len(seen) >= math.ceil(self.pop_size * 0.75)):
                    seen.add(key)
                    for task_idx, node_idx in enumerate(repair.assignment):
                        self.res[individual_idx][dnn_index][task_idx] = node_idx
                    break
            else:
                return False
        self.operator_stats["initial_unique_count"] = len(seen)
        self.operator_stats["initial_duplicate_count"] = self.pop_size - len(seen)
        self.population_roles = [
            self.customized_membrane_roles[
                individual_idx % len(self.customized_membrane_roles)
            ]
            for individual_idx in range(self.pop_size)
        ]
        return True

    def _mutate_assignment(self, dnn_index: int, assignment: List[int]) -> List[int]:
        """对临时子代执行一次独立变异并返回结果。"""
        mutated = list(assignment)
        if random.random() >= self.mutate_pm:
            return mutated
        dnn = self.env.ds[dnn_index]
        task_count = len(dnn.tasks)
        attempts = 0
        while attempts < 200:
            attempts += 1
            task_idx = int(random.random() * task_count)
            node_idx = int(random.random() * len(self.env.nodes))
            original = mutated[task_idx]
            mutated[task_idx] = node_idx
            if self._is_assignment_valid(dnn_index, mutated):
                return mutated
            mutated[task_idx] = original
        return list(assignment)

    def _write_child(self, dnn_index: int, child: List[int], target_index: int) -> None:
        """把子代写入联合种群指定槽位。"""
        task_count = len(self.env.ds[dnn_index].tasks)
        for x in range(task_count):
            self.res_pq[target_index][dnn_index][x] = child[x]

    def _pick_parent_from_membrane(self, role_index: int) -> int:
        """从指定子膜中选择一个父代个体。"""
        role = self.customized_membrane_roles[role_index]
        candidates = [
            idx for idx, current_role in enumerate(self.population_roles)
            if current_role == role
        ]
        if not candidates:
            return int(random.random() * self.pop_size)
        return candidates[int(random.random() * len(candidates))]

    def _constraint_violation(
        self,
        dnn_index: int,
        assignment: List[int],
        context: DNNContext,
        delay: float | None = None,
    ) -> float:
        """从统一候选快照读取归一化约束违反度。"""
        del context, delay
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        snapshot = self._active_search_snapshot
        cache_key = (
            dnn_index,
            tuple(assignment_slice),
            (
                snapshot.environment_state_version
                if snapshot is not None
                else self.env.environment_state_version
            ),
            self.env.objective_semantics_version,
            snapshot.snapshot_version if snapshot is not None else -1,
        )
        cached = self._constraint_violation_cache.get(cache_key)
        if cached is not None:
            return cached
        candidate = self._candidate_state_for_assignment(
            dnn_index,
            assignment_slice,
        )
        violation = candidate.constraint_violation
        self._constraint_violation_cache[cache_key] = violation
        return violation

    def _customized_epsilon(self, generation: int) -> float:
        """动态约束阈值，随代数从宽松收缩到严格可行。"""
        if self.iteration_limit <= 1:
            return 0.0
        progress = min(max(generation / (self.iteration_limit - 1), 0.0), 1.0)
        return self.cv_initial_epsilon * (1.0 - progress)

    def _dominates_with_cv(self, i: int, j: int, epsilon: float) -> bool:
        """按动态约束支配原则判断 i 是否支配 j。"""
        cv_i = self.constraint_violations[i]
        cv_j = self.constraint_violations[j]
        feasible_i = cv_i <= epsilon
        feasible_j = cv_j <= epsilon
        if feasible_i and not feasible_j:
            return True
        if not feasible_i and feasible_j:
            return False
        if not feasible_i and not feasible_j:
            return cv_i < cv_j
        return self._dominates_objectives(
            (
                self.function1_values[i],
                self.function2_values[i],
                self.function3_values[i],
            ),
            (
                self.function1_values[j],
                self.function2_values[j],
                self.function3_values[j],
            ),
        )

    def dominated_sort_with_cv(self, o: int, population_size: int, epsilon: float) -> None:
        """结合动态约束违反度执行非支配排序。"""
        rank = 1
        dominates = [[0 for _ in range(population_size)] for _ in range(population_size)]
        assigned_count = 0
        assigned = [0 for _ in range(population_size)]
        for i in range(population_size):
            for j in range(population_size):
                if i != j and self._dominates_with_cv(i, j, epsilon):
                    dominates[i][j] = 1
        while assigned_count < population_size:
            changed = False
            for j in range(population_size):
                if assigned[j] == 1:
                    continue
                dominated_count = 0
                for i in range(population_size):
                    if dominates[i][j] == 1:
                        dominated_count += 1
                if dominated_count == 0:
                    assigned[j] = 1
                    self.pareto_level_sort[o][j] = rank
                    assigned_count += 1
                    changed = True
            for i in range(population_size):
                if assigned[i] == 1:
                    for j in range(population_size):
                        dominates[i][j] = 0
            if not changed:
                for j in range(population_size):
                    if assigned[j] == 0:
                        assigned[j] = 1
                        self.pareto_level_sort[o][j] = rank
                        assigned_count += 1
                break
            rank += 1

    def update_res_with_cv(
        self,
        pq: List[List[List[int]]],
        rank: List[List[int]],
        dis: List[List[int]],
        i: int,
        epsilon: float,
    ) -> None:
        """ε 可行解按拥挤距离选择，ε 不可行解再按违反度选择。"""
        def selection_key(idx: int) -> tuple[float, ...]:
            return (
                rank[i][idx],
                0 if self.constraint_violations[idx] <= epsilon else 1,
                0.0 if self.constraint_violations[idx] <= epsilon else self.constraint_violations[idx],
                -dis[i][idx],
            )

        ordered = sorted(range(len(pq)), key=selection_key)
        joint_roles = (
            self._joint_population_roles
            if len(self._joint_population_roles) == len(pq)
            else [
                self.customized_membrane_roles[
                    idx % len(self.customized_membrane_roles)
                ]
                for idx in range(len(pq))
            ]
        )
        selected: List[int] = []
        base_quota = self.pop_size // len(self.customized_membrane_roles)
        remainder = self.pop_size % len(self.customized_membrane_roles)
        for role_index, role in enumerate(self.customized_membrane_roles):
            quota = base_quota + (1 if role_index < remainder else 0)
            role_candidates = [idx for idx in ordered if joint_roles[idx] == role]
            selected.extend(role_candidates[:quota])
        if len(selected) < self.pop_size:
            selected_set = set(selected)
            selected.extend(
                idx for idx in ordered
                if idx not in selected_set
            )
        selected = selected[: self.pop_size]
        for target_idx, source_idx in enumerate(selected):
            for task_idx in range(len(pq[source_idx][i])):
                self.res[target_idx][i][task_idx] = pq[source_idx][i][task_idx]
        self.population_roles = [joint_roles[source_idx] for source_idx in selected]

    def _local_search_score(
        self,
        dnn_index: int,
        assignment: List[int],
        w_a: float,
        w_r: float,
        w_t: float,
        w_e: float,
    ) -> tuple[ObjectiveValues, float]:
        """返回局部搜索使用的目标值、时延和综合得分。"""
        values = self._evaluate_customized_assignment(dnn_index, assignment)
        if not values.deadline_feasible:
            score = -math.inf
        else:
            score = self._score(*values.vector, 0.0, w_a, w_r, w_t, w_e)
        return values, score

    def _local_search_elites(
        self,
        dnn_index: int,
        context: DNNContext,
        w_a: float,
        w_r: float,
        w_t: float,
        w_e: float,
    ) -> int:
        """对少量精英个体执行关键路径邻域搜索。"""
        elite_count = max(1, int(self.pop_size * self.local_search_elite_ratio))
        scored = []
        for idx in range(self.pop_size):
            assignment = self._assignment_slice(dnn_index, self.res[idx][dnn_index])
            values = self._local_search_score(
                dnn_index,
                assignment,
                w_a,
                w_r,
                w_t,
                w_e,
            )
            objectives, score = values
            cv = self._constraint_violation(dnn_index, assignment, context, objectives.estimated_delay)
            scored.append((cv, -score, idx, values))
        improved = 0
        task_order = list(context.critical_path_tasks) or list(range(len(self.env.ds[dnn_index].tasks)))
        for _, _, individual_idx, base_values in sorted(scored)[:elite_count]:
            assignment = self._assignment_slice(dnn_index, self.res[individual_idx][dnn_index])
            best_assignment = list(assignment)
            best_objectives, best_score = base_values
            best_cv = self._constraint_violation(
                dnn_index,
                best_assignment,
                context,
                best_objectives.estimated_delay,
            )
            for task_idx in task_order:
                ranked_candidates = sorted(
                    context.candidate_nodes[task_idx],
                    key=lambda node_idx: self._candidate_score(
                        dnn_index,
                        task_idx,
                        node_idx,
                        best_assignment,
                        context,
                        "balanced",
                    ),
                    reverse=True,
                )
                for node_idx in ranked_candidates[: self.local_search_top_k]:
                    if node_idx == best_assignment[task_idx]:
                        continue
                    candidate = list(best_assignment)
                    candidate[task_idx] = node_idx
                    candidate = self._repair_assignment(dnn_index, candidate, context)
                    candidate_values = self._local_search_score(
                        dnn_index,
                        candidate,
                        w_a,
                        w_r,
                        w_t,
                        w_e,
                    )
                    candidate_objectives, candidate_score = candidate_values
                    candidate_cv = self._constraint_violation(
                        dnn_index,
                        candidate,
                        context,
                        candidate_objectives.estimated_delay,
                    )
                    pareto_not_worse = all(
                        candidate_value >= best_value
                        for candidate_value, best_value in zip(
                            candidate_objectives.vector,
                            best_objectives.vector,
                        )
                    )
                    if (
                        candidate_cv <= best_cv
                        and pareto_not_worse
                        and candidate_score > best_score
                    ):
                        best_assignment = candidate
                        best_objectives = candidate_objectives
                        best_score = candidate_score
                        best_cv = candidate_cv
                        improved += 1
                        break
            for task_idx, node_idx in enumerate(best_assignment):
                self.res[individual_idx][dnn_index][task_idx] = node_idx
        return improved

    def cross_over1(self, i: int, m: int, n: int, next_index: int) -> None:
        """执行按后缀交换的交叉操作，并生成两个独立子代。"""
        dnn = self.env.ds[i]
        task_count = len(dnn.tasks)
        parent_m = list(self.res[m][i])
        parent_n = list(self.res[n][i])
        child_m = list(parent_m)
        child_n = list(parent_n)
        success = False
        self.start_time = time.monotonic()
        while True:
            if (time.monotonic() - self.start_time) * 1000 > 10000:
                break
            index_m = int(task_count * random.random())
            child_m = list(parent_m)
            child_n = list(parent_n)
            for j in range(index_m, task_count):
                child_m[j] = parent_n[j]
                child_n[j] = parent_m[j]
            if self._is_assignment_valid(i, child_m) and self._is_assignment_valid(i, child_n):
                success = True
                break
        if not success:
            child_m = list(parent_m)
            child_n = list(parent_n)
        child_m = self._mutate_assignment(i, child_m)
        child_n = self._mutate_assignment(i, child_n)
        self._write_child(i, child_m, next_index)
        self._write_child(i, child_n, next_index + 1)

    def count_values(self, i: int, m: int) -> None:
        """一次性计算并写回四个统一满意度目标及其原始指标。"""
        self._write_objective_values(m, self._evaluate_assignment(i, self.res_pq[m][i]))

    def dominated_sort(self, o: int) -> None:
        """对当前种群执行非支配排序。"""
        rank = 1
        x = [[0 for _ in range(2 * self.pop_size)] for _ in range(2 * self.pop_size)]
        y = 0
        z = [0 for _ in range(2 * self.pop_size)]
        for i in range(len(self.function1_values)):
            for j in range(len(self.function1_values)):
                if i == j:
                    continue
                if self.deadline_feasible_values[i] and not self.deadline_feasible_values[j]:
                    x[i][j] = 1
                elif not self.deadline_feasible_values[i] and not self.deadline_feasible_values[j]:
                    if self.estimated_delay_values[i] < self.estimated_delay_values[j]:
                        x[i][j] = 1
                elif self.deadline_feasible_values[i] and self._dominates_objectives(
                    (
                        self.function1_values[i],
                        self.function2_values[i],
                        self.function3_values[i],
                    ),
                    (
                        self.function1_values[j],
                        self.function2_values[j],
                        self.function3_values[j],
                    ),
                ):
                    x[i][j] = 1
        while y < 2 * self.pop_size:
            for j in range(2 * self.pop_size):
                if z[j] == 1:
                    continue
                b = 0
                for i in range(2 * self.pop_size):
                    if x[i][j] == 1:
                        b += 1
                if b == 0:
                    z[j] = 1
                    self.pareto_level_sort[o][j] = rank
                    y += 1
            for i in range(2 * self.pop_size):
                if z[i] == 1:
                    for j in range(2 * self.pop_size):
                        x[i][j] = 0
            rank += 1

    def get_res(
        self,
        i: int,
        pq: List[List[List[int]]],
        values1: List[float],
        values2: List[float],
        values3: List[float],
    ) -> List[List[int]]:
        """按 Pareto 层分别计算联合种群的拥挤距离。"""
        inf = 2**31 - 1
        population_size = len(pq)

        # 先清空当前 DNN 对应的一行距离，避免残留旧轮次结果。
        for m in range(population_size):
            self.distance[i][m] = 0

        # 拥挤距离只在同一 Pareto 层内有意义，因此先按 rank 分组。
        fronts: dict[int, List[int]] = {}
        for m in range(population_size):
            rank = self.pareto_level_sort[i][m]
            fronts.setdefault(rank, []).append(m)

        objectives = (values1, values2, values3)
        for front in fronts.values():
            if not front:
                continue
            if len(front) <= 2:
                for idx in front:
                    self.distance[i][idx] = inf
                continue

            for values in objectives:
                sorted_front = sorted(front, key=lambda idx: values[idx])
                left = sorted_front[0]
                right = sorted_front[-1]
                self.distance[i][left] = inf
                self.distance[i][right] = inf

                min_value = values[left]
                max_value = values[right]
                denominator = max_value - min_value
                if denominator == 0:
                    continue

                for pos in range(1, len(sorted_front) - 1):
                    curr = sorted_front[pos]
                    if self.distance[i][curr] == inf:
                        continue
                    prev_idx = sorted_front[pos - 1]
                    next_idx = sorted_front[pos + 1]
                    self.distance[i][curr] += _java_div(
                        abs(values[next_idx] - values[prev_idx]),
                        denominator,
                    )
        return self.distance

    def update_res(self, pq: List[List[List[int]]], rank: List[List[int]], dis: List[List[int]], i: int) -> None:
        """根据排序层级和距离更新下一代种群。"""
        index = 0
        level = 1
        has = [0 for _ in range(len(pq))]
        while index < self.pop_size:
            level_num = 0
            for m in range(len(rank[i])):
                if rank[i][m] == level:
                    level_num += 1
            if (self.pop_size - index) >= level_num:
                for m in range(len(rank[i])):
                    if rank[i][m] == level:
                        for n in range(len(pq[m][i])):
                            self.res[index][i][n] = pq[m][i][n]
                        has[m] = 1
                        index += 1
                level += 1
            else:
                while index < self.pop_size:
                    post = -1
                    for m in range(len(rank[i])):
                        if rank[i][m] == level and has[m] == 0:
                            if post == -1 or (
                                self.deadline_feasible_values[m]
                                and not self.deadline_feasible_values[post]
                            ):
                                post = m
                            elif dis[i][m] > dis[i][post]:
                                post = m
                    has[post] = 1
                    for n in range(len(pq[post][i])):
                        self.res[index][i][n] = pq[post][i][n]
                    index += 1
                level += 1

    def _score(
        self,
        value1: float,
        value2: float,
        value3: float,
        value4: float,
        w_a: float,
        w_r: float,
        w_t: float,
        w_e: float,
    ) -> float:
        """对统一满意度向量加权。"""
        return (
            w_a * self._clamp_satisfaction(value1)
            + w_r * self._clamp_satisfaction(value2)
            + w_t * self._clamp_satisfaction(value3)
            + w_e * self._clamp_satisfaction(value4)
        )

    def _best_strict_feasible_index(
        self,
        population_size: int,
        weights: tuple[float, float, float, float],
        tolerance: float = 1e-12,
    ) -> int | None:
        """从已评价种群中选择严格可行的偏好最优解。"""
        w_a, w_r, w_t, w_e = weights
        candidates = [
            idx
            for idx in range(population_size)
            if self.deadline_feasible_values[idx]
            and self.constraint_violations[idx] <= tolerance
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda idx: self._score(
                self.function1_values[idx],
                self.function2_values[idx],
                self.function3_values[idx],
                self.function4_values[idx],
                w_a,
                w_r,
                w_t,
                w_e,
            ),
        )

    def generate_customized_candidates(
        self,
        *,
        dnn_index: int,
        snapshot: ObservationSnapshot,
        search_config: CustomizedSearchConfig | None = None,
    ) -> CustomizedSearchResult:
        """在固定观测快照上运行原 customized 搜索核心，不提交环境状态。"""
        if snapshot.purpose not in {"planning", "online_admission"}:
            raise ValueError("customized search requires a planning or admission snapshot")
        config = search_config or CustomizedSearchConfig(
            population_size=self.pop_size,
            generations=self.iteration_limit,
        )
        seed_material = repr(
            (
                config.random_seed,
                snapshot.snapshot_version,
                snapshot.environment_state_version,
                dnn_index,
                self.env.profile_id_for_dnn(dnn_index),
                "customized_periodic_pool_v1",
            )
        ).encode()
        import hashlib

        effective_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            "big",
        )
        started = time.monotonic()
        global_rng_state = random.getstate()
        previous_snapshot = self._active_search_snapshot
        previous_pop_size = self.pop_size
        previous_iteration_limit = self.iteration_limit
        previous_distance = self.distance
        archive: dict[
            tuple[int, ...],
            tuple[CandidateStateSnapshot, int, str],
        ] = {}
        evaluated_states: dict[tuple[int, ...], CandidateStateSnapshot] = {}
        generations_completed = 0
        termination_reason = "generation_limit"

        def collect(population: List[List[List[int]]], generation: int, source: str) -> None:
            for individual in population:
                assignment = tuple(self._assignment_slice(dnn_index, individual[dnn_index]))
                if assignment in evaluated_states:
                    candidate = evaluated_states[assignment]
                else:
                    candidate = self._candidate_state_for_assignment(
                        dnn_index,
                        list(assignment),
                        snapshot=snapshot,
                    )
                    evaluated_states[assignment] = candidate
                if (
                    candidate.is_strictly_feasible
                    and candidate.fixed_point_converged
                    and candidate.constraint_violation <= 1e-12
                ):
                    archive.setdefault(assignment, (candidate, generation, source))

        try:
            random.seed(effective_seed)
            self._active_search_snapshot = snapshot
            self.pop_size = config.population_size
            self.iteration_limit = config.generations
            self.distance = [
                [0 for _ in range(2 * self.pop_size)]
                for _ in range(max(1000, len(self.env.ds)))
            ]
            self._path_transfer_cost_cache = {}
            self._assignment_evaluation_cache = {}
            self._candidate_state_cache = {}
            self._constraint_violation_cache = {}
            self.operator_stats = {}
            self.res = self._new_population(self.pop_size)
            self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
            self.estimated_delay_values = [math.inf for _ in range(2 * self.pop_size)]
            self.total_energy_values = [math.inf for _ in range(2 * self.pop_size)]
            self.deadline_feasible_values = [False for _ in range(2 * self.pop_size)]
            self.constraint_violations = [0.0 for _ in range(2 * self.pop_size)]
            context = self._build_dnn_context(dnn_index, snapshot)
            if not all(context.candidate_nodes.values()):
                termination_reason = "no_legal_nodes"
            elif not self._initialize_customized_population(dnn_index, context):
                termination_reason = "initialization_failed"
            else:
                collect(self.res, 0, "initialization")
                for generation in range(config.generations):
                    if (
                        config.max_evaluations is not None
                        and len(evaluated_states) >= config.max_evaluations
                    ):
                        termination_reason = "evaluation_budget"
                        break
                    if (
                        config.wall_time_budget_ms is not None
                        and (time.monotonic() - started) * 1000.0
                        >= config.wall_time_budget_ms
                    ):
                        termination_reason = "wall_time_budget"
                        break
                    self.pareto_level_sort = [
                        [0 for _ in range(2 * self.pop_size)]
                        for _ in range(len(self.env.ds))
                    ]
                    self.res_pq = self._new_population(2 * self.pop_size)
                    for individual_idx in range(len(self.res)):
                        for current_dnn in range(len(self.res[individual_idx])):
                            self.res_pq[individual_idx][current_dnn] = list(
                                self.res[individual_idx][current_dnn]
                            )
                    self._joint_population_roles = [
                        *self.population_roles,
                        *["balanced" for _ in range(self.pop_size)],
                    ]
                    for child_offset in range(0, self.pop_size, 2):
                        role_index = (child_offset // 2) % len(self.customized_membrane_roles)
                        role = self.customized_membrane_roles[role_index]
                        parent_a = self._pick_parent_from_membrane(role_index)
                        if generation > 0 and generation % 5 == 0:
                            parent_b = int(random.random() * self.pop_size)
                        else:
                            parent_b = self._pick_parent_from_membrane(role_index)
                        target_index = self.pop_size + child_offset
                        self._write_customized_child(
                            dnn_index,
                            parent_a,
                            parent_b,
                            target_index,
                            context,
                            role,
                        )
                        self._joint_population_roles[target_index] = role
                        self._joint_population_roles[target_index + 1] = role

                    for individual_idx in range(len(self.res_pq)):
                        delay = self._count_customized_values(dnn_index, individual_idx)
                        self.constraint_violations[individual_idx] = self._constraint_violation(
                            dnn_index,
                            self.res_pq[individual_idx][dnn_index],
                            context,
                            delay,
                        )
                    collect(self.res_pq, generation, "evolution")
                    epsilon = self._customized_epsilon(generation)
                    self.dominated_sort_with_cv(dnn_index, len(self.res_pq), epsilon)
                    self.get_res(
                        dnn_index,
                        self.res_pq,
                        self.function1_values,
                        self.function2_values,
                        self.function3_values,
                    )
                    self.update_res_with_cv(
                        self.res_pq,
                        self.pareto_level_sort,
                        self.distance,
                        dnn_index,
                        epsilon,
                    )
                    if self.use_elite_local_search:
                        weights = self._resolve_preference_weights(
                            dnn_index,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            200.0,
                            2000.0,
                        )
                        self._local_search_elites(dnn_index, context, *weights)
                        collect(self.res, generation, "local_search")
                    generations_completed = generation + 1

            def dominates(
                first: CandidateStateSnapshot,
                second: CandidateStateSnapshot,
            ) -> bool:
                first_values = (
                    first.operational_stability,
                    first.delay_satisfaction,
                    first.energy_satisfaction,
                )
                second_values = (
                    second.operational_stability,
                    second.delay_satisfaction,
                    second.energy_satisfaction,
                )
                return all(
                    left >= right - 1e-12
                    for left, right in zip(first_values, second_values)
                ) and any(
                    left > right + 1e-12
                    for left, right in zip(first_values, second_values)
                )

            nondominated = [
                entry
                for assignment, entry in archive.items()
                if not any(
                    other_assignment != assignment
                    and dominates(other_entry[0], entry[0])
                    for other_assignment, other_entry in archive.items()
                )
            ]
            nondominated.sort(
                key=lambda entry: (
                    -sum(
                        (
                            entry[0].operational_stability,
                            entry[0].delay_satisfaction,
                            entry[0].energy_satisfaction,
                        )
                    ),
                    entry[0].assignment,
                )
            )
            selected_archive = nondominated[: config.archive_capacity]
            candidates = tuple(
                CustomizedCandidate(
                    assignment=candidate.assignment,
                    objectives=(
                        candidate.operational_stability,
                        candidate.delay_satisfaction,
                        candidate.energy_satisfaction,
                    ),
                    constraint_violation=candidate.constraint_violation,
                    generation=generation,
                    source=source,
                )
                for candidate, generation, source in selected_archive
            )
            nonconverged = sum(
                not candidate.fixed_point_converged
                for candidate in evaluated_states.values()
            )
            constraint_rejected = sum(
                candidate.fixed_point_converged and not candidate.is_strictly_feasible
                for candidate in evaluated_states.values()
            )
            result = CustomizedSearchResult(
                candidates=candidates,
                evaluations=len(evaluated_states),
                generations_completed=generations_completed,
                feasible_evaluations=sum(
                    candidate.fixed_point_converged and candidate.is_strictly_feasible
                    for candidate in evaluated_states.values()
                ),
                constraint_rejected_evaluations=constraint_rejected,
                nonconverged_evaluations=nonconverged,
                archive_size_before_selection=len(nondominated),
                termination_reason=termination_reason,
                wall_time_ms=(time.monotonic() - started) * 1000.0,
                seed=effective_seed,
            )
            self.last_customized_search_result = result
            return result
        finally:
            random.setstate(global_rng_state)
            self._active_search_snapshot = previous_snapshot
            self.pop_size = previous_pop_size
            self.iteration_limit = previous_iteration_limit
            self.distance = previous_distance

    def search_customized_plan_pool(
        self,
        dnn_index: int,
        snapshot: ObservationSnapshot,
        profile_id: str,
        origin_group: int,
        max_pool_size: int = 20,
        ttl_slots: int = 10,
        semantic_versions: tuple[tuple[str, str], ...] | None = None,
        search_config: CustomizedSearchConfig | None = None,
    ):
        """使用 customized 搜索档案构建周期性部署方案池。"""
        from planning.repository import make_deployment_plan, select_diverse_plans

        if snapshot.purpose != "planning":
            raise ValueError("search_customized_plan_pool requires a planning snapshot")
        if profile_id != self.env.profile_id_for_dnn(dnn_index):
            raise ValueError("profile_id does not match the selected DNN")
        if max_pool_size <= 0:
            raise ValueError("max_pool_size must be positive")
        versions = semantic_versions or PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        result = self.generate_customized_candidates(
            dnn_index=dnn_index,
            snapshot=snapshot,
            search_config=search_config,
        )
        plans = []
        for archived in result.candidates:
            candidate = self.env.predict_candidate_state(
                dnn_index,
                list(archived.assignment),
                snapshot,
            )
            if not candidate.is_strictly_feasible or not candidate.fixed_point_converged:
                continue
            plans.append(
                make_deployment_plan(
                    self.env,
                    dnn_index,
                    candidate,
                    profile_id=profile_id,
                    origin_group=origin_group,
                    created_slot=snapshot.slot,
                    ttl_slots=ttl_slots,
                    semantic_versions=versions,
                    generator_mode="customized",
                    generator_version="customized_periodic_pool_v1",
                    search_seed=result.seed,
                    source_generation=archived.generation,
                )
            )
        selected = select_diverse_plans(plans, max_size=max_pool_size)
        self.last_plan_pool_search_statistics = PlanPoolSearchStatistics(
            generated_assignments=result.evaluations,
            constraint_rejected_candidates=result.constraint_rejected_evaluations,
            nonconverged_candidates=result.nonconverged_evaluations,
            feasible_candidates=result.feasible_evaluations,
            pareto_candidates=result.archive_size_before_selection,
            selected_candidates=len(selected),
        )
        return selected

    def search_plan_pool(
        self,
        dnn_index: int,
        snapshot: ObservationSnapshot,
        profile_id: str,
        origin_group: int,
        max_pool_size: int = 20,
        sample_count: int = 80,
        ttl_slots: int = 10,
        semantic_versions: tuple[tuple[str, str], ...] | None = None,
    ):
        """基于显式规划快照生成无副作用、结构多样的候选方案池。"""
        from planning.repository import make_deployment_plan, select_diverse_plans

        if snapshot.purpose != "planning":
            raise ValueError("search_plan_pool requires a planning snapshot")
        if profile_id != self.env.profile_id_for_dnn(dnn_index):
            raise ValueError("profile_id does not match the selected DNN")
        if max_pool_size <= 0 or sample_count <= 0:
            raise ValueError("pool and sample sizes must be positive")
        versions = semantic_versions or PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        dnn = self.env.ds[dnn_index]
        legal_nodes = [
            node_index
            for node_index, node in enumerate(self.env.nodes)
            if snapshot.node_online[node_index]
            and (node.level != 1 or node_index == dnn.initiateNode)
        ]
        if not legal_nodes:
            self.last_plan_pool_search_statistics = PlanPoolSearchStatistics()
            return ()

        assignments: set[tuple[int, ...]] = set()
        for node_index in legal_nodes:
            assignments.add(tuple(node_index for _ in dnn.tasks))
        for offset in range(min(len(legal_nodes), max_pool_size)):
            assignments.add(
                tuple(
                    legal_nodes[(task_index + offset) % len(legal_nodes)]
                    for task_index in range(len(dnn.tasks))
                )
            )

        seed = (
            snapshot.snapshot_version * 1_000_003
            + dnn_index * 10_007
            + origin_group * 101
            + len(dnn.tasks)
        )
        local_rng = random.Random(seed)
        for _ in range(sample_count):
            remaining = list(snapshot.node_available_cpu)
            assignment: list[int] = []
            for task in dnn.tasks:
                choices = [
                    node_index
                    for node_index in legal_nodes
                    if remaining[node_index] + 1e-12 >= task.cpu_need
                ]
                if not choices:
                    assignment = []
                    break
                node_index = choices[local_rng.randrange(len(choices))]
                assignment.append(node_index)
                remaining[node_index] -= task.cpu_need
            if assignment:
                assignments.add(tuple(assignment))

        evaluated: list[CandidateStateSnapshot] = []
        constraint_rejected = 0
        nonconverged = 0
        for assignment in sorted(assignments):
            candidate = self.env.predict_candidate_state(
                dnn_index,
                list(assignment),
                snapshot,
            )
            if not candidate.is_strictly_feasible:
                constraint_rejected += 1
                continue
            if not candidate.fixed_point_converged:
                nonconverged += 1
                continue
            evaluated.append(candidate)

        def dominates(first: CandidateStateSnapshot, second: CandidateStateSnapshot) -> bool:
            first_vector = (
                first.operational_stability,
                first.delay_satisfaction,
                first.energy_satisfaction,
            )
            second_vector = (
                second.operational_stability,
                second.delay_satisfaction,
                second.energy_satisfaction,
            )
            return all(
                left >= right - 1e-12
                for left, right in zip(first_vector, second_vector)
            ) and any(
                left > right + 1e-12
                for left, right in zip(first_vector, second_vector)
            )

        pareto = [
            candidate
            for candidate in evaluated
            if not any(
                dominates(other, candidate)
                for other in evaluated
                if other is not candidate
            )
        ]
        plans = [
            make_deployment_plan(
                self.env,
                dnn_index,
                candidate,
                profile_id=profile_id,
                origin_group=origin_group,
                created_slot=snapshot.slot,
                ttl_slots=ttl_slots,
                semantic_versions=versions,
                generator_mode="lightweight",
                generator_version="lightweight_snapshot_pool_v1",
                search_seed=seed,
            )
            for candidate in pareto
        ]
        selected = select_diverse_plans(plans, max_size=max_pool_size)
        self.last_plan_pool_search_statistics = PlanPoolSearchStatistics(
            generated_assignments=len(assignments),
            constraint_rejected_candidates=constraint_rejected,
            nonconverged_candidates=nonconverged,
            feasible_candidates=len(evaluated),
            pareto_candidates=len(pareto),
            selected_candidates=len(selected),
        )
        return selected

    def run_proposed(self) -> ExperimentMetrics:
        """运行主算法并输出平均指标。"""
        self.pareto_points = []
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        self.res = self._new_population(self.pop_size)
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        run_time = time.monotonic()
        next_dnn_index = 0
        # 外层循环按 DNN 到达顺序驱动系统运行；
        # 当没有新 DNN 时，只继续推进运行中的任务。
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            # 当所有 DNN 都已到达后，只需要继续推进系统，
            # 直到已接纳的运行中 DNN 全部执行完成。
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            self._log_proposed_stage(
                t,
                "search_start",
                tasks=len(self.env.ds[t].tasks),
                deadline=self.env.ds[t].delay,
                iteration_limit=self.iteration_limit,
            )
            # 为当前 DNN 重置本轮搜索状态，并记录当前已知最优解。
            best_value = [0.0, 0.0, -1.0, 0.0]
            best_assignment = [0 for _ in range(self.max_dnn_num)]
            self.res = self._new_population(self.pop_size)
            self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
            self.estimated_delay_values = [math.inf for _ in range(2 * self.pop_size)]
            self.total_energy_values = [math.inf for _ in range(2 * self.pop_size)]
            self.deadline_feasible_values = [False for _ in range(2 * self.pop_size)]
            self.constraint_violations = [0.0 for _ in range(2 * self.pop_size)]
            w_a, w_r, w_t, w_e = self._resolve_preference_weights(
                t,
                min_a,
                max_a,
                min_r,
                max_r,
                min_t,
                max_t,
            )
            o = 0
            index = 0
            best_gen = 0
            # 阶段一：进入当前 DNN 的进化搜索过程。
            while index < self.iteration_limit:
                if o == 1:
                    break
                if index == 0 or index == self.iteration_limit or index % 10 == 0:
                    self._log_proposed_stage(t, "generation_start", generation=index)
                self.pareto_level_sort = [[0 for _ in range(2 * self.pop_size)] for _ in range(len(self.env.ds))]
                if index == 0:
                    self._log_proposed_stage(t, "initialize_population", population=self.pop_size)
                    # 首代种群由三部分组成：
                    # 一个全云解、若干随机可行解、以及两个额外满足时延约束的补充解。
                    cloud_index = 0
                    for i in range(len(self.env.nodes)):
                        if self.env.nodes[i].level == 3:
                            cloud_index = i
                            break
                    for i in range(len(self.env.ds[t].tasks)):
                        self.res[0][t][i] = cloud_index
                    for j in range(1, self.pop_size - 2):
                        for k in range(self.max_dnn_num):
                            self.res[j][t][k] = -1
                        for i in range(len(self.env.ds[t].tasks)):
                            xs = [[0 for _ in range(len(self.env.nodes))] for _ in range(len(self.env.ds[t].tasks))]
                            for p in range(len(self.env.ds[t].tasks)):
                                if self.res[j][t][p] != -1:
                                    xs[p][self.res[j][t][p]] = 1
                            self.start_time = time.monotonic()
                            while True:
                                if (time.monotonic() - self.start_time) * 1000 > 10000:
                                    o = 1
                                    break
                                n = int(random.random() * len(self.env.nodes))
                                xs[i][n] = 1
                                if self.env.check_resource(self.env.ds[t], xs) and (self.env.nodes[n].level != 1 or n == self.env.ds[t].initiateNode):
                                    self.res[j][t][i] = n
                                    break
                                xs[i][n] = 0
                            if o == 1:
                                break
                        if o == 1:
                            break
                    if o == 1:
                        self._log_proposed_stage(t, "initialize_population_timeout")
                        continue
                    self.start_time = time.monotonic()
                    for j in range(self.pop_size - 2, self.pop_size):
                        for k in range(self.max_dnn_num):
                            self.res[j][t][k] = -1
                        while True:
                            if (time.monotonic() - self.start_time) * 1000 > 20000:
                                for i in range(len(self.env.ds[t].tasks)):
                                    self.res[j][t][i] = cloud_index
                                o = 0
                                break
                            xs = [[0 for _ in range(len(self.env.nodes))] for _ in range(len(self.env.ds[t].tasks))]
                            p = 0
                            while p < len(self.env.ds[t].tasks):
                                n = int(random.random() * len(self.env.nodes))
                                xs[p][n] = 1
                                if self.res[j][t][p] != -1:
                                    xs[p][self.res[j][t][p]] = 1
                                if self.env.check_resource(self.env.ds[t], xs) and (self.env.nodes[n].level != 1 or n == self.env.ds[t].initiateNode):
                                    p += 1
                                else:
                                    xs[p][n] = 0
                            if self.env.check_delay_random(self.env.ds[t], xs):
                                for p in range(len(self.env.ds[t].tasks)):
                                    for n in range(len(self.env.nodes)):
                                        if xs[p][n] == 1:
                                            self.res[j][t][p] = n
                                break
                        if o == 1:
                            break
                    if o == 1:
                        self._log_proposed_stage(t, "deadline_seed_timeout")
                        continue
                # 阶段二：基于父代构造 P+Q 联合种群。
                # 把父代复制到 P+Q，再通过交叉和变异生成新的候选个体。
                self.res_pq = self._new_population(2 * self.pop_size)
                for m in range(len(self.res)):
                    for m1 in range(len(self.res[m])):
                        for m2 in range(len(self.res[m][m1])):
                            self.res_pq[m][m1][m2] = self.res[m][m1][m2]
                for m in range(0, self.pop_size, 2):
                    a1 = int(random.random() * self.pop_size)
                    b1 = int(random.random() * self.pop_size)
                    self.cross_over1(t, a1, b1, self.pop_size + m)
                # 阶段三：评估联合种群并更新当前已知最优解。
                # 对联合种群逐个计算三目标值，并挑出本代最优候选。
                for m in range(len(self.res_pq)):
                    self.count_values(t, m)
                bi = 0
                for i in range(2 * self.pop_size):
                    if self.deadline_feasible_values[i] and not self.deadline_feasible_values[bi]:
                        bi = i
                    elif self.deadline_feasible_values[i]:
                        if not self.deadline_feasible_values[bi] or self._score(self.function1_values[i], self.function2_values[i], self.function3_values[i], self.function4_values[i], w_a, w_r, w_t, w_e) > self._score(self.function1_values[bi], self.function2_values[bi], self.function3_values[bi], self.function4_values[bi], w_a, w_r, w_t, w_e):
                            bi = i
                if (
                    (best_value[2] < 0 and self.deadline_feasible_values[bi])
                    or (
                        self.deadline_feasible_values[bi]
                        and self._score(best_value[0], best_value[1], best_value[2], best_value[3], w_a, w_r, w_t, w_e)
                        < self._score(self.function1_values[bi], self.function2_values[bi], self.function3_values[bi], self.function4_values[bi], w_a, w_r, w_t, w_e)
                    )
                ):
                    for j in range(len(self.env.ds[t].tasks)):
                        best_assignment[j] = self.res_pq[bi][t][j]
                    best_value[0] = self.function1_values[bi]
                    best_value[1] = self.function2_values[bi]
                    best_value[2] = self.function3_values[bi]
                    best_value[3] = self.function4_values[bi]
                    best_gen = index
                    self._log_proposed_stage(
                        t,
                        "generation_best_updated",
                        generation=index,
                        value1=f"{best_value[0]:.4f}",
                        value2=f"{best_value[1]:.4f}",
                        value3=f"{best_value[2]:.4f}",
                        value4=f"{best_value[3]:.6f}",
                    )
                # 阶段四：通过非支配排序和拥挤距离更新下一代父代。
                # 使用非支配排序和拥挤距离从 P+Q 中筛出下一代父代。
                self.dominated_sort(t)
                self.get_res(
                    t,
                    self.res_pq,
                    self.function1_values,
                    self.function2_values,
                    self.function3_values,
                )
                self.update_res(self.res_pq, self.pareto_level_sort, self.distance, t)
                index += 1
            if o == 1:
                self._log_proposed_stage(t, "search_failed")
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            # 阶段五：对保留下来的父代做最终选择。
            # 进化结束后，再对保留下来的父代种群做一次最终选择，
            # 防止历史 best 被后续更新漏掉。
            self.res_pq = self._new_population(self.pop_size)
            self.res_pq = self.res
            for m in range(len(self.res_pq)):
                self.count_values(t, m)
            self._append_current_pareto_points("proposed", t, len(self.res_pq))
            best_i = 0
            for i in range(self.pop_size):
                if self.deadline_feasible_values[i] and not self.deadline_feasible_values[best_i]:
                    best_i = i
                elif self.deadline_feasible_values[i]:
                    if not self.deadline_feasible_values[best_i] or self._score(self.function1_values[i], self.function2_values[i], self.function3_values[i], self.function4_values[i], w_a, w_r, w_t, w_e) > self._score(self.function1_values[best_i], self.function2_values[best_i], self.function3_values[best_i], self.function4_values[best_i], w_a, w_r, w_t, w_e):
                        best_i = i
            if (
                (best_value[2] < 0 and self.deadline_feasible_values[best_i])
                or (
                    self.deadline_feasible_values[best_i]
                    and self._score(best_value[0], best_value[1], best_value[2], best_value[3], w_a, w_r, w_t, w_e)
                    < self._score(self.function1_values[best_i], self.function2_values[best_i], self.function3_values[best_i], self.function4_values[best_i], w_a, w_r, w_t, w_e)
                )
            ):
                for j in range(len(self.env.ds[t].tasks)):
                    best_assignment[j] = self.res_pq[best_i][t][j]
                best_value[0] = self.function1_values[best_i]
                best_value[1] = self.function2_values[best_i]
                best_value[2] = self.function3_values[best_i]
                best_value[3] = self.function4_values[best_i]
                best_gen = index
            _ = best_gen
            self._log_proposed_stage(
                t,
                "final_selection",
                best_generation=best_gen,
                value1=f"{best_value[0]:.4f}",
                value2=f"{best_value[1]:.4f}",
                value3=f"{best_value[2]:.4f}",
                value4=f"{best_value[3]:.6f}",
            )
            # 若当前最优解仍不满足时延收益条件，则本次接纳失败。
            if best_value[2] < 0:
                self._log_proposed_stage(t, "rejected", reason="deadline_or_score")
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            # 阶段六：接纳最优解并把 DNN 注册为运行块。
            # 接纳成功后先把 DNN 放入运行集合，
            # 下一个时隙再统一更新负载、热度和动态可靠性。
            selected_values = self._evaluate_assignment(t, best_assignment)
            delay = selected_values.estimated_delay
            total_energy = selected_values.total_energy
            self._log_proposed_stage(
                t,
                "accepted",
                delay=f"{delay:.2f}",
                energy=f"{total_energy:.2f}",
                best_generation=best_gen,
            )
            self._register_running_dnn(t, best_assignment, delay)
            t_res += delay
            r_res += selected_values.operational_stability
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operational_stability_score=r_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_customized_proposed(self) -> ExperimentMetrics:
        """通过共享 customized 搜索内核运行逐请求对照基线。"""
        self.pareto_points = []
        total_delay = 0.0
        total_stability = 0.0
        total_energy = 0.0
        failures = 0
        successes = 0
        started = time.monotonic()
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            dnn_index = next_dnn_index
            self.env.print_pending_dnn_info(dnn_index)
            snapshot = self.env.capture_observation_snapshot("online_admission")
            result = self.generate_customized_candidates(
                dnn_index=dnn_index,
                snapshot=snapshot,
                search_config=CustomizedSearchConfig(
                    population_size=self.pop_size,
                    generations=self.iteration_limit,
                    archive_capacity=max(self.pop_size, 64),
                    random_seed=0,
                ),
            )
            weights = self._resolve_preference_weights(
                dnn_index,
                0.0,
                1.0,
                0.0,
                1.0,
                200.0,
                2000.0,
            )
            if not result.candidates:
                self._log_proposed_stage(
                    dnn_index,
                    "customized_rejected",
                    reason=result.termination_reason,
                )
                failures += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            selected = max(
                result.candidates,
                key=lambda candidate: self._score(
                    *candidate.objectives, 0.0, *weights
                ),
            )
            candidate_state = self.env.predict_candidate_state(
                dnn_index,
                list(selected.assignment),
                snapshot,
            )
            if (
                not candidate_state.is_strictly_feasible
                or not candidate_state.fixed_point_converged
            ):
                self._log_proposed_stage(
                    dnn_index,
                    "customized_rejected",
                    reason="final_revalidation_failed",
                )
                failures += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            assignment = list(selected.assignment)
            delay = candidate_state.estimated_delay_ms
            self._register_running_dnn(dnn_index, assignment, delay)
            total_delay += delay
            total_stability += candidate_state.operational_stability
            total_energy += candidate_state.total_energy
            successes += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        return ExperimentMetrics(
            avg_delay=total_delay / successes if successes else 0.0,
            avg_operational_stability_score=(
                total_stability / successes if successes else 0.0
            ),
            avg_energy=total_energy / successes if successes else 0.0,
            failure_count=failures,
            runtime_ms=int((time.monotonic() - started) * 1000),
        )

    def _run_customized_proposed_legacy(self) -> ExperimentMetrics:
        """运行面向 DNN 拓扑与边缘网络状态定制的 RDODA-DSSA。"""
        self.pareto_points = []
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        self.res = self._new_population(self.pop_size)
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        run_time = time.monotonic()
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            self._log_proposed_stage(
                t,
                "customized_search_start",
                tasks=len(self.env.ds[t].tasks),
                deadline=self.env.ds[t].delay,
                iteration_limit=self.iteration_limit,
            )
            context = self._build_dnn_context(t)
            self._path_transfer_cost_cache = {}
            self._assignment_evaluation_cache = {}
            self._constraint_violation_cache = {}
            best_value = [0.0, 0.0, -1.0, 0.0]
            best_assignment = [0 for _ in range(self.max_dnn_num)]
            self.res = self._new_population(self.pop_size)
            self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
            self.estimated_delay_values = [math.inf for _ in range(2 * self.pop_size)]
            self.total_energy_values = [math.inf for _ in range(2 * self.pop_size)]
            self.deadline_feasible_values = [False for _ in range(2 * self.pop_size)]
            self.constraint_violations = [0.0 for _ in range(2 * self.pop_size)]
            w_a, w_r, w_t, w_e = self._resolve_preference_weights(
                t,
                min_a,
                max_a,
                min_r,
                max_r,
                min_t,
                max_t,
            )
            preference_weights = (w_a, w_r, w_t, w_e)
            search_failed = False
            index = 0
            best_gen = 0
            while index < self.iteration_limit:
                if index == 0 or index % 10 == 0:
                    self._log_proposed_stage(t, "customized_generation_start", generation=index)
                self.pareto_level_sort = [[0 for _ in range(2 * self.pop_size)] for _ in range(len(self.env.ds))]
                if index == 0:
                    self._log_proposed_stage(t, "customized_initialize_population", population=self.pop_size)
                    if not self._initialize_customized_population(t, context):
                        self._log_proposed_stage(t, "customized_initialize_population_failed")
                        search_failed = True
                        break

                self.res_pq = self._new_population(2 * self.pop_size)
                for m in range(len(self.res)):
                    for m1 in range(len(self.res[m])):
                        for m2 in range(len(self.res[m][m1])):
                            self.res_pq[m][m1][m2] = self.res[m][m1][m2]
                self._joint_population_roles = [
                    *self.population_roles,
                    *["balanced" for _ in range(self.pop_size)],
                ]
                for m in range(0, self.pop_size, 2):
                    role_index = (m // 2) % len(self.customized_membrane_roles)
                    role = self.customized_membrane_roles[role_index]
                    a1 = self._pick_parent_from_membrane(role_index)
                    if index > 0 and index % 5 == 0:
                        b1 = int(random.random() * self.pop_size)
                    else:
                        b1 = self._pick_parent_from_membrane(role_index)
                    self._write_customized_child(t, a1, b1, self.pop_size + m, context, role)
                    self._joint_population_roles[self.pop_size + m] = role
                    if self.pop_size + m + 1 < len(self._joint_population_roles):
                        self._joint_population_roles[self.pop_size + m + 1] = role

                for m in range(len(self.res_pq)):
                    delay = self._count_customized_values(t, m)
                    self.constraint_violations[m] = self._constraint_violation(
                        t,
                        self.res_pq[m][t],
                        context,
                        delay,
                    )
                bi = self._best_strict_feasible_index(
                    2 * self.pop_size,
                    preference_weights,
                )
                if bi is not None and (
                    best_value[2] < 0
                    or self._score(
                        best_value[0],
                        best_value[1],
                        best_value[2],
                        best_value[3],
                        *preference_weights,
                    )
                    < self._score(
                        self.function1_values[bi],
                        self.function2_values[bi],
                        self.function3_values[bi],
                        self.function4_values[bi],
                        *preference_weights,
                    )
                ):
                    for j in range(len(self.env.ds[t].tasks)):
                        best_assignment[j] = self.res_pq[bi][t][j]
                    best_value[0] = self.function1_values[bi]
                    best_value[1] = self.function2_values[bi]
                    best_value[2] = self.function3_values[bi]
                    best_value[3] = self.function4_values[bi]
                    best_gen = index
                    self._log_proposed_stage(
                        t,
                        "customized_generation_best_updated",
                        generation=index,
                        value1=f"{best_value[0]:.4f}",
                        value2=f"{best_value[1]:.4f}",
                        value3=f"{best_value[2]:.4f}",
                        value4=f"{best_value[3]:.6f}",
                    )

                epsilon = self._customized_epsilon(index)
                self.dominated_sort_with_cv(t, len(self.res_pq), epsilon)
                self.get_res(
                    t,
                    self.res_pq,
                    self.function1_values,
                    self.function2_values,
                    self.function3_values,
                )
                self.update_res_with_cv(
                    self.res_pq,
                    self.pareto_level_sort,
                    self.distance,
                    t,
                    epsilon,
                )
                if self.use_elite_local_search:
                    local_improvements = self._local_search_elites(
                        t,
                        context,
                        w_a,
                        w_r,
                        w_t,
                        w_e,
                    )
                else:
                    local_improvements = 0
                if local_improvements and (index == 0 or index % 10 == 0):
                    self._log_proposed_stage(
                        t,
                        "customized_local_search",
                        generation=index,
                        improvements=local_improvements,
                        epsilon=f"{epsilon:.4f}",
                    )
                index += 1

            if search_failed:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue

            self.res_pq = self._new_population(self.pop_size)
            self.res_pq = self.res
            for m in range(len(self.res_pq)):
                delay = self._count_customized_values(t, m)
                self.constraint_violations[m] = self._constraint_violation(
                    t,
                    self.res_pq[m][t],
                    context,
                    delay,
                )
            self._append_current_pareto_points("customized", t, len(self.res_pq))
            best_i = self._best_strict_feasible_index(
                self.pop_size,
                preference_weights,
            )
            if best_i is not None and (
                best_value[2] < 0
                or self._score(
                    best_value[0],
                    best_value[1],
                    best_value[2],
                    best_value[3],
                    *preference_weights,
                )
                < self._score(
                    self.function1_values[best_i],
                    self.function2_values[best_i],
                    self.function3_values[best_i],
                    self.function4_values[best_i],
                    *preference_weights,
                )
            ):
                for j in range(len(self.env.ds[t].tasks)):
                    best_assignment[j] = self.res_pq[best_i][t][j]
                best_value[0] = self.function1_values[best_i]
                best_value[1] = self.function2_values[best_i]
                best_value[2] = self.function3_values[best_i]
                best_value[3] = self.function4_values[best_i]
                best_gen = index
            self._log_proposed_stage(
                t,
                "customized_final_selection",
                best_generation=best_gen,
                value1=f"{best_value[0]:.4f}",
                value2=f"{best_value[1]:.4f}",
                value3=f"{best_value[2]:.4f}",
                value4=f"{best_value[3]:.6f}",
            )
            if best_value[2] < 0:
                self._log_proposed_stage(t, "customized_rejected", reason="no_strictly_feasible_solution")
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            selected_values = self._evaluate_customized_assignment(t, best_assignment)
            delay = selected_values.estimated_delay
            assignment_slice = self._assignment_slice(t, best_assignment)
            self._constraint_violation_cache.pop((t, tuple(assignment_slice)), None)
            final_cv = self._constraint_violation(t, assignment_slice, context, delay)
            if final_cv > 1e-12:
                self._log_proposed_stage(
                    t,
                    "customized_rejected",
                    reason="final_constraint_violation",
                    cv=f"{final_cv:.6g}",
                )
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            total_energy = selected_values.total_energy
            self._log_proposed_stage(
                t,
                "customized_accepted",
                delay=f"{delay:.2f}",
                energy=f"{total_energy:.2f}",
                best_generation=best_gen,
            )
            self._register_running_dnn(t, best_assignment, delay)
            t_res += delay
            r_res += selected_values.operational_stability
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operational_stability_score=r_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def _check_max_resource_assignment(self, r: List[int], dnn: DNN) -> bool:
        """检查 maxresource 完整部署向量的时延并缓存结果。"""
        assignment = tuple(r[: len(dnn.tasks)])
        delay = self._max_resource_delay_cache.get(assignment)
        if delay is None:
            delay = self.env.count_delay_by_assignment(dnn, list(assignment))
            self._max_resource_delay_cache[assignment] = delay
        if dnn.delay > delay:
            self._max_resource_last_delay = delay
            return True
        return False

    def max_resource_placement1(self, r: List[int], dnn: DNN, i: int, node_list: List[Node]) -> bool:
        """在不优先云节点时递归寻找高资源部署。"""
        if (time.monotonic() - self.start_time) * 1000 > 10000:
            return False
        if i == len(dnn.tasks):
            return self._check_max_resource_assignment(r, dnn)
        candidates = [idx for idx in range(len(node_list)) if self.env.nodes[idx].level != 3]
        candidates.sort(key=lambda idx: node_list[idx].cpu, reverse=True)
        for index in candidates:
            r[i] = index
            if not (node_list[index].cpu >= dnn.tasks[i].cpu_need):
                continue
            node_list[index].cpu -= dnn.tasks[i].cpu_need
            if (node_list[index].level != 1 or index == dnn.initiateNode) and self.max_resource_placement1(r, dnn, i + 1, node_list):
                return True
            node_list[index].cpu += dnn.tasks[i].cpu_need
            r[i] = 0
        return False

    def max_resource_placement(self, r: List[int], dnn: DNN, i: int, node_list: List[Node]) -> bool:
        """按剩余 CPU 从高到低递归构造部署。"""
        if (time.monotonic() - self.start_time) * 1000 > 10000:
            return False
        if i == len(dnn.tasks):
            return self._check_max_resource_assignment(r, dnn)
        candidates = list(range(len(node_list)))
        candidates.sort(key=lambda idx: node_list[idx].cpu, reverse=True)
        for index in candidates:
            r[i] = index
            if not (node_list[index].cpu >= dnn.tasks[i].cpu_need):
                continue
            node_list[index].cpu -= dnn.tasks[i].cpu_need
            if (node_list[index].level != 1 or index == dnn.initiateNode) and self.max_resource_placement(r, dnn, i + 1, node_list):
                return True
            node_list[index].cpu += dnn.tasks[i].cpu_need
            r[i] = 0
        return False

    def _max_resource_fast_score(self, remaining_cpu: List[float], assignment: List[int]) -> tuple[float, float, int]:
        """返回 fast beam 使用的资源优先评分。"""
        used_nodes = len({node_idx for node_idx in assignment if node_idx >= 0})
        return max(remaining_cpu), sum(remaining_cpu), -used_nodes

    def _max_resource_fast_beam(self, r: List[int], dnn: DNN, node_list: List[Node]) -> bool:
        """使用 Top-K beam search 快速寻找资源优先部署。"""
        node_count = len(node_list)
        task_count = len(dnn.tasks)
        initial_cpu = [node.cpu for node in node_list]
        beam: List[tuple[List[int], List[float]]] = [([], initial_cpu)]
        for task_idx, task in enumerate(dnn.tasks):
            if (time.monotonic() - self.start_time) * 1000 > self.max_resource_fast_budget_ms:
                return False
            next_beam: List[tuple[List[int], List[float]]] = []
            for assignment, remaining_cpu in beam:
                candidates = list(range(node_count))
                candidates.sort(key=lambda idx: remaining_cpu[idx], reverse=True)
                expanded = 0
                for node_idx in candidates:
                    if expanded >= self.max_resource_fast_top_k:
                        break
                    if remaining_cpu[node_idx] < task.cpu_need:
                        continue
                    if node_list[node_idx].level == 1 and node_idx != dnn.initiateNode:
                        continue
                    next_assignment = [*assignment, node_idx]
                    next_cpu = list(remaining_cpu)
                    next_cpu[node_idx] -= task.cpu_need
                    next_beam.append((next_assignment, next_cpu))
                    expanded += 1
            if not next_beam:
                return False
            next_beam.sort(
                key=lambda item: self._max_resource_fast_score(item[1], item[0]),
                reverse=True,
            )
            beam = next_beam[: self.max_resource_fast_beam_width]
        for assignment, _ in beam:
            for task_idx in range(task_count):
                r[task_idx] = assignment[task_idx]
            if self._check_max_resource_assignment(r, dnn):
                return True
        return False

    def _max_resource_placement_budget(
        self,
        r: List[int],
        dnn: DNN,
        i: int,
        node_list: List[Node],
    ) -> bool:
        """在 fast 总预算内按原始候选顺序执行 DFS 兜底。"""
        if (time.monotonic() - self.start_time) * 1000 > self.max_resource_fast_budget_ms:
            return False
        if i == len(dnn.tasks):
            return self._check_max_resource_assignment(r, dnn)
        candidates = list(range(len(node_list)))
        candidates.sort(key=lambda idx: node_list[idx].cpu, reverse=True)
        for index in candidates:
            r[i] = index
            if not (node_list[index].cpu >= dnn.tasks[i].cpu_need):
                continue
            if node_list[index].level == 1 and index != dnn.initiateNode:
                continue
            node_list[index].cpu -= dnn.tasks[i].cpu_need
            if self._max_resource_placement_budget(r, dnn, i + 1, node_list):
                return True
            node_list[index].cpu += dnn.tasks[i].cpu_need
            r[i] = 0
        return False

    def _location_band(self, current_node: Node, candidate_node: Node) -> int:
        """返回当前节点到候选节点的带宽近似值。"""
        band = 0
        if candidate_node is current_node:
            band = 10000
        for link_node in self.env.link_nodes:
            if link_node.s_node is current_node and link_node.e_node is candidate_node:
                band = link_node.band_width
        if candidate_node.level == 3:
            band = -1
        return band

    def location_placement1(self, r: List[int], dnn: DNN, i: int, node_list: List[Node], node: Node) -> bool:
        """在限制更强的局部候选集上递归寻找就近部署。"""
        if (time.monotonic() - self.start_time) * 1000 > 10000:
            return False
        if i == len(dnn.tasks):
            xs = [[0 for _ in range(len(node_list))] for _ in range(len(dnn.tasks))]
            for j in range(len(dnn.tasks)):
                xs[j][r[j]] = 1
            return self.env.check_delay_random(dnn, xs)
        candidates = [
            idx
            for idx in range(len(node_list))
            if self.env.nodes[idx].level != 1 and node_list[idx].cpu > dnn.tasks[i].cpu_need
        ]
        candidates.sort(key=lambda idx: self._location_band(node, node_list[idx]), reverse=True)
        for index in candidates:
            r[i] = index
            if not (node_list[index].cpu >= dnn.tasks[i].cpu_need):
                continue
            node_list[index].cpu -= dnn.tasks[i].cpu_need
            if (node_list[index].level != 1 or index == dnn.initiateNode) and self.location_placement(r, dnn, i + 1, node_list, node_list[index]):
                node_list[index].cpu += dnn.tasks[i].cpu_need
                return True
            node_list[index].cpu += dnn.tasks[i].cpu_need
            r[i] = 0
        return False

    def location_placement(self, r: List[int], dnn: DNN, i: int, node_list: List[Node], node: Node) -> bool:
        """按链路带宽优先的策略递归构造部署。"""
        if (time.monotonic() - self.start_time) * 1000 > 20000:
            return False
        if i == len(dnn.tasks):
            xs = [[0 for _ in range(len(node_list))] for _ in range(len(dnn.tasks))]
            for j in range(len(dnn.tasks)):
                xs[j][r[j]] = 1
            return self.env.check_delay_random(dnn, xs)
        candidates = [idx for idx in range(len(node_list)) if node_list[idx].cpu > dnn.tasks[i].cpu_need]
        candidates.sort(key=lambda idx: self._location_band(node, node_list[idx]), reverse=True)
        for index in candidates:
            r[i] = index
            x = [[0 for _ in range(len(node_list))] for _ in range(len(dnn.tasks))]
            for p in range(i + 1):
                x[p][r[p]] = 1
            if not (node_list[index].cpu >= dnn.tasks[i].cpu_need):
                continue
            node_list[index].cpu -= dnn.tasks[i].cpu_need
            if (node_list[index].level != 1 or index == dnn.initiateNode) and self.count_delay_random1(dnn, x, i) < dnn.delay and self.location_placement(r, dnn, i + 1, node_list, node_list[index]):
                node_list[index].cpu += dnn.tasks[i].cpu_need
                return True
            node_list[index].cpu += dnn.tasks[i].cpu_need
            r[i] = -1
        return False

    def count_delay_random1(self, dnn: DNN, x: List[List[int]], index1: int) -> float:
        """转调环境中的局部时延估计函数。"""
        return self.env.count_delay_random1(dnn, x, index1)

    def _random_valid_assignment(self, dnn_index: int, max_attempts: int = 500) -> List[int] | None:
        """随机生成一个满足资源和节点层级约束的部署向量。"""
        dnn = self.env.ds[dnn_index]
        task_count = len(dnn.tasks)
        candidate_nodes = [
            [
                node_idx
                for node_idx, node in enumerate(self.env.nodes)
                if node.level != 1 or node_idx == dnn.initiateNode
            ]
            for _ in range(task_count)
        ]
        for _ in range(max_attempts):
            assignment = [
                candidates[int(random.random() * len(candidates))]
                for candidates in candidate_nodes
            ]
            if self._is_assignment_valid(dnn_index, assignment):
                return assignment
        cloud_candidates = [idx for idx, node in enumerate(self.env.nodes) if node.level == 3]
        if cloud_candidates:
            assignment = [cloud_candidates[0] for _ in range(task_count)]
            if self._is_assignment_valid(dnn_index, assignment):
                return assignment
        return None

    def _sa_neighbor_assignment(self, dnn_index: int, assignment: List[int]) -> List[int]:
        """对一个部署向量执行单任务迁移邻域扰动。"""
        dnn = self.env.ds[dnn_index]
        task_count = len(dnn.tasks)
        for _ in range(100):
            candidate = list(assignment)
            task_idx = int(random.random() * task_count)
            node_idx = int(random.random() * len(self.env.nodes))
            candidate[task_idx] = node_idx
            if self._is_assignment_valid(dnn_index, candidate):
                return candidate
        return list(assignment)

    def _sa_candidate_score(
        self,
        dnn_index: int,
        assignment: List[int],
        weights: tuple[float, float, float, float],
    ) -> tuple[float, ObjectiveValues]:
        """返回 SA 接受准则使用的偏好评分和候选目标值。"""
        values = self._evaluate_assignment(dnn_index, assignment)
        if not values.deadline_feasible:
            return -math.inf, values
        w_a, w_r, w_t, w_e = weights
        score = self._score(*values.vector, 0.0, w_a, w_r, w_t, w_e)
        return score, values

    def _sample_random_pareto_archive(
        self,
        dnn_index: int,
        algorithm: str,
        sample_count: int,
        weights: tuple[float, float, float, float],
    ) -> tuple[
        List[Dict[str, float | int | str]],
        List[int] | None,
        ObjectiveValues | None,
    ]:
        """随机采样可行部署，返回非支配 archive 和偏好最优解。"""
        archive: List[Dict[str, float | int | str]] = []
        best_assignment: List[int] | None = None
        best_values: ObjectiveValues | None = None
        best_score = -math.inf
        point_index = 0
        for _ in range(sample_count):
            assignment = self._random_valid_assignment(dnn_index, max_attempts=50)
            if assignment is None:
                continue
            score, values = self._sa_candidate_score(dnn_index, assignment, weights)
            point = self._assignment_pareto_point(algorithm, dnn_index, assignment, point_index)
            if point is not None:
                archive.append(point)
                point_index += 1
            if score > best_score:
                best_score = score
                best_assignment = list(assignment)
                best_values = values
        return self._filter_pareto_points(archive), best_assignment, best_values

    def run_simulated_annealing(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行多目标模拟退火基线，并在 Pareto 实验中导出四目标候选前沿。"""
        self.pareto_points = []
        run_time = time.monotonic()
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            context = self._build_dnn_context(t)
            seed_assignments: List[List[int]] = []
            random_seed = self._random_valid_assignment(t)
            if random_seed is not None:
                seed_assignments.append(random_seed)
            for mode in ("cloud", "local", "resource", "stability", "topology", "random"):
                seed = self._heuristic_assignment(t, context, mode)
                if self._is_assignment_valid(t, seed):
                    seed_assignments.append(seed)
            if not seed_assignments:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            archive: List[Dict[str, float | int | str]] = []
            point_index = 0
            final_weights = self._resolve_preference_weights(t, min_a, max_a, min_r, max_r, min_t, max_t)
            best_assignment = list(seed_assignments[0])
            best_score = -math.inf
            best_values = self._evaluate_assignment(t, best_assignment)
            original_preference = self.preference
            weight_profiles = list(dict.fromkeys(self.sa_weight_profiles))
            if "adaptive" not in weight_profiles:
                weight_profiles.insert(0, "adaptive")
            steps = max(1, self.iteration_limit * max(1, self.sa_steps_multiplier))
            try:
                for profile in weight_profiles:
                    self.preference = profile
                    weights = self._resolve_preference_weights(t, min_a, max_a, min_r, max_r, min_t, max_t)
                    scored_seeds = []
                    for seed in seed_assignments:
                        seed_score, seed_values = self._sa_candidate_score(t, seed, weights)
                        final_seed_score, _ = self._sa_candidate_score(t, seed, final_weights)
                        scored_seeds.append((seed_score, seed, seed_values))
                        seed_point = self._assignment_pareto_point("sa", t, seed, point_index)
                        if seed_point is not None:
                            archive.append(seed_point)
                            point_index += 1
                        if final_seed_score > best_score:
                            best_assignment = list(seed)
                            best_score = final_seed_score
                            best_values = seed_values
                    current_score, current, _ = max(scored_seeds, key=lambda item: item[0])
                    current = list(current)

                    for step in range(steps):
                        progress = step / max(steps - 1, 1)
                        temperature = self.sa_initial_temperature * (
                            self.sa_final_temperature / self.sa_initial_temperature
                        ) ** progress
                        candidate = self._sa_neighbor_assignment(t, current)
                        candidate_score, candidate_values = self._sa_candidate_score(t, candidate, weights)
                        final_candidate_score, _ = self._sa_candidate_score(t, candidate, final_weights)
                        candidate_point = self._assignment_pareto_point("sa", t, candidate, point_index)
                        if candidate_point is not None:
                            archive.append(candidate_point)
                            point_index += 1
                        delta = candidate_score - current_score
                        if current_score == -math.inf:
                            accept = True
                        elif delta >= 0:
                            accept = True
                        else:
                            accept = random.random() < math.exp(delta / max(temperature, 1e-9))
                        if accept:
                            current = candidate
                            current_score = candidate_score
                        if final_candidate_score > best_score:
                            best_assignment = list(candidate)
                            best_score = final_candidate_score
                            best_values = candidate_values
            finally:
                self.preference = original_preference

            archive = self._filter_pareto_points(archive)
            self.pareto_points.extend(archive)
            if not best_values.deadline_feasible:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            delay = best_values.estimated_delay
            total_energy = best_values.total_energy
            self._register_running_dnn(t, best_assignment, delay)
            t_res += delay
            r_res += best_values.operational_stability
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        return ExperimentMetrics(
            avg_delay=t_res / success_dnn if success_dnn else 0.0,
            avg_operational_stability_score=r_res / success_dnn if success_dnn else 0.0,
            avg_energy=e_res / success_dnn if success_dnn else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_random(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行随机部署基线。"""
        self.pareto_points = []
        run_time = time.monotonic()
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            weights = self._resolve_preference_weights(t, min_a, max_a, min_r, max_r, min_t, max_t)
            sample_count = max(self.iteration_limit, self.pop_size, 1)
            archive, assignment, values = self._sample_random_pareto_archive(
                t,
                "random",
                sample_count,
                weights,
            )
            self.pareto_points.extend(archive)
            if assignment is None or values is None or not values.deadline_feasible:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            # 随机基线保留原来的随机放置逻辑，
            # 只把评价方式和执行语义切换到新的动态环境模型。
            f2 = values.operational_stability
            delay = values.estimated_delay
            total_energy = values.total_energy
            self._register_running_dnn(t, assignment, delay)
            t_res += delay
            r_res += f2
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        return ExperimentMetrics(
            avg_delay=t_res / success_dnn if success_dnn else 0.0,
            avg_operational_stability_score=r_res / success_dnn if success_dnn else 0.0,
            avg_energy=e_res / success_dnn if success_dnn else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_max_resource(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行按剩余资源优先的基线算法。"""
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            node_list = self.env.clone_nodes()
            r = [0 for _ in range(len(self.env.ds[t].tasks))]
            self._max_resource_delay_cache = {}
            self._max_resource_last_delay = None
            self.start_time = time.monotonic()
            if self.max_resource_placement(r, self.env.ds[t], 0, node_list):
                pass
            else:
                self.start_time = time.monotonic()
                if not self.max_resource_placement1(r, self.env.ds[t], 0, node_list):
                    failure_dnn += 1
                    next_dnn_index += 1
                    self.env.advance_time_slot()
                    continue
            values = self._evaluate_assignment(t, r, self._max_resource_last_delay)
            f2, delay = values.operational_stability, values.estimated_delay
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, r))
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operational_stability_score=r_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_max_resource_fast(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行有界快速搜索版最大剩余资源基线。"""
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            node_list = self.env.clone_nodes()
            r = [0 for _ in range(len(self.env.ds[t].tasks))]
            self._max_resource_delay_cache = {}
            self._max_resource_last_delay = None
            self.start_time = time.monotonic()
            if self._max_resource_fast_beam(r, self.env.ds[t], node_list):
                pass
            else:
                r = [0 for _ in range(len(self.env.ds[t].tasks))]
                node_list = self.env.clone_nodes()
                if not self._max_resource_placement_budget(r, self.env.ds[t], 0, node_list):
                    failure_dnn += 1
                    next_dnn_index += 1
                    self.env.advance_time_slot()
                    continue
            values = self._evaluate_assignment(t, r, self._max_resource_last_delay)
            f2, delay = values.operational_stability, values.estimated_delay
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, r))
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operational_stability_score=r_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_local_first(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行按本地链路优先的基线算法。"""
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            t = next_dnn_index
            self.env.print_pending_dnn_info(t)
            r = [0 for _ in range(len(self.env.ds[t].tasks))]
            self.start_time = time.monotonic()
            if self.location_placement(r, self.env.ds[t], 0, self.env.nodes, self.env.nodes[self.env.ds[t].initiateNode]):
                pass
            else:
                self.start_time = time.monotonic()
                if not self.location_placement1(r, self.env.ds[t], 0, self.env.nodes, self.env.nodes[self.env.ds[t].initiateNode]):
                    failure_dnn += 1
                    next_dnn_index += 1
                    self.env.advance_time_slot()
                    continue
            values = self._evaluate_assignment(t, r)
            f2, delay = values.operational_stability, values.estimated_delay
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, r))
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operational_stability_score=r_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )
