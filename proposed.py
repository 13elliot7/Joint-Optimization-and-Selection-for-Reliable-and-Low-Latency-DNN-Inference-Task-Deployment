from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set

from core.environment import Environment
from metrics import ExperimentMetrics
from models import DNN, LinkNode, Node


@dataclass
class DNNContext:
    """保存定制化进化算子需要复用的 DNN 拓扑信息。"""

    topological_order: List[int]
    predecessors: Dict[int, List[int]]
    successors: Dict[int, List[int]]
    edge_data: Dict[tuple[int, int], float]
    critical_path_tasks: Set[int]
    candidate_nodes: Dict[int, List[int]]


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
        self.gen = 1000
        self.iteration_limit = 120
        self.max_dnn_num = 20
        self.mutate_pm = 0.1
        self.cross_over_pm = 0.5
        self.customized_membrane_roles = ["reliability", "latency", "energy", "feasibility"]
        self.cv_initial_epsilon = 0.35
        self.local_search_elite_ratio = 0.05
        self.local_search_top_k = 3
        self.start_time = 0.0
        self.res: List[List[List[int]]] = []
        self.res_next: List[List[List[int]]] = []
        self.res_pq: List[List[List[int]]] = []
        self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
        self.constraint_violations = [0.0 for _ in range(2 * self.pop_size)]
        self.pareto_level_sort: List[List[int]] = []
        self.distance = [[0 for _ in range(2 * self.pop_size)] for _ in range(1000)]
        self._path_transfer_cost_cache: Dict[tuple[int, int, float], float] = {}
        self._assignment_evaluation_cache: Dict[
            tuple[int, tuple[int, ...]],
            tuple[float, float, float, float, float],
        ] = {}
        self._constraint_violation_cache: Dict[tuple[int, tuple[int, ...]], float] = {}

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

    def _evaluate_assignment(self, dnn_index: int, assignment: List[int]) -> tuple[float, float, float, float, float]:
        """计算一个候选部署的三目标值、能耗收益与估计时延。"""
        # 所有算法统一复用这套动态评价：当前节点/链路可靠性
        # 加上基于原始关键路径估计得到的时延收益。
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        value1 = float(self.env.count_dynamic_accuracy_by_assignment(dnn_index, assignment_slice))
        value2 = float(self.env.count_dynamic_operation_by_assignment(dnn_index, assignment_slice))
        value4 = float(self.env.count_energy_utility_by_assignment(dnn_index, assignment_slice))
        delay = self.env.estimate_delay_from_assignment(dnn_index, assignment_slice)
        if delay > self.env.ds[dnn_index].delay:
            value3 = float(self.env.ds[dnn_index].delay - delay)
        else:
            value3 = float(1 / delay)
        return value1, value2, value3, value4, delay

    def _evaluate_customized_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> tuple[float, float, float, float, float]:
        """在当前 DNN 搜索期间缓存定制化算法的候选评价。"""
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        cache_key = (dnn_index, tuple(assignment_slice))
        cached = self._assignment_evaluation_cache.get(cache_key)
        if cached is not None:
            return cached
        values = self._evaluate_assignment(dnn_index, assignment_slice)
        self._assignment_evaluation_cache[cache_key] = values
        return values

    def _count_customized_values(self, dnn_index: int, individual_idx: int) -> float:
        """评价定制化联合种群个体并返回其估计时延。"""
        value1, value2, value3, value4, delay = self._evaluate_customized_assignment(
            dnn_index,
            self.res_pq[individual_idx][dnn_index],
        )
        self.function1_values[individual_idx] = value1
        self.function2_values[individual_idx] = value2
        self.function3_values[individual_idx] = value3
        self.function4_values[individual_idx] = value4
        return delay

    def _register_running_dnn(self, dnn_index: int, assignment: List[int], delay: float) -> None:
        """把已接纳的 DNN 注册成时隙级运行实体。"""
        # DNN 被接纳后不再是“部署即结束”，而是转成一个运行块，
        # 在若干个时隙内持续占用资源。
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        remaining_slots = max(1, math.ceil(delay / self.env.slot_length))
        self.env.add_running_dnn(
            dnn_index=dnn_index,
            assignment=assignment_slice,
            arrival_time=self.env.current_slot,
            start_time=self.env.current_slot,
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

    def _is_assignment_valid(self, dnn_index: int, assignment: List[int]) -> bool:
        """检查一维部署向量是否满足资源和节点层级约束。"""
        dnn = self.env.ds[dnn_index]
        task_count = len(dnn.tasks)
        xs = self._build_assignment_matrix(dnn_index, assignment[:task_count])
        if not self.env.check_resource(dnn, xs):
            return False
        for task_idx in range(task_count):
            node_idx = assignment[task_idx]
            if self.env.nodes[node_idx].level == 1 and node_idx != dnn.initiateNode:
                return False
        return True

    def _build_dnn_context(self, dnn_index: int) -> DNNContext:
        """构建定制化 RDODA 使用的 DAG 拓扑上下文。"""
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
                if node.cpu >= task.cpu_need and (node.level != 1 or node_idx == dnn.initiateNode)
            ]
            if not candidates:
                candidates = [
                    node_idx
                    for node_idx, node in enumerate(self.env.nodes)
                    if node.level != 1 or node_idx == dnn.initiateNode
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
        cost = self.env.path_transfer_delay(start_node_idx, end_node_idx, data_amount)
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
        cpu_score = node.cpu / node.max_cpu if node.max_cpu else 0.0
        exec_time = task.float_num / node.float_rate if node.float_rate else float("inf")
        exec_time_score = 1.0 / (1.0 + exec_time)
        reliability_score = (node.o_reliability + node.a_reliability) / 2.0
        heat_score = 1.0 / (1.0 + node.heat + node.load_ratio)
        energy = node.comp_power * exec_time if math.isfinite(exec_time) else float("inf")
        energy_score = 1.0 / (1.0 + energy)

        link_cost = 0.0
        path_reliability_score = 1.0
        path_feasibility_score = 1.0
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
                path_reliability, path_feasibility = self.env.predict_path_state(
                    pred_node_idx,
                    node_idx,
                    data_amount,
                    predicted_runtime,
                )
                path_reliability_score *= path_reliability
                path_feasibility_score *= path_feasibility
        for succ in context.successors[task_idx]:
            succ_node_idx = assignment[succ]
            if succ_node_idx >= 0:
                data_amount = context.edge_data.get((task_idx, succ), 0.0)
                link_cost += self._path_transfer_cost(
                    node_idx,
                    succ_node_idx,
                    data_amount,
                )
                path_reliability, path_feasibility = self.env.predict_path_state(
                    node_idx,
                    succ_node_idx,
                    data_amount,
                    predicted_runtime,
                )
                path_reliability_score *= path_reliability
                path_feasibility_score *= path_feasibility
        link_score = 1.0 / (1.0 + link_cost) if math.isfinite(link_cost) else 0.0
        predicted_node_load = node.load_ratio + task.cpu_need / max(float(node.max_cpu), 1.0)
        node_feasibility_score = 1.0 / (1.0 + max(0.0, predicted_node_load - 1.0))
        feasibility_score = path_feasibility_score * node_feasibility_score

        weights_by_role = {
            "balanced": (0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10),
            "reliability": (0.10, 0.10, 0.10, 0.20, 0.25, 0.05, 0.05, 0.15),
            "latency": (0.10, 0.25, 0.25, 0.10, 0.05, 0.05, 0.05, 0.15),
            "energy": (0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.30, 0.15),
            "feasibility": (0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.25),
        }
        w_cpu, w_time, w_link, w_node_rel, w_path_rel, w_heat, w_e, w_feasibility = weights_by_role.get(
            role,
            weights_by_role["balanced"],
        )
        return (
            w_cpu * cpu_score
            + w_time * exec_time_score
            + w_link * link_score
            + w_node_rel * reliability_score
            + w_path_rel * path_reliability_score
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

    def _repair_assignment(self, dnn_index: int, assignment: List[int], context: DNNContext) -> List[int]:
        """修复交叉/变异后违反层级或资源约束的部署方案。"""
        repaired = self._assignment_slice(dnn_index, assignment)
        dnn = self.env.ds[dnn_index]
        for task_idx in context.topological_order:
            if repaired[task_idx] < 0:
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

        if self._is_assignment_valid(dnn_index, repaired):
            return repaired

        # 资源仍冲突时，优先迁移非关键路径任务，降低关键路径扰动。
        task_order = sorted(
            range(len(dnn.tasks)),
            key=lambda idx: (idx in context.critical_path_tasks, -len(context.successors[idx])),
        )
        for task_idx in task_order:
            original = repaired[task_idx]
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
                if self._is_assignment_valid(dnn_index, repaired):
                    return repaired
            repaired[task_idx] = original
        return self._assignment_slice(dnn_index, assignment)

    def _weighted_choice(self, weighted_candidates: List[tuple[int, float]]) -> int:
        """按候选权重随机选择一个节点。"""
        total = sum(max(weight, 0.0) for _, weight in weighted_candidates)
        if total <= 0:
            return weighted_candidates[int(random.random() * len(weighted_candidates))][0]
        cursor = random.random() * total
        acc = 0.0
        for node_idx, weight in weighted_candidates:
            acc += max(weight, 0.0)
            if acc >= cursor:
                return node_idx
        return weighted_candidates[-1][0]

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
        task_pool = list(context.critical_path_tasks) or list(range(len(mutated)))
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
        ]
        if not weighted:
            return mutated
        mutated[task_idx] = self._weighted_choice(weighted)
        return self._repair_assignment(dnn_index, mutated, context)

    def _dag_block_crossover(
        self,
        dnn_index: int,
        parent_a: List[int],
        parent_b: List[int],
        context: DNNContext,
    ) -> tuple[List[int], List[int]]:
        """按 DAG 拓扑边界交换父代片段。"""
        task_count = len(self.env.ds[dnn_index].tasks)
        child_a = self._assignment_slice(dnn_index, parent_a)
        child_b = self._assignment_slice(dnn_index, parent_b)
        if task_count <= 1:
            return child_a, child_b
        boundary_pool = sorted(context.critical_path_tasks)
        boundary_pool = [idx for idx in boundary_pool if 0 < idx < task_count]
        if boundary_pool:
            boundary = boundary_pool[int(random.random() * len(boundary_pool))]
        else:
            boundary = 1 + int(random.random() * (task_count - 1))
        swapped_tasks = [idx for idx in context.topological_order if idx >= boundary]
        for task_idx in swapped_tasks:
            child_a[task_idx] = parent_b[task_idx]
            child_b[task_idx] = parent_a[task_idx]
        child_a = self._repair_assignment(dnn_index, child_a, context)
        child_b = self._repair_assignment(dnn_index, child_b, context)
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
        child_a, child_b = self._dag_block_crossover(
            dnn_index,
            self.res[parent_a_idx][dnn_index],
            self.res[parent_b_idx][dnn_index],
            context,
        )
        child_a = self._skew_mutate_assignment(dnn_index, child_a, context, role)
        child_b = self._skew_mutate_assignment(dnn_index, child_b, context, role)
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
            elif mode == "reliability":
                candidates = context.candidate_nodes[task_idx]
                assignment[task_idx] = max(
                    candidates,
                    key=lambda idx: self.env.nodes[idx].o_reliability * self.env.nodes[idx].a_reliability,
                )
            elif mode == "topology":
                assignment[task_idx] = self._best_candidate_node(dnn_index, task_idx, assignment, context)
            else:
                candidates = context.candidate_nodes[task_idx]
                assignment[task_idx] = candidates[int(random.random() * len(candidates))]
        return self._repair_assignment(dnn_index, assignment, context)

    def _initialize_customized_population(self, dnn_index: int, context: DNNContext) -> bool:
        """用拓扑和边缘网络启发式初始化改进算法种群。"""
        modes = ["cloud", "local", "resource", "reliability", "topology", "random"]
        for individual_idx in range(self.pop_size):
            mode = modes[individual_idx % len(modes)]
            assignment = self._heuristic_assignment(dnn_index, context, mode)
            if not self._is_assignment_valid(dnn_index, assignment):
                return False
            for task_idx, node_idx in enumerate(assignment):
                self.res[individual_idx][dnn_index][task_idx] = node_idx
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

    def _membrane_range(self, role_index: int) -> tuple[int, int]:
        """返回指定子膜在父代种群中的连续索引范围。"""
        role_count = len(self.customized_membrane_roles)
        start = role_index * self.pop_size // role_count
        end = (role_index + 1) * self.pop_size // role_count
        return start, max(start + 1, end)

    def _pick_parent_from_membrane(self, role_index: int) -> int:
        """从指定子膜中选择一个父代个体。"""
        start, end = self._membrane_range(role_index)
        return start + int(random.random() * (end - start))

    def _constraint_violation(
        self,
        dnn_index: int,
        assignment: List[int],
        context: DNNContext,
        delay: float | None = None,
    ) -> float:
        """计算时延、资源和层级约束的归一化违反度。"""
        dnn = self.env.ds[dnn_index]
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        cache_key = (dnn_index, tuple(assignment_slice))
        cached = self._constraint_violation_cache.get(cache_key)
        if cached is not None:
            return cached
        if delay is None:
            delay = self.env.estimate_delay_from_assignment(dnn_index, assignment_slice)
        deadline = max(float(dnn.delay), 1.0)
        delay_violation = max(0.0, delay - deadline) / deadline

        cpu_has = [float(node.cpu) for node in self.env.nodes]
        total_need = 0.0
        overload = 0.0
        for task_idx, node_idx in enumerate(assignment_slice):
            need = float(dnn.tasks[task_idx].cpu_need)
            total_need += need
            if node_idx < 0 or node_idx >= len(self.env.nodes):
                overload += need
                continue
            cpu_has[node_idx] -= need
        for remaining in cpu_has:
            if remaining < 0:
                overload += -remaining
        resource_violation = overload / max(total_need, 1.0)

        hierarchy_bad = 0
        hierarchy_total = 0
        for task_idx in context.topological_order:
            node_idx = assignment_slice[task_idx]
            if node_idx < 0 or node_idx >= len(self.env.nodes):
                hierarchy_bad += 1
                hierarchy_total += 1
                continue
            if self.env.nodes[node_idx].level == 1 and node_idx != dnn.initiateNode:
                hierarchy_bad += 1
                hierarchy_total += 1
            for pred in context.predecessors[task_idx]:
                hierarchy_total += 1
                pred_node_idx = assignment_slice[pred]
                if pred_node_idx < 0 or pred_node_idx >= len(self.env.nodes):
                    hierarchy_bad += 1
                    continue
                if self.env.nodes[node_idx].level < self.env.nodes[pred_node_idx].level:
                    hierarchy_bad += 1
        hierarchy_violation = hierarchy_bad / max(hierarchy_total, 1)
        predicted_link_loads = self.env.predict_link_load_ratios(dnn_index, assignment_slice, delay)
        link_overload_violation = sum(
            max(0.0, load_ratio - 1.0)
            for load_ratio in predicted_link_loads.values()
        ) / max(len(predicted_link_loads), 1)
        violation = delay_violation + resource_violation + hierarchy_violation + link_overload_violation
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
        return (
            self.function1_values[i] >= self.function1_values[j]
            and self.function2_values[i] >= self.function2_values[j]
            and self.function3_values[i] >= self.function3_values[j]
            and (
                self.function1_values[i] != self.function1_values[j]
                or self.function2_values[i] != self.function2_values[j]
                or self.function3_values[i] != self.function3_values[j]
            )
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
    ) -> None:
        """根据 Pareto 层、CV 和拥挤距离更新 customized 下一代种群。"""
        selected = sorted(
            range(len(pq)),
            key=lambda idx: (
                rank[i][idx],
                self.constraint_violations[idx],
                -dis[i][idx],
            ),
        )[: self.pop_size]
        for target_idx, source_idx in enumerate(selected):
            for task_idx in range(len(pq[source_idx][i])):
                self.res[target_idx][i][task_idx] = pq[source_idx][i][task_idx]

    def _local_search_score(
        self,
        dnn_index: int,
        assignment: List[int],
        w_a: float,
        w_r: float,
        w_t: float,
        w_e: float,
        min_a: float,
        max_a: float,
        min_r: float,
        max_r: float,
        min_t: float,
        max_t: float,
        min_e: float,
        max_e: float,
    ) -> tuple[float, float, float, float, float, float]:
        """返回局部搜索使用的目标值、时延和综合得分。"""
        value1, value2, value3, value4, delay = self._evaluate_customized_assignment(dnn_index, assignment)
        if value3 <= 0:
            score = -math.inf
        else:
            score = self._score(
                value1,
                value2,
                value3,
                value4,
                w_a,
                w_r,
                w_t,
                w_e,
                min_a,
                max_a,
                min_r,
                max_r,
                min_t,
                max_t,
                min_e,
                max_e,
            )
        return value1, value2, value3, value4, delay, score

    def _local_search_elites(
        self,
        dnn_index: int,
        context: DNNContext,
        w_a: float,
        w_r: float,
        w_t: float,
        w_e: float,
        min_a: float,
        max_a: float,
        min_r: float,
        max_r: float,
        min_t: float,
        max_t: float,
        min_e: float,
        max_e: float,
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
                min_a,
                max_a,
                min_r,
                max_r,
                min_t,
                max_t,
                min_e,
                max_e,
            )
            cv = self._constraint_violation(dnn_index, assignment, context, values[4])
            scored.append((cv, -values[5], idx, values))
        improved = 0
        task_order = list(context.critical_path_tasks) or list(range(len(self.env.ds[dnn_index].tasks)))
        for _, _, individual_idx, base_values in sorted(scored)[:elite_count]:
            assignment = self._assignment_slice(dnn_index, self.res[individual_idx][dnn_index])
            best_assignment = list(assignment)
            best_values = base_values
            best_cv = self._constraint_violation(dnn_index, best_assignment, context, best_values[4])
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
                        min_a,
                        max_a,
                        min_r,
                        max_r,
                        min_t,
                        max_t,
                        min_e,
                        max_e,
                    )
                    candidate_cv = self._constraint_violation(dnn_index, candidate, context, candidate_values[4])
                    pareto_not_worse = (
                        candidate_values[0] >= best_values[0]
                        and candidate_values[1] >= best_values[1]
                        and candidate_values[2] >= best_values[2]
                    )
                    if (
                        candidate_cv <= best_cv
                        and pareto_not_worse
                        and candidate_values[5] > best_values[5]
                    ):
                        best_assignment = candidate
                        best_values = candidate_values
                        best_cv = candidate_cv
                        improved += 1
                        break
            for task_idx, node_idx in enumerate(best_assignment):
                self.res[individual_idx][dnn_index][task_idx] = node_idx
        return improved

    def mutate(self, i: int, m: int) -> None:
        """对指定个体执行一次随机变异。"""
        rate = random.random()
        dnn = self.env.ds[i]
        xs = self._build_assignment_matrix(i, self.res[m][i])
        if rate < self.mutate_pm:
            while True:
                x = int(random.random() * len(dnn.tasks))
                y = int(random.random() * len(self.env.nodes))
                xs[x][self.res[m][i][x]] = 0
                xs[x][y] = 1
                if self.env.check_resource(dnn, xs) and (self.env.nodes[y].level != 1 or y == dnn.initiateNode):
                    self.res[m][i][x] = y
                    break
                xs[x][self.res[m][i][x]] = 1
                xs[x][y] = 0

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


    def cross_over(self, i: int, m: int, n: int, next_index: int) -> None:
        """执行按单点交换的交叉操作，并生成两个独立子代。"""
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
            index_n = int(task_count * random.random())
            child_m = list(parent_m)
            child_n = list(parent_n)
            child_m[index_m] = parent_n[index_n]
            child_n[index_n] = parent_m[index_m]
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

    def count_values1(self, i: int, m: int) -> None:
        """计算第一个目标值并写回缓存。"""
        self.function1_values[m] = self._evaluate_assignment(i, self.res_pq[m][i])[0]

    def count_values2(self, i: int, m: int) -> None:
        """计算第二个目标值并写回缓存。"""
        self.function2_values[m] = self._evaluate_assignment(i, self.res_pq[m][i])[1]

    def count_values3(self, i: int, m: int) -> None:
        """计算第三个目标值并写回缓存。"""
        self.function3_values[m] = self._evaluate_assignment(i, self.res_pq[m][i])[2]

    def count_values(self, i: int, m: int) -> None:
        """一次性计算三个目标值、能耗收益并写回缓存。"""
        value1, value2, value3, value4, _ = self._evaluate_assignment(i, self.res_pq[m][i])
        self.function1_values[m] = value1
        self.function2_values[m] = value2
        self.function3_values[m] = value3
        self.function4_values[m] = value4

    def dominated_sort(self, o: int) -> None:
        """对当前种群执行非支配排序。"""
        rank = 1
        x = [[0 for _ in range(2 * self.pop_size)] for _ in range(2 * self.pop_size)]
        y = 0
        z = [0 for _ in range(2 * self.pop_size)]
        for i in range(len(self.function1_values)):
            for j in range(len(self.function1_values)):
                if (
                    self.function1_values[i] >= self.function1_values[j]
                    and self.function2_values[i] >= self.function2_values[j]
                    and self.function3_values[i] >= self.function3_values[j]
                    and (
                        self.function1_values[i] != self.function1_values[j]
                        or self.function2_values[i] != self.function2_values[j]
                        or self.function3_values[i] != self.function3_values[j]
                    )
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

    def get_res(self, i: int, pq: List[List[List[int]]], values1: List[float], values2: List[float], values3: List[float]) -> List[List[int]]:
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
                            if post == -1 or (self.function3_values[m] > 0 and self.function3_values[post] < 0):
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
        min_a: float,
        max_a: float,
        min_r: float,
        max_r: float,
        min_t: float,
        max_t: float,
        min_e: float,
        max_e: float,
    ) -> float:
        """对满足时延的候选解做归一化加权打分，并纳入能耗收益。"""
        return (
            w_a * _java_div(value1 - min_a, max_a - min_a)
            + w_r * _java_div(value2 - min_r, max_r - min_r)
            + w_t * _java_div(max_t - _java_div(1, value3), max_t - min_t)
            + w_e * _java_div(value4 - min_e, max_e - min_e)
        )

    def run_proposed(self) -> ExperimentMetrics:
        """运行主算法并输出平均指标。"""
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        max_e = 1.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        min_e = 0.0
        self.res = self._new_population(self.pop_size)
        self.res_next = self._new_population(self.pop_size)
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
            self.res_next = self._new_population(self.pop_size)
            self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
            self.constraint_violations = [0.0 for _ in range(2 * self.pop_size)]
            raw_w_t = _java_div(max_t - self.env.ds[t].delay, max_t - min_t)
            raw_w_a = _java_div(self.env.ds[t].preA - min_a, max_a - min_a)
            raw_w_r = _java_div(self.env.ds[t].preR - min_r, max_r - min_r)
            w_e = 0.2
            remaining_weight = 1.0 - w_e
            base_weight_sum = raw_w_t + raw_w_a + raw_w_r
            if base_weight_sum > 0:
                scale = remaining_weight / base_weight_sum
                w_t = raw_w_t * scale
                w_a = raw_w_a * scale
                w_r = raw_w_r * scale
            else:
                w_t = remaining_weight / 3
                w_a = remaining_weight / 3
                w_r = remaining_weight / 3
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
                    if self.function3_values[i] > 0 and self.function3_values[bi] < 0:
                        bi = i
                    elif self.function3_values[i] > 0:
                        if self.function3_values[bi] < 0 or self._score(self.function1_values[i], self.function2_values[i], self.function3_values[i], self.function4_values[i], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e) > self._score(self.function1_values[bi], self.function2_values[bi], self.function3_values[bi], self.function4_values[bi], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e):
                            bi = i
                if (
                    (best_value[2] < 0 and self.function3_values[bi] > 0)
                    or (
                        self.function3_values[bi] > 0
                        and self._score(best_value[0], best_value[1], best_value[2], best_value[3], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e)
                        < self._score(self.function1_values[bi], self.function2_values[bi], self.function3_values[bi], self.function4_values[bi], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e)
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
                self.get_res(t, self.res_pq, self.function1_values, self.function2_values, self.function3_values)
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
            best_i = 0
            for i in range(self.pop_size):
                if self.function3_values[i] > 0 and self.function3_values[best_i] < 0:
                    best_i = i
                elif self.function3_values[i] > 0:
                    if self.function3_values[best_i] < 0 or self._score(self.function1_values[i], self.function2_values[i], self.function3_values[i], self.function4_values[i], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e) > self._score(self.function1_values[best_i], self.function2_values[best_i], self.function3_values[best_i], self.function4_values[best_i], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e):
                        best_i = i
            if (
                (best_value[2] < 0 and self.function3_values[best_i] > 0)
                or (
                    self.function3_values[best_i] > 0
                    and self._score(best_value[0], best_value[1], best_value[2], best_value[3], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e)
                    < self._score(self.function1_values[best_i], self.function2_values[best_i], self.function3_values[best_i], self.function4_values[best_i], w_a, w_r, w_t, w_e, min_a, max_a, min_r, max_r, min_t, max_t, min_e, max_e)
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
            _, _, _, _, delay = self._evaluate_assignment(t, best_assignment)
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, best_assignment))
            self._log_proposed_stage(
                t,
                "accepted",
                delay=f"{delay:.2f}",
                energy=f"{total_energy:.2f}",
                best_generation=best_gen,
            )
            self._register_running_dnn(t, best_assignment, delay)
            t_res += delay
            r_res += best_value[1]
            a_res += best_value[0]
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_customized_proposed(self) -> ExperimentMetrics:
        """运行面向 DNN 拓扑与边缘网络状态定制的 RDODA-DSSA。"""
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        max_e = 1.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        min_e = 0.0
        self.res = self._new_population(self.pop_size)
        self.res_next = self._new_population(self.pop_size)
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
            self.res_next = self._new_population(self.pop_size)
            self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function4_values = [0.0 for _ in range(2 * self.pop_size)]
            raw_w_t = _java_div(max_t - self.env.ds[t].delay, max_t - min_t)
            raw_w_a = _java_div(self.env.ds[t].preA - min_a, max_a - min_a)
            raw_w_r = _java_div(self.env.ds[t].preR - min_r, max_r - min_r)
            w_e = 0.2
            remaining_weight = 1.0 - w_e
            base_weight_sum = raw_w_t + raw_w_a + raw_w_r
            if base_weight_sum > 0:
                scale = remaining_weight / base_weight_sum
                w_t = raw_w_t * scale
                w_a = raw_w_a * scale
                w_r = raw_w_r * scale
            else:
                w_t = remaining_weight / 3
                w_a = remaining_weight / 3
                w_r = remaining_weight / 3

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
                for m in range(0, self.pop_size, 2):
                    role_index = (m // 2) % len(self.customized_membrane_roles)
                    role = self.customized_membrane_roles[role_index]
                    a1 = self._pick_parent_from_membrane(role_index)
                    if index > 0 and index % 5 == 0:
                        b1 = int(random.random() * self.pop_size)
                    else:
                        b1 = self._pick_parent_from_membrane(role_index)
                    self._write_customized_child(t, a1, b1, self.pop_size + m, context, role)

                for m in range(len(self.res_pq)):
                    delay = self._count_customized_values(t, m)
                    self.constraint_violations[m] = self._constraint_violation(
                        t,
                        self.res_pq[m][t],
                        context,
                        delay,
                    )
                bi = 0
                for i in range(2 * self.pop_size):
                    if self.function3_values[i] > 0 and self.function3_values[bi] < 0:
                        bi = i
                    elif self.function3_values[i] > 0:
                        if self.function3_values[bi] < 0 or self._score(
                            self.function1_values[i],
                            self.function2_values[i],
                            self.function3_values[i],
                            self.function4_values[i],
                            w_a,
                            w_r,
                            w_t,
                            w_e,
                            min_a,
                            max_a,
                            min_r,
                            max_r,
                            min_t,
                            max_t,
                            min_e,
                            max_e,
                        ) > self._score(
                            self.function1_values[bi],
                            self.function2_values[bi],
                            self.function3_values[bi],
                            self.function4_values[bi],
                            w_a,
                            w_r,
                            w_t,
                            w_e,
                            min_a,
                            max_a,
                            min_r,
                            max_r,
                            min_t,
                            max_t,
                            min_e,
                            max_e,
                        ):
                            bi = i
                if (
                    (best_value[2] < 0 and self.function3_values[bi] > 0)
                    or (
                        self.function3_values[bi] > 0
                        and self._score(
                            best_value[0],
                            best_value[1],
                            best_value[2],
                            best_value[3],
                            w_a,
                            w_r,
                            w_t,
                            w_e,
                            min_a,
                            max_a,
                            min_r,
                            max_r,
                            min_t,
                            max_t,
                            min_e,
                            max_e,
                        )
                        < self._score(
                            self.function1_values[bi],
                            self.function2_values[bi],
                            self.function3_values[bi],
                            self.function4_values[bi],
                            w_a,
                            w_r,
                            w_t,
                            w_e,
                            min_a,
                            max_a,
                            min_r,
                            max_r,
                            min_t,
                            max_t,
                            min_e,
                            max_e,
                        )
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
                self.get_res(t, self.res_pq, self.function1_values, self.function2_values, self.function3_values)
                self.update_res_with_cv(self.res_pq, self.pareto_level_sort, self.distance, t)
                local_improvements = self._local_search_elites(
                    t,
                    context,
                    w_a,
                    w_r,
                    w_t,
                    w_e,
                    min_a,
                    max_a,
                    min_r,
                    max_r,
                    min_t,
                    max_t,
                    min_e,
                    max_e,
                )
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
                self._count_customized_values(t, m)
            best_i = 0
            for i in range(self.pop_size):
                if self.function3_values[i] > 0 and self.function3_values[best_i] < 0:
                    best_i = i
                elif self.function3_values[i] > 0:
                    if self.function3_values[best_i] < 0 or self._score(
                        self.function1_values[i],
                        self.function2_values[i],
                        self.function3_values[i],
                        self.function4_values[i],
                        w_a,
                        w_r,
                        w_t,
                        w_e,
                        min_a,
                        max_a,
                        min_r,
                        max_r,
                        min_t,
                        max_t,
                        min_e,
                        max_e,
                    ) > self._score(
                        self.function1_values[best_i],
                        self.function2_values[best_i],
                        self.function3_values[best_i],
                        self.function4_values[best_i],
                        w_a,
                        w_r,
                        w_t,
                        w_e,
                        min_a,
                        max_a,
                        min_r,
                        max_r,
                        min_t,
                        max_t,
                        min_e,
                        max_e,
                    ):
                        best_i = i
            if (
                (best_value[2] < 0 and self.function3_values[best_i] > 0)
                or (
                    self.function3_values[best_i] > 0
                    and self._score(
                        best_value[0],
                        best_value[1],
                        best_value[2],
                        best_value[3],
                        w_a,
                        w_r,
                        w_t,
                        w_e,
                        min_a,
                        max_a,
                        min_r,
                        max_r,
                        min_t,
                        max_t,
                        min_e,
                        max_e,
                    )
                    < self._score(
                        self.function1_values[best_i],
                        self.function2_values[best_i],
                        self.function3_values[best_i],
                        self.function4_values[best_i],
                        w_a,
                        w_r,
                        w_t,
                        w_e,
                        min_a,
                        max_a,
                        min_r,
                        max_r,
                        min_t,
                        max_t,
                        min_e,
                        max_e,
                    )
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
                self._log_proposed_stage(t, "customized_rejected", reason="deadline_or_score")
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            _, _, _, _, delay = self._evaluate_customized_assignment(t, best_assignment)
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, best_assignment))
            self._log_proposed_stage(
                t,
                "customized_accepted",
                delay=f"{delay:.2f}",
                energy=f"{total_energy:.2f}",
                best_generation=best_gen,
            )
            self._register_running_dnn(t, best_assignment, delay)
            t_res += delay
            r_res += best_value[1]
            a_res += best_value[0]
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def max_resource_placement1(self, r: List[int], dnn: DNN, i: int, node_list: List[Node]) -> bool:
        """在不优先云节点时递归寻找高资源部署。"""
        if (time.monotonic() - self.start_time) * 1000 > 10000:
            return False
        if i == len(dnn.tasks):
            xs = [[0 for _ in range(len(node_list))] for _ in range(len(dnn.tasks))]
            for j in range(len(dnn.tasks)):
                xs[j][r[j]] = 1
            return self.env.check_delay_random(dnn, xs) and self.env.check_resource(dnn, xs)
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
            xs = [[0 for _ in range(len(node_list))] for _ in range(len(dnn.tasks))]
            for j in range(len(dnn.tasks)):
                xs[j][r[j]] = 1
            return self.env.check_delay_random(dnn, xs) and self.env.check_resource(dnn, xs)
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

    def run_random(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行随机部署基线。"""
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
            assignment = [-1 for _ in range(self.max_dnn_num)]
            started = time.monotonic()
            o = 0
            while True:
                if (time.monotonic() - started) * 1000 > 20000:
                    o = 1
                    break
                xs = [[0 for _ in range(len(self.env.nodes))] for _ in range(len(self.env.ds[t].tasks))]
                p = 0
                while p < len(self.env.ds[t].tasks):
                    n = int(random.random() * len(self.env.nodes))
                    xs[p][n] = 1
                    if self.env.check_resource(self.env.ds[t], xs) and (self.env.nodes[n].level != 1 or n == self.env.ds[t].initiateNode):
                        p += 1
                    else:
                        xs[p][n] = 0
                if self.env.check_delay_random(self.env.ds[t], xs):
                    for p in range(len(self.env.ds[t].tasks)):
                        for n in range(len(self.env.nodes)):
                            if xs[p][n] == 1:
                                assignment[p] = n
                    break
            if o == 1:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            # 随机基线保留原来的随机放置逻辑，
            # 只把评价方式和执行语义切换到新的动态环境模型。
            f1, f2, f3, _, delay = self._evaluate_assignment(t, assignment)
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, assignment))
            self._register_running_dnn(t, assignment, delay)
            t_res += delay
            r_res += f2
            a_res += f1
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        return ExperimentMetrics(
            avg_delay=t_res / success_dnn if success_dnn else 0.0,
            avg_operation=r_res / success_dnn if success_dnn else 0.0,
            avg_accuracy=a_res / success_dnn if success_dnn else 0.0,
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
            f1, f2, f3, _, delay = self._evaluate_assignment(t, r)
            _ = f3
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, r))
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            a_res += f1
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
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
            f1, f2, f3, _, delay = self._evaluate_assignment(t, r)
            _ = f3
            total_energy = self.env.count_total_energy_by_assignment(t, self._assignment_slice(t, r))
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            a_res += f1
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            avg_energy=e_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )
