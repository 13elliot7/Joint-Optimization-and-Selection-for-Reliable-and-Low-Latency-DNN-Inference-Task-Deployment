from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.topology import TopologyConfig, create_dnns, create_nodes
from models import (
    CandidateQualityScores,
    DNN,
    LinkDNN,
    LinkNode,
    Node,
    PhysicalLinkState,
    PostAdmissionMetrics,
    RunningDNN,
    Task,
)


@dataclass
class Environment:
    t_max: int = 40
    max_dnn_num: int = 20
    availability_node_count: int = 11
    shape: float = 1.5
    scale: float = 100.0
    mean: float = 2.5
    sigma: float = 0.8
    h_off: int = 100
    slot_length: float = 100.0
    alpha_r: float = 0.8
    beta_r: float = 0.5
    alpha_a: float = 0.5
    beta_a: float = 0.3
    lambda_h: float = 0.7
    o_min: float = 0.5
    a_min: float = 0.5
    alpha_l: float = 0.7
    beta_l: float = 0.4
    lambda_g: float = 0.7
    l_min: float = 0.5
    bandwidth_heat_gamma: float = 0.5
    bandwidth_load_gamma: float = 0.3
    min_bandwidth_ratio: float = 0.2
    node_stability_weight: float = 0.5
    link_stability_weight: float = 0.5
    quality_log_epsilon: float = 1e-12
    objective_semantics_version: str = "stability_fidelity_v2_return_energy"
    topology_config: TopologyConfig | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        """初始化拓扑、DNN 请求和动态运行状态。"""
        self._validate_quality_configuration()
        self.nodes, self.link_nodes = create_nodes(self.topology_config)
        self._build_physical_link_index()
        self.ds = create_dnns(self.t_max, self.nodes)
        self._build_path_cache()
        self.rj_history = self._generate_rj_history()
        self.up = [True for _ in range(self.availability_node_count)]
        self.remaining_time = [self._sample_uptime() for _ in range(self.availability_node_count)]
        self.running_dnns: List[RunningDNN] = []
        self.current_slot = 0
        self.finished_total = 0
        self._initialize_dynamic_state()
        self._validate_component_quality_priors()
        self._print_initialization_summary()

    def _validate_quality_configuration(self) -> None:
        """拒绝无法形成有界对数域质量分数的配置。"""
        if not 0.0 < self.quality_log_epsilon < 1.0:
            raise ValueError("quality_log_epsilon must be in (0, 1)")
        for name, value in (
            ("o_min", self.o_min),
            ("a_min", self.a_min),
            ("l_min", self.l_min),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name, value in (
            ("alpha_r", self.alpha_r),
            ("beta_r", self.beta_r),
            ("alpha_a", self.alpha_a),
            ("beta_a", self.beta_a),
            ("alpha_l", self.alpha_l),
            ("beta_l", self.beta_l),
            ("node_stability_weight", self.node_stability_weight),
            ("link_stability_weight", self.link_stability_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.node_stability_weight + self.link_stability_weight <= 0.0:
            raise ValueError("at least one stability aggregation weight must be positive")
        for name, value in (("lambda_h", self.lambda_h), ("lambda_g", self.lambda_g)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def _validate_component_quality_priors(self) -> None:
        """校验节点和物理链路的经验质量先验。"""
        for node in self.nodes:
            if not self.o_min <= node.base_operational_stability <= 1.0:
                raise ValueError("node operational stability prior must be in [o_min, 1]")
            if not self.a_min <= node.base_inference_fidelity <= 1.0:
                raise ValueError("node inference fidelity prior must be in [a_min, 1]")
        for physical_link in self.physical_links:
            if not self.l_min <= physical_link.base_transmission_stability <= 1.0:
                raise ValueError("link transmission stability prior must be in [l_min, 1]")

    def reset_nodes(self, snapshot: List[Node]) -> None:
        """用快照恢复节点状态并清空动态运行上下文。"""
        for idx, node in enumerate(snapshot):
            self.nodes[idx].cpu = node.cpu
            self.nodes[idx].max_cpu = node.max_cpu
            self.nodes[idx].level = node.level
            self.nodes[idx].operational_stability = node.operational_stability
            self.nodes[idx].inference_fidelity = node.inference_fidelity
            self.nodes[idx].float_rate = node.float_rate
            self.nodes[idx].base_operational_stability = node.base_operational_stability
            self.nodes[idx].base_inference_fidelity = node.base_inference_fidelity
            self.nodes[idx].load_ratio = node.load_ratio
            self.nodes[idx].heat = node.heat
            self.nodes[idx].comp_power = node.comp_power
        self.running_dnns = []
        self.current_slot = 0
        self.finished_total = 0
        self._reset_link_state()

    def clone_nodes(self) -> List[Node]:
        """复制当前全部节点状态。"""
        return [node.clone() for node in self.nodes]

    def _build_physical_link_index(self) -> None:
        """建立物理链路及其有向传输边的索引。"""
        states_by_id: Dict[int, PhysicalLinkState] = {}
        links_by_id: Dict[int, List[LinkNode]] = {}
        for link in self.link_nodes:
            physical_id = link.physical_link_id
            assert link.physical_state is not None
            existing_state = states_by_id.get(physical_id)
            if existing_state is not None and existing_state is not link.physical_state:
                raise ValueError(f"duplicate physical link id: {physical_id}")
            states_by_id[physical_id] = link.physical_state
            links_by_id.setdefault(physical_id, []).append(link)
        self.physical_links = list(states_by_id.values())
        self._physical_link_by_id = states_by_id
        self._directed_links_by_physical_id = links_by_id

    def _build_path_cache(self) -> None:
        """预计算固定网络拓扑中所有节点对的最短路径。"""
        self._node_index_by_id = {id(node): idx for idx, node in enumerate(self.nodes)}
        self._link_by_node_pair = {
            (self._node_index_by_id[id(link.s_node)], self._node_index_by_id[id(link.e_node)]): link
            for link in self.link_nodes
        }
        adjacency: List[List[tuple[int, float]]] = [[] for _ in range(len(self.nodes))]
        for link in self.link_nodes:
            start = self._node_index_by_id[id(link.s_node)]
            end = self._node_index_by_id[id(link.e_node)]
            adjacency[start].append((end, 1.0 / max(float(link.band_width), 1.0)))
        self._path_links: Dict[Tuple[int, int], tuple[LinkNode, ...]] = {}
        self._path_delay_factors: Dict[Tuple[int, int], float] = {}
        self._path_energy_factors: Dict[Tuple[int, int], float] = {}
        for start in range(len(self.nodes)):
            parents = self._get_shortest_path_parents(start, adjacency)
            for end in range(len(self.nodes)):
                if start == end:
                    links: tuple[LinkNode, ...] = ()
                else:
                    path = self._restore_shortest_path(parents, end)
                    links = tuple(
                        self._link_by_node_pair[(path[pos], path[pos + 1])]
                        for pos in range(len(path) - 1)
                    )
                key = (start, end)
                self._path_links[key] = links
                self._path_delay_factors[key] = sum(
                    1.0 / max(float(link.band_width), 1.0)
                    for link in links
                )
                self._path_energy_factors[key] = sum(link.energy_per_mb for link in links)

    def _get_shortest_path_parents(
        self,
        start: int,
        adjacency: List[List[tuple[int, float]]],
    ) -> List[int]:
        """计算一个起点对应的最短路径树。"""
        parent = [-1 for _ in range(len(self.nodes))]
        distance = [float("inf") for _ in range(len(self.nodes))]
        distance[start] = 0.0
        heap = [(0.0, start)]
        while heap:
            current_distance, current = heapq.heappop(heap)
            if current_distance > distance[current]:
                continue
            for neighbor, weight in adjacency[current]:
                next_distance = current_distance + weight
                if next_distance < distance[neighbor]:
                    distance[neighbor] = next_distance
                    parent[neighbor] = current
                    heapq.heappush(heap, (next_distance, neighbor))
        return parent

    @staticmethod
    def _restore_shortest_path(parent: List[int], end: int) -> List[int]:
        """从最短路径树恢复指定终点的节点路径。"""
        path: List[int] = []
        current = end
        while current != -1:
            path.insert(0, current)
            current = parent[current]
        return path

    def check_resource(self, dnn: DNN, x: List[List[int]]) -> bool:
        """检查一个部署矩阵是否满足节点 CPU 约束。"""
        cpu_has = [node.cpu for node in self.nodes]
        b = 0
        for i in range(len(x)):
            for j in range(len(self.nodes)):
                if x[i][j] == 1:
                    cpu_has[j] -= dnn.tasks[i].cpu_need
                    if cpu_has[j] < 0:
                        b = 1
        return b == 0

    def _initialize_dynamic_state(self) -> None:
        """初始化节点和链路的动态可靠性状态。"""
        for node in self.nodes:
            if node.base_operational_stability is None:
                node.base_operational_stability = node.operational_stability
            if node.base_inference_fidelity is None:
                node.base_inference_fidelity = node.inference_fidelity
            node.load_ratio = 0.0
            node.heat = 0.0
            node.operational_stability = node.base_operational_stability
            node.inference_fidelity = node.base_inference_fidelity
        self._reset_link_state()

    def _print_initialization_summary(self) -> None:
        """打印环境初始化摘要。"""
        if not self.verbose:
            return
        cloud_count = sum(1 for node in self.nodes if node.level == 3)
        edge_count = sum(1 for node in self.nodes if node.level == 2)
        user_count = sum(1 for node in self.nodes if node.level == 1)
        print(
            "[INIT] "
            f"t_max={self.t_max} "
            f"nodes={len(self.nodes)}(cloud={cloud_count} edge={edge_count} user={user_count}) "
            f"directed_links={len(self.link_nodes)} "
            f"physical_links={len(self.physical_links)} "
            f"dnns={len(self.ds)}"
        )
        # if self.ds:
        #     first_dnn = self.ds[0]
        #     print(
        #         "[INIT] "
        #         f"first_dnn tasks={len(first_dnn.tasks)} "
        #         f"deadline={first_dnn.delay} "
        #         f"initiate={first_dnn.initiateNode}"
        #     )
        print(
            "[INIT] "
            f"slot_length={self.slot_length} "
            f"alpha_r={self.alpha_r} beta_r={self.beta_r} "
            f"alpha_l={self.alpha_l} beta_l={self.beta_l}"
        )

    def print_pending_dnn_info(self, dnn_index: int) -> None:
        """打印当前待接纳 DNN 的基础信息。"""
        if not self.verbose:
            return
        dnn = self.ds[dnn_index]
        print(
            "[PENDING] "
            f"slot={self.current_slot} "
            f"dnn={dnn_index} "
            f"tasks={len(dnn.tasks)} "
            f"deadline={dnn.delay} "
            f"initiate={dnn.initiateNode} "
            f"preference_fidelity={dnn.preference_fidelity:.3f} "
            f"preference_stability={dnn.preference_stability:.3f}"
        )

    def _reset_link_state(self) -> None:
        """重置链路的先验可靠性和动态状态。"""
        for link in self.link_nodes:
            link.base_band_width = link.band_width
            link.effective_band_width = float(link.band_width)
            link.load_ratio = 0.0
        for physical_link in self.physical_links:
            representative = self._directed_links_by_physical_id[
                physical_link.physical_link_id
            ][0]
            physical_link.base_transmission_stability = self._infer_link_base_stability(
                representative
            )
            physical_link.transmission_stability = physical_link.base_transmission_stability
            physical_link.load_ratio = 0.0
            physical_link.heat = 0.0

    def get_effective_bandwidth(self, link: LinkNode) -> float:
        """返回链路在当前负载和热度下的有效带宽。"""
        base = float(link.base_band_width or link.band_width)
        degradation = 1.0 + self.bandwidth_heat_gamma * link.heat + self.bandwidth_load_gamma * link.load_ratio
        effective = base / max(degradation, 1.0)
        return max(base * self.min_bandwidth_ratio, effective, 1.0)

    def refresh_effective_bandwidth(self) -> None:
        """根据当前链路负载和热度刷新有效带宽。"""
        for link in self.link_nodes:
            link.effective_band_width = self.get_effective_bandwidth(link)

    def _link_transfer_delay(self, link: LinkNode, data_amount: float) -> float:
        """使用当前有效带宽计算单条链路传输时延。"""
        return data_amount / max(self.get_effective_bandwidth(link), 1.0)

    def _infer_link_base_stability(self, link: LinkNode) -> float:
        """根据链路层级组合和带宽推断静态先验可靠性。"""
        start_level = link.s_node.level
        end_level = link.e_node.level
        if start_level == end_level == 1:
            return 0.99
        if 3 in (start_level, end_level) and 2 in (start_level, end_level):
            min_rel = 0.90
            max_rel = 0.97
        elif 2 in (start_level, end_level) and 1 in (start_level, end_level):
            min_rel = 0.96
            max_rel = 0.99
        elif start_level == end_level == 2:
            min_rel = 0.95
            max_rel = 0.99
        else:
            min_rel = 0.94
            max_rel = 0.98
        bandwidth = max(float(link.band_width), 1.0)
        score = bandwidth / (bandwidth + 20.0)
        return min_rel + (max_rel - min_rel) * score

    def estimate_delay_from_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """复用原始静态时延模型估计部署完成时间。"""
        return self.count_delay_by_assignment(self.ds[dnn_index], assignment)

    def collect_used_nodes(self, dnn_index: int, assignment: List[int]) -> List[int]:
        """提取部署方案实际使用的唯一节点集合。"""
        del dnn_index
        return sorted({node_idx for node_idx in assignment if node_idx >= 0})

    def path_transfer_delay(self, start_node_idx: int, end_node_idx: int, data_amount: float) -> float:
        """使用缓存路径和当前有效带宽返回两节点间传输时延。"""
        return sum(
            self._link_transfer_delay(link, data_amount)
            for link in self._path_links[(start_node_idx, end_node_idx)]
        )

    def predict_node_load_ratios(self, dnn_index: int, assignment: List[int]) -> Dict[int, float]:
        """预测接纳候选部署后相关节点的负载率。"""
        dnn = self.ds[dnn_index]
        added_cpu: Dict[int, float] = {}
        for task_idx, node_idx in enumerate(assignment):
            added_cpu[node_idx] = added_cpu.get(node_idx, 0.0) + dnn.tasks[task_idx].cpu_need
        return {
            node_idx: node.load_ratio + added_cpu.get(node_idx, 0.0) / max(float(node.max_cpu), 1.0)
            for node_idx, node in enumerate(self.nodes)
            if node.load_ratio > 0 or node_idx in added_cpu
        }

    def predict_link_load_ratios(
        self,
        dnn_index: int,
        assignment: List[int],
        estimated_runtime: float,
    ) -> Dict[LinkNode, float]:
        """预测接纳候选部署后相关链路的负载率。"""
        runtime = max(float(estimated_runtime), self.slot_length, 1.0)
        link_weights = self._collect_link_weights(dnn_index, assignment)
        return {
            link: link.load_ratio + data_amount / runtime / max(self.get_effective_bandwidth(link), 1.0)
            for link, data_amount in link_weights.items()
        }

    def _physical_load_ratios_from_directional(
        self,
        directional_loads: Dict[LinkNode, float] | None = None,
        physical_links: List[PhysicalLinkState] | None = None,
    ) -> Dict[PhysicalLinkState, float]:
        """按全双工口径取两个方向负载的最大值。"""
        selected = self.physical_links if physical_links is None else physical_links
        predicted = directional_loads or {}
        return {
            physical_link: max(
                (
                    predicted.get(link, link.load_ratio)
                    for link in self._directed_links_by_physical_id[
                        physical_link.physical_link_id
                    ]
                ),
                default=0.0,
            )
            for physical_link in selected
        }

    def predict_paths_state(
        self,
        transfers: List[tuple[int, int, float]],
        estimated_runtime: float,
    ) -> tuple[float, float]:
        """汇总候选流量后按唯一物理链路预测稳定性与可行性。"""
        runtime = max(float(estimated_runtime), self.slot_length, 1.0)
        overload = 0.0
        added_bandwidth: Dict[LinkNode, float] = {}
        for start_node_idx, end_node_idx, data_amount in transfers:
            if data_amount <= 0:
                continue
            for link in self._path_links[(start_node_idx, end_node_idx)]:
                added_bandwidth[link] = (
                    added_bandwidth.get(link, 0.0) + data_amount / runtime
                )
        directional_loads = {
            link: link.load_ratio
            + added / max(self.get_effective_bandwidth(link), 1.0)
            for link, added in added_bandwidth.items()
        }
        for predicted_load in directional_loads.values():
            overload += max(0.0, predicted_load - 1.0)
        physical_links = self._unique_physical_links(list(directional_loads))
        physical_loads = self._physical_load_ratios_from_directional(
            directional_loads,
            physical_links,
        )
        predicted_stabilities: List[float] = []
        for physical_link in physical_links:
            load = physical_loads[physical_link]
            heat = self.lambda_g * physical_link.heat + (1.0 - self.lambda_g) * load
            predicted_stability = physical_link.base_transmission_stability * math.exp(
                -self.alpha_l * load - self.beta_l * heat
            )
            predicted_stabilities.append(min(
                physical_link.base_transmission_stability,
                max(self.l_min, predicted_stability),
            ))
        stability = (
            self._geometric_mean(predicted_stabilities)
            if predicted_stabilities
            else 1.0
        )
        feasibility = 1.0 / (1.0 + overload)
        return stability, feasibility

    def _add_path_link_weights(self, weights: Dict[LinkNode, float], links: List[LinkNode], data_amount: float) -> None:
        """把一段路径上的流量累计到链路权重表。"""
        if data_amount <= 0:
            return
        for link in links:
            weights[link] = weights.get(link, 0.0) + data_amount

    def _collect_link_weights(self, dnn_index: int, assignment: List[int], include_return: bool = True) -> Dict[LinkNode, float]:
        """收集一个 DNN 部署对各链路造成的总数据量。"""
        dnn = self.ds[dnn_index]
        weights: Dict[LinkNode, float] = {}
        if not assignment:
            return weights
        first_node = self.nodes[assignment[0]]
        # 一个 DNN 对链路的占用分三部分统计：
        # 输入上传、DAG 依赖边上的中间传输、最终结果回传。
        self._add_path_link_weights(
            weights,
            self.get_arrive_link(self.nodes[dnn.initiateNode], first_node),
            dnn.startFloat,
        )
        for dnn_link in dnn.links:
            start_index = dnn.tasks.index(dnn_link.s_task)
            end_index = dnn.tasks.index(dnn_link.e_task)
            start_node = self.nodes[assignment[start_index]]
            end_node = self.nodes[assignment[end_index]]
            self._add_path_link_weights(
                weights,
                self.get_arrive_link(start_node, end_node),
                dnn_link.float_tran,
            )
        if include_return:
            last_node = self.nodes[assignment[len(dnn.tasks) - 1]]
            self._add_path_link_weights(
                weights,
                self.get_arrive_link(last_node, self.nodes[dnn.initiateNode]),
                dnn.backFloat,
            )
        return weights

    def collect_used_directed_links(self, dnn_index: int, assignment: List[int]) -> List[LinkNode]:
        """提取部署方案实际经过的唯一有向传输边集合。"""
        return list(self._collect_link_weights(dnn_index, assignment).keys())

    @staticmethod
    def _unique_physical_links(links: List[LinkNode] | tuple[LinkNode, ...]) -> List[PhysicalLinkState]:
        """按显式物理链路 ID 去重并保持首次出现顺序。"""
        result: List[PhysicalLinkState] = []
        seen: set[int] = set()
        for link in links:
            if link.physical_link_id in seen:
                continue
            seen.add(link.physical_link_id)
            assert link.physical_state is not None
            result.append(link.physical_state)
        return result

    def collect_used_physical_links(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> List[PhysicalLinkState]:
        """提取部署方案使用的唯一物理链路故障域集合。"""
        return self._unique_physical_links(
            self.collect_used_directed_links(dnn_index, assignment)
        )

    def count_compute_energy_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """统计部署方案在节点计算侧的总能耗。"""
        dnn = self.ds[dnn_index]
        total_energy = 0.0
        for task_idx, node_idx in enumerate(assignment):
            task = dnn.tasks[task_idx]
            node = self.nodes[node_idx]
            exec_time = task.float_num / node.float_rate if node.float_rate else 0.0
            total_energy += node.comp_power * exec_time
        return total_energy

    def count_link_energy_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """统计输入上传、DAG 中间传输和最终结果回传的链路能耗。"""
        total_energy = 0.0
        for link, data_amount in self._collect_link_weights(
            dnn_index,
            assignment,
            include_return=True,
        ).items():
            total_energy += link.energy_per_mb * data_amount
        return total_energy

    def count_total_energy_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """统计部署方案的总能耗。"""
        return self.count_compute_energy_by_assignment(dnn_index, assignment) + self.count_link_energy_by_assignment(
            dnn_index, assignment
        )

    def energy_reference_bounds(self, dnn_index: int) -> tuple[float, float]:
        """返回与算法无关的确定性能耗参考边界，用于偏好评分和 Pareto 归一化。"""
        dnn = self.ds[dnn_index]
        legal_nodes = [
            (idx, node)
            for idx, node in enumerate(self.nodes)
            if node.level != 1 or idx == dnn.initiateNode
        ]
        if not legal_nodes:
            return 0.0, 1.0

        compute_lower = 0.0
        compute_upper = 0.0
        for task in dnn.tasks:
            costs = [
                node.comp_power * task.float_num / max(float(node.float_rate), 1e-12)
                for _, node in legal_nodes
            ]
            compute_lower += min(costs)
            compute_upper += max(costs)

        path_energy_factors = [
            self._path_energy_factors[(start_idx, end_idx)]
            for start_idx, _ in legal_nodes
            for end_idx, _ in legal_nodes
        ]
        max_path_energy = max(path_energy_factors, default=0.0)
        input_factors = [
            self._path_energy_factors[(dnn.initiateNode, node_idx)]
            for node_idx, _ in legal_nodes
        ]
        transfer_upper = dnn.startFloat * max(input_factors, default=0.0)
        transfer_upper += sum(link.float_tran for link in dnn.links) * max_path_energy
        return_factors = [
            self._path_energy_factors[(node_idx, dnn.initiateNode)]
            for node_idx, _ in legal_nodes
        ]
        transfer_upper += dnn.backFloat * max(return_factors, default=0.0)

        lower = max(0.0, compute_lower)
        upper = max(lower + 1e-9, compute_upper + transfer_upper)
        return lower, upper

    def energy_satisfaction(self, dnn_index: int, total_energy: float) -> float:
        """把原始总能耗映射到统一的越大越好区间。"""
        lower, upper = self.energy_reference_bounds(dnn_index)
        if upper <= lower:
            return 1.0 if total_energy <= lower else 0.0
        return max(0.0, min(1.0, (upper - total_energy) / (upper - lower)))

    def _geometric_mean(self, values: List[float]) -> float:
        """在对数域计算几何平均，避免大规模 DAG 下数值下溢。"""
        if not values:
            raise ValueError("geometric mean requires at least one value")
        log_sum = sum(math.log(max(value, self.quality_log_epsilon)) for value in values)
        return math.exp(log_sum / len(values))

    def _aggregate_operational_stability(
        self,
        node_stability: float,
        link_stability: float | None,
    ) -> float:
        """按节点域和物理链路域分层聚合运行稳定性。"""
        if link_stability is None:
            return node_stability
        weight_sum = self.node_stability_weight + self.link_stability_weight
        node_weight = self.node_stability_weight / weight_sum
        link_weight = self.link_stability_weight / weight_sum
        return math.exp(
            node_weight * math.log(max(node_stability, self.quality_log_epsilon))
            + link_weight * math.log(max(link_stability, self.quality_log_epsilon))
        )

    def predict_candidate_scores_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
        estimated_runtime: float | None = None,
    ) -> CandidateQualityScores:
        """预测候选接纳后的 OSS/IFS，不修改环境真实状态。"""
        dnn = self.ds[dnn_index]
        if len(assignment) != len(dnn.tasks):
            raise ValueError("assignment length must equal the DNN task count")
        if any(node_idx < 0 or node_idx >= len(self.nodes) for node_idx in assignment):
            raise ValueError("assignment contains an invalid node index")
        if estimated_runtime is None:
            estimated_runtime = self.estimate_delay_from_assignment(dnn_index, assignment)
        predicted_node_loads = self.predict_node_load_ratios(dnn_index, assignment)
        predicted_link_loads = self.predict_link_load_ratios(dnn_index, assignment, estimated_runtime)
        used_nodes = self.collect_used_nodes(dnn_index, assignment)
        used_physical_links = self.collect_used_physical_links(dnn_index, assignment)
        predicted_physical_loads = self._physical_load_ratios_from_directional(
            predicted_link_loads,
            used_physical_links,
        )

        predicted_node_stability: Dict[int, float] = {}
        predicted_node_fidelity: Dict[int, float] = {}
        for node_idx in used_nodes:
            node = self.nodes[node_idx]
            load = predicted_node_loads.get(node_idx, node.load_ratio)
            heat = self.lambda_h * node.heat + (1.0 - self.lambda_h) * load
            base_o = node.base_operational_stability
            predicted_o = base_o * math.exp(-self.alpha_r * load - self.beta_r * heat)
            predicted_node_stability[node_idx] = min(base_o, max(self.o_min, predicted_o))

            base_a = node.base_inference_fidelity
            predicted_a = base_a * math.exp(-self.alpha_a * load - self.beta_a * heat)
            predicted_node_fidelity[node_idx] = min(base_a, max(self.a_min, predicted_a))

        predicted_link_stabilities: List[float] = []
        for physical_link in used_physical_links:
            load = predicted_physical_loads[physical_link]
            heat = self.lambda_g * physical_link.heat + (1.0 - self.lambda_g) * load
            base_stability = physical_link.base_transmission_stability
            predicted_stability = base_stability * math.exp(
                -self.alpha_l * load - self.beta_l * heat
            )
            predicted_link_stabilities.append(
                min(base_stability, max(self.l_min, predicted_stability))
            )

        node_values = [predicted_node_stability[node_idx] for node_idx in used_nodes]
        node_stability = self._geometric_mean(node_values)
        link_stability = (
            self._geometric_mean(predicted_link_stabilities)
            if predicted_link_stabilities
            else None
        )
        operational_stability = self._aggregate_operational_stability(
            node_stability,
            link_stability,
        )
        raw_joint_product = math.prod(node_values) * math.prod(predicted_link_stabilities)

        total_float = sum(task.float_num for task in dnn.tasks)
        if total_float <= 0:
            inference_fidelity = 0.0
        else:
            float_by_node: Dict[int, float] = {}
            for task_idx, node_idx in enumerate(assignment):
                float_by_node[node_idx] = (
                    float_by_node.get(node_idx, 0.0) + dnn.tasks[task_idx].float_num
                )
            weighted_log_fidelity = sum(
                (node_float / total_float)
                * math.log(
                    max(predicted_node_fidelity[node_idx], self.quality_log_epsilon)
                )
                for node_idx, node_float in float_by_node.items()
            )
            inference_fidelity = math.exp(weighted_log_fidelity)

        return CandidateQualityScores(
            operational_stability=operational_stability,
            inference_fidelity=inference_fidelity,
            node_stability=node_stability,
            link_stability=link_stability,
            raw_joint_product=raw_joint_product,
            used_node_count=len(used_nodes),
            used_physical_link_count=len(used_physical_links),
        )

    def add_running_dnn(
        self,
        dnn_index: int,
        assignment: List[int],
        estimated_runtime: float,
        remaining_slots: int,
    ) -> RunningDNN:
        """把一个已接纳 DNN 注册到运行队列。"""
        running_dnn = RunningDNN(
            dnn_index=dnn_index,
            assignment=list(assignment),
            estimated_runtime=estimated_runtime,
            remaining_slots=remaining_slots,
        )
        self.running_dnns.append(running_dnn)
        # 接纳成功后立即把新 DNN 纳入当前占用，
        # 但热度和动态可靠性仍然只在时隙推进时更新。
        self.allocate_running_resources()
        if self.verbose:
            print(
                "[ACCEPTED] "
                f"slot={self.current_slot} "
                f"dnn={dnn_index} "
                f"estimated_runtime={estimated_runtime:.2f} "
                f"required_slots={remaining_slots}"
            )
        return running_dnn

    def advance_time_slot(self) -> None:
        """推进一个离散时隙并刷新系统动态状态。"""
        # 一个时隙内先按当前占用累计热度和可靠性，
        # 再结算运行时长并释放本时隙结束时已完成的 DNN。
        slot_id = self.current_slot
        self.allocate_running_resources()
        self.update_node_heat()
        self.update_link_heat()
        self.refresh_effective_bandwidth()
        self.refresh_dynamic_stability()
        for running_dnn in self.running_dnns:
            if running_dnn.remaining_slots > 0:
                running_dnn.remaining_slots -= 1
        finished_count = sum(1 for running_dnn in self.running_dnns if running_dnn.remaining_slots <= 0)
        self.release_finished_dnns()
        self._print_slot_summary(slot_id, finished_count)
        self.current_slot += 1

    def release_finished_dnns(self) -> None:
        """移除已完成的运行块并释放其占用资源。"""
        active: List[RunningDNN] = []
        for running_dnn in self.running_dnns:
            if running_dnn.remaining_slots <= 0:
                self.free_running_resources(running_dnn)
                self.finished_total += 1
                continue
            active.append(running_dnn)
        self.running_dnns = active
        self.allocate_running_resources()

    def allocate_running_resources(self) -> None:
        """按运行队列统一回填节点和链路占用状态。"""
        self.update_node_loads()
        self.update_link_loads()

    def free_running_resources(self, running_dnn: RunningDNN) -> None:
        """释放一个完成运行块占用的节点 CPU。"""
        dnn = self.ds[running_dnn.dnn_index]
        for task_idx, node_idx in enumerate(running_dnn.assignment):
            self.nodes[node_idx].cpu = min(
                self.nodes[node_idx].max_cpu,
                self.nodes[node_idx].cpu + dnn.tasks[task_idx].cpu_need,
            )

    def update_node_loads(self) -> None:
        """根据运行队列重算各节点当前负载率。"""
        used_cpu = [0 for _ in range(len(self.nodes))]
        for running_dnn in self.running_dnns:
            dnn = self.ds[running_dnn.dnn_index]
            for task_idx, node_idx in enumerate(running_dnn.assignment):
                used_cpu[node_idx] += dnn.tasks[task_idx].cpu_need
        for idx, node in enumerate(self.nodes):
            node.cpu = max(node.max_cpu - used_cpu[idx], 0)
            node.load_ratio = used_cpu[idx] / node.max_cpu if node.max_cpu else 0.0

    def update_link_loads(self) -> None:
        """根据运行队列重算各链路当前负载率。"""
        used_bandwidth: Dict[int, float] = {id(link): 0.0 for link in self.link_nodes}
        for running_dnn in self.running_dnns:
            # 简化模型把链路数据量按估计运行时长均摊，
            # 而不是显式做逐包级别的传输仿真。
            runtime = max(running_dnn.estimated_runtime, self.slot_length, 1.0)
            link_weights = self._collect_link_weights(running_dnn.dnn_index, running_dnn.assignment)
            for link, data_amount in link_weights.items():
                used_bandwidth[id(link)] += data_amount / runtime
        for link in self.link_nodes:
            current_used = used_bandwidth.get(id(link), 0.0)
            bandwidth = max(self.get_effective_bandwidth(link), 1.0)
            link.load_ratio = current_used / bandwidth
        physical_loads = self._physical_load_ratios_from_directional()
        for physical_link, load_ratio in physical_loads.items():
            physical_link.load_ratio = load_ratio

    def update_node_heat(self) -> None:
        """使用指数移动平均更新节点历史压力代理量。"""
        for node in self.nodes:
            node.heat = self.lambda_h * node.heat + (1.0 - self.lambda_h) * node.load_ratio

    def update_link_heat(self) -> None:
        """使用物理链路负载的指数移动平均更新共享压力代理量。"""
        for physical_link in self.physical_links:
            physical_link.heat = (
                self.lambda_g * physical_link.heat
                + (1.0 - self.lambda_g) * physical_link.load_ratio
            )

    def refresh_dynamic_stability(self) -> None:
        """依据负载和历史压力刷新节点与链路的动态质量分数。"""
        # 质量分数由经验先验和当前负载/历史压力共同决定，
        # 同时做上下界裁剪，避免高于先验或低于下界。
        for node in self.nodes:
            base_o = node.base_operational_stability
            base_a = node.base_inference_fidelity
            next_o = base_o * math.exp(-self.alpha_r * node.load_ratio - self.beta_r * node.heat)
            next_a = base_a * math.exp(-self.alpha_a * node.load_ratio - self.beta_a * node.heat)
            node.operational_stability = min(base_o, max(self.o_min, next_o))
            node.inference_fidelity = min(base_a, max(self.a_min, next_a))
        for physical_link in self.physical_links:
            next_stability = physical_link.base_transmission_stability * math.exp(
                -self.alpha_l * physical_link.load_ratio
                - self.beta_l * physical_link.heat
            )
            physical_link.transmission_stability = min(
                physical_link.base_transmission_stability,
                max(self.l_min, next_stability),
            )

    def _print_slot_summary(self, slot_id: int, finished_count: int) -> None:
        """打印单个时隙的运行摘要。"""
        if not self.verbose:
            return
        node_loads = [node.load_ratio for node in self.nodes]
        link_loads = [link.load_ratio for link in self.link_nodes]
        node_stabilities = [node.operational_stability for node in self.nodes]
        link_stabilities = [
            physical_link.transmission_stability for physical_link in self.physical_links
        ]
        node_load_avg = sum(node_loads) / len(node_loads) if node_loads else 0.0
        node_load_max = max(node_loads) if node_loads else 0.0
        link_load_avg = sum(link_loads) / len(link_loads) if link_loads else 0.0
        link_load_max = max(link_loads) if link_loads else 0.0
        node_r_avg = sum(node_stabilities) / len(node_stabilities) if node_stabilities else 0.0
        link_r_avg = sum(link_stabilities) / len(link_stabilities) if link_stabilities else 0.0
        print(
            f"[SLOT {slot_id}] "
            f"running={len(self.running_dnns)} "
            f"finished={finished_count} "
            f"finished_total={self.finished_total} "
            f"node_load_avg={node_load_avg:.2f} "
            f"node_load_max={node_load_max:.2f} "
            f"link_load_avg={link_load_avg:.2f} "
            f"link_load_max={link_load_max:.2f} "
            f"node_r_avg={node_r_avg:.2f} "
            f"link_r_avg={link_r_avg:.2f}\n"
        )

    def count_raw_link_product_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> float:
        """兼容诊断接口：计算唯一物理链路稳定性原始连乘。"""
        raw_link_product = 1.0
        for physical_link in self.collect_used_physical_links(dnn_index, assignment):
            raw_link_product *= physical_link.transmission_stability
        return raw_link_product

    def count_link_stability_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> float | None:
        """计算部署使用的唯一物理链路稳定性几何平均。"""
        values = [
            physical_link.transmission_stability
            for physical_link in self.collect_used_physical_links(dnn_index, assignment)
        ]
        return self._geometric_mean(values) if values else None

    def count_raw_joint_product_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> float:
        """计算旧式节点与物理链路连乘，仅用于诊断。"""
        node_product = math.prod(
            self.nodes[node_idx].operational_stability
            for node_idx in self.collect_used_nodes(dnn_index, assignment)
        )
        return node_product * self.count_raw_link_product_by_assignment(
            dnn_index,
            assignment,
        )

    def count_operational_stability_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> float:
        """按唯一节点域和物理链路域计算当前 OSS。"""
        node_values = [
            self.nodes[node_idx].operational_stability
            for node_idx in self.collect_used_nodes(dnn_index, assignment)
        ]
        node_stability = self._geometric_mean(node_values)
        return self._aggregate_operational_stability(
            node_stability,
            self.count_link_stability_by_assignment(dnn_index, assignment),
        )

    def count_inference_fidelity_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> float:
        """按节点承载 FLOPs 权重计算当前 IFS。"""
        dnn = self.ds[dnn_index]
        total_float = sum(task.float_num for task in dnn.tasks)
        if total_float <= 0:
            return 0.0
        float_by_node: Dict[int, float] = {}
        for task_idx, node_idx in enumerate(assignment):
            float_by_node[node_idx] = float_by_node.get(node_idx, 0.0) + dnn.tasks[task_idx].float_num
        weighted_log_sum = sum(
            (node_float / total_float)
            * math.log(
                max(self.nodes[node_idx].inference_fidelity, self.quality_log_epsilon)
            )
            for node_idx, node_float in float_by_node.items()
        )
        return math.exp(weighted_log_sum)

    def evaluate_post_admission_metrics(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> PostAdmissionMetrics:
        """按候选接纳后的预测状态统一计算四类指标。"""
        delay = self.estimate_delay_from_assignment(dnn_index, assignment)
        scores = self.predict_candidate_scores_by_assignment(
            dnn_index,
            assignment,
            delay,
        )
        energy = self.count_total_energy_by_assignment(dnn_index, assignment)
        return PostAdmissionMetrics(
            delay=delay,
            operational_stability=scores.operational_stability,
            inference_fidelity=scores.inference_fidelity,
            total_energy=energy,
            raw_joint_product=scores.raw_joint_product,
        )

    def check_delay_random(self, dnn: DNN, x: List[List[int]]) -> bool:
        """检查二维部署矩阵是否满足时延约束。"""
        return dnn.delay > self.count_delay_random(dnn, x)

    def count_tran_random(self, i: int, dnn: DNN, task: Task, x: List[List[int]]) -> float:
        """递归计算二维部署下到达某任务的关键路径时延。"""
        index = -1
        for j, current_task in enumerate(dnn.tasks):
            if task is current_task:
                index = j
        tloc = 0.0
        for j in range(len(self.nodes)):
            if x[index][j] == 1:
                tloc = task.float_num / self.nodes[j].float_rate
                break
        if dnn.tasks[0] is task:
            return tloc
        tran = 0.0
        for link in dnn.links:
            if link.e_task is task:
                tran = max(
                    tran,
                    tloc
                    + self.count_tran_delay_random(dnn, i, link.s_task, task, link.float_tran, x)
                    + self.count_tran_random(i, dnn, link.s_task, x),
                )
        return tran

    def count_delay_random1(self, dnn: DNN, x: List[List[int]], index1: int) -> float:
        """估计到指定任务为止的部分时延。"""
        tran = 0.0
        tloc = 0.0
        for i, task in enumerate(dnn.tasks):
            node = None
            nx = 0
            for j in range(len(self.nodes)):
                if x[i][j] == 1:
                    node = self.nodes[j]
                    nx = j
            if i == 0:
                links = self.get_arrive_link(self.nodes[dnn.initiateNode], node)
                for link in links:
                    tran += self._link_transfer_delay(link, dnn.startFloat)
            if node is not None:
                tloc += task.float_num / node.float_rate
            next_tasks = self.get_next_dnn_tasks(dnn, task)
            if not next_tasks:
                continue
        tran += self.count_tran_random(self.ds.index(dnn), dnn, dnn.tasks[index1], x)
        tloc = 0.0
        for j in range(len(self.nodes)):
            if x[len(dnn.tasks) - 1][j] == 1:
                links = self.get_arrive_link(self.nodes[j], self.nodes[dnn.initiateNode])
                for link in links:
                    tloc += self._link_transfer_delay(link, dnn.backFloat)
        return tloc + tran

    def count_delay_random(self, dnn: DNN, x: List[List[int]]) -> float:
        """计算二维部署矩阵对应的端到端总时延。"""
        # 这里保留原始静态时延模型：
        # 输入上传 + DAG 关键路径 + 结果回传。
        # 动态层把它复用为运行时长代理。
        tran = 0.0
        tloc = 0.0
        for i, task in enumerate(dnn.tasks):
            node = None
            for j in range(len(self.nodes)):
                if x[i][j] == 1:
                    node = self.nodes[j]
            if i == 0:
                links = self.get_arrive_link(self.nodes[dnn.initiateNode], node)
                for link in links:
                    tran += self._link_transfer_delay(link, dnn.startFloat)
            if node is not None:
                tloc += task.float_num / node.float_rate
            next_tasks = self.get_next_dnn_tasks(dnn, task)
            if not next_tasks:
                continue
        tran += self.count_tran_random(self.ds.index(dnn), dnn, dnn.tasks[len(dnn.tasks) - 1], x)
        tloc = 0.0
        for j in range(len(self.nodes)):
            if x[len(dnn.tasks) - 1][j] == 1:
                links = self.get_arrive_link(self.nodes[j], self.nodes[dnn.initiateNode])
                for link in links:
                    tloc += self._link_transfer_delay(link, dnn.backFloat)
        return tloc + tran

    def count_delay_by_assignment(self, dnn: DNN, assignment: List[int]) -> float:
        """计算一维部署向量对应的端到端总时延。"""
        task_index_by_id = {id(task): idx for idx, task in enumerate(dnn.tasks)}
        tran = 0.0
        for i, task in enumerate(dnn.tasks):
            node = self.nodes[assignment[i]]
            if i == 0:
                links = self.get_arrive_link(self.nodes[dnn.initiateNode], node)
                for link in links:
                    tran += self._link_transfer_delay(link, dnn.startFloat)
            if not self.get_next_dnn_tasks(dnn, task):
                continue
        tran += self.count_tran_by_assignment(
            dnn,
            dnn.tasks[len(dnn.tasks) - 1],
            assignment,
            task_index_by_id,
        )
        tloc = 0.0
        last_node = self.nodes[assignment[len(dnn.tasks) - 1]]
        links = self.get_arrive_link(last_node, self.nodes[dnn.initiateNode])
        for link in links:
            tloc += self._link_transfer_delay(link, dnn.backFloat)
        return tloc + tran

    def count_tran_by_assignment(
        self,
        dnn: DNN,
        task: Task,
        assignment: List[int],
        task_index_by_id: Dict[int, int],
    ) -> float:
        """递归计算一维部署向量下到达某任务的关键路径时延。"""
        index = task_index_by_id[id(task)]
        tloc = task.float_num / self.nodes[assignment[index]].float_rate
        if dnn.tasks[0] is task:
            return tloc
        tran = 0.0
        for link in dnn.links:
            if link.e_task is task:
                tran = max(
                    tran,
                    tloc
                    + self.count_tran_delay_by_assignment(
                        dnn,
                        link.s_task,
                        task,
                        link.float_tran,
                        assignment,
                        task_index_by_id,
                    )
                    + self.count_tran_by_assignment(
                        dnn,
                        link.s_task,
                        assignment,
                        task_index_by_id,
                    ),
                )
        return tran

    def count_tran_delay_by_assignment(
        self,
        dnn: DNN,
        stask: Task,
        etask: Task,
        float_num: int,
        assignment: List[int],
        task_index_by_id: Dict[int, int],
    ) -> float:
        """计算一维部署向量下两任务间的数据传输时延。"""
        del dnn
        snode = self.nodes[assignment[task_index_by_id[id(stask)]]]
        enode = self.nodes[assignment[task_index_by_id[id(etask)]]]
        tran = 0.0
        for link in self.get_arrive_link(snode, enode):
            tran += self._link_transfer_delay(link, float_num)
        return tran

    def count_tran_delay_random(
        self,
        dnn: DNN,
        i: int,
        stask: Task,
        etask: Task,
        float_num: int,
        x: List[List[int]],
    ) -> float:
        """计算二维部署下两任务间的数据传输时延。"""
        m = -1
        n = -1
        for j, task in enumerate(dnn.tasks):
            if task is stask:
                m = j
            elif task is etask:
                n = j
        snode = None
        enode = None
        for j in range(len(self.nodes)):
            if x[m][j] == 1:
                snode = self.nodes[j]
            if x[n][j] == 1:
                enode = self.nodes[j]
        tran = 0.0
        for link in self.get_arrive_link(snode, enode):
            tran += self._link_transfer_delay(link, float_num)
        return tran

    def get_arrive_link(self, s: Node | None, e: Node | None) -> List[LinkNode]:
        """返回两个节点间最短路对应的链路序列。"""
        if s is e:
            return []
        if s is None or e is None:
            raise ValueError("Node assignment missing")
        snode = self._node_index_by_id[id(s)]
        enode = self._node_index_by_id[id(e)]
        return list(self._path_links[(snode, enode)])

    def get_next_dnn_tasks(self, dnn: DNN, s: Task) -> List[Task]:
        """返回指定任务的后继任务集合。"""
        return [link.e_task for link in dnn.links if link.s_task is s]

    def calculate_task_delay_costs(self, dnn_index: int, task_index: int, x: List[List[int]]) -> List[float]:
        """估计 RTBL 中单个任务部署到各节点的时延代价。"""
        dnn = self.ds[dnn_index]
        task = dnn.tasks[task_index]
        delay_costs = [0.0 for _ in range(len(self.nodes))]
        for node_index, target_node in enumerate(self.nodes):
            computation_delay = task.float_num / target_node.float_rate
            transmission_delay = 0.0
            if task_index == 0:
                links = self.get_arrive_link(self.nodes[dnn.initiateNode], target_node)
                for link in links:
                    transmission_delay += self._link_transfer_delay(link, dnn.startFloat)
            return_delay = 0.0
            if task_index == len(dnn.tasks) - 1:
                links = self.get_arrive_link(target_node, self.nodes[dnn.initiateNode])
                for link in links:
                    return_delay += self._link_transfer_delay(link, dnn.backFloat)
            max_inter_task_delay = 0.0
            for link in dnn.links:
                if link.e_task is task:
                    predecessor_task = link.s_task
                    predecessor_index = dnn.tasks.index(predecessor_task)
                    predecessor_node_index = 0
                    for i in range(len(self.nodes)):
                        if x[predecessor_index][i] == 1:
                            predecessor_node_index = i
                            break
                    predecessor_node = self.nodes[predecessor_node_index]
                    inter_links = self.get_arrive_link(predecessor_node, target_node)
                    for inter_link in inter_links:
                        inter_task_delay1 = self._link_transfer_delay(inter_link, link.float_tran)
                        if inter_task_delay1 > max_inter_task_delay:
                            max_inter_task_delay = inter_task_delay1
            delay_costs[node_index] = computation_delay + transmission_delay + return_delay + max_inter_task_delay
        return delay_costs

    def generate_availability(self) -> List[int]:
        """推进节点在线状态并返回当前可用性快照。"""
        availability = [0 for _ in range(self.availability_node_count)]
        for j in range(self.availability_node_count):
            if self.up[j]:
                if self.remaining_time[j] > 0:
                    self.remaining_time[j] -= 1
                else:
                    self.up[j] = False
                    self.remaining_time[j] = self._sample_downtime()
            else:
                if self.remaining_time[j] > 0:
                    self.remaining_time[j] -= 1
                else:
                    self.up[j] = True
                    self.remaining_time[j] = self._sample_uptime()
            availability[j] = 1 if self.up[j] else 0
        return availability

    def _generate_rj_history(self) -> List[List[int]]:
        """生成一段离线可用性历史供 RTBL 初始化。"""
        history = [[] for _ in range(self.availability_node_count)]
        up_virtual = [True for _ in range(self.availability_node_count)]
        rem_virtual = [self._sample_uptime() for _ in range(self.availability_node_count)]
        for _ in range(self.h_off):
            snapshot = [0 for _ in range(self.availability_node_count)]
            for j in range(self.availability_node_count):
                if up_virtual[j]:
                    if rem_virtual[j] > 0:
                        rem_virtual[j] -= 1
                    else:
                        up_virtual[j] = False
                        rem_virtual[j] = self._sample_downtime()
                else:
                    if rem_virtual[j] > 0:
                        rem_virtual[j] -= 1
                    else:
                        up_virtual[j] = True
                        rem_virtual[j] = self._sample_uptime()
                snapshot[j] = 1 if up_virtual[j] else 0
            for j in range(self.availability_node_count):
                history[j].append(snapshot[j])
        return history

    def _sample_uptime(self) -> int:
        """按 Weibull 分布采样在线时长。"""
        u = random.random()
        return int(self.scale * math.pow(-math.log(1 - u), 1.0 / self.shape))

    def _sample_downtime(self) -> int:
        """按对数正态分布采样离线时长。"""
        u = random.random()
        v = random.random()
        z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
        return int(math.exp(self.mean + self.sigma * z))
