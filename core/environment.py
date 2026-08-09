from __future__ import annotations

import heapq
import math
import random
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.topology import TopologyConfig, create_dnns, create_nodes
from models import (
    CandidateStateSnapshot,
    CandidateQualityScores,
    DNN,
    LinkDNN,
    LinkNode,
    Node,
    ObservationPurpose,
    ObservationSnapshot,
    PhysicalLinkState,
    PostAdmissionMetrics,
    RunningDNN,
    Task,
)
from semantics import (
    ACTIVE_OBJECTIVE_SEMANTICS_VERSION,
    LEGACY_OBJECTIVE_SEMANTICS_VERSION,
    PERIODIC_SEMANTIC_VERSIONS,
)


class StaleEnvironmentStateError(RuntimeError):
    """提交时环境版本已经不同于在线评价所用版本。"""


class InfeasibleAssignmentError(RuntimeError):
    """最终复验发现候选不满足当前硬约束。"""


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
    availability_history_limit: int = 1000
    availability_beta_success: float = 1.0
    availability_beta_failure: float = 1.0
    availability_seed: int | None = None
    slot_length: float = 100.0
    alpha_r: float = 0.8
    beta_r: float = 0.5
    lambda_h: float = 0.7
    o_min: float = 0.5
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
    candidate_fixed_point_max_iterations: int = 20
    candidate_fixed_point_relative_tolerance: float = 1e-4
    candidate_fixed_point_damping: float = 0.5
    objective_semantics_version: str = ACTIVE_OBJECTIVE_SEMANTICS_VERSION
    topology_config: TopologyConfig | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        """初始化拓扑、DNN 请求和动态运行状态。"""
        self._validate_quality_configuration()
        self.nodes, self.link_nodes = create_nodes(self.topology_config)
        if self.availability_node_count < 0:
            raise ValueError("availability_node_count must be non-negative")
        self.availability_node_count = min(self.availability_node_count, len(self.nodes))
        for index, node in enumerate(self.nodes):
            node.availability_mode = (
                "stochastic" if index < self.availability_node_count else "always_on"
            )
        self._availability_rng = (
            random.Random(self.availability_seed)
            if self.availability_seed is not None
            else random
        )
        self._build_physical_link_index()
        self.ds = create_dnns(self.t_max, self.nodes)
        self._build_path_cache()
        self.rj_history = self._generate_rj_history()
        self.up = [True for _ in self.nodes]
        self.remaining_time = [
            self._sample_uptime() if node.availability_mode == "stochastic" else 0
            for node in self.nodes
        ]
        self._initial_rj_history = [list(history) for history in self.rj_history]
        self._initial_up = list(self.up)
        self._initial_remaining_time = list(self.remaining_time)
        self._initial_availability_rng_state = (
            self._availability_rng.getstate()
            if self.availability_seed is not None
            else None
        )
        self.running_dnns: List[RunningDNN] = []
        self.current_slot = 0
        self.finished_total = 0
        self.topology_version = 0
        self.environment_state_version = 0
        self._snapshot_version = 0
        self._commit_lock = threading.Lock()
        self._initialize_dynamic_state()
        self._validate_component_quality_priors()
        self._print_initialization_summary()

    def _validate_quality_configuration(self) -> None:
        """拒绝无法形成有界对数域质量分数的配置。"""
        if not 0.0 < self.quality_log_epsilon < 1.0:
            raise ValueError("quality_log_epsilon must be in (0, 1)")
        if self.h_off <= 0 or self.availability_history_limit < self.h_off:
            raise ValueError("availability history limits must be positive and ordered")
        if self.availability_beta_success <= 0.0 or self.availability_beta_failure <= 0.0:
            raise ValueError("availability Beta prior parameters must be positive")
        for name, value in (
            ("o_min", self.o_min),
            ("l_min", self.l_min),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name, value in (
            ("alpha_r", self.alpha_r),
            ("beta_r", self.beta_r),
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
        if self.candidate_fixed_point_max_iterations <= 0:
            raise ValueError("candidate_fixed_point_max_iterations must be positive")
        if self.candidate_fixed_point_relative_tolerance <= 0.0:
            raise ValueError("candidate_fixed_point_relative_tolerance must be positive")
        if not 0.0 < self.candidate_fixed_point_damping <= 1.0:
            raise ValueError("candidate_fixed_point_damping must be in (0, 1]")

    def _validate_component_quality_priors(self) -> None:
        """校验节点和物理链路的经验质量先验。"""
        for node in self.nodes:
            if not self.o_min <= node.base_operational_stability <= 1.0:
                raise ValueError("node operational stability prior must be in [o_min, 1]")
        for physical_link in self.physical_links:
            if not self.l_min <= physical_link.base_transmission_stability <= 1.0:
                raise ValueError("link transmission stability prior must be in [l_min, 1]")

    @property
    def uses_periodic_semantics(self) -> bool:
        """仅在显式选择当前目标语义时启用时域可用性 OSS。"""
        return self.objective_semantics_version == PERIODIC_SEMANTIC_VERSIONS.objective

    def profile_id_for_dnn(self, dnn_index: int) -> str:
        """返回请求绑定的画像 ID；旧请求使用稳定的索引兼容 ID。"""
        profile_id = self.ds[dnn_index].profile_id
        return profile_id or f"legacy_dnn_{dnn_index}"

    def reset_nodes(self, snapshot: List[Node]) -> None:
        """用快照恢复节点状态并清空动态运行上下文。"""
        for idx, node in enumerate(snapshot):
            self.nodes[idx].cpu = node.cpu
            self.nodes[idx].max_cpu = node.max_cpu
            self.nodes[idx].level = node.level
            self.nodes[idx].operational_stability = node.operational_stability
            self.nodes[idx].float_rate = node.float_rate
            self.nodes[idx].base_operational_stability = node.base_operational_stability
            self.nodes[idx].load_ratio = node.load_ratio
            self.nodes[idx].heat = node.heat
            self.nodes[idx].comp_power = node.comp_power
            self.nodes[idx].availability_mode = node.availability_mode
        self.running_dnns = []
        self.current_slot = 0
        self.finished_total = 0
        self.environment_state_version = 0
        self._snapshot_version = 0
        self.rj_history = [list(history) for history in self._initial_rj_history]
        self.up = list(self._initial_up)
        self.remaining_time = list(self._initial_remaining_time)
        if self._initial_availability_rng_state is not None:
            self._availability_rng.setstate(self._initial_availability_rng_state)
        self._reset_link_state()

    def capture_observation_snapshot(
        self,
        purpose: ObservationPurpose = "planning",
    ) -> ObservationSnapshot:
        """捕获算法当前可见的不可变状态，不暴露未来请求或故障轨迹。"""
        if purpose not in {
            "planning",
            "publication_revalidation",
            "online_admission",
        }:
            raise ValueError(f"unsupported observation purpose: {purpose}")
        snapshot_version = self._snapshot_version
        self._snapshot_version += 1
        node_online = tuple(bool(value) for value in self.up)
        availability_history = tuple(
            tuple(bool(value) for value in self.rj_history[index])
            for index in range(len(self.nodes))
        )
        effective_bandwidths = tuple(
            self.get_effective_bandwidth(link)
            for link in self.link_nodes
        )
        offered_rates = self.aggregate_directional_offered_rates()
        directional_offered_rates = tuple(
            float(offered_rates.get(link, 0.0))
            for link in self.link_nodes
        )
        return ObservationSnapshot(
            snapshot_version=snapshot_version,
            environment_state_version=self.environment_state_version,
            topology_version=self.topology_version,
            slot=self.current_slot,
            purpose=purpose,
            node_online=node_online,
            node_availability_modes=tuple(
                node.availability_mode for node in self.nodes
            ),
            node_available_cpu=tuple(float(node.cpu) for node in self.nodes),
            node_loads=tuple(float(node.load_ratio) for node in self.nodes),
            node_heats=tuple(float(node.heat) for node in self.nodes),
            node_availability_history=availability_history,
            directional_offered_rates=directional_offered_rates,
            directional_link_loads=tuple(float(link.load_ratio) for link in self.link_nodes),
            physical_link_heats=tuple(float(link.heat) for link in self.physical_links),
            effective_bandwidths=effective_bandwidths,
        )

    def origin_group_for_node(self, node_index: int) -> int:
        """把请求源节点映射为稳定的接入边缘组。"""
        if not 0 <= node_index < len(self.nodes):
            raise ValueError("invalid node_index")
        if self.nodes[node_index].level == 2:
            return node_index
        adjacent_edges = sorted(
            {
                self._node_index_by_id[id(link.e_node)]
                for link in self.link_nodes
                if self._node_index_by_id[id(link.s_node)] == node_index
                and link.e_node.level == 2
            }
        )
        if adjacent_edges:
            return adjacent_edges[0]
        edge_nodes = [
            index for index, node in enumerate(self.nodes) if node.level == 2
        ]
        if not edge_nodes:
            raise ValueError("topology has no edge node for origin grouping")
        return min(
            edge_nodes,
            key=lambda edge_index: self._path_delay_factors[(node_index, edge_index)],
        )

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
            node.load_ratio = 0.0
            node.heat = 0.0
            node.operational_stability = node.base_operational_stability
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
            f"preference_stability={dnn.preference_stability:.3f} "
            f"preference_delay={dnn.preference_delay:.3f} "
            f"preference_energy={dnn.preference_energy:.3f}"
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

    def _effective_bandwidth_from_predicted_state(
        self,
        link: LinkNode,
        directional_load: float,
        physical_heat: float,
    ) -> float:
        """在不修改链路对象的情况下计算预测有效带宽。"""
        base = float(link.base_band_width or link.band_width)
        degradation = (
            1.0
            + self.bandwidth_heat_gamma * physical_heat
            + self.bandwidth_load_gamma * directional_load
        )
        effective = base / max(degradation, 1.0)
        return max(base * self.min_bandwidth_ratio, effective, 1.0)

    def estimate_delay_from_assignment_with_bandwidths(
        self,
        dnn_index: int,
        assignment: List[int],
        effective_bandwidths: Dict[LinkNode, float],
    ) -> float:
        """使用显式带宽映射计算输入、关键路径和结果回传时延。"""
        dnn = self.ds[dnn_index]
        if len(assignment) != len(dnn.tasks):
            raise ValueError("assignment length must equal the DNN task count")
        task_index_by_id = {id(task): index for index, task in enumerate(dnn.tasks)}
        incoming: Dict[int, List[tuple[int, float]]] = {
            index: [] for index in range(len(dnn.tasks))
        }
        for dnn_link in dnn.links:
            source = task_index_by_id[id(dnn_link.s_task)]
            target = task_index_by_id[id(dnn_link.e_task)]
            incoming[target].append((source, float(dnn_link.float_tran)))

        def path_delay(start: int, end: int, data_amount: float) -> float:
            return sum(
                data_amount / max(effective_bandwidths.get(link, 1.0), 1.0)
                for link in self._path_links[(start, end)]
            )

        memo: Dict[int, float] = {}

        def critical_delay(task_index: int) -> float:
            cached = memo.get(task_index)
            if cached is not None:
                return cached
            node_index = assignment[task_index]
            computation = (
                dnn.tasks[task_index].float_num
                / max(float(self.nodes[node_index].float_rate), 1e-12)
            )
            if task_index == 0:
                result = computation
            else:
                predecessors = incoming[task_index]
                result = max(
                    (
                        critical_delay(source)
                        + path_delay(assignment[source], node_index, data_amount)
                        + computation
                        for source, data_amount in predecessors
                    ),
                    default=0.0,
                )
            memo[task_index] = result
            return result

        first_node = assignment[0]
        last_node = assignment[-1]
        upload = path_delay(dnn.initiateNode, first_node, float(dnn.startFloat))
        critical = critical_delay(len(dnn.tasks) - 1)
        result_return = path_delay(last_node, dnn.initiateNode, float(dnn.backFloat))
        return upload + critical + result_return

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
        candidate_rates = self.candidate_directional_offered_rates(
            dnn_index,
            assignment,
            estimated_runtime,
        )
        offered_rates = self.aggregate_directional_offered_rates(candidate_rates)
        return self.directional_load_ratios_from_offered_rates(offered_rates)

    @staticmethod
    def directional_offered_rates_from_data(
        directional_data: Dict[LinkNode, float],
        active_duration_ms: float,
    ) -> Dict[LinkNode, float]:
        """把方向链路数据量统一换算为当前模型的 kbit/ms offered rate。"""
        duration = max(float(active_duration_ms), 1.0)
        return {
            link: data_amount / duration
            for link, data_amount in directional_data.items()
            if data_amount > 0.0
        }

    def candidate_directional_offered_rates(
        self,
        dnn_index: int,
        assignment: List[int],
        estimated_runtime: float,
    ) -> Dict[LinkNode, float]:
        """使用唯一数据量和时长口径计算候选方向 offered rate。"""
        runtime = max(float(estimated_runtime), self.slot_length, 1.0)
        return self.directional_offered_rates_from_data(
            self._collect_link_weights(dnn_index, assignment),
            runtime,
        )

    def aggregate_directional_offered_rates(
        self,
        additional_rates: Dict[LinkNode, float] | None = None,
    ) -> Dict[LinkNode, float]:
        """汇总全部运行任务及可选候选任务的方向 offered rate。"""
        offered_rates: Dict[LinkNode, float] = {}
        for running_dnn in self.running_dnns:
            runtime = max(running_dnn.estimated_runtime, self.slot_length, 1.0)
            running_rates = self.directional_offered_rates_from_data(
                self._collect_link_weights(
                    running_dnn.dnn_index,
                    running_dnn.assignment,
                ),
                runtime,
            )
            for link, rate in running_rates.items():
                offered_rates[link] = offered_rates.get(link, 0.0) + rate
        for link, rate in (additional_rates or {}).items():
            if rate > 0.0:
                offered_rates[link] = offered_rates.get(link, 0.0) + rate
        return offered_rates

    def directional_load_ratios_from_offered_rates(
        self,
        offered_rates: Dict[LinkNode, float],
    ) -> Dict[LinkNode, float]:
        """使用当前有效带宽把 offered rate 唯一映射为方向负载率。"""
        return {
            link: rate / max(self.get_effective_bandwidth(link), 1.0)
            for link, rate in offered_rates.items()
            if rate > 0.0
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
        overload = 0.0
        directional_data: Dict[LinkNode, float] = {}
        for start_node_idx, end_node_idx, data_amount in transfers:
            if data_amount <= 0:
                continue
            for link in self._path_links[(start_node_idx, end_node_idx)]:
                directional_data[link] = directional_data.get(link, 0.0) + data_amount
        added_rates = self.directional_offered_rates_from_data(
            directional_data,
            max(float(estimated_runtime), self.slot_length, 1.0),
        )
        offered_rates = self.aggregate_directional_offered_rates(added_rates)
        directional_loads = self.directional_load_ratios_from_offered_rates(offered_rates)
        for predicted_load in directional_loads.values():
            overload += max(0.0, predicted_load - 1.0)
        physical_links = self._unique_physical_links(list(added_rates))
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

    def service_exposure_slots(self, estimated_delay_ms: float) -> int:
        """固定服务时长 v1：统一资源占用、完成事件和故障暴露时域。"""
        return max(1, math.ceil(max(float(estimated_delay_ms), 0.0) / self.slot_length))

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

    def predict_node_continuous_availability(
        self,
        node_index: int,
        horizon_slots: int,
        observation_snapshot: ObservationSnapshot,
    ) -> tuple[float, float]:
        """用快照内历史窗口和 Beta 平滑估计连续在线概率及置信度。"""
        if not 0 <= node_index < len(self.nodes):
            raise ValueError("invalid node_index")
        if horizon_slots <= 0:
            raise ValueError("horizon_slots must be positive")
        if observation_snapshot.node_availability_modes[node_index] == "always_on":
            return 1.0, 1.0
        if not observation_snapshot.node_online[node_index]:
            return 0.0, 1.0

        history = observation_snapshot.node_availability_history[node_index]
        alpha = self.availability_beta_success
        beta = self.availability_beta_failure
        estimates: List[float] = []
        horizon_eligible = 0
        for window in range(1, horizon_slots + 1):
            eligible = 0
            success = 0
            for start in range(0, len(history) - window):
                if not history[start]:
                    continue
                eligible += 1
                if all(history[start + 1 : start + window + 1]):
                    success += 1
            estimates.append((success + alpha) / (eligible + alpha + beta))
            if window == horizon_slots:
                horizon_eligible = eligible
        # 有限窗口的分母不同会产生微小反常；取前缀最小值保证生存概率随时域不增。
        probability = min(estimates)
        confidence = horizon_eligible / (horizon_eligible + alpha + beta)
        return probability, confidence

    def aggregate_candidate_availability(
        self,
        dnn_index: int,
        assignment: List[int],
        horizon_slots: int,
        observation_snapshot: ObservationSnapshot,
    ) -> tuple[float, float]:
        """按任务 FLOPs 权重聚合部署节点在执行时域内的连续可用性。"""
        dnn = self.ds[dnn_index]
        float_by_node: Dict[int, float] = {}
        for task_index, node_index in enumerate(assignment):
            float_by_node[node_index] = (
                float_by_node.get(node_index, 0.0) + dnn.tasks[task_index].float_num
            )
        total_float = sum(float_by_node.values())
        if total_float <= 0.0:
            return 0.0, 0.0
        weighted_log = 0.0
        confidence = 0.0
        for node_index, node_float in float_by_node.items():
            weight = node_float / total_float
            probability, node_confidence = self.predict_node_continuous_availability(
                node_index,
                horizon_slots,
                observation_snapshot,
            )
            if probability <= 0.0:
                return 0.0, node_confidence
            weighted_log += weight * math.log(probability)
            confidence += weight * node_confidence
        # The normalized weights can sum to a value infinitesimally above one
        # (for example 1.0000000000000002).  Keep the probability contract exact
        # at this numerical boundary before constructing CandidateStateSnapshot.
        confidence = max(0.0, min(1.0, confidence))
        return math.exp(weighted_log), confidence

    def predict_candidate_scores_by_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
        estimated_runtime: float | None = None,
    ) -> CandidateQualityScores:
        """预测候选接纳后的 OSS，不修改环境真实状态。"""
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

        return self._quality_scores_from_predicted_loads(
            dnn_index,
            assignment,
            predicted_node_loads,
            predicted_physical_loads,
        )

    def _quality_scores_from_predicted_loads(
        self,
        dnn_index: int,
        assignment: List[int],
        predicted_node_loads: Dict[int, float],
        predicted_physical_loads: Dict[PhysicalLinkState, float],
        base_node_heats: Dict[int, float] | None = None,
        base_physical_heats: Dict[PhysicalLinkState, float] | None = None,
    ) -> CandidateQualityScores:
        """从统一候选负载状态计算当前版本的 OSS。"""
        used_nodes = self.collect_used_nodes(dnn_index, assignment)
        used_physical_links = self.collect_used_physical_links(dnn_index, assignment)

        predicted_node_stability: Dict[int, float] = {}
        for node_idx in used_nodes:
            node = self.nodes[node_idx]
            load = predicted_node_loads.get(node_idx, node.load_ratio)
            current_heat = (
                base_node_heats.get(node_idx, node.heat)
                if base_node_heats is not None
                else node.heat
            )
            heat = self.lambda_h * current_heat + (1.0 - self.lambda_h) * load
            base_o = node.base_operational_stability
            predicted_o = base_o * math.exp(-self.alpha_r * load - self.beta_r * heat)
            predicted_node_stability[node_idx] = min(base_o, max(self.o_min, predicted_o))


        predicted_link_stabilities: List[float] = []
        for physical_link in used_physical_links:
            load = predicted_physical_loads[physical_link]
            current_heat = (
                base_physical_heats.get(physical_link, physical_link.heat)
                if base_physical_heats is not None
                else physical_link.heat
            )
            heat = self.lambda_g * current_heat + (1.0 - self.lambda_g) * load
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

        return CandidateQualityScores(
            operational_stability=operational_stability,
            node_stability=node_stability,
            link_stability=link_stability,
            raw_joint_product=raw_joint_product,
            used_node_count=len(used_nodes),
            used_physical_link_count=len(used_physical_links),
        )

    def predict_candidate_state(
        self,
        dnn_index: int,
        assignment: List[int],
        observation_snapshot: ObservationSnapshot | None = None,
        initial_delay_hint: float | None = None,
    ) -> CandidateStateSnapshot:
        """基于一个显式观测快照无副作用地预测完整候选接纳后状态。"""
        dnn = self.ds[dnn_index]
        if len(assignment) != len(dnn.tasks):
            raise ValueError("assignment length must equal the DNN task count")
        if any(node_idx < 0 or node_idx >= len(self.nodes) for node_idx in assignment):
            raise ValueError("assignment contains an invalid node index")
        snapshot = observation_snapshot or self.capture_observation_snapshot(
            "online_admission"
        )
        if snapshot.topology_version != self.topology_version:
            raise ValueError("observation topology_version is stale")
        if len(snapshot.effective_bandwidths) != len(self.link_nodes):
            raise ValueError("observation link count does not match the topology")

        current_bandwidths = {
            link: float(snapshot.effective_bandwidths[index])
            for index, link in enumerate(self.link_nodes)
        }
        base_offered_rates = {
            link: float(snapshot.directional_offered_rates[index])
            for index, link in enumerate(self.link_nodes)
            if snapshot.directional_offered_rates[index] > 0.0
        }
        physical_heat_by_link = {
            physical_link: float(snapshot.physical_link_heats[index])
            for index, physical_link in enumerate(self.physical_links)
        }
        node_heat_by_index = {
            index: float(snapshot.node_heats[index])
            for index in range(len(self.nodes))
        }

        added_cpu = [0.0 for _ in self.nodes]
        total_cpu_need = 0.0
        for task_index, node_index in enumerate(assignment):
            need = float(dnn.tasks[task_index].cpu_need)
            added_cpu[node_index] += need
            total_cpu_need += need
        predicted_node_cpu = tuple(
            float(snapshot.node_available_cpu[index]) - added_cpu[index]
            for index in range(len(self.nodes))
        )
        predicted_node_loads = {
            index: float(snapshot.node_loads[index])
            + added_cpu[index] / max(float(self.nodes[index].max_cpu), 1.0)
            for index in range(len(self.nodes))
            if snapshot.node_loads[index] > 0.0 or added_cpu[index] > 0.0
        }

        if initial_delay_hint is None:
            delay = self.estimate_delay_from_assignment_with_bandwidths(
                dnn_index,
                assignment,
                current_bandwidths,
            )
        else:
            delay = max(float(initial_delay_hint), 0.0)
        max_delay_seen = delay
        bandwidths = current_bandwidths
        converged = False
        iterations = 0
        relative_change = 0.0

        for iteration in range(1, self.candidate_fixed_point_max_iterations + 1):
            iterations = iteration
            candidate_rates = self.directional_offered_rates_from_data(
                self._collect_link_weights(dnn_index, assignment),
                max(delay, self.slot_length, 1.0),
            )
            offered_rates = dict(base_offered_rates)
            for link, rate in candidate_rates.items():
                offered_rates[link] = offered_rates.get(link, 0.0) + rate
            directional_loads = {
                link: rate / max(bandwidths.get(link, 1.0), 1.0)
                for link, rate in offered_rates.items()
                if rate > 0.0
            }
            physical_loads = {
                physical_link: max(
                    (
                        directional_loads.get(link, 0.0)
                        for link in self._directed_links_by_physical_id[
                            physical_link.physical_link_id
                        ]
                    ),
                    default=0.0,
                )
                for physical_link in self.physical_links
            }
            predicted_heats = {
                physical_link: (
                    self.lambda_g * physical_heat_by_link[physical_link]
                    + (1.0 - self.lambda_g) * physical_loads[physical_link]
                )
                for physical_link in self.physical_links
            }
            next_bandwidths = {
                link: self._effective_bandwidth_from_predicted_state(
                    link,
                    directional_loads.get(link, 0.0),
                    predicted_heats[link.physical_state],
                )
                for link in self.link_nodes
            }
            raw_delay = self.estimate_delay_from_assignment_with_bandwidths(
                dnn_index,
                assignment,
                next_bandwidths,
            )
            next_delay = (
                (1.0 - self.candidate_fixed_point_damping) * delay
                + self.candidate_fixed_point_damping * raw_delay
            )
            max_delay_seen = max(max_delay_seen, raw_delay, next_delay)
            relative_change = abs(next_delay - delay) / max(abs(delay), 1.0)
            delay = next_delay
            bandwidths = next_bandwidths
            if relative_change <= self.candidate_fixed_point_relative_tolerance:
                converged = True
                break

        if not converged:
            delay = max_delay_seen

        candidate_rates = self.directional_offered_rates_from_data(
            self._collect_link_weights(dnn_index, assignment),
            max(delay, self.slot_length, 1.0),
        )
        offered_rates = dict(base_offered_rates)
        for link, rate in candidate_rates.items():
            offered_rates[link] = offered_rates.get(link, 0.0) + rate
        directional_loads = {
            link: rate / max(bandwidths.get(link, 1.0), 1.0)
            for link, rate in offered_rates.items()
            if rate > 0.0
        }
        physical_loads = {
            physical_link: max(
                (
                    directional_loads.get(link, 0.0)
                    for link in self._directed_links_by_physical_id[
                        physical_link.physical_link_id
                    ]
                ),
                default=0.0,
            )
            for physical_link in self.physical_links
        }
        predicted_heats = {
            physical_link: (
                self.lambda_g * physical_heat_by_link[physical_link]
                + (1.0 - self.lambda_g) * physical_loads[physical_link]
            )
            for physical_link in self.physical_links
        }
        bandwidths = {
            link: self._effective_bandwidth_from_predicted_state(
                link,
                directional_loads.get(link, 0.0),
                predicted_heats[link.physical_state],
            )
            for link in self.link_nodes
        }

        quality_scores = self._quality_scores_from_predicted_loads(
            dnn_index,
            assignment,
            predicted_node_loads,
            physical_loads,
            node_heat_by_index,
            physical_heat_by_link,
        )
        service_exposure_slots = self.service_exposure_slots(delay)
        node_availability, availability_confidence = (
            self.aggregate_candidate_availability(
                dnn_index,
                assignment,
                service_exposure_slots,
                snapshot,
            )
        )
        if self.uses_periodic_semantics:
            operational_stability = self._aggregate_operational_stability(
                node_availability,
                quality_scores.link_stability,
            )
            quality_scores = CandidateQualityScores(
                operational_stability=operational_stability,
                node_stability=node_availability,
                link_stability=quality_scores.link_stability,
                raw_joint_product=node_availability
                * (quality_scores.link_stability or 1.0),
                used_node_count=quality_scores.used_node_count,
                used_physical_link_count=quality_scores.used_physical_link_count,
            )
        total_energy = float(
            self.count_total_energy_by_assignment(dnn_index, assignment)
        )
        deadline = max(float(dnn.delay), 1.0)
        delay_violation = max(0.0, delay - deadline) / deadline
        resource_overload = sum(max(0.0, -cpu) for cpu in predicted_node_cpu)
        resource_violation = resource_overload / max(total_cpu_need, 1.0)

        hierarchy_bad = 0
        hierarchy_total = 0
        task_index_by_id = {id(task): index for index, task in enumerate(dnn.tasks)}
        for task_index, node_index in enumerate(assignment):
            if self.nodes[node_index].level == 1 and node_index != dnn.initiateNode:
                hierarchy_bad += 1
            hierarchy_total += 1
            for link in dnn.links:
                if link.e_task is not dnn.tasks[task_index]:
                    continue
                hierarchy_total += 1
                predecessor = task_index_by_id[id(link.s_task)]
                if self.nodes[node_index].level < self.nodes[assignment[predecessor]].level:
                    hierarchy_bad += 1
        hierarchy_violation = hierarchy_bad / max(hierarchy_total, 1)
        link_overload = [
            max(0.0, load - 1.0)
            for load in directional_loads.values()
        ]
        link_violation = sum(link_overload) / max(len(link_overload), 1)
        used_nodes = self.collect_used_nodes(dnn_index, assignment)
        offline_nodes = [
            node_index
            for node_index in used_nodes
            if not snapshot.node_online[node_index]
        ]
        availability_violation = len(offline_nodes) / max(len(used_nodes), 1)
        constraint_violation = (
            delay_violation
            + resource_violation
            + hierarchy_violation
            + link_violation
            + availability_violation
        )
        violations = []
        if offline_nodes:
            violations.append("node_offline")
        if resource_overload > 0.0:
            violations.append("insufficient_cpu")
        if hierarchy_bad:
            violations.append("hierarchy_violation")
        if any(value > 0.0 for value in link_overload):
            violations.append("link_overload")
        if delay > deadline:
            violations.append("deadline_violation")

        delay_satisfaction = max(0.0, min(1.0, (deadline - delay) / deadline))
        energy_satisfaction = self.energy_satisfaction(dnn_index, total_energy)
        return CandidateStateSnapshot(
            observation_snapshot_version=snapshot.snapshot_version,
            environment_state_version=snapshot.environment_state_version,
            assignment=tuple(assignment),
            predicted_node_cpu=predicted_node_cpu,
            directional_offered_rates=tuple(
                offered_rates.get(link, 0.0) for link in self.link_nodes
            ),
            directional_link_loads=tuple(
                directional_loads.get(link, 0.0) for link in self.link_nodes
            ),
            physical_link_loads=tuple(
                physical_loads[physical_link] for physical_link in self.physical_links
            ),
            predicted_link_heats=tuple(
                predicted_heats[physical_link] for physical_link in self.physical_links
            ),
            predicted_effective_bandwidths=tuple(
                bandwidths[link] for link in self.link_nodes
            ),
            estimated_delay_ms=float(delay),
            total_energy=total_energy,
            node_stability_score=quality_scores.node_stability,
            node_availability_score=node_availability,
            service_exposure_slots=service_exposure_slots,
            availability_prediction_confidence=availability_confidence,
            link_stability_score=quality_scores.link_stability,
            operational_stability=quality_scores.operational_stability,
            delay_satisfaction=delay_satisfaction,
            energy_satisfaction=energy_satisfaction,
            raw_joint_product=quality_scores.raw_joint_product,
            deadline_feasible=delay <= deadline,
            fixed_point_converged=converged,
            fixed_point_iterations=iterations,
            fixed_point_relative_residual=float(relative_change),
            constraint_violation=constraint_violation,
            constraint_violations=tuple(violations),
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
        self.environment_state_version += 1
        if self.verbose:
            print(
                "[ACCEPTED] "
                f"slot={self.current_slot} "
                f"dnn={dnn_index} "
                f"estimated_runtime={estimated_runtime:.2f} "
                f"required_slots={remaining_slots}"
            )
        return running_dnn

    def commit_assignment(
        self,
        dnn_index: int,
        assignment: List[int],
        expected_environment_state_version: int,
    ) -> CandidateStateSnapshot:
        """在版本一致时最终复验并原子注册一个运行请求。"""
        with self._commit_lock:
            if self.environment_state_version != expected_environment_state_version:
                raise StaleEnvironmentStateError(
                    "environment changed after online candidate evaluation"
                )
            snapshot = self.capture_observation_snapshot("online_admission")
            candidate = self.predict_candidate_state(dnn_index, assignment, snapshot)
            if not candidate.is_strictly_feasible:
                reasons = ",".join(candidate.constraint_violations) or "constraint_violation"
                raise InfeasibleAssignmentError(reasons)
            if self.environment_state_version != expected_environment_state_version:
                raise StaleEnvironmentStateError(
                    "environment changed during final candidate revalidation"
                )
            self.add_running_dnn(
                dnn_index=dnn_index,
                assignment=assignment,
                estimated_runtime=candidate.estimated_delay_ms,
                remaining_slots=candidate.service_exposure_slots,
            )
            return candidate

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
        self.update_node_availability()
        self._print_slot_summary(slot_id, finished_count)
        self.current_slot += 1
        self.environment_state_version += 1

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

    def abort_running_dnn(self, dnn_index: int) -> bool:
        """因执行期节点故障终止一个运行请求并立即释放资源。"""
        for index, running_dnn in enumerate(self.running_dnns):
            if running_dnn.dnn_index != dnn_index:
                continue
            self.free_running_resources(running_dnn)
            del self.running_dnns[index]
            self.allocate_running_resources()
            self.environment_state_version += 1
            return True
        return False

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
        offered_rates = self.aggregate_directional_offered_rates()
        directional_loads = self.directional_load_ratios_from_offered_rates(offered_rates)
        for link in self.link_nodes:
            link.load_ratio = directional_loads.get(link, 0.0)
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
        """依据负载和历史压力刷新节点与链路的动态稳定性。"""
        # 稳定性由经验先验和当前负载/历史压力共同决定，
        # 同时做上下界裁剪，避免高于先验或低于下界。
        for node in self.nodes:
            base_o = node.base_operational_stability
            next_o = base_o * math.exp(-self.alpha_r * node.load_ratio - self.beta_r * node.heat)
            node.operational_stability = min(base_o, max(self.o_min, next_o))
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

    def evaluate_post_admission_metrics(
        self,
        dnn_index: int,
        assignment: List[int],
    ) -> PostAdmissionMetrics:
        """按候选接纳后的预测状态统一计算三类指标。"""
        candidate = self.predict_candidate_state(dnn_index, assignment)
        return PostAdmissionMetrics(
            delay=candidate.estimated_delay_ms,
            operational_stability=candidate.operational_stability,
            total_energy=candidate.total_energy,
            raw_joint_product=candidate.raw_joint_product,
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
        """兼容 RTBL：只读取当前状态，不在算法调用处推进故障过程。"""
        return [1 if self.up[index] else 0 for index in range(self.availability_node_count)]

    def update_node_availability(self) -> None:
        """在系统时隙边界统一推进一次节点可用性状态机。"""
        for index, node in enumerate(self.nodes):
            if node.availability_mode == "always_on":
                self.up[index] = True
            elif self.up[index]:
                if self.remaining_time[index] > 0:
                    self.remaining_time[index] -= 1
                else:
                    self.up[index] = False
                    self.remaining_time[index] = self._sample_downtime()
            elif self.remaining_time[index] > 0:
                self.remaining_time[index] -= 1
            else:
                self.up[index] = True
                self.remaining_time[index] = self._sample_uptime()
            self.rj_history[index].append(1 if self.up[index] else 0)
            overflow = len(self.rj_history[index]) - self.availability_history_limit
            if overflow > 0:
                del self.rj_history[index][:overflow]

    def _generate_rj_history(self) -> List[List[int]]:
        """生成一段离线可用性历史供 RTBL 初始化。"""
        history = [[] for _ in self.nodes]
        up_virtual = [True for _ in self.nodes]
        rem_virtual = [
            self._sample_uptime() if node.availability_mode == "stochastic" else 0
            for node in self.nodes
        ]
        for _ in range(self.h_off):
            snapshot = [1 for _ in self.nodes]
            for j, node in enumerate(self.nodes):
                if node.availability_mode == "always_on":
                    continue
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
            for j in range(len(self.nodes)):
                history[j].append(snapshot[j])
        return history

    def _sample_uptime(self) -> int:
        """按 Weibull 分布采样在线时长。"""
        u = self._availability_rng.random()
        return int(self.scale * math.pow(-math.log(1 - u), 1.0 / self.shape))

    def _sample_downtime(self) -> int:
        """按对数正态分布采样离线时长。"""
        u = self._availability_rng.random()
        v = self._availability_rng.random()
        z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
        return int(math.exp(self.mean + self.sigma * z))
