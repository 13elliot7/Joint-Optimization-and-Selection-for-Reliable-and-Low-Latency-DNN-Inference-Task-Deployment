from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

from core.topology import create_dnns, create_nodes
from models import DNN, LinkDNN, LinkNode, Node, RunningDNN, Task


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
    k_range: tuple[float, float] = (2e3, 4e3)
    i_range: tuple[int, int] = (5000, 10000)
    c_range: tuple[float, float] = (0.15, 0.20)
    alpha_1_range: tuple[float, float] = (0.45, 0.65)
    alpha2_range: tuple[float, float] = (0.75, 1.0)
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

    def __post_init__(self) -> None:
        """初始化拓扑、DNN 请求和动态运行状态。"""
        self.nodes, self.link_nodes = create_nodes()
        self.ds = create_dnns(self.t_max, self.nodes)
        self.graph_node: List[List[float]] = []
        self.get_graph_matrix()
        self.k = self._random_uniform(*self.k_range)
        self.rj_history = self._generate_rj_history()
        self.up = [True for _ in range(self.availability_node_count)]
        self.remaining_time = [self._sample_uptime() for _ in range(self.availability_node_count)]
        self.running_dnns: List[RunningDNN] = []
        self.current_slot = 0
        self.finished_total = 0
        self._initialize_dynamic_state()
        self._print_initialization_summary()

    def reset_nodes(self, snapshot: List[Node]) -> None:
        """用快照恢复节点状态并清空动态运行上下文。"""
        for idx, node in enumerate(snapshot):
            self.nodes[idx].cpu = node.cpu
            self.nodes[idx].max_cpu = node.max_cpu
            self.nodes[idx].level = node.level
            self.nodes[idx].o_reliability = node.o_reliability
            self.nodes[idx].a_reliability = node.a_reliability
            self.nodes[idx].float_rate = node.float_rate
            self.nodes[idx].base_o_reliability = node.base_o_reliability
            self.nodes[idx].base_a_reliability = node.base_a_reliability
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

    def get_graph_matrix(self) -> None:
        """根据链路带宽构建最短路计算所需的图矩阵。"""
        size = len(self.nodes)
        self.graph_node = [[float("inf") for _ in range(size)] for _ in range(size)]
        for link_node in self.link_nodes:
            s = self.nodes.index(link_node.s_node)
            e = self.nodes.index(link_node.e_node)
            self.graph_node[s][e] = 1.0 / link_node.band_width
            self.graph_node[e][s] = 1.0 / link_node.band_width

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

    def release_resource(self, dnn: DNN, assignment: List[int]) -> None:
        """按旧静态模型扣减已部署 DNN 的节点资源。"""
        for i, task in enumerate(dnn.tasks):
            self.nodes[assignment[i]].cpu = self.nodes[assignment[i]].cpu - task.cpu_need

    def _initialize_dynamic_state(self) -> None:
        """初始化节点和链路的动态可靠性状态。"""
        for node in self.nodes:
            if node.base_o_reliability is None:
                node.base_o_reliability = node.o_reliability
            if node.base_a_reliability is None:
                node.base_a_reliability = node.a_reliability
            node.load_ratio = 0.0
            node.heat = 0.0
            node.o_reliability = node.base_o_reliability
            node.a_reliability = node.base_a_reliability
        self._reset_link_state()

    def _print_initialization_summary(self) -> None:
        """打印环境初始化摘要。"""
        cloud_count = sum(1 for node in self.nodes if node.level == 3)
        edge_count = sum(1 for node in self.nodes if node.level == 2)
        user_count = sum(1 for node in self.nodes if node.level == 1)
        print(
            "[INIT] "
            f"t_max={self.t_max} "
            f"nodes={len(self.nodes)}(cloud={cloud_count} edge={edge_count} user={user_count}) "
            f"links={len(self.link_nodes)} dnns={len(self.ds)}"
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
        dnn = self.ds[dnn_index]
        print(
            "[PENDING] "
            f"slot={self.current_slot} "
            f"dnn={dnn_index} "
            f"tasks={len(dnn.tasks)} "
            f"deadline={dnn.delay} "
            f"initiate={dnn.initiateNode} "
            f"preA={dnn.preA:.3f} "
            f"preR={dnn.preR:.3f}"
        )

    def _reset_link_state(self) -> None:
        """重置链路的先验可靠性和动态状态。"""
        for link in self.link_nodes:
            link.base_reliability = self._infer_link_base_reliability(link)
            link.reliability = link.base_reliability
            link.load_ratio = 0.0
            link.heat = 0.0

    def _infer_link_base_reliability(self, link: LinkNode) -> float:
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

    def _build_assignment_matrix(self, dnn_index: int, assignment: List[int]) -> List[List[int]]:
        """把一维部署向量转成二维任务-节点矩阵。"""
        task_count = len(self.ds[dnn_index].tasks)
        x = [[0 for _ in range(len(self.nodes))] for _ in range(task_count)]
        for task_idx in range(task_count):
            node_idx = assignment[task_idx]
            if node_idx < 0:
                raise IndexError("assignment index -1")
            x[task_idx][node_idx] = 1
        return x

    def estimate_delay_from_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """复用原始静态时延模型估计部署完成时间。"""
        x = self._build_assignment_matrix(dnn_index, assignment)
        return self.count_delay_random(self.ds[dnn_index], x)

    def collect_used_nodes(self, dnn_index: int, assignment: List[int]) -> List[int]:
        """提取部署方案实际使用的唯一节点集合。"""
        del dnn_index
        return sorted({node_idx for node_idx in assignment if node_idx >= 0})

    def _add_path_link_weights(self, weights: Dict[LinkNode, float], links: List[LinkNode], data_amount: float) -> None:
        """把一段路径上的流量累计到链路权重表。"""
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

    def collect_used_links(self, dnn_index: int, assignment: List[int]) -> List[LinkNode]:
        """提取部署方案实际经过的唯一链路集合。"""
        return list(self._collect_link_weights(dnn_index, assignment).keys())

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
        """统计部署方案的链路传输能耗，不计最终结果回传。"""
        total_energy = 0.0
        for link, data_amount in self._collect_link_weights(dnn_index, assignment, include_return=False).items():
            total_energy += link.energy_per_mb * data_amount
        return total_energy

    def count_total_energy_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """统计部署方案的总能耗。"""
        return self.count_compute_energy_by_assignment(dnn_index, assignment) + self.count_link_energy_by_assignment(
            dnn_index, assignment
        )

    def count_energy_utility_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """把总能耗转成越大越好的能耗收益。"""
        total_energy = self.count_total_energy_by_assignment(dnn_index, assignment)
        return 1.0 / total_energy if total_energy > 0 else 0.0

    def add_running_dnn(
        self,
        dnn_index: int,
        assignment: List[int],
        arrival_time: int,
        start_time: int,
        estimated_runtime: float,
        remaining_slots: int,
    ) -> RunningDNN:
        """把一个已接纳 DNN 注册到运行队列。"""
        running_dnn = RunningDNN(
            dnn_index=dnn_index,
            assignment=list(assignment),
            arrival_time=arrival_time,
            start_time=start_time,
            estimated_runtime=estimated_runtime,
            remaining_slots=remaining_slots,
            used_nodes=self.collect_used_nodes(dnn_index, assignment),
            used_links=self.collect_used_links(dnn_index, assignment),
        )
        self.running_dnns.append(running_dnn)
        # 接纳成功后立即把新 DNN 纳入当前占用，
        # 但热度和动态可靠性仍然只在时隙推进时更新。
        self.allocate_running_resources()
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
        self.refresh_dynamic_reliability()
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
            bandwidth = max(float(link.band_width), 1.0)
            link.load_ratio = min(1.0, current_used / bandwidth)

    def update_node_heat(self) -> None:
        """按遗忘系数更新节点热度。"""
        for node in self.nodes:
            node.heat = self.lambda_h * node.heat + node.load_ratio

    def update_link_heat(self) -> None:
        """按遗忘系数更新链路热度。"""
        for link in self.link_nodes:
            link.heat = self.lambda_g * link.heat + link.load_ratio

    def refresh_dynamic_reliability(self) -> None:
        """依据负载和热度刷新节点与链路的动态可靠性。"""
        # 可靠性由静态先验和当前负载/热度共同决定，
        # 同时做上下界裁剪，避免高于先验或低于下界。
        for node in self.nodes:
            base_o = node.base_o_reliability if node.base_o_reliability is not None else node.o_reliability
            base_a = node.base_a_reliability if node.base_a_reliability is not None else node.a_reliability
            next_o = base_o * math.exp(-self.alpha_r * node.load_ratio - self.beta_r * node.heat)
            next_a = base_a * math.exp(-self.alpha_a * node.load_ratio - self.beta_a * node.heat)
            node.o_reliability = min(base_o, max(self.o_min, next_o))
            node.a_reliability = min(base_a, max(self.a_min, next_a))
        for link in self.link_nodes:
            base_reliability = link.base_reliability if link.base_reliability is not None else 1.0
            next_reliability = base_reliability * math.exp(-self.alpha_l * link.load_ratio - self.beta_l * link.heat)
            link.reliability = min(base_reliability, max(self.l_min, next_reliability))

    def _print_slot_summary(self, slot_id: int, finished_count: int) -> None:
        """打印单个时隙的运行摘要。"""
        node_loads = [node.load_ratio for node in self.nodes]
        link_loads = [link.load_ratio for link in self.link_nodes]
        node_reliabilities = [node.o_reliability for node in self.nodes]
        link_reliabilities = [
            link.reliability if link.reliability is not None else link.base_reliability or 1.0
            for link in self.link_nodes
        ]
        node_load_avg = sum(node_loads) / len(node_loads) if node_loads else 0.0
        node_load_max = max(node_loads) if node_loads else 0.0
        link_load_avg = sum(link_loads) / len(link_loads) if link_loads else 0.0
        link_load_max = max(link_loads) if link_loads else 0.0
        node_r_avg = sum(node_reliabilities) / len(node_reliabilities) if node_reliabilities else 0.0
        link_r_avg = sum(link_reliabilities) / len(link_reliabilities) if link_reliabilities else 0.0
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

    def count_dynamic_link_reliability_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """计算部署方案经过链路的联合可靠性。"""
        reliability = 1.0
        for link in self.collect_used_links(dnn_index, assignment):
            link_reliability = link.reliability if link.reliability is not None else link.base_reliability or 1.0
            reliability *= link_reliability
        return reliability

    def count_dynamic_operation_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """按唯一节点和唯一链路计算动态运行可靠性。"""
        # 运行可靠性按路径级口径计算：
        # 唯一执行节点和唯一经过链路各自只乘一次。
        reliability = 1.0
        for node_index in self.collect_used_nodes(dnn_index, assignment):
            reliability *= self.nodes[node_index].o_reliability
        reliability *= self.count_dynamic_link_reliability_by_assignment(dnn_index, assignment)
        return reliability

    def count_dynamic_accuracy_by_assignment(self, dnn_index: int, assignment: List[int]) -> float:
        """按节点承载计算量权重计算动态精度可靠性。"""
        dnn = self.ds[dnn_index]
        total_float = sum(task.float_num for task in dnn.tasks)
        if total_float == 0:
            return 0.0
        float_by_node: Dict[int, float] = {}
        for task_idx, node_idx in enumerate(assignment):
            float_by_node[node_idx] = float_by_node.get(node_idx, 0.0) + dnn.tasks[task_idx].float_num
        reliability = 1.0
        for node_idx, node_float in float_by_node.items():
            weight = node_float / total_float
            reliability *= self.nodes[node_idx].a_reliability ** weight
        return reliability

    def check_delay_random_3d(self, dnn: DNN, x: List[List[List[int]]]) -> bool:
        """检查三维部署矩阵是否满足时延约束。"""
        t = 0.0
        tran = 0.0
        tloc = 0.0
        index = self.ds.index(dnn)
        for i, task in enumerate(dnn.tasks):
            node = None
            for j in range(len(self.nodes)):
                if x[index][i][j] == 1:
                    node = self.nodes[j]
            if node is not None:
                tloc += node.float_rate / task.float_num
            next_tasks = self.get_next_dnn_tasks(dnn, task)
            if not next_tasks:
                continue
            for next_task in next_tasks:
                for link in dnn.links:
                    if link.s_task is task and link.e_task is next_task:
                        tran += self.count_tran_delay(dnn, index, task, next_task, link.float_tran, x)
        return dnn.delay < t

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
                    tran += dnn.startFloat / link.band_width
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
                    tloc += dnn.backFloat / link.band_width
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
                    tran += dnn.startFloat / link.band_width
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
                    tloc += dnn.backFloat / link.band_width
        return tloc + tran

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
            tran += float_num / link.band_width
        return tran

    def count_tran_delay(
        self,
        dnn: DNN,
        i: int,
        stask: Task,
        etask: Task,
        float_num: int,
        x: List[List[List[int]]],
    ) -> float:
        """计算三维部署下两任务间的数据传输时延。"""
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
            if x[i][m][j] == 1:
                snode = self.nodes[j]
            if x[i][n][j] == 1:
                enode = self.nodes[j]
        tran = 0.0
        for link in self.get_arrive_link(snode, enode):
            tran += float_num / link.band_width
        return tran

    def get_arrive_link(self, s: Node | None, e: Node | None) -> List[LinkNode]:
        """返回两个节点间最短路对应的链路序列。"""
        if s is e:
            return []
        if s is None or e is None:
            raise ValueError("Node assignment missing")
        snode = self.nodes.index(s)
        enode = self.nodes.index(e)
        path = self.get_shortest_path(snode, enode)
        links: List[LinkNode] = []
        for p in range(len(path) - 1):
            start = self.nodes[path[p]]
            end = self.nodes[path[p + 1]]
            for link in self.link_nodes:
                if link.s_node is start and link.e_node is end:
                    links.append(link)
        return links

    def get_shortest_path(self, start: int, end: int) -> List[int]:
        """基于带宽倒数权重计算节点间最短路径。"""
        parent = [-1 for _ in range(len(self.nodes))]
        distance = [float("inf") for _ in range(len(self.nodes))]
        distance[start] = 0.0
        n = len(self.nodes)
        for _ in range(n - 1):
            for j in range(n):
                for k in range(n):
                    if (
                        self.graph_node[j][k] != float("inf")
                        and distance[j] != float("inf")
                        and distance[j] + self.graph_node[j][k] < distance[k]
                    ):
                        distance[k] = distance[j] + self.graph_node[j][k]
                        parent[k] = j
        path: List[int] = []
        current = end
        while current != -1:
            path.insert(0, current)
            current = parent[current]
        return path

    def count_accuracy(self, dnn: DNN, m: int, x: List[List[List[int]]]) -> float:
        """按原始静态模型计算精度可靠性。"""
        accuracy = 1.0
        for i in range(len(dnn.tasks)):
            for j in range(len(self.nodes)):
                if x[m][i][j] == 1:
                    accuracy = self.nodes[j].a_reliability * accuracy * self.nodes[j].o_reliability
        return accuracy

    def count_operation(self, dnn: DNN, m: int, x: List[List[List[int]]]) -> float:
        """按原始静态模型计算运行可靠性。"""
        operation = 1.0
        for i in range(len(dnn.tasks)):
            for j in range(len(self.nodes)):
                if x[m][i][j] == 1:
                    operation = self.nodes[j].o_reliability * operation
        return operation

    def get_next_dnn_tasks(self, dnn: DNN, s: Task) -> List[Task]:
        """返回指定任务的后继任务集合。"""
        return [link.e_task for link in dnn.links if link.s_task is s]

    def count_tran(self, i: int, dnn: DNN, task: Task, x: List[List[List[int]]]) -> float:
        """递归计算三维部署下到达某任务的关键路径时延。"""
        index = -1
        for j, current_task in enumerate(dnn.tasks):
            if task is current_task:
                index = j
        tloc = 0.0
        for j in range(len(self.nodes)):
            if x[i][index][j] == 1:
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
                    + self.count_tran_delay(dnn, i, link.s_task, task, link.float_tran, x)
                    + self.count_tran(i, dnn, link.s_task, x),
                )
        return tran

    def count_delay(self, dnn: DNN, x: List[List[List[int]]]) -> float:
        """计算三维部署矩阵对应的端到端总时延。"""
        tran = 0.0
        tloc = 0.0
        index = self.ds.index(dnn)
        for i, task in enumerate(dnn.tasks):
            node = None
            for j in range(len(self.nodes)):
                if x[index][i][j] == 1:
                    node = self.nodes[j]
            if i == 0:
                links = self.get_arrive_link(self.nodes[dnn.initiateNode], node)
                for link in links:
                    tran += dnn.startFloat / link.band_width
            if node is not None:
                tloc += task.float_num / node.float_rate
            next_tasks = self.get_next_dnn_tasks(dnn, task)
            if not next_tasks:
                continue
        tran += self.count_tran(index, dnn, dnn.tasks[len(dnn.tasks) - 1], x)
        tloc = 0.0
        for j in range(len(self.nodes)):
            if x[index][len(dnn.tasks) - 1][j] == 1:
                links = self.get_arrive_link(self.nodes[j], self.nodes[dnn.initiateNode])
                for link in links:
                    tloc += dnn.backFloat / link.band_width
        return tloc + tran

    def count_values1_random(self, i: int, m: int, r: List[List[List[int]]]) -> float:
        """按旧基线接口计算精度目标值。"""
        x = [[[0 for _ in range(len(self.nodes))] for _ in range(len(self.ds[i].tasks))] for _ in range(i + 1)]
        for j in range(len(self.ds[i].tasks)):
            if r[m][i][j] == -1:
                print("error")
                raise IndexError("assignment index -1")
            x[i][j][r[m][i][j]] = 1
        return float(self.count_accuracy(self.ds[i], i, x))

    def count_values2_random(self, i: int, m: int, r: List[List[List[int]]]) -> float:
        """按旧基线接口计算运行可靠性目标值。"""
        x = [[[0 for _ in range(len(self.nodes))] for _ in range(len(self.ds[i].tasks))] for _ in range(i + 1)]
        for j in range(len(self.ds[i].tasks)):
            if r[m][i][j] == -1:
                raise IndexError("assignment index -1")
            x[i][j][r[m][i][j]] = 1
        return float(self.count_operation(self.ds[i], i, x))

    def count_values3_random(self, i: int, m: int, r: List[List[List[int]]]) -> float:
        """按旧基线接口计算时延收益目标值。"""
        x = [[[0 for _ in range(len(self.nodes))] for _ in range(len(self.ds[i].tasks))] for _ in range(i + 1)]
        for j in range(len(self.ds[i].tasks)):
            if r[m][i][j] == -1:
                raise IndexError("assignment index -1")
            x[i][j][r[m][i][j]] = 1
        delay = self.count_delay(self.ds[i], x)
        if delay > self.ds[i].delay:
            return float(self.ds[i].delay - delay)
        return float(1 / delay)

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
                    transmission_delay += dnn.startFloat / link.band_width
            return_delay = 0.0
            if task_index == len(dnn.tasks) - 1:
                links = self.get_arrive_link(target_node, self.nodes[dnn.initiateNode])
                for link in links:
                    return_delay += dnn.backFloat / link.band_width
            inter_task_delay = 0.0
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
                        inter_task_delay1 = link.getFloatTran() / inter_link.band_width
                        if inter_task_delay1 > max_inter_task_delay:
                            max_inter_task_delay = inter_task_delay1
            delay_costs[node_index] = computation_delay + transmission_delay + return_delay + inter_task_delay
        return delay_costs

    def generate_task(self) -> tuple[int, int]:
        """随机生成 RTBL 任务大小与计算需求。"""
        i_t = random.randint(self.i_range[0], self.i_range[1])
        c_t = self._random_uniform(*self.c_range)
        return i_t, int(c_t * 1e9)

    def generate_accuracy(self) -> List[float]:
        """生成 RTBL 场景下各节点的精度参数。"""
        result = [0.0 for _ in range(self.availability_node_count)]
        result[0] = self._random_uniform(*self.alpha_1_range)
        for j in range(1, self.availability_node_count):
            result[j] = self._random_uniform(*self.alpha2_range)
        return result

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

    @staticmethod
    def _random_uniform(min_value: float, max_value: float) -> float:
        """在给定区间内采样均匀随机数。"""
        return min_value + (max_value - min_value) * random.random()
