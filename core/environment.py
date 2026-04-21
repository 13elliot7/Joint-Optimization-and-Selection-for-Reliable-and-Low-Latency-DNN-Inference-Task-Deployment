from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from src_python.core.topology import create_dnns, create_nodes
from src_python.models import DNN, LinkDNN, LinkNode, Node, Task


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

    def __post_init__(self) -> None:
        self.nodes, self.link_nodes = create_nodes()
        self.ds = create_dnns(self.t_max, self.nodes)
        self.graph_node: List[List[float]] = []
        self.get_graph_matrix()
        self.k = self._random_uniform(*self.k_range)
        self.rj_history = self._generate_rj_history()
        self.up = [True for _ in range(self.availability_node_count)]
        self.remaining_time = [self._sample_uptime() for _ in range(self.availability_node_count)]

    def reset_nodes(self, snapshot: List[Node]) -> None:
        for idx, node in enumerate(snapshot):
            self.nodes[idx].cpu = node.cpu
            self.nodes[idx].max_cpu = node.max_cpu
            self.nodes[idx].level = node.level
            self.nodes[idx].o_reliability = node.o_reliability
            self.nodes[idx].a_reliability = node.a_reliability
            self.nodes[idx].float_rate = node.float_rate

    def clone_nodes(self) -> List[Node]:
        return [node.clone() for node in self.nodes]

    def get_graph_matrix(self) -> None:
        size = len(self.nodes)
        self.graph_node = [[float("inf") for _ in range(size)] for _ in range(size)]
        for link_node in self.link_nodes:
            s = self.nodes.index(link_node.s_node)
            e = self.nodes.index(link_node.e_node)
            self.graph_node[s][e] = 1.0 / link_node.band_width
            self.graph_node[e][s] = 1.0 / link_node.band_width

    def check_resource(self, dnn: DNN, x: List[List[int]]) -> bool:
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
        for i, task in enumerate(dnn.tasks):
            self.nodes[assignment[i]].cpu = self.nodes[assignment[i]].cpu - task.cpu_need

    def check_delay_random_3d(self, dnn: DNN, x: List[List[List[int]]]) -> bool:
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
        return dnn.delay > self.count_delay_random(dnn, x)

    def count_tran_random(self, i: int, dnn: DNN, task: Task, x: List[List[int]]) -> float:
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
        accuracy = 1.0
        for i in range(len(dnn.tasks)):
            for j in range(len(self.nodes)):
                if x[m][i][j] == 1:
                    accuracy = self.nodes[j].a_reliability * accuracy * self.nodes[j].o_reliability
        return accuracy

    def count_operation(self, dnn: DNN, m: int, x: List[List[List[int]]]) -> float:
        operation = 1.0
        for i in range(len(dnn.tasks)):
            for j in range(len(self.nodes)):
                if x[m][i][j] == 1:
                    operation = self.nodes[j].o_reliability * operation
        return operation

    def get_next_dnn_tasks(self, dnn: DNN, s: Task) -> List[Task]:
        return [link.e_task for link in dnn.links if link.s_task is s]

    def count_tran(self, i: int, dnn: DNN, task: Task, x: List[List[List[int]]]) -> float:
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
        x = [[[0 for _ in range(len(self.nodes))] for _ in range(len(self.ds[i].tasks))] for _ in range(i + 1)]
        for j in range(len(self.ds[i].tasks)):
            if r[m][i][j] == -1:
                print("error")
                raise IndexError("assignment index -1")
            x[i][j][r[m][i][j]] = 1
        return float(self.count_accuracy(self.ds[i], i, x))

    def count_values2_random(self, i: int, m: int, r: List[List[List[int]]]) -> float:
        x = [[[0 for _ in range(len(self.nodes))] for _ in range(len(self.ds[i].tasks))] for _ in range(i + 1)]
        for j in range(len(self.ds[i].tasks)):
            if r[m][i][j] == -1:
                raise IndexError("assignment index -1")
            x[i][j][r[m][i][j]] = 1
        return float(self.count_operation(self.ds[i], i, x))

    def count_values3_random(self, i: int, m: int, r: List[List[List[int]]]) -> float:
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
        i_t = random.randint(self.i_range[0], self.i_range[1])
        c_t = self._random_uniform(*self.c_range)
        return i_t, int(c_t * 1e9)

    def generate_accuracy(self) -> List[float]:
        result = [0.0 for _ in range(self.availability_node_count)]
        result[0] = self._random_uniform(*self.alpha_1_range)
        for j in range(1, self.availability_node_count):
            result[j] = self._random_uniform(*self.alpha2_range)
        return result

    def generate_availability(self) -> List[int]:
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
        u = random.random()
        return int(self.scale * math.pow(-math.log(1 - u), 1.0 / self.shape))

    def _sample_downtime(self) -> int:
        u = random.random()
        v = random.random()
        z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
        return int(math.exp(self.mean + self.sigma * z))

    @staticmethod
    def _random_uniform(min_value: float, max_value: float) -> float:
        return min_value + (max_value - min_value) * random.random()
