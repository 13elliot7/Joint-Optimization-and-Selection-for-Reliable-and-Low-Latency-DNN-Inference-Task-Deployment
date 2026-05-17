from __future__ import annotations

import math
import random
import time
from typing import List

from core.environment import Environment
from metrics import ExperimentMetrics
from models import DNN, LinkNode, Node


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
        self.start_time = 0.0
        self.res: List[List[List[int]]] = []
        self.res_next: List[List[List[int]]] = []
        self.res_pq: List[List[List[int]]] = []
        self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
        self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
        self.pareto_level_sort: List[List[int]] = []
        self.distance = [[0 for _ in range(2 * self.pop_size)] for _ in range(1000)]

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

    def _evaluate_assignment(self, dnn_index: int, assignment: List[int]) -> tuple[float, float, float, float]:
        """计算一个候选部署的三目标值与估计时延。"""
        # 所有算法统一复用这套动态评价：当前节点/链路可靠性
        # 加上基于原始关键路径估计得到的时延收益。
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        value1 = float(self.env.count_dynamic_accuracy_by_assignment(dnn_index, assignment_slice))
        value2 = float(self.env.count_dynamic_operation_by_assignment(dnn_index, assignment_slice))
        delay = self.env.estimate_delay_from_assignment(dnn_index, assignment_slice)
        if delay > self.env.ds[dnn_index].delay:
            value3 = float(self.env.ds[dnn_index].delay - delay)
        else:
            value3 = float(1 / delay)
        return value1, value2, value3, delay

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
        """执行按后缀交换的交叉操作。"""
        dnn = self.env.ds[i]
        d_m = len(dnn.tasks)
        self.start_time = time.monotonic()
        while True:
            if (time.monotonic() - self.start_time) * 1000 > 10000:
                break
            xm = self._build_assignment_matrix(i, self.res[m][i])
            xn = self._build_assignment_matrix(i, self.res[n][i])
            index_m = int(d_m * random.random())
            index_n = index_m
            for j in range(index_m, d_m):
                xm[j][self.res[m][i][j]] = 0
                xm[j][self.res[n][i][j]] = 1
                xn[j][self.res[n][i][j]] = 0
                xn[j][self.res[m][i][j]] = 1
            if index_n == -1:
                continue
            if self.env.check_resource(dnn, xn) and self.env.check_resource(dnn, xm):
                for j in range(index_m, d_m):
                    temp = self.res[n][i][j]
                    self.res[n][i][j] = self.res[m][i][j]
                    self.res[m][i][j] = temp
                break
        self.mutate(i, m)
        for x in range(len(self.res[m][i])):
            self.res_pq[next_index][i][x] = self.res[m][i][x]
        self.mutate(i, n)
        for x in range(len(self.res[n][i])):
            self.res_pq[next_index][i][x] = self.res[n][i][x]


    def cross_over(self, i: int, m: int, n: int, next_index: int) -> None:
        """执行按单点交换的交叉操作。"""
        dnn = self.env.ds[i]
        d_m = len(dnn.tasks)
        d_n = len(dnn.tasks)
        xm = self._build_assignment_matrix(i, self.res[m][i])
        xn = self._build_assignment_matrix(i, self.res[n][i])
        self.start_time = time.monotonic()
        while True:
            if (time.monotonic() - self.start_time) * 1000 > 10000:
                break
            index_m = int(d_m * random.random())
            index_n = int(d_n * random.random())
            xm[index_m][self.res[n][i][index_n]] = 1
            xm[index_m][self.res[m][i][index_m]] = 0
            xn[index_n][self.res[m][i][index_m]] = 1
            xn[index_n][self.res[n][i][index_n]] = 0
            if (
                self.env.nodes[self.res[m][i][index_m]].level != 1
                and self.env.nodes[self.res[n][i][index_n]].level != 1
                and self.env.check_resource(dnn, xn)
                and self.env.check_resource(dnn, xm)
            ):
                temp = self.res[n][i][index_n]
                self.res[n][i][index_n] = self.res[m][i][index_m]
                self.res[m][i][index_m] = temp
                break
            xm[index_m][self.res[n][i][index_n]] = 0
            xm[index_m][self.res[m][i][index_m]] = 1
            xn[index_n][self.res[m][i][index_m]] = 0
            xn[index_n][self.res[n][i][index_n]] = 1
        self.mutate(i, m)
        for x in range(len(self.res[m][i])):
            self.res_pq[next_index][i][x] = self.res[m][i][x]
        self.mutate(i, n)
        for x in range(len(self.res[n][i])):
            self.res_pq[next_index][i][x] = self.res[n][i][x]

    def count_values1(self, i: int, m: int) -> None:
        """计算第一个目标值并写回缓存。"""
        self.function1_values[m] = self._evaluate_assignment(i, self.res_pq[m][i])[0]

    def count_values2(self, i: int, m: int) -> None:
        """计算第二个目标值并写回缓存。"""
        self.function2_values[m] = self._evaluate_assignment(i, self.res_pq[m][i])[1]

    def count_values3(self, i: int, m: int) -> None:
        """计算第三个目标值并写回缓存。"""
        self.function3_values[m] = self._evaluate_assignment(i, self.res_pq[m][i])[2]

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
        """按拥挤距离规则计算同层个体的距离值。"""
        v1 = list(range(len(values1)))
        v2 = list(range(len(values2)))
        v3 = list(range(len(values3)))
        for m in range(len(v1)):
            for n in range(len(v1) - 1 - m):
                if values1[n] > values1[n + 1]:
                    values1[n], values1[n + 1] = values1[n + 1], values1[n]
                    v1[n], v1[n + 1] = v1[n + 1], v1[n]
        for m in range(len(v2)):
            for n in range(len(v2) - 1 - m):
                if values2[n] > values2[n + 1]:
                    values2[n], values2[n + 1] = values2[n + 1], values2[n]
                    v2[n], v2[n + 1] = v2[n + 1], v2[n]
        for m in range(len(v3)):
            for n in range(len(v3) - 1 - m):
                if values3[n] > values3[n + 1]:
                    values3[n], values3[n + 1] = values3[n + 1], values3[n]
                    v3[n], v3[n + 1] = v3[n + 1], v3[n]
        for m in range(len(pq)):
            if v1[m] == 0 or v1[m] == len(values1) - 1:
                self.distance[i][m] = 2**31 - 1
            else:
                self.distance[i][m] += _java_div(abs(values1[v1[m] + 1] - values1[v1[m] - 1]), values1[-1] - values1[0])
        for m in range(len(pq)):
            if v2[m] == 0 or v2[m] == len(values2) - 1:
                self.distance[i][m] = 2**31 - 1
            else:
                self.distance[i][m] += _java_div(abs(values2[v2[m] + 1] - values2[v2[m] - 1]), values2[-1] - values2[0])
        for m in range(len(pq)):
            if v3[m] == 0 or v3[m] == len(values3) - 1:
                self.distance[i][m] = 2**31 - 1
            else:
                self.distance[i][m] += _java_div(abs(values3[v3[m] + 1] - values3[v3[m] - 1]), values3[-1] - values3[0])
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

    def _score(self, value1: float, value2: float, value3: float, w_a: float, w_r: float, w_t: float, min_a: float, max_a: float, min_r: float, max_r: float, min_t: float, max_t: float) -> float:
        """对满足时延的候选解做归一化加权打分。"""
        return (
            w_a * _java_div(value1 - min_a, max_a - min_a)
            + w_r * _java_div(value2 - min_r, max_r - min_r)
            + w_t * _java_div(max_t - _java_div(1, value3), max_t - min_t)
        )

    def run_proposed(self) -> ExperimentMetrics:
        """运行主算法并输出平均指标。"""
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        self.res = self._new_population(self.pop_size)
        self.res_next = self._new_population(self.pop_size)
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
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
            # 为当前 DNN 重置本轮搜索状态，并记录当前已知最优解。
            best_value = [0.0, 0.0, -1.0]
            best_assignment = [0 for _ in range(self.max_dnn_num)]
            self.res = self._new_population(self.pop_size)
            self.res_next = self._new_population(self.pop_size)
            self.function1_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function2_values = [0.0 for _ in range(2 * self.pop_size)]
            self.function3_values = [0.0 for _ in range(2 * self.pop_size)]
            w_t = _java_div(max_t - self.env.ds[t].delay, max_t - min_t)
            w_a = _java_div(self.env.ds[t].preA - min_a, max_a - min_a)
            w_r = _java_div(self.env.ds[t].preR - min_r, max_r - min_r)
            o = 0
            index = 0
            best_gen = 0
            # 内层循环是当前 DNN 的进化搜索过程。
            while index <= self.iteration_limit:
                if o == 1:
                    break
                self.pareto_level_sort = [[0 for _ in range(2 * self.pop_size)] for _ in range(len(self.env.ds))]
                if index == 0:
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
                        continue
                # 把父代复制到 P+Q，再通过交叉和变异生成新的候选个体。
                self.res_pq = self._new_population(2 * self.pop_size)
                for m in range(len(self.res)):
                    for m1 in range(len(self.res[m])):
                        for m2 in range(len(self.res[m][m1])):
                            self.res_pq[m][m1][m2] = self.res[m][m1][m2]
                for m in range(self.pop_size):
                    a1 = int(random.random() * self.pop_size)
                    b1 = int(random.random() * self.pop_size)
                    self.cross_over1(t, a1, b1, m + self.pop_size)
                # 对联合种群逐个计算三目标值，并挑出本代最优候选。
                for m in range(len(self.res_pq)):
                    self.count_values1(t, m)
                    self.count_values2(t, m)
                    self.count_values3(t, m)
                bi = 0
                for i in range(2 * self.pop_size):
                    if self.function3_values[i] > 0 and self.function3_values[bi] < 0:
                        bi = i
                    elif self.function3_values[i] > 0:
                        if self.function3_values[bi] < 0 or self._score(self.function1_values[i], self.function2_values[i], self.function3_values[i], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t) > self._score(self.function1_values[bi], self.function2_values[bi], self.function3_values[bi], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t):
                            bi = i
                if (
                    (best_value[2] < 0 and self.function3_values[bi] > 0)
                    or (
                        self.function3_values[bi] > 0
                        and self._score(best_value[0], best_value[1], best_value[2], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t)
                        < self._score(self.function1_values[bi], self.function2_values[bi], self.function3_values[bi], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t)
                    )
                ):
                    for j in range(len(self.env.ds[t].tasks)):
                        best_assignment[j] = self.res_pq[bi][t][j]
                    best_value[0] = self.function1_values[bi]
                    best_value[1] = self.function2_values[bi]
                    best_value[2] = self.function3_values[bi]
                    best_gen = index
                # 使用非支配排序和拥挤距离从 P+Q 中筛出下一代父代。
                self.dominated_sort(t)
                self.get_res(t, self.res_pq, self.function1_values, self.function2_values, self.function3_values)
                self.update_res(self.res_pq, self.pareto_level_sort, self.distance, t)
                index += 1
            if o == 1:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            # 进化结束后，再对保留下来的父代种群做一次最终选择，
            # 防止历史 best 被后续更新漏掉。
            self.res_pq = self._new_population(self.pop_size)
            self.res_pq = self.res
            for m in range(len(self.res_pq)):
                self.count_values1(t, m)
                self.count_values2(t, m)
                self.count_values3(t, m)
            best_i = 0
            for i in range(self.pop_size):
                if self.function3_values[i] > 0 and self.function3_values[best_i] < 0:
                    best_i = i
                elif self.function3_values[i] > 0:
                    if self.function3_values[best_i] < 0 or self._score(self.function1_values[i], self.function2_values[i], self.function3_values[i], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t) > self._score(self.function1_values[best_i], self.function2_values[best_i], self.function3_values[best_i], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t):
                        best_i = i
            if (
                (best_value[2] < 0 and self.function3_values[best_i] > 0)
                or (
                    self.function3_values[best_i] > 0
                    and self._score(best_value[0], best_value[1], best_value[2], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t)
                    < self._score(self.function1_values[best_i], self.function2_values[best_i], self.function3_values[best_i], w_a, w_r, w_t, min_a, max_a, min_r, max_r, min_t, max_t)
                )
            ):
                for j in range(len(self.env.ds[t].tasks)):
                    best_assignment[j] = self.res_pq[best_i][t][j]
                best_value[0] = self.function1_values[best_i]
                best_value[1] = self.function2_values[best_i]
                best_value[2] = self.function3_values[best_i]
                best_gen = index
            _ = best_gen
            # 若当前最优解仍不满足时延收益条件，则本次接纳失败。
            if best_value[2] < 0:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            # 接纳成功后先把 DNN 放入运行集合，
            # 下一个时隙再统一更新负载、热度和动态可靠性。
            _, _, _, delay = self._evaluate_assignment(t, best_assignment)
            self._register_running_dnn(t, best_assignment, delay)
            t_res += delay
            r_res += best_value[1]
            a_res += best_value[0]
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
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
            f1, f2, f3, delay = self._evaluate_assignment(t, assignment)
            self._register_running_dnn(t, assignment, delay)
            t_res += delay
            r_res += f2
            a_res += f1
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        return ExperimentMetrics(
            avg_delay=t_res / success_dnn if success_dnn else 0.0,
            avg_operation=r_res / success_dnn if success_dnn else 0.0,
            avg_accuracy=a_res / success_dnn if success_dnn else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_max_resource(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行按剩余资源优先的基线算法。"""
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
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
            f1, f2, f3, delay = self._evaluate_assignment(t, r)
            _ = f3
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            a_res += f1
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_local_first(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        """运行按本地链路优先的基线算法。"""
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
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
            f1, f2, f3, delay = self._evaluate_assignment(t, r)
            _ = f3
            self._register_running_dnn(t, r, delay)
            t_res += delay
            r_res += f2
            a_res += f1
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        self._advance_until_drained()
        denominator = success_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )
