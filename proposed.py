from __future__ import annotations

import math
import random
import time
from typing import List

from src_python.core.environment import Environment
from src_python.metrics import ExperimentMetrics
from src_python.models import DNN, LinkNode, Node


def _java_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        if numerator == 0:
            return math.nan
        return math.inf
    return numerator / denominator


class AllDNNRefactor:
    def __init__(self, env: Environment | None = None) -> None:
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
        return [
            [[0 for _ in range(self.max_dnn_num)] for _ in range(len(self.env.ds))]
            for _ in range(population_size)
        ]

    def _build_assignment_matrix(self, dnn_index: int, assignment: List[int]) -> List[List[int]]:
        task_count = len(self.env.ds[dnn_index].tasks)
        x = [[0 for _ in range(len(self.env.nodes))] for _ in range(task_count)]
        for task_idx in range(task_count):
            node_idx = assignment[task_idx]
            if node_idx < 0:
                raise IndexError("assignment index -1")
            x[task_idx][node_idx] = 1
        return x

    def mutate(self, i: int, m: int) -> None:
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
        x = [
            [[0 for _ in range(len(self.env.nodes))] for _ in range(len(self.env.ds[i].tasks))]
            for _ in range(i + 1)
        ]
        for j in range(len(self.env.ds[i].tasks)):
            x[i][j][self.res_pq[m][i][j]] = 1
        self.function1_values[m] = float(self.env.count_accuracy(self.env.ds[i], i, x))

    def count_values2(self, i: int, m: int) -> None:
        x = [
            [[0 for _ in range(len(self.env.nodes))] for _ in range(len(self.env.ds[i].tasks))]
            for _ in range(i + 1)
        ]
        for j in range(len(self.env.ds[i].tasks)):
            x[i][j][self.res_pq[m][i][j]] = 1
        self.function2_values[m] = float(self.env.count_operation(self.env.ds[i], i, x))

    def count_values3(self, i: int, m: int) -> None:
        x = [
            [[0 for _ in range(len(self.env.nodes))] for _ in range(len(self.env.ds[i].tasks))]
            for _ in range(i + 1)
        ]
        for j in range(len(self.env.ds[i].tasks)):
            x[i][j][self.res_pq[m][i][j]] = 1
        delay = self.env.count_delay(self.env.ds[i], x)
        if delay > self.env.ds[i].delay:
            self.function3_values[m] = float(self.env.ds[i].delay - delay)
            return
        self.function3_values[m] = float(1 / delay)

    def dominated_sort(self, o: int) -> None:
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
        return (
            w_a * _java_div(value1 - min_a, max_a - min_a)
            + w_r * _java_div(value2 - min_r, max_r - min_r)
            + w_t * _java_div(max_t - _java_div(1, value3), max_t - min_t)
        )

    def run_proposed(self) -> ExperimentMetrics:
        max_a = 1.0
        max_r = 1.0
        max_t = 2000.0
        min_a = 0.0
        min_r = 0.0
        min_t = 200.0
        self.res = self._new_population(self.pop_size)
        self.res_next = self._new_population(self.pop_size)
        t = 0
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        failure_dnn = 0
        run_time = time.monotonic()
        while t < self.env.t_max:
            best_res = [[0 for _ in range(self.max_dnn_num)] for _ in range(len(self.env.ds))]
            best_value = [0.0, 0.0, -1.0]
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
            while index <= self.iteration_limit:
                if o == 1:
                    break
                self.pareto_level_sort = [[0 for _ in range(2 * self.pop_size)] for _ in range(len(self.env.ds))]
                if index == 0:
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
                self.res_pq = self._new_population(2 * self.pop_size)
                for m in range(len(self.res)):
                    for m1 in range(len(self.res[m])):
                        for m2 in range(len(self.res[m][m1])):
                            self.res_pq[m][m1][m2] = self.res[m][m1][m2]
                for m in range(self.pop_size):
                    a1 = int(random.random() * self.pop_size)
                    b1 = int(random.random() * self.pop_size)
                    self.cross_over1(t, a1, b1, m + self.pop_size)
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
                    for i in range(len(self.env.ds)):
                        for j in range(len(self.env.ds[t].tasks)):
                            best_res[i][j] = self.res_pq[bi][i][j]
                    best_value[0] = self.function1_values[bi]
                    best_value[1] = self.function2_values[bi]
                    best_value[2] = self.function3_values[bi]
                    best_gen = index
                self.dominated_sort(t)
                self.get_res(t, self.res_pq, self.function1_values, self.function2_values, self.function3_values)
                self.update_res(self.res_pq, self.pareto_level_sort, self.distance, t)
                index += 1
            if o == 1:
                t += 1
                failure_dnn += 1
                continue
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
                for i in range(len(self.env.ds)):
                    for j in range(len(self.env.ds[t].tasks)):
                        best_res[i][j] = self.res_pq[best_i][i][j]
                best_value[0] = self.function1_values[best_i]
                best_value[1] = self.function2_values[best_i]
                best_value[2] = self.function3_values[best_i]
                best_gen = index
            _ = best_gen
            if best_value[2] < 0:
                t += 1
                failure_dnn += 1
                continue
            self.env.release_resource(self.env.ds[t], best_res[t])
            t_res += _java_div(1, best_value[2])
            r_res += best_value[1]
            a_res += best_value[0]
            t += 1
        denominator = self.env.t_max - failure_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def max_resource_placement1(self, r: List[int], dnn: DNN, i: int, node_list: List[Node]) -> bool:
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
        return self.env.count_delay_random1(dnn, x, index1)

    def run_random(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        failure_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        t = 0
        while t < self.env.t_max:
            suiji = [[[0 for _ in range(self.max_dnn_num)] for _ in range(len(self.env.ds))] for _ in range(1)]
            for value_idx in range(self.max_dnn_num):
                suiji[0][t][value_idx] = -1
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
                    if suiji[0][t][p] != -1:
                        xs[p][suiji[0][t][p]] = 1
                    if self.env.check_resource(self.env.ds[t], xs) and (self.env.nodes[n].level != 1 or n == self.env.ds[t].initiateNode):
                        p += 1
                    else:
                        xs[p][n] = 0
                if self.env.check_delay_random(self.env.ds[t], xs):
                    for p in range(len(self.env.ds[t].tasks)):
                        for n in range(len(self.env.nodes)):
                            if xs[p][n] == 1:
                                suiji[0][t][p] = n
                    break
            if o == 1:
                t += 1
                failure_dnn += 1
                continue
            f1 = self.env.count_values1_random(t, 0, suiji)
            f2 = self.env.count_values2_random(t, 0, suiji)
            f3 = self.env.count_values3_random(t, 0, suiji)
            self.env.release_resource(self.env.ds[t], suiji[0][t])
            t_res += _java_div(1, f3)
            r_res += f2
            a_res += f1
            t += 1
        return ExperimentMetrics(
            avg_delay=t_res / self.env.t_max if self.env.t_max else 0.0,
            avg_operation=r_res / self.env.t_max if self.env.t_max else 0.0,
            avg_accuracy=a_res / self.env.t_max if self.env.t_max else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_max_resource(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        failure_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        node_list = self.env.clone_nodes()
        t = 0
        while t < self.env.t_max:
            r = [0 for _ in range(len(self.env.ds[t].tasks))]
            self.start_time = time.monotonic()
            if self.max_resource_placement(r, self.env.ds[t], 0, node_list):
                pass
            else:
                self.start_time = time.monotonic()
                if not self.max_resource_placement1(r, self.env.ds[t], 0, node_list):
                    failure_dnn += 1
                    t += 1
                    continue
            suiji = [[[0 for _ in range(len(r))] for _ in range(len(self.env.ds))] for _ in range(1)]
            for idx, value in enumerate(r):
                suiji[0][t][idx] = value
            f1 = self.env.count_values1_random(t, 0, suiji)
            f2 = self.env.count_values2_random(t, 0, suiji)
            f3 = self.env.count_values3_random(t, 0, suiji)
            self.env.release_resource(self.env.ds[t], suiji[0][t])
            t_res += _java_div(1, f3)
            r_res += f2
            a_res += f1
            t += 1
        denominator = self.env.t_max - failure_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )

    def run_local_first(self, reset_snapshot: List[Node]) -> ExperimentMetrics:
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        failure_dnn = 0
        self.env.reset_nodes(reset_snapshot)
        t = 0
        while t < self.env.t_max:
            r = [0 for _ in range(len(self.env.ds[t].tasks))]
            self.start_time = time.monotonic()
            if self.location_placement(r, self.env.ds[t], 0, self.env.nodes, self.env.nodes[self.env.ds[t].initiateNode]):
                pass
            else:
                self.start_time = time.monotonic()
                if not self.location_placement1(r, self.env.ds[t], 0, self.env.nodes, self.env.nodes[self.env.ds[t].initiateNode]):
                    failure_dnn += 1
                    t += 1
                    continue
            suiji = [[[0 for _ in range(len(r))] for _ in range(len(self.env.ds))] for _ in range(1)]
            for idx, value in enumerate(r):
                suiji[0][t][idx] = value
            f1 = self.env.count_values1_random(t, 0, suiji)
            f2 = self.env.count_values2_random(t, 0, suiji)
            f3 = self.env.count_values3_random(t, 0, suiji)
            self.env.release_resource(self.env.ds[t], suiji[0][t])
            t_res += _java_div(1, f3)
            r_res += f2
            a_res += f1
            t += 1
        denominator = self.env.t_max - failure_dnn
        return ExperimentMetrics(
            avg_delay=t_res / denominator if denominator else 0.0,
            avg_operation=r_res / denominator if denominator else 0.0,
            avg_accuracy=a_res / denominator if denominator else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )
