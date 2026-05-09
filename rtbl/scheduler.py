from __future__ import annotations

import math
from typing import List

from core.environment import Environment
from metrics import ExperimentMetrics
from rtbl.config import SimulationConfig
from rtbl.sota_selector import SOTASelector


class RTBLScheduler:
    def __init__(self, config: SimulationConfig, rj_history: List[List[int]], env: Environment) -> None:
        """初始化 RTBL 调度器的在线估计状态。"""
        self.config = config
        self.env = env
        self.m = config.M
        self.v = config.V_range
        self.lambda_a = config.lambda_a
        self.lambda_r = config.lambda_r
        self.r_min = config.r_min
        self.r_max = config.r_max
        self.d_bug = config.D_bug
        self.h_off = config.H_off
        self.d_max = config.D_Max
        self.q = 0.0
        self.hj = [0 for _ in range(self.m)]
        self.rj_bar = self._initialize_rj_bar(rj_history)
        self.rj_tilde = [self.r_min for _ in range(self.m)]
        self.selector = SOTASelector(self.m, self.lambda_a, self.lambda_r, self.v)

    def _initialize_rj_bar(self, rj_history: List[List[int]]) -> List[float]:
        """根据历史可用性样本初始化经验均值。"""
        result = [0.0 for _ in range(self.m)]
        for j in range(self.m):
            result[j] = sum(rj_history[j]) / self.h_off
        return result

    def update_rj_tilde(self, t: int) -> None:
        """按 UCB 风格公式更新保守可用性估计。"""
        r0 = self.r_max - self.r_min
        for j in range(self.m):
            if self.hj[j] > 0:
                exploration = r0 * math.sqrt(2 * math.log(t + 1) / (self.h_off + self.hj[j]))
                self.rj_tilde[j] = max(self.rj_bar[j] - exploration, self.r_min)
            else:
                self.rj_tilde[j] = self.r_min

    def reset_q(self) -> None:
        """重置当前 DNN 的排队代价累积量。"""
        self.q = 0.0

    def step(self, dnn_index: int, task_index: int, x: List[List[int]], a_j: List[float], rj_t: List[int]) -> List[int] | None:
        """为当前任务选择一个满足约束的节点集合。"""
        self.update_rj_tilde(task_index)
        d_j = self.env.calculate_task_delay_costs(dnn_index, task_index, x)
        xij_t = self.selector.select_multiple(a_j, self.rj_tilde, self.q, d_j, self.env.ds[dnn_index].getDelay())
        xj_t = [0 for _ in range(self.m)]
        selected_i = 10
        for i in range(10):
            for j in range(self.m):
                if xij_t[i][j] == 1 and rj_t[j] == 1:
                    x[task_index][j] = 1
            if self.env.check_resource(self.env.ds[dnn_index], x):
                xj_t = xij_t[i]
                selected_i = i
                break
        if selected_i == 10 or xj_t == [0 for _ in range(self.m)]:
            return None
        d_t = 0.0
        for j in range(self.m):
            if xj_t[j] == 1 and rj_t[j] == 1:
                d_t += d_j[j]
        self.q = max(self.q - self.d_bug + d_t, 0.0)
        for j in range(self.m):
            hj_prev = self.hj[j]
            self.hj[j] += xj_t[j]
            hj_plus_hj = self.h_off + self.hj[j]
            self.rj_bar[j] = (((hj_prev + self.h_off) / hj_plus_hj) * self.rj_bar[j] + rj_t[j] * xj_t[j] / hj_plus_hj)
        return xj_t


class RTBLRunner:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        """构建 RTBL 实验运行器及其环境。"""
        self.config = config or SimulationConfig()
        self.env = Environment(
            t_max=self.config.tMax,
            max_dnn_num=20,
            availability_node_count=self.config.M,
            shape=self.config.shape,
            scale=self.config.scale,
            mean=self.config.mean,
            sigma=self.config.sigma,
            h_off=self.config.H_off,
            k_range=self.config.k_range,
            i_range=self.config.I_range,
            c_range=self.config.C_range,
            alpha_1_range=self.config.alpha_1_range,
            alpha2_range=self.config.alpha2_range,
        )
        self.scheduler = RTBLScheduler(self.config, self.env.rj_history, self.env)

    def run(self) -> ExperimentMetrics:
        """按 RTBL 逻辑依次部署全部 DNN 并统计结果。"""
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        failure_dnn = 0
        run_time = __import__("time").monotonic()
        for t in range(self.env.t_max):
            self.scheduler.reset_q()
            task_num = len(self.env.ds[t].tasks)
            x = [[0 for _ in range(len(self.env.nodes))] for _ in range(task_num)]
            suiji = [[[0 for _ in range(self.env.max_dnn_num)] for _ in range(len(self.env.ds))] for _ in range(1)]
            for idx in range(self.env.max_dnn_num):
                suiji[0][t][idx] = -1
            a_j = [0.0 for _ in range(self.config.M)]
            rj_t = self.env.generate_availability()
            for j in range(self.config.M):
                a_j[j] = self.env.nodes[j].a_reliability
            success_deploy = True
            for i in range(task_num):
                x_i = self.scheduler.step(t, i, x, a_j, rj_t)
                if x_i is None:
                    success_deploy = False
                    break
                for j in range(self.config.M):
                    if x_i[j] == 1:
                        x[i][j] = 1
            if not success_deploy:
                failure_dnn += 1
                continue
            if self.env.check_delay_random(self.env.ds[t], x):
                for p in range(task_num):
                    for j in range(self.config.M):
                        if x[p][j] == 1:
                            suiji[0][t][p] = j
            f1 = self.env.count_values1_random(t, 0, suiji)
            f2 = self.env.count_values2_random(t, 0, suiji)
            f3 = self.env.count_values3_random(t, 0, suiji)
            self.env.release_resource(self.env.ds[t], suiji[0][t])
            t_res += 1 / f3
            r_res += f2
            a_res += f1
        return ExperimentMetrics(
            avg_delay=t_res / self.env.t_max if self.env.t_max else 0.0,
            avg_operation=r_res / self.env.t_max if self.env.t_max else 0.0,
            avg_accuracy=a_res / self.env.t_max if self.env.t_max else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((__import__("time").monotonic() - run_time) * 1000),
        )
