from __future__ import annotations

import math
import time
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
        self.q = 0.0
        self.decision_round = 0
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

    def _task_delay_budget(self, dnn_index: int) -> float:
        """返回单个任务在虚拟队列更新中使用的时延预算。"""
        task_count = max(len(self.env.ds[dnn_index].tasks), 1)
        dnn_budget = min(float(self.d_bug), float(self.env.ds[dnn_index].delay))
        return dnn_budget / task_count

    def step(self, dnn_index: int, task_index: int, x: List[List[int]], a_j: List[float], rj_t: List[int]) -> List[int] | None:
        """为当前任务选择一个满足约束的节点集合。"""
        self.decision_round += 1
        self.update_rj_tilde(self.decision_round)
        d_j = self.env.calculate_task_delay_costs(dnn_index, task_index, x)
        xij_t = self.selector.select_multiple(a_j, self.rj_tilde, self.q, d_j, self.env.ds[dnn_index].delay)
        xj_t = [0 for _ in range(self.m)]
        selected_i = 10
        for i in range(10):
            candidate_nodes = [j for j in range(self.m) if xij_t[i][j] == 1]
            if not candidate_nodes or any(rj_t[j] != 1 for j in candidate_nodes):
                continue
            for j in range(self.m):
                if xij_t[i][j] == 1:
                    x[task_index][j] = 1
            if self.env.check_resource(self.env.ds[dnn_index], x):
                xj_t = xij_t[i]
                selected_i = i
                break
            for j in candidate_nodes:
                x[task_index][j] = 0
        if selected_i == 10 or xj_t == [0 for _ in range(self.m)]:
            return None
        d_t = 0.0
        for j in range(self.m):
            if xj_t[j] == 1 and rj_t[j] == 1:
                d_t += d_j[j]
        task_budget = max(self._task_delay_budget(dnn_index), 1.0)
        self.q = max(self.q + d_t / task_budget - 1.0, 0.0)
        for j in range(self.m):
            hj_prev = self.hj[j]
            self.hj[j] += xj_t[j]
            hj_plus_hj = self.h_off + self.hj[j]
            self.rj_bar[j] = (((hj_prev + self.h_off) / hj_plus_hj) * self.rj_bar[j] + rj_t[j] * xj_t[j] / hj_plus_hj)
        return xj_t


class RTBLRunner:
    def __init__(self, env: Environment | None = None, config: SimulationConfig | None = None) -> None:
        """构建 RTBL 实验运行器及其环境。"""
        self.config = config or SimulationConfig()
        if env is None:
            self.env = Environment(
                t_max=self.config.tMax,
                max_dnn_num=20,
                availability_node_count=self.config.M,
                shape=self.config.shape,
                scale=self.config.scale,
                mean=self.config.mean,
                sigma=self.config.sigma,
                h_off=self.config.H_off,
            )
        else:
            self.env = env
            if self.config.M > len(self.env.nodes):
                raise ValueError("RTBL config.M cannot exceed the number of environment nodes")
            if self.config.M > self.env.availability_node_count:
                raise ValueError("RTBL config.M cannot exceed environment availability_node_count")
        self.scheduler = RTBLScheduler(self.config, self.env.rj_history, self.env)

    def _reset_scheduler(self) -> None:
        """重置 RTBL 在线估计状态，不改变节点选择规则。"""
        self.scheduler = RTBLScheduler(self.config, self.env.rj_history, self.env)

    def _build_assignment(self, dnn_index: int) -> List[int] | None:
        """按原 RTBL 逐任务逻辑生成当前 DNN 的部署向量。"""
        task_num = len(self.env.ds[dnn_index].tasks)
        x = [[0 for _ in range(len(self.env.nodes))] for _ in range(task_num)]
        assignment = [-1 for _ in range(self.env.max_dnn_num)]
        a_j = [0.0 for _ in range(self.config.M)]
        rj_t = self.env.generate_availability()
        for j in range(self.config.M):
            a_j[j] = self.env.nodes[j].inference_fidelity
        for task_index in range(task_num):
            x_i = self.scheduler.step(dnn_index, task_index, x, a_j, rj_t)
            if x_i is None:
                return None
            for node_index in range(self.config.M):
                if x_i[node_index] == 1:
                    x[task_index][node_index] = 1
                    assignment[task_index] = node_index
        if not self.env.check_delay_random(self.env.ds[dnn_index], x):
            return None
        return assignment

    def run_dynamic(self, reset_snapshot=None) -> ExperimentMetrics:
        """在共享动态环境中运行 RTBL，复用原节点选择逻辑。"""
        run_time = time.monotonic()
        t_res = 0.0
        r_res = 0.0
        a_res = 0.0
        e_res = 0.0
        failure_dnn = 0
        success_dnn = 0
        if reset_snapshot is not None:
            self.env.reset_nodes(reset_snapshot)
        self._reset_scheduler()
        next_dnn_index = 0
        while next_dnn_index < self.env.t_max or self.env.running_dnns:
            if next_dnn_index >= self.env.t_max:
                self.env.advance_time_slot()
                continue
            dnn_index = next_dnn_index
            self.env.print_pending_dnn_info(dnn_index)
            assignment = self._build_assignment(dnn_index)
            if assignment is None:
                failure_dnn += 1
                next_dnn_index += 1
                self.env.advance_time_slot()
                continue
            assignment_slice = assignment[: len(self.env.ds[dnn_index].tasks)]
            post_admission = self.env.evaluate_post_admission_metrics(
                dnn_index,
                assignment_slice,
            )
            delay = post_admission.delay
            operational_stability = post_admission.operational_stability
            inference_fidelity = post_admission.inference_fidelity
            total_energy = post_admission.total_energy
            remaining_slots = max(1, math.ceil(delay / self.env.slot_length))
            self.env.add_running_dnn(
                dnn_index=dnn_index,
                assignment=assignment_slice,
                estimated_runtime=delay,
                remaining_slots=remaining_slots,
            )
            t_res += delay
            r_res += operational_stability
            a_res += inference_fidelity
            e_res += total_energy
            success_dnn += 1
            next_dnn_index += 1
            self.env.advance_time_slot()
        while self.env.running_dnns:
            self.env.advance_time_slot()
        return ExperimentMetrics(
            avg_delay=t_res / success_dnn if success_dnn else 0.0,
            avg_operational_stability_score=r_res / success_dnn if success_dnn else 0.0,
            avg_inference_fidelity_score=a_res / success_dnn if success_dnn else 0.0,
            avg_energy=e_res / success_dnn if success_dnn else 0.0,
            failure_count=failure_dnn,
            runtime_ms=int((time.monotonic() - run_time) * 1000),
        )
