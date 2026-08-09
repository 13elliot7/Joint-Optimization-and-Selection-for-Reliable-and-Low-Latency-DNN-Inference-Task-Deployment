from __future__ import annotations

from typing import Iterable, List, Set


class SOTASelector:
    def __init__(self, m: int, lambda_r: float, v: float) -> None:
        """初始化 RTBL 中的单步候选节点选择器。"""
        self.m = m
        self.lambda_r = lambda_r
        self.v = v

    def compute_f1(self, selected: Set[int], rj_tilde: List[float]) -> float:
        """计算已选节点集合的冗余可用性收益。"""
        if not selected:
            return 0.0
        availability_complement = 1.0
        for j in selected:
            availability_complement *= 1 - rj_tilde[j]
        availability = 1 - availability_complement
        return self.v * self.lambda_r * availability

    def compute_f2(self, selected: Iterable[int], q: float, d_j: List[float]) -> float:
        """计算已选节点集合带来的时延代价。"""
        result = 0.0
        for j in selected:
            result += q * d_j[j]
        return result

    def select(self, rj_tilde: List[float], q: float, d_j: List[float], d_max: float, remaining: Set[int]) -> List[int]:
        """从候选节点中选择一组满足收益条件的部署位置。"""
        selected: Set[int] = set()
        if not remaining:
            return [0 for _ in range(self.m)]
        best_u = None
        is_zero_cost = False
        best_ratio = float("-inf")
        best_gain = float("-inf")
        for u in remaining:
            plus_u = set(selected)
            plus_u.add(u)
            gain = self.compute_f1(plus_u, rj_tilde) - self.compute_f1(selected, rj_tilde)
            cost = self.compute_f2({u}, q, d_j)
            if cost == 0:
                if gain > best_gain:
                    best_gain = gain
                    is_zero_cost = True
                    best_u = u
            elif not is_zero_cost:
                ratio = gain / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_u = u
        if best_u is not None:
            plus_best = set(selected)
            plus_best.add(best_u)
            gain = self.compute_f1(plus_best, rj_tilde) - self.compute_f1(selected, rj_tilde)
            cost = self.compute_f2({best_u}, q, d_j)
            if gain - cost >= 0 and d_j[best_u] <= d_max:
                selected.add(best_u)
        return [1 if j in selected else 0 for j in range(self.m)]

    def select_multiple(self, rj_tilde: List[float], q: float, d_j: List[float], d_max: float) -> List[List[int]]:
        """连续生成多组互不重复的候选选择结果。"""
        remaining = set(range(self.m))
        xij_t = [[0 for _ in range(self.m)] for _ in range(10)]
        for i in range(10):
            xij_t[i] = self.select(rj_tilde, q, d_j, d_max, remaining)
            for j in range(self.m):
                if xij_t[i][j] == 1:
                    remaining.discard(j)
        return xij_t
