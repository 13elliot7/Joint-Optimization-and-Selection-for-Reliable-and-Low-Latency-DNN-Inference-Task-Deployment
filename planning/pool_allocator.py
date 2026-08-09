from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PoolDemand:
    profile_id: str
    origin_group: int
    arrival_rate: float
    hit_failure_risk: float = 0.0
    availability_risk: float = 0.0
    diversity_need: float = 0.0

    def __post_init__(self) -> None:
        if not self.profile_id or self.origin_group < 0 or self.arrival_rate < 0.0:
            raise ValueError("invalid pool demand key or arrival rate")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.hit_failure_risk,
                self.availability_risk,
                self.diversity_need,
            )
        ):
            raise ValueError("pool demand risks must be in [0, 1]")

    @property
    def key(self) -> tuple[str, int]:
        return self.profile_id, self.origin_group

    @property
    def score(self) -> float:
        return self.arrival_rate * (1.0 + self.hit_failure_risk) * (
            1.0 + self.availability_risk
        ) * (1.0 + self.diversity_need)


class PoolBudgetAllocator:
    """使用最小配额和最大余数法确定性分配候选池预算。"""

    def __init__(self, minimum_per_key: int = 2, maximum_per_key: int = 40) -> None:
        if minimum_per_key <= 0 or maximum_per_key < minimum_per_key:
            raise ValueError("invalid per-key pool bounds")
        self.minimum_per_key = minimum_per_key
        self.maximum_per_key = maximum_per_key

    def allocate(
        self,
        total_budget: int,
        demands: tuple[PoolDemand, ...],
        mode: str = "demand_weighted",
    ) -> tuple[tuple[str, int, int], ...]:
        unique = {demand.key: demand for demand in demands}
        ordered = [unique[key] for key in sorted(unique)]
        if not ordered:
            return ()
        minimum_total = len(ordered) * self.minimum_per_key
        maximum_total = len(ordered) * self.maximum_per_key
        if not minimum_total <= total_budget <= maximum_total:
            raise ValueError("total budget cannot satisfy per-key bounds")
        allocation = {demand.key: self.minimum_per_key for demand in ordered}
        remaining = total_budget - minimum_total
        while remaining > 0:
            eligible = [
                demand
                for demand in ordered
                if allocation[demand.key] < self.maximum_per_key
            ]
            if not eligible:
                break
            if mode == "uniform":
                scores = {demand.key: 1.0 for demand in eligible}
            elif mode == "demand_weighted":
                scores = {demand.key: max(demand.score, 0.0) for demand in eligible}
                if sum(scores.values()) <= 0.0:
                    scores = {demand.key: 1.0 for demand in eligible}
            else:
                raise ValueError("unsupported pool allocation mode")
            score_sum = sum(scores.values())
            ideal = {
                key: remaining * score / score_sum for key, score in scores.items()
            }
            floor_grants = {
                key: min(
                    math.floor(value),
                    self.maximum_per_key - allocation[key],
                )
                for key, value in ideal.items()
            }
            granted = sum(floor_grants.values())
            for key, grant in floor_grants.items():
                allocation[key] += grant
            remaining -= granted
            if remaining <= 0:
                break
            ranked = sorted(
                eligible,
                key=lambda demand: (
                    -(ideal[demand.key] - math.floor(ideal[demand.key])),
                    demand.profile_id,
                    demand.origin_group,
                ),
            )
            progress = False
            for demand in ranked:
                if remaining <= 0:
                    break
                if allocation[demand.key] >= self.maximum_per_key:
                    continue
                allocation[demand.key] += 1
                remaining -= 1
                progress = True
            if not progress:
                raise RuntimeError("pool allocation made no progress")
        return tuple(
            (profile_id, origin_group, allocation[(profile_id, origin_group)])
            for profile_id, origin_group in sorted(allocation)
        )

    def allocate_smoothed(
        self,
        total_budget: int,
        demands: tuple[PoolDemand, ...],
        previous: tuple[tuple[str, int, int], ...],
        mode: str = "demand_weighted",
        smoothing_alpha: float = 0.5,
        max_change_ratio: float = 0.5,
    ) -> tuple[tuple[str, int, int], ...]:
        """在保持整数总量的同时平滑相邻控制周期的分键配额。"""
        if not 0.0 <= smoothing_alpha <= 1.0 or max_change_ratio < 0.0:
            raise ValueError("invalid pool smoothing configuration")
        raw = self.allocate(total_budget, demands, mode)
        if not previous:
            return raw
        raw_map = {(profile, origin): value for profile, origin, value in raw}
        previous_map = {
            (profile, origin): value for profile, origin, value in previous
        }
        allocation: dict[tuple[str, int], int] = {}
        lower: dict[tuple[str, int], int] = {}
        upper: dict[tuple[str, int], int] = {}
        for key in sorted(raw_map):
            old = previous_map.get(key, self.minimum_per_key)
            lower[key] = max(
                self.minimum_per_key,
                math.floor(old * (1.0 - max_change_ratio)),
            )
            upper[key] = min(
                self.maximum_per_key,
                math.ceil(old * (1.0 + max_change_ratio)),
            )
            blended = round(
                smoothing_alpha * raw_map[key] + (1.0 - smoothing_alpha) * old
            )
            allocation[key] = min(upper[key], max(lower[key], blended))

        def adjust(delta: int, respect_change_limit: bool) -> int:
            while delta != 0:
                if delta > 0:
                    candidates = [
                        key
                        for key in allocation
                        if allocation[key]
                        < (upper[key] if respect_change_limit else self.maximum_per_key)
                    ]
                    candidates.sort(
                        key=lambda key: (-(raw_map[key] - allocation[key]), key)
                    )
                    step = 1
                else:
                    candidates = [
                        key
                        for key in allocation
                        if allocation[key]
                        > (lower[key] if respect_change_limit else self.minimum_per_key)
                    ]
                    candidates.sort(
                        key=lambda key: (-(allocation[key] - raw_map[key]), key)
                    )
                    step = -1
                if not candidates:
                    break
                for key in candidates:
                    if delta == 0:
                        break
                    allocation[key] += step
                    delta -= step
            return delta

        difference = total_budget - sum(allocation.values())
        difference = adjust(difference, respect_change_limit=True)
        # 动作总预算变化过大时，守恒优先于变化率限制，但仍保持每键硬上下界。
        if difference:
            difference = adjust(difference, respect_change_limit=False)
        if difference:
            raise ValueError("cannot conserve total pool budget after smoothing")
        return tuple(
            (profile, origin, allocation[(profile, origin)])
            for profile, origin in sorted(allocation)
        )
