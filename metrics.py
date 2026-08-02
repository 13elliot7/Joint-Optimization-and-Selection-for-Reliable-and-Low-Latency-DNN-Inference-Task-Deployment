from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExperimentMetrics:
    """保存一次实验运行后的聚合指标。"""

    avg_delay: float
    avg_operational_stability_score: float
    avg_inference_fidelity_score: float
    avg_energy: float
    failure_count: int
    runtime_ms: int
    objective_semantics_version: str = "stability_fidelity_v2_return_energy"

    @property
    def avg_estimated_delay(self) -> float:
        """返回动态模型口径下的平均估计完成时延。"""
        return self.avg_delay

    @property
    def rejected_or_failed_count(self) -> int:
        """返回接纳失败或部署失败的任务数量。"""
        return self.failure_count

    @property
    def avg_total_energy(self) -> float:
        """返回平均总能耗。"""
        return self.avg_energy

    def to_report_rows(self) -> List[Tuple[str, float | int | str]]:
        """按动态模型语义生成统一输出项。"""
        return [
            ("avg_estimated_delay", self.avg_estimated_delay),
            ("avg_operational_stability_score", self.avg_operational_stability_score),
            ("avg_inference_fidelity_score", self.avg_inference_fidelity_score),
            ("objective_semantics_version", self.objective_semantics_version),
            ("avg_total_energy", self.avg_total_energy),
            ("rejected_or_failed_count", self.rejected_or_failed_count),
            ("runtime_ms", self.runtime_ms),
        ]
