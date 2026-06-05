from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExperimentMetrics:
    """保存一次实验运行后的聚合指标。"""

    avg_delay: float
    avg_operation: float
    avg_accuracy: float
    avg_energy: float
    failure_count: int
    runtime_ms: int

    @property
    def avg_estimated_delay(self) -> float:
        """返回动态模型口径下的平均估计完成时延。"""
        return self.avg_delay

    @property
    def avg_dynamic_operation_reliability(self) -> float:
        """返回动态模型口径下的平均运行可靠性。"""
        return self.avg_operation

    @property
    def avg_dynamic_accuracy_reliability(self) -> float:
        """返回动态模型口径下的平均精度可靠性。"""
        return self.avg_accuracy

    @property
    def rejected_or_failed_count(self) -> int:
        """返回接纳失败或部署失败的任务数量。"""
        return self.failure_count

    @property
    def avg_total_energy(self) -> float:
        """返回平均总能耗。"""
        return self.avg_energy

    def to_report_rows(self) -> List[Tuple[str, float | int]]:
        """按动态模型语义生成统一输出项。"""
        return [
            ("avg_estimated_delay", self.avg_estimated_delay),
            ("avg_dynamic_operation_reliability", self.avg_dynamic_operation_reliability),
            ("avg_dynamic_accuracy_reliability", self.avg_dynamic_accuracy_reliability),
            ("avg_total_energy", self.avg_total_energy),
            ("rejected_or_failed_count", self.rejected_or_failed_count),
            ("runtime_ms", self.runtime_ms),
        ]
