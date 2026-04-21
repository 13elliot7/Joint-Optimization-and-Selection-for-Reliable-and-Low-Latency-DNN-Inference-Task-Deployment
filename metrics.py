from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentMetrics:
    avg_delay: float
    avg_operation: float
    avg_accuracy: float
    failure_count: int
    runtime_ms: int

