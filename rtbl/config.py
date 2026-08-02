from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    M: int = 11
    r_min: float = 0.70
    r_max: float = 0.99
    D_bug: float = 1000.0
    lambda_a: float = 1.0
    lambda_r: float = 10.0
    H_off: int = 100
    V_range: float = 200.0
    shape: float = 1.5
    scale: float = 100.0
    mean: float = 2.5
    sigma: float = 0.8
    tMax: int = 4
