from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    M: int = 11
    r_min: float = 0.70
    r_max: float = 0.99
    D_bug: float = 1000.0
    D_Max: float = 100.0
    lambda_a: float = 1.0
    lambda_r: float = 10.0
    T: int = 1000
    H_off: int = 100
    I_range: tuple[int, int] = (5000, 10000)
    C_range: tuple[float, float] = (0.15, 0.20)
    alpha_1_range: tuple[float, float] = (0.45, 0.65)
    alpha2_range: tuple[float, float] = (0.75, 1.0)
    k_range: tuple[float, float] = (2e3, 4e3)
    G_range: tuple[float, float] = (2e-7, 5e-7)
    V_range: float = 200.0
    shape: float = 1.5
    scale: float = 100.0
    mean: float = 2.5
    sigma: float = 0.8
    tMax: int = 4

