from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import List


_physical_link_id_sequence = count()


def _next_physical_link_id() -> int:
    """返回进程内唯一的物理链路标识。"""
    return next(_physical_link_id_sequence)


@dataclass(eq=False)
class Node:
    cpu: int
    max_cpu: int
    level: int
    operational_stability: float
    inference_fidelity: float
    float_rate: float
    base_operational_stability: float | None = None
    base_inference_fidelity: float | None = None
    load_ratio: float = 0.0
    heat: float = 0.0
    comp_power: float = 1.0

    def __post_init__(self) -> None:
        """补齐状态相关经验质量分数的基线字段。"""
        if self.base_operational_stability is None:
            self.base_operational_stability = self.operational_stability
        if self.base_inference_fidelity is None:
            self.base_inference_fidelity = self.inference_fidelity

    def clone(self) -> "Node":
        """复制节点对象及其动态状态。"""
        return Node(
            cpu=self.cpu,
            max_cpu=self.max_cpu,
            level=self.level,
            operational_stability=self.operational_stability,
            inference_fidelity=self.inference_fidelity,
            float_rate=self.float_rate,
            base_operational_stability=self.base_operational_stability,
            base_inference_fidelity=self.base_inference_fidelity,
            load_ratio=self.load_ratio,
            heat=self.heat,
            comp_power=self.comp_power,
        )


@dataclass(eq=False)
class Task:
    cpu_need: int
    float_num: float


@dataclass(eq=False)
class PhysicalLinkState:
    """由一对有向传输边共享的物理链路稳定性状态。"""

    node_a: Node
    node_b: Node
    physical_link_id: int = field(default_factory=_next_physical_link_id)
    base_transmission_stability: float = 1.0
    transmission_stability: float = 1.0
    load_ratio: float = 0.0
    heat: float = 0.0


@dataclass(frozen=True)
class CandidateQualityScores:
    """候选部署接纳后一步预测得到的无量纲质量分数。"""

    operational_stability: float
    inference_fidelity: float
    node_stability: float
    link_stability: float | None
    raw_joint_product: float
    used_node_count: int
    used_physical_link_count: int


@dataclass(frozen=True)
class PostAdmissionMetrics:
    """候选部署的接纳后预测指标。"""

    delay: float
    operational_stability: float
    inference_fidelity: float
    total_energy: float
    raw_joint_product: float

@dataclass(eq=False)
class LinkNode:
    """有向传输边；稳定性字段代理到共享的物理链路状态。"""

    s_node: Node
    e_node: Node
    band_width: int
    base_band_width: int | None = None
    effective_band_width: float | None = None
    load_ratio: float = 0.0
    energy_per_mb: float = 0.0
    physical_state: PhysicalLinkState | None = None

    def __post_init__(self) -> None:
        """在有向链路创建后补齐带宽和共享物理状态。"""
        if self.base_band_width is None:
            self.base_band_width = self.band_width
        if self.effective_band_width is None:
            self.effective_band_width = float(self.band_width)
        if self.physical_state is None:
            self.physical_state = PhysicalLinkState(self.s_node, self.e_node)

    @property
    def physical_link_id(self) -> int:
        """返回该方向边所属的物理链路 ID。"""
        assert self.physical_state is not None
        return self.physical_state.physical_link_id

    @property
    def heat(self) -> float:
        """兼容旧接口：返回共享物理链路热度。"""
        assert self.physical_state is not None
        return self.physical_state.heat

    @heat.setter
    def heat(self, value: float) -> None:
        assert self.physical_state is not None
        self.physical_state.heat = value

@dataclass(eq=False)
class RunningDNN:
    dnn_index: int
    assignment: List[int]
    estimated_runtime: float
    remaining_slots: int


@dataclass(eq=False)
class LinkDNN:
    s_task: Task
    e_task: Task
    float_tran: int

@dataclass(eq=False)
class DNN:
    tasks: List[Task]
    links: List[LinkDNN]
    delay: int
    initiateNode: int
    startFloat: float = 0.0
    preference_stability: float = 0.0
    preference_fidelity: float = 0.0
    backFloat: float = 0.0
