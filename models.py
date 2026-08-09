from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count
from typing import List, Literal


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
    float_rate: float
    base_operational_stability: float | None = None
    load_ratio: float = 0.0
    heat: float = 0.0
    comp_power: float = 1.0
    availability_mode: Literal["stochastic", "always_on"] = "stochastic"

    def __post_init__(self) -> None:
        """补齐状态相关经验质量分数的基线字段。"""
        if self.base_operational_stability is None:
            self.base_operational_stability = self.operational_stability

    def clone(self) -> "Node":
        """复制节点对象及其动态状态。"""
        return Node(
            cpu=self.cpu,
            max_cpu=self.max_cpu,
            level=self.level,
            operational_stability=self.operational_stability,
            float_rate=self.float_rate,
            base_operational_stability=self.base_operational_stability,
            load_ratio=self.load_ratio,
            heat=self.heat,
            comp_power=self.comp_power,
            availability_mode=self.availability_mode,
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
    preference_delay: float = 0.0
    preference_energy: float = 0.0
    backFloat: float = 0.0
    profile_id: str | None = None


@dataclass(frozen=True)
class TaskProfile:
    """可跨请求复用的不可变 DNN 子任务描述。"""

    cpu_need: int
    float_num: float

    def __post_init__(self) -> None:
        if self.cpu_need <= 0:
            raise ValueError("cpu_need must be positive")
        if self.float_num < 0.0:
            raise ValueError("float_num must be non-negative")


@dataclass(frozen=True)
class DNNLinkProfile:
    """使用任务索引表达的不可变 DNN 依赖边。"""

    source_task_index: int
    target_task_index: int
    float_tran: int

    def __post_init__(self) -> None:
        if self.source_task_index < 0 or self.target_task_index < 0:
            raise ValueError("task indices must be non-negative")
        if self.source_task_index == self.target_task_index:
            raise ValueError("a DNN link cannot be a self-loop")
        if self.float_tran < 0:
            raise ValueError("float_tran must be non-negative")


@dataclass(frozen=True)
class DNNProfile:
    """请求之间共享的 DNN DAG、计算量和传输画像。"""

    profile_id: str
    tasks: tuple[TaskProfile, ...]
    links: tuple[DNNLinkProfile, ...]
    input_size_kbit: float
    output_size_kbit: float
    semantic_version: str = "dnn_profile_v1"

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")
        if not self.tasks:
            raise ValueError("a DNN profile must contain at least one task")
        if self.input_size_kbit < 0.0 or self.output_size_kbit < 0.0:
            raise ValueError("profile input/output sizes must be non-negative")
        task_count = len(self.tasks)
        for link in self.links:
            if link.source_task_index >= task_count or link.target_task_index >= task_count:
                raise ValueError("DNN link task index is outside the profile")


@dataclass(frozen=True)
class InferenceRequest:
    """运行时到达的轻量请求实例，不重复保存 DNN DAG。"""

    request_id: int
    profile_id: str
    arrival_slot: int
    initiate_node: int
    deadline_ms: float
    preference_stability: float
    preference_delay: float
    preference_energy: float
    preference_mode: str = "adaptive"
    priority: int = 0

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")
        if self.arrival_slot < 0:
            raise ValueError("arrival_slot must be non-negative")
        if self.initiate_node < 0:
            raise ValueError("initiate_node must be non-negative")
        if self.deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive")
        for name, value in (
            ("preference_stability", self.preference_stability),
            ("preference_delay", self.preference_delay),
            ("preference_energy", self.preference_energy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.preference_stability + self.preference_delay + self.preference_energy <= 0.0:
            raise ValueError("at least one request preference must be positive")


@dataclass(frozen=True)
class DNNProfileCatalog:
    """有限 DNN 服务画像目录及其请求采样权重。"""

    profiles: tuple[DNNProfile, ...]
    sampling_weights: tuple[float, ...]
    semantic_version: str = "dnn_profile_catalog_v1"

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("profile catalog must not be empty")
        if len(self.profiles) != len(self.sampling_weights):
            raise ValueError("profiles and sampling_weights must have equal length")
        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("profile_id values must be unique")
        if any(weight < 0.0 for weight in self.sampling_weights):
            raise ValueError("sampling weights must be non-negative")
        if sum(self.sampling_weights) <= 0.0:
            raise ValueError("at least one sampling weight must be positive")

    def get(self, profile_id: str) -> DNNProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(profile_id)


ObservationPurpose = Literal[
    "planning",
    "publication_revalidation",
    "online_admission",
]


@dataclass(frozen=True)
class ObservationSnapshot:
    """算法可读取的显式、不可变系统观测。"""

    snapshot_version: int
    environment_state_version: int
    topology_version: int
    slot: int
    purpose: ObservationPurpose
    node_online: tuple[bool, ...]
    node_availability_modes: tuple[Literal["stochastic", "always_on"], ...]
    node_available_cpu: tuple[float, ...]
    node_loads: tuple[float, ...]
    node_heats: tuple[float, ...]
    node_availability_history: tuple[tuple[bool, ...], ...]
    directional_offered_rates: tuple[float, ...]
    directional_link_loads: tuple[float, ...]
    physical_link_heats: tuple[float, ...]
    effective_bandwidths: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.snapshot_version < 0 or self.environment_state_version < 0:
            raise ValueError("snapshot versions must be non-negative")
        if (
            self.topology_version < 0
            or self.slot < 0
        ):
            raise ValueError("topology version and slot must be non-negative")
        if self.purpose not in {
            "planning",
            "publication_revalidation",
            "online_admission",
        }:
            raise ValueError(f"unsupported observation purpose: {self.purpose}")
        node_count = len(self.node_online)
        if any(
            len(values) != node_count
            for values in (
                self.node_available_cpu,
                self.node_loads,
                self.node_heats,
                self.node_availability_history,
                self.node_availability_modes,
            )
        ):
            raise ValueError("all node observation fields must have equal length")


@dataclass(frozen=True)
class CandidateStateSnapshot:
    """一个候选接纳后用于目标和约束评价的唯一预测状态。"""

    observation_snapshot_version: int
    environment_state_version: int
    assignment: tuple[int, ...]
    predicted_node_cpu: tuple[float, ...]
    directional_offered_rates: tuple[float, ...]
    directional_link_loads: tuple[float, ...]
    physical_link_loads: tuple[float, ...]
    predicted_link_heats: tuple[float, ...]
    predicted_effective_bandwidths: tuple[float, ...]
    estimated_delay_ms: float
    total_energy: float
    node_stability_score: float
    node_availability_score: float
    service_exposure_slots: int
    availability_prediction_confidence: float
    link_stability_score: float | None
    operational_stability: float
    delay_satisfaction: float
    energy_satisfaction: float
    raw_joint_product: float
    deadline_feasible: bool
    fixed_point_converged: bool
    fixed_point_iterations: int
    fixed_point_relative_residual: float
    constraint_violation: float
    constraint_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observation_snapshot_version < 0 or self.environment_state_version < 0:
            raise ValueError("candidate state versions must be non-negative")
        if self.estimated_delay_ms < 0.0 or self.total_energy < 0.0:
            raise ValueError("candidate delay and energy must be non-negative")
        if self.fixed_point_iterations < 0:
            raise ValueError("fixed_point_iterations must be non-negative")
        if (
            not math.isfinite(self.fixed_point_relative_residual)
            or self.fixed_point_relative_residual < 0.0
        ):
            raise ValueError("fixed_point_relative_residual must be finite and non-negative")
        if self.service_exposure_slots <= 0:
            raise ValueError("service_exposure_slots must be positive")
        if self.constraint_violation < 0.0:
            raise ValueError("constraint_violation must be non-negative")
        for name, value in (
            ("node_stability_score", self.node_stability_score),
            ("node_availability_score", self.node_availability_score),
            (
                "availability_prediction_confidence",
                self.availability_prediction_confidence,
            ),
            ("operational_stability", self.operational_stability),
            ("delay_satisfaction", self.delay_satisfaction),
            ("energy_satisfaction", self.energy_satisfaction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.link_stability_score is not None and not 0.0 <= self.link_stability_score <= 1.0:
            raise ValueError("link_stability_score must be in [0, 1]")

    @property
    def is_strictly_feasible(self) -> bool:
        return not self.constraint_violations and self.constraint_violation <= 1e-12

    @property
    def availability_horizon_slots(self) -> int:
        """可用性预测时域必须与资源实际占用时域相同。"""
        return self.service_exposure_slots


def profile_from_dnn(dnn: DNN, profile_id: str) -> DNNProfile:
    """把旧 `DNN` 对象转换为不含请求属性的可复用画像。"""
    task_indices = {id(task): index for index, task in enumerate(dnn.tasks)}
    return DNNProfile(
        profile_id=profile_id,
        tasks=tuple(
            TaskProfile(cpu_need=task.cpu_need, float_num=task.float_num)
            for task in dnn.tasks
        ),
        links=tuple(
            DNNLinkProfile(
                source_task_index=task_indices[id(link.s_task)],
                target_task_index=task_indices[id(link.e_task)],
                float_tran=link.float_tran,
            )
            for link in dnn.links
        ),
        input_size_kbit=dnn.startFloat,
        output_size_kbit=dnn.backFloat,
    )


def materialize_dnn(profile: DNNProfile, request: InferenceRequest) -> DNN:
    """在请求到达时把画像和请求实例适配为当前算法使用的 `DNN`。"""
    if profile.profile_id != request.profile_id:
        raise ValueError("request profile_id does not match the selected profile")
    tasks = [Task(task.cpu_need, task.float_num) for task in profile.tasks]
    links = [
        LinkDNN(
            tasks[link.source_task_index],
            tasks[link.target_task_index],
            link.float_tran,
        )
        for link in profile.links
    ]
    return DNN(
        tasks=tasks,
        links=links,
        delay=int(round(request.deadline_ms)),
        initiateNode=request.initiate_node,
        startFloat=profile.input_size_kbit,
        preference_stability=request.preference_stability,
        preference_delay=request.preference_delay,
        preference_energy=request.preference_energy,
        backFloat=profile.output_size_kbit,
        profile_id=profile.profile_id,
    )
