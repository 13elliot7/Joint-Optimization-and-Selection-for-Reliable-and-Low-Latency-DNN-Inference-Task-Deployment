from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(eq=False)
class Node:
    cpu: int
    max_cpu: int
    level: int
    o_reliability: float
    a_reliability: float
    float_rate: float
    base_o_reliability: float | None = None
    base_a_reliability: float | None = None
    load_ratio: float = 0.0
    heat: float = 0.0

    def __post_init__(self) -> None:
        """在节点创建后补齐动态可靠性建模所需的基线字段。"""
        if self.base_o_reliability is None:
            self.base_o_reliability = self.o_reliability
        if self.base_a_reliability is None:
            self.base_a_reliability = self.a_reliability

    def getCpu(self) -> int:
        """返回节点当前剩余 CPU。"""
        return self.cpu

    def setCpu(self, cpu: int) -> None:
        """更新节点当前剩余 CPU。"""
        self.cpu = cpu

    def getMaxCpu(self) -> int:
        """返回节点的 CPU 上限。"""
        return self.max_cpu

    def getLevel(self) -> int:
        """返回节点层级。"""
        return self.level

    def getoReliability(self) -> float:
        """返回节点当前运行可靠性。"""
        return self.o_reliability

    def getaReliability(self) -> float:
        """返回节点当前精度可靠性。"""
        return self.a_reliability

    def getFloatRate(self) -> float:
        """返回节点浮点计算速率。"""
        return self.float_rate

    def clone(self) -> "Node":
        """复制节点对象及其动态状态。"""
        return Node(
            self.cpu,
            self.max_cpu,
            self.level,
            self.o_reliability,
            self.a_reliability,
            self.float_rate,
            self.base_o_reliability,
            self.base_a_reliability,
            self.load_ratio,
            self.heat,
        )


@dataclass(eq=False)
class Task:
    cpu_need: int
    float_num: float
    weight_file: float = 0.0

    def getCpuNeed(self) -> int:
        """返回子任务所需的 CPU 量。"""
        return self.cpu_need

    def getFloatNum(self) -> float:
        """返回子任务的计算量。"""
        return self.float_num


@dataclass(eq=False)
class LinkNode:
    s_node: Node
    e_node: Node
    band_width: int
    base_reliability: float | None = None
    reliability: float | None = None
    load_ratio: float = 0.0
    heat: float = 0.0

    def __post_init__(self) -> None:
        """在链路创建后补齐动态可靠性字段。"""
        if self.base_reliability is None:
            self.base_reliability = 1.0
        if self.reliability is None:
            self.reliability = self.base_reliability

    def getsNode(self) -> Node:
        """返回链路起点节点。"""
        return self.s_node

    def geteNode(self) -> Node:
        """返回链路终点节点。"""
        return self.e_node

    def getBandWidth(self) -> int:
        """返回链路带宽。"""
        return self.band_width


@dataclass(eq=False)
class RunningDNN:
    dnn_index: int
    assignment: List[int]
    arrival_time: int
    start_time: int
    estimated_runtime: float
    remaining_slots: int
    used_nodes: List[int] = field(default_factory=list)
    used_links: List[LinkNode] = field(default_factory=list)


@dataclass(eq=False)
class LinkDNN:
    s_task: Task
    e_task: Task
    float_tran: int

    def getsTask(self) -> Task:
        """返回 DNN 依赖边的起点任务。"""
        return self.s_task

    def geteTask(self) -> Task:
        """返回 DNN 依赖边的终点任务。"""
        return self.e_task

    def getFloatTran(self) -> int:
        """返回依赖边上传输的数据量。"""
        return self.float_tran


@dataclass(eq=False)
class DNN:
    tasks: List[Task]
    links: List[LinkDNN]
    delay: int
    initiateNode: int
    startFloat: float = 0.0
    preR: float = 0.0
    preA: float = 0.0
    backFloat: float = 0.0

    def getTasks(self) -> List[Task]:
        """返回 DNN 的全部子任务。"""
        return self.tasks

    def getLinks(self) -> List[LinkDNN]:
        """返回 DNN 内部 DAG 依赖边。"""
        return self.links

    def getDelay(self) -> int:
        """返回 DNN 的时延约束。"""
        return self.delay
