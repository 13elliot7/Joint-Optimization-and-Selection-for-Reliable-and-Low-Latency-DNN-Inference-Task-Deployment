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

    def getCpu(self) -> int:
        return self.cpu

    def setCpu(self, cpu: int) -> None:
        self.cpu = cpu

    def getMaxCpu(self) -> int:
        return self.max_cpu

    def getLevel(self) -> int:
        return self.level

    def getoReliability(self) -> float:
        return self.o_reliability

    def getaReliability(self) -> float:
        return self.a_reliability

    def getFloatRate(self) -> float:
        return self.float_rate

    def clone(self) -> "Node":
        return Node(
            self.cpu,
            self.max_cpu,
            self.level,
            self.o_reliability,
            self.a_reliability,
            self.float_rate,
        )


@dataclass(eq=False)
class Task:
    cpu_need: int
    float_num: float
    weight_file: float = 0.0

    def getCpuNeed(self) -> int:
        return self.cpu_need

    def getFloatNum(self) -> float:
        return self.float_num


@dataclass(eq=False)
class LinkNode:
    s_node: Node
    e_node: Node
    band_width: int

    def getsNode(self) -> Node:
        return self.s_node

    def geteNode(self) -> Node:
        return self.e_node

    def getBandWidth(self) -> int:
        return self.band_width


@dataclass(eq=False)
class LinkDNN:
    s_task: Task
    e_task: Task
    float_tran: int

    def getsTask(self) -> Task:
        return self.s_task

    def geteTask(self) -> Task:
        return self.e_task

    def getFloatTran(self) -> int:
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
        return self.tasks

    def getLinks(self) -> List[LinkDNN]:
        return self.links

    def getDelay(self) -> int:
        return self.delay

