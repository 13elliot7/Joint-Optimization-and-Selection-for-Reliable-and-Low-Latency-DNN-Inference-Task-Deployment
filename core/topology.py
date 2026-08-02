from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple

from core.dag_generator import DAGGenerator
from models import DNN, LinkDNN, LinkNode, Node, PhysicalLinkState, Task


@dataclass(frozen=True)
class TopologyConfig:
    """可扩展云-边-端拓扑配置。"""

    cloud_count: int = 1
    edge_count: int = 10
    user_count: int | None = None
    edge_link_factor: float = 2.0


def _default_comp_power(level: int) -> float:
    """按节点层级返回固定计算功率。"""
    if level == 3:
        return 18.0
    if level == 2:
        return 10.0
    return 4.0


def _default_link_energy_per_mb(start_level: int, end_level: int) -> float:
    """按链路层级组合返回固定单位传输能耗。"""
    levels = {start_level, end_level}
    if levels == {2, 3}:
        return 0.12
    if levels == {1, 2}:
        return 0.05
    if levels == {2}:
        return 0.08
    if levels == {1}:
        return 0.03
    return 0.10


def create_dnns(num: int, nodes: List[Node]) -> List[DNN]:
    """批量生成随机 DNN 请求。"""
    ds: List[DNN] = []
    dag_generator = DAGGenerator()
    for _ in range(num):
        task_num = 8 + int(random.random() * 12)
        ds.append(create_dnn(task_num, nodes, dag_generator))
    return ds


def create_dnn(task_count: int, nodes: List[Node], dag_generator: DAGGenerator) -> DNN:
    """构造一个包含随机任务、依赖和约束的 DNN。"""
    graph = dag_generator.get_graph(task_count)
    tasks: List[Task] = []
    link_dnns: List[LinkDNN] = []
    delay = int(1500 * random.random()) + 500
    initiate = 0

    for _ in range(task_count):
        cpu_need = 1 + int(random.random() * 2)
        float_num = (random.random() * 0.95 + 0.05) * 1000
        task = Task(cpu_need, float_num)
        tasks.append(task)

    for j in range(len(graph)):
        for p in range(len(graph[j])):
            if graph[j][p] == 1:
                float_tran = int(200 * random.random()) + 300
                link_dnns.append(LinkDNN(tasks[j], tasks[p], 8 * float_tran))

    dnn = DNN(tasks, link_dnns, delay, initiate)
    dnn.preference_fidelity = 0.7 + 0.3 * random.random()
    dnn.preference_stability = 0.7 + 0.3 * random.random()
    dnn.startFloat = 588 * 8
    dnn.backFloat = 8 * (int(90 * random.random()) + 10)

    user_indices = [idx for idx, node in enumerate(nodes) if node.level == 1]
    dnn.initiateNode = user_indices[int(random.random() * len(user_indices))]
    return dnn


def _create_edge_node() -> Node:
    """随机生成一个边缘节点。"""
    cpu = int(16 + random.random() * 16)
    node = Node(
        cpu=cpu,
        max_cpu=cpu,
        level=2,
        operational_stability=random.random() * 0.05 + 0.94,
        inference_fidelity=random.random() * 0.05 + 0.94,
        float_rate=int(random.random() * 8 + 16),
    )
    node.comp_power = _default_comp_power(node.level)
    return node


def _add_bidirectional_link(
    link_nodes: List[LinkNode],
    a: Node,
    b: Node,
    band_width: int,
) -> tuple[LinkNode, LinkNode]:
    """为两个节点添加带宽一致的双向链路。"""
    physical_state = PhysicalLinkState(a, b)
    link_ab = LinkNode(a, b, band_width, physical_state=physical_state)
    link_ab.energy_per_mb = _default_link_energy_per_mb(a.level, b.level)
    link_nodes.append(link_ab)
    link_ba = LinkNode(b, a, band_width, physical_state=physical_state)
    link_ba.energy_per_mb = _default_link_energy_per_mb(b.level, a.level)
    link_nodes.append(link_ba)
    return link_ab, link_ba


def _create_parametric_nodes(config: TopologyConfig) -> Tuple[List[Node], List[LinkNode]]:
    """按给定规模生成可扩展云-边-端拓扑。"""
    cloud_count = max(1, config.cloud_count)
    edge_count = max(1, config.edge_count)
    user_count = config.user_count if config.user_count is not None else edge_count * 3
    user_count = max(1, user_count)
    nodes: List[Node] = []
    link_nodes: List[LinkNode] = []
    cloud_nodes: List[Node] = []
    edge_nodes: List[Node] = []

    for _ in range(cloud_count):
        cloud_node = Node(2**31 - 1, 2**31 - 1, 3, 0.99, 0.99, 28)
        cloud_node.comp_power = _default_comp_power(cloud_node.level)
        cloud_nodes.append(cloud_node)
        nodes.append(cloud_node)

    for _ in range(edge_count):
        edge_node = _create_edge_node()
        edge_nodes.append(edge_node)
        nodes.append(edge_node)

    # Edge ring keeps the graph connected; extra random chords improve path diversity.
    ring_edges: set[tuple[int, int]] = set()
    for idx, edge_node in enumerate(edge_nodes):
        next_edge = edge_nodes[(idx + 1) % edge_count]
        if edge_count > 1:
            ring_key = tuple(sorted((id(edge_node), id(next_edge))))
            if ring_key not in ring_edges:
                ring_edges.add(ring_key)
                _add_bidirectional_link(
                    link_nodes,
                    edge_node,
                    next_edge,
                    int(random.random() * 50 + 100),
                )
        cloud_node = cloud_nodes[idx % cloud_count]
        _add_bidirectional_link(link_nodes, edge_node, cloud_node, int(random.random() * 10 + 1))

    target_undirected_edges = int(edge_count * max(config.edge_link_factor, 1.0))
    existing = {
        tuple(sorted((id(link.s_node), id(link.e_node))))
        for link in link_nodes
        if link.s_node.level == link.e_node.level == 2
    }
    attempts = 0
    while len(existing) < target_undirected_edges and attempts < edge_count * edge_count:
        attempts += 1
        a_idx = int(random.random() * edge_count)
        b_idx = int(random.random() * edge_count)
        if a_idx == b_idx:
            continue
        key = tuple(sorted((id(edge_nodes[a_idx]), id(edge_nodes[b_idx]))))
        if key in existing:
            continue
        existing.add(key)
        _add_bidirectional_link(
            link_nodes,
            edge_nodes[a_idx],
            edge_nodes[b_idx],
            int(random.random() * 50 + 100),
        )

    for user_idx in range(user_count):
        cpu = 8 + int(8 * random.random())
        user_node = Node(
            cpu,
            cpu,
            1,
            0.9 + (random.random() * 0.05),
            0.9 + (random.random() * 0.05),
            1.0 + (random.random() * 3),
        )
        user_node.comp_power = _default_comp_power(user_node.level)
        nodes.append(user_node)
        edge = edge_nodes[user_idx % edge_count]
        _add_bidirectional_link(link_nodes, user_node, edge, 20 + int(random.random() * 31))

    return nodes, link_nodes


def create_nodes(config: TopologyConfig | None = None) -> Tuple[List[Node], List[LinkNode]]:
    """生成云边端节点集合以及双向链路拓扑。"""
    if config is not None and config != TopologyConfig():
        return _create_parametric_nodes(config)

    nodes: List[Node] = []
    link_nodes: List[LinkNode] = []
    cloud_node = Node(2**31 - 1, 2**31 - 1, 3, 0.99, 0.99, 28)
    cloud_node.comp_power = _default_comp_power(cloud_node.level)
    nodes.append(cloud_node)
    edge_nodes: List[Node] = []

    def create_default_edge_node(max_cpu: int) -> Node:
        """使用最终采样值构造节点，确保当前分数与基准先验一致。"""
        operational_stability = random.random() * 0.05 + 0.94
        inference_fidelity = random.random() * 0.05 + 0.94
        float_rate = int(random.random() * 8 + 16)
        cpu = int(16 + random.random() * 16)
        return Node(
            cpu=cpu,
            max_cpu=max_cpu,
            level=2,
            operational_stability=operational_stability,
            inference_fidelity=inference_fidelity,
            float_rate=float_rate,
        )

    edge_node1 = create_default_edge_node(20)
    edge_node2 = create_default_edge_node(22)
    edge_node3 = create_default_edge_node(18)
    edge_node4 = create_default_edge_node(16)
    edge_node5 = create_default_edge_node(24)
    edge_node6 = create_default_edge_node(28)
    edge_node7 = create_default_edge_node(32)
    edge_node8 = create_default_edge_node(26)
    edge_node9 = create_default_edge_node(32)
    edge_node10 = create_default_edge_node(25)

    edge_nodes.extend(
        [
            edge_node1,
            edge_node2,
            edge_node3,
            edge_node4,
            edge_node5,
            edge_node6,
            edge_node7,
            edge_node8,
        ]
    )
    nodes.extend(
        [
            edge_node1,
            edge_node2,
            edge_node3,
            edge_node4,
            edge_node5,
            edge_node6,
            edge_node7,
            edge_node8,
            edge_node9,
            edge_node10,
        ]
    )

    def add_bidirectional_link(a: Node, b: Node, default_band: int, sampled_min: int, sampled_span: int) -> None:
        """为两个节点补充一对带宽一致的双向链路。"""
        del default_band
        sampled_band = int(random.random() * sampled_span + sampled_min)
        _add_bidirectional_link(link_nodes, a, b, sampled_band)

    add_bidirectional_link(edge_node1, edge_node2, 120, 100, 50)
    add_bidirectional_link(edge_node1, edge_node9, 70, 100, 50)
    add_bidirectional_link(edge_node1, edge_node6, 120, 100, 50)

    _add_bidirectional_link(
        link_nodes,
        edge_node2,
        cloud_node,
        int(random.random() * 10 + 1),
    )

    add_bidirectional_link(edge_node2, edge_node3, 110, 100, 50)
    add_bidirectional_link(edge_node3, edge_node4, 130, 100, 50)
    add_bidirectional_link(edge_node3, edge_node9, 80, 100, 50)
    add_bidirectional_link(edge_node4, edge_node5, 135, 100, 50)
    add_bidirectional_link(edge_node4, edge_node10, 130, 100, 50)
    add_bidirectional_link(edge_node5, edge_node8, 134, 100, 50)

    _add_bidirectional_link(
        link_nodes,
        edge_node5,
        cloud_node,
        int(random.random() * 10 + 1),
    )

    add_bidirectional_link(edge_node6, edge_node7, 138, 100, 50)
    add_bidirectional_link(edge_node7, edge_node8, 142, 100, 50)

    _add_bidirectional_link(
        link_nodes,
        edge_node7,
        cloud_node,
        int(random.random() * 10 + 1),
    )

    add_bidirectional_link(edge_node8, edge_node10, 130, 100, 50)

    _add_bidirectional_link(
        link_nodes,
        edge_node9,
        cloud_node,
        int(random.random() * 10 + 1),
    )

    _add_bidirectional_link(
        link_nodes,
        edge_node10,
        cloud_node,
        int(random.random() * 10 + 1),
    )

    for edge in edge_nodes:
        random_count = 2 + int(random.random() * 3)
        for _ in range(random_count):
            cpu = 8 + int(8 * random.random())
            operational_stability = 0.9 + (random.random() * 0.05)
            inference_fidelity = 0.9 + (random.random() * 0.05)
            float_num = 1.0 + (random.random() * 3)
            band = 20 + int(random.random() * 31)
            user_node = Node(
                cpu,
                cpu,
                1,
                operational_stability,
                inference_fidelity,
                float_num,
            )
            user_node.comp_power = _default_comp_power(user_node.level)
            _add_bidirectional_link(link_nodes, user_node, edge, band)
            nodes.append(user_node)

    for edge_node in nodes:
        if edge_node.level in (2, 3):
            edge_node.comp_power = _default_comp_power(edge_node.level)

    return nodes, link_nodes
