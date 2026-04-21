from __future__ import annotations

import random
from typing import List, Tuple

from core.dag_generator import DAGGenerator
from models import DNN, LinkDNN, LinkNode, Node, Task


def create_dnns(num: int, nodes: List[Node]) -> List[DNN]:
    ds: List[DNN] = []
    dag_generator = DAGGenerator()
    for _ in range(num):
        task_num = 8 + int(random.random() * 12)
        task_num = 8 + int(random.random() * 12)
        ds.append(create_dnn(task_num, nodes, dag_generator))
    return ds


def create_dnn(task_count: int, nodes: List[Node], dag_generator: DAGGenerator) -> DNN:
    graph = dag_generator.get_graph(task_count)
    tasks: List[Task] = []
    link_dnns: List[LinkDNN] = []
    delay = int(1500 * random.random()) + 500
    initiate = 0

    for _ in range(task_count):
        cpu_need = 1 + int(random.random() * 2)
        float_num = (random.random() * 0.95 + 0.05) * 1000
        task = Task(cpu_need, float_num)
        task.weight_file = (0.5 + (9.5 * random.random())) * 8
        tasks.append(task)

    for j in range(len(graph)):
        for p in range(len(graph[j])):
            if graph[j][p] == 1:
                float_tran = int(200 * random.random()) + 300
                link_dnns.append(LinkDNN(tasks[j], tasks[p], 8 * float_tran))

    dnn = DNN(tasks, link_dnns, delay, initiate)
    dnn.preA = 0.7 + 0.3 * random.random()
    dnn.preR = 0.7 + 0.3 * random.random()
    dnn.startFloat = 588 * 8
    dnn.backFloat = 8 * (int(90 * random.random()) + 10)

    user_indices = [idx for idx, node in enumerate(nodes) if node.level == 1]
    dnn.initiateNode = user_indices[int(random.random() * len(user_indices))]
    return dnn


def create_nodes() -> Tuple[List[Node], List[LinkNode]]:
    nodes: List[Node] = []
    link_nodes: List[LinkNode] = []
    cloud_node = Node(2**31 - 1, 2**31 - 1, 3, 0.99, 0.99, 28)
    nodes.append(cloud_node)
    edge_nodes: List[Node] = []

    edge_node1 = Node(20, 20, 2, 0.98, 0.99, 17)
    edge_node1.o_reliability = random.random() * 0.05 + 0.94
    edge_node1.a_reliability = random.random() * 0.05 + 0.94
    edge_node1.float_rate = int(random.random() * 8 + 16)
    edge_node1.cpu = int(16 + random.random() * 16)

    edge_node2 = Node(28, 22, 2, 0.97, 0.96, 7)
    edge_node2.o_reliability = random.random() * 0.05 + 0.94
    edge_node2.a_reliability = random.random() * 0.05 + 0.94
    edge_node2.float_rate = int(random.random() * 8 + 16)
    edge_node2.cpu = int(16 + random.random() * 16)

    edge_node3 = Node(15, 18, 2, 0.98, 0.98, 20)
    edge_node3.o_reliability = random.random() * 0.05 + 0.94
    edge_node3.a_reliability = random.random() * 0.05 + 0.94
    edge_node3.float_rate = int(random.random() * 8 + 16)
    edge_node3.cpu = int(16 + random.random() * 16)

    edge_node4 = Node(14, 16, 2, 0.98, 0.98, 16)
    edge_node4.o_reliability = random.random() * 0.05 + 0.94
    edge_node4.a_reliability = random.random() * 0.05 + 0.94
    edge_node4.float_rate = int(random.random() * 8 + 16)
    edge_node4.cpu = int(16 + random.random() * 16)

    edge_node5 = Node(18, 24, 2, 0.97, 0.97, 24)
    edge_node5.o_reliability = random.random() * 0.05 + 0.94
    edge_node5.a_reliability = random.random() * 0.05 + 0.94
    edge_node5.float_rate = int(random.random() * 8 + 16)
    edge_node5.cpu = int(16 + random.random() * 16)

    edge_node6 = Node(14, 28, 2, 0.97, 0.97, 22)
    edge_node6.o_reliability = random.random() * 0.05 + 0.94
    edge_node6.a_reliability = random.random() * 0.05 + 0.94
    edge_node6.float_rate = int(random.random() * 8 + 16)
    edge_node6.cpu = int(16 + random.random() * 16)

    edge_node7 = Node(18, 32, 2, 0.95, 0.95, 18)
    edge_node7.o_reliability = random.random() * 0.05 + 0.94
    edge_node7.a_reliability = random.random() * 0.05 + 0.94
    edge_node7.float_rate = int(random.random() * 8 + 16)
    edge_node7.cpu = int(16 + random.random() * 16)

    edge_node8 = Node(17, 26, 2, 0.96, 0.96, 12)
    edge_node8.o_reliability = random.random() * 0.05 + 0.94
    edge_node8.a_reliability = random.random() * 0.05 + 0.94
    edge_node8.float_rate = int(random.random() * 8 + 16)
    edge_node8.cpu = int(16 + random.random() * 16)

    edge_node9 = Node(30, 32, 2, 0.97, 0.97, 8)
    edge_node9.o_reliability = random.random() * 0.05 + 0.94
    edge_node9.a_reliability = random.random() * 0.05 + 0.94
    edge_node9.float_rate = int(random.random() * 8 + 16)
    edge_node9.cpu = int(16 + random.random() * 16)

    edge_node10 = Node(21, 25, 2, 0.96, 0.96, 19)
    edge_node10.o_reliability = random.random() * 0.05 + 0.94
    edge_node10.a_reliability = random.random() * 0.05 + 0.94
    edge_node10.float_rate = int(random.random() * 8 + 16)
    edge_node10.cpu = int(16 + random.random() * 16)

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
        link_ab = LinkNode(a, b, default_band)
        link_ab.band_width = int(random.random() * sampled_span + sampled_min)
        link_nodes.append(link_ab)
        link_ba = LinkNode(b, a, default_band)
        link_ba.band_width = link_ab.band_width
        link_nodes.append(link_ba)

    add_bidirectional_link(edge_node1, edge_node2, 120, 100, 50)
    add_bidirectional_link(edge_node1, edge_node9, 70, 100, 50)
    add_bidirectional_link(edge_node1, edge_node6, 120, 100, 50)

    link_node2c = LinkNode(edge_node2, cloud_node, 6)
    link_node2c.band_width = int(random.random() * 10 + 1)
    link_nodes.append(link_node2c)
    link_nodec2 = LinkNode(cloud_node, edge_node2, 6)
    link_nodec2.band_width = link_node2c.band_width
    link_nodes.append(link_nodec2)

    add_bidirectional_link(edge_node2, edge_node3, 110, 100, 50)
    add_bidirectional_link(edge_node3, edge_node4, 130, 100, 50)
    add_bidirectional_link(edge_node3, edge_node9, 80, 100, 50)
    add_bidirectional_link(edge_node4, edge_node5, 135, 100, 50)
    add_bidirectional_link(edge_node4, edge_node10, 130, 100, 50)
    add_bidirectional_link(edge_node5, edge_node8, 134, 100, 50)

    link_node5c = LinkNode(edge_node5, cloud_node, 4)
    link_node5c.band_width = int(random.random() * 10 + 1)
    link_nodes.append(link_node5c)
    link_nodec5 = LinkNode(cloud_node, edge_node5, 4)
    link_nodec5.band_width = link_node5c.band_width
    link_nodes.append(link_nodec5)

    add_bidirectional_link(edge_node6, edge_node7, 138, 100, 50)
    add_bidirectional_link(edge_node7, edge_node8, 142, 100, 50)

    link_node7c = LinkNode(edge_node7, cloud_node, 5)
    link_node7c.band_width = int(random.random() * 10 + 1)
    link_nodes.append(link_node7c)
    link_nodec7 = LinkNode(cloud_node, edge_node7, 5)
    link_node7c.band_width = link_node7c.getBandWidth()
    link_nodes.append(link_nodec7)

    add_bidirectional_link(edge_node8, edge_node10, 130, 100, 50)

    link_node9c = LinkNode(edge_node9, cloud_node, 4)
    link_node9c.band_width = int(random.random() * 10 + 1)
    link_nodes.append(link_node9c)
    link_nodec9 = LinkNode(cloud_node, edge_node9, 4)
    link_nodec9.band_width = link_node9c.band_width
    link_nodes.append(link_nodec9)

    link_node10c = LinkNode(edge_node10, cloud_node, 5)
    link_node10c.band_width = int(random.random() * 10 + 1)
    link_nodes.append(link_node10c)
    link_nodec10 = LinkNode(cloud_node, edge_node10, 5)
    link_nodec10.band_width = link_node10c.band_width
    link_nodes.append(link_nodec10)

    for edge in edge_nodes:
        random_count = 2 + int(random.random() * 3)
        for _ in range(random_count):
            cpu = 8 + int(8 * random.random())
            reliable = 0.9 + (random.random() * 0.05)
            accuracy = 0.9 + (random.random() * 0.05)
            float_num = 1.0 + (random.random() * 3)
            band = 20 + int(random.random() * 31)
            user_node = Node(cpu, cpu, 1, reliable, accuracy, float_num)
            link_nodes.append(LinkNode(user_node, edge, band))
            link_nodes.append(LinkNode(edge, user_node, band))
            nodes.append(user_node)

    return nodes, link_nodes
