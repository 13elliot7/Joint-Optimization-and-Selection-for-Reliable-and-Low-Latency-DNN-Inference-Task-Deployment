from __future__ import annotations

import random
from typing import Dict, List, Set


class DAGGenerator:
    def __init__(self) -> None:
        """初始化 DAG 生成器的内部状态。"""
        self.num_vertices = 0
        self.graph: List[List[int]] = []
        self.y = 0

    def get_graph(self, x: int) -> List[List[int]]:
        """生成指定顶点数的 DAG 邻接矩阵。"""
        self.num_vertices = x
        self.y = x
        self.graph = [[0 for _ in range(x)] for _ in range(x)]
        vertices = list(range(x))
        graph_map = self._generate_graph(vertices)
        self._fill_graph(graph_map)
        return self.graph

    def _fill_graph(self, graph_map: Dict[int, Set[int]]) -> None:
        """把稀疏图映射补成邻接矩阵并保证首尾可达。"""
        visit1 = [0 for _ in range(self.num_vertices)]
        visit2 = [0 for _ in range(self.num_vertices)]
        for start, neighbors in graph_map.items():
            for neighbor in neighbors:
                self.graph[start][neighbor] = 1
                visit1[start] += 1
                visit2[neighbor] += 1
        for i in range(1, self.num_vertices - 1):
            if visit2[i] == 0:
                self.graph[0][i] = 1
            if visit1[i] == 0:
                self.graph[i][self.y - 1] = 1

    def _generate_graph(self, vertices_list: List[int]) -> Dict[int, Set[int]]:
        """按随机拓扑顺序生成无环边集合。"""
        graph_map: Dict[int, Set[int]] = {}
        while vertices_list:
            current_vertex = vertices_list.pop(random.randrange(len(vertices_list)))
            if current_vertex not in graph_map:
                graph_map[current_vertex] = set()
                for neighbor in vertices_list:
                    is_added = False
                    if neighbor > current_vertex and not self._is_cyclic(graph_map, current_vertex, neighbor):
                        graph_map[current_vertex].add(neighbor)
                        is_added = True
                    if is_added:
                        break
        return graph_map

    def _is_cyclic(self, graph_map: Dict[int, Set[int]], vertex_a: int, vertex_b: int) -> bool:
        """检查加入一条边后是否会形成环。"""
        stack = [vertex_a]
        visited = {vertex_a}
        while stack and vertex_b not in visited:
            current = stack.pop()
            for neighbor in graph_map.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
                    visited.add(neighbor)
                else:
                    return True
        return vertex_b in visited
