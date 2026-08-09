from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass
from typing import Callable

from core.environment import Environment, InfeasibleAssignmentError, StaleEnvironmentStateError
from models import CandidateStateSnapshot, InferenceRequest
from planning.online_dispatcher import DispatchDecision
from planning.repository import PlanRepository
from proposed import AllDNNRefactor, CustomizedSearchConfig


@dataclass(frozen=True)
class PerRequestCustomizedConfig:
    population_size: int = 12
    generations: int = 4
    archive_capacity: int = 64
    random_seed: int = 0
    wall_time_budget_ms: float | None = None


@dataclass
class PerRequestSearchStatistics:
    requests_searched: int = 0
    assignments_generated: int = 0
    constraint_rejected_candidates: int = 0
    nonconverged_candidates: int = 0
    feasible_candidates: int = 0
    selected_candidates: int = 0


class PerRequestCustomizedDispatcher:
    """在请求到达后运行原 customized 搜索，并通过环境接口原子提交。"""

    def __init__(
        self,
        environment: Environment,
        config: PerRequestCustomizedConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.environment = environment
        self.config = config
        self._clock = clock
        self.algorithm = AllDNNRefactor(environment)
        self.statistics = PerRequestSearchStatistics()

    @staticmethod
    def _score(candidate: CandidateStateSnapshot, request: InferenceRequest) -> float:
        weighted = (
            request.preference_stability * candidate.operational_stability
            + request.preference_delay * candidate.delay_satisfaction
            + request.preference_energy * candidate.energy_satisfaction
        )
        return weighted / max(
            request.preference_stability
            + request.preference_delay
            + request.preference_energy,
            1e-12,
        )

    def dispatch(
        self,
        dnn_index: int,
        request: InferenceRequest,
        repository: PlanRepository,
    ) -> DispatchDecision:
        started_at = self._clock()
        snapshot = self.environment.capture_observation_snapshot("online_admission")
        search = self.algorithm.generate_customized_candidates(
            dnn_index=dnn_index,
            snapshot=snapshot,
            search_config=CustomizedSearchConfig(
                population_size=self.config.population_size,
                generations=self.config.generations,
                archive_capacity=self.config.archive_capacity,
                random_seed=self.config.random_seed + request.request_id,
                wall_time_budget_ms=self.config.wall_time_budget_ms,
            ),
        )
        self.statistics.requests_searched += 1
        self.statistics.assignments_generated += search.evaluations
        self.statistics.constraint_rejected_candidates += (
            search.constraint_rejected_evaluations
        )
        self.statistics.nonconverged_candidates += search.nonconverged_evaluations
        self.statistics.feasible_candidates += search.feasible_evaluations

        evaluated: list[tuple[float, tuple[int, ...], CandidateStateSnapshot]] = []
        for item in search.candidates:
            candidate = self.environment.predict_candidate_state(
                dnn_index, list(item.assignment), snapshot
            )
            if candidate.is_strictly_feasible:
                evaluated.append((self._score(candidate, request), item.assignment, candidate))
        evaluated.sort(key=lambda item: (-item[0], item[1]))

        committed: CandidateStateSnapshot | None = None
        assignment: tuple[int, ...] | None = None
        rejection_reason = search.termination_reason
        for _, proposed_assignment, _ in evaluated:
            try:
                committed = self.environment.commit_assignment(
                    dnn_index,
                    list(proposed_assignment),
                    expected_environment_state_version=snapshot.environment_state_version,
                )
            except (InfeasibleAssignmentError, StaleEnvironmentStateError) as error:
                rejection_reason = str(error)
                continue
            assignment = proposed_assignment
            self.statistics.selected_candidates += 1
            break

        elapsed_ms = (self._clock() - started_at) * 1000.0
        accepted = committed is not None
        budget_ms = self.config.wall_time_budget_ms or max(
            request.deadline_ms * 0.25, 0.0
        )
        return DispatchDecision(
            accepted=accepted,
            path="per_request_customized" if accepted else "rejected",
            rejection_reason=None if accepted else rejection_reason or "no_feasible_candidate",
            plan_id=None,
            assignment=assignment,
            candidate=committed,
            repository_version=repository.repository_version,
            evaluated_environment_state_version=snapshot.environment_state_version,
            committed_environment_state_version=(
                self.environment.environment_state_version if accepted else None
            ),
            full_revalidation_count=len(evaluated),
            online_runtime_ms=elapsed_ms,
            configured_online_budget_ms=budget_ms,
            request_online_budget_ms=budget_ms,
            feasible_candidate_count=len(evaluated),
            budget_exhausted=(
                self.config.wall_time_budget_ms is not None
                and search.termination_reason == "wall_time_budget"
            ),
            early_stop_reason=(
                search.termination_reason
                if search.termination_reason != "generation_limit"
                else None
            ),
            best_utility_after_each_evaluation=tuple(
                item[0] for item in sorted(evaluated, key=lambda item: item[0])
            ),
            control_decision_id=None,
        )


@dataclass(frozen=True)
class OnlineBaselineConfig:
    algorithm: str
    evaluation_budget: int = 48
    random_seed: int = 0
    beam_width: int = 8

    def __post_init__(self) -> None:
        if self.algorithm not in {
            "random",
            "sa",
            "localfirst",
            "max_resource_fast",
            "rtbl",
        }:
            raise ValueError("unsupported online baseline")
        if self.evaluation_budget <= 0 or self.beam_width <= 0:
            raise ValueError("baseline budgets must be positive")


class OnlineBaselineDispatcher:
    """将旧基线的节点选择原则适配为无推进、无提交副作用的在线策略。"""

    def __init__(
        self,
        environment: Environment,
        config: OnlineBaselineConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.environment = environment
        self.config = config
        self._clock = clock
        self.evaluations = 0
        self.selected = 0
        self._rtbl_q = 0.0
        self._rtbl_selection_counts = [0 for _ in environment.nodes]

    def _legal_nodes(self, dnn_index: int, snapshot) -> tuple[int, ...]:
        origin = self.environment.ds[dnn_index].initiateNode
        return tuple(
            index
            for index, node in enumerate(self.environment.nodes)
            if snapshot.node_online[index] and (node.level != 1 or index == origin)
        )

    def _topology(self, dnn_index: int) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
        dnn = self.environment.ds[dnn_index]
        task_index = {id(task): index for index, task in enumerate(dnn.tasks)}
        predecessors: dict[int, list[int]] = {index: [] for index in range(len(dnn.tasks))}
        successors: dict[int, list[int]] = {index: [] for index in range(len(dnn.tasks))}
        indegree = [0 for _ in dnn.tasks]
        for link in dnn.links:
            start = task_index[id(link.s_task)]
            end = task_index[id(link.e_task)]
            predecessors[end].append(start)
            successors[start].append(end)
            indegree[end] += 1
        ready = [index for index, value in enumerate(indegree) if value == 0]
        order: list[int] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        return tuple(order), {
            index: tuple(values) for index, values in predecessors.items()
        }

    def _greedy_assignment(self, dnn_index: int, snapshot, mode: str) -> tuple[int, ...] | None:
        dnn = self.environment.ds[dnn_index]
        legal = self._legal_nodes(dnn_index, snapshot)
        order, predecessors = self._topology(dnn_index)
        remaining = list(snapshot.node_available_cpu)
        assignment = [-1 for _ in dnn.tasks]
        for task_index in order:
            minimum_level = max(
                (self.environment.nodes[assignment[parent]].level for parent in predecessors[task_index]),
                default=1,
            )
            candidates = [
                node_index
                for node_index in legal
                if self.environment.nodes[node_index].level >= minimum_level
                and remaining[node_index] >= dnn.tasks[task_index].cpu_need
            ]
            if not candidates:
                return None
            if mode == "localfirst":
                node_index = min(
                    candidates,
                    key=lambda index: (
                        self.environment.nodes[index].level,
                        -remaining[index],
                        index,
                    ),
                )
            elif mode == "rtbl":
                round_id = 1 + sum(self._rtbl_selection_counts)

                def rtbl_score(index: int) -> tuple[float, float, int]:
                    history = snapshot.node_availability_history[index]
                    mean = sum(history) / len(history) if history else 1.0
                    confidence = math.sqrt(
                        2.0 * math.log(round_id + 1) / max(len(history) + self._rtbl_selection_counts[index], 1)
                    )
                    conservative = max(0.0, mean - confidence)
                    load_cost = snapshot.node_loads[index] + (
                        dnn.tasks[task_index].cpu_need
                        / max(self.environment.nodes[index].max_cpu, 1.0)
                    )
                    utility = 2200.0 * conservative
                    return utility - self._rtbl_q * load_cost, remaining[index], -index

                node_index = max(candidates, key=rtbl_score)
                self._rtbl_selection_counts[node_index] += 1
            else:
                node_index = max(
                    candidates,
                    key=lambda index: (remaining[index], -len(candidates), -index),
                )
            assignment[task_index] = node_index
            remaining[node_index] -= dnn.tasks[task_index].cpu_need
        return tuple(assignment)

    def _random_assignment(self, dnn_index: int, snapshot, rng: random.Random) -> tuple[int, ...] | None:
        dnn = self.environment.ds[dnn_index]
        legal = self._legal_nodes(dnn_index, snapshot)
        order, predecessors = self._topology(dnn_index)
        remaining = list(snapshot.node_available_cpu)
        assignment = [-1 for _ in dnn.tasks]
        for task_index in order:
            minimum_level = max(
                (self.environment.nodes[assignment[parent]].level for parent in predecessors[task_index]),
                default=1,
            )
            candidates = [
                index for index in legal
                if self.environment.nodes[index].level >= minimum_level
                and remaining[index] >= dnn.tasks[task_index].cpu_need
            ]
            if not candidates:
                return None
            selected = rng.choice(candidates)
            assignment[task_index] = selected
            remaining[selected] -= dnn.tasks[task_index].cpu_need
        return tuple(assignment)

    def _beam_assignments(self, dnn_index: int, snapshot, mode: str) -> tuple[tuple[int, ...], ...]:
        dnn = self.environment.ds[dnn_index]
        legal = self._legal_nodes(dnn_index, snapshot)
        order, predecessors = self._topology(dnn_index)
        beam: list[tuple[list[int], list[float]]] = [
            ([-1 for _ in dnn.tasks], list(snapshot.node_available_cpu))
        ]
        for task_index in order:
            expanded: list[tuple[list[int], list[float]]] = []
            for assignment, remaining in beam:
                minimum_level = max(
                    (
                        self.environment.nodes[assignment[parent]].level
                        for parent in predecessors[task_index]
                    ),
                    default=1,
                )
                candidates = [
                    index for index in legal
                    if self.environment.nodes[index].level >= minimum_level
                    and remaining[index] >= dnn.tasks[task_index].cpu_need
                ]
                candidates.sort(
                    key=(
                        (lambda index: (
                            self.environment.nodes[index].level,
                            -remaining[index],
                            index,
                        ))
                        if mode == "localfirst"
                        else (lambda index: (-remaining[index], index))
                    )
                )
                for node_index in candidates[:3]:
                    next_assignment = list(assignment)
                    next_remaining = list(remaining)
                    next_assignment[task_index] = node_index
                    next_remaining[node_index] -= dnn.tasks[task_index].cpu_need
                    expanded.append((next_assignment, next_remaining))
            if not expanded:
                return ()
            if mode == "localfirst":
                expanded.sort(
                    key=lambda item: (
                        sum(
                            self.environment.nodes[index].level
                            for index in item[0]
                            if index >= 0
                        ),
                        -sum(item[1]),
                        tuple(item[0]),
                    )
                )
            else:
                expanded.sort(
                    key=lambda item: (
                        -min(item[1]),
                        -sum(item[1]),
                        tuple(item[0]),
                    )
                )
            beam = expanded[: self.config.beam_width]
        return tuple(tuple(assignment) for assignment, _ in beam)

    @staticmethod
    def _score(candidate: CandidateStateSnapshot, request: InferenceRequest) -> float:
        return PerRequestCustomizedDispatcher._score(candidate, request)

    def _candidate_assignments(self, dnn_index: int, request: InferenceRequest, snapshot):
        algorithm = self.config.algorithm
        if algorithm == "localfirst":
            return tuple(
                dict.fromkeys(
                    (
                        *self._beam_assignments(dnn_index, snapshot, "localfirst"),
                        *self._beam_assignments(dnn_index, snapshot, "max_resource_fast"),
                    )
                )
            )
        if algorithm == "max_resource_fast":
            return self._beam_assignments(dnn_index, snapshot, algorithm)
        if algorithm == "rtbl":
            assignment = self._greedy_assignment(dnn_index, snapshot, algorithm)
            return () if assignment is None else (assignment,)
        rng = random.Random(self.config.random_seed + request.request_id)
        if algorithm == "random":
            return tuple(
                assignment
                for assignment in (
                    self._random_assignment(dnn_index, snapshot, rng)
                    for _ in range(self.config.evaluation_budget)
                )
                if assignment is not None
            )

        seed = self._greedy_assignment(dnn_index, snapshot, "max_resource_fast")
        if seed is None:
            seed = self._random_assignment(dnn_index, snapshot, rng)
        if seed is None:
            return ()
        current = seed
        current_state = self.environment.predict_candidate_state(dnn_index, list(current), snapshot)
        current_score = self._score(current_state, request) if current_state.is_strictly_feasible else -math.inf
        best = current
        best_score = current_score
        visited = [current]
        legal = self._legal_nodes(dnn_index, snapshot)
        for step in range(1, self.config.evaluation_budget):
            neighbor = list(current)
            neighbor[rng.randrange(len(neighbor))] = rng.choice(legal)
            state = self.environment.predict_candidate_state(dnn_index, neighbor, snapshot)
            score = self._score(state, request) if state.is_strictly_feasible else -math.inf
            temperature = max(0.01, 1.0 - step / self.config.evaluation_budget)
            if score >= current_score or rng.random() < math.exp(
                min(0.0, score - current_score) / temperature
            ):
                current = tuple(neighbor)
                current_score = score
            if score > best_score:
                best = tuple(neighbor)
                best_score = score
            visited.append(tuple(neighbor))
        return tuple(dict.fromkeys((*visited, best)))

    def dispatch(
        self,
        dnn_index: int,
        request: InferenceRequest,
        repository: PlanRepository,
    ) -> DispatchDecision:
        started_at = self._clock()
        snapshot = self.environment.capture_observation_snapshot("online_admission")
        assignments = self._candidate_assignments(dnn_index, request, snapshot)
        evaluated: list[tuple[float, tuple[int, ...], CandidateStateSnapshot]] = []
        for assignment in assignments:
            candidate = self.environment.predict_candidate_state(
                dnn_index, list(assignment), snapshot
            )
            self.evaluations += 1
            if candidate.is_strictly_feasible:
                evaluated.append((self._score(candidate, request), assignment, candidate))
        if self.config.algorithm in {"random", "sa"}:
            evaluated.sort(key=lambda item: (-item[0], item[1]))
        committed = None
        selected_assignment = None
        rejection_reason = "no_feasible_candidate"
        for _, assignment, _ in evaluated:
            try:
                committed = self.environment.commit_assignment(
                    dnn_index,
                    list(assignment),
                    expected_environment_state_version=snapshot.environment_state_version,
                )
            except (InfeasibleAssignmentError, StaleEnvironmentStateError) as error:
                rejection_reason = str(error)
                continue
            selected_assignment = assignment
            self.selected += 1
            break
        if self.config.algorithm == "rtbl":
            self._rtbl_q = max(
                0.0,
                self._rtbl_q
                + ((committed.estimated_delay_ms / max(request.deadline_ms, 1.0)) if committed else 1.0)
                - 1.0,
            )
        elapsed_ms = (self._clock() - started_at) * 1000.0
        accepted = committed is not None
        return DispatchDecision(
            accepted=accepted,
            path=f"baseline_{self.config.algorithm}" if accepted else "rejected",
            rejection_reason=None if accepted else rejection_reason,
            plan_id=None,
            assignment=selected_assignment,
            candidate=committed,
            repository_version=repository.repository_version,
            evaluated_environment_state_version=snapshot.environment_state_version,
            committed_environment_state_version=(
                self.environment.environment_state_version if accepted else None
            ),
            full_revalidation_count=len(assignments),
            online_runtime_ms=elapsed_ms,
            configured_online_budget_ms=0.0,
            request_online_budget_ms=0.0,
            feasible_candidate_count=len(evaluated),
            budget_exhausted=False,
            early_stop_reason=None,
            best_utility_after_each_evaluation=tuple(
                score for score, _, _ in sorted(evaluated)
            ),
            control_decision_id=None,
        )
