from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.environment import Environment
from experiment_framework.artifacts import suite_lock, validate_suite_output, write_run_manifest
from experiment_framework.identity import task_id
from experiment_runner import _pareto_hypervolume
from proposed import AllDNNRefactor, CustomizedSearchConfig
from semantics import ACTIVE_OBJECTIVE_SEMANTICS_VERSION


PROTOCOL_VERSION = "isolated_snapshot_v1"
SUITE = "search_quality"
NUMERIC_FIELDS = (
    "instance_count",
    "feasible_instances",
    "feasible_rate",
    "mean_candidate_count",
    "mean_evaluations",
    "mean_constraint_rejection_rate",
    "mean_nonconvergence_rate",
    "mean_archive_size",
    "mean_hypervolume",
    "avg_estimated_delay_ms",
    "avg_operational_stability_score",
    "avg_total_energy",
    "search_runtime_ms",
)


@dataclass(frozen=True)
class SearchQualityScenario:
    name: str
    method: str
    evidence_roles: tuple[str, ...]
    population_size: int
    generations: int
    archive_capacity: int
    instance_count: int = 30
    use_dag_aware_initialization: bool = True
    use_dag_block_crossover: bool = True
    use_skew_mutation: bool = True
    use_elite_local_search: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"customized", "proposed", "sa", "no_dag_operators"}:
            raise ValueError("unsupported search-quality method")
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("population_size must be an even integer >= 2")
        if self.generations <= 0 or self.archive_capacity <= 0 or self.instance_count <= 0:
            raise ValueError("search-quality budgets and instance count must be positive")


@dataclass(frozen=True)
class _SearchResult:
    states: tuple[object, ...]
    evaluations: int
    constraint_rejected: int
    nonconverged: int
    archive_size: int
    wall_time_ms: float


def build_scenarios(*, quick: bool = False, instance_count: int = 30) -> tuple[SearchQualityScenario, ...]:
    if quick:
        instance_count = min(instance_count, 2)
        return (
            SearchQualityScenario("customized_quick", "customized", ("budget", "algorithm", "operator"), 4, 2, 8, instance_count),
            SearchQualityScenario(
                "proposed_quick",
                "proposed",
                ("algorithm", "operator"),
                4,
                2,
                8,
                instance_count,
                False,
                False,
                False,
                False,
            ),
            SearchQualityScenario("sa_quick", "sa", ("algorithm",), 4, 2, 8, instance_count),
            SearchQualityScenario(
                "no_dag_quick",
                "no_dag_operators",
                ("operator",),
                4,
                2,
                8,
                instance_count,
                False,
                False,
                False,
                True,
            ),
        )
    return (
        SearchQualityScenario("customized_medium", "customized", ("budget", "algorithm", "operator"), 12, 4, 64, instance_count),
        SearchQualityScenario(
            "proposed_medium",
            "proposed",
            ("algorithm", "operator"),
            12,
            4,
            64,
            instance_count,
            False,
            False,
            False,
            False,
        ),
        SearchQualityScenario("sa_medium", "sa", ("algorithm",), 12, 4, 64, instance_count),
        SearchQualityScenario(
            "no_dag_medium",
            "no_dag_operators",
            ("operator",),
            12,
            4,
            64,
            instance_count,
            False,
            False,
            False,
            True,
        ),
    )


def _background_environment(seed: int, instance_index: int) -> tuple[Environment, object, float]:
    random.seed(seed)
    environment = Environment(t_max=1, verbose=False)
    background_load = (0.0, 0.35, 0.65)[instance_index % 3]
    for node in environment.nodes:
        node.cpu = max(0, round(node.cpu * (1.0 - background_load)))
        node.load_ratio = background_load
        node.heat = background_load
    for link in environment.link_nodes:
        link.load_ratio = background_load
    for physical_link in environment.physical_links:
        physical_link.load_ratio = background_load
        physical_link.heat = background_load
    environment.refresh_effective_bandwidth()
    environment.refresh_dynamic_stability()
    snapshot = environment.capture_observation_snapshot("planning")
    return environment, snapshot, background_load


def _nondominated(states: Iterable[object]) -> tuple[object, ...]:
    unique = {tuple(state.assignment): state for state in states}

    def dominates(first: object, second: object) -> bool:
        left = (first.operational_stability, first.delay_satisfaction, first.energy_satisfaction)
        right = (second.operational_stability, second.delay_satisfaction, second.energy_satisfaction)
        return all(a >= b - 1e-12 for a, b in zip(left, right)) and any(
            a > b + 1e-12 for a, b in zip(left, right)
        )

    return tuple(
        state
        for assignment, state in unique.items()
        if not any(
            other_assignment != assignment and dominates(other, state)
            for other_assignment, other in unique.items()
        )
    )


def _run_evolutionary(
    environment: Environment,
    snapshot: object,
    scenario: SearchQualityScenario,
    seed: int,
) -> _SearchResult:
    refactor = AllDNNRefactor(environment)
    refactor.use_dag_aware_initialization = scenario.use_dag_aware_initialization
    refactor.use_dag_block_crossover = scenario.use_dag_block_crossover
    refactor.use_skew_mutation = scenario.use_skew_mutation
    refactor.use_elite_local_search = scenario.use_elite_local_search
    result = refactor.generate_customized_candidates(
        dnn_index=0,
        snapshot=snapshot,
        search_config=CustomizedSearchConfig(
            population_size=scenario.population_size,
            generations=scenario.generations,
            archive_capacity=scenario.archive_capacity,
            random_seed=seed,
        ),
    )
    states = tuple(
        environment.predict_candidate_state(0, list(candidate.assignment), snapshot)
        for candidate in result.candidates
    )
    return _SearchResult(
        states=states,
        evaluations=result.evaluations,
        constraint_rejected=result.constraint_rejected_evaluations,
        nonconverged=result.nonconverged_evaluations,
        archive_size=result.archive_size_before_selection,
        wall_time_ms=result.wall_time_ms,
    )


def _run_sa(
    environment: Environment,
    snapshot: object,
    scenario: SearchQualityScenario,
    seed: int,
) -> _SearchResult:
    started = time.monotonic()
    refactor = AllDNNRefactor(environment)
    global_state = random.getstate()
    previous_snapshot = refactor._active_search_snapshot
    evaluated: dict[tuple[int, ...], object] = {}
    rejected = 0
    nonconverged = 0

    def evaluate(assignment: list[int]) -> object:
        nonlocal rejected, nonconverged
        key = tuple(assignment)
        if key in evaluated:
            return evaluated[key]
        state = environment.predict_candidate_state(0, assignment, snapshot)
        evaluated[key] = state
        if not state.fixed_point_converged:
            nonconverged += 1
        elif not state.is_strictly_feasible:
            rejected += 1
        return state

    try:
        random.seed(seed)
        refactor._active_search_snapshot = snapshot
        context = refactor._build_dnn_context(0, snapshot)
        seeds: list[list[int]] = []
        random_assignment = refactor._random_valid_assignment(0)
        if random_assignment is not None:
            seeds.append(random_assignment)
        for mode in ("cloud", "local", "resource", "stability", "topology", "random"):
            assignment = refactor._heuristic_assignment(0, context, mode)
            if refactor._is_assignment_valid(0, assignment):
                seeds.append(assignment)
        if not seeds:
            return _SearchResult((), 0, 0, 0, 0, (time.monotonic() - started) * 1000.0)
        for assignment in seeds:
            evaluate(assignment)
        current = max(
            seeds,
            key=lambda assignment: sum(
                (
                    evaluate(assignment).operational_stability,
                    evaluate(assignment).delay_satisfaction,
                    evaluate(assignment).energy_satisfaction,
                )
            ),
        )
        current = list(current)
        current_score = sum(
            (
                evaluate(current).operational_stability,
                evaluate(current).delay_satisfaction,
                evaluate(current).energy_satisfaction,
            )
        )
        steps = max(1, scenario.population_size * scenario.generations)
        for step in range(steps):
            progress = step / max(steps - 1, 1)
            temperature = 1.0 * (0.01**progress)
            candidate = refactor._sa_neighbor_assignment(0, current)
            state = evaluate(candidate)
            score = sum(
                (
                    state.operational_stability,
                    state.delay_satisfaction,
                    state.energy_satisfaction,
                )
            ) if state.is_strictly_feasible else -math.inf
            delta = score - current_score
            if delta >= 0 or random.random() < math.exp(delta / max(temperature, 1e-9)):
                current = candidate
                current_score = score
        feasible = (
            state
            for state in evaluated.values()
            if state.fixed_point_converged and state.is_strictly_feasible
        )
        front = _nondominated(feasible)[: scenario.archive_capacity]
        return _SearchResult(
            states=front,
            evaluations=len(evaluated),
            constraint_rejected=rejected,
            nonconverged=nonconverged,
            archive_size=len(front),
            wall_time_ms=(time.monotonic() - started) * 1000.0,
        )
    finally:
        refactor._active_search_snapshot = previous_snapshot
        random.setstate(global_state)


def run_once(scenario: SearchQualityScenario, seed: int) -> dict[str, float | int | str]:
    delays: list[float] = []
    stability: list[float] = []
    energy: list[float] = []
    candidate_counts: list[float] = []
    evaluations: list[float] = []
    rejection_rates: list[float] = []
    nonconvergence_rates: list[float] = []
    archive_sizes: list[float] = []
    hypervolumes: list[float] = []
    runtime_ms = 0.0

    for instance_index in range(scenario.instance_count):
        instance_seed = seed + instance_index * 1_000_003
        environment, snapshot, _ = _background_environment(instance_seed, instance_index)
        initial_slot = environment.current_slot
        initial_version = environment.environment_state_version
        result = (
            _run_sa(environment, snapshot, scenario, instance_seed + 500_009)
            if scenario.method == "sa"
            else _run_evolutionary(environment, snapshot, scenario, instance_seed + 500_009)
        )
        if environment.current_slot != initial_slot or environment.environment_state_version != initial_version or environment.running_dnns:
            raise RuntimeError("search-quality method mutated the environment")
        runtime_ms += result.wall_time_ms
        candidate_counts.append(float(len(result.states)))
        evaluations.append(float(result.evaluations))
        denominator = max(result.evaluations, 1)
        rejection_rates.append(result.constraint_rejected / denominator)
        nonconvergence_rates.append(result.nonconverged / denominator)
        archive_sizes.append(float(result.archive_size))
        hypervolumes.append(
            _pareto_hypervolume(
                [
                    {
                        "operational_stability_norm": state.operational_stability,
                        "delay_norm": state.delay_satisfaction,
                        "energy_norm": state.energy_satisfaction,
                    }
                    for state in result.states
                ]
            )
        )
        if result.states:
            selected = max(
                result.states,
                key=lambda state: state.operational_stability + state.delay_satisfaction + state.energy_satisfaction,
            )
            delays.append(selected.estimated_delay_ms)
            stability.append(selected.operational_stability)
            energy.append(selected.total_energy)

    mean = lambda values: statistics.fmean(values) if values else 0.0
    return {
        "instance_count": scenario.instance_count,
        "feasible_instances": len(delays),
        "feasible_rate": len(delays) / scenario.instance_count,
        "mean_candidate_count": mean(candidate_counts),
        "mean_evaluations": mean(evaluations),
        "mean_constraint_rejection_rate": mean(rejection_rates),
        "mean_nonconvergence_rate": mean(nonconvergence_rates),
        "mean_archive_size": mean(archive_sizes),
        "mean_hypervolume": mean(hypervolumes),
        "avg_estimated_delay_ms": mean(delays),
        "avg_operational_stability_score": mean(stability),
        "avg_total_energy": mean(energy),
        "search_runtime_ms": runtime_ms,
    }


def _append_row(path: Path, row: dict[str, object], fields: list[str] | None) -> list[str]:
    current = list(row)
    if fields is not None and fields != current:
        raise ValueError("existing search-quality result schema does not match")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=current)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    return current


def _write_summary(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["algorithm"]), []).append(row)
    fields = ["experiment_layer", "protocol_version", "suite", "scenario", "algorithm", "runs"]
    for metric in NUMERIC_FIELDS:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (scenario, algorithm), group in sorted(groups.items()):
            output: dict[str, object] = {
                "experiment_layer": "search_quality",
                "protocol_version": PROTOCOL_VERSION,
                "suite": SUITE,
                "scenario": scenario,
                "algorithm": algorithm,
                "runs": len(group),
            }
            for metric in NUMERIC_FIELDS:
                values = [float(row[metric]) for row in group]
                output[f"{metric}_mean"] = statistics.fmean(values)
                output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow(output)


def _run_scenarios_unlocked(
    scenarios: Iterable[SearchQualityScenario],
    *,
    repeats: int,
    base_seed: int,
    output_dir: Path,
) -> None:
    scenarios = tuple(scenarios)
    raw_path = output_dir / "raw_results.csv"
    completed: set[str] = set()
    fields: list[str] | None = None
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            existing = list(reader)
            fields = reader.fieldnames
        completed = {row["task_id"] for row in existing}
    for scenario in scenarios:
        for repeat in range(repeats):
            seed = base_seed + repeat
            identifier = task_id(
                experiment_layer="search_quality",
                protocol_version=PROTOCOL_VERSION,
                suite=SUITE,
                scenario=asdict(scenario),
                algorithm=scenario.method,
                repeat=repeat,
                seed=seed,
            )
            if identifier in completed:
                print(f"[skip] {identifier}")
                continue
            print(f"[run] {identifier}")
            metrics = run_once(scenario, seed)
            row: dict[str, object] = {
                "task_id": identifier,
                "experiment_layer": "search_quality",
                "protocol_version": PROTOCOL_VERSION,
                "suite": SUITE,
                "scenario": scenario.name,
                "comparison_group": "search_quality",
                "comparison_method": scenario.name,
                "algorithm": scenario.method,
                "repeat": repeat,
                "seed": seed,
                "search_budget_id": f"p{scenario.population_size}_g{scenario.generations}_a{scenario.archive_capacity}",
                "objective_semantics_version": ACTIVE_OBJECTIVE_SEMANTICS_VERSION,
                **metrics,
            }
            fields = _append_row(raw_path, row, fields)
            completed.add(identifier)
    _write_summary(raw_path, output_dir / "summary.csv")
    write_run_manifest(
        output_dir,
        layer="search_quality",
        suite=SUITE,
        scenarios=scenarios,
        repeats=repeats,
        base_seed=base_seed,
        timing_mode="measured",
        semantic_versions={"objective_semantics_version": ACTIVE_OBJECTIVE_SEMANTICS_VERSION},
    )
    report = validate_suite_output(output_dir, expected_runs=len(scenarios) * repeats)
    if not report["valid"]:
        raise RuntimeError(f"search-quality result validation failed: {report['failures']}")


def run_scenarios(
    scenarios: Iterable[SearchQualityScenario],
    *,
    repeats: int,
    base_seed: int,
    output_dir: Path,
) -> None:
    with suite_lock(output_dir):
        _run_scenarios_unlocked(
            scenarios,
            repeats=repeats,
            base_seed=base_seed,
            output_dir=output_dir,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Converged isolated-snapshot search-quality experiments")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/search_quality"))
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats <= 0 or args.instances <= 0:
        raise SystemExit("repeats and instances must be positive")
    if args.quick:
        args.repeats = 1
    run_scenarios(
        build_scenarios(quick=args.quick, instance_count=args.instances),
        repeats=args.repeats,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
