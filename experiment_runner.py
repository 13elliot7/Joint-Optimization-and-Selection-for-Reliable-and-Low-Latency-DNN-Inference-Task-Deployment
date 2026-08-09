from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from core.environment import Environment
from core.topology import TopologyConfig
from metrics import ExperimentMetrics
from proposed import AllDNNRefactor, CustomizedSearchConfig
from rtbl.scheduler import RTBLRunner
from semantics import ACTIVE_OBJECTIVE_SEMANTICS_VERSION


def run_periodic_experiment(environment, catalog, arrival_trace, config):
    """运行新版周期规划事件循环；旧基线结果口径保持不变。"""
    from periodic_experiment import PeriodicExperimentRunner

    return PeriodicExperimentRunner(
        environment,
        catalog,
        arrival_trace,
        config,
    ).run()


PARETO_ALGORITHMS = ("proposed", "customized", "random", "sa")
DEPLOYMENT_ALGORITHMS = ("proposed", "customized", "random", "localfirst", "maxresource_fast", "rtbl") # maxresource
ALGORITHMS = tuple(dict.fromkeys((*PARETO_ALGORITHMS, *DEPLOYMENT_ALGORITHMS)))
BASE_METRIC_FIELDS = (
    "avg_estimated_delay",
    "avg_operational_stability_score",
    "avg_total_energy",
    "rejected_or_failed_count",
    "runtime_ms",
)
PARETO_METRIC_FIELDS = (
    "pareto_point_count",
    "pareto_point_count_mean",
    "pareto_hypervolume",
    "pareto_hypervolume_sum",
    "pareto_hypervolume_std",
    "pareto_postprocess_ms",
)
SENSITIVITY_METRIC_FIELDS = (
    "quality_instance_count",
    "quality_feasible_instances",
    "quality_feasible_rate",
    "quality_mean_candidate_count",
    "quality_mean_evaluations",
    "quality_mean_constraint_rejection_rate",
    "quality_mean_nonconvergence_rate",
    "quality_mean_archive_size",
    "quality_mean_hypervolume",
    "quality_mean_background_load",
)


@dataclass(frozen=True)
class Scenario:
    suite: str
    name: str
    dnn_count: int = 30
    population_size: int = 60
    iteration_limit: int = 200
    mutation_probability: float = 0.25
    crossover_probability: float = 0.50
    archive_capacity: int = 64
    sensitivity_instance_count: int = 1
    alpha_r: float = 0.80
    beta_r: float = 0.50
    alpha_l: float = 0.70
    beta_l: float = 0.40
    lambda_h: float = 0.70
    lambda_g: float = 0.70
    node_stability_weight: float = 0.50
    link_stability_weight: float = 0.50
    preference: str = "adaptive"
    use_dag_aware_initialization: bool = True
    use_dag_block_crossover: bool = True
    use_skew_mutation: bool = True
    use_elite_local_search: bool = True
    cloud_count: int = 1
    edge_count: int = 10
    user_count: int | None = None
    edge_link_factor: float = 2.0

    def environment_kwargs(self) -> Dict[str, float | int | bool]:
        kwargs = {
            "t_max": self.dnn_count,
            "alpha_r": self.alpha_r,
            "beta_r": self.beta_r,
            "alpha_l": self.alpha_l,
            "beta_l": self.beta_l,
            "lambda_h": self.lambda_h,
            "lambda_g": self.lambda_g,
            "node_stability_weight": self.node_stability_weight,
            "link_stability_weight": self.link_stability_weight,
            "verbose": False,
        }
        topology_config = TopologyConfig(
            cloud_count=self.cloud_count,
            edge_count=self.edge_count,
            user_count=self.user_count,
            edge_link_factor=self.edge_link_factor,
        )
        if topology_config != TopologyConfig():
            kwargs["topology_config"] = topology_config
        return kwargs


@dataclass
class RunOutput:
    metrics: ExperimentMetrics
    pareto_points: List[Dict[str, float | int | str]]
    diagnostics: Dict[str, float | int | str] | None = None


def _parse_csv_values(raw: str, value_type):
    return [value_type(value.strip()) for value in raw.split(",") if value.strip()]


def _scenario_key(scenario: Scenario) -> str:
    encoded = json.dumps(asdict(scenario), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:10]
    return f"{scenario.suite}:{scenario.name}:{digest}"


def _configure_search(refactor: AllDNNRefactor, scenario: Scenario) -> None:
    refactor.pop_size = scenario.population_size
    refactor.iteration_limit = scenario.iteration_limit
    refactor.mutate_pm = scenario.mutation_probability
    refactor.cross_over_pm = scenario.crossover_probability
    refactor.preference = scenario.preference
    refactor.use_dag_aware_initialization = scenario.use_dag_aware_initialization
    refactor.use_dag_block_crossover = scenario.use_dag_block_crossover
    refactor.use_skew_mutation = scenario.use_skew_mutation
    refactor.use_elite_local_search = scenario.use_elite_local_search
    refactor.collect_pareto_points = scenario.suite == "pareto"
    refactor.function1_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.function2_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.function3_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.function4_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.estimated_delay_values = [float("inf") for _ in range(2 * refactor.pop_size)]
    refactor.total_energy_values = [float("inf") for _ in range(2 * refactor.pop_size)]
    refactor.deadline_feasible_values = [False for _ in range(2 * refactor.pop_size)]
    refactor.constraint_violations = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.distance = [[0 for _ in range(2 * refactor.pop_size)] for _ in range(1000)]


def _apply_dynamic_parameters(env: Environment, scenario: Scenario) -> None:
    for field in (
        "alpha_r",
        "beta_r",
        "alpha_l",
        "beta_l",
        "lambda_h",
        "lambda_g",
    ):
        setattr(env, field, getattr(scenario, field))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _pareto_hypervolume(
    points: Sequence[Dict[str, float | int | str]],
    objectives: Sequence[str] = (
        "operational_stability_norm",
        "delay_norm",
        "energy_norm",
    ),
) -> float:
    """Compute hypervolume against zero; fall back to deterministic sampling for large fronts."""
    feasible_points = [
        {objective: max(0.0, float(point[objective])) for objective in objectives}
        for point in points
    ]
    if not feasible_points:
        return 0.0
    axes = []
    for objective in objectives:
        values = sorted({0.0, *(point[objective] for point in feasible_points)})
        axes.append(values)
    cell_count = 1
    for axis in axes:
        cell_count *= max(len(axis) - 1, 1)
    if cell_count > 500_000:
        upper_bounds = [max(point[objective] for point in feasible_points) for objective in objectives]
        box_volume = 1.0
        for upper_bound in upper_bounds:
            box_volume *= upper_bound
        if box_volume == 0:
            return 0.0
        samples = 20_000
        rng = random.Random(20260613)
        dominated = 0
        for _ in range(samples):
            probe = [rng.random() * upper_bound for upper_bound in upper_bounds]
            if any(
                all(point[objective] >= probe[idx] for idx, objective in enumerate(objectives))
                for point in feasible_points
            ):
                dominated += 1
        return box_volume * dominated / samples
    hv = 0.0
    ranges = [range(1, len(axis)) for axis in axes]
    for indices in itertools.product(*ranges):
        upper = tuple(axes[dim][idx] for dim, idx in enumerate(indices))
        if any(
            all(point[objective] >= upper[idx] for idx, objective in enumerate(objectives))
            for point in feasible_points
        ):
            volume = 1.0
            for dim, idx in enumerate(indices):
                volume *= axes[dim][idx] - axes[dim][idx - 1]
            hv += volume
    return hv


def _group_pareto_points_by_dnn(
    points: Sequence[Dict[str, float | int | str]],
) -> Dict[int, List[Dict[str, float | int | str]]]:
    groups: Dict[int, List[Dict[str, float | int | str]]] = {}
    for point in points:
        groups.setdefault(int(point["dnn_index"]), []).append(point)
    return groups


def _normalize_pareto_points(
    points: Sequence[Dict[str, float | int | str]],
) -> List[Dict[str, float | int | str]]:
    """复用搜索阶段的统一满意度表示，不再按算法前沿二次归一化。"""
    if not points:
        return []
    normalized = []
    for point in points:
        required = (
            "operational_stability_satisfaction",
            "delay_satisfaction",
            "energy_satisfaction",
        )
        missing = [field for field in required if field not in point]
        if missing:
            raise ValueError(
                "Pareto points must use the unified three-objective representation "
                f"(missing: {', '.join(missing)})"
            )
        enriched = dict(point)
        stability = float(point["operational_stability_satisfaction"])
        enriched["operational_stability_norm"] = _clamp(stability)
        enriched["delay_norm"] = _clamp(float(point["delay_satisfaction"]))
        enriched["energy_norm"] = _clamp(float(point["energy_satisfaction"]))
        normalized.append(enriched)
    return normalized


def _pareto_hypervolume_by_dnn(
    points: Sequence[Dict[str, float | int | str]],
    *,
    enabled: bool,
) -> Dict[str, float | str]:
    if not enabled:
        raise RuntimeError("Hypervolume is only available in the pareto suite")
    started_at = __import__("time").monotonic()
    groups = _group_pareto_points_by_dnn(points)
    point_counts = [len(group_points) for group_points in groups.values()]
    hypervolumes = [
        _pareto_hypervolume(
            _normalize_pareto_points(group_points),
            objectives=(
                "operational_stability_norm",
                "delay_norm",
                "energy_norm",
            ),
        )
        for group_points in groups.values()
    ]
    return {
        "pareto_point_count": float(len(points)),
        "pareto_point_count_mean": statistics.fmean(point_counts) if point_counts else 0.0,
        "pareto_hypervolume": statistics.fmean(hypervolumes) if hypervolumes else 0.0,
        "pareto_hypervolume_sum": sum(hypervolumes),
        "pareto_hypervolume_std": statistics.stdev(hypervolumes) if len(hypervolumes) > 1 else 0.0,
        "pareto_postprocess_ms": (__import__("time").monotonic() - started_at) * 1000.0,
        "pareto_hv_version": "3d_oss_delay_energy_v1",
    }


def _run_isolated_sensitivity(scenario: Scenario, seed: int) -> RunOutput:
    """在共享实例seed库上评估单DNN搜索质量，不提交资源或推进环境。"""
    delays: List[float] = []
    stability: List[float] = []
    energy: List[float] = []
    candidate_counts: List[float] = []
    evaluation_counts: List[float] = []
    constraint_rejection_rates: List[float] = []
    nonconvergence_rates: List[float] = []
    archive_sizes: List[float] = []
    hypervolumes: List[float] = []
    total_runtime_ms = 0.0
    background_loads: List[float] = []

    for instance_index in range(scenario.sensitivity_instance_count):
        instance_seed = seed + instance_index * 1_000_003
        random.seed(instance_seed)
        kwargs = scenario.environment_kwargs()
        kwargs["t_max"] = 1
        env = Environment(**kwargs)
        background_load = (0.0, 0.35, 0.65)[instance_index % 3]
        background_loads.append(background_load)
        for node in env.nodes:
            node.cpu = max(0, round(node.cpu * (1.0 - background_load)))
            node.load_ratio = background_load
            node.heat = background_load
        for link in env.link_nodes:
            link.load_ratio = background_load
        for physical_link in env.physical_links:
            physical_link.load_ratio = background_load
            physical_link.heat = background_load
        env.refresh_effective_bandwidth()
        env.refresh_dynamic_stability()
        refactor = AllDNNRefactor(env)
        _configure_search(refactor, scenario)
        snapshot = env.capture_observation_snapshot("planning")
        initial_slot = env.current_slot
        initial_state_version = env.environment_state_version
        result = refactor.generate_customized_candidates(
            dnn_index=0,
            snapshot=snapshot,
            search_config=CustomizedSearchConfig(
                population_size=scenario.population_size,
                generations=scenario.iteration_limit,
                archive_capacity=scenario.archive_capacity,
                random_seed=instance_seed + 500_009,
            ),
        )
        if (
            env.current_slot != initial_slot
            or env.environment_state_version != initial_state_version
            or env.running_dnns
        ):
            raise RuntimeError("isolated sensitivity search mutated the environment")

        total_runtime_ms += result.wall_time_ms
        candidate_counts.append(float(len(result.candidates)))
        evaluation_counts.append(float(result.evaluations))
        denominator = max(result.evaluations, 1)
        constraint_rejection_rates.append(
            result.constraint_rejected_evaluations / denominator
        )
        nonconvergence_rates.append(result.nonconverged_evaluations / denominator)
        archive_sizes.append(float(result.archive_size_before_selection))
        hypervolumes.append(
            _pareto_hypervolume(
                [
                    {
                        "operational_stability_norm": candidate.objectives[0],
                        "delay_norm": candidate.objectives[1],
                        "energy_norm": candidate.objectives[2],
                    }
                    for candidate in result.candidates
                ]
            )
        )
        if not result.candidates:
            continue
        selected = max(result.candidates, key=lambda candidate: sum(candidate.objectives))
        state = env.predict_candidate_state(0, list(selected.assignment), snapshot)
        delays.append(state.estimated_delay_ms)
        stability.append(state.operational_stability)
        energy.append(state.total_energy)

    instance_count = scenario.sensitivity_instance_count
    feasible_count = len(delays)
    metrics = ExperimentMetrics(
        avg_delay=_mean(delays),
        avg_operational_stability_score=_mean(stability),
        avg_energy=_mean(energy),
        failure_count=instance_count - feasible_count,
        runtime_ms=round(total_runtime_ms),
    )
    return RunOutput(
        metrics=metrics,
        pareto_points=[],
        diagnostics={
            "experiment_layer": "algorithm_quality",
            "workload_mode": "isolated_single_dnn_instance_bank",
            "sensitivity_protocol_version": "isolated_single_dnn_v1",
            "quality_instance_count": instance_count,
            "quality_feasible_instances": feasible_count,
            "quality_feasible_rate": feasible_count / instance_count,
            "quality_mean_candidate_count": _mean(candidate_counts),
            "quality_mean_evaluations": _mean(evaluation_counts),
            "quality_mean_constraint_rejection_rate": _mean(
                constraint_rejection_rates
            ),
            "quality_mean_nonconvergence_rate": _mean(nonconvergence_rates),
            "quality_mean_archive_size": _mean(archive_sizes),
            "quality_mean_hypervolume": _mean(hypervolumes),
            "quality_mean_background_load": _mean(background_loads),
            "quality_load_strata": ",".join(
                ("low", "medium", "high")[: min(instance_count, 3)]
            ),
        },
    )


def run_once(algorithm: str, scenario: Scenario, seed: int) -> RunOutput:
    """Run one reproducible algorithm/scenario/seed combination."""
    if scenario.suite == "sensitivity":
        if algorithm != "customized":
            raise ValueError("isolated sensitivity currently supports customized only")
        return _run_isolated_sensitivity(scenario, seed)
    random.seed(seed)
    env = Environment(**scenario.environment_kwargs())
    initial_nodes = env.clone_nodes()
    refactor = AllDNNRefactor(env)
    _configure_search(refactor, scenario)
    runners = {
        "random": lambda: refactor.run_random(initial_nodes),
        "maxresource": lambda: refactor.run_max_resource(initial_nodes),
        "maxresource_fast": lambda: refactor.run_max_resource_fast(initial_nodes),
        "localfirst": lambda: refactor.run_local_first(initial_nodes),
        "rtbl": lambda: RTBLRunner(env).run_dynamic(initial_nodes),
        "sa": lambda: refactor.run_simulated_annealing(initial_nodes),
        "proposed": refactor.run_proposed,
        "customized": refactor.run_customized_proposed,
    }
    metrics = runners[algorithm]()
    return RunOutput(
        metrics=metrics,
        pareto_points=list(refactor.pareto_points) if scenario.suite == "pareto" else [],
        diagnostics={
            "experiment_layer": "legacy_algorithm_auxiliary",
            "workload_mode": "sequential_one_dnn_per_slot",
        },
    )


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    if args.suite == "overall":
        return [
            Scenario(
                suite="overall",
                name=f"dnn_{dnn_count}",
                dnn_count=dnn_count,
                population_size=args.population_size,
                iteration_limit=args.iteration_limit,
            )
            for dnn_count in args.dnn_counts
        ]
    if args.suite == "dynamic":
        levels = {
            "weak": dict(
                alpha_r=0.40,
                beta_r=0.25,
                alpha_l=0.35,
                beta_l=0.20,
                lambda_h=0.50,
                lambda_g=0.50,
            ),
            "medium": {},
            "strong": dict(
                alpha_r=1.20,
                beta_r=0.80,
                alpha_l=1.10,
                beta_l=0.70,
                lambda_h=0.90,
                lambda_g=0.90,
            ),
        }
        return [
            Scenario(
                suite="dynamic",
                name=level,
                dnn_count=args.dnn_count,
                population_size=args.population_size,
                iteration_limit=args.iteration_limit,
                **levels[level],
            )
            for level in ("weak", "medium", "strong")
        ]
    if args.suite == "sensitivity":
        scenarios: List[Scenario] = []
        for population_size in args.population_sizes:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"population_{population_size}",
                    dnn_count=1,
                    population_size=population_size,
                    iteration_limit=args.iteration_limit,
                    archive_capacity=args.archive_capacity,
                    sensitivity_instance_count=args.sensitivity_instances,
                )
            )
        for iteration_limit in args.iteration_limits:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"iterations_{iteration_limit}",
                    dnn_count=1,
                    population_size=args.population_size,
                    iteration_limit=iteration_limit,
                    archive_capacity=args.archive_capacity,
                    sensitivity_instance_count=args.sensitivity_instances,
                )
            )
        for probability in args.mutation_probabilities:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"mutation_{probability:g}",
                    dnn_count=1,
                    population_size=args.population_size,
                    iteration_limit=args.iteration_limit,
                    mutation_probability=probability,
                    archive_capacity=args.archive_capacity,
                    sensitivity_instance_count=args.sensitivity_instances,
                )
            )
        for probability in args.crossover_probabilities:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"crossover_{probability:g}",
                    dnn_count=1,
                    population_size=args.population_size,
                    iteration_limit=args.iteration_limit,
                    crossover_probability=probability,
                    archive_capacity=args.archive_capacity,
                    sensitivity_instance_count=args.sensitivity_instances,
                )
            )
        for archive_capacity in args.archive_capacities:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"archive_{archive_capacity}",
                    dnn_count=1,
                    population_size=args.population_size,
                    iteration_limit=args.iteration_limit,
                    archive_capacity=archive_capacity,
                    sensitivity_instance_count=args.sensitivity_instances,
                )
            )
        return scenarios
    if args.suite == "ablation":
        variants = {
            "full": {},
            "wo_dag_init": {"use_dag_aware_initialization": False},
            "wo_block_crossover": {"use_dag_block_crossover": False},
            "wo_skew_mutation": {"use_skew_mutation": False},
            "wo_elite_local_search": {"use_elite_local_search": False},
            "nsga_only": {
                "use_dag_aware_initialization": False,
                "use_dag_block_crossover": False,
                "use_skew_mutation": False,
                "use_elite_local_search": False,
            },
        }
        return [
            Scenario(
                suite="ablation",
                name=name,
                dnn_count=args.dnn_count,
                population_size=args.population_size,
                iteration_limit=args.iteration_limit,
                **settings,
            )
            for name, settings in variants.items()
        ]
    if args.suite == "pareto":
        return [
            Scenario(
                suite="pareto",
                name=f"dnn_{args.dnn_count}",
                dnn_count=args.dnn_count,
                population_size=args.population_size,
                iteration_limit=args.iteration_limit,
            )
        ]
    if args.suite == "scale":
        scales = {
            "small": dict(edge_count=10, user_count=30, dnn_count=args.small_dnn_count),
            "medium": dict(edge_count=30, user_count=120, dnn_count=args.medium_dnn_count),
            "large": dict(edge_count=60, user_count=300, dnn_count=args.large_dnn_count),
        }
        return [
            Scenario(
                suite="scale",
                name=name,
                population_size=args.population_size,
                iteration_limit=args.iteration_limit,
                **settings,
            )
            for name, settings in scales.items()
        ]
    if args.suite == "preference":
        return [
            Scenario(
                suite="preference",
                name=preference,
                dnn_count=args.dnn_count,
                population_size=args.population_size,
                iteration_limit=args.iteration_limit,
                preference=preference,
            )
            for preference in (
                "delay_sensitive",
                "stability_sensitive",
                "energy_sensitive",
                "balanced",
            )
        ]
    raise ValueError(f"unsupported suite: {args.suite}")


def _result_row(algorithm: str, scenario: Scenario, repeat: int, seed: int, output: RunOutput) -> Dict[str, object]:
    row: Dict[str, object] = {
        "task_id": f"{_scenario_key(scenario)}:{algorithm}:repeat_{repeat}:seed_{seed}",
        "suite": scenario.suite,
        "scenario": scenario.name,
        "algorithm": algorithm,
        "repeat": repeat,
        "seed": seed,
        "scenario_parameters": json.dumps(asdict(scenario), sort_keys=True),
    }
    row.update(dict(output.metrics.to_report_rows()))
    if output.diagnostics:
        row.update(output.diagnostics)
    row["result_schema_version"] = "semantic_names_v5_three_objective"
    if algorithm == "rtbl":
        row["metric_version"] = "post_admission_v1"
    if scenario.suite == "pareto":
        if algorithm not in PARETO_ALGORITHMS:
            raise ValueError(f"algorithm {algorithm!r} does not export a Pareto front")
        row.update(_pareto_hypervolume_by_dnn(output.pareto_points, enabled=True))
    return row


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["task_id"] for row in csv.DictReader(stream)}


def _validate_existing_result_schema(raw_path: Path, suite: str) -> None:
    """Prevent silently appending incompatible 3D/4D or Pareto/non-Pareto rows."""
    if not raw_path.exists():
        return
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    pareto_fields = {field for field in fieldnames if field.startswith("pareto_")}
    if suite == "pareto":
        required = {"pareto_hv_version", *PARETO_METRIC_FIELDS}
        missing = sorted(required - fieldnames)
        if missing:
            raise RuntimeError(
                f"{raw_path} uses an incompatible Pareto result schema "
                f"(missing: {', '.join(missing)}). Use a fresh output directory "
                "or archive results from an older Pareto schema."
            )
        invalid_versions = {
            row.get("pareto_hv_version", "")
            for row in rows
            if row.get("pareto_hv_version", "") != "3d_oss_delay_energy_v1"
        }
        if invalid_versions:
            raise RuntimeError(
                f"{raw_path} contains incompatible Pareto rows. Use a fresh output "
                "directory or archive the legacy results."
            )
    elif pareto_fields:
        raise RuntimeError(
            f"{raw_path} contains Pareto-only fields in the {suite!r} suite. "
            "Use a fresh output directory so non-Pareto runtime results remain HV-free."
        )
    required_semantics = ACTIVE_OBJECTIVE_SEMANTICS_VERSION
    if "objective_semantics_version" not in fieldnames or any(
        row.get("objective_semantics_version", "") != required_semantics
        for row in rows
    ):
        raise RuntimeError(
            f"{raw_path} does not use {required_semantics!r} objective semantics. "
            "Use a fresh output directory or archive the legacy product-based results."
        )
    if "result_schema_version" not in fieldnames or any(
        row.get("result_schema_version", "") != "semantic_names_v5_three_objective"
        for row in rows
    ):
        raise RuntimeError(
            f"{raw_path} does not use the semantic_names_v5_three_objective result schema. "
            "Migrate the legacy CSV or use a fresh output directory."
        )
    if suite == "sensitivity" and (
        "sensitivity_protocol_version" not in fieldnames
        or any(
            row.get("sensitivity_protocol_version", "") != "isolated_single_dnn_v1"
            for row in rows
        )
    ):
        raise RuntimeError(
            f"{raw_path} contains legacy sequential sensitivity results. "
            "Use a fresh output directory for isolated_single_dnn_v1."
        )


def _validate_existing_pareto_points_schema(path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "total_energy",
        "operational_stability",
        "operational_stability_satisfaction",
        "delay_satisfaction",
        "energy_satisfaction",
        "energy_norm",
        "operational_stability_norm",
        "pareto_hv_version",
        "objective_semantics_version",
        "result_schema_version",
    }
    missing = sorted(required - fieldnames)
    invalid_version = any(
        row.get("pareto_hv_version", "") != "3d_oss_delay_energy_v1"
        or row.get("result_schema_version", "") != "semantic_names_v5_three_objective"
        for row in rows
    )
    if missing or invalid_version:
        detail = (
            f"missing: {', '.join(missing)}"
            if missing
            else "rows from an incompatible three-objective schema"
        )
        raise RuntimeError(
            f"{path} uses an incompatible Pareto-point schema ({detail}). "
            "Use a fresh output directory or archive the legacy Pareto points."
        )


def _append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        missing_fieldnames = [field for field in row if field not in existing_fieldnames]
        if missing_fieldnames:
            fieldnames = [*existing_fieldnames, *missing_fieldnames]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
        else:
            fieldnames = existing_fieldnames
    else:
        fieldnames = list(row)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_pareto_points(
    path: Path,
    task_id: str,
    algorithm: str,
    scenario: Scenario,
    repeat: int,
    seed: int,
    points: Sequence[Dict[str, float | int | str]],
) -> None:
    if not points:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "suite",
        "scenario",
        "algorithm",
        "repeat",
        "seed",
        "dnn_index",
        "individual_index",
        "operational_stability",
        "delay_utility",
        "total_energy",
        "operational_stability_satisfaction",
        "energy_satisfaction",
        "estimated_delay",
        "deadline",
        "delay_satisfaction",
        "operational_stability_norm",
        "delay_norm",
        "energy_norm",
        "pareto_hv_version",
        "objective_semantics_version",
        "result_schema_version",
    ]
    write_header = not path.exists()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        missing_fieldnames = [field for field in fieldnames if field not in existing_fieldnames]
        if missing_fieldnames:
            fieldnames = [*existing_fieldnames, *missing_fieldnames]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
    normalized_points: List[Dict[str, float | int | str]] = []
    for group_points in _group_pareto_points_by_dnn(points).values():
        normalized_points.extend(_normalize_pareto_points(group_points))
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for point in normalized_points:
            writer.writerow(
                {
                    "task_id": task_id,
                    "suite": scenario.suite,
                    "scenario": scenario.name,
                    "algorithm": algorithm,
                    "repeat": repeat,
                    "seed": seed,
                    "dnn_index": point["dnn_index"],
                    "individual_index": point["individual_index"],
                    "operational_stability": point["operational_stability"],
                    "delay_utility": point["delay_utility"],
                    "total_energy": point["total_energy"],
                    "operational_stability_satisfaction": point["operational_stability_satisfaction"],
                    "energy_satisfaction": point["energy_satisfaction"],
                    "estimated_delay": point.get("estimated_delay", ""),
                    "deadline": point.get("deadline", ""),
                    "delay_satisfaction": point.get("delay_satisfaction", ""),
                    "operational_stability_norm": point["operational_stability_norm"],
                    "delay_norm": point["delay_norm"],
                    "energy_norm": point["energy_norm"],
                    "pareto_hv_version": "3d_oss_delay_energy_v1",
                    "objective_semantics_version": ACTIVE_OBJECTIVE_SEMANTICS_VERSION,
                    "result_schema_version": "semantic_names_v5_three_objective",
                }
            )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_summary(raw_path: Path, summary_path: Path, algorithms: Sequence[str] | None = None) -> None:
    allowed_algorithms = set(algorithms or [])
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if not allowed_algorithms or row["algorithm"] in allowed_algorithms
        ]
    groups: Dict[tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["suite"], row["scenario"], row["algorithm"])
        groups.setdefault(key, []).append(row)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    is_pareto = bool(rows) and all(row["suite"] == "pareto" for row in rows)
    is_sensitivity = bool(rows) and all(row["suite"] == "sensitivity" for row in rows)
    metric_fields = (
        BASE_METRIC_FIELDS
        + (PARETO_METRIC_FIELDS if is_pareto else ())
        + (SENSITIVITY_METRIC_FIELDS if is_sensitivity else ())
    )
    fields = ["suite", "scenario", "algorithm", "runs"]
    for metric in metric_fields:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (suite, scenario, algorithm), group_rows in sorted(groups.items()):
            result: Dict[str, object] = {
                "suite": suite,
                "scenario": scenario,
                "algorithm": algorithm,
                "runs": len(group_rows),
            }
            for metric in metric_fields:
                values = [float(row[metric]) for row in group_rows]
                result[f"{metric}_mean"] = _mean(values)
                result[f"{metric}_std"] = _std(values)
            writer.writerow(result)


def run_experiments(
    scenarios: Iterable[Scenario],
    algorithms: Sequence[str],
    repeats: int,
    base_seed: int,
    output_dir: Path,
) -> None:
    scenarios = list(scenarios)
    suites = {scenario.suite for scenario in scenarios}
    if len(suites) > 1:
        raise ValueError("one output directory may contain only one experiment suite")
    suite = next(iter(suites), "")
    raw_path = output_dir / "raw_results.csv"
    pareto_path = output_dir / "pareto_points.csv"
    _validate_existing_result_schema(raw_path, suite)
    if suite == "pareto":
        _validate_existing_pareto_points_schema(pareto_path)
    completed = _read_completed(raw_path)
    for scenario in scenarios:
        for repeat in range(repeats):
            seed = base_seed + repeat
            for algorithm in algorithms:
                task_id = f"{_scenario_key(scenario)}:{algorithm}:repeat_{repeat}:seed_{seed}"
                if task_id in completed:
                    print(f"[skip] {task_id}")
                    continue
                print(f"[run] {task_id}")
                output = run_once(algorithm, scenario, seed)
                _append_row(raw_path, _result_row(algorithm, scenario, repeat, seed, output))
                if scenario.suite == "pareto" and algorithm in PARETO_ALGORITHMS:
                    _append_pareto_points(pareto_path, task_id, algorithm, scenario, repeat, seed, output.pareto_points)
                completed.add(task_id)
    write_summary(raw_path, output_dir / "summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible batch experiments for DNN deployment algorithms")
    parser.add_argument(
        "--suite",
        choices=("overall", "dynamic", "sensitivity", "ablation", "pareto", "scale", "preference"),
        required=True,
    )
    parser.add_argument(
        "--algorithms",
        help="comma-separated algorithms; defaults depend on the selected suite",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/compatibility/legacy_algorithm_quality"),
    )
    parser.add_argument("--dnn-count", type=int, default=30)
    parser.add_argument("--dnn-counts", default="5,10,20,30,40")
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--iteration-limit", type=int, default=200)
    parser.add_argument("--population-sizes", default="20,40,60,80,100")
    parser.add_argument("--iteration-limits", default="80,120,160,200,240,280")
    parser.add_argument("--mutation-probabilities", default="0.10,0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("--crossover-probabilities", default="0.25,0.50,0.75")
    parser.add_argument("--archive-capacity", type=int, default=64)
    parser.add_argument("--archive-capacities", default="16,32,64,128")
    parser.add_argument("--sensitivity-instances", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small-dnn-count", type=int, default=20)
    parser.add_argument("--medium-dnn-count", type=int, default=60)
    parser.add_argument("--large-dnn-count", type=int, default=120)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(
        "[deprecated] experiment_runner.py is a compatibility backend; "
        "use experiments.py and the search_quality suite for formal results"
    )
    default_algorithms = {
        "overall": list(DEPLOYMENT_ALGORITHMS),
        "dynamic": list(DEPLOYMENT_ALGORITHMS),
        "sensitivity": ["customized"],
        "ablation": ["customized"],
        "pareto": list(PARETO_ALGORITHMS),
        "scale": list(DEPLOYMENT_ALGORITHMS),
        "preference": ["customized"],
    }
    args.algorithms = (
        _parse_csv_values(args.algorithms, str)
        if args.algorithms
        else default_algorithms[args.suite]
    )
    invalid_algorithms = sorted(set(args.algorithms) - set(ALGORITHMS))
    if invalid_algorithms:
        parser.error(f"unsupported algorithms: {', '.join(invalid_algorithms)}")
    if args.suite == "sensitivity" and args.algorithms != ["customized"]:
        parser.error("isolated sensitivity supports only the customized algorithm")
    args.dnn_counts = _parse_csv_values(args.dnn_counts, int)
    args.population_sizes = _parse_csv_values(args.population_sizes, int)
    args.iteration_limits = _parse_csv_values(args.iteration_limits, int)
    args.mutation_probabilities = _parse_csv_values(args.mutation_probabilities, float)
    args.crossover_probabilities = _parse_csv_values(args.crossover_probabilities, float)
    args.archive_capacities = _parse_csv_values(args.archive_capacities, int)
    if args.quick:
        args.repeats = 1
        args.sensitivity_instances = min(args.sensitivity_instances, 2)
        args.population_size = 4
        args.iteration_limit = 2
        args.population_sizes = [4]
        args.iteration_limits = [2]
        args.mutation_probabilities = [0.25]
        args.crossover_probabilities = [0.5]
        args.archive_capacity = min(args.archive_capacity, 8)
        args.archive_capacities = [args.archive_capacity]
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if (
        args.sensitivity_instances < 1
        or args.population_size < 2
        or args.population_size % 2
        or any(value < 2 or value % 2 for value in args.population_sizes)
        or args.iteration_limit < 1
        or any(value < 1 for value in args.iteration_limits)
        or args.archive_capacity < 1
        or any(value < 1 for value in args.archive_capacities)
    ):
        parser.error("invalid sensitivity instance or search budget")
    run_experiments(
        build_scenarios(args),
        args.algorithms,
        args.repeats,
        args.base_seed,
        args.output_dir / args.suite,
    )


if __name__ == "__main__":
    main()
