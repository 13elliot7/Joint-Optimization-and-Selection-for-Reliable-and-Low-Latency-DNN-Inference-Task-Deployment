from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from core.environment import Environment
from metrics import ExperimentMetrics
from proposed import AllDNNRefactor
from rtbl.config import SimulationConfig
from rtbl.scheduler import RTBLRunner


ALGORITHMS = ("random", "maxresource", "localfirst", "proposed", "customized")
METRIC_FIELDS = (
    "avg_estimated_delay",
    "avg_dynamic_operation_reliability",
    "avg_dynamic_accuracy_reliability",
    "avg_total_energy",
    "rejected_or_failed_count",
    "runtime_ms",
)


@dataclass(frozen=True)
class Scenario:
    suite: str
    name: str
    dnn_count: int = 20
    population_size: int = 60
    iteration_limit: int = 120
    mutation_probability: float = 0.10
    crossover_probability: float = 0.50
    alpha_r: float = 0.80
    beta_r: float = 0.50
    alpha_a: float = 0.50
    beta_a: float = 0.30
    alpha_l: float = 0.70
    beta_l: float = 0.40
    lambda_h: float = 0.70
    lambda_g: float = 0.70

    def environment_kwargs(self) -> Dict[str, float | int | bool]:
        return {
            "t_max": self.dnn_count,
            "alpha_r": self.alpha_r,
            "beta_r": self.beta_r,
            "alpha_a": self.alpha_a,
            "beta_a": self.beta_a,
            "alpha_l": self.alpha_l,
            "beta_l": self.beta_l,
            "lambda_h": self.lambda_h,
            "lambda_g": self.lambda_g,
            "verbose": False,
        }


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
    refactor.function1_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.function2_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.function3_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.function4_values = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.constraint_violations = [0.0 for _ in range(2 * refactor.pop_size)]
    refactor.distance = [[0 for _ in range(2 * refactor.pop_size)] for _ in range(1000)]


def _apply_dynamic_parameters(env: Environment, scenario: Scenario) -> None:
    for field in (
        "alpha_r",
        "beta_r",
        "alpha_a",
        "beta_a",
        "alpha_l",
        "beta_l",
        "lambda_h",
        "lambda_g",
    ):
        setattr(env, field, getattr(scenario, field))


def run_once(algorithm: str, scenario: Scenario, seed: int) -> ExperimentMetrics:
    """Run one reproducible algorithm/scenario/seed combination."""
    random.seed(seed)
    if algorithm == "rtbl":
        runner = RTBLRunner(SimulationConfig(tMax=scenario.dnn_count))
        runner.env.verbose = False
        _apply_dynamic_parameters(runner.env, scenario)
        return runner.run()

    env = Environment(**scenario.environment_kwargs())
    initial_nodes = env.clone_nodes()
    refactor = AllDNNRefactor(env)
    _configure_search(refactor, scenario)
    runners = {
        "random": lambda: refactor.run_random(initial_nodes),
        "maxresource": lambda: refactor.run_max_resource(initial_nodes),
        "localfirst": lambda: refactor.run_local_first(initial_nodes),
        "proposed": refactor.run_proposed,
        "customized": refactor.run_customized_proposed,
    }
    return runners[algorithm]()


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
                alpha_a=0.25,
                beta_a=0.15,
                alpha_l=0.35,
                beta_l=0.20,
                lambda_h=0.50,
                lambda_g=0.50,
            ),
            "medium": {},
            "strong": dict(
                alpha_r=1.20,
                beta_r=0.80,
                alpha_a=0.80,
                beta_a=0.50,
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
                    dnn_count=args.dnn_count,
                    population_size=population_size,
                    iteration_limit=args.iteration_limit,
                )
            )
        for iteration_limit in args.iteration_limits:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"iterations_{iteration_limit}",
                    dnn_count=args.dnn_count,
                    population_size=args.population_size,
                    iteration_limit=iteration_limit,
                )
            )
        for probability in args.mutation_probabilities:
            scenarios.append(
                Scenario(
                    suite="sensitivity",
                    name=f"mutation_{probability:g}",
                    dnn_count=args.dnn_count,
                    population_size=args.population_size,
                    iteration_limit=args.iteration_limit,
                    mutation_probability=probability,
                )
            )
        return scenarios
    raise ValueError(f"unsupported suite: {args.suite}")


def _result_row(algorithm: str, scenario: Scenario, repeat: int, seed: int, metrics: ExperimentMetrics) -> Dict[str, object]:
    row: Dict[str, object] = {
        "task_id": f"{_scenario_key(scenario)}:{algorithm}:repeat_{repeat}:seed_{seed}",
        "suite": scenario.suite,
        "scenario": scenario.name,
        "algorithm": algorithm,
        "repeat": repeat,
        "seed": seed,
        "scenario_parameters": json.dumps(asdict(scenario), sort_keys=True),
    }
    row.update(dict(metrics.to_report_rows()))
    return row


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["task_id"] for row in csv.DictReader(stream)}


def _append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_summary(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: Dict[tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["suite"], row["scenario"], row["algorithm"])
        groups.setdefault(key, []).append(row)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["suite", "scenario", "algorithm", "runs"]
    for metric in METRIC_FIELDS:
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
            for metric in METRIC_FIELDS:
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
    raw_path = output_dir / "raw_results.csv"
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
                metrics = run_once(algorithm, scenario, seed)
                _append_row(raw_path, _result_row(algorithm, scenario, repeat, seed, metrics))
                completed.add(task_id)
    write_summary(raw_path, output_dir / "summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible batch experiments for DNN deployment algorithms")
    parser.add_argument("--suite", choices=("overall", "dynamic", "sensitivity"), required=True)
    parser.add_argument(
        "--algorithms",
        help="comma-separated algorithms; defaults depend on the selected suite",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    parser.add_argument("--dnn-count", type=int, default=20)
    parser.add_argument("--dnn-counts", default="5,10,20,30,40")
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--iteration-limit", type=int, default=120)
    parser.add_argument("--population-sizes", default="20,40,60,80,100")
    parser.add_argument("--iteration-limits", default="30,60,120,180")
    parser.add_argument("--mutation-probabilities", default="0.05,0.10,0.20,0.30")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    default_algorithms = {
        "overall": list(ALGORITHMS),
        "dynamic": [algorithm for algorithm in ALGORITHMS if algorithm != "rtbl"],
        "sensitivity": ["proposed", "customized"],
    }
    args.algorithms = (
        _parse_csv_values(args.algorithms, str)
        if args.algorithms
        else default_algorithms[args.suite]
    )
    invalid_algorithms = sorted(set(args.algorithms) - set(ALGORITHMS))
    if invalid_algorithms:
        parser.error(f"unsupported algorithms: {', '.join(invalid_algorithms)}")
    args.dnn_counts = _parse_csv_values(args.dnn_counts, int)
    args.population_sizes = _parse_csv_values(args.population_sizes, int)
    args.iteration_limits = _parse_csv_values(args.iteration_limits, int)
    args.mutation_probabilities = _parse_csv_values(args.mutation_probabilities, float)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run_experiments(
        build_scenarios(args),
        args.algorithms,
        args.repeats,
        args.base_seed,
        args.output_dir / args.suite,
    )


if __name__ == "__main__":
    main()
