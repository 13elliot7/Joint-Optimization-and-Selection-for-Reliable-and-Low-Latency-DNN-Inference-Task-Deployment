from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

from core.arrival import PoissonArrivalConfig, PoissonArrivalProcess
from core.environment import Environment
from core.topology import TopologyConfig
from experiment_framework.artifacts import suite_lock, validate_suite_output, write_run_manifest
from experiment_framework.identity import stable_hash, task_id
from metrics import PeriodicExperimentMetrics
from models import DNNProfileCatalog, profile_from_dnn
from periodic_experiment import PeriodicExperimentConfig, PeriodicExperimentRunner
from planning.periodic_planner import PublicationCandidateAudit
from semantics import PERIODIC_SEMANTIC_VERSIONS


PERIODIC_SUITES = (
    "load",
    "algorithm_baseline",
    "robustness",
    "scale",
)

PERIODIC_PROTOCOL_VERSION = "poisson_lifecycle_v1"

NUMERIC_METRICS = (
    "arrival_slots",
    "elapsed_slots",
    "arrived_requests",
    "accepted_requests",
    "rejected_requests",
    "completed_requests",
    "runtime_failed_requests",
    "deadline_missed_requests",
    "unfinished_requests",
    "goodput_utility",
    "avg_estimated_delay_ms",
    "avg_operational_stability_score",
    "avg_total_energy",
    "plan_direct_count",
    "plan_repaired_count",
    "fast_fallback_count",
    "per_request_customized_count",
    "online_baseline_count",
    "planning_jobs_started",
    "planning_jobs_published",
    "planning_jobs_retained",
    "total_planning_runtime_ms",
    "total_online_runtime_ms",
    "avg_online_runtime_ms",
    "p95_online_runtime_ms",
    "publication_valid_candidates",
    "publication_rejected_candidates",
    "acceptance_rate",
    "completion_rate",
    "runtime_failure_rate",
    "plan_direct_rate",
    "plan_repaired_rate",
    "fast_fallback_rate",
)

PUBLICATION_AUDIT_FIELDS = (
    "audit_id",
    "task_id",
    "suite",
    "scenario",
    "repeat",
    "seed",
    "experiment_layer",
    "workload_mode",
    "algorithm",
    "search_budget_id",
    "job_id",
    "plan_id",
    "trigger_reason",
    "profile_id",
    "origin_group",
    "generator_mode",
    "generator_version",
    "search_seed",
    "source_generation",
    "publication_slot",
    "accepted",
    "primary_reason",
    "reasons",
    "offline_nodes",
    "constraint_violations",
    "constraint_violation",
    "fixed_point_converged",
    "fixed_point_iterations",
    "fixed_point_relative_residual",
    "estimated_delay_ms",
    "source_snapshot_version",
    "publication_snapshot_version",
    "source_environment_state_version",
    "publication_environment_state_version",
)


@dataclass(frozen=True)
class PeriodicScenario:
    suite: str
    name: str
    lambda_per_slot: float
    slot_count: int
    comparison_group: str = "reference"
    comparison_method: str = ""
    experiment_layer: str = "system_poisson"
    workload_mode: str = "poisson_profile_catalog"
    algorithm: str = "periodic_customized"
    slot_length_ms: float = 100.0
    control_mode: str = "fixed"
    bootstrap_mode: str = "prewarm"
    fixed_planning_interval_slots: int = 5
    fixed_total_pool_budget: int = 20
    fixed_online_budget_ms: float = 10.0
    planning_intervals: tuple[int, ...] = (5, 10, 20)
    total_pool_budgets: tuple[int, ...] = (8, 20, 40)
    online_budgets_ms: tuple[float, ...] = (5.0, 10.0, 20.0)
    planning_runtime_ms: float = 100.0
    planning_timing_mode: str = "measured"
    planning_sample_count: int = 20
    planning_candidate_generator_mode: str = "customized"
    planning_customized_population_size: int = 12
    planning_customized_generations: int = 4
    planning_customized_archive_capacity: int = 64
    candidate_fixed_point_max_iterations: int = 20
    candidate_fixed_point_relative_tolerance: float = 1e-4
    candidate_fixed_point_damping: float = 0.5
    plan_ttl_slots: int = 20
    minimum_replanning_interval_slots: int = 2
    minimum_control_window_slots: int = 5
    planning_deadline_ms: float = 2_000.0
    profile_count: int = 2
    origin_group_count: int = 2
    cloud_count: int = 1
    edge_count: int = 5
    user_count: int = 10
    edge_link_factor: float = 2.0
    availability_fraction: float = 0.5
    uptime_shape: float = 1.5
    uptime_scale: float = 100.0
    downtime_log_mean: float = 2.5
    downtime_log_sigma: float = 0.8
    alpha_r: float = 0.80
    beta_r: float = 0.50
    alpha_l: float = 0.70
    beta_l: float = 0.40
    lambda_h: float = 0.70
    lambda_g: float = 0.70
    pool_budget_mode: str = "fixed_total"
    min_deadline_ms: float = 500.0
    max_deadline_ms: float = 1999.0
    online_timing_mode: str = "measured"

    def __post_init__(self) -> None:
        if self.suite not in PERIODIC_SUITES:
            raise ValueError("unsupported periodic suite")
        if self.lambda_per_slot < 0.0 or self.slot_count <= 0 or self.slot_length_ms <= 0.0:
            raise ValueError("arrival rate and slot count are invalid")
        if self.profile_count <= 0 or self.origin_group_count <= 0:
            raise ValueError("profile and origin-group counts must be positive")
        if self.planning_candidate_generator_mode not in {
            "customized",
            "lightweight",
            "customized_with_fallback",
        }:
            raise ValueError("unsupported planning candidate generator")
        if self.algorithm not in {
            "periodic_customized",
            "periodic_lightweight",
            "online_fast_fallback",
            "per_request_customized",
            "random",
            "sa",
            "localfirst",
            "max_resource_fast",
            "rtbl",
        }:
            raise ValueError("unsupported scheduling algorithm")
        if (
            self.planning_customized_population_size < 2
            or self.planning_customized_population_size % 2 != 0
            or self.planning_customized_generations <= 0
            or self.planning_customized_archive_capacity <= 0
        ):
            raise ValueError("invalid customized search budget")
        if self.candidate_fixed_point_max_iterations <= 0:
            raise ValueError("candidate fixed-point iteration limit must be positive")
        if self.candidate_fixed_point_relative_tolerance <= 0.0:
            raise ValueError("candidate fixed-point tolerance must be positive")
        if not 0.0 < self.candidate_fixed_point_damping <= 1.0:
            raise ValueError("candidate fixed-point damping must be in (0, 1]")
        if not 0.0 <= self.availability_fraction <= 1.0:
            raise ValueError("availability_fraction must be in [0, 1]")
        if self.pool_budget_mode not in {"fixed_total", "fixed_per_key"}:
            raise ValueError("unsupported pool_budget_mode")


def _csv_values(raw: str, value_type):
    values = tuple(value_type(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must not be empty")
    return values


def _csv_search_budgets(raw: str) -> tuple[tuple[int, int, int], ...]:
    budgets: list[tuple[int, int, int]] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        try:
            population, generations, archive = (
                int(value) for value in item.lower().split("x")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "search budgets must use populationxgenerationsxarchive"
            ) from error
        if population < 2 or population % 2 or generations <= 0 or archive <= 0:
            raise ValueError("invalid customized search budget")
        budgets.append((population, generations, archive))
    if not budgets:
        raise ValueError("search budgets must not be empty")
    return tuple(budgets)


def _search_budget_id(scenario: PeriodicScenario) -> str:
    if scenario.algorithm in {
        "online_fast_fallback",
        "periodic_lightweight",
        "localfirst",
        "max_resource_fast",
        "rtbl",
    }:
        return "not_applicable"
    return (
        f"p{scenario.planning_customized_population_size}_"
        f"g{scenario.planning_customized_generations}_"
        f"a{scenario.planning_customized_archive_capacity}"
    )


def _candidate_generator_identity(scenario: PeriodicScenario) -> str:
    return (
        scenario.planning_candidate_generator_mode
        if scenario.algorithm.startswith("periodic_")
        else "not_applicable"
    )


def _scenario_id(scenario: PeriodicScenario, repeat: int, seed: int) -> str:
    return task_id(
        experiment_layer=scenario.experiment_layer,
        protocol_version=PERIODIC_PROTOCOL_VERSION,
        suite=scenario.suite,
        scenario=asdict(scenario),
        algorithm=scenario.algorithm,
        repeat=repeat,
        seed=seed,
    )


def _select_origins(environment: Environment, requested_count: int) -> tuple[int, ...]:
    selected: list[int] = []
    groups: set[int] = set()
    for node_index, node in enumerate(environment.nodes):
        if node.level != 1:
            continue
        group = environment.origin_group_for_node(node_index)
        if group in groups:
            continue
        selected.append(node_index)
        groups.add(group)
        if len(selected) >= requested_count:
            break
    if len(selected) != requested_count:
        raise ValueError(
            f"topology exposes {len(selected)} origin groups, expected {requested_count}"
        )
    return tuple(selected)


def _build_inputs(
    scenario: PeriodicScenario,
    seed: int,
) -> tuple[Environment, DNNProfileCatalog, object, PeriodicExperimentConfig]:
    random.seed(seed)
    topology = TopologyConfig(
        cloud_count=scenario.cloud_count,
        edge_count=scenario.edge_count,
        user_count=scenario.user_count,
        edge_link_factor=scenario.edge_link_factor,
    )
    node_count = scenario.cloud_count + scenario.edge_count + scenario.user_count
    stochastic_nodes = round(node_count * scenario.availability_fraction)
    environment = Environment(
        t_max=scenario.profile_count,
        alpha_r=scenario.alpha_r,
        beta_r=scenario.beta_r,
        alpha_l=scenario.alpha_l,
        beta_l=scenario.beta_l,
        lambda_h=scenario.lambda_h,
        lambda_g=scenario.lambda_g,
        availability_node_count=stochastic_nodes,
        shape=scenario.uptime_shape,
        scale=scenario.uptime_scale,
        mean=scenario.downtime_log_mean,
        sigma=scenario.downtime_log_sigma,
        availability_seed=seed + 1_000_003,
        slot_length=scenario.slot_length_ms,
        candidate_fixed_point_max_iterations=(
            scenario.candidate_fixed_point_max_iterations
        ),
        candidate_fixed_point_relative_tolerance=(
            scenario.candidate_fixed_point_relative_tolerance
        ),
        candidate_fixed_point_damping=scenario.candidate_fixed_point_damping,
        topology_config=topology,
        objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
        verbose=False,
    )
    profiles = tuple(
        profile_from_dnn(environment.ds[index], f"profile_{index}")
        for index in range(scenario.profile_count)
    )
    catalog = DNNProfileCatalog(
        profiles=profiles,
        sampling_weights=tuple(1.0 for _ in profiles),
    )
    origins = _select_origins(environment, scenario.origin_group_count)
    arrival_process = PoissonArrivalProcess(
        catalog,
        PoissonArrivalConfig(
            lambda_per_slot=scenario.lambda_per_slot,
            origin_nodes=origins,
            min_deadline_ms=scenario.min_deadline_ms,
            max_deadline_ms=scenario.max_deadline_ms,
        ),
        seed=seed + 2_000_003,
    )
    trace = arrival_process.generate_trace(scenario.slot_count)
    config = PeriodicExperimentConfig(
        origin_nodes=origins,
        bootstrap_mode=scenario.bootstrap_mode,
        control_mode=scenario.control_mode,
        planning_intervals=scenario.planning_intervals,
        total_pool_budgets=scenario.total_pool_budgets,
        online_budgets_ms=scenario.online_budgets_ms,
        fixed_planning_interval_slots=scenario.fixed_planning_interval_slots,
        fixed_total_pool_budget=scenario.fixed_total_pool_budget,
        fixed_online_budget_ms=scenario.fixed_online_budget_ms,
        planning_runtime_ms=scenario.planning_runtime_ms,
        planning_timing_mode=scenario.planning_timing_mode,
        planning_sample_count=scenario.planning_sample_count,
        planning_candidate_generator_mode=scenario.planning_candidate_generator_mode,
        planning_customized_population_size=scenario.planning_customized_population_size,
        planning_customized_generations=scenario.planning_customized_generations,
        planning_customized_archive_capacity=scenario.planning_customized_archive_capacity,
        planning_customized_random_seed=seed,
        algorithm=scenario.algorithm,
        plan_ttl_slots=scenario.plan_ttl_slots,
        minimum_replanning_interval_slots=scenario.minimum_replanning_interval_slots,
        minimum_control_window_slots=scenario.minimum_control_window_slots,
        planning_deadline_ms=scenario.planning_deadline_ms,
        online_timing_mode=scenario.online_timing_mode,
    )
    return environment, catalog, trace, config


def run_once(
    scenario: PeriodicScenario,
    seed: int,
    audit_sink: Callable[[tuple[PublicationCandidateAudit, ...]], None] | None = None,
    event_sink: Callable[[dict[str, tuple[dict[str, object], ...]]], None] | None = None,
) -> PeriodicExperimentMetrics:
    environment, catalog, trace, config = _build_inputs(scenario, seed)
    runner = PeriodicExperimentRunner(environment, catalog, trace, config)
    metrics = runner.run()
    if audit_sink is not None:
        audit_sink(tuple(runner.planner.publication_audits))
    if event_sink is not None:
        event_sink(
            {
                "request_events": tuple(asdict(record) for record in runner.dispatch_records),
                "planning_jobs": tuple(asdict(record) for record in runner.planning_records),
                "control_actions": tuple(
                    {"event_kind": "decision", **asdict(record)}
                    for record in runner.control_loop.metrics.decisions
                )
                + tuple(
                    {"event_kind": "outcome", **asdict(record)}
                    for record in runner.control_loop.metrics.outcomes
                ),
            }
        )
    return metrics


def _read_jsonl_event_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                ids.add(str(json.loads(line)["event_id"]))
    return ids


def _append_jsonl_events(
    path: Path,
    *,
    task_identifier: str,
    records: Iterable[dict[str, object]],
    existing_ids: set[str],
) -> None:
    pending: list[dict[str, object]] = []
    for index, record in enumerate(records):
        event_id = hashlib.sha256(
            f"{task_identifier}:{path.name}:{index}".encode("utf-8")
        ).hexdigest()[:24]
        if event_id in existing_ids:
            continue
        pending.append({"event_id": event_id, "task_id": task_identifier, **record})
    if not pending:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in pending:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            existing_ids.add(str(record["event_id"]))


def _audit_rows(
    scenario: PeriodicScenario,
    repeat: int,
    seed: int,
    audits: Iterable[PublicationCandidateAudit],
) -> list[dict[str, object]]:
    task_id = _scenario_id(scenario, repeat, seed)
    rows: list[dict[str, object]] = []
    for audit in audits:
        audit_id = hashlib.sha256(
            repr(
                (
                    task_id,
                    audit.job_id,
                    audit.plan_id,
                    audit.publication_snapshot_version,
                )
            ).encode("utf-8")
        ).hexdigest()[:24]
        row = {
            "audit_id": audit_id,
            "task_id": task_id,
            "suite": scenario.suite,
            "scenario": scenario.name,
            "repeat": repeat,
            "seed": seed,
            "experiment_layer": scenario.experiment_layer,
            "workload_mode": scenario.workload_mode,
            "algorithm": scenario.algorithm,
            "search_budget_id": _search_budget_id(scenario),
            **asdict(audit),
            "accepted": int(audit.accepted),
            "reasons": json.dumps(audit.reasons, separators=(",", ":")),
            "offline_nodes": json.dumps(audit.offline_nodes, separators=(",", ":")),
            "constraint_violations": json.dumps(
                audit.constraint_violations,
                separators=(",", ":"),
            ),
        }
        rows.append({field: row[field] for field in PUBLICATION_AUDIT_FIELDS})
    return rows


def _read_audit_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["audit_id"] for row in csv.DictReader(stream)}


def _append_audit_rows(
    path: Path,
    rows: Iterable[dict[str, object]],
    existing_ids: set[str],
) -> None:
    pending = [row for row in rows if str(row["audit_id"]) not in existing_ids]
    if not pending:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLICATION_AUDIT_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(pending)
    existing_ids.update(str(row["audit_id"]) for row in pending)


def _derived_metrics(
    metrics: PeriodicExperimentMetrics,
) -> dict[str, float]:
    arrived = max(metrics.arrived_requests, 1)
    accepted = max(metrics.accepted_requests, 1)
    return {
        "acceptance_rate": metrics.accepted_requests / arrived,
        "completion_rate": metrics.completed_requests / arrived,
        "runtime_failure_rate": metrics.runtime_failed_requests / accepted,
        "plan_direct_rate": metrics.plan_direct_count / arrived,
        "plan_repaired_rate": metrics.plan_repaired_count / arrived,
        "fast_fallback_rate": metrics.fast_fallback_count / arrived,
    }


def _result_row(
    scenario: PeriodicScenario,
    repeat: int,
    seed: int,
    metrics: PeriodicExperimentMetrics,
) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": _scenario_id(scenario, repeat, seed),
        "suite": scenario.suite,
        "scenario": scenario.name,
        "comparison_group": scenario.comparison_group,
        "comparison_method": scenario.comparison_method or scenario.algorithm,
        "repeat": repeat,
        "seed": seed,
        "experiment_layer": scenario.experiment_layer,
        "protocol_version": PERIODIC_PROTOCOL_VERSION,
        "algorithm": scenario.algorithm,
        "candidate_generator_mode": _candidate_generator_identity(scenario),
        "search_budget_id": _search_budget_id(scenario),
        "bootstrap_mode_config": scenario.bootstrap_mode,
        "online_timing_mode": scenario.online_timing_mode,
        "planning_timing_mode": scenario.planning_timing_mode,
        "pool_budget_mode": scenario.pool_budget_mode,
        "lambda_per_slot": scenario.lambda_per_slot,
        "topology_hash": stable_hash(
            {
                "cloud_count": scenario.cloud_count,
                "edge_count": scenario.edge_count,
                "user_count": scenario.user_count,
                "edge_link_factor": scenario.edge_link_factor,
                "seed": seed,
            }
        ),
        "profile_catalog_hash": stable_hash(
            {"profile_count": scenario.profile_count, "seed": seed}
        ),
        "workload_trace_hash": stable_hash(
            {
                "lambda_per_slot": scenario.lambda_per_slot,
                "slot_count": scenario.slot_count,
                "origin_group_count": scenario.origin_group_count,
                "seed": seed + 2_000_003,
            }
        ),
        "environment_trace_hash": stable_hash(
            {
                "availability_fraction": scenario.availability_fraction,
                "uptime_shape": scenario.uptime_shape,
                "uptime_scale": scenario.uptime_scale,
                "downtime_log_mean": scenario.downtime_log_mean,
                "downtime_log_sigma": scenario.downtime_log_sigma,
                "alpha_r": scenario.alpha_r,
                "beta_r": scenario.beta_r,
                "alpha_l": scenario.alpha_l,
                "beta_l": scenario.beta_l,
                "lambda_h": scenario.lambda_h,
                "lambda_g": scenario.lambda_g,
                "seed": seed + 1_000_003,
            }
        ),
    }
    row.update(dict(metrics.to_report_rows()))
    row.update(_derived_metrics(metrics))
    return row


def _read_existing(path: Path) -> tuple[set[str], list[str] | None]:
    if not path.exists():
        return set(), None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return {row["task_id"] for row in reader}, reader.fieldnames


def _append_row(path: Path, row: dict[str, object], expected_fields: list[str] | None) -> list[str]:
    fields = list(row)
    if expected_fields is not None and expected_fields != fields:
        raise ValueError(f"existing periodic result schema does not match: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    return fields


def _write_summary(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    identity_fields = (
        "experiment_layer",
        "protocol_version",
        "suite",
        "scenario",
        "comparison_group",
        "comparison_method",
        "algorithm",
        "candidate_generator_mode",
        "search_budget_id",
        "bootstrap_mode_config",
        "online_timing_mode",
        "planning_timing_mode",
        "pool_budget_mode",
        "objective_semantics_version",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in identity_fields)
        groups.setdefault(key, []).append(row)
    fields = [*identity_fields, "runs"]
    for metric in NUMERIC_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for identity, group in sorted(groups.items()):
            output: dict[str, object] = dict(zip(identity_fields, identity))
            output["runs"] = len(group)
            for metric in NUMERIC_METRICS:
                values = [float(row[metric]) for row in group]
                output[f"{metric}_mean"] = statistics.fmean(values)
                output[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            writer.writerow(output)


def _run_scenarios_unlocked(
    scenarios: Iterable[PeriodicScenario],
    repeats: int,
    base_seed: int,
    output_dir: Path,
) -> None:
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("periodic suite produced no feasible scenarios")
    suites = {scenario.suite for scenario in scenarios}
    if len(suites) != 1:
        raise ValueError("one periodic output directory may contain only one suite")
    raw_path = output_dir / "raw_results.csv"
    audit_path = output_dir / "planning_candidate_audits.csv"
    event_paths = {
        "request_events": output_dir / "request_events.jsonl",
        "planning_jobs": output_dir / "planning_jobs.jsonl",
        "control_actions": output_dir / "control_actions.jsonl",
    }
    completed, fields = _read_existing(raw_path)
    existing_audit_ids = _read_audit_ids(audit_path)
    existing_event_ids = {
        name: _read_jsonl_event_ids(path) for name, path in event_paths.items()
    }
    for scenario in scenarios:
        for repeat in range(repeats):
            seed = base_seed + repeat
            task_id = _scenario_id(scenario, repeat, seed)
            if task_id in completed:
                print(f"[skip] {task_id}")
                continue
            print(f"[run] {task_id}")
            audits: list[PublicationCandidateAudit] = []
            events: dict[str, tuple[dict[str, object], ...]] = {}
            metrics = run_once(scenario, seed, audits.extend, events.update)
            _append_audit_rows(
                audit_path,
                _audit_rows(scenario, repeat, seed, audits),
                existing_audit_ids,
            )
            for name, path in event_paths.items():
                _append_jsonl_events(
                    path,
                    task_identifier=task_id,
                    records=events.get(name, ()),
                    existing_ids=existing_event_ids[name],
                )
            fields = _append_row(
                raw_path,
                _result_row(scenario, repeat, seed, metrics),
                fields,
            )
            completed.add(task_id)
    _write_summary(raw_path, output_dir / "summary.csv")
    write_run_manifest(
        output_dir,
        layer="system_poisson",
        suite=next(iter(suites)),
        scenarios=scenarios,
        repeats=repeats,
        base_seed=base_seed,
        timing_mode=scenarios[0].online_timing_mode,
        semantic_versions=PERIODIC_SEMANTIC_VERSIONS.as_pairs(),
    )
    report = validate_suite_output(
        output_dir,
        expected_runs=len(scenarios) * repeats,
    )
    if not report["valid"]:
        raise RuntimeError(f"periodic result validation failed: {report['failures']}")


def run_scenarios(
    scenarios: Iterable[PeriodicScenario],
    repeats: int,
    base_seed: int,
    output_dir: Path,
) -> None:
    with suite_lock(output_dir):
        _run_scenarios_unlocked(scenarios, repeats, base_seed, output_dir)


def _base_scenario(args: argparse.Namespace, suite: str, name: str) -> PeriodicScenario:
    generator_mode = getattr(args, "candidate_generator_mode", "customized")
    return PeriodicScenario(
        suite=suite,
        name=name,
        lambda_per_slot=args.reference_lambda,
        slot_count=args.slots,
        slot_length_ms=args.slot_length_ms,
        fixed_planning_interval_slots=args.fixed_planning_interval,
        fixed_total_pool_budget=args.fixed_pool_budget,
        fixed_online_budget_ms=args.fixed_online_budget_ms,
        planning_intervals=args.planning_intervals,
        total_pool_budgets=args.pool_budgets,
        online_budgets_ms=args.online_budgets_ms,
        planning_runtime_ms=args.planning_runtime_ms,
        planning_timing_mode=args.planning_timing_mode,
        planning_sample_count=args.planning_sample_count,
        algorithm=(
            "periodic_lightweight"
            if generator_mode == "lightweight"
            else "periodic_customized"
        ),
        planning_candidate_generator_mode=generator_mode,
        planning_customized_population_size=getattr(
            args, "customized_population_size", 12
        ),
        planning_customized_generations=getattr(args, "customized_generations", 4),
        planning_customized_archive_capacity=getattr(
            args, "customized_archive_capacity", 64
        ),
        candidate_fixed_point_max_iterations=(
            getattr(args, "candidate_fixed_point_max_iterations", 20)
        ),
        candidate_fixed_point_relative_tolerance=(
            getattr(args, "candidate_fixed_point_relative_tolerance", 1e-4)
        ),
        candidate_fixed_point_damping=getattr(
            args,
            "candidate_fixed_point_damping",
            0.5,
        ),
        plan_ttl_slots=args.plan_ttl_slots,
        minimum_replanning_interval_slots=args.minimum_replanning_interval,
        minimum_control_window_slots=args.minimum_control_window,
        profile_count=args.profile_count,
        origin_group_count=args.origin_group_count,
        cloud_count=args.cloud_count,
        edge_count=args.edge_count,
        user_count=args.user_count,
        edge_link_factor=args.edge_link_factor,
        availability_fraction=args.availability_fraction,
        uptime_shape=args.uptime_shape,
        uptime_scale=args.uptime_scale,
        downtime_log_mean=args.downtime_log_mean,
        downtime_log_sigma=args.downtime_log_sigma,
        online_timing_mode=args.online_timing_mode,
    )


def build_scenarios(args: argparse.Namespace, suite: str) -> tuple[PeriodicScenario, ...]:
    base = _base_scenario(args, suite, "reference")
    minimum_pool = 2 * args.profile_count * args.origin_group_count
    maximum_pool = 40 * args.profile_count * args.origin_group_count

    def valid_pool(value: int) -> bool:
        return minimum_pool <= value <= maximum_pool

    if not valid_pool(args.fixed_pool_budget):
        raise ValueError(
            f"fixed pool budget must be in [{minimum_pool}, {maximum_pool}] "
            "for the configured profile/origin keys"
        )
    if suite == "algorithm_baseline":
        return (
            replace(
                base,
                name="periodic_customized",
                algorithm="periodic_customized",
                planning_candidate_generator_mode="customized",
            ),
            replace(
                base,
                name="periodic_lightweight",
                algorithm="periodic_lightweight",
                planning_candidate_generator_mode="lightweight",
            ),
            replace(
                base,
                name="per_request_customized",
                algorithm="per_request_customized",
                planning_candidate_generator_mode="customized",
                bootstrap_mode="cold_start",
            ),
            replace(
                base,
                name="periodic_customized_cold_start",
                algorithm="periodic_customized",
                comparison_method="periodic_customized_cold_start",
                planning_candidate_generator_mode="customized",
                bootstrap_mode="cold_start",
            ),
            *(
                replace(
                    base,
                    name=algorithm,
                    algorithm=algorithm,
                    planning_candidate_generator_mode="lightweight",
                    bootstrap_mode="cold_start",
                )
                for algorithm in (
                    "random",
                    "sa",
                    "localfirst",
                    "max_resource_fast",
                    "rtbl",
                )
            ),
        )
    if suite == "load":
        values = (
            tuple(dict.fromkeys((args.lambda_values[0], args.reference_lambda)))
            if args.quick
            else args.lambda_values
        )
        algorithms = (
            ("periodic_customized", "customized", "prewarm"),
            ("periodic_lightweight", "lightweight", "prewarm"),
            ("per_request_customized", "customized", "cold_start"),
        )
        return tuple(
            replace(
                base,
                name=f"lambda_{value:g}_{algorithm}",
                comparison_group=f"lambda_{value:g}",
                comparison_method=algorithm,
                lambda_per_slot=value,
                algorithm=algorithm,
                planning_candidate_generator_mode=generator,
                bootstrap_mode=bootstrap,
            )
            for value in values
            for algorithm, generator, bootstrap in algorithms
        )
    if suite == "robustness":
        environments = (
            (
                "stable",
                dict(
                    availability_fraction=0.25,
                    uptime_scale=200.0,
                    alpha_r=0.40,
                    beta_r=0.25,
                    alpha_l=0.35,
                    beta_l=0.20,
                    lambda_h=0.50,
                    lambda_g=0.50,
                ),
            ),
            (
                "failure_prone",
                dict(availability_fraction=1.0, uptime_scale=50.0),
            ),
            (
                "fast_drift",
                dict(
                    availability_fraction=0.5,
                    uptime_scale=100.0,
                    alpha_r=1.20,
                    beta_r=0.80,
                    alpha_l=1.10,
                    beta_l=0.70,
                    lambda_h=0.90,
                    lambda_g=0.90,
                ),
            ),
        )
        algorithms = (
            ("periodic_customized", "customized", "prewarm"),
            ("periodic_lightweight", "lightweight", "prewarm"),
            ("per_request_customized", "customized", "cold_start"),
        )
        if args.quick:
            algorithms = algorithms[:1]
        return tuple(
            replace(
                base,
                name=f"{environment_name}_{algorithm}",
                comparison_group=environment_name,
                comparison_method=algorithm,
                algorithm=algorithm,
                planning_candidate_generator_mode=generator,
                bootstrap_mode=bootstrap,
                **settings,
            )
            for environment_name, settings in environments
            for algorithm, generator, bootstrap in algorithms
        )
    if suite == "scale":
        presets = (
            ("small", 1, 2, 4, 1, 1, 0.5),
            ("medium", 1, 5, 10, 2, 2, 1.0),
            ("large", 1, 10, 20, 4, 4, 2.0),
        )
        if args.quick:
            presets = presets[:1]
        algorithms = (
            ("periodic_customized", "customized", "prewarm"),
            ("periodic_lightweight", "lightweight", "prewarm"),
            ("per_request_customized", "customized", "cold_start"),
        )
        if args.quick:
            algorithms = algorithms[:1]
        scenarios: list[PeriodicScenario] = []
        for name, clouds, edges, users, profiles, origins, arrival_rate in presets:
            per_key_pool = 5 * profiles * origins
            for algorithm, generator, bootstrap in algorithms:
                scenarios.append(
                    replace(
                        base,
                        name=f"{name}_{algorithm}_per_key",
                        comparison_group=f"{name}_per_key",
                        comparison_method=algorithm,
                        algorithm=algorithm,
                        planning_candidate_generator_mode=generator,
                        bootstrap_mode=bootstrap,
                        cloud_count=clouds,
                        edge_count=edges,
                        user_count=users,
                        profile_count=profiles,
                        origin_group_count=origins,
                        lambda_per_slot=arrival_rate,
                        fixed_total_pool_budget=per_key_pool,
                        pool_budget_mode="fixed_per_key",
                    )
                )
        return tuple(scenarios)
    raise ValueError(f"unsupported periodic suite: {suite}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible batch experiments for periodic DNN planning"
    )
    parser.add_argument("--suite", choices=("all", *PERIODIC_SUITES), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/system_poisson"),
    )
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--slot-length-ms", type=float, default=100.0)
    parser.add_argument(
        "--lambda-values",
        default="0.1,0.25,0.5,1,2",
    )
    parser.add_argument("--reference-lambda", type=float, default=1.0)
    parser.add_argument("--planning-intervals", default="5,10,20")
    parser.add_argument("--pool-budgets", default="8,20,40")
    parser.add_argument("--online-budgets-ms", default="5,10,20")
    parser.add_argument("--fixed-planning-interval", type=int, default=5)
    parser.add_argument("--fixed-pool-budget", type=int, default=20)
    parser.add_argument("--fixed-online-budget-ms", type=float, default=10.0)
    parser.add_argument("--planning-runtime-ms", type=float, default=100.0)
    parser.add_argument(
        "--planning-timing-mode",
        choices=("measured", "configured"),
        default="measured",
        help="map measured planning wall time or a configured runtime to ready slots",
    )
    parser.add_argument("--planning-runtime-values", default="0,100,500,2000")
    parser.add_argument("--planning-sample-count", type=int, default=20)
    parser.add_argument(
        "--candidate-generator-mode",
        choices=("customized", "lightweight", "customized_with_fallback"),
        default="customized",
    )
    parser.add_argument("--customized-population-size", type=int, default=12)
    parser.add_argument("--customized-generations", type=int, default=4)
    parser.add_argument("--customized-archive-capacity", type=int, default=64)
    parser.add_argument(
        "--search-budgets",
        default="12x4x64,24x10x64,40x25x128",
        help="comma-separated populationxgenerationsxarchive budgets",
    )
    parser.add_argument(
        "--candidate-fixed-point-max-iterations",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--candidate-fixed-point-relative-tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--candidate-fixed-point-damping",
        type=float,
        default=0.5,
    )
    parser.add_argument("--plan-ttl-slots", type=int, default=20)
    parser.add_argument("--minimum-replanning-interval", type=int, default=2)
    parser.add_argument("--minimum-control-window", type=int, default=5)
    parser.add_argument("--profile-count", type=int, default=2)
    parser.add_argument("--origin-group-count", type=int, default=2)
    parser.add_argument("--cloud-count", type=int, default=1)
    parser.add_argument("--edge-count", type=int, default=5)
    parser.add_argument("--user-count", type=int, default=10)
    parser.add_argument("--edge-link-factor", type=float, default=2.0)
    parser.add_argument("--availability-fraction", type=float, default=0.5)
    parser.add_argument("--availability-fractions", default="0.25,0.5,1.0")
    parser.add_argument("--uptime-shape", type=float, default=1.5)
    parser.add_argument("--uptime-scale", type=float, default=100.0)
    parser.add_argument("--uptime-scales", default="50,100,200")
    parser.add_argument("--downtime-log-mean", type=float, default=2.5)
    parser.add_argument("--downtime-log-sigma", type=float, default=0.8)
    parser.add_argument(
        "--online-timing-mode",
        choices=("measured", "deterministic"),
        default="measured",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a reduced one-repeat smoke matrix with deterministic timing",
    )
    return parser


def _normalize_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        args.lambda_values = _csv_values(args.lambda_values, float)
        args.planning_intervals = _csv_values(args.planning_intervals, int)
        args.pool_budgets = _csv_values(args.pool_budgets, int)
        args.online_budgets_ms = _csv_values(args.online_budgets_ms, float)
        args.planning_runtime_values = _csv_values(args.planning_runtime_values, float)
        args.availability_fractions = _csv_values(args.availability_fractions, float)
        args.uptime_scales = _csv_values(args.uptime_scales, float)
        args.search_budgets = _csv_search_budgets(args.search_budgets)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.repeats <= 0
        or args.slots <= 0
        or args.planning_sample_count <= 0
        or getattr(args, "customized_population_size", 12) < 2
        or getattr(args, "customized_population_size", 12) % 2 != 0
        or getattr(args, "customized_generations", 4) <= 0
        or getattr(args, "customized_archive_capacity", 64) <= 0
        or args.candidate_fixed_point_max_iterations <= 0
        or args.candidate_fixed_point_relative_tolerance <= 0.0
        or not 0.0 < args.candidate_fixed_point_damping <= 1.0
    ):
        parser.error("experiment counts and fixed-point parameters are invalid")
    if args.quick:
        args.repeats = 1
        args.slots = min(args.slots, 10)
        args.planning_sample_count = min(args.planning_sample_count, 3)
        args.customized_population_size = min(
            getattr(args, "customized_population_size", 12), 4
        )
        if args.customized_population_size % 2 != 0:
            args.customized_population_size -= 1
        args.customized_generations = min(
            getattr(args, "customized_generations", 4), 2
        )
        args.search_budgets = (
            (
                args.customized_population_size,
                args.customized_generations,
                args.customized_archive_capacity,
            ),
        )
        args.online_timing_mode = "deterministic"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(
        "[backend] periodic_experiment_runner.py executes system_poisson suites; "
        "use experiments.py for the unified formal plan"
    )
    _normalize_args(parser, args)
    suites: Sequence[str] = PERIODIC_SUITES if args.suite == "all" else (args.suite,)
    for suite in suites:
        try:
            scenarios = build_scenarios(args, suite)
        except ValueError as error:
            parser.error(str(error))
        print(f"[suite] {suite}: {len(scenarios)} scenarios × {args.repeats} repeats")
        run_scenarios(
            scenarios,
            args.repeats,
            args.base_seed,
            args.output_dir / suite,
        )


if __name__ == "__main__":
    main()
