from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from cleanup_legacy_results import (
    delete_verified,
    dry_run as cleanup_dry_run,
    verify as verify_cleanup,
)
from experiment_framework.artifacts import validate_suite_output
from experiment_framework.plans import FORMAL_SUITES, load_plan


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RESULTS = WORKSPACE / "results"


def _resolve_plan(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.suffix:
        return candidate if candidate.is_absolute() else WORKSPACE / candidate
    return WORKSPACE / "experiment_plans" / f"{raw}.json"


def _run_search_quality(
    *,
    quick: bool,
    repeats: int,
    base_seed: int,
    output_root: Path,
    instances: int,
) -> None:
    from search_quality_runner import build_scenarios, run_scenarios

    run_scenarios(
        build_scenarios(quick=quick, instance_count=instances),
        repeats=1 if quick else repeats,
        base_seed=base_seed,
        output_dir=output_root / "search_quality",
    )


def _periodic_args(
    suite: str,
    *,
    quick: bool,
    repeats: int,
    base_seed: int,
    output_root: Path,
    slots: int,
    lambda_values: Iterable[float],
    reference_lambda: float,
    timing_mode: str,
    planning_timing_mode: str,
):
    from periodic_experiment_runner import _normalize_args, build_parser

    arguments = [
        "--suite",
        suite,
        "--repeats",
        str(repeats),
        "--base-seed",
        str(base_seed),
        "--output-dir",
        str(output_root / "system_poisson"),
        "--slots",
        str(slots),
        "--lambda-values",
        ",".join(str(value) for value in lambda_values),
        "--reference-lambda",
        str(reference_lambda),
        "--online-timing-mode",
        timing_mode,
        "--planning-timing-mode",
        planning_timing_mode,
    ]
    if quick:
        arguments.append("--quick")
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    _normalize_args(parser, parsed)
    return parsed


def _run_periodic_suite(
    suite: str,
    *,
    quick: bool,
    repeats: int,
    base_seed: int,
    output_root: Path,
    slots: int,
    lambda_values: Iterable[float],
    reference_lambda: float,
    timing_mode: str,
    planning_timing_mode: str,
) -> None:
    from periodic_experiment_runner import build_scenarios, run_scenarios

    args = _periodic_args(
        suite,
        quick=quick,
        repeats=repeats,
        base_seed=base_seed,
        output_root=output_root,
        slots=slots,
        lambda_values=lambda_values,
        reference_lambda=reference_lambda,
        timing_mode=timing_mode,
        planning_timing_mode=planning_timing_mode,
    )
    scenarios = build_scenarios(args, suite)
    print(f"[suite] {suite}: {len(scenarios)} scenarios × {args.repeats} repeats")
    run_scenarios(
        scenarios,
        args.repeats,
        args.base_seed,
        args.output_dir / suite,
    )


def _run_selected(
    suites: Iterable[str],
    *,
    quick: bool,
    repeats: int,
    base_seed: int,
    output_root: Path,
    instances: int,
    slots: int,
    lambda_values: Iterable[float],
    reference_lambda: float,
    timing_mode: str,
    planning_timing_mode: str,
) -> None:
    for suite in suites:
        if suite == "search_quality":
            _run_search_quality(
                quick=quick,
                repeats=repeats,
                base_seed=base_seed,
                output_root=output_root,
                instances=instances,
            )
        else:
            _run_periodic_suite(
                suite,
                quick=quick,
                repeats=repeats,
                base_seed=base_seed,
                output_root=output_root,
                slots=slots,
                lambda_values=lambda_values,
                reference_lambda=reference_lambda,
                timing_mode=timing_mode,
                planning_timing_mode=planning_timing_mode,
            )


def _suite_path(output_root: Path, suite: str) -> Path:
    return (
        output_root / "search_quality"
        if suite == "search_quality"
        else output_root / "system_poisson" / suite
    )


def validate_results(output_root: Path, suites: Iterable[str]) -> bool:
    valid = True
    for suite in suites:
        path = _suite_path(output_root, suite)
        if not path.exists():
            print(f"[missing] {suite}: {path}")
            valid = False
            continue
        report = validate_suite_output(path)
        print(f"[{'ok' if report['valid'] else 'invalid'}] {suite}: {report['row_count']} rows")
        valid = valid and bool(report["valid"])
    return valid


def audit_workspace(output_root: Path) -> bool:
    failures: list[str] = []
    for legacy in (
        "experiment_results",
        "periodic_experiment_results",
        "periodic_experiment_results_v2",
        "periodic_experiment_results_v3",
    ):
        if (WORKSPACE / legacy).exists():
            failures.append(f"legacy result root still exists: {legacy}")
    if output_root.resolve() == WORKSPACE.resolve():
        failures.append("output root must not be the workspace root")
    report = {"valid": not failures, "failures": failures, "output_root": str(output_root)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return not failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified five-module experiment entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list the five formal experiment modules")

    run = subparsers.add_parser("run", help="run a layer, suite, or declared plan")
    run.add_argument("--plan")
    run.add_argument("--layer", choices=("all", "search_quality", "system_poisson"), default="all")
    run.add_argument("--suite", choices=("all", *FORMAL_SUITES), default="all")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--repeats", type=int, default=20)
    run.add_argument("--base-seed", type=int, default=20260719)
    run.add_argument("--instances", type=int, default=30)
    run.add_argument("--slots", type=int, default=200)
    run.add_argument("--lambda-values", default="0.1,0.25,0.5,1,2")
    run.add_argument("--reference-lambda", type=float, default=1.0)
    run.add_argument("--timing-mode", choices=("measured", "deterministic"), default="measured")
    run.add_argument(
        "--planning-timing-mode",
        choices=("measured", "configured"),
        default="measured",
    )
    run.add_argument("--output-root", type=Path)

    for command in ("validate", "summarize", "audit", "plot"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
        child.add_argument("--layer", choices=("all", "search_quality", "system_poisson"), default="all")

    cleanup = subparsers.add_parser("cleanup-legacy")
    cleanup_actions = cleanup.add_mutually_exclusive_group(required=True)
    cleanup_actions.add_argument("--dry-run", action="store_true")
    cleanup_actions.add_argument("--verify", action="store_true")
    cleanup_actions.add_argument("--delete-verified", action="store_true")
    return parser


def _suites_for_layer(layer: str) -> tuple[str, ...]:
    if layer == "search_quality":
        return ("search_quality",)
    if layer == "system_poisson":
        return FORMAL_SUITES[1:]
    return FORMAL_SUITES


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        for index, suite in enumerate(FORMAL_SUITES, 1):
            layer = "search_quality" if suite == "search_quality" else "system_poisson"
            print(f"{index}. {suite} [{layer}]")
        return
    if args.command == "cleanup-legacy":
        if args.dry_run:
            cleanup_dry_run()
        elif args.verify:
            result = verify_cleanup()
            if not result["verified"]:
                raise SystemExit(1)
        else:
            delete_verified()
        return
    if args.command == "run":
        if args.plan:
            plan = load_plan(_resolve_plan(args.plan))
            suites = tuple(plan["suites"])
            repeats = int(plan["repeats"])
            base_seed = int(plan["base_seed"])
            output_root = args.output_root or (WORKSPACE / str(plan["output_root"]))
            search_config = dict(plan["search_quality"])
            system_config = dict(plan["system_poisson"])
            instances = int(search_config["instances"])
            slots = int(system_config["slots"])
            lambda_values = tuple(float(value) for value in system_config["lambda_values"])
            reference_lambda = float(system_config["reference_lambda"])
            timing_mode = str(system_config["online_timing_mode"])
            planning_timing_mode = str(
                system_config.get("planning_timing_mode", "measured")
            )
        else:
            selected = _suites_for_layer(args.layer)
            suites = selected if args.suite == "all" else (args.suite,)
            if any(suite not in selected for suite in suites):
                raise SystemExit("selected suite does not belong to the selected layer")
            repeats = args.repeats
            base_seed = args.base_seed
            output_root = args.output_root or DEFAULT_RESULTS
            instances = args.instances
            slots = args.slots
            lambda_values = tuple(float(value) for value in args.lambda_values.split(","))
            reference_lambda = args.reference_lambda
            timing_mode = args.timing_mode
            planning_timing_mode = args.planning_timing_mode
        _run_selected(
            suites,
            quick=args.quick,
            repeats=repeats,
            base_seed=base_seed,
            output_root=output_root,
            instances=instances,
            slots=slots,
            lambda_values=lambda_values,
            reference_lambda=reference_lambda,
            timing_mode=timing_mode,
            planning_timing_mode=planning_timing_mode,
        )
        return
    suites = _suites_for_layer(args.layer)
    if args.command in {"validate", "summarize"}:
        if not validate_results(args.output_root, suites):
            raise SystemExit(1)
        if args.command == "summarize":
            from experiment_framework.statistics import build_statistical_report

            report = build_statistical_report(
                args.output_root,
                ((suite, _suite_path(args.output_root, suite)) for suite in suites),
            )
            print(f"[report] {report}")
    elif args.command == "audit":
        if not audit_workspace(args.output_root) or not validate_results(args.output_root, suites):
            raise SystemExit(1)
    else:
        from plot_converged_results import generate_plots

        generate_plots(args.output_root, suites)


if __name__ == "__main__":
    main()
