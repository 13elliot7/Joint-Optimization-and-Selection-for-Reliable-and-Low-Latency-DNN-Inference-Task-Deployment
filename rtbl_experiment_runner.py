from __future__ import annotations

import argparse
import csv
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List

from experiment_runner import (
    Scenario,
    _parse_csv_values,
    build_scenarios,
    run_experiments,
    write_summary,
)


# RTBL_SUITES = ("overall", "dynamic", "scale")
RTBL_SUITES = ("overall", "dynamic")


def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _merge_rtbl_results(target_path: Path, staged_path: Path) -> tuple[int, int]:
    """Atomically replace only RTBL rows while preserving every other algorithm."""
    existing_fields, existing_rows = _read_csv(target_path)
    staged_fields, staged_rows = _read_csv(staged_path)
    if not staged_rows:
        raise RuntimeError(f"no staged RTBL results were produced at {staged_path}")
    invalid_algorithms = sorted({
        row.get("algorithm", "")
        for row in staged_rows
        if row.get("algorithm") != "rtbl"
    })
    if invalid_algorithms:
        raise RuntimeError(
            "staged RTBL results contain unexpected algorithms: "
            + ", ".join(invalid_algorithms)
        )

    preserved_rows = [row for row in existing_rows if row.get("algorithm") != "rtbl"]
    replaced_count = len(existing_rows) - len(preserved_rows)
    fieldnames = list(existing_fields)
    for field in staged_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    if not fieldnames:
        fieldnames = list(staged_fields)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(target_path.suffix + ".rtbl-tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(preserved_rows)
            writer.writerows(staged_rows)
        temporary_path.replace(target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return replaced_count, len(staged_rows)


def _scenario_args(args: argparse.Namespace, suite: str) -> Namespace:
    return Namespace(
        suite=suite,
        dnn_count=args.dnn_count,
        dnn_counts=args.dnn_counts,
        population_size=args.population_size,
        iteration_limit=args.iteration_limit,
        population_sizes=[],
        iteration_limits=[],
        mutation_probabilities=[],
        small_dnn_count=args.small_dnn_count,
        medium_dnn_count=args.medium_dnn_count,
        large_dnn_count=args.large_dnn_count,
    )


def _run_suite(
    suite: str,
    scenarios: Iterable[Scenario],
    args: argparse.Namespace,
    staging_root: Path,
) -> None:
    staged_suite_dir = staging_root / suite
    run_experiments(
        scenarios,
        ["rtbl"],
        args.repeats,
        args.base_seed,
        staged_suite_dir,
    )
    target_suite_dir = args.output_dir / suite
    replaced, inserted = _merge_rtbl_results(
        target_suite_dir / "raw_results.csv",
        staged_suite_dir / "raw_results.csv",
    )
    write_summary(
        target_suite_dir / "raw_results.csv",
        target_suite_dir / "summary.csv",
    )
    print(
        f"[merged] suite={suite} replaced_rtbl_rows={replaced} "
        f"inserted_rtbl_rows={inserted}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only RTBL deployment experiments and atomically replace "
            "RTBL rows in existing result files."
        )
    )
    parser.add_argument(
        "--suites",
        help=(
            "comma-separated RTBL suites: overall,dynamic,scale; "
            "defaults to suites with an existing raw_results.csv"
        ),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    parser.add_argument("--dnn-count", type=int, default=30)
    parser.add_argument("--dnn-counts", default="5,10,20,30,40")
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--iteration-limit", type=int, default=200)
    parser.add_argument("--small-dnn-count", type=int, default=20)
    parser.add_argument("--medium-dnn-count", type=int, default=60)
    parser.add_argument("--large-dnn-count", type=int, default=120)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    suites = (
        _parse_csv_values(args.suites, str)
        if args.suites
        else [
            suite
            for suite in RTBL_SUITES
            if (args.output_dir / suite / "raw_results.csv").exists()
        ]
    )
    invalid_suites = sorted(set(suites) - set(RTBL_SUITES))
    if invalid_suites:
        parser.error(f"unsupported RTBL suites: {', '.join(invalid_suites)}")
    if not suites:
        parser.error(
            "no existing RTBL suite results were found; pass --suites explicitly "
            "to create a new RTBL-only result set"
        )
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.dnn_counts = _parse_csv_values(args.dnn_counts, int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rtbl-rerun-",
        dir=args.output_dir,
    ) as staging_directory:
        staging_root = Path(staging_directory)
        for suite in suites:
            suite_args = _scenario_args(args, suite)
            _run_suite(
                suite,
                build_scenarios(suite_args),
                args,
                staging_root,
            )


if __name__ == "__main__":
    main()
