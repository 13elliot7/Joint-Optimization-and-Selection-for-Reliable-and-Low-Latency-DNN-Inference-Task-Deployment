from __future__ import annotations

import csv
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .identity import stable_hash


STANDARD_FILES = ("raw_results.csv", "summary.csv", "run_manifest.json", "validation_report.json")


@contextmanager
def suite_lock(output_dir: Path):
    """Prevent concurrent appenders from corrupting a resumable suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".run.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(
            f"suite output is already locked: {output_dir} (owner {owner or 'unknown'})"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def git_revision(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_manifest(
    output_dir: Path,
    *,
    layer: str,
    suite: str,
    scenarios: Iterable[object],
    repeats: int,
    base_seed: int,
    timing_mode: str,
    semantic_versions: Mapping[str, str] | Iterable[tuple[str, str]],
    formal: bool = False,
) -> Path:
    scenario_list = list(scenarios)
    payload = {
        "manifest_version": "converged_experiment_manifest_v1",
        "experiment_layer": layer,
        "suite": suite,
        "scenario_count": len(scenario_list),
        "scenario_hashes": [stable_hash(item) for item in scenario_list],
        "scenario_definitions": [
            asdict(item) if is_dataclass(item) else dict(item) for item in scenario_list
        ],
        "repeats": repeats,
        "base_seed": base_seed,
        "timing_mode": timing_mode,
        "semantic_versions": dict(semantic_versions),
        "formal": formal,
        "git_revision": git_revision(Path(__file__).resolve().parents[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def validate_suite_output(output_dir: Path, expected_runs: int | None = None) -> dict[str, object]:
    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary.csv"
    failures: list[str] = []
    rows: list[dict[str, str]] = []
    if not raw_path.exists():
        failures.append("raw_results.csv is missing")
    else:
        with raw_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        task_ids = [row.get("task_id", "") for row in rows]
        if not all(task_ids):
            failures.append("one or more rows have no task_id")
        if len(set(task_ids)) != len(task_ids):
            failures.append("duplicate task_id values detected")
        required = {"experiment_layer", "protocol_version", "suite", "scenario", "algorithm", "seed"}
        missing = sorted(required - set(rows[0] if rows else ()))
        if missing:
            failures.append(f"missing identity fields: {', '.join(missing)}")
    if not summary_path.exists():
        failures.append("summary.csv is missing")
    if expected_runs is not None and len(rows) != expected_runs:
        failures.append(f"expected {expected_runs} rows, found {len(rows)}")
    report = {
        "validation_version": "converged_output_validation_v1",
        "valid": not failures,
        "formal_complete": not failures and expected_runs is not None,
        "row_count": len(rows),
        "expected_runs": expected_runs,
        "failures": failures,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
