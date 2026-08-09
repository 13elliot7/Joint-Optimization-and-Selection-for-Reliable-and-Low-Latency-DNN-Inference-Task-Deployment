from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


WORKSPACE = Path(__file__).resolve().parent
ARCHIVE_BASE = WORKSPACE / "archives" / "experiment_results_legacy"
_existing_archives = sorted(path for path in ARCHIVE_BASE.glob("????-??-??") if path.is_dir())
ARCHIVE_ROOT = _existing_archives[-1] if _existing_archives else ARCHIVE_BASE / str(date.today())
MANIFEST_PATH = ARCHIVE_ROOT / "archive_manifest.csv"
CHECKSUM_PATH = ARCHIVE_ROOT / "checksums.sha256"
REPORT_PATH = ARCHIVE_ROOT / "cleanup_report.json"

MANIFEST_FIELDS = (
    "source_path",
    "archive_file",
    "status",
    "experiment_layer",
    "protocol_version",
    "semantic_version",
    "suite",
    "row_count",
    "file_count",
    "byte_size",
    "sha256",
    "created_at",
    "eligible_for_formal_statistics",
    "reason",
)


@dataclass(frozen=True)
class ArchiveGroup:
    archive_file: str
    roots: tuple[str, ...]
    status: str
    experiment_layer: str
    protocol_version: str
    reason: str
    generated_figures: bool = False


GROUPS = (
    ArchiveGroup(
        archive_file="legacy_a_layer.tar.gz",
        roots=("experiment_results/archive",),
        status="historical",
        experiment_layer="deployment_quality",
        protocol_version="legacy_sequential_mixed",
        reason="superseded by isolated snapshot and Poisson lifecycle protocols",
    ),
    ArchiveGroup(
        archive_file="partial_a_sensitivity.tar.gz",
        roots=("experiment_results/sensitivity",),
        status="partial",
        experiment_layer="search_quality",
        protocol_version="isolated_single_dnn_v1",
        reason="incomplete parameter matrix; formal plan restarts from an empty result root",
    ),
    ArchiveGroup(
        archive_file="partial_b_algorithm_baseline_v3.tar.gz",
        roots=("periodic_experiment_results_v3/algorithm_baseline",),
        status="partial_diagnostic",
        experiment_layer="system_poisson",
        protocol_version="poisson_lifecycle_partial_v3",
        reason="seven-algorithm partial run; not eligible for completion by append",
    ),
    ArchiveGroup(
        archive_file="legacy_generated_figures.tar.gz",
        roots=("plot/figures",),
        status="generated_legacy",
        experiment_layer="derived_artifact",
        protocol_version="legacy_plot_inputs",
        reason="generated from superseded results and must be regenerated",
        generated_figures=True,
    ),
)

EMPTY_RESULT_ROOTS = (
    "periodic_experiment_results",
    "periodic_experiment_results_v2",
)
RESULT_ROOTS_TO_DELETE = (
    "experiment_results",
    "periodic_experiment_results",
    "periodic_experiment_results_v2",
    "periodic_experiment_results_v3",
)


def _safe_workspace_path(relative: str) -> Path:
    if not relative or relative.startswith(("/", "~")):
        raise ValueError(f"unsafe cleanup path: {relative!r}")
    candidate = (WORKSPACE / relative).resolve()
    if candidate == WORKSPACE or WORKSPACE not in candidate.parents:
        raise ValueError(f"cleanup path escapes workspace: {relative!r}")
    unresolved = WORKSPACE / relative
    if unresolved.is_symlink() or candidate.is_symlink():
        raise ValueError(f"cleanup target must not be a symbolic link: {relative!r}")
    return candidate


def _figure_files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_file()
            and (path.suffix.lower() in {".svg", ".pdf"} or path.name == "manifest.json")
        )
    )


def _group_files(group: ArchiveGroup) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative in group.roots:
        root = _safe_workspace_path(relative)
        if group.generated_figures:
            files.extend(_figure_files(root))
        elif root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    checked: list[Path] = []
    for path in sorted(set(files)):
        if path.is_symlink():
            raise ValueError(f"archive source must not be a symbolic link: {path}")
        if path.name == ".DS_Store" or "__pycache__" in path.parts:
            continue
        checked.append(path)
    return tuple(checked)


def _sha256_stream(stream: io.BufferedReader | tarfile.ExFileObject) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _csv_row_count(path: Path) -> int:
    if path.suffix.lower() != ".csv":
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        return max(sum(1 for _ in csv.reader(stream)) - 1, 0)


def _suite_from_path(path: Path) -> str:
    parts = path.relative_to(WORKSPACE).parts
    known = {
        "sensitivity",
        "pareto",
        "ablation",
        "overall",
        "dynamic",
        "scale",
        "algorithm_baseline",
    }
    return next((part for part in parts if part in known), "not_applicable")


def _semantic_version(path: Path) -> str:
    if path.suffix.lower() != ".csv":
        return "not_applicable"
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream)
            row = next(reader, None)
        if not row:
            return "unknown"
        return (
            row.get("objective_semantics_version")
            or row.get("semantic_version")
            or row.get("result_schema_version")
            or "unknown"
        )
    except (csv.Error, OSError):
        return "unknown"


def inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        files = _group_files(group)
        for path in files:
            relative = path.relative_to(WORKSPACE).as_posix()
            rows.append(
                {
                    "source_path": relative,
                    "archive_file": group.archive_file,
                    "status": group.status,
                    "experiment_layer": group.experiment_layer,
                    "protocol_version": group.protocol_version,
                    "semantic_version": _semantic_version(path),
                    "suite": _suite_from_path(path),
                    "row_count": _csv_row_count(path),
                    "file_count": 1,
                    "byte_size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "eligible_for_formal_statistics": "false",
                    "reason": group.reason,
                }
            )
    return rows


def dry_run() -> dict[str, object]:
    rows = inventory()
    report = {
        "workspace": str(WORKSPACE),
        "archive_root": str(ARCHIVE_ROOT),
        "files_to_archive": len(rows),
        "bytes_to_archive": sum(int(row["byte_size"]) for row in rows),
        "result_roots_to_delete": list(RESULT_ROOTS_TO_DELETE),
        "generated_figure_files_to_delete": [
            row["source_path"]
            for row in rows
            if row["archive_file"] == "legacy_generated_figures.tar.gz"
        ],
        "reference_assets_preserved": ["plot/figures/reference figures"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _write_readme() -> None:
    content = """# Legacy experiment result archive

This directory contains recoverable snapshots of superseded or partial experiment
outputs removed from the active workspace. Every archived result is ineligible for
formal statistics. Verify `checksums.sha256` and `archive_manifest.csv` before use.
"""
    (ARCHIVE_ROOT / "README.md").write_text(content, encoding="utf-8")


def archive() -> None:
    rows = inventory()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists() or any((ARCHIVE_ROOT / group.archive_file).exists() for group in GROUPS):
        raise FileExistsError(
            f"archive already exists at {ARCHIVE_ROOT}; verify it instead of overwriting"
        )
    rows_by_archive: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_archive.setdefault(str(row["archive_file"]), []).append(row)
    for group in GROUPS:
        archive_path = ARCHIVE_ROOT / group.archive_file
        with tarfile.open(archive_path, "w:gz") as bundle:
            for row in rows_by_archive.get(group.archive_file, []):
                source = _safe_workspace_path(str(row["source_path"]))
                bundle.add(source, arcname=str(row["source_path"]), recursive=False)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with CHECKSUM_PATH.open("w", encoding="utf-8") as stream:
        for group in GROUPS:
            archive_path = ARCHIVE_ROOT / group.archive_file
            stream.write(f"{_sha256_file(archive_path)}  {group.archive_file}\n")
    _write_readme()
    print(f"archived {len(rows)} files under {ARCHIVE_ROOT}")


def _manifest_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"archive manifest is missing: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict[str, object]:
    rows = _manifest_rows()
    expected_by_archive: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        expected_by_archive.setdefault(row["archive_file"], {})[row["source_path"]] = row
    failures: list[str] = []
    verified_files = 0
    for group in GROUPS:
        archive_path = ARCHIVE_ROOT / group.archive_file
        expected = expected_by_archive.get(group.archive_file, {})
        if not archive_path.exists():
            failures.append(f"missing archive: {group.archive_file}")
            continue
        with tarfile.open(archive_path, "r:gz") as bundle:
            members = {member.name: member for member in bundle.getmembers() if member.isfile()}
            if set(members) != set(expected):
                failures.append(f"member mismatch: {group.archive_file}")
                continue
            for name, member in members.items():
                extracted = bundle.extractfile(member)
                if extracted is None:
                    failures.append(f"cannot read member: {name}")
                    continue
                digest = _sha256_stream(extracted)
                if digest != expected[name]["sha256"]:
                    failures.append(f"checksum mismatch: {name}")
                elif member.size != int(expected[name]["byte_size"]):
                    failures.append(f"size mismatch: {name}")
                else:
                    verified_files += 1
    result = {
        "verified": not failures and verified_files == len(rows),
        "verified_files": verified_files,
        "manifest_files": len(rows),
        "failures": failures,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _remove_generated_figures(rows: Iterable[dict[str, str]]) -> list[str]:
    removed: list[str] = []
    for row in rows:
        if row["archive_file"] != "legacy_generated_figures.tar.gz":
            continue
        path = _safe_workspace_path(row["source_path"])
        if path.exists():
            path.unlink()
            removed.append(row["source_path"])
    return removed


def delete_verified() -> None:
    verification = verify()
    if not verification["verified"]:
        raise RuntimeError("archive verification failed; refusing to delete legacy results")
    rows = _manifest_rows()
    removed_files = _remove_generated_figures(rows)
    removed_roots: list[str] = []
    for relative in RESULT_ROOTS_TO_DELETE:
        path = _safe_workspace_path(relative)
        if path.exists():
            shutil.rmtree(path)
            removed_roots.append(relative)
    reference_root = _safe_workspace_path("plot/figures/reference figures")
    if not reference_root.exists():
        raise RuntimeError("reference figure assets were unexpectedly removed")
    report = {
        **verification,
        "removed_result_roots": removed_roots,
        "removed_generated_figure_files": removed_files,
        "reference_assets_preserved": [reference_root.relative_to(WORKSPACE).as_posix()],
        "deleted_at": datetime.now(timezone.utc).isoformat(),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely archive, verify, and remove fixed legacy experiment outputs"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--archive", action="store_true")
    actions.add_argument("--verify", action="store_true")
    actions.add_argument("--delete-verified", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run()
    elif args.archive:
        archive()
    elif args.verify:
        verification = verify()
        if not verification["verified"]:
            raise SystemExit(1)
    else:
        delete_verified()


if __name__ == "__main__":
    main()
