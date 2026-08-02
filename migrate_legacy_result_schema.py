from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


FIELD_RENAMES = {
    "accuracy": "inference_fidelity",
    "operation": "operational_stability",
    "accuracy_satisfaction": "inference_fidelity_satisfaction",
    "operation_satisfaction": "operational_stability_satisfaction",
    "accuracy_norm": "inference_fidelity_norm",
    "operation_norm": "operational_stability_norm",
    "avg_operation": "avg_operational_stability_score",
    "avg_accuracy": "avg_inference_fidelity_score",
    "avg_dynamic_operation_reliability": "avg_operational_stability_score",
    "avg_dynamic_accuracy_reliability": "avg_inference_fidelity_score",
}
RESULT_SCHEMA_VERSION = "semantic_names_v4_global"
PARETO_SCHEMA_VERSION = "4d_unified_v4_global_semantics"
MIGRATABLE_PARETO_VERSIONS = {
    "4d_unified_v2",
    "4d_unified_v3_semantic_names",
    PARETO_SCHEMA_VERSION,
}


def _merge_value(row: dict[str, str], old: str, new: str) -> None:
    old_value = row.get(old, "")
    new_value = row.get(new, "")
    if old_value and new_value and old_value != new_value:
        raise ValueError(f"conflicting values for {old!r} and {new!r}")
    if not new_value and old_value:
        row[new] = old_value
    row.pop(old, None)


def migrate_rows(
    rows: Iterable[dict[str, str]],
    *,
    semantics_version: str | None = None,
) -> list[dict[str, str]]:
    migrated: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        for old, new in FIELD_RENAMES.items():
            _merge_value(row, old, new)

        for field in ("scenario", "preference"):
            if row.get(field) == "reliability_sensitive":
                row[field] = "stability_sensitive"
        if row.get("scenario_parameters"):
            parameters = json.loads(row["scenario_parameters"])
            if parameters.get("preference") == "reliability_sensitive":
                parameters["preference"] = "stability_sensitive"
            row["scenario_parameters"] = json.dumps(parameters, sort_keys=True)

        stored_semantics = row.get("objective_semantics_version", "")
        if stored_semantics and semantics_version and stored_semantics != semantics_version:
            raise ValueError("the requested objective semantics conflicts with the CSV value")
        if not stored_semantics:
            if not semantics_version:
                raise ValueError(
                    "objective_semantics_version is missing; pass --semantics-version "
                    "only after confirming the legacy experiment semantics"
                )
            row["objective_semantics_version"] = semantics_version

        pareto_version = row.get("pareto_hv_version", "")
        has_pareto_objectives = any(
            row.get(field, "")
            for field in (
                "inference_fidelity_norm",
                "operational_stability_norm",
                "delay_norm",
                "energy_norm",
            )
        )
        if pareto_version or has_pareto_objectives:
            if pareto_version not in MIGRATABLE_PARETO_VERSIONS:
                raise ValueError(
                    "only versioned four-dimensional Pareto rows can be renamed safely; "
                    "legacy 3D or unversioned fronts must be regenerated"
                )
            row["pareto_hv_version"] = PARETO_SCHEMA_VERSION

        row["result_schema_version"] = RESULT_SCHEMA_VERSION
        migrated.append(row)
    return migrated


def migrate_csv(
    source: Path,
    destination: Path,
    *,
    semantics_version: str | None = None,
    overwrite: bool = False,
) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = migrate_rows(reader, semantics_version=semantics_version)
        source_fields = list(reader.fieldnames or [])

    fieldnames = [field for field in source_fields if field not in FIELD_RENAMES]
    for field in (
        *FIELD_RENAMES.values(),
        "objective_semantics_version",
        "pareto_hv_version",
        "result_schema_version",
    ):
        if any(field in row for row in rows) and field not in fieldnames:
            fieldnames.append(field)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy experiment CSV data to semantic_names_v4_global."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--semantics-version")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    migrate_csv(
        args.source,
        args.destination,
        semantics_version=args.semantics_version,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
