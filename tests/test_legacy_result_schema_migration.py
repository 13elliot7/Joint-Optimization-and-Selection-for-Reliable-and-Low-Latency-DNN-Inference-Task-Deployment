from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from migrate_legacy_result_schema import migrate_csv, migrate_rows


class LegacyResultSchemaMigrationTests(unittest.TestCase):
    def test_semantic_columns_are_renamed_without_dual_output(self) -> None:
        rows = migrate_rows(
            [
                {
                    "accuracy": "0.91",
                    "operation": "0.82",
                    "accuracy_norm": "0.90",
                    "operation_norm": "0.80",
                    "pareto_hv_version": "4d_unified_v2",
                    "objective_semantics_version": "stability_fidelity_v2_return_energy",
                }
            ]
        )
        row = rows[0]
        self.assertEqual(row["inference_fidelity"], "0.91")
        self.assertEqual(row["operational_stability"], "0.82")
        self.assertEqual(row["pareto_hv_version"], "4d_unified_v4_global_semantics")
        self.assertEqual(row["result_schema_version"], "semantic_names_v4_global")
        self.assertNotIn("accuracy", row)
        self.assertNotIn("operation", row)

    def test_missing_semantics_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "--semantics-version"):
            migrate_rows([{"avg_accuracy": "0.9"}])

    def test_preference_enum_and_scenario_parameters_are_migrated(self) -> None:
        row = migrate_rows(
            [
                {
                    "scenario": "reliability_sensitive",
                    "scenario_parameters": '{"preference": "reliability_sensitive"}',
                }
            ],
            semantics_version="stability_fidelity_v2_return_energy",
        )[0]
        self.assertEqual(row["scenario"], "stability_sensitive")
        self.assertIn('"preference": "stability_sensitive"', row["scenario_parameters"])

    def test_unversioned_pareto_front_is_not_relabelled(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be regenerated"):
            migrate_rows(
                [{"accuracy_norm": "0.9", "operation_norm": "0.8"}],
                semantics_version="stability_fidelity_v2_return_energy",
            )

    def test_csv_migration_writes_only_new_metric_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.csv"
            destination = Path(directory) / "migrated.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "task_id",
                        "avg_operation",
                        "avg_accuracy",
                        "objective_semantics_version",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "sample",
                        "avg_operation": "0.8",
                        "avg_accuracy": "0.9",
                        "objective_semantics_version": "stability_fidelity_v2_return_energy",
                    }
                )
            migrate_csv(source, destination)
            with destination.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                row = next(reader)
                self.assertNotIn("avg_operation", reader.fieldnames or [])
                self.assertNotIn("avg_accuracy", reader.fieldnames or [])
            self.assertEqual(row["avg_operational_stability_score"], "0.8")
            self.assertEqual(row["avg_inference_fidelity_score"], "0.9")


if __name__ == "__main__":
    unittest.main()
