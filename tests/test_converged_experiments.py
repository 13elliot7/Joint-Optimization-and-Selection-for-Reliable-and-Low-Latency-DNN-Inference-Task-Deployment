from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cleanup_legacy_results import _safe_workspace_path, verify
from experiment_framework.artifacts import suite_lock
from experiment_framework.identity import canonical_json, derive_seed, stable_hash, task_id
from experiment_framework.plans import FORMAL_SUITES, load_plan
from search_quality_runner import build_scenarios, run_once, run_scenarios


class ConvergedExperimentTests(unittest.TestCase):
    def test_identity_is_stable_across_mapping_order(self) -> None:
        first = {"b": [2, 3], "a": 1}
        second = {"a": 1, "b": (2, 3)}
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(stable_hash(first), stable_hash(second))
        self.assertEqual(
            task_id(
                experiment_layer="search_quality",
                protocol_version="v1",
                suite="search_quality",
                scenario=first,
                algorithm="customized",
                repeat=0,
                seed=7,
            ),
            task_id(
                experiment_layer="search_quality",
                protocol_version="v1",
                suite="search_quality",
                scenario=second,
                algorithm="customized",
                repeat=0,
                seed=7,
            ),
        )

    def test_external_random_streams_are_independent(self) -> None:
        arrival = derive_seed(7, "arrival")
        availability = derive_seed(7, "availability")
        algorithm = derive_seed(7, "algorithm", "customized")
        self.assertEqual(len({arrival, availability, algorithm}), 3)

    def test_formal_plan_contains_exactly_five_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plan = load_plan(root / "experiment_plans" / "formal_experiment_v1.json")
        self.assertEqual(tuple(plan["suites"]), FORMAL_SUITES)

    def test_cleanup_rejects_workspace_escape(self) -> None:
        with self.assertRaises(ValueError):
            _safe_workspace_path("../outside")

    def test_cleanup_archive_is_verified(self) -> None:
        self.assertTrue(verify()["verified"])

    def test_suite_lock_rejects_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with suite_lock(output):
                with self.assertRaises(RuntimeError):
                    with suite_lock(output):
                        pass
            self.assertFalse((output / ".run.lock").exists())

    def test_search_quality_quick_matrix_is_complete(self) -> None:
        scenarios = build_scenarios(quick=True, instance_count=1)
        self.assertEqual(
            {scenario.method for scenario in scenarios},
            {"customized", "proposed", "sa", "no_dag_operators"},
        )
        for scenario in scenarios:
            metrics = run_once(scenario, 20260719)
            self.assertEqual(metrics["instance_count"], 1)

    def test_search_quality_writes_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scenarios = build_scenarios(quick=True, instance_count=1)[:1]
            run_scenarios(scenarios, repeats=1, base_seed=7, output_dir=output)
            self.assertTrue((output / "raw_results.csv").exists())
            self.assertTrue((output / "summary.csv").exists())
            self.assertTrue((output / "run_manifest.json").exists())
            self.assertTrue((output / "validation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
