from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.environment import Environment
from experiment_runner import Scenario
from models import PostAdmissionMetrics
from rtbl.scheduler import RTBLRunner
from rtbl_experiment_runner import _merge_rtbl_results


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RTBLCorrectionTests(unittest.TestCase):
    def test_dynamic_runner_uses_post_admission_metrics(self) -> None:
        random.seed(20260719)
        scenario = Scenario(suite="overall", name="dnn_1", dnn_count=1)
        environment = Environment(**scenario.environment_kwargs())
        runner = RTBLRunner(environment)
        assignment = [0 for _ in range(environment.max_dnn_num)]

        with (
            patch.object(runner, "_build_assignment", return_value=assignment),
            patch.object(
                environment,
                "evaluate_post_admission_metrics",
                return_value=PostAdmissionMetrics(
                    delay=123.0,
                    operational_stability=0.42,
                    inference_fidelity=0.84,
                    total_energy=456.0,
                    raw_joint_product=0.35,
                ),
            ) as evaluator,
            patch.object(environment, "add_running_dnn"),
        ):
            metrics = runner.run_dynamic(environment.clone_nodes())

        evaluator.assert_called_once()
        self.assertEqual(metrics.avg_delay, 123.0)
        self.assertEqual(metrics.avg_operational_stability_score, 0.42)
        self.assertEqual(metrics.avg_inference_fidelity_score, 0.84)
        self.assertEqual(metrics.avg_energy, 456.0)
        self.assertEqual(metrics.failure_count, 0)

    def test_merge_replaces_only_rtbl_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "overall" / "raw_results.csv"
            staged = root / "staged" / "raw_results.csv"
            _write_rows(
                target,
                [
                    {"task_id": "customized-old", "algorithm": "customized", "value": 1},
                    {"task_id": "rtbl-old", "algorithm": "rtbl", "value": 2},
                    {"task_id": "random-old", "algorithm": "random", "value": 3},
                ],
            )
            _write_rows(
                staged,
                [
                    {
                        "task_id": "rtbl-new",
                        "algorithm": "rtbl",
                        "value": 4,
                        "metric_version": "post_admission_v1",
                    },
                ],
            )

            replaced, inserted = _merge_rtbl_results(target, staged)

            self.assertEqual((replaced, inserted), (1, 1))
            with target.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["task_id"] for row in rows],
                ["customized-old", "random-old", "rtbl-new"],
            )
            self.assertEqual(rows[0]["metric_version"], "")
            self.assertEqual(rows[-1]["metric_version"], "post_admission_v1")

    def test_merge_rejects_non_rtbl_staging_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "raw_results.csv"
            staged = root / "staged.csv"
            _write_rows(
                target,
                [{"task_id": "rtbl-old", "algorithm": "rtbl"}],
            )
            _write_rows(
                staged,
                [{"task_id": "wrong", "algorithm": "customized"}],
            )

            with self.assertRaisesRegex(RuntimeError, "unexpected algorithms"):
                _merge_rtbl_results(target, staged)


if __name__ == "__main__":
    unittest.main()
