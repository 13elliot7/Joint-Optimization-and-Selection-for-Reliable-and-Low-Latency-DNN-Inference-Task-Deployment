from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable


PRIMARY_METRICS = {
    "search_quality": ("mean_hypervolume", "feasible_rate", "search_runtime_ms"),
    "load": ("goodput_utility", "acceptance_rate", "runtime_failure_rate"),
    "algorithm_baseline": ("goodput_utility", "acceptance_rate", "avg_online_runtime_ms"),
    "robustness": ("goodput_utility", "runtime_failure_rate", "publication_rejected_candidates"),
    "scale": ("goodput_utility", "total_planning_runtime_ms", "p95_online_runtime_ms"),
}

LOWER_IS_BETTER = {
    "search_runtime_ms",
    "runtime_failure_rate",
    "avg_online_runtime_ms",
    "p95_online_runtime_ms",
    "control_fallback_count",
    "publication_rejected_candidates",
    "total_planning_runtime_ms",
}

REFERENCE_METHODS = {
    "search_quality": "customized_medium",
    "load": "periodic_customized",
    "algorithm_baseline": "periodic_customized",
    "robustness": "periodic_customized",
    "scale": "periodic_customized",
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    return 1.96 * _std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _sign_test_pvalue(differences: list[float]) -> float:
    nonzero = [value for value in differences if abs(value) > 1e-12]
    if not nonzero:
        return 1.0
    positive = sum(value > 0.0 for value in nonzero)
    tail = min(positive, len(nonzero) - positive)
    probability = sum(math.comb(len(nonzero), index) for index in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in ordered[start:end]:
            ranks[position] = average
        start = end
    return ranks


def _regularized_gamma_q(shape: float, value: float) -> float:
    if value < 0.0 or shape <= 0.0:
        raise ValueError("invalid incomplete gamma arguments")
    if value == 0.0:
        return 1.0
    epsilon = 1e-14
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        current = shape
        for _ in range(1000):
            current += 1.0
            term *= value / current
            total += term
            if abs(term) < abs(total) * epsilon:
                break
        p_value = total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
        return max(0.0, min(1.0, 1.0 - p_value))
    tiny = 1e-300
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for index in range(1, 1000):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return max(
        0.0,
        min(1.0, math.exp(-value + shape * math.log(value) - math.lgamma(shape)) * fraction),
    )


def _friedman(method_values: dict[str, dict[int, float]]) -> tuple[int, int, float, float, float] | None:
    methods = sorted(method_values)
    if len(methods) < 3:
        return None
    common_seeds = set.intersection(*(set(method_values[method]) for method in methods))
    if len(common_seeds) < 2:
        return None
    rank_sums = [0.0] * len(methods)
    for seed in sorted(common_seeds):
        ranks = _ranks([method_values[method][seed] for method in methods])
        rank_sums = [left + right for left, right in zip(rank_sums, ranks)]
    blocks = len(common_seeds)
    method_count = len(methods)
    statistic = (
        12.0 / (blocks * method_count * (method_count + 1)) * sum(value * value for value in rank_sums)
        - 3.0 * blocks * (method_count + 1)
    )
    p_value = _regularized_gamma_q((method_count - 1) / 2.0, statistic / 2.0)
    kendall_w = statistic / (blocks * (method_count - 1))
    return blocks, method_count, statistic, p_value, kendall_w


def _write_inferential_reports(
    reports: Path,
    all_rows: dict[str, list[dict[str, str]]],
) -> None:
    paired_rows: list[dict[str, object]] = []
    omnibus_rows: list[dict[str, object]] = []
    for suite, rows in all_rows.items():
        for metric in PRIMARY_METRICS[suite]:
            by_group: dict[str, dict[str, dict[int, float]]] = {}
            for row in rows:
                if metric not in row or row[metric] == "":
                    continue
                group = row.get("comparison_group") or row["scenario"]
                method = row.get("comparison_method") or row["algorithm"]
                by_group.setdefault(group, {}).setdefault(method, {})[int(row["seed"])] = float(row[metric])
            for group, methods in sorted(by_group.items()):
                friedman = _friedman(methods)
                if friedman is not None:
                    blocks, method_count, statistic, p_value, kendall_w = friedman
                    omnibus_rows.append(
                        {
                            "suite": suite,
                            "comparison_group": group,
                            "metric": metric,
                            "blocks": blocks,
                            "methods": method_count,
                            "friedman_statistic": statistic,
                            "p_value": p_value,
                            "kendall_w": kendall_w,
                        }
                    )
                reference = REFERENCE_METHODS[suite]
                if reference not in methods:
                    continue
                group_pairs: list[dict[str, object]] = []
                for method, values in sorted(methods.items()):
                    if method == reference:
                        continue
                    common = sorted(set(methods[reference]) & set(values))
                    if len(common) < 2:
                        continue
                    raw_differences = [values[seed] - methods[reference][seed] for seed in common]
                    oriented = [
                        -value if metric in LOWER_IS_BETTER else value
                        for value in raw_differences
                    ]
                    reference_mean = _mean([methods[reference][seed] for seed in common])
                    difference_mean = _mean(raw_differences)
                    group_pairs.append(
                        {
                            "suite": suite,
                            "comparison_group": group,
                            "metric": metric,
                            "reference": reference,
                            "comparison": method,
                            "pairs": len(common),
                            "mean_difference_comparison_minus_reference": difference_mean,
                            "relative_difference": difference_mean / abs(reference_mean) if reference_mean else 0.0,
                            "paired_effect_dz": _mean(oriented) / _std(oriented) if _std(oriented) else 0.0,
                            "sign_test_p_value": _sign_test_pvalue(oriented),
                        }
                    )
                ordered = sorted(group_pairs, key=lambda row: float(row["sign_test_p_value"]))
                count = len(ordered)
                running = 0.0
                for index, row in enumerate(ordered):
                    adjusted = min(1.0, float(row["sign_test_p_value"]) * (count - index))
                    running = max(running, adjusted)
                    row["holm_adjusted_p_value"] = running
                    paired_rows.append(row)
    paired_fields = (
        "suite",
        "comparison_group",
        "metric",
        "reference",
        "comparison",
        "pairs",
        "mean_difference_comparison_minus_reference",
        "relative_difference",
        "paired_effect_dz",
        "sign_test_p_value",
        "holm_adjusted_p_value",
    )
    with (reports / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    omnibus_fields = (
        "suite",
        "comparison_group",
        "metric",
        "blocks",
        "methods",
        "friedman_statistic",
        "p_value",
        "kendall_w",
    )
    with (reports / "omnibus_tests.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=omnibus_fields)
        writer.writeheader()
        writer.writerows(omnibus_rows)


def build_statistical_report(
    output_root: Path,
    suite_paths: Iterable[tuple[str, Path]],
) -> Path:
    report_rows: list[dict[str, object]] = []
    all_rows: dict[str, list[dict[str, str]]] = {}
    for suite, suite_path in suite_paths:
        raw_path = suite_path / "raw_results.csv"
        if not raw_path.exists():
            continue
        with raw_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        all_rows[suite] = rows
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            group = row.get("comparison_group") or row["scenario"]
            method = row.get("comparison_method") or row["algorithm"]
            groups.setdefault((group, method), []).append(row)
        for (group, method), method_rows in sorted(groups.items()):
            for metric in PRIMARY_METRICS[suite]:
                if metric not in method_rows[0] or method_rows[0][metric] == "":
                    continue
                values = [float(row[metric]) for row in method_rows]
                report_rows.append(
                    {
                        "suite": suite,
                        "comparison_group": group,
                        "comparison_method": method,
                        "metric": metric,
                        "runs": len(values),
                        "mean": _mean(values),
                        "std": _std(values),
                        "ci95_half_width": _ci95(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "descriptive_statistics.csv"
    fields = (
        "suite",
        "comparison_group",
        "comparison_method",
        "metric",
        "runs",
        "mean",
        "std",
        "ci95_half_width",
        "minimum",
        "maximum",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    _write_inferential_reports(reports, all_rows)
    return path
