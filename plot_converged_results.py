from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiment_framework.statistics import PRIMARY_METRICS


def _suite_path(output_root: Path, suite: str) -> Path:
    return output_root / "search_quality" if suite == "search_quality" else output_root / "system_poisson" / suite


def _write_svg(
    path: Path,
    *,
    title: str,
    metric: str,
    labels: list[str],
    values: list[float],
    errors: list[float],
) -> None:
    width = max(760, 110 * len(values) + 140)
    height = 480
    left, top, right, bottom = 80, 55, 30, 135
    chart_width = width - left - right
    chart_height = height - top - bottom
    upper = max((value + error for value, error in zip(values, errors)), default=1.0)
    upper = upper if upper > 0.0 else 1.0
    slot = chart_width / max(len(values), 1)
    bar_width = slot * 0.62
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333"/>',
        f'<text x="18" y="{top + chart_height / 2:.1f}" transform="rotate(-90 18 {top + chart_height / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(metric.replace("_", " "))}</text>',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + chart_height * (1 - ratio)
        value = upper * ratio
        elements.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3g}</text>',
            )
        )
    for index, (label, value, error) in enumerate(zip(labels, values, errors)):
        center = left + slot * (index + 0.5)
        bar_height = chart_height * value / upper
        x = center - bar_width / 2
        y = top + chart_height - bar_height
        error_height = chart_height * error / upper
        elements.extend(
            (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#4c78a8"/>',
                f'<line x1="{center:.1f}" y1="{max(top, y - error_height):.1f}" x2="{center:.1f}" y2="{min(top + chart_height, y + error_height):.1f}" stroke="#222"/>',
                f'<text x="{center:.1f}" y="{top + chart_height + 18:.1f}" transform="rotate(35 {center:.1f} {top + chart_height + 18:.1f})" text-anchor="start" font-family="sans-serif" font-size="10">{html.escape(label)}</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def generate_plots(output_root: Path, suites: Iterable[str]) -> Path:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for suite in suites:
        summary_path = _suite_path(output_root, suite) / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        metric = PRIMARY_METRICS[suite][0]
        mean_field = f"{metric}_mean"
        std_field = f"{metric}_std"
        if mean_field not in rows[0]:
            continue
        labels = [
            row.get("comparison_method")
            or row.get("scenario")
            or row.get("algorithm")
            or str(index)
            for index, row in enumerate(rows)
        ]
        values = [float(row[mean_field]) for row in rows]
        errors = [float(row.get(std_field, 0.0) or 0.0) for row in rows]
        path = figure_root / f"{suite}_{metric}.svg"
        _write_svg(
            path,
            title=suite.replace("_", " ").title(),
            metric=metric,
            labels=labels,
            values=values,
            errors=errors,
        )
        outputs = [path.relative_to(output_root).as_posix()]
        manifest.append(
            {
                "suite": suite,
                "metric": metric,
                "source": summary_path.relative_to(output_root).as_posix(),
                "outputs": outputs,
            }
        )
    manifest_path = figure_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "converged_figures_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "figures": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(manifest)} converged figures under {figure_root}")
    return manifest_path
