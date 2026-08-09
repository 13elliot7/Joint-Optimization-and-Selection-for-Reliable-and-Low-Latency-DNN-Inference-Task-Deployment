from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence


DEFAULT_METRICS = (
    "avg_estimated_delay",
    "avg_operational_stability_score",
    "avg_total_energy",
    "rejected_or_failed_count",
    "runtime_ms",
)
PARETO_METRICS = (
    "pareto_point_count",
    "pareto_point_count_mean",
    "pareto_hypervolume",
    "pareto_hypervolume_sum",
)
COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)
SUITE_ORDER = {
    "dynamic": ("weak", "medium", "strong"),
    "scale": ("small", "medium", "large"),
    "preference": ("delay_sensitive", "stability_sensitive", "energy_sensitive", "balanced"),
    "ablation": (
        "full",
        "wo_dag_init",
        "wo_block_crossover",
        "wo_skew_mutation",
        "wo_elite_local_search",
        "nsga_only",
    ),
}


@dataclass(frozen=True)
class SeriesPoint:
    label: str
    x_value: float
    y_value: float
    y_std: float


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def metric_mean(row: Dict[str, str], metric: str) -> float:
    return as_float(row.get(f"{metric}_mean", row.get(metric)))


def metric_std(row: Dict[str, str], metric: str) -> float:
    return as_float(row.get(f"{metric}_std"), 0.0)


def scenario_sort_key(suite: str, scenario: str) -> tuple[int, float | str]:
    ordered = SUITE_ORDER.get(suite)
    if ordered and scenario in ordered:
        return 0, ordered.index(scenario)
    match = re.search(r"(-?\d+(?:\.\d+)?)", scenario)
    if match:
        return 0, float(match.group(1))
    return 1, scenario


def scenario_x_value(suite: str, scenario: str, index: int) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", scenario)
    if match and suite not in {"dynamic", "ablation", "scale", "preference"}:
        return float(match.group(1))
    return float(index)


def escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10 ** exponent
    scaled = value / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def svg_frame(width: int, height: int, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
        f"{escape(title)}</text>\n"
        f"{body}\n</svg>\n"
    )


def axis_svg(
    width: int,
    height: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
    y_max: float,
    x_labels: Sequence[str],
) -> tuple[str, Callable[[float], float], Callable[[float], float], tuple[int, int, int, int]]:
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    y_top = nice_upper(y_max)

    def x_map(x: float) -> float:
        if len(x_labels) <= 1:
            return plot_left + plot_width / 2
        return plot_left + x * plot_width / (len(x_labels) - 1)

    def y_map(y: float) -> float:
        return plot_bottom - (y / y_top) * plot_height

    parts = [
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = y_top * tick / 5
        y = y_map(value)
        parts.append(f'<line x1="{plot_left - 4}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{value:.3g}</text>'
        )
    for idx, label in enumerate(x_labels):
        x = x_map(float(idx))
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 4}" stroke="#333"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{escape(label)}</text>'
        )
    return "\n".join(parts), x_map, y_map, (plot_left, plot_top, plot_right, plot_bottom)


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_series(summary_rows: Sequence[Dict[str, str]], suite: str, metric: str) -> tuple[List[str], Dict[str, List[SeriesPoint]]]:
    scenarios = sorted({row["scenario"] for row in summary_rows}, key=lambda item: scenario_sort_key(suite, item))
    scenario_index = {scenario: idx for idx, scenario in enumerate(scenarios)}
    by_algorithm: Dict[str, List[SeriesPoint]] = {}
    for row in summary_rows:
        scenario = row["scenario"]
        algorithm = row["algorithm"]
        idx = scenario_index[scenario]
        by_algorithm.setdefault(algorithm, []).append(
            SeriesPoint(
                label=scenario,
                x_value=scenario_x_value(suite, scenario, idx),
                y_value=metric_mean(row, metric),
                y_std=metric_std(row, metric),
            )
        )
    for points in by_algorithm.values():
        points.sort(key=lambda point: scenario_index[point.label])
    return scenarios, by_algorithm


def line_chart(summary_rows: Sequence[Dict[str, str]], suite: str, metric: str, output_path: Path) -> None:
    x_labels, by_algorithm = build_series(summary_rows, suite, metric)
    if not x_labels or not by_algorithm:
        return
    y_max = max((point.y_value + point.y_std for points in by_algorithm.values() for point in points), default=1.0)
    width, height = 980, 560
    axes, x_map, y_map, bounds = axis_svg(width, height, 78, 210, 52, 72, y_max, x_labels)
    plot_left, _, plot_right, _ = bounds
    parts = [axes]
    for idx, (algorithm, points) in enumerate(sorted(by_algorithm.items())):
        color = COLORS[idx % len(COLORS)]
        coords = [
            f"{x_map(float(x_labels.index(point.label))):.1f},{y_map(point.y_value):.1f}"
            for point in points
        ]
        if len(coords) > 1:
            parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for point in points:
            x = x_map(float(x_labels.index(point.label)))
            y = y_map(point.y_value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            if point.y_std > 0:
                y_low = y_map(max(0.0, point.y_value - point.y_std))
                y_high = y_map(point.y_value + point.y_std)
                parts.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{color}"/>')
                parts.append(f'<line x1="{x - 4:.1f}" y1="{y_high:.1f}" x2="{x + 4:.1f}" y2="{y_high:.1f}" stroke="{color}"/>')
                parts.append(f'<line x1="{x - 4:.1f}" y1="{y_low:.1f}" x2="{x + 4:.1f}" y2="{y_low:.1f}" stroke="{color}"/>')
        legend_y = 68 + idx * 22
        parts.append(f'<rect x="{plot_right + 22}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{plot_right + 40}" y="{legend_y}" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(algorithm)}</text>'
        )
    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 22}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="13">scenario</text>'
    )
    parts.append(
        f'<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{escape(metric)}</text>'
    )
    write_svg(output_path, svg_frame(width, height, f"{suite}: {metric}", "\n".join(parts)))


def bar_chart(summary_rows: Sequence[Dict[str, str]], suite: str, metric: str, output_path: Path) -> None:
    x_labels, by_algorithm = build_series(summary_rows, suite, metric)
    if not x_labels or not by_algorithm:
        return
    algorithms = sorted(by_algorithm)
    y_max = max((point.y_value + point.y_std for points in by_algorithm.values() for point in points), default=1.0)
    width, height = 1060, 580
    axes, _, y_map, bounds = axis_svg(width, height, 84, 220, 54, 92, y_max, x_labels)
    plot_left, _, plot_right, plot_bottom = bounds
    group_width = (plot_right - plot_left) / max(len(x_labels), 1)
    bar_width = group_width * 0.72 / max(len(algorithms), 1)
    parts = [axes]
    point_lookup = {
        (algorithm, point.label): point
        for algorithm, points in by_algorithm.items()
        for point in points
    }
    for scenario_idx, scenario in enumerate(x_labels):
        group_left = plot_left + scenario_idx * group_width + group_width * 0.14
        for algorithm_idx, algorithm in enumerate(algorithms):
            point = point_lookup.get((algorithm, scenario))
            if point is None:
                continue
            color = COLORS[algorithm_idx % len(COLORS)]
            x = group_left + algorithm_idx * bar_width
            y = y_map(point.y_value)
            height_value = plot_bottom - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.86:.1f}" '
                f'height="{height_value:.1f}" fill="{color}" opacity="0.86"/>'
            )
            if point.y_std > 0:
                cx = x + bar_width * 0.43
                y_low = y_map(max(0.0, point.y_value - point.y_std))
                y_high = y_map(point.y_value + point.y_std)
                parts.append(f'<line x1="{cx:.1f}" y1="{y_high:.1f}" x2="{cx:.1f}" y2="{y_low:.1f}" stroke="{color}" opacity="0.72"/>')
                parts.append(f'<line x1="{cx - 4:.1f}" y1="{y_high:.1f}" x2="{cx + 4:.1f}" y2="{y_high:.1f}" stroke="{color}" opacity="0.72"/>')
                parts.append(f'<line x1="{cx - 4:.1f}" y1="{y_low:.1f}" x2="{cx + 4:.1f}" y2="{y_low:.1f}" stroke="{color}" opacity="0.72"/>')
    for idx, algorithm in enumerate(algorithms):
        color = COLORS[idx % len(COLORS)]
        legend_y = 70 + idx * 22
        parts.append(f'<rect x="{plot_right + 24}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{plot_right + 42}" y="{legend_y}" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(algorithm)}</text>'
        )
    parts.append(
        f'<text x="22" y="{height / 2:.1f}" transform="rotate(-90 22 {height / 2:.1f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{escape(metric)}</text>'
    )
    write_svg(output_path, svg_frame(width, height, f"{suite}: {metric}", "\n".join(parts)))


def scatter_chart(rows: Sequence[Dict[str, str]], x_metric: str, y_metric: str, output_path: Path) -> None:
    points = [
        row for row in rows
        if row.get(x_metric) not in (None, "") and row.get(y_metric) not in (None, "")
    ]
    if not points:
        return
    width, height = 840, 560
    margin_left, margin_right, margin_top, margin_bottom = 78, 180, 52, 70
    x_max = nice_upper(max(as_float(row[x_metric]) for row in points))
    y_max = nice_upper(max(as_float(row[y_metric]) for row in points))
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom
    plot_width, plot_height = plot_right - plot_left, plot_bottom - plot_top

    def x_map(x: float) -> float:
        return plot_left + (x / x_max) * plot_width if x_max else plot_left

    def y_map(y: float) -> float:
        return plot_bottom - (y / y_max) * plot_height if y_max else plot_bottom

    algorithms = sorted({row["algorithm"] for row in points})
    color_by_algorithm = {algorithm: COLORS[idx % len(COLORS)] for idx, algorithm in enumerate(algorithms)}
    parts = [
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333"/>',
    ]
    for tick in range(6):
        xv = x_max * tick / 5
        x = x_map(xv)
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_top}" stroke="#eee"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{xv:.3g}</text>'
        )
        yv = y_max * tick / 5
        y = y_map(yv)
        parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{yv:.3g}</text>'
        )
    for row in points:
        color = color_by_algorithm[row["algorithm"]]
        x = x_map(as_float(row[x_metric]))
        y = y_map(as_float(row[y_metric]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}" opacity="0.72"/>')
    for idx, algorithm in enumerate(algorithms):
        color = color_by_algorithm[algorithm]
        legend_y = 70 + idx * 22
        parts.append(f'<rect x="{plot_right + 22}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{plot_right + 40}" y="{legend_y}" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(algorithm)}</text>'
        )
    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 24}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="13">{escape(x_metric)}</text>'
    )
    parts.append(
        f'<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{escape(y_metric)}</text>'
    )
    write_svg(output_path, svg_frame(width, height, f"Pareto: {x_metric} vs {y_metric}", "\n".join(parts)))


def pareto_3d_chart(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    normalized_metrics = (
        "operational_stability_norm",
        "delay_norm",
        "energy_norm",
    )
    raw_metrics = ("operational_stability", "delay_utility", "total_energy")
    if all(any(row.get(metric) not in (None, "") for row in rows) for metric in normalized_metrics):
        x_metric, y_metric, z_metric = normalized_metrics
        x_max = y_max = z_max = 1.0
        title_suffix = "normalized"
    else:
        x_metric, y_metric, z_metric = raw_metrics
        points_with_values = [
            row for row in rows
            if row.get(x_metric) not in (None, "") and row.get(y_metric) not in (None, "") and row.get(z_metric) not in (None, "")
        ]
        if not points_with_values:
            return
        x_max = nice_upper(max(as_float(row[x_metric]) for row in points_with_values))
        y_max = nice_upper(max(as_float(row[y_metric]) for row in points_with_values))
        z_max = nice_upper(max(as_float(row[z_metric]) for row in points_with_values))
        title_suffix = "raw"

    points = [
        row for row in rows
        if row.get(x_metric) not in (None, "") and row.get(y_metric) not in (None, "") and row.get(z_metric) not in (None, "")
    ]
    if not points:
        return

    width, height = 1040, 660
    origin = (120.0, 520.0)
    x_vec = (520.0, 0.0)
    y_vec = (0.0, -360.0)
    z_vec = (190.0, -130.0)

    def normalize(value: float, upper: float) -> float:
        return max(0.0, min(1.0, value / upper)) if upper > 0 else 0.0

    def project(x_value: float, y_value: float, z_value: float) -> tuple[float, float]:
        x_norm = normalize(x_value, x_max)
        y_norm = normalize(y_value, y_max)
        z_norm = normalize(z_value, z_max)
        return (
            origin[0] + x_vec[0] * x_norm + y_vec[0] * y_norm + z_vec[0] * z_norm,
            origin[1] + x_vec[1] * x_norm + y_vec[1] * y_norm + z_vec[1] * z_norm,
        )

    algorithms = sorted({row["algorithm"] for row in points})
    color_by_algorithm = {algorithm: COLORS[idx % len(COLORS)] for idx, algorithm in enumerate(algorithms)}
    parts: List[str] = []

    axis_specs = (
        ((0.0, 0.0, 0.0), (x_max, 0.0, 0.0), x_metric),
        ((0.0, 0.0, 0.0), (0.0, y_max, 0.0), y_metric),
        ((0.0, 0.0, 0.0), (0.0, 0.0, z_max), z_metric),
    )
    for start, end, label in axis_specs:
        x1, y1 = project(*start)
        x2, y2 = project(*end)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#222" stroke-width="1.4"/>')
        parts.append(
            f'<text x="{x2 + 8:.1f}" y="{y2 - 6:.1f}" font-family="Arial, sans-serif" '
            f'font-size="12" font-weight="700">{escape(label)}</text>'
        )
        for tick in range(1, 6):
            t = tick / 5
            tick_point = (
                end[0] * t,
                end[1] * t,
                end[2] * t,
            )
            tx, ty = project(*tick_point)
            parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="1.5" fill="#555"/>')
            tick_value = (x_max if label == x_metric else y_max if label == y_metric else z_max) * t
            parts.append(
                f'<text x="{tx + 5:.1f}" y="{ty + 13:.1f}" font-family="Arial, sans-serif" '
                f'font-size="10" fill="#555">{tick_value:.2g}</text>'
            )

    for grid in range(1, 6):
        t = grid / 5
        x0, y0 = project(0.0, y_max * t, 0.0)
        x1, y1 = project(x_max, y_max * t, 0.0)
        x2, y2 = project(0.0, y_max * t, z_max)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#e8e8e8"/>')
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e8e8e8"/>')

    sorted_points = sorted(
        points,
        key=lambda row: as_float(row[x_metric]) / max(x_max, 1e-12) + as_float(row[z_metric]) / max(z_max, 1e-12),
    )
    for row in sorted_points:
        color = color_by_algorithm[row["algorithm"]]
        x, y = project(as_float(row[x_metric]), as_float(row[y_metric]), as_float(row[z_metric]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" opacity="0.68"/>')

    legend_x = 825
    for idx, algorithm in enumerate(algorithms):
        color = color_by_algorithm[algorithm]
        legend_y = 78 + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="13" height="13" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 20}" y="{legend_y}" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(algorithm)}</text>'
        )
    parts.append(
        f'<text x="{legend_x}" y="210" font-family="Arial, sans-serif" font-size="11" fill="#555">'
        f'points: {len(points)}</text>'
    )
    write_svg(output_path, svg_frame(width, height, f"Pareto 3D comparison ({title_suffix})", "\n".join(parts)))


def suite_plot_style(suite: str) -> str:
    if suite in {"overall", "sensitivity", "scale"}:
        return "line"
    return "bar"


def plot_suite(suite_dir: Path, output_dir: Path, metrics: Sequence[str]) -> List[Path]:
    suite = suite_dir.name
    summary_rows = read_csv(suite_dir / "summary.csv")
    if not summary_rows:
        return []
    output_paths: List[Path] = []
    plotter = line_chart if suite_plot_style(suite) == "line" else bar_chart
    available_fields = set(summary_rows[0])
    for metric in metrics:
        if f"{metric}_mean" not in available_fields and metric not in available_fields:
            continue
        path = output_dir / suite / f"{metric}.svg"
        plotter(summary_rows, suite, metric, path)
        output_paths.append(path)

    pareto_rows = read_csv(suite_dir / "pareto_points.csv")
    if pareto_rows:
        scatter_specs = (
            ("delay_utility", "operational_stability"),
            ("energy_satisfaction", "operational_stability"),
            ("delay_utility", "energy_satisfaction"),
        )
        for x_metric, y_metric in scatter_specs:
            path = output_dir / suite / f"pareto_{x_metric}_vs_{y_metric}.svg"
            scatter_chart(pareto_rows, x_metric, y_metric, path)
            output_paths.append(path)
        path = output_dir / suite / "pareto_3d_comparison.svg"
        pareto_3d_chart(pareto_rows, path)
        if path.exists():
            output_paths.append(path)
    return output_paths


def discover_suite_dirs(results_dir: Path, requested_suites: Sequence[str]) -> List[Path]:
    if (results_dir / "summary.csv").exists():
        return [results_dir]
    suite_dirs = [
        path for path in results_dir.iterdir()
        if path.is_dir() and (path / "summary.csv").exists()
    ]
    if requested_suites:
        requested = set(requested_suites)
        suite_dirs = [path for path in suite_dirs if path.name in requested]
    return sorted(suite_dirs, key=lambda path: path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SVG figures from experiment result CSV files")
    parser.add_argument("--results-dir", type=Path, default=Path("experiment_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_figures"))
    parser.add_argument("--suites", default="", help="comma-separated suite names; empty means all available")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS + PARETO_METRICS),
        help="comma-separated metric base names to plot",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    requested_suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    suite_dirs = discover_suite_dirs(args.results_dir, requested_suites)
    if not suite_dirs:
        parser.error(f"no result suites found under {args.results_dir}")

    written: List[Path] = []
    for suite_dir in suite_dirs:
        written.extend(plot_suite(suite_dir, args.output_dir, metrics))
    manifest = {
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
        "figure_count": len(written),
        "figures": [str(path) for path in written],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(written)} figure(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
