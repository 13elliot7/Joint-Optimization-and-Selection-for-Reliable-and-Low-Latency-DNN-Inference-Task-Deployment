from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiment_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

COLORS = {
    "DAREED": "#1b6ca8",
    "NSGA-ED": "#d95f02",
    "RTBL": "#7570b3",
    "RDA": "#666666",
    "MRD": "#1b9e77",
    "LPD": "#e7298a",
    "SAA": "#66a61e",
    "Full DAREED": "#1b6ca8",
}
FALLBACK_COLORS = ("#1b6ca8", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#666666")

ALGORITHM_LABELS = {
    "customized": "DAREED",
    "proposed": "NSGA-ED",
    "rtbl": "RTBL",
    "random": "RDA",
    "maxresource_fast": "MRD",
    "localfirst": "LPD",
    "sa": "SAA",
}
ALGORITHM_ORDER = ("DAREED", "NSGA-ED", "RTBL", "RDA", "MRD", "LPD", "SAA")

VARIANT_LABELS = {
    "full": "Full DAREED",
    "wo_dag_init": "w/o DAG init.",
    "wo_block_crossover": "w/o block crossover",
    "wo_skew_mutation": "w/o skew mutation",
    "wo_elite_local_search": "w/o elite search",
    "nsga_only": "Plain evolutionary",
}
VARIANT_ORDER = tuple(VARIANT_LABELS.values())

METRIC_LABELS = {
    "avg_operational_stability_score": "Operational stability score",
    "avg_estimated_delay": "Delay (ms)",
    "avg_total_energy": "Energy",
    "rejected_or_failed_count": "Failed deployments",
    "runtime_ms": "Runtime (ms)",
    "pareto_hypervolume": "Hypervolume",
    "pareto_point_count_mean": "Pareto points/request",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fnum(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def mean(row: dict[str, str], metric: str) -> float:
    return fnum(row.get(f"{metric}_mean", row.get(metric)))


def std(row: dict[str, str], metric: str) -> float:
    return fnum(row.get(f"{metric}_std"), 0.0)


def algorithm_label(raw: str) -> str:
    return ALGORITHM_LABELS.get(raw, raw)


def variant_label(raw: str) -> str:
    return VARIANT_LABELS.get(raw, raw)


def scenario_number(scenario: str) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", scenario)
    return float(match.group(1)) if match else 0.0


def scenario_label(scenario: str) -> str:
    labels = {
        "weak": "Weak",
        "medium": "Medium",
        "strong": "Strong",
        "small": "Small",
        "large": "Large",
        "delay_sensitive": "Delay-sensitive",
        "stability_sensitive": "Stability-sensitive",
        "energy_sensitive": "Energy-sensitive",
        "balanced": "Balanced",
    }
    if scenario.startswith("dnn_"):
        return scenario.split("_", 1)[1]
    if scenario.startswith("population_"):
        return scenario.split("_", 1)[1]
    if scenario.startswith("iterations_"):
        return scenario.split("_", 1)[1]
    if scenario.startswith("mutation_"):
        return scenario.split("_", 1)[1]
    return labels.get(scenario, scenario)


def escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def segment_polygon(x1: float, y1: float, x2: float, y2: float, width: float) -> str:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return f"{x1:.1f},{y1:.1f}"
    nx = -dy / length * width / 2
    ny = dx / length * width / 2
    return " ".join(
        (
            f"{x1 + nx:.1f},{y1 + ny:.1f}",
            f"{x2 + nx:.1f},{y2 + ny:.1f}",
            f"{x2 - nx:.1f},{y2 - ny:.1f}",
            f"{x1 - nx:.1f},{y1 - ny:.1f}",
        )
    )


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
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


def svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style><![CDATA[\n'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#222} '
        '.title{font-size:18px;font-weight:700} .axis{stroke:#333;stroke-width:1} '
        '.grid{stroke:#e6e6e6;stroke-width:1} .tick{font-size:10px;fill:#555} '
        '.label{font-size:12px;fill:#222} .legend{font-size:11px;fill:#222}\n'
        ']]></style>\n'
        f"{body}\n</svg>\n"
    )


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(width, height, body), encoding="utf-8")


def convert_to_pdf(svg_path: Path) -> Path | None:
    convert = shutil.which("magick") or shutil.which("convert")
    if not convert:
        return None
    pdf_path = svg_path.with_suffix(".pdf")
    font_path = Path("/System/Library/Fonts/SFNS.ttf")
    command = [convert]
    if font_path.exists():
        command.extend(["-font", str(font_path)])
    command.extend([str(svg_path), str(pdf_path)])
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        return None
    return pdf_path if pdf_path.exists() else None


def draw_legend(items: Sequence[str], x: float, y: float) -> str:
    parts: list[str] = []
    for idx, item in enumerate(items):
        color = COLORS.get(item, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
        yy = y + idx * 18
        parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 16}" y2="{yy}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<circle cx="{x + 8}" cy="{yy}" r="3.2" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{x + 23}" y="{yy + 4}">{escape(item)}</text>')
    return "\n".join(parts)


def draw_line_panel(
    rows: Sequence[dict[str, str]],
    metric: MetricSpec,
    x_labels: Sequence[str],
    algorithms: Sequence[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
) -> str:
    top, right, bottom, left = 26.0, 12.0, 44.0, 58.0
    px0, py0 = x + left, y + top
    px1, py1 = x + width - right, y + height - bottom
    plot_w, plot_h = px1 - px0, py1 - py0
    lookup = {(scenario_label(r["scenario"]), algorithm_label(r["algorithm"])): r for r in rows}
    y_max = max(
        (mean(r, metric.key) + std(r, metric.key) for r in rows if algorithm_label(r["algorithm"]) in algorithms),
        default=1.0,
    )
    if "stability" in metric.key:
        raw_min = min((mean(r, metric.key) - std(r, metric.key) for r in rows), default=0.0)
        raw_max = max((mean(r, metric.key) + std(r, metric.key) for r in rows), default=1.0)
        y_min = max(0.0, math.floor(max(0.0, raw_min - 0.02) * 20) / 20)
        y_max = min(1.0, math.ceil(min(1.0, raw_max + 0.01) * 20) / 20)
        if y_max - y_min < 0.1:
            y_min = max(0.0, y_min - 0.05)
            y_max = min(1.0, y_max + 0.05)
    else:
        y_min = 0.0
        y_max = nice_upper(y_max)

    def xm(index: int) -> float:
        if len(x_labels) == 1:
            return px0 + plot_w / 2
        return px0 + index * plot_w / (len(x_labels) - 1)

    def ym(value: float) -> float:
        return py1 - ((value - y_min) / max(y_max - y_min, 1e-12)) * plot_h

    parts = [f'<text class="label" x="{x + width / 2}" y="{y + 15}" text-anchor="middle">{escape(title)}</text>']
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        yy = ym(value)
        parts.append(f'<line class="grid" x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="tick" x="{px0 - 7}" y="{yy + 3:.1f}" text-anchor="end">{value:.3g}</text>')
    parts.append(f'<line class="axis" x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}"/>')
    parts.append(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}"/>')
    for idx, label in enumerate(x_labels):
        xx = xm(idx)
        parts.append(f'<line class="axis" x1="{xx:.1f}" y1="{py1}" x2="{xx:.1f}" y2="{py1 + 4}"/>')
        parts.append(f'<text class="tick" x="{xx:.1f}" y="{py1 + 18}" text-anchor="middle">{escape(label)}</text>')
    parts.append(
        f'<text class="tick" x="{x + 12}" y="{(py0 + py1) / 2:.1f}" '
        f'transform="rotate(-90 {x + 12} {(py0 + py1) / 2:.1f})" text-anchor="middle">{escape(metric.label)}</text>'
    )

    for aidx, algorithm in enumerate(algorithms):
        color = COLORS.get(algorithm, FALLBACK_COLORS[aidx % len(FALLBACK_COLORS)])
        coords: list[str] = []
        for idx, label in enumerate(x_labels):
            row = lookup.get((label, algorithm))
            if row is None:
                continue
            coords.append(f"{xm(idx):.1f},{ym(mean(row, metric.key)):.1f}")
        for idx, label in enumerate(x_labels):
            row = lookup.get((label, algorithm))
            if row is None:
                continue
            xx, yy = xm(idx), ym(mean(row, metric.key))
            err = std(row, metric.key)
            if err:
                ylo, yhi = ym(max(y_min, mean(row, metric.key) - err)), ym(min(y_max, mean(row, metric.key) + err))
                cap = 4.2 if len(algorithms) > 1 else 5.5
                parts.append(
                    f'<line x1="{xx:.1f}" y1="{yhi:.1f}" x2="{xx:.1f}" y2="{ylo:.1f}" '
                    f'stroke="{color}" stroke-width="1.0" opacity="0.68"/>'
                )
                parts.append(
                    f'<line x1="{xx - cap:.1f}" y1="{yhi:.1f}" x2="{xx + cap:.1f}" y2="{yhi:.1f}" '
                    f'stroke="{color}" stroke-width="1.0" opacity="0.68"/>'
                )
                parts.append(
                    f'<line x1="{xx - cap:.1f}" y1="{ylo:.1f}" x2="{xx + cap:.1f}" y2="{ylo:.1f}" '
                    f'stroke="{color}" stroke-width="1.0" opacity="0.68"/>'
                )
        if len(coords) > 1:
            for start, end in zip(coords, coords[1:]):
                x1, y1 = start.split(",", 1)
                x2, y2 = end.split(",", 1)
                parts.append(
                    f'<polygon points="{segment_polygon(float(x1), float(y1), float(x2), float(y2), 2.4)}" '
                    f'fill="{color}" opacity="0.92"/>'
                )
        for idx, label in enumerate(x_labels):
            row = lookup.get((label, algorithm))
            if row is None:
                continue
            xx, yy = xm(idx), ym(mean(row, metric.key))
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.5" fill="{color}" stroke="white" stroke-width="0.9"/>')
    return "\n".join(parts)


def draw_grouped_bar_panel(
    rows: Sequence[dict[str, str]],
    metric: MetricSpec,
    x_labels: Sequence[str],
    series: Sequence[str],
    row_label_fn,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
) -> str:
    top, right, bottom, left = 26.0, 12.0, 50.0, 58.0
    px0, py0 = x + left, y + top
    px1, py1 = x + width - right, y + height - bottom
    plot_w, plot_h = px1 - px0, py1 - py0
    lookup = {(scenario_label(r["scenario"]), row_label_fn(r)): r for r in rows}
    y_max = nice_upper(max((mean(r, metric.key) + std(r, metric.key) for r in rows), default=1.0))

    def ym(value: float) -> float:
        return py1 - (value / y_max) * plot_h

    parts = [f'<text class="label" x="{x + width / 2}" y="{y + 15}" text-anchor="middle">{escape(title)}</text>']
    for tick in range(5):
        value = y_max * tick / 4
        yy = ym(value)
        parts.append(f'<line class="grid" x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="tick" x="{px0 - 7}" y="{yy + 3:.1f}" text-anchor="end">{value:.3g}</text>')
    parts.append(f'<line class="axis" x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}"/>')
    parts.append(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}"/>')
    group_w = plot_w / max(len(x_labels), 1)
    bar_w = group_w * 0.74 / max(len(series), 1)
    for gidx, label in enumerate(x_labels):
        gx = px0 + gidx * group_w + group_w * 0.13
        for sidx, item in enumerate(series):
            row = lookup.get((label, item))
            if row is None:
                continue
            color = COLORS.get(item, FALLBACK_COLORS[sidx % len(FALLBACK_COLORS)])
            value = mean(row, metric.key)
            bx = gx + sidx * bar_w
            by = ym(value)
            err = std(row, metric.key)
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w * 0.84:.1f}" height="{py1 - by:.1f}" '
                f'fill="{color}" opacity="0.86"/>'
            )
            if err:
                cx = bx + bar_w * 0.42
                ylo, yhi = ym(max(0.0, value - err)), ym(min(y_max, value + err))
                cap = max(3.0, min(6.0, bar_w * 0.22))
                parts.append(
                    f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" '
                    f'stroke="{color}" stroke-width="1.0" opacity="0.72"/>'
                )
                parts.append(
                    f'<line x1="{cx - cap:.1f}" y1="{yhi:.1f}" x2="{cx + cap:.1f}" y2="{yhi:.1f}" '
                    f'stroke="{color}" stroke-width="1.0" opacity="0.72"/>'
                )
                parts.append(
                    f'<line x1="{cx - cap:.1f}" y1="{ylo:.1f}" x2="{cx + cap:.1f}" y2="{ylo:.1f}" '
                    f'stroke="{color}" stroke-width="1.0" opacity="0.72"/>'
                )
                parts.append(
                    f'<line x1="{bx + bar_w * 0.16:.1f}" y1="{by:.1f}" '
                    f'x2="{bx + bar_w * 0.68:.1f}" y2="{by:.1f}" '
                    f'stroke="{color}" stroke-width="1.1" opacity="0.42"/>'
                )
        tx = px0 + gidx * group_w + group_w / 2
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{py1 + 17}" text-anchor="middle">{escape(label)}</text>')
    parts.append(
        f'<text class="tick" x="{x + 12}" y="{(py0 + py1) / 2:.1f}" '
        f'transform="rotate(-90 {x + 12} {(py0 + py1) / 2:.1f})" text-anchor="middle">{escape(metric.label)}</text>'
    )
    return "\n".join(parts)


def ordered_algorithms(rows: Sequence[dict[str, str]]) -> list[str]:
    present = {algorithm_label(row["algorithm"]) for row in rows}
    return [item for item in ALGORITHM_ORDER if item in present]


def make_multi_line_figure(
    rows: Sequence[dict[str, str]],
    path: Path,
    title: str,
    metrics: Sequence[MetricSpec],
    x_labels: Sequence[str],
    algorithms: Sequence[str],
) -> Path:
    cols = 2
    panel_w = 420
    panel_h = 250
    x0, y0 = 70, 58
    gap_x, gap_y = 50, 46
    row_count = math.ceil(len(metrics) / cols)
    width = 1180
    height = int(y0 + row_count * panel_h + max(0, row_count - 1) * gap_y + 56)
    parts = [f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">{escape(title)}</text>']
    for midx, metric in enumerate(metrics):
        col, row = midx % cols, midx // cols
        parts.append(
            draw_line_panel(
                rows,
                metric,
                x_labels,
                algorithms,
                x0 + col * (panel_w + gap_x),
                y0 + row * (panel_h + gap_y),
                panel_w,
                panel_h,
                metric.label,
            )
        )
    parts.append(draw_legend(algorithms, 1010, 78))
    write_svg(path, width, height, "\n".join(parts))
    return path


def make_multi_bar_figure(
    rows: Sequence[dict[str, str]],
    path: Path,
    title: str,
    metrics: Sequence[MetricSpec],
    x_labels: Sequence[str],
    series: Sequence[str],
    row_label_fn,
) -> Path:
    cols = 2
    panel_w = 420
    panel_h = 250
    x0, y0 = 70, 58
    gap_x, gap_y = 50, 46
    row_count = math.ceil(len(metrics) / cols)
    width = 1180
    height = int(y0 + row_count * panel_h + max(0, row_count - 1) * gap_y + 56)
    parts = [f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">{escape(title)}</text>']
    for midx, metric in enumerate(metrics):
        col, row = midx % cols, midx // cols
        parts.append(
            draw_grouped_bar_panel(
                rows,
                metric,
                x_labels,
                series,
                row_label_fn,
                x0 + col * (panel_w + gap_x),
                y0 + row * (panel_h + gap_y),
                panel_w,
                panel_h,
                metric.label,
            )
        )
    legend_x = 1010
    legend_parts = []
    for idx, item in enumerate(series):
        color = COLORS.get(item, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
        yy = 80 + idx * 18
        legend_parts.append(f'<rect x="{legend_x}" y="{yy - 10}" width="12" height="12" fill="{color}" opacity="0.86"/>')
        legend_parts.append(f'<text class="legend" x="{legend_x + 19}" y="{yy}">{escape(item)}</text>')
    parts.extend(legend_parts)
    write_svg(path, width, height, "\n".join(parts))
    return path


def filter_sensitivity(rows: Sequence[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    result = [row for row in rows if row["scenario"].startswith(prefix) and row["algorithm"] == "customized"]
    return sorted(result, key=lambda row: scenario_number(row["scenario"]))


def make_sensitivity_figures(written: list[Path]) -> None:
    rows = read_csv(RESULTS_DIR / "sensitivity" / "summary.csv")
    specs = [
        ("population_", "fig_param_population.svg", "Sensitivity under Different Population Sizes"),
        ("iterations_", "fig_param_iteration.svg", "Sensitivity under Different Iteration Numbers"),
        ("mutation_", "fig_param_mutation.svg", "Sensitivity under Different Mutation Probabilities"),
    ]
    metrics = [
        MetricSpec("avg_operational_stability_score", "OSS"),
        MetricSpec("avg_total_energy", "Energy"),
        MetricSpec("rejected_or_failed_count", "Failed"),
        MetricSpec("runtime_ms", "Runtime (ms)"),
    ]
    for prefix, filename, title in specs:
        subset = filter_sensitivity(rows, prefix)
        labels = [scenario_label(row["scenario"]) for row in subset]
        written.append(make_multi_line_figure(subset, OUTPUT_DIR / filename, title, metrics, labels, ["DAREED"]))


def make_overall_figures(written: list[Path]) -> None:
    rows = sorted(read_csv(RESULTS_DIR / "overall" / "summary.csv"), key=lambda row: scenario_number(row["scenario"]))
    x_labels = [str(v) for v in sorted({int(scenario_number(row["scenario"])) for row in rows})]
    algorithms = ordered_algorithms(rows)
    figure_specs = [
        (
            "fig_overall_quality.svg",
            "Average Deployment Quality under Different DNN Request Numbers",
            [
                MetricSpec("avg_operational_stability_score", "OSS"),
            ],
        ),
        (
            "fig_overall_delay_energy.svg",
            "Average Delay and Energy under Different DNN Request Numbers",
            [
                MetricSpec("avg_estimated_delay", "Delay (ms)"),
                MetricSpec("avg_total_energy", "Energy"),
            ],
        ),
        (
            "fig_overall_failure_runtime.svg",
            "Failed Deployments and Runtime under Different DNN Request Numbers",
            [
                MetricSpec("rejected_or_failed_count", "Failed"),
                MetricSpec("runtime_ms", "Runtime (ms)"),
            ],
        ),
    ]
    for filename, title, metrics in figure_specs:
        written.append(make_multi_line_figure(rows, OUTPUT_DIR / filename, title, metrics, x_labels, algorithms))


def make_pareto_figures(written: list[Path]) -> None:
    rows = read_csv(RESULTS_DIR / "pareto" / "summary.csv")
    algorithms = ordered_algorithms(rows)
    written.append(
        make_multi_bar_figure(
            rows,
            OUTPUT_DIR / "fig_pareto_hv.svg",
            "Three-Objective Pareto Front Quality Comparison",
            [
                MetricSpec("pareto_hypervolume", "Hypervolume"),
                MetricSpec("pareto_point_count_mean", "Pareto points/request"),
            ],
            ["30"],
            algorithms,
            lambda row: algorithm_label(row["algorithm"]),
        )
    )
    pareto_rows = read_csv(RESULTS_DIR / "pareto" / "pareto_points.csv")
    written.append(make_pareto_3d(pareto_rows, OUTPUT_DIR / "fig_pareto_3d.svg"))


def make_pareto_3d(rows: Sequence[dict[str, str]], path: Path) -> Path:
    metrics = (
        "operational_stability_norm",
        "delay_norm",
        "energy_norm",
    )
    valid_rows = [
        row
        for row in rows
        if all(row.get(metric) not in (None, "") for metric in metrics)
        and row.get("repeat") not in (None, "")
        and row.get("dnn_index") not in (None, "")
    ]
    if not valid_rows:
        raise ValueError("Pareto projection requires repeat, dnn_index, and three normalized objectives")

    # A Pareto front is defined for one optimization problem. Select the shared
    # repeat/request pair with the median combined set size as a reproducible,
    # representative visualization case instead of pooling heterogeneous cases.
    raw_algorithms = sorted({row["algorithm"] for row in valid_rows}, key=lambda item: ALGORITHM_ORDER.index(
        algorithm_label(item)
    ) if algorithm_label(item) in ALGORITHM_ORDER else len(ALGORITHM_ORDER))
    if len(raw_algorithms) > 4:
        raise ValueError("Pareto small multiples support at most four algorithms")
    case_algorithms: dict[tuple[int, int], set[str]] = {}
    for row in valid_rows:
        case = (int(row["repeat"]), int(row["dnn_index"]))
        case_algorithms.setdefault(case, set()).add(row["algorithm"])
    complete_cases = [
        case for case, algorithms in case_algorithms.items() if set(raw_algorithms).issubset(algorithms)
    ]
    if not complete_cases:
        raise ValueError("Pareto projection requires a repeat/DNN case shared by every algorithm")
    case_sizes = {
        case: sum(
            int(row["repeat"]) == case[0] and int(row["dnn_index"]) == case[1]
            for row in valid_rows
        )
        for case in complete_cases
    }
    median_size = sorted(case_sizes.values())[len(case_sizes) // 2]
    selected_repeat, selected_dnn = min(
        complete_cases,
        key=lambda case: (abs(case_sizes[case] - median_size), case),
    )
    points = [
        row
        for row in valid_rows
        if int(row["repeat"]) == selected_repeat and int(row["dnn_index"]) == selected_dnn
    ]

    def dominates(left: dict[str, str], right: dict[str, str]) -> bool:
        not_worse = all(fnum(left[metric]) >= fnum(right[metric]) for metric in metrics)
        strictly_better = any(fnum(left[metric]) > fnum(right[metric]) for metric in metrics)
        return not_worse and strictly_better

    reference_ids = {
        id(point)
        for point in points
        if not any(other is not point and dominates(other, point) for other in points)
    }

    def upper_tenth(values: Sequence[float]) -> float:
        return min(1.0, max(0.1, math.ceil(max(values) * 10 - 1e-9) / 10))

    stability_values = [fnum(row["operational_stability_norm"]) for row in points]
    axis_bounds = (
        (
            max(0.0, min(0.9, math.floor(min(stability_values) * 10) / 10)),
            1.0,
        ),
        (0.0, upper_tenth([fnum(row["delay_norm"]) for row in points])),
        (0.0, upper_tenth([fnum(row["energy_norm"]) for row in points])),
    )
    energy_values = [fnum(row["energy_norm"]) for row in points]
    energy_bounds = (
        max(0.0, math.floor(min(energy_values) * 10) / 10),
        min(1.0, math.ceil(max(energy_values) * 10 - 1e-9) / 10),
    )

    def energy_color(value: float) -> str:
        # ColorBrewer YlGnBu: pale colors are low energy satisfaction and dark
        # colors are high. Piecewise interpolation preserves contrast in print.
        stops = (
            (0.0, (255, 255, 204)),
            (0.5, (65, 182, 196)),
            (1.0, (34, 94, 168)),
        )
        value = (value - energy_bounds[0]) / max(energy_bounds[1] - energy_bounds[0], 1e-12)
        value = max(0.0, min(1.0, value))
        for (left_t, left_rgb), (right_t, right_rgb) in zip(stops, stops[1:]):
            if value <= right_t:
                ratio = (value - left_t) / (right_t - left_t)
                rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(left_rgb, right_rgb))
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        return "#225ea8"

    width, height = 1120, 760
    panel_width, panel_height = 535.0, 285.0
    panel_positions = ((15.0, 72.0), (570.0, 72.0), (15.0, 382.0), (570.0, 382.0))
    parts = [
        f'<text class="title" x="{width / 2}" y="27" text-anchor="middle">'
        "Three-objective Nondominated Sets: 3D Projection</text>",
        f'<text class="tick" x="{width / 2}" y="48" text-anchor="middle">'
        f"Median-size matched case: repeat {selected_repeat}, DNN request {selected_dnn}; "
        "larger values are better</text>",
    ]

    for panel_index, algorithm in enumerate(raw_algorithms):
        if panel_index >= len(panel_positions):
            break
        panel_x, panel_y = panel_positions[panel_index]
        algorithm_points = [row for row in points if row["algorithm"] == algorithm]
        # Symmetric axonometric projection: delay satisfaction is vertical,
        # while operational stability and energy recede in opposite directions. A complete
        # cube is drawn below so depth remains legible in a static paper figure.
        origin = (panel_x + 255.0, panel_y + 250.0)
        x_vec = (175.0, -42.0)
        y_vec = (0.0, -165.0)
        z_vec = (-145.0, -52.0)

        def normalize(value: float, bounds: tuple[float, float]) -> float:
            lower, upper = bounds
            return max(0.0, min(1.0, (value - lower) / max(upper - lower, 1e-12)))

        def project(x_value: float, y_value: float, z_value: float) -> tuple[float, float]:
            x_norm = normalize(x_value, axis_bounds[0])
            y_norm = normalize(y_value, axis_bounds[1])
            z_norm = normalize(z_value, axis_bounds[2])
            return project_normalized(x_norm, y_norm, z_norm)

        def project_normalized(x_norm: float, y_norm: float, z_norm: float) -> tuple[float, float]:
            return (
                origin[0] + x_vec[0] * x_norm + y_vec[0] * y_norm + z_vec[0] * z_norm,
                origin[1] + x_vec[1] * x_norm + y_vec[1] * y_norm + z_vec[1] * z_norm,
            )

        def tick_values(bounds: tuple[float, float]) -> list[float]:
            lower, upper = bounds
            step = 0.1 if upper - lower <= 0.5 + 1e-12 else 0.2
            values = []
            value = math.ceil((lower + 1e-9) / step) * step
            while value <= upper + 1e-9:
                values.append(min(value, upper))
                value += step
            if not values or abs(values[-1] - upper) > 1e-9:
                values.append(upper)
            return values

        parts.append(
            f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_width:.1f}" '
            f'height="{panel_height:.1f}" rx="4" fill="#fff" stroke="#d9d9d9"/>'
        )
        cube = {
            (x, y, z): project_normalized(x, y, z)
            for x in (0.0, 1.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        }

        def polygon(corners: Sequence[tuple[float, float, float]], fill: str) -> None:
            coordinates = " ".join(
                f"{cube[corner][0]:.1f},{cube[corner][1]:.1f}" for corner in corners
            )
            parts.append(f'<polygon points="{coordinates}" fill="{fill}" stroke="none"/>')

        def grid_line(
            start: tuple[float, float, float],
            end: tuple[float, float, float],
        ) -> None:
            x1, y1 = project_normalized(*start)
            x2, y2 = project_normalized(*end)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#d5dde5" stroke-width="0.7"/>'
            )

        # Three softly shaded coordinate planes make the viewing direction and
        # relative depth unambiguous without covering the data.
        polygon(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)), "#f5f7fa")
        polygon(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0)), "#eef3f7")
        polygon(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)), "#f8fafc")

        for grid_value in (0.25, 0.5, 0.75):
            # All three coordinate planes use the same normalized grid spacing.
            grid_line((grid_value, 0.0, 0.0), (grid_value, 0.0, 1.0))
            grid_line((0.0, 0.0, grid_value), (1.0, 0.0, grid_value))
            grid_line((0.0, grid_value, 0.0), (0.0, grid_value, 1.0))
            grid_line((0.0, 0.0, grid_value), (0.0, 1.0, grid_value))
            grid_line((grid_value, 0.0, 1.0), (grid_value, 1.0, 1.0))
            grid_line((0.0, grid_value, 1.0), (1.0, grid_value, 1.0))

        cube_edges = []
        for axis_index in range(3):
            other_axes = [axis for axis in range(3) if axis != axis_index]
            for first_fixed in (0.0, 1.0):
                for second_fixed in (0.0, 1.0):
                    start = [0.0, 0.0, 0.0]
                    end = [0.0, 0.0, 0.0]
                    start[axis_index] = 0.0
                    end[axis_index] = 1.0
                    start[other_axes[0]] = end[other_axes[0]] = first_fixed
                    start[other_axes[1]] = end[other_axes[1]] = second_fixed
                    cube_edges.append((tuple(start), tuple(end)))
        for start, end in cube_edges:
            x1, y1 = project_normalized(*start)
            x2, y2 = project_normalized(*end)
            front_edge = sum(start) <= 1e-12 or sum(end) <= 1e-12
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{"#4b5563" if front_edge else "#8b98a5"}" '
                f'stroke-width="{1.15 if front_edge else 0.8}"/>'
            )

        axes = (
            (0, axis_bounds[0], (1.0, 0.0, 0.0), "Operational stability"),
            (1, axis_bounds[1], (0.0, 1.0, 0.0), "Delay satisfaction"),
            (2, axis_bounds[2], (0.0, 0.0, 1.0), "Energy satisfaction"),
        )
        for axis_index, bounds, normalized_end, label in axes:
            x2, y2 = project_normalized(*normalized_end)
            if axis_index == 0:
                label_x, label_y, anchor = x2 + 10, y2 + 29, "start"
            elif axis_index == 1:
                label_x, label_y, anchor = x2, y2 - 18, "middle"
            else:
                label_x, label_y, anchor = x2 - 10, y2 + 29, "end"
            parts.append(
                f'<text class="tick" x="{label_x:.1f}" y="{label_y:.1f}" '
                f'text-anchor="{anchor}" font-weight="700">{escape(label)}</text>'
            )
            for tick_value in tick_values(bounds):
                tick_position = [0.0, 0.0, 0.0]
                tick_position[axis_index] = normalize(tick_value, bounds)
                tx, ty = project_normalized(*tick_position)
                if axis_index == 0:
                    tick_x, tick_y, tick_anchor = tx + 2, ty + 12, "middle"
                elif axis_index == 1:
                    tick_x, tick_y, tick_anchor = tx - 5, ty + 3, "end"
                else:
                    tick_x, tick_y, tick_anchor = tx - 3, ty + 12, "end"
                parts.append(
                    f'<text class="tick" x="{tick_x:.1f}" y="{tick_y:.1f}" '
                    f'text-anchor="{tick_anchor}">'
                    f"{tick_value:.1f}</text>"
                )

        ordered_points = sorted(
            algorithm_points,
            key=lambda row: (
                id(row) in reference_ids,
                project(
                    fnum(row["operational_stability_norm"]),
                    fnum(row["delay_norm"]),
                    fnum(row["energy_norm"]),
                )[1],
            ),
        )
        for row in ordered_points:
            xx, yy = project(
                fnum(row["operational_stability_norm"]),
                fnum(row["delay_norm"]),
                fnum(row["energy_norm"]),
            )
            is_reference = id(row) in reference_ids
            if is_reference:
                parts.append(
                    f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.0" fill="#111" opacity="0.9"/>'
                )
                parts.append(
                    f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.45" '
                    f'fill="{energy_color(fnum(row["energy_norm"]))}" opacity="0.96"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.2" '
                    f'fill="{energy_color(fnum(row["energy_norm"]))}" opacity="0.42" '
                    f'stroke="#fff" stroke-width="0.35"/>'
                )
        reference_count = sum(id(row) in reference_ids for row in algorithm_points)
        title_width = max(72.0, 9.0 * len(algorithm_label(algorithm)))
        parts.append(
            f'<rect x="{panel_x + panel_width / 2 - title_width / 2:.1f}" '
            f'y="{panel_y + 5:.1f}" width="{title_width:.1f}" height="22" rx="3" '
            'fill="#fff" fill-opacity="0.94" stroke="#d9d9d9" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text class="label" x="{panel_x + panel_width / 2:.1f}" y="{panel_y + 20:.1f}" '
            f'text-anchor="middle" font-weight="700">{escape(algorithm_label(algorithm))}</text>'
        )
        parts.append(
            f'<text class="tick" x="{panel_x + panel_width - 10:.1f}" '
            f'y="{panel_y + panel_height - 9:.1f}" text-anchor="end">'
            f"solutions {len(algorithm_points)} | reference contribution {reference_count}</text>"
        )

    colorbar_x, colorbar_y, colorbar_width = 330.0, 712.0, 280.0
    parts.append(
        f'<text class="legend" x="{colorbar_x - 12:.1f}" y="{colorbar_y + 10:.1f}" '
        'text-anchor="end">Energy satisfaction</text>'
    )
    color_steps = 28
    for color_index in range(color_steps):
        fraction = color_index / max(color_steps - 1, 1)
        actual_value = energy_bounds[0] + (energy_bounds[1] - energy_bounds[0]) * fraction
        rect_x = colorbar_x + colorbar_width * color_index / color_steps
        parts.append(
            f'<rect x="{rect_x:.1f}" y="{colorbar_y:.1f}" '
            f'width="{colorbar_width / color_steps + 0.2:.1f}" height="12" '
            f'fill="{energy_color(actual_value)}" stroke="none"/>'
        )
    parts.extend(
        (
            f'<rect x="{colorbar_x:.1f}" y="{colorbar_y:.1f}" width="{colorbar_width:.1f}" '
            'height="12" fill="none" stroke="#777" stroke-width="0.5"/>',
            f'<text class="tick" x="{colorbar_x:.1f}" y="{colorbar_y + 27:.1f}">'
            f"{energy_bounds[0]:.1f} (low)</text>",
            f'<text class="tick" x="{colorbar_x + colorbar_width:.1f}" y="{colorbar_y + 27:.1f}" '
            f'text-anchor="end">{energy_bounds[1]:.1f} (high)</text>',
            '<circle cx="745" cy="718" r="5" fill="#111"/>',
            '<circle cx="745" cy="718" r="3.4" fill="#41b6c4"/>',
            '<text class="legend" x="756" y="722">Member of joint four-objective reference front</text>',
        )
    )
    write_svg(path, width, height, "\n".join(parts))
    return path


def make_dynamic_figure(written: list[Path]) -> None:
    rows = read_csv(RESULTS_DIR / "dynamic" / "summary.csv")
    x_labels = ["Weak", "Medium", "Strong"]
    algorithms = ordered_algorithms(rows)
    written.append(
        make_multi_bar_figure(
            rows,
            OUTPUT_DIR / "fig_dynamic_degradation.svg",
            "Performance under Dynamic Quality Degradation",
            [
                MetricSpec("avg_operational_stability_score", "OSS"),
                MetricSpec("avg_total_energy", "Energy"),
                MetricSpec("rejected_or_failed_count", "Failed"),
            ],
            x_labels,
            algorithms,
            lambda row: algorithm_label(row["algorithm"]),
        )
    )


def make_ablation_figure(written: list[Path]) -> None:
    rows = read_csv(RESULTS_DIR / "ablation" / "summary.csv")
    for row in rows:
        row["variant"] = row["scenario"]
        row["scenario"] = "dnn_30"
    x_labels = ["30"]
    written.append(
        make_multi_bar_figure(
            rows,
            OUTPUT_DIR / "fig_ablation.svg",
            "Ablation Study of DAREED Components",
            [
                MetricSpec("avg_operational_stability_score", "OSS"),
                MetricSpec("avg_total_energy", "Energy"),
                MetricSpec("rejected_or_failed_count", "Failed"),
            ],
            x_labels,
            VARIANT_ORDER,
            lambda row: variant_label(row["variant"]),
        )
    )


def make_scale_figure(written: list[Path]) -> None:
    rows = read_csv(RESULTS_DIR / "scale" / "summary.csv")
    x_labels = ["Small", "Medium", "Large"]
    written.append(
        make_multi_line_figure(
            rows,
            OUTPUT_DIR / "fig_scale.svg",
            "Scalability Evaluation under Different Network Scales",
            [
                MetricSpec("avg_estimated_delay", "Delay (ms)"),
                MetricSpec("avg_total_energy", "Energy"),
                MetricSpec("rejected_or_failed_count", "Failed"),
                MetricSpec("runtime_ms", "Runtime (ms)"),
            ],
            x_labels,
            ["NSGA-ED"],
        )
    )


def make_preference_figure(written: list[Path]) -> None:
    rows = read_csv(RESULTS_DIR / "preference" / "summary.csv")
    order = ["Delay-sensitive", "Stability-sensitive", "Energy-sensitive", "Balanced"]
    raw = {scenario_label(row["scenario"]): row for row in rows}
    metrics = [
        ("Delay utility", "avg_estimated_delay", False),
        ("OSS", "avg_operational_stability_score", True),
        ("Energy utility", "avg_total_energy", False),
    ]
    values = {label: [] for label in order}
    for _name, key, higher_better in metrics:
        column = [mean(raw[label], key) for label in order if label in raw]
        cmin, cmax = min(column), max(column)
        for label in order:
            value = mean(raw[label], key)
            if abs(cmax - cmin) < 1e-12:
                norm = 1.0
            elif higher_better:
                norm = (value - cmin) / (cmax - cmin)
            else:
                norm = (cmax - value) / (cmax - cmin)
            values[label].append(0.1 + 0.9 * norm)

    width, height = 760, 620
    cx, cy, radius = 360.0, 310.0, 205.0
    parts = [f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">Performance under Application Preferences</text>']
    axis_count = len(metrics)
    for ring in range(1, 6):
        r = radius * ring / 5
        coords = []
        for idx in range(axis_count):
            angle = -math.pi / 2 + idx * 2 * math.pi / axis_count
            coords.append(f"{cx + math.cos(angle) * r:.1f},{cy + math.sin(angle) * r:.1f}")
        parts.append(f'<polygon points="{" ".join(coords)}" fill="none" stroke="#e0e0e0"/>')
    for idx, (label, _key, _higher_better) in enumerate(metrics):
        angle = -math.pi / 2 + idx * 2 * math.pi / axis_count
        x2, y2 = cx + math.cos(angle) * radius, cy + math.sin(angle) * radius
        tx, ty = cx + math.cos(angle) * (radius + 38), cy + math.sin(angle) * (radius + 32)
        parts.append(f'<line class="axis" x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        parts.append(f'<text class="label" x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle">{escape(label)}</text>')
    for idx, label in enumerate(order):
        color = FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
        coords = []
        for midx, value in enumerate(values[label]):
            angle = -math.pi / 2 + midx * 2 * math.pi / axis_count
            coords.append(f"{cx + math.cos(angle) * radius * value:.1f},{cy + math.sin(angle) * radius * value:.1f}")
        parts.append(f'<polygon points="{" ".join(coords)}" fill="{color}" opacity="0.12" stroke="{color}" stroke-width="2"/>')
        yy = 90 + idx * 20
        parts.append(f'<rect x="600" y="{yy - 10}" width="12" height="12" fill="{color}" opacity="0.7"/>')
        parts.append(f'<text class="legend" x="618" y="{yy}">{escape(label)}</text>')
    write_svg(OUTPUT_DIR / "fig_preference.svg", width, height, "\n".join(parts))
    written.append(OUTPUT_DIR / "fig_preference.svg")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    make_sensitivity_figures(written)
    make_pareto_figures(written)
    make_overall_figures(written)
    make_dynamic_figure(written)
    make_ablation_figure(written)
    # make_scale_figure(written)
    # make_preference_figure(written)

    pdfs: list[Path] = []
    for svg in written:
        pdf = convert_to_pdf(svg)
        if pdf is not None:
            pdfs.append(pdf)

    manifest = {
        "results_dir": str(RESULTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "svg_count": len(written),
        "pdf_count": len(pdfs),
        "svg_files": [str(path.relative_to(ROOT)) for path in written],
        "pdf_files": [str(path.relative_to(ROOT)) for path in pdfs],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(written)} SVG figure(s) and {len(pdfs)} PDF figure(s) to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
