from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, List

from core.environment import Environment
from proposed import AllDNNRefactor


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _product(values: List[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _capture_stability_diagnostics(refactor: AllDNNRefactor) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    original_register: Callable[..., None] = refactor._register_running_dnn

    def wrapped_register(self: AllDNNRefactor, dnn_index: int, assignment: List[int], delay: float) -> None:
        assignment_slice = self._assignment_slice(dnn_index, assignment)
        used_nodes = self.env.collect_used_nodes(dnn_index, assignment_slice)
        used_links = self.env.collect_used_directed_links(dnn_index, assignment_slice)
        used_physical_links = self.env.collect_used_physical_links(
            dnn_index,
            assignment_slice,
        )
        node_stabilities = [
            self.env.nodes[node_idx].operational_stability for node_idx in used_nodes
        ]
        link_stabilities = [
            physical_link.transmission_stability for physical_link in used_physical_links
        ]
        node_loads = [self.env.nodes[node_idx].load_ratio for node_idx in used_nodes]
        node_heats = [self.env.nodes[node_idx].heat for node_idx in used_nodes]
        link_loads = [link.load_ratio for link in used_links]
        physical_link_loads = [
            physical_link.load_ratio for physical_link in used_physical_links
        ]
        link_heats = [
            physical_link.heat for physical_link in used_physical_links
        ]
        directed_pairs = {
            (
                self.env._node_index_by_id[id(link.s_node)],
                self.env._node_index_by_id[id(link.e_node)],
            )
            for link in used_links
        }
        reverse_pair_count = sum(
            1
            for start, end in directed_pairs
            if start < end and (end, start) in directed_pairs
        )
        node_product = _product(node_stabilities)
        link_product = _product(link_stabilities)
        operational_stability = self.env.count_operational_stability_by_assignment(
            dnn_index,
            assignment_slice,
        )
        raw_joint_product = self.env.count_raw_joint_product_by_assignment(
            dnn_index,
            assignment_slice,
        )
        rows.append(
            {
                "dnn_index": dnn_index,
                "task_count": len(self.env.ds[dnn_index].tasks),
                "deadline": self.env.ds[dnn_index].delay,
                "delay": delay,
                "used_node_count": len(used_nodes),
                "used_link_count": len(used_physical_links),
                "used_directed_link_count": len(used_links),
                "used_physical_link_count": len(used_physical_links),
                "reverse_pair_count": reverse_pair_count,
                "node_product": node_product,
                "link_product": link_product,
                "operational_stability_score": operational_stability,
                "raw_joint_product": raw_joint_product,
                "avg_node_stability": _mean(node_stabilities),
                "min_node_stability": min(node_stabilities) if node_stabilities else 0.0,
                "avg_link_stability": _mean(link_stabilities),
                "min_link_stability": min(link_stabilities) if link_stabilities else 0.0,
                "avg_node_load": _mean(node_loads),
                "max_node_load": max(node_loads) if node_loads else 0.0,
                "avg_link_load": _mean(link_loads),
                "max_link_load": max(link_loads) if link_loads else 0.0,
                "avg_physical_link_load": _mean(physical_link_loads),
                "max_physical_link_load": (
                    max(physical_link_loads) if physical_link_loads else 0.0
                ),
                "avg_node_heat": _mean(node_heats),
                "avg_link_heat": _mean(link_heats),
            }
        )
        original_register(dnn_index, assignment, delay)

    refactor._register_running_dnn = MethodType(wrapped_register, refactor)
    return rows


def _run_algorithm(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Any]:
    random.seed(args.seed)
    env = Environment(
        t_max=args.tmax,
        alpha_r=args.alpha_r,
        beta_r=args.beta_r,
        alpha_a=args.alpha_a,
        beta_a=args.beta_a,
        alpha_l=args.alpha_l,
        beta_l=args.beta_l,
        lambda_h=args.lambda_h,
        lambda_g=args.lambda_g,
        verbose=False,
    )
    refactor = AllDNNRefactor(env)
    refactor.pop_size = args.population_size
    refactor.iteration_limit = args.iteration_limit
    refactor.mutate_pm = args.mutation_probability
    refactor.cross_over_pm = args.crossover_probability
    diagnostics = _capture_stability_diagnostics(refactor)
    if args.algorithm == "proposed":
        metrics = refactor.run_proposed()
    elif args.algorithm == "customized":
        metrics = refactor.run_customized_proposed()
    else:
        raise ValueError(f"unsupported algorithm: {args.algorithm}")
    return diagnostics, metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict[str, Any]], metrics: Any) -> None:
    print("metrics")
    for key, value in metrics.to_report_rows():
        print(f"  {key}={value}")
    if not rows:
        print("diagnostics")
        print("  accepted_count=0")
        return
    print("diagnostics")
    print(f"  accepted_count={len(rows)}")
    for key in (
        "operational_stability_score",
        "raw_joint_product",
        "node_product",
        "link_product",
        "used_node_count",
        "used_link_count",
        "used_directed_link_count",
        "used_physical_link_count",
        "reverse_pair_count",
        "avg_link_stability",
        "min_link_stability",
        "avg_link_load",
        "max_link_load",
        "avg_physical_link_load",
        "max_physical_link_load",
        "avg_link_heat",
    ):
        values = [float(row[key]) for row in rows]
        print(f"  avg_{key}={_mean(values)}")
    weakest = sorted(rows, key=lambda row: row["operational_stability_score"])[:5]
    print("weakest_accepted")
    for row in weakest:
        print(
            "  "
            f"dnn={row['dnn_index']} "
            f"oss={row['operational_stability_score']:.6f} "
            f"raw_product={row['raw_joint_product']:.6f} "
            f"node_product={row['node_product']:.6f} "
            f"link_product={row['link_product']:.6f} "
            f"nodes={row['used_node_count']} "
            f"directed_links={row['used_directed_link_count']} "
            f"physical_links={row['used_physical_link_count']} "
            f"reverse_pairs={row['reverse_pair_count']} "
            f"avg_link_s={row['avg_link_stability']:.6f} "
            f"min_link_s={row['min_link_stability']:.6f} "
            f"max_link_load={row['max_link_load']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose operational stability score composition.")
    parser.add_argument("--algorithm", choices=["proposed", "customized"], default="customized")
    parser.add_argument("--tmax", type=int, default=20)
    parser.add_argument("--iteration-limit", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--mutation-probability", type=float, default=0.10)
    parser.add_argument("--crossover-probability", type=float, default=0.50)
    parser.add_argument("--alpha-r", type=float, default=0.80)
    parser.add_argument("--beta-r", type=float, default=0.50)
    parser.add_argument("--alpha-a", type=float, default=0.50)
    parser.add_argument("--beta-a", type=float, default=0.30)
    parser.add_argument("--alpha-l", type=float, default=0.70)
    parser.add_argument("--beta-l", type=float, default=0.40)
    parser.add_argument("--lambda-h", type=float, default=0.70)
    parser.add_argument("--lambda-g", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path("experiment_results/stability_diagnostics.csv"))
    args = parser.parse_args()

    rows, metrics = _run_algorithm(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output, rows)
    _print_summary(rows, metrics)
    print(f"csv={args.output}")


if __name__ == "__main__":
    main()
