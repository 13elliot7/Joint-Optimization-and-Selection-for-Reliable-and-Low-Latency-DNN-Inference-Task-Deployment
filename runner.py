from __future__ import annotations

import argparse

from core.environment import Environment
from proposed import AllDNNRefactor
from rtbl.scheduler import RTBLRunner


def print_metrics(name: str, metrics) -> None:
    """按动态模型口径打印实验指标。"""
    print(name)
    for key, value in metrics.to_report_rows():
        print(f"  {key}={value}")


def main() -> None:
    """解析命令行参数并运行指定算法。"""
    parser = argparse.ArgumentParser(description="Python refactor for src_backup_origin")
    parser.add_argument(
        "--algorithm",
        choices=["proposed", "customized", "random", "maxresource", "maxresource_fast", "localfirst", "rtbl", "sa", "all"],
        default="all",
    )
    parser.add_argument("--tmax", type=int, default=40)
    parser.add_argument("--iteration-limit", type=int, default=200)
    parser.add_argument("--quiet", action="store_true", help="只输出最终实验指标")
    args = parser.parse_args()

    env = Environment(t_max=args.tmax, verbose=not args.quiet)
    refactor = AllDNNRefactor(env)
    refactor.iteration_limit = args.iteration_limit
    initial_nodes = env.clone_nodes()

    if args.algorithm == "proposed":
        print_metrics("PROPOSED", refactor.run_proposed())
        return
    if args.algorithm == "customized":
        print_metrics("CUSTOMIZED", refactor.run_customized_proposed())
        return
    if args.algorithm == "random":
        print_metrics("RANDOM", refactor.run_random(initial_nodes))
        return
    if args.algorithm == "maxresource":
        print_metrics("MAXRESOURCE", refactor.run_max_resource(initial_nodes))
        return
    if args.algorithm == "maxresource_fast":
        print_metrics("MAXRESOURCE_FAST", refactor.run_max_resource_fast(initial_nodes))
        return
    if args.algorithm == "localfirst":
        print_metrics("LOCALFIRST", refactor.run_local_first(initial_nodes))
        return
    if args.algorithm == "rtbl":
        print_metrics("RTBL", RTBLRunner(env).run_dynamic(initial_nodes))
        return
    if args.algorithm == "sa":
        print_metrics("SA", refactor.run_simulated_annealing(initial_nodes))
        return


    print_metrics("PROPOSED", refactor.run_proposed())
    env.reset_nodes(initial_nodes)
    print_metrics("CUSTOMIZED", refactor.run_customized_proposed())
    env.reset_nodes(initial_nodes)
    print_metrics("RANDOM", refactor.run_random(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("MAXRESOURCE", refactor.run_max_resource(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("MAXRESOURCE_FAST", refactor.run_max_resource_fast(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("LOCALFIRST", refactor.run_local_first(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("RTBL", RTBLRunner(env).run_dynamic(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("SA", refactor.run_simulated_annealing(initial_nodes))


if __name__ == "__main__":
    main()
