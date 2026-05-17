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
        choices=["proposed", "random", "maxresource", "localfirst", "rtbl", "all"],
        default="all",
    )
    parser.add_argument("--tmax", type=int, default=40)
    parser.add_argument("--iteration-limit", type=int, default=120)
    args = parser.parse_args()

    if args.algorithm == "rtbl":
        metrics = RTBLRunner().run()
        print_metrics("RTBL", metrics)
        return

    env = Environment(t_max=args.tmax)
    refactor = AllDNNRefactor(env)
    refactor.iteration_limit = args.iteration_limit
    initial_nodes = env.clone_nodes()

    if args.algorithm == "proposed":
        print_metrics("PROPOSED", refactor.run_proposed())
        return
    if args.algorithm == "random":
        print_metrics("RANDOM", refactor.run_random(initial_nodes))
        return
    if args.algorithm == "maxresource":
        print_metrics("MAXRESOURCE", refactor.run_max_resource(initial_nodes))
        return
    if args.algorithm == "localfirst":
        print_metrics("LOCALFIRST", refactor.run_local_first(initial_nodes))
        return


    print_metrics("PROPOSED", refactor.run_proposed())
    env.reset_nodes(initial_nodes)
    print_metrics("RANDOM", refactor.run_random(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("MAXRESOURCE", refactor.run_max_resource(initial_nodes))
    env.reset_nodes(initial_nodes)
    print_metrics("LOCALFIRST", refactor.run_local_first(initial_nodes))
    print_metrics("RTBL", RTBLRunner().run())


if __name__ == "__main__":
    main()
