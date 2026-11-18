import argparse
import csv
from dataclasses import asdict, replace
import json
from math import pi, sqrt
import os
import platform
from pathlib import Path
import statistics

import numpy as np
import scipy

from src import PlannerConfig, RRT, RRTStar, load_environment
from src.utils import validate_path


def machine_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run_benchmark(output, config, seeds, maps, repeats=1):
    seeds = list(seeds)
    environments = [load_environment(name) for name in maps]
    if not seeds or not environments:
        raise ValueError("Choose at least one seed and one map")
    for environment in environments:
        environment.validate_query(config.robot_radius)

    output = Path(output)
    filenames = ("runs.csv", "convergence.csv", "paths.jsonl", "settings.json")
    if any((output / name).exists() for name in filenames):
        raise FileExistsError(f"Benchmark already exists in {output}; choose another output folder")
    output.mkdir(parents=True, exist_ok=True)

    settings = {
        "config": asdict(config),
        "seeds": seeds,
        "timing_repeats": repeats,
        "machine": machine_info(),
        "effective_gamma": {
            environment.name: config.neighbor_gamma or 2.2 * sqrt(1.5 * environment.area / pi)
            for environment in environments
        },
        "maps": [environment.to_dict() for environment in environments],
    }
    (output / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    # Warm up both algorithms before collecting timings.
    warmup_config = replace(config, max_iterations=150)
    for algorithm in (RRT, RRTStar):
        algorithm(environments[0], warmup_config, seed=0).run()

    run_fields = [
        "environment", "algorithm", "seed", "success", "path_length",
        "planning_time", "time_to_first_solution", "first_solution_iteration",
        "iterations", "number_of_nodes", "rewires",
    ]
    history_fields = [
        "environment", "algorithm", "seed", "iteration", "nodes", "elapsed", "best_cost",
    ]
    completed = 0
    total = len(environments) * len(seeds) * 2

    with (
        (output / "runs.csv").open("w", newline="", encoding="utf-8") as run_file,
        (output / "convergence.csv").open("w", newline="", encoding="utf-8") as history_file,
        (output / "paths.jsonl").open("w", encoding="utf-8") as path_file,
    ):
        run_writer = csv.DictWriter(run_file, fieldnames=run_fields)
        history_writer = csv.DictWriter(history_file, fieldnames=history_fields)
        run_writer.writeheader()
        history_writer.writeheader()

        for environment in environments:
            for seed in seeds:
                # Alternate which algorithm runs first.
                algorithms = (RRT, RRTStar) if seed % 2 == 0 else (RRTStar, RRT)
                for algorithm in algorithms:
                    planner = algorithm(environment, config, seed)
                    result = planner.run()
                    if result.success and not validate_path(
                        result.path, environment, config.robot_radius
                    ):
                        raise ValueError(f"Invalid path: {environment.name}, {algorithm.name}, seed {seed}")

                    if repeats > 1:
                        # Kết quả hình học là tất định; chỉ thời gian chạy dao động, nên lấy trung vị.
                        again = [algorithm(environment, config, seed).run() for _ in range(repeats - 1)]
                        runs = [result, *again]
                        result.planning_time = statistics.median(r.planning_time for r in runs)
                        firsts = [r.time_to_first_solution for r in runs if r.time_to_first_solution is not None]
                        if firsts:
                            result.time_to_first_solution = statistics.median(firsts)

                    metrics = asdict(result)
                    metrics.pop("path")
                    run_writer.writerow({"environment": environment.name, **metrics})

                    for step in planner.history:
                        history_writer.writerow({
                            "environment": environment.name,
                            "algorithm": algorithm.name,
                            "seed": seed,
                            "iteration": step.iteration,
                            "nodes": step.nodes,
                            "elapsed": step.elapsed,
                            "best_cost": step.best_cost,
                        })

                    path_file.write(json.dumps({
                        "environment": environment.name,
                        "algorithm": algorithm.name,
                        "seed": seed,
                        "success": result.success,
                        "path": result.path.tolist() if result.path is not None else None,
                    }) + "\n")

                    run_file.flush()
                    history_file.flush()
                    path_file.flush()
                    completed += 1
                    length = "no path" if result.path_length is None else f"{result.path_length:.3f} m"
                    print(
                        f"[{completed}/{total}] {environment.name} | {algorithm.name} | "
                        f"seed {seed} | {length} | {result.planning_time:.3f} s",
                        flush=True,
                    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare RRT and RRT* over several random seeds")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds, starting at zero")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--maps", nargs="+", default=["simple", "complex", "narrow"])
    parser.add_argument("--step-size", type=float, default=0.8)
    parser.add_argument("--goal-bias", type=float, default=0.05)
    parser.add_argument("--robot-radius", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=1, help="Số lần đo lặp mỗi (bản đồ, thuật toán, seed); báo cáo trung vị thời gian")
    parser.add_argument("--protocol", choices=["fixed", "first"], default="fixed")
    parser.add_argument("--charts-only", action="store_true", help="Skip report map and tree snapshots")
    args = parser.parse_args()
    if args.seeds < 1 or args.repeats < 1:
        parser.error("--seeds and --repeats must be positive")

    config = PlannerConfig(
        step_size=args.step_size,
        goal_sample_rate=args.goal_bias,
        max_iterations=args.iterations,
        robot_radius=args.robot_radius,
        stop_on_first=args.protocol == "first",
    )
    output = run_benchmark(args.output, config, range(args.seeds), args.maps, args.repeats)
    from .plots import generate
    generate(output, snapshots=not args.charts_only)


if __name__ == "__main__":
    main()
