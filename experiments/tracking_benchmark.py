"""Thực nghiệm lập kế hoạch -> làm mượt -> bám đường (Pure Pursuit) cho robot 2 bánh vi sai.

Với mỗi bản đồ, thuật toán (RRT, RRT*), seed, biên an toàn (margin),
kiểu đường (raw / smooth) và khoảng nhìn trước L: lập kế hoạch với bán kính robot + margin,
rồi mô phỏng bám đường và kiểm tra va chạm với bán kính robot thật.

    python -m experiments.tracking_benchmark --output results/tracking
"""

import argparse
import csv
from pathlib import Path
import statistics

from src import PlannerConfig, RRT, RRTStar, load_environment
from src.smoothing import shortcut_path
from src.tracking import PurePursuitConfig, track_path
from src.utils import path_length, validate_path

FIELDS = [
    "environment", "algorithm", "seed", "margin", "variant", "lookahead", "path_length",
    "reached", "collided", "success", "time", "traveled_length", "rms_cte", "max_cte",
]


def plan(algorithm, environment, plan_radius, seed, args):
    cls = RRT if algorithm == "RRT" else RRTStar
    config = PlannerConfig(max_iterations=args.iterations, robot_radius=plan_radius)
    return cls(environment, config, seed).run()


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["environment"], row["algorithm"], row["margin"], row["variant"], row["lookahead"])
        groups.setdefault(key, []).append(row)
    print(f"\n{'map':8s} {'alg':4s} {'margin':>6s} {'variant':7s} {'L':>4s} | "
          f"{'plans':>5s} {'success':>7s} {'median T':>9s} {'median RMS':>11s} {'median max':>11s}")
    for key in sorted(groups):
        group = groups[key]
        ok = [r for r in group if r["success"]]
        def med(field):
            return f"{statistics.median(r[field] for r in ok):.3f}" if ok else "-"
        env, alg, margin, variant, look = key
        print(f"{env:8s} {alg:4s} {margin:6g} {variant:7s} {look:4g} | {len(group):5d} "
              f"{len(ok):3d}/{len(group):<3d} {med('time'):>9s} {med('rms_cte'):>11s} {med('max_cte'):>11s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/tracking"))
    parser.add_argument("--maps", nargs="+", default=["simple", "complex", "narrow"])
    parser.add_argument("--algorithms", nargs="+", choices=("RRT", "RRT*"), default=["RRT", "RRT*"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--robot-radius", type=float, default=0.2)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--lookaheads", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    parser.add_argument("--v-max", type=float, default=0.5)
    parser.add_argument("--omega-max", type=float, default=1.5)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.maps:
        environment = load_environment(name)
        for algorithm in args.algorithms:
            seeds = range(args.seeds)
            for margin in args.margins:
                plan_radius = args.robot_radius + margin
                for seed in seeds:
                    result = plan(algorithm, environment, plan_radius, seed, args)
                    if not result.success:
                        continue
                    assert validate_path(result.path, environment, plan_radius)
                    variants = {"raw": result.path,
                                "smooth": shortcut_path(result.path, environment, plan_radius)}
                    for variant, reference in variants.items():
                        for lookahead in args.lookaheads:
                            tracking = PurePursuitConfig(
                                lookahead=lookahead, v_max=args.v_max, omega_max=args.omega_max,
                                robot_radius=args.robot_radius)
                            outcome = track_path(reference, environment, tracking)
                            rows.append({
                                "environment": name, "algorithm": algorithm, "seed": seed,
                                "margin": margin, "variant": variant, "lookahead": lookahead,
                                "path_length": path_length(reference),
                                "reached": outcome.reached, "collided": outcome.collided,
                                "success": outcome.success, "time": outcome.time,
                                "traveled_length": outcome.traveled_length,
                                "rms_cte": outcome.rms_cte, "max_cte": outcome.max_cte,
                            })
        print(f"done {name}", flush=True)

    with (args.output / "tracking_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output / 'tracking_runs.csv'} ({len(rows)} rows)")
    summarize(rows)


if __name__ == "__main__":
    main()
