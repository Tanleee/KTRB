"""Các thực nghiệm bổ sung cho báo cáo (độ phức tạp, bộ nhớ, baseline, độ nhạy...).

Cách dùng (chạy từ thư mục gốc dự án):
    python -m experiments.extra radius
    python -m experiments.extra scaling      --input results/benchmark
    python -m experiments.extra first-solution --input results/benchmark
    python -m experiments.extra reference    --input results/benchmark
    python -m experiments.extra memory       --seeds 5
    python -m experiments.extra equal-time   --budget 1.0 --seeds 20
    python -m experiments.extra sensitivity  --seeds 10
Mỗi lệnh ghi CSV (và hình nếu có) vào --output (mặc định results/extra).
"""
import argparse
import csv
from collections import defaultdict
from dataclasses import replace
import heapq
from math import cos, log, pi, sin, sqrt
from pathlib import Path
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import PlannerConfig, RRT, RRTStar, load_environment
from src.collision import Circle, Rectangle

MAPS = ("simple", "complex", "narrow")
ALGORITHMS = (RRT, RRTStar)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {path}")


def mean_std(values):
    values = [v for v in values if v is not None]
    return (
        float(np.mean(values)) if values else None,
        float(np.std(values, ddof=1)) if len(values) > 1 else None,
    )


def read_runs(directory):
    with (Path(directory) / "runs.csv").open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_convergence(directory):
    """Trả về {(môi trường, thuật toán, seed): (elapsed[], best_cost[])} theo vòng lặp."""
    data = defaultdict(lambda: ([], []))
    with (Path(directory) / "convergence.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["environment"], row["algorithm"], int(row["seed"]))
            data[key][0].append(float(row["elapsed"]))
            data[key][1].append(float(row["best_cost"]) if row["best_cost"] else np.nan)
    return {key: (np.array(a), np.array(b)) for key, (a, b) in data.items()}


# ---------------------------------------------------------------- radius
def cmd_radius(args):
    """Bán kính lân cận r_n thực tế và ngưỡng γ lý thuyết (công thức 9 và 15 trong báo cáo)."""
    rows = []
    for name in args.maps:
        env = load_environment(name)
        gamma = 2.2 * sqrt(1.5 * env.area / pi)
        gamma_min = 2 * sqrt(1.5) * sqrt(env.area / pi)  # dùng μ(C_free) ≤ diện tích bản đồ
        # Từ n nào r_n < η? (chỉ xét n sau đỉnh của log n / n)
        n = 3
        while min(args.step_size, gamma * sqrt(log(n) / n)) >= args.step_size and n < 10**6:
            n += 1
        row = {
            "environment": name,
            "gamma_used": gamma,
            "gamma_threshold_upper_bound": gamma_min,
            "n_where_radius_drops_below_eta": n,
        }
        for count in (100, 500, 1000, 2500, 5000, 10000):
            row[f"r_n_at_{count}"] = min(args.step_size, gamma * sqrt(log(count) / count))
        rows.append(row)
        print(f"{name}: γ={gamma:.2f}, r_n < η={args.step_size} khi n ≥ {n}; r_2500={row['r_n_at_2500']:.3f}")
    write_csv(Path(args.output) / "radius.csv", rows)


# ---------------------------------------------------------------- scaling
def cmd_scaling(args):
    """Đường cong thời gian tích lũy theo vòng lặp và số mũ thực nghiệm b trong t ≈ a·n^b."""
    data = read_convergence(args.input)
    groups = defaultdict(list)
    for (env, algorithm, _), (elapsed, _) in data.items():
        groups[(env, algorithm)].append(elapsed)

    rows = []
    fig, axes = plt.subplots(1, len(args.maps), figsize=(4.2 * len(args.maps), 3.6), squeeze=False)
    for ax, env in zip(axes[0], args.maps):
        for algorithm in ("RRT", "RRT*"):
            mean_elapsed = np.mean(np.vstack(groups[(env, algorithm)]), axis=0)
            iterations = np.arange(len(mean_elapsed))
            mask = (iterations >= args.min_iteration) & (mean_elapsed > 0)
            slope, intercept = np.polyfit(np.log(iterations[mask]), np.log(mean_elapsed[mask]), 1)
            rows.append({
                "environment": env, "algorithm": algorithm,
                "loglog_exponent_b": slope,
                "time_at_500": mean_elapsed[500], "time_at_1000": mean_elapsed[1000],
                "time_at_2500": mean_elapsed[-1],
            })
            ax.loglog(iterations[1:], mean_elapsed[1:], label=f"{algorithm} (b≈{slope:.2f})".replace(".", ","))
        ax.set_title(env.capitalize())
        ax.set_xlabel("Vòng lặp")
        ax.set_ylabel("Thời gian tích lũy (s)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "scaling.png", dpi=180)
    write_csv(out / "scaling.csv", rows)


# ---------------------------------------------------------------- first-solution
def cmd_first_solution(args):
    """RRT chạy hết ngân sách có lợi hơn RRT dừng ở lời giải đầu bao nhiêu? RRT* còn giảm được bao nhiêu so với RRT chuẩn?"""
    data = read_convergence(args.input)
    rows = []
    for env in args.maps:
        keep_improve, vs_standard, vs_full = [], [], []
        for (e, algorithm, seed), (_, costs) in data.items():
            if e != env or algorithm != "RRT":
                continue
            valid = np.flatnonzero(~np.isnan(costs))
            star = data.get((env, "RRT*", seed))
            if len(valid) == 0 or star is None or np.isnan(star[1][-1]):
                continue
            first, final = costs[valid[0]], costs[-1]
            keep_improve.append(100 * (first - final) / first)
            vs_standard.append(100 * (first - star[1][-1]) / first)
            vs_full.append(100 * (final - star[1][-1]) / final)
        rows.append({
            "environment": env, "paired_runs": len(keep_improve),
            "rrt_gain_from_continuing_percent": mean_std(keep_improve)[0],
            "rrt_star_vs_rrt_first_solution_percent": mean_std(vs_standard)[0],
            "rrt_star_vs_rrt_full_budget_percent": mean_std(vs_full)[0],
        })
    write_csv(Path(args.output) / "first_solution.csv", rows)
    for row in rows:
        print(row)


# ---------------------------------------------------------------- reference (visibility graph)
def reference_points(env, radius, margin=1e-6, arc_points=16, ring_points=96):
    """Điểm ứng viên bám quanh vật cản đã nở r: đường ngắn nhất chỉ uốn tại các điểm này."""
    pts = [tuple(env.start), tuple(env.goal)]
    for ob in env.obstacles:
        if isinstance(ob, Circle):
            ring = (ob.radius + radius + margin) / cos(pi / ring_points)
            pts += [(ob.x + ring * cos(2 * pi * k / ring_points), ob.y + ring * sin(2 * pi * k / ring_points))
                    for k in range(ring_points)]
        else:
            x0, y0, x1, y1 = ob.x, ob.y, ob.x + ob.width, ob.y + ob.height
            step = (pi / 2) / arc_points
            ring = (radius + margin) / cos(step / 2)
            for (cx, cy), start_angle in (((x1, y1), 0), ((x0, y1), pi / 2), ((x0, y0), pi), ((x1, y0), 1.5 * pi)):
                for k in range(arc_points + 1):
                    a = start_angle + k * step
                    pts.append((cx + ring * cos(a), cy + ring * sin(a)))
    return [p for p in pts if env.point_free(p, radius)]


def reference_length(env, radius):
    """Chiều dài đường ngắn nhất xấp xỉ (đồ thị tầm nhìn + tìm đường ngắn nhất trên đồ thị), làm mốc để tính optimality gap."""
    pts = reference_points(env, radius)
    goal = pts[1]
    best = {0: 0.0}
    heap = [(float(np.hypot(pts[0][0] - goal[0], pts[0][1] - goal[1])), 0.0, 0)]
    closed = set()
    while heap:
        _, g, i = heapq.heappop(heap)
        if i in closed:
            continue
        if i == 1:
            return g
        closed.add(i)
        for j, q in enumerate(pts):
            if j in closed or j == i:
                continue
            d = float(np.hypot(pts[i][0] - q[0], pts[i][1] - q[1]))
            if g + d >= best.get(j, float("inf")):
                continue
            if env.edge_free(pts[i], q, radius):
                best[j] = g + d
                heapq.heappush(heap, (g + d + float(np.hypot(q[0] - goal[0], q[1] - goal[1])), g + d, j))
    return None


def cmd_reference(args):
    refs = {name: reference_length(load_environment(name), args.robot_radius) for name in args.maps}
    runs = read_runs(args.input)
    rows = []
    for name in args.maps:
        row = {"environment": name, "reference_length": refs[name]}
        for algorithm in ("RRT", "RRT*"):
            lengths = [float(r["path_length"]) for r in runs
                       if r["environment"] == name and r["algorithm"] == algorithm and r["success"] == "True"]
            gaps = [100 * (L - refs[name]) / refs[name] for L in lengths]
            mean, std = mean_std(gaps)
            key = algorithm.replace("*", "_star")
            row[f"{key}_gap_mean_percent"], row[f"{key}_gap_std_percent"] = mean, std
        rows.append(row)
        print(row)
    write_csv(Path(args.output) / "optimality_gap.csv", rows)


# ---------------------------------------------------------------- memory
def cmd_memory(args):
    rows = []
    for name in args.maps:
        env = load_environment(name)
        config = PlannerConfig(max_iterations=args.iterations, robot_radius=args.robot_radius)
        for seed in range(args.seeds):
            for algorithm in ALGORITHMS:
                tracemalloc.start()
                result = algorithm(env, config, seed).run()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append({
                    "environment": name, "algorithm": algorithm.name, "seed": seed,
                    "peak_kib": peak / 1024, "nodes": result.number_of_nodes,
                    "bytes_per_node": peak / max(result.number_of_nodes, 1),
                })
    write_csv(Path(args.output) / "memory_runs.csv", rows)
    summary = []
    for name in args.maps:
        for algorithm in ALGORITHMS:
            sel = [r for r in rows if r["environment"] == name and r["algorithm"] == algorithm.name]
            mean, std = mean_std([r["peak_kib"] for r in sel])
            summary.append({"environment": name, "algorithm": algorithm.name,
                            "peak_kib_mean": mean, "peak_kib_std": std,
                            "bytes_per_node_mean": mean_std([r["bytes_per_node"] for r in sel])[0]})
    write_csv(Path(args.output) / "memory_summary.csv", summary)
    for row in summary:
        print(row)


# ---------------------------------------------------------------- equal time
def run_for_time(algorithm, env, config, seed, budget):
    planner = algorithm(env, config, seed)
    while planner.elapsed < budget and planner.iteration < config.max_iterations:
        planner.step()
    return planner.result()


def cmd_equal_time(args):
    rows = []
    for name in args.maps:
        env = load_environment(name)
        config = PlannerConfig(max_iterations=200000, robot_radius=args.robot_radius)
        for seed in range(args.seeds):
            for algorithm in ALGORITHMS:
                r = run_for_time(algorithm, env, config, seed, args.budget)
                rows.append({"environment": name, "algorithm": algorithm.name, "seed": seed,
                             "success": r.success, "path_length": r.path_length,
                             "iterations": r.iterations, "nodes": r.number_of_nodes, "time": r.planning_time})
    write_csv(Path(args.output) / "equal_time_runs.csv", rows)
    summary = []
    for name in args.maps:
        for algorithm in ALGORITHMS:
            sel = [r for r in rows if r["environment"] == name and r["algorithm"] == algorithm.name]
            mean, std = mean_std([r["path_length"] for r in sel])
            summary.append({"environment": name, "algorithm": algorithm.name, "budget_s": args.budget,
                            "successes": sum(r["success"] for r in sel), "runs": len(sel),
                            "path_length_mean": mean, "path_length_std": std,
                            "iterations_mean": mean_std([r["iterations"] for r in sel])[0]})
    write_csv(Path(args.output) / "equal_time_summary.csv", summary)
    for row in summary:
        print(row)


# ---------------------------------------------------------------- sensitivity
def cmd_sensitivity(args):
    base = dict(step_size=0.8, goal_sample_rate=0.05, neighbor_gamma=None)
    settings = [("baseline", {})]
    settings += [("step_size", {"step_size": v}) for v in (0.4, 1.2)]
    settings += [("goal_sample_rate", {"goal_sample_rate": v}) for v in (0.0, 0.2)]
    settings += [("neighbor_gamma", {"neighbor_gamma": v}) for v in (6.0, 3.0)]
    rows = []
    for name in args.maps:
        env = load_environment(name)
        for factor, change in settings:
            config = PlannerConfig(max_iterations=args.iterations, robot_radius=args.robot_radius, **{**base, **change})
            for algorithm in ALGORITHMS:
                if factor == "neighbor_gamma" and algorithm is RRT:
                    continue  # γ chỉ ảnh hưởng RRT*
                results = [algorithm(env, config, seed).run() for seed in range(args.seeds)]
                ok = [r for r in results if r.success]
                rows.append({
                    "environment": name, "algorithm": algorithm.name, "factor": factor,
                    "value": next(iter(change.values()), "default"),
                    "success": f"{len(ok)}/{len(results)}",
                    "path_length_mean": mean_std([r.path_length for r in ok])[0],
                    "path_length_std": mean_std([r.path_length for r in ok])[1],
                    "time_mean": mean_std([r.planning_time for r in results])[0],
                    "rewires_mean": mean_std([r.rewires for r in results])[0],
                })
                print(rows[-1], flush=True)
    write_csv(Path(args.output) / "sensitivity.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("radius", "scaling", "first-solution", "reference",
                                            "memory", "equal-time", "sensitivity"))
    parser.add_argument("--input", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("results/extra"))
    parser.add_argument("--maps", nargs="+", default=list(MAPS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--step-size", type=float, default=0.8)
    parser.add_argument("--robot-radius", type=float, default=0.2)
    parser.add_argument("--budget", type=float, default=1.0, help="Ngân sách thời gian (giây) cho equal-time")
    parser.add_argument("--min-iteration", type=int, default=250, help="Chỉ khớp log-log từ vòng lặp này")
    args = parser.parse_args()
    {
        "radius": cmd_radius, "scaling": cmd_scaling, "first-solution": cmd_first_solution,
        "reference": cmd_reference, "memory": cmd_memory, "equal-time": cmd_equal_time,
        "sensitivity": cmd_sensitivity,
    }[args.command](args)


if __name__ == "__main__":
    main()
