import argparse
import csv
from collections import defaultdict
import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import PermutationMethod, wilcoxon

ALGORITHMS = ("RRT", "RRT*")
COLORS = ("#3779ad", "#168b82")


def save_figure(fig, destination):
    for ax in fig.axes:
        if ax.get_xlabel() != "Bản đồ":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
METRICS = (
    "planning_time", "path_length", "time_to_first_solution",
    "number_of_nodes", "first_solution_iteration", "rewires",
)


def read_runs(directory):
    with (Path(directory) / "runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["success"] = row["success"] == "True"
        for metric in METRICS:
            row[metric] = float(row[metric]) if row[metric] else None
    return rows


def mean_std(values):
    values = [value for value in values if value is not None]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None, None
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return max(0, center - margin), min(1, center + margin)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["environment"], row["algorithm"])].append(row)

    summary = []
    for (environment, algorithm), runs in sorted(groups.items()):
        successes = sum(run["success"] for run in runs)
        low, high = wilson(successes, len(runs))
        row = {
            "environment": environment,
            "algorithm": algorithm,
            "runs": len(runs),
            "successes": successes,
            "success_rate": successes / len(runs),
            "success_ci_low": low,
            "success_ci_high": high,
        }
        for metric in METRICS:
            stats = mean_std([run[metric] for run in runs])
            for field, value in stats.items():
                row[f"{metric}_{field}"] = value
        summary.append(row)
    return summary


def wilcoxon_statistics(differences):
    # Round subtraction noise before ranking; differences are measured in metres.
    values = np.round(np.asarray(differences, dtype=float), decimals=9)
    if not np.isfinite(values).all():
        raise ValueError("Wilcoxon differences must be finite")
    nonzero = values[values != 0]
    result = {
        "paired_successes": len(values),
        "nonzero_pairs": len(nonzero),
        "zero_pairs": len(values) - len(nonzero),
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "difference_rounding_decimals": 9,
        "statistic": None,
        "p_value": None,
        "method": "no_successful_pairs",
    }
    if not len(values):
        return result
    if not len(nonzero):
        # No signed ranks remain: explicitly label this degenerate case.
        result.update(statistic=0.0, p_value=1.0, method="all_differences_zero")
        return result
    ties = len(np.unique(np.abs(nonzero))) != len(nonzero)
    if len(nonzero) <= 50 and not ties:
        method = "exact"
        result["method"] = "exact"
    elif len(nonzero) <= 50:
        method = PermutationMethod(n_resamples=99999, rng=0)
        result.update(method="permutation", max_resamples=99999, random_seed=0)
    else:
        method = "asymptotic"
        result["method"] = "asymptotic"
    # Dropping zeros explicitly implements the Wilcox zero convention.
    calculation = wilcoxon(nonzero, alternative="two-sided", method=method)
    result.update(statistic=float(calculation.statistic), p_value=float(calculation.pvalue))
    return result


def paired_comparison(rows):
    lookup = {(row["environment"], row["algorithm"], row["seed"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate environment/algorithm/seed in benchmark results")
    comparisons = []
    for environment in sorted({row["environment"] for row in rows}):
        improvements = []
        differences = []
        time_ratios = []
        seeds = sorted({row["seed"] for row in rows if row["environment"] == environment})
        for seed in seeds:
            rrt = lookup.get((environment, "RRT", seed))
            rrt_star = lookup.get((environment, "RRT*", seed))
            if rrt is None or rrt_star is None:
                continue
            if rrt["planning_time"] > 0:
                time_ratios.append(rrt_star["planning_time"] / rrt["planning_time"])
            if rrt["success"] and rrt_star["success"]:
                difference = rrt["path_length"] - rrt_star["path_length"]
                differences.append(difference)
                if rrt["path_length"] > 0:
                    improvements.append(100 * difference / rrt["path_length"])

        comparisons.append({
            "environment": environment,
            "paired_successes": len(differences),
            "length_reduction_percent": mean_std(improvements),
            "length_difference_m": mean_std(differences),
            "compute_time_ratio": mean_std(time_ratios),
            "rrt_star_strictly_better": sum(difference > 1e-9 for difference in differences),
            "wilcoxon": wilcoxon_statistics(differences),
        })
    return comparisons


def metric_figures(rows, summary, directory):
    maps = list(dict.fromkeys(row["environment"] for row in rows))
    lookup = {(row["environment"], row["algorithm"]): row for row in summary}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    metrics = [
        ("planning_time", "Thời gian lập kế hoạch (s)"),
        ("path_length", "Độ dài đường (m)"),
        ("time_to_first_solution", "Thời gian tìm lời giải đầu (s)"),
        ("first_solution_iteration", "Vòng lặp tìm lời giải đầu"),
    ]
    positions = np.arange(len(maps))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for index, algorithm in enumerate(ALGORITHMS):
            values = []
            errors = []
            for environment in maps:
                stats = lookup[(environment, algorithm)]
                mean = stats[f"{metric}_mean"]
                values.append(np.nan if mean is None else mean)
                errors.append(stats[f"{metric}_std"] or 0)

            bar_positions = positions + (index - 0.5) * 0.36
            error_bars = np.array([np.minimum(np.nan_to_num(values), errors), errors])
            ax.bar(
                bar_positions, values, width=0.36, color=COLORS[index], label=algorithm,
                yerr=error_bars, capsize=3, error_kw={"linewidth": 1},
            )
        ax.set_xticks(positions, [name.title() for name in maps])
        ax.set_xlabel("Bản đồ")
        ax.set_ylim(bottom=0)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0, 0].legend()
    fig.suptitle("So sánh RRT và RRT* trên các bản đồ", fontsize=16)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, directory / "comparison.png")


def convergence_figures(directory, rows):
    histories = defaultdict(list)
    with (directory / "convergence.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["environment"], row["algorithm"], int(row["seed"]))
            cost = float(row["best_cost"]) if row["best_cost"] else np.nan
            histories[key].append((int(row["iteration"]), cost))

    maps = list(dict.fromkeys(row["environment"] for row in rows))
    fig, axes = plt.subplots(2, len(maps), figsize=(5 * len(maps), 8), squeeze=False)
    trace_fig, trace_axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4), squeeze=False)
    for column, environment in enumerate(maps):
        max_iteration = max(
            iteration
            for (map_name, _, _), history in histories.items() if map_name == environment
            for iteration, _ in history
        )
        iterations = np.arange(max_iteration + 1)
        trace_has_solution = False
        for index, algorithm in enumerate(ALGORITHMS):
            runs = {
                seed: history
                for (map_name, planner_name, seed), history in histories.items()
                if map_name == environment and planner_name == algorithm
            }
            costs = np.full((len(runs), len(iterations)), np.nan)
            for run_index, history in enumerate(runs.values()):
                for iteration, cost in history:
                    costs[run_index, iteration] = cost
                # Keep the final cost after a run stops at its first solution.
                last_iteration, last_cost = history[-1]
                if np.isfinite(last_cost):
                    costs[run_index, last_iteration:] = last_cost

            solved = np.isfinite(costs).sum(axis=0)
            has_solution = solved > 0
            median = np.full(len(iterations), np.nan)
            lower_quartile = median.copy()
            upper_quartile = median.copy()
            if has_solution.any():
                quartiles = np.nanquantile(costs[:, has_solution], [0.25, 0.5, 0.75], axis=0)
                lower_quartile[has_solution], median[has_solution], upper_quartile[has_solution] = quartiles

            color = COLORS[index]
            axes[0, column].plot(iterations, median, color=color, label=algorithm)
            axes[0, column].fill_between(
                iterations, lower_quartile, upper_quartile, color=color, alpha=0.17
            )
            axes[1, column].plot(
                iterations, solved / len(runs) * 100, color=color, label=algorithm,
                linestyle="-" if index == 0 else "--",
            )

            seed = min(runs)
            history = runs[seed]
            trace_has_solution |= any(np.isfinite(cost) for _, cost in history)
            trace_axes[0, column].plot(
                [iteration for iteration, _ in history],
                [cost for _, cost in history],
                label=f"{algorithm}, seed {seed}", color=color,
            )

        axes[0, column].set_title(f"Bản đồ {environment.title()}")
        axes[0, column].set_ylabel("Độ dài đường (m)")
        axes[0, column].set_xlabel("Vòng lặp")
        if not trace_has_solution:
            trace_axes[0, column].set_yticks([])
            trace_axes[0, column].text(0.5, 0.5, "Chưa tìm được đường", ha="center", transform=trace_axes[0, column].transAxes)
        axes[1, column].set(xlabel="Vòng lặp", ylabel="Tỷ lệ tìm được đường (%)", ylim=(-3, 103))
        trace_axes[0, column].set(title=f"Bản đồ {environment.title()}", xlabel="Vòng lặp",
                                  ylabel="Độ dài đường (m)", xlim=(0, max(1, max_iteration)))
        trace_axes[0, column].legend()
        trace_axes[0, column].grid(alpha=0.2)
        for ax in axes[:, column]:
            ax.legend()
            ax.grid(alpha=0.2)

    fig.suptitle("Hội tụ chi phí và tỷ lệ tìm được đường qua các vòng lặp", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(fig, directory / "convergence.png")
    trace_fig.suptitle("Diễn biến chi phí đường đi theo seed")
    trace_fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(trace_fig, directory / "convergence_example.png")


def report_snapshots(directory, rows):
    from src import Environment, PlannerConfig, RRT, RRTStar
    from src.collision import Circle, Rectangle
    from src.visualization import PlannerView, draw_environment, status_text

    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    config = PlannerConfig(**settings["config"])
    environments = []
    for data in settings["maps"]:
        obstacles = []
        for obstacle in data["obstacles"]:
            values = dict(obstacle)
            kind = values.pop("type")
            if kind not in ("circle", "rectangle"):
                raise ValueError(f"Unknown obstacle type: {kind}")
            obstacles.append((Circle if kind == "circle" else Rectangle)(**values))
        environments.append(Environment(data["name"], tuple(data["bounds"]),
                                        tuple(data["start"]), tuple(data["goal"]), tuple(obstacles)))

    fig, axes = plt.subplots(1, len(environments), figsize=(5 * len(environments), 5), squeeze=False)
    for ax, env in zip(axes[0], environments):
        draw_environment(ax, env, config.robot_radius)
        ax.set_title(f"Bản đồ {env.name.title()}", fontweight="bold")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Điểm đầu", "Đích"], loc="upper left")
    radius = f"{config.robot_radius:g}".replace(".", ",")
    fig.suptitle(f"Môi trường tĩnh | Bán kính robot: {radius} m", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, directory / "environments.png")

    lookup = {(r["environment"], r["algorithm"], r["seed"]): r for r in rows}
    with (directory / "paths.jsonl").open(encoding="utf-8") as stream:
        paths = {(r["environment"], r["algorithm"], r["seed"]): r["path"]
                 for r in map(json.loads, stream)}
    for env in environments:
        seeds = sorted({r["seed"] for r in rows if r["environment"] == env.name})
        # The report uses seed 0 on every map, plus seed 1 on Narrow.
        selected = [seeds[0]]
        if env.name.lower() == "narrow" and 1 in seeds and 1 not in selected:
            selected.append(1)
        for seed in selected:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            try:
                for ax, algorithm in zip(axes, (RRT, RRTStar)):
                    key = (env.name, algorithm.name, seed)
                    row = lookup[key]
                    planner = algorithm(env, config, seed)
                    result = planner.run()
                    recorded_path = paths[key]
                    path_matches = (result.path is None and recorded_path is None) or (
                        result.path is not None and recorded_path is not None
                        and result.path.shape == np.asarray(recorded_path).shape
                        and np.allclose(result.path, recorded_path, atol=1e-10, rtol=0))
                    if (not path_matches or result.success != row["success"]
                            or result.number_of_nodes != row["number_of_nodes"]
                            or result.iterations != int(row["iterations"])
                            or result.rewires != int(row["rewires"])
                            or (result.success and not np.isclose(result.path_length, row["path_length"], atol=1e-10, rtol=0))):
                        raise ValueError(f"Cannot reproduce recorded snapshot: {key}. Run a new benchmark.")
                    view = PlannerView(ax, planner)
                    # Replay reconstructs geometry; displayed times come from the saved run.
                    view.status.set_text(status_text(
                        row['iterations'], config.max_iterations, row['number_of_nodes'],
                        row['planning_time'], row['time_to_first_solution'], row['path_length'], row['rewires']))
                fig.suptitle("Lập kế hoạch đường đi hình học bằng RRT và RRT*", fontweight="bold", fontsize=17)
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                # Use a filename-safe map name for custom environments.
                name = "".join(c if c.isalnum() or c in "-_" else "_" for c in env.name)
                save_figure(fig, directory / f"{name}_seed{seed}.png")
                print(f"Saved snapshot: {env.name}, seed {seed}", flush=True)
            finally:
                plt.close(fig)


def generate(directory, snapshots=True):
    directory = Path(directory)
    rows = read_runs(directory)
    if not rows:
        raise ValueError("No benchmark runs found")
    summary = summarize(rows)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    comparisons = paired_comparison(rows)
    (directory / "paired_comparison.json").write_text(
        json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8"
    )
    statistics = [{"environment": row["environment"], **row["wilcoxon"]} for row in comparisons]
    with (directory / "wilcoxon.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in statistics for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(statistics)
    metric_figures(rows, summary, directory)
    convergence_figures(directory, rows)
    if snapshots:
        report_snapshots(directory, rows)
    print(f"Saved statistics and plots in {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved benchmark results")
    parser.add_argument("--input", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--charts-only", action="store_true", help="Skip map and tree snapshots; only saved CSV data is needed")
    args = parser.parse_args()
    generate(args.input, snapshots=not args.charts_only)

