import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_returns(path: Path) -> List[float]:
    with open(path, "r") as f:
        data = json.load(f)
    returns: List[float] = []
    for ep in data.get("episodes", []):
        start = ep.get("start_portfolio_value")
        if start is None:
            steps = ep.get("steps", [])
            if steps:
                start = steps[0].get("portfolio_value")
        final = ep.get("final_portfolio_value")
        if start is None or final is None:
            continue
        start_f = float(start)
        final_f = float(final)
        returns.append((final_f - start_f) / max(start_f, 1e-8))
    return returns


def parse_series(raw_series: List[str]) -> List[Tuple[str, List[Path]]]:
    parsed: List[Tuple[str, List[Path]]] = []
    for raw in raw_series:
        if ":" not in raw:
            raise ValueError(f"--series expects label:paths, got '{raw}'")
        label, paths_str = raw.split(":", 1)
        paths = [Path(p.strip()) for p in paths_str.split(",") if p.strip()]
        if not paths:
            raise ValueError(f"No paths provided for series '{label}'")
        parsed.append((label, paths))
    return parsed


def compute_stats(series: List[Tuple[str, List[Path]]]) -> Dict[str, List[Dict[str, float]]]:
    stats: Dict[str, List[Dict[str, float]]] = {}
    for label, paths in series:
        entries: List[Dict[str, float]] = []
        for path in paths:
            returns = load_returns(path)
            if not returns:
                print(f"[warn] No usable returns found in {path}, skipping.")
                continue
            mean = float(np.mean(returns))
            variance = float(np.var(returns))
            entries.append(
                {
                    "name": path.stem,
                    "mean": mean,
                    "variance": variance,
                    "returns": returns,
                }
            )
        if entries:
            stats[label] = entries
    return stats


def plot_mean_variance(stats: Dict[str, List[Dict[str, float]]], output: Path, title: str):
    if not stats:
        raise ValueError("No statistics available to plot.")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    # Track how many points land on the same coordinates to offset annotations and reduce overlap.
    hit_counts: Dict[Tuple[float, float], int] = {}

    for idx, (label, entries) in enumerate(stats.items()):
        color = colors[idx % len(colors)]
        for entry in entries:
            mean_pct = entry["mean"] * 100.0
            var_pct = entry["variance"] * (100.0 ** 2)
            point_key = (round(var_pct, 3), round(mean_pct, 3))
            count = hit_counts.get(point_key, 0)
            hit_counts[point_key] = count + 1
            # Offset each subsequent annotation a bit further out to avoid text overlap.
            offset = 5 + count * 6
            ax.scatter(var_pct, mean_pct, color=color, alpha=0.8, label=label if entry is entries[0] else "")
            ax.annotate(
                entry["name"],
                (var_pct, mean_pct),
                textcoords="offset points",
                xytext=(offset, offset),
                fontsize=9,
                ha="left",
                va="bottom",
            )
        all_returns = [r for entry in entries for r in entry["returns"]]
        if all_returns:
            overall_mean = float(np.mean(all_returns) * 100.0)
            overall_var = float(np.var(all_returns) * (100.0 ** 2))
            overall_key = (round(overall_var, 3), round(overall_mean, 3))
            count = hit_counts.get(overall_key, 0)
            hit_counts[overall_key] = count + 1
            offset = 8 + count * 6
            ax.scatter(overall_var, overall_mean, color=color, marker="X", s=90, edgecolors="k", linewidths=0.8, alpha=0.9, label="" if entries else label)
            ax.annotate(
                f"{label}-overall",
                (overall_var, overall_mean),
                textcoords="offset points",
                xytext=(offset, -offset),
                fontsize=9,
                weight="bold",
                ha="left",
                va="top",
            )

    ax.set_xlabel("Variance of return (%)")
    ax.set_ylabel("Mean return (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close(fig)
    print(f"Saved mean-variance plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot mean-variance points from action log JSON files.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Label and comma-separated JSONs, e.g. 'dqn:seen.json,mixed.json,unseen.json'. Can be repeated.",
    )
    parser.add_argument("--output", type=str, default="mean_variance.png", help="Output image path")
    parser.add_argument("--title", type=str, default="Mean-Variance Comparison", help="Plot title")
    args = parser.parse_args()

    series = parse_series(args.series)
    stats = compute_stats(series)
    plot_mean_variance(stats, Path(args.output), args.title)


if __name__ == "__main__":
    main()
