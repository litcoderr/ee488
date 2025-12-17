import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def load_actions(path: str):
    with open(path, "r") as f:
        return json.load(f)


def plot_all_tickers(data: dict, episode_idx: int, output_dir: Path, prefix: str):
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError("No episodes found in action log")
    if episode_idx >= len(episodes):
        raise IndexError(f"Episode index {episode_idx} out of range (found {len(episodes)})")
    episode = episodes[episode_idx]
    steps = episode.get("steps", [])
    tickers = data.get("tickers") or ([f"ticker_{i}" for i in range(len(steps[0].get('closes', [])))] if steps else [])
    inferred_value = steps[0].get("portfolio_value") if steps else episode.get("final_portfolio_value")
    start_value = episode.get("start_portfolio_value", inferred_value)
    if start_value is None:
        raise ValueError("Could not infer starting portfolio value for episode")
    start_value = float(start_value)
    end_value = float(episode.get("final_portfolio_value", start_value))
    profit_pct = ((end_value - start_value) / max(start_value, 1e-8)) * 100.0
    output_dir.mkdir(parents=True, exist_ok=True)

    if not steps:
        print(f"Episode {episode_idx} has no executed trades; skipping price/action plot.")
        return

    for ticker_index, ticker_name in enumerate(tickers):
        closes = [step["closes"][ticker_index] for step in steps]
        dates = list(range(len(steps)))  # simple index for x-axis
        if all("executed_trades" in step for step in steps):
            action_key = "executed_trades"
        elif all("requested_trades" in step for step in steps):
            action_key = "requested_trades"
        elif all("action_vector" in step for step in steps):
            action_key = "action_vector"
        else:
            raise ValueError("No compatible action key found in steps")
        actions = [step[action_key][ticker_index] for step in steps]

        buys_x = [i for i, a in enumerate(actions) if a > 0]
        buys_y = [closes[i] for i in buys_x]
        sells_x = [i for i, a in enumerate(actions) if a < 0]
        sells_y = [closes[i] for i in sells_x]

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure(figsize=(11, 5))
        plt.plot(dates, closes, label="Close", linewidth=2, color="#1f77b4")
        if buys_x:
            plt.scatter(buys_x, buys_y, color="#2ca02c", marker="^", label="Buy", zorder=3, s=60, edgecolors="k", linewidths=0.5)
        if sells_x:
            plt.scatter(sells_x, sells_y, color="#d62728", marker="v", label="Sell", zorder=3, s=60, edgecolors="k", linewidths=0.5)
        plt.xlabel("Step")
        plt.ylabel("Price")
        title = (
            f"{ticker_name} | Episode {episode_idx} | "
            f"Final: {end_value:.2f} ({profit_pct:+.2f}%)"
        )
        plt.title(title)
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.legend()
        plt.tight_layout()
        out_name = f"{prefix}_ep{episode_idx}_{ticker_name}.png"
        plt.savefig(output_dir / out_name)
        plt.close()
        print(f"Saved plot to {output_dir / out_name}")


def main():
    parser = argparse.ArgumentParser(description="Plot price and buy/sell actions from saved action log JSON.")
    parser.add_argument("--json", required=True, nargs="+", help="Path(s) to action log JSON")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Episode indices to plot, comma-separated, or 'all' (default). Example: 0,2",
    )
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to write plots")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output files; defaults to JSON stem")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for idx, json_path in enumerate(args.json):
        data = load_actions(json_path)
        total_eps = len(data.get("episodes", []))
        if total_eps == 0:
            print(f"No episodes in {json_path}, skipping.")
            continue
        if args.episodes.lower() == "all":
            ep_idxs = list(range(total_eps))
        else:
            ep_idxs = [int(x) for x in args.episodes.split(",") if x.strip().isdigit()]
            ep_idxs = [e for e in ep_idxs if 0 <= e < total_eps]
        base_prefix = args.prefix if args.prefix else Path(json_path).stem
        # Ensure unique prefix when multiple JSON files are supplied with a single prefix
        prefix = f"{base_prefix}_{idx}" if args.prefix and len(args.json) > 1 else base_prefix
        target_dir = output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for ep in ep_idxs:
            plot_all_tickers(data, ep, target_dir, prefix)


if __name__ == "__main__":
    main()
