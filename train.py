import argparse
import os
from typing import List, Optional

from rl_trader.env import EnvConfig
from rl_trader.trainer import TrainConfig, Trainer


def parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def build_env_cfg(args, test: bool = False, random_start: bool = False) -> EnvConfig:
    start = args.test_start if test else args.start_date
    end = args.test_end if test else args.end_date
    return EnvConfig(
        tickers=parse_tickers(args.tickers if not test else args.test_tickers or args.tickers),
        start_date=start,
        end_date=end,
        init_balance=args.init_balance,
        max_shares_per_trade=args.max_shares,
        window_size=args.window_size,
        action_mode=args.action_mode,
        transaction_cost_pct=args.fee_pct,
        reward_scaling=args.reward_scale,
        random_start=random_start,
        seed=args.seed,
        source=args.source,
        data_dir=args.data_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate trading agents.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--tickers", type=str, default="AAPL")
    parser.add_argument("--test-tickers", type=str, default=None, dest="test_tickers")
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date", type=str, default="2022-12-31")
    parser.add_argument("--test-start", type=str, default="2023-01-01")
    parser.add_argument("--test-end", type=str, default="2023-12-31")
    parser.add_argument("--init-balance", type=float, default=10_000)
    parser.add_argument("--max-shares", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-normalize", action="store_true", help="Enable per-episode reward normalization")
    parser.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="synthetic")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--rollout-length", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/user name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Name for the wandb run")
    parser.add_argument("--save-actions", type=str, default=None, help="Path to save actions taken during eval as JSON")
    args = parser.parse_args()

    train_random_start = args.mode == "train"
    env_cfg = build_env_cfg(args, test=False, random_start=train_random_start)
    env_cfg.reward_normalize = args.reward_normalize
    test_cfg = build_env_cfg(args, test=True, random_start=False)
    test_cfg.reward_normalize = False
    train_cfg = TrainConfig(
        algorithm=args.algo,
        episodes=args.episodes,
        max_steps_per_episode=args.steps_per_episode,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        rollout_length=args.rollout_length,
    )

    wandb_run: Optional[object] = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install wandb to enable logging: pip install wandb") from exc
        run_name = args.wandb_run_name or f"{args.mode}-{args.algo}-{args.tickers.replace(',', '_')}"
        wandb_run = wandb.init(
            project=args.wandb_project or os.getenv("WANDB_PROJECT", "ee488"),
            entity=args.wandb_entity or os.getenv("WANDB_ENTITY"),
            name=run_name,
            config={
                "mode": args.mode,
                "algo": args.algo,
                "tickers": args.tickers,
                "test_tickers": args.test_tickers or args.tickers,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "test_start": args.test_start,
                "test_end": args.test_end,
                "init_balance": args.init_balance,
                "max_shares": args.max_shares,
                "window_size": args.window_size,
                "action_mode": args.action_mode,
                "fee_pct": args.fee_pct,
                "reward_scale": args.reward_scale,
                "reward_normalize": args.reward_normalize,
                "random_start": train_random_start,
                "source": args.source,
                "data_dir": args.data_dir,
                "episodes": args.episodes,
                "steps_per_episode": args.steps_per_episode,
                "eval_interval": args.eval_interval,
                "rollout_length": args.rollout_length,
                "seed": args.seed,
            },
        )

    trainer = Trainer(
        env_cfg,
        train_cfg,
        test_cfg,
        wandb_run=wandb_run,
        create_checkpoint_dir=args.mode == "train",
    )
    if args.checkpoint:
        trainer.load(args.checkpoint)

    try:
        if args.mode == "train":
            trainer.train()
        else:
            value = trainer.evaluate(episodes=args.eval_episodes, save_actions_path=args.save_actions)
            print(f"Average portfolio value over {args.eval_episodes} eval runs: {value:.2f}")
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
