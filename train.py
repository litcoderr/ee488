import argparse
from typing import List

from rl_trader.env import EnvConfig
from rl_trader.trainer import TrainConfig, Trainer


def parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def build_env_cfg(args, test: bool = False) -> EnvConfig:
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
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
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
    args = parser.parse_args()

    env_cfg = build_env_cfg(args, test=False)
    test_cfg = build_env_cfg(args, test=True)
    train_cfg = TrainConfig(
        algorithm=args.algo,
        episodes=args.episodes,
        max_steps_per_episode=args.steps_per_episode,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        rollout_length=args.rollout_length,
    )

    trainer = Trainer(env_cfg, train_cfg, test_cfg)
    if args.checkpoint:
        trainer.load(args.checkpoint)

    if args.mode == "train":
        trainer.train()
    else:
        value = trainer.evaluate(episodes=args.eval_episodes)
        print(f"Average portfolio value over {args.eval_episodes} eval runs: {value:.2f}")


if __name__ == "__main__":
    main()
