# RL Trading Boilerplate

Flexible training and evaluation harness for the stock-trading assignment. Includes a Gymnasium environment, DQN and PPO agents, single- or multi-ticker support, and multi-GPU training via `torchrun`.

## Features
- Trading environment with configurable tickers, window size, discrete or continuous actions, transaction fees, and reward scaling.
- Data sources: `yfinance`, local CSVs (`ticker.csv` per symbol), or offline `synthetic` data for quick testing without network access.
- Two algorithms: value-based DQN (discrete action grid) and actor-critic PPO (discrete or continuous actions).
- Multi-GPU/host support through PyTorch Distributed (`torchrun --nproc_per_node=<gpus>`). CPU-only also works.
- Training, evaluation, and checkpointing in a single CLI (`train.py`).

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install torch gymnasium pandas numpy
# Optional for remote data
pip install yfinance
```

## Quick Start
- Synthetic data, single ticker, DQN (CPU):
```
python train.py --algo dqn --tickers AAPL --source synthetic --episodes 5 --steps-per-episode 200
```
- Multi-GPU PPO (continuous actions) on four tickers with CSV data:
```
torchrun --standalone --nproc_per_node=2 train.py \
  --algo ppo --action-mode continuous \
  --tickers AAPL,MSFT,GOOG,AMZN --source csv --data-dir ./data \
  --episodes 10 --rollout-length 256
```
- Evaluate a saved checkpoint on unseen tickers/date range:
```
python train.py --mode eval --algo dqn \
  --checkpoint checkpoints/dqn/final.pt \
  --tickers AAPL --test-tickers TSLA,NVDA,META \
  --test-start 2023-01-01 --test-end 2023-12-31 --eval-episodes 5
```

## Download Real Data to CSV
- Grab offline CSVs once, then point training at them with `--source csv --data-dir ./data` to avoid repeated API calls.
- Example:
```
python collect_data.py --tickers AAPL,MSFT,GOOG,AMZN \
  --start-date 2018-01-01 --end-date 2024-12-31 \
  --data-dir ./data --interval 1d --auto-adjust
```

## Key Arguments
- `--algo {dqn,ppo}`: choose algorithm.
- `--mode {train,eval}`: train or evaluate an existing checkpoint.
- `--tickers` / `--test-tickers`: comma-separated symbols for train/test.
- `--start-date`, `--end-date`, `--test-start`, `--test-end`: data windows.
- `--action-mode {discrete,continuous}`: discrete uses a 3^N action grid (sell/hold/buy per ticker); continuous uses per-ticker values in [-1, 1] scaled by `--max-shares`.
- `--source {synthetic,csv,yfinance}` and `--data-dir` for CSVs.
- `--episodes`, `--steps-per-episode`, `--rollout-length` (PPO), `--eval-interval`, `--checkpoint-dir`.

## Files
- `rl_trader/env.py`: `TradingEnv` (Gymnasium) with portfolio tracking, multi-ticker observations, discrete/continuous actions, transaction fees, and reward based on portfolio delta.
- `rl_trader/data.py`: load price data from `yfinance`, CSV, or synthetic generator.
- `rl_trader/agents.py`: DQN and PPO implementations with replay buffer or on-policy storage, PyTorch models, and DDP awareness.
- `rl_trader/trainer.py`: training loops, evaluation, checkpoint save/load, distributed initialization.
- `train.py`: CLI entry for training and evaluation.

## Workflow Suggestions
1) Train single-ticker agent on a train period, then `--mode eval` on a disjoint period and new tickers (N=1 tests).
2) Train with four tickers (`--tickers T1,T2,T3,T4`), then evaluate:
   - Seen-ticker test: same four tickers.
   - Mixed: `--test-tickers T1,T2,U1,U2`.
   - Unseen: four new symbols.
3) For multi-GPU runs, prefer `torchrun --standalone --nproc_per_node=<gpus> ...`; checkpoints land in `checkpoints/<algo>/`.

## Notes
- Default data source is `synthetic` so runs work without network; switch to `--source yfinance` for real prices or `--source csv` for offline snapshots.
- Transaction fees (`--fee-pct`) and reward scaling (`--reward-scale`) let you shape learning signals.
- Use `--action-mode continuous` with PPO when discrete grids become large for many tickers.
