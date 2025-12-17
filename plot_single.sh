#!/usr/bin/env bash
set -euo pipefail

# Plot single-ticker DQN runs
python plot_actions.py \
    --json single_dqn_actions_goog.json single_dqn_actions_msft.json single_dqn_actions_nvda.json single_dqn_actions_aapl.json \
    --output-dir checkpoints/single/dqn/plots \
    --prefix single_dqn

# Plot single-ticker PPO runs
python plot_actions.py \
    --json single_ppo_actions_goog.json single_ppo_actions_msft.json single_ppo_actions_nvda.json single_ppo_actions_aapl.json \
    --output-dir checkpoints/single/ppo/plots \
    --prefix single_ppo

# Mean-variance overlay comparing DQN vs PPO
python plot_mean_variance.py \
    --series "dqn:single_dqn_actions_goog.json,single_dqn_actions_msft.json,single_dqn_actions_nvda.json,single_dqn_actions_aapl.json" \
    --series "ppo:single_ppo_actions_goog.json,single_ppo_actions_msft.json,single_ppo_actions_nvda.json,single_ppo_actions_aapl.json" \
    --output checkpoints/single/mean_variance_single.png \
    --title "Single-ticker mean-variance (DQN vs PPO)"
