python plot_actions.py \
    --json single_ppo_actions_goog.json single_ppo_actions_msft.json single_ppo_actions_nvda.json single_ppo_actions_aapl.json \
    --output-dir checkpoints/single/ppo/plots \
    --prefix single_ppo

python plot_mean_variance.py \
    --series "ppo:single_ppo_actions_goog.json,single_ppo_actions_msft.json,single_ppo_actions_nvda.json,single_ppo_actions_aapl.json" \
    --output checkpoints/single/ppo/plots/mean_variance_single_ppo.png \
    --title "Single-ticker mean-variance (PPO)"
