python plot_actions.py \
    --json single_dqn_actions_goog.json single_dqn_actions_msft.json single_dqn_actions_nvda.json single_dqn_actions_aapl.json \
    --output-dir checkpoints/single/dqn/plots \
    --prefix single_dqn

python plot_mean_variance.py \
    --series "dqn:single_dqn_actions_goog.json,single_dqn_actions_msft.json,single_dqn_actions_nvda.json,single_dqn_actions_aapl.json" \
    --output checkpoints/single/dqn/plots/mean_variance_single_dqn.png \
    --title "Single-ticker mean-variance (DQN)"
