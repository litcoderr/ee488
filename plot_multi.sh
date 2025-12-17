# plot dqn results
python plot_actions.py \
    --json multi_seen_dqn_actions.json \
    --output-dir checkpoints/multi/dqn/plots \
    --prefix multi_seen_dqn \
    --episodes 0

python plot_actions.py \
    --json multi_mixed_dqn_actions.json \
    --output-dir checkpoints/multi/dqn/plots \
    --prefix multi_mixed_dqn \
    --episodes 0

python plot_actions.py \
    --json multi_unseen_dqn_actions.json \
    --output-dir checkpoints/multi/dqn/plots \
    --prefix multi_unseen_dqn \
    --episodes 0

# plot ppo results
python plot_actions.py \
    --json multi_seen_ppo_actions.json \
    --output-dir checkpoints/multi/ppo/plots \
    --prefix multi_seen_ppo \
    --episodes 0

python plot_actions.py \
    --json multi_mixed_ppo_actions.json \
    --output-dir checkpoints/multi/ppo/plots \
    --prefix multi_mixed_ppo \
    --episodes 0

python plot_actions.py \
    --json multi_unseen_ppo_actions.json \
    --output-dir checkpoints/multi/ppo/plots \
    --prefix multi_unseen_ppo \
    --episodes 0

python plot_mean_variance.py \
    --series "dqn:multi_seen_dqn_actions.json,multi_mixed_dqn_actions.json,multi_unseen_dqn_actions.json" \
    --series "ppo:multi_seen_ppo_actions.json,multi_mixed_ppo_actions.json,multi_unseen_ppo_actions.json" \
    --output checkpoints/multi/mean_variance_multi.png \
    --title "Multi-ticker mean-variance (final returns)"
