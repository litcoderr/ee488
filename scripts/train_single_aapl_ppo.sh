#!/usr/bin/env bash
cd ..

set -euo pipefail

RUN_NAME="train_single_aapl_ppo_w30"
DATA_DIR="/mnt/hard2/litcoderr/project/ee488/data"
CHECKPOINT_DIR="/mnt/hard2/litcoderr/project/ee488/checkpoints/single"
START_DATE="2016-01-01"
END_DATE="2020-12-31"
TEST_START="2021-01-01"
TEST_END="2024-12-31"
INIT_BALANCE=10000

CUDA_VISIBLE_DEVICES=6 \
python train.py \
  --mode train \
  --algo ppo \
  --tickers AAPL \
  --action-mode continuous \
  --source csv \
  --data-dir "${DATA_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --init-balance "${INIT_BALANCE}" \
  --episodes 500 \
  --steps-per-episode 256 \
  --rollout-length 256 \
  --window-size 30 \
  --eval-interval 5 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --wandb --wandb-entity litcoderr --wandb-project ee488 --wandb-run-name "${RUN_NAME}"
