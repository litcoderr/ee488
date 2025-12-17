#!/usr/bin/env bash
cd ..

set -euo pipefail

CHECKPOINT_DIR="/mnt/hard2/litcoderr/project/ee488/checkpoints/multi"
CKPT="${CHECKPOINT_DIR}/dqn/final.pt"
DATA_DIR="/mnt/hard2/litcoderr/project/ee488/data"
ACTION_DIR="/mnt/hard2/litcoderr/project/ee488"
START_DATE="2016-01-01"
END_DATE="2020-12-31"
TEST_START="2021-01-01"
TEST_END="2024-12-31"
INIT_BALANCE=10000
TRAIN_TICKERS="AAPL,MSFT,GOOG,NVDA"

mkdir -p "${ACTION_DIR}"

# Seen tickers (same as training)
python train.py \
  --mode eval \
  --algo dqn \
  --checkpoint "${CKPT}" \
  --tickers "${TRAIN_TICKERS}" \
  --test-tickers "${TRAIN_TICKERS}" \
  --source csv \
  --data-dir "${DATA_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --init-balance "${INIT_BALANCE}" \
  --eval-episodes 5 \
  --save-actions "${ACTION_DIR}/multi_seen_dqn_actions.json"

# Mixed: two seen + two new
python train.py \
  --mode eval \
  --algo dqn \
  --checkpoint "${CKPT}" \
  --tickers "${TRAIN_TICKERS}" \
  --test-tickers "AAPL,MSFT,TSLA,AMZN" \
  --source "${DATA_SOURCE:-synthetic}" \
  --data-dir "${DATA_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --init-balance "${INIT_BALANCE}" \
  --eval-episodes 5 \
  --save-actions "${ACTION_DIR}/multi_mixed_dqn_actions.json"

# Unseen: four new tickers
python train.py \
  --mode eval \
  --algo dqn \
  --checkpoint "${CKPT}" \
  --tickers "${TRAIN_TICKERS}" \
  --test-tickers "TSLA,AMZN,META,IBM" \
  --source "${DATA_SOURCE:-synthetic}" \
  --data-dir "${DATA_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --init-balance "${INIT_BALANCE}" \
  --eval-episodes 5 \
  --save-actions "${ACTION_DIR}/multi_unseen_dqn_actions.json"
