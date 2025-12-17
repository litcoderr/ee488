#!/usr/bin/env bash
cd ..

set -euo pipefail

DATA_DIR="/mnt/hard2/litcoderr/project/ee488/data"
CHECKPOINT_DIR="/mnt/hard2/litcoderr/project/ee488/checkpoints/single"
CKPT="${CHECKPOINT_DIR}/dqn/final.pt"
START_DATE="2016-01-01"
END_DATE="2020-12-31"
TEST_START="2021-01-01"
TEST_END="2024-12-31"

for TICKER in GOOG MSFT NVDA; do
  RUN_NAME="eval_single_aapl_on_${TICKER,,}_dqn"
  INIT_BALANCE=10000
  python train.py \
    --mode eval \
    --algo dqn \
    --checkpoint "${CKPT}" \
    --tickers AAPL \
    --test-tickers "${TICKER}" \
    --source csv \
    --data-dir "${DATA_DIR}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --test-start "${TEST_START}" \
    --test-end "${TEST_END}" \
    --init-balance "${INIT_BALANCE}" \
    --eval-episodes 5 \
    --window-size 30 \
    --save-actions single_dqn_actions_${TICKER,,}.json
done
