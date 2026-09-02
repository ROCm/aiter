#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 GLOBAL_RANK:LOCAL_DEVICE [...]" >&2
  exit 2
fi

cd /home/hzm/aiter
export PYTHONPATH=/home/hzm/aiter
export FLYDSL_RUNTIME_ENABLE_CACHE=1
unset FLYDSL_DEBUG_ENABLE_DEBUG_INFO

for spec in "$@"; do
  rank="${spec%%:*}"
  device="${spec##*:}"
  start="$(date +%s)"
  echo "START rank=${rank} device=${device} epoch=${start}"
  set +e
  HIP_VISIBLE_DEVICES="${device}" python3 \
    scripts/megamoe_tile/compile_ep16_stage2_fused.py \
    --rank "${rank}" \
    --worker-blocks 176 \
    --diagnostic-mode full \
    --final-combine-blocks 4 \
    --gmm-schedule persistent_queue \
    --return-chunk-tokens 8 \
    --bf16-atomic-kind buffer \
    --node-accumulation-mode rank_local \
    --rank-accumulation-mode atomic \
    --node-reduce-blocks 16 \
    --node-reduce-vec-bytes 8 \
    --node-reduce-load-schedule load_first \
    --node-reduce-work-schedule dynamic_head \
    --node-reduce-rejoin-blocks 0 \
    --rank-epilogue-lds-addressing expanded \
    --rail-return-schedule compact \
    --epilogue-schedule lane32_meta \
    --n-tile-group 2 \
    --group-pipeline-schedule a_double_buffer \
    --scoreboard-schedule wave0 \
    --atomic-issue-schedule interleaved
  rc=$?
  set -e
  end="$(date +%s)"
  echo "DONE rank=${rank} device=${device} rc=${rc} seconds=$((end - start))"
  (( rc == 0 )) || exit "${rc}"
done
