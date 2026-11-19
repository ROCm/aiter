#!/bin/bash
# Prefill attention, full-attention layer, fp8 like the TP1 server, BLOCK_M 256, chunks 1-4:
#   1) MIXED batch as today (1 launch: prefill chunk + 3 decode rows)
#   2) MIXED batch split (the changed SGLang: prefill chunk, then the decode rows through the decode route)
#   3) pure EXTEND (no decode rows): the target the split should get close to
# Logs go to ./bench_logs/<date>_<time>/
cd /dockerx/gpt_oss/opt/attention
export PYTHONPATH=$HOME/draftbase_1111/aiter
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
LOG=bench_logs/$(date +%m%d_%H%M); mkdir -p "$LOG"
CFG='{"BLOCK_M": 256}'

echo "== 1) MIXED, single launch (as today)"
python3 bench_prefill_attention.py --fp8 --cfg "$CFG"                 2>&1 | tee "$LOG/1_mixed_single.txt"
echo "== 2) MIXED, decode rows split off (changed SGLang)"
python3 bench_prefill_attention.py --fp8 --cfg "$CFG" --split-decode  2>&1 | tee "$LOG/2_mixed_split.txt"
echo "== 3) pure EXTEND (no decode rows)"
python3 bench_prefill_attention.py --fp8 --cfg "$CFG" --decode-rows 0 2>&1 | tee "$LOG/3_extend.txt"

echo "logs in $LOG"
