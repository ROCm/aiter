#!/bin/bash
# Measure the gpt-oss prefill attention on gfx1250 (run inside the container, on a free GPU).
# DT=fp8 (default) = the TP1 launch (--kv-cache-dtype fp8_e4m3, SGLANG_6003_PREFILL_Q=static); DT=bf16 = bf16 KV.
#   1) original vs new full-attention config (BLOCK_M 128 vs 256) at chunks 1..4, with accuracy checks
#   2) the installed table config for both layer types, with accuracy checks
#   3) sweep launch configs at the chunk-3 prefix (32k); each prints the DEFAULT.json entry of its winner
# Needs aiter importable with the changed DEFAULT.json installed (apply.sh), e.g.
#   PYTHONPATH=$HOME/draftbase_1111/aiter bash run_prefill_sweep.sh
#   DT=bf16 HIP_VISIBLE_DEVICES=6 bash run_prefill_sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DT="${DT:-fp8}"
OUT="${OUT:-$HERE/results_${DT}_$(date +%m%d_%H%M)}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
mkdir -p "$OUT"
B="python3 $HERE/bench_prefill_attention.py"
[ "$DT" = fp8 ] && B="$B --fp8"
echo "dtype=$DT  out=$OUT  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"

echo "== full-attention layer: ORIGINAL config (BLOCK_M 128) at chunks 1..4"
$B --sw 0 --cfg '{"BLOCK_M": 128}' 2>&1 | tee "$OUT/full_blockm128.txt"
echo "== full-attention layer: installed table config (new entry, BLOCK_M 256) at chunks 1..4"
$B --sw 0 2>&1 | tee "$OUT/full_table.txt"
echo "== full-attention layer: same, decode rows split off (the changed SGLang _forward_extend_unified)"
$B --sw 0 --split-decode 2>&1 | tee "$OUT/full_table_split.txt"
echo "== full-attention layer: pure EXTEND (no decode rows) for reference"
$B --sw 0 --decode-rows 0 2>&1 | tee "$OUT/full_table_extend.txt"
echo "== sliding-window layer: installed table config"
$B --sw 127 2>&1 | tee "$OUT/sw_table.txt"

echo "== sweep, full-attention layer, prefix 32768"
$B --sw 0 --sweep --prefix 32768 2>&1 | tee "$OUT/sweep_full_p32k.txt"
echo "== sweep, sliding-window layer, prefix 32768"
$B --sw 127 --sweep --prefix 32768 2>&1 | tee "$OUT/sweep_sw_p32k.txt"

echo "results in $OUT"
