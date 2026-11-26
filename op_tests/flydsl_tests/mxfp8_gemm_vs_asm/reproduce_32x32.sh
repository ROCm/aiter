#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# FlyDSL 32x32 (native E8M0 group32 w_scale) MXFP8 GEMM on gfx1250, 12 shapes.
# Uses the tuned rows in aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv
# (kernelId bmm -> compute_wmma t256x256x128) and dsv41's model_configs table.
#
# Runs op_tests/test_gemm_a8w8_blockscale.py --w-scale-block 32x32 once per shape
# (one process each) and prints a markdown table of FlyDSL us / TFLOPS. The 32x32
# mode does not run ASM (the test skips it), so this script has no ASM column; use
# reproduce-1.sh / reproduce-2.sh (1x32) for the ASM ratio.
# Stops at the first "GPU Hang".
#
# Usage (gfx1250 machine; wrap in your GPU lock if the GPU is shared):
#   op_tests/flydsl_tests/mxfp8_gemm_vs_asm/reproduce_32x32.sh
#
# Env vars:
#   FLYDSL_PYTHONPATH  FlyDSL python path put in front of PYTHONPATH (optional)
#   AITER_DIR          aiter checkout                 (default: this repo)
#   SHAPES             "M,N,K ..."                    (default: shapes below)
#   MODES              "data:scale ..."               (default: "uniform:auto")
#   ROUNDS             repeats per shape, median      (default: 1)
#   IDLE_MAX_USE       rocm-smi GPU use (%) counted as idle (default: 15; some
#                      boards idle at a nonzero floor, e.g. 13)
#   SHAPE_TIMEOUT      seconds per process            (default: 900)
#   OUT_DIR            raw logs + tables              (default: /tmp/mxfp8_repro_32x32)
SHAPES=${SHAPES:-"4096,4096,4096 8192,8192,8192 16384,1792,5120 16384,4608,5120 16384,5120,2304 16384,5120,4224 16384,5120,4352 16384,5120,8192 16384,16384,16384 16384,25600,6144 16384,32768,8192 16384,32768,1280"}
OUT_DIR=${OUT_DIR:-/tmp/mxfp8_repro_32x32}
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AITER_DIR=${AITER_DIR:-$(cd "$HERE/../../.." && pwd)}
P=${FLYDSL_PYTHONPATH:+$FLYDSL_PYTHONPATH:}$AITER_DIR
MODES=${MODES:-"uniform:auto"}
ROUNDS=${ROUNDS:-1}
IDLE_MAX_USE=${IDLE_MAX_USE:-15}
SHAPE_TIMEOUT=${SHAPE_TIMEOUT:-900}
mkdir -p "$OUT_DIR"
cd "$AITER_DIR" || exit 1

wait_idle() {
  local t=0 ok=0 use
  while [ $t -lt 900 ]; do
    use=$(rocm-smi --showuse 2>/dev/null | sed -n 's/.*GPU use (%): \([0-9]*\)$/\1/p' | head -1)
    if [ -n "$use" ] && [ "$use" -le "$IDLE_MAX_USE" ]; then
      ok=$((ok + 1)); [ $ok -ge 3 ] && return 0
    else
      ok=0
    fi
    sleep 1; t=$((t + 1))
  done
  echo "wait_idle timeout" >&2
}

HANG=0
for mode in $MODES; do
  di=${mode%%:*}; si=${mode##*:}
  JL=$OUT_DIR/$di.jsonl
  : > "$JL"
  for r in $(seq 1 "$ROUNDS"); do
    for s in $SHAPES; do
      [ $HANG = 1 ] && break
      IFS=, read -r m n k <<< "$s"
      log=$OUT_DIR/$di-m$m-n$n-k$k-r$r.log
      wait_idle
      env -u PYTHONPATH PYTHONPATH=$P HIP_VISIBLE_DEVICES=0 timeout "$SHAPE_TIMEOUT" \
        python3 -u op_tests/test_gemm_a8w8_blockscale.py --w-scale-block 32x32 \
        -m "$m" -nk "$n,$k" --data-init "$di" --scale-init "$si" > "$log" 2>&1
      grep '^{"name": "gemm_a8w8_blockscale summary"' "$log" >> "$JL" \
        || echo "{\"fail\": \"${m}x${n}x${k}\"}" >> "$JL"
      if grep -q 'GPU Hang' "$log"; then
        echo "GPU Hang at ${m}x${n}x${k}, stopping" >&2
        HANG=1
      fi
    done
  done
  echo "== data=$di scale=$si (blockscale test: FlyDSL 32x32, perftest)"
  python3 - "$JL" <<'PYEOF' | tee "$OUT_DIR/$di.tbl"
import collections
import json
import statistics
import sys

runs = collections.defaultdict(list)
kernel, fails = {}, []
for line in open(sys.argv[1]):
    d = json.loads(line)
    if "fail" in d:
        fails.append(d["fail"])
        continue
    for row in d["rows"]:
        key = (row["m"], row["n"], row["k"])
        runs[key].append(row)
        kernel[key] = str(row.get("kernelName", "")).split("_t256x256x128_")[-1]


def med(rows, name):
    vals = [r[name] for r in rows if name in r]
    return statistics.median(vals) if vals else None


print("| Shape | FlyDSL | TFLOPS |")
print("|---|---|---|")
for key in sorted(runs):
    m, n, k = key
    flops = 2 * m * n * k
    f = med(runs[key], "ck us")
    tf = f"{flops / f / 1e6:.0f}" if f else "n/a"
    print(f"| {m}x{n}x{k} | {f:.1f}us | {tf} |" if f else f"| {m}x{n}x{k} | n/a | n/a |")
print("fail:", fails or "none")
PYEOF
done
