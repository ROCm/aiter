#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Reproduce the FlyDSL-vs-ASM MXFP8 GEMM comparison on gfx1250.
#
# METHOD=blockscale (default, like-for-like)
#   Calls op_tests/test_gemm_a8w8_blockscale.py --w-scale-block 1x32, one process
#   per shape. FlyDSL (gemm_a8w8_mxfp8_bpreshuffle) and ASM (gemm_a8w8_mxfp8) are
#   timed by the same aiter perftest in the same process, so the ratio is fair.
#
# METHOD=legacy (reproduces the older, FlyDSL-favoring numbers)
#   ASM    : op_tests/test_mxfp8fp4gemm.py --mode perf --intype a8w8 --apre 0
#            (perftest, rotating inputs, fresh process = first perftest call)
#   FlyDSL : CUDA-event loop (3 warmup + 50 iters) over ONE buffer set (embedded).
#   The two sides are timed differently, which favors FlyDSL: FlyDSL loses ~20% at
#   4096^3 once inputs rotate (18.1us -> ~22us) while ASM loses ~5%, and the first
#   perftest call of a process is ~10-20% slower at small shapes (ASM's case here).
#
# Usage (gfx1250 machine; wrap in your GPU lock if the GPU is shared):
#   op_tests/flydsl_tests/mxfp8_gemm_vs_asm/reproduce.sh
#   METHOD=legacy op_tests/flydsl_tests/mxfp8_gemm_vs_asm/reproduce.sh
#
# Env vars:
#   METHOD             blockscale | legacy            (default: blockscale)
#   FLYDSL_PYTHONPATH  FlyDSL python path put in front of PYTHONPATH (optional)
#   AITER_DIR          aiter checkout                 (default: this repo)
#   SHAPES             "M,N,K ..."                    (default: 12 shapes below)
#   MODES              "data:scale ..."               (default: "uniform:auto constant:constant")
#   ROUNDS             repeats per shape, median      (default: 1 blockscale, 3 legacy)
#   IDLE_MAX_USE       rocm-smi GPU use (%) counted as idle (default: 0; some boards
#                      idle at a nonzero floor, e.g. 13)
#   SHAPE_TIMEOUT      seconds per process            (default: 900)
#   OUT_DIR            raw logs + tables              (default: /tmp/mxfp8_gemm_vs_asm_repro)
#
# Output: per data mode, a markdown table of median us / TFLOPS and FlyDSL/ASM.
# legacy only: K <= 1536 shapes use the ps4 short-K FlyDSL variant.
# blockscale stops at the first "GPU Hang" instead of piling more work on the GPU.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AITER_DIR=${AITER_DIR:-$(cd "$HERE/../../.." && pwd)}
P=${FLYDSL_PYTHONPATH:+$FLYDSL_PYTHONPATH:}$AITER_DIR
METHOD=${METHOD:-blockscale}
SHAPES=${SHAPES:-"4096,4096,4096 8192,8192,8192 16384,1792,5120 16384,4608,5120 16384,5120,2304 16384,5120,4224 16384,5120,4352 16384,5120,8192 16384,16384,16384 16384,25600,6144 16384,32768,8192 16384,32768,1280"}
MODES=${MODES:-"uniform:auto constant:constant"}
if [ "$METHOD" = legacy ]; then ROUNDS=${ROUNDS:-3}; else ROUNDS=${ROUNDS:-1}; fi
IDLE_MAX_USE=${IDLE_MAX_USE:-0}
SHAPE_TIMEOUT=${SHAPE_TIMEOUT:-900}
OUT_DIR=${OUT_DIR:-/tmp/mxfp8_gemm_vs_asm_repro}
case "$METHOD" in blockscale | legacy) ;; *) echo "METHOD must be blockscale or legacy" >&2; exit 2 ;; esac
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

# ---------------------------------------------------------------- blockscale
if [ "$METHOD" = blockscale ]; then
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
          python3 -u op_tests/test_gemm_a8w8_blockscale.py --w-scale-block 1x32 \
          -m "$m" -nk "$n,$k" --data-init "$di" --scale-init "$si" > "$log" 2>&1
        grep '^{"name": "gemm_a8w8_blockscale summary"' "$log" >> "$JL" \
          || echo "{\"fail\": \"${m}x${n}x${k}\"}" >> "$JL"
        if grep -q 'GPU Hang' "$log"; then
          echo "GPU Hang at ${m}x${n}x${k}, stopping" >&2
          HANG=1
        fi
      done
    done
    echo "== data=$di scale=$si (blockscale test: FlyDSL vs ASM, same perftest)"
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


print("| Shape | ASM | FlyDSL | FlyDSL/ASM | kernel |")
print("|---|---|---|---|---|")
for key in sorted(runs):
    m, n, k = key
    flops = 2 * m * n * k
    a, f = med(runs[key], "asm us"), med(runs[key], "ck us")
    fmt = lambda us: f"{us:.1f}us / {flops / us / 1e6:.0f} TF" if us else "n/a"
    rel = f"{a / f * 100:.0f}%" if a and f else "n/a"
    print(f"| {m}x{n}x{k} | {fmt(a)} | {fmt(f)} | {rel} | {kernel[key]} |")
print("fail:", fails or "none")
PYEOF
  done
  exit 0
fi

# -------------------------------------------------------------------- legacy
BENCH=$(mktemp /tmp/mxfp8_fly_bench.XXXXXX.py)
trap 'rm -f "$BENCH"' EXIT
cat > "$BENCH" <<'PYEOF'
import os
import re

import torch
from aiter import dtypes
from aiter.benchmark_data_init import fill_fp8, fill_scale_e8m0, make_generator
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
)
from aiter.ops.gemm_op_a8w8 import _mxfp8_32_fallback_kernel_name
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight

shape, ps = os.environ["PS_LIST"].split(":")
ps = int(ps)
ITERS = int(os.environ.get("PS_ITERS", "50"))
M, N, K = map(int, shape.split("x"))
DI = os.environ.get("DATA_INIT", "uniform")
SI = os.environ.get("SCALE_INIT", "auto")


def time_us(fn, warmup=3, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def deq(x, s):
    e = s.view(torch.uint8).float().sub(127).exp2()
    return (x.float().view(x.shape[0], -1, 32) * e.unsqueeze(-1)).view(x.shape[0], -1)


torch.zeros(1, device="cuda")
kn = re.sub(r"_ps\d+", "", _mxfp8_32_fallback_kernel_name(M, N, K))
if ps > 1:
    kn = re.sub(r"(_cn\d+(?:_apre)?)", rf"\1_ps{ps}", kn, count=1)
gen = make_generator(0)
A = fill_fp8((M, K), DI, gen)
B_raw = fill_fp8((N, K), DI, gen)
B = shuffle_weight(B_raw, (16, 16))
sA = fill_scale_e8m0((M, K // 32), SI, gen).view(dtypes.fp8_e8m0)
sB_raw = fill_scale_e8m0((N, K // 32), SI, gen).view(dtypes.fp8_e8m0)
sB = shuffle_mxfp8fp4_scale(sB_raw)
ref = deq(A, sA) @ deq(B_raw, sB_raw).t()
ref_norm = ref.abs().max().item()
out = torch.empty((M, N), dtype=dtypes.bf16, device="cuda")


def fn():
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(A, B, sA, sB, out, kn, a_scale_preshuffled=False)


try:
    fn()
    torch.cuda.synchronize()
    err = (out.float() - ref).abs().max().item() / ref_norm
    us = time_us(fn)
    print(f"RES {shape} fly ps={ps} us={us:.2f} TFLOPS={2*M*N*K/us/1e6:.1f} relerr={err:.2e} kn={kn}", flush=True)
except Exception as ex:  # noqa: BLE001
    print(f"RES {shape} fly ps={ps} FAIL {type(ex).__name__}: {str(ex)[:300]}", flush=True)
PYEOF

for mode in $MODES; do
  di=${mode%%:*}; si=${mode##*:}
  OUT=$OUT_DIR/$di.log
  rm -f "$OUT"
  rm -rf "$OUT_DIR/flydsl-cache-$di"
  for r in $(seq 1 "$ROUNDS"); do
    for s in $SHAPES; do
      shp=${s//,/x}
      k=${s##*,}
      ps=1; [ "$k" -le 1536 ] && ps=4
      wait_idle
      # ASM baseline (fresh process, aiter perftest)
      env -u PYTHONPATH PYTHONPATH=$P HIP_VISIBLE_DEVICES=0 timeout "$SHAPE_TIMEOUT" python3 -u op_tests/test_mxfp8fp4gemm.py \
        --mode perf --intype a8w8 --apre 0 --data-init "$di" --scale-init "$si" -s "$s" 2>&1 \
        | python3 -c '
import sys
shp = sys.argv[1]
rows = [l for l in sys.stdin if l.startswith("|")]
hdr = [c.strip() for c in rows[0].strip("|\n").split("|")] if rows else []
for l in rows[2:]:
    c = [x.strip() for x in l.strip("|\n").split("|")]
    d = dict(zip(hdr, c)); print("asm RES %s asm ps=1 us=%s TFLOPS=0 relerr=0" % (shp, d.get("asm us", "nan")))
if len(rows) < 3:
    print("asm RES %s asm ps=1 FAIL no-table" % shp)
' "$shp" >> "$OUT"
      wait_idle
      # FlyDSL (CUDA-event loop over one buffer set)
      env -u PYTHONPATH PYTHONPATH=$P HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_ENABLE_CACHE=1 \
        FLYDSL_RUNTIME_CACHE_DIR="$OUT_DIR/flydsl-cache-$di" DATA_INIT=$di SCALE_INIT=$si \
        PS_ITERS=50 PS_LIST="$shp:$ps" timeout "$SHAPE_TIMEOUT" python3 -u "$BENCH" 2>&1 \
        | grep RES | sed 's/^/fly /' >> "$OUT"
    done
  done
  echo "== data=$di scale=$si (legacy method: Fly event loop vs ASM perftest)"
  python3 - "$OUT" <<'PYEOF' | tee "$OUT_DIR/$di.tbl"
import collections
import re
import statistics
import sys

d = collections.defaultdict(list)
bad, fail = set(), set()
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) RES (\S+) (\S+)\s+ps=(\d+) us=([\d.]+|nan) TFLOPS=\S+ relerr=(\S+)", l)
    if m:
        if m[5] != "nan":
            d[(m[2], m[1])].append(float(m[5]))
        if float(m[6]) > 1e-2:
            bad.add((m[2], m[1], m[6]))
    elif "FAIL" in l:
        fail.add(" ".join(l.split()[:3]))
shapes = sorted({s for s, _ in d}, key=lambda s: [int(x) for x in s.split("x")])


def cell(s, t):
    xs = d.get((s, t))
    if not xs:
        return "n/a"
    M, N, K = map(int, s.split("x"))
    med = statistics.median(xs)
    return f"{med:.1f}us / {2 * M * N * K / med / 1e6:.0f} TF"


print("| Shape | ASM | FlyDSL | FlyDSL/ASM |")
print("|---|---|---|---|")
for s in shapes:
    a, f = d.get((s, "asm")), d.get((s, "fly"))
    rel = f"{statistics.median(a) / statistics.median(f) * 100:.0f}%" if a and f else "n/a"
    print(f"| {s} | {cell(s, 'asm')} | {cell(s, 'fly')} | {rel} |")
print("relerr>1e-2:", bad or "none")
print("fail:", fail or "none")
PYEOF
done
