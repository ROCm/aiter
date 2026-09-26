#!/bin/bash
# Accuracy / functional UT for the TP MegaMoE layer (single-kernel fused engine).
#
#   ./run_megamoe_tp_ut.sh [-m "dsv3 glm5 kimi3 dsv4"] [-t "256 512 1024 2048"] [-n 4]
#                          [-c "ag_rs ar_ar"] [-s] [-o DIR]
#
# Every (mode, model, tokens) cell runs op_tests/multigpu_tests/test_mega_moe_TP.py
# in its own process with --impl both --no-perf and the torch reference enabled,
# so each cell checks:
#   split  vs torch reference   rel_l2 < 0.06   (test gate)
#   fused  vs split             rel_l2 < 0.06   (test gate; split runs FP8 route outputs)
#   fused  vs torch reference   rel_l2 < 0.06   (test gate)
#                                      < UT_REF_TOL (default 0.04, this script)
# plus a NaN check on both outputs. Exit status is non-zero if any cell fails.
#
# -c: communication modes, ag_rs (AllGather / ReduceScatter) and/or ar_ar
#     (AllReduce before and after). -s: one process drives all -n GPUs
#     (--single-process) instead of torchrun.
# GPUs: before each cell, waits for -n idle GPUs (prefers 4-7). Set GPUS=4,5,6,7
# to pin them instead. Environment overrides: PYTHON, TORCHRUN, UT_REF_TOL.
# The fused kernel keeps FP8 route rows by default (~0.027 vs torch, the split
# baseline is ~0.034); AITER_MEGAMOE_ROUTE_FP8=0 tests bf16 routes (~0.003,
# then UT_REF_TOL=0.01 is a meaningful gate).
set -u
AITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="dsv3 glm5 kimi3 dsv4"
TOKENS="256 512 1024 2048"
NP=4
MODES="ag_rs ar_ar"
SP=0
OUT="$AITER_DIR/megamoe_tp_results/ut_$(date +%m%d_%H%M%S)"
while getopts "m:t:n:c:so:h" opt; do
  case $opt in
    m) MODELS=$OPTARG ;;
    c) MODES=$OPTARG ;;
    s) SP=1 ;;
    t) TOKENS=$OPTARG ;;
    n) NP=$OPTARG ;;
    o) OUT=$OPTARG ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
  esac
done
PYTHON=${PYTHON:-/tmp/aiter_venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
TORCHRUN=${TORCHRUN:-$(dirname "$PYTHON")/torchrun}
UT_REF_TOL=${UT_REF_TOL:-0.04}
mkdir -p "$OUT"

export PYTHONPATH=$AITER_DIR
export AITER_USE_SYSTEM_TRITON=1 AITER_SITUV2_A4W4=1 AITER_FLYDSL_STAGE2_FP8=1
unset AITER_CONFIG_FMOE

# idle = <5% use and <2% VRAM in 5 samples; prefers GPUs 4-7
pick_gpus() {
  "$PYTHON" - "$1" <<'EOF'
import json, subprocess, sys, time
n, busy, d = int(sys.argv[1]), set(), {}
for _ in range(5):
    d = json.loads(subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--json"],
                                  capture_output=True, text=True).stdout)
    for k, v in d.items():
        if k.startswith("card") and (float(v.get("GPU use (%)", 0)) >= 5
                                     or float(v.get("GPU Memory Allocated (VRAM%)", 0)) >= 2):
            busy.add(int(k[4:]))
    time.sleep(0.4)
cards = sorted(int(k[4:]) for k in d if k.startswith("card"))
free = [g for g in cards if g not in busy]
free = [g for g in free if 4 <= g <= 7] + [g for g in free if not 4 <= g <= 7]
if len(free) < n:
    sys.exit(1)
print(",".join(map(str, sorted(free[:n]))))
EOF
}
wait_gpus() {
  if [ -n "${GPUS:-}" ]; then export HIP_VISIBLE_DEVICES=$GPUS; return 0; fi
  local g
  for _ in $(seq 1 40); do
    g=$(pick_gpus "$1") && { export HIP_VISIBLE_DEVICES=$g; return 0; }
    sleep 30
  done
  return 1
}

cd "$AITER_DIR"
fail=0
for C in $MODES; do
 for M in $MODELS; do
  for T in $TOKENS; do
    base="$OUT/${C}_${M}_${T}"
    if ! wait_gpus "$NP"; then
      echo "FAIL  $C $M/$T: no $NP idle GPUs"; fail=1; continue
    fi
    rm -f /dev/shm/nccl-*
    args=(op_tests/multigpu_tests/test_mega_moe_TP.py --models "$M" --tokens "$T"
          --comm-mode "$C" --impl both --no-perf --accuracy-max-tokens "$T" --csv "$base.csv")
    if [ "$SP" = 1 ]; then
      timeout ${TMO:-1800} "$PYTHON" "${args[@]}" --single-process --tp "$NP" > "$base.log" 2>&1
    else
      timeout ${TMO:-1800} "$TORCHRUN" --nproc_per_node="$NP" "${args[@]}" > "$base.log" 2>&1
    fi
    rc=$?
    verdict=$("$PYTHON" - "$base.csv" "$rc" "$UT_REF_TOL" "$base.log" <<'EOF'
import math, os, sys
import pandas as pd
csv, rc, tol, log = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
if rc != 0 or not os.path.exists(csv):
    err = [l.strip() for l in open(log) if "AssertionError" in l or "Error:" in l]
    print(f"FAIL  rc={rc} {err[-1][:160] if err else ''}")
    sys.exit()
if "[SKIP]" in open(log).read():
    print("FAIL  cell skipped (see log)")
    sys.exit()
r = pd.read_csv(csv).iloc[-1]
split, fvs, fref = (float(r.get(c, math.nan)) for c in ("rel_l2", "fused_rel_l2", "fused_ref_rel_l2"))
ok = fref == fref and fref < tol
print(f"{'PASS' if ok else 'FAIL'}  split_vs_ref={split:.5f} fused_vs_split={fvs:.5f} "
      f"fused_vs_ref={fref:.5f}{'' if ok else f' (>= {tol})'}")
EOF
)
    echo "$verdict" | grep -q '^PASS' || fail=1
    printf '%-6s %-6s %-10s gpus=%s\n' "${verdict%% *}" "$C" "$M/$T" "$HIP_VISIBLE_DEVICES"
    echo "       ${verdict#* }"
  done
 done
done
echo "logs: $OUT"
[ $fail -eq 0 ] && echo "ALL PASS" || echo "SOME CELLS FAILED"
exit $fail
