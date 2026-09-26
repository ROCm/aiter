#!/bin/bash
# Performance of the TP MegaMoE layer: split baseline vs single-kernel fused engine.
#
#   ./bench_megamoe_tp.sh [-m "dsv3 glm5 kimi3 dsv4"] [-t "256 512 1024 2048"] [-n 4]
#                         [-c "ag_rs ar_ar"] [-s] [-k 5] [-o DIR] [-r REF_DIR] [-x 0.05]
#
# Every (mode, model, tokens) cell runs op_tests/multigpu_tests/test_mega_moe_TP.py
# in its own process (--impl both, CUDA-graph replay, best of -k rounds; the
# test also gates fused-vs-split accuracy). Prints split / fused us and
# split/fused speedup per cell. With -r, compares fused time against a previous
# run's output directory and flags cells slower by more than -x (fraction).
#
# -c: communication modes -- ag_rs (AllGather before, ReduceScatter after) and/or
#     ar_ar (AllReduce before and after); the fused kernel dispatches on it.
# -s: one process drives all -n GPUs (--single-process: peer access, the split
#     baseline on the one-shot P2P kernels of p2p_collectives.py) instead of
#     torchrun -- for nodes whose cross-process GPU sync is too slow to time.
#
# The fused kernel uses FP8 route rows (AITER_MEGAMOE_ROUTE_FP8=0: bf16). GPUs:
# waits for -n idle GPUs before each cell (prefers 4-7); GPUS=4,5,6,7 pins them.
# Overrides: PYTHON, TORCHRUN.
set -u
AITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="dsv3 glm5 kimi3 dsv4"
TOKENS="256 512 1024 2048"
NP=4
MODES="ag_rs ar_ar"
SP=0
ROUNDS=5
REF=""
TOL=0.05
OUT="$AITER_DIR/megamoe_tp_results/bench_$(date +%m%d_%H%M%S)"
while getopts "m:t:n:c:sk:o:r:x:h" opt; do
  case $opt in
    m) MODELS=$OPTARG ;;
    c) MODES=$OPTARG ;;
    s) SP=1 ;;
    t) TOKENS=$OPTARG ;;
    n) NP=$OPTARG ;;
    k) ROUNDS=$OPTARG ;;
    o) OUT=$OPTARG ;;
    r) REF=$OPTARG ;;
    x) TOL=$OPTARG ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
  esac
done
PYTHON=${PYTHON:-/tmp/aiter_venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
TORCHRUN=${TORCHRUN:-$(dirname "$PYTHON")/torchrun}
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
for C in $MODES; do
 for M in $MODELS; do
  for T in $TOKENS; do
    base="$OUT/${C}_${M}_${T}"
    if ! wait_gpus "$NP"; then
      echo "no $NP idle GPUs" > "$base.log"; echo "exit=3" >> "$base.log"
      echo "$C $M/$T: no $NP idle GPUs"; continue
    fi
    rm -f /dev/shm/nccl-*
    args=(op_tests/multigpu_tests/test_mega_moe_TP.py --models "$M" --tokens "$T"
          --comm-mode "$C" --impl both --rounds "$ROUNDS" --csv "$base.csv")
    if [ "$SP" = 1 ]; then
      timeout ${TMO:-1800} "$PYTHON" "${args[@]}" --single-process --tp "$NP" > "$base.log" 2>&1
    else
      timeout ${TMO:-1800} "$TORCHRUN" --nproc_per_node="$NP" "${args[@]}" > "$base.log" 2>&1
    fi
    rc=$?
    printf 'gpus=%s\nexit=%s\n' "$HIP_VISIBLE_DEVICES" "$rc" >> "$base.log"
    echo "$C $M/$T done (exit=$rc, gpus=$HIP_VISIBLE_DEVICES)"
  done
 done
done

"$PYTHON" - "$OUT" "$REF" "$TOL" <<'EOF'
import glob, os, re, sys
import pandas as pd
out, ref, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
order = ["dsv3", "glm5", "kimi3", "dsv4"]

def load(d):
    rows = {}
    for p in glob.glob(os.path.join(d, "*_*.csv")):
        m = re.match(r"(?:(ag_rs|ar_ar)_)?(.+)_(\d+)\.csv$", os.path.basename(p))
        if not m:
            continue
        log = p[:-4] + ".log"
        rc = re.findall(r"^exit=(\d+)", open(log).read(), re.M) if os.path.exists(log) else []
        r = pd.read_csv(p).iloc[-1]
        rows[(m.group(1) or "ag_rs", m.group(2), int(m.group(3)))] = (
            float(r["split_graph_us"]), float(r["fused_graph_us"]), rc[-1] if rc else "?")
    return rows

cur, base = load(out), (load(ref) if ref else {})
key = lambda k: (k[0], order.index(k[1]) if k[1] in order else 99, k[1], k[2])
head = f"{'mode':<7}{'cell':<12}{'split us':>10}{'fused us':>10}{'speedup':>9}  status"
if base:
    head += f"{'ref fused':>11}{'delta':>9}"
print(head)
bad = slow = 0
for k in sorted(cur, key=key):
    s, f, rc = cur[k]
    ok = rc == "0"
    bad += not ok
    line = f"{k[0]:<7}{k[1] + '/' + str(k[2]):<12}{s:>10.1f}{f:>10.1f}{s / f:>9.3f}  {'PASS' if ok else 'FAIL'}  "
    if k in base:
        d = f / base[k][1] - 1
        slow += d > tol
        line += f"{base[k][1]:>9.1f}{d * 100:>8.1f}%" + ("  <-- slower" if d > tol else "")
    print(line)
print(f"{len(cur)} cells, {bad} failed" + (f", {slow} slower than ref by > {tol:.0%}" if base else ""))
print(f"results: {out}")
sys.exit(1 if bad else 0)
EOF
