#!/bin/bash
# Performance of the TP MegaMoE layer: split baseline vs single-kernel fused engine.
#
#   ./bench_megamoe_tp.sh [-m "dsv3 glm5 kimi3 dsv4"] [-t "256 512 1024 2048"] [-n 4]
#                         [-k 5] [-o DIR] [-r REF_DIR] [-x 0.05]
#
# Every (model, tokens) cell runs op_tests/multigpu_tests/test_mega_moe_TP.py in
# its own process (--impl both, CUDA-graph replay, best of -k rounds; the
# test also gates fused-vs-split accuracy). Prints split / fused us and
# split/fused speedup per cell. With -r, compares fused time against a previous
# run's output directory and flags cells slower by more than -x (fraction).
#
# AITER_MEGAMOE_ROUTE_FP8=1 runs the fused kernel with FP8 route rows. GPUs:
# waits for -n idle GPUs before each cell (prefers 4-7); GPUS=4,5,6,7 pins them.
# Overrides: PYTHON, TORCHRUN.
set -u
AITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="dsv3 glm5 kimi3 dsv4"
TOKENS="256 512 1024 2048"
NP=4
ROUNDS=5
REF=""
TOL=0.05
OUT="$AITER_DIR/megamoe_tp_results/bench_$(date +%m%d_%H%M%S)"
while getopts "m:t:n:k:o:r:x:h" opt; do
  case $opt in
    m) MODELS=$OPTARG ;;
    t) TOKENS=$OPTARG ;;
    n) NP=$OPTARG ;;
    k) ROUNDS=$OPTARG ;;
    o) OUT=$OPTARG ;;
    r) REF=$OPTARG ;;
    x) TOL=$OPTARG ;;
    *) sed -n '2,18p' "$0"; exit 2 ;;
  esac
done
PYTHON=${PYTHON:-/tmp/aiter_venv/bin/python}
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
for M in $MODELS; do
  for T in $TOKENS; do
    base="$OUT/${M}_${T}"
    if ! wait_gpus "$NP"; then
      echo "no $NP idle GPUs" > "$base.log"; echo "exit=3" >> "$base.log"
      echo "$M/$T: no $NP idle GPUs"; continue
    fi
    rm -f /dev/shm/nccl-*
    timeout ${TMO:-1800} "$TORCHRUN" --nproc_per_node="$NP" \
      op_tests/multigpu_tests/test_mega_moe_TP.py --models "$M" --tokens "$T" \
      --impl both --rounds "$ROUNDS" \
      --csv "$base.csv" > "$base.log" 2>&1
    rc=$?
    printf 'gpus=%s\nexit=%s\n' "$HIP_VISIBLE_DEVICES" "$rc" >> "$base.log"
    echo "$M/$T done (exit=$rc, gpus=$HIP_VISIBLE_DEVICES)"
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
        m = re.match(r"(.+)_(\d+)\.csv$", os.path.basename(p))
        if not m:
            continue
        log = p[:-4] + ".log"
        rc = re.findall(r"^exit=(\d+)", open(log).read(), re.M) if os.path.exists(log) else []
        r = pd.read_csv(p).iloc[-1]
        rows[(m.group(1), int(m.group(2)))] = (
            float(r["split_graph_us"]), float(r["fused_graph_us"]), rc[-1] if rc else "?")
    return rows

cur, base = load(out), (load(ref) if ref else {})
key = lambda k: (order.index(k[0]) if k[0] in order else 99, k[0], k[1])
head = f"{'cell':<12}{'split us':>10}{'fused us':>10}{'speedup':>9}  status"
if base:
    head += f"{'ref fused':>11}{'delta':>9}"
print(head)
bad = slow = 0
for k in sorted(cur, key=key):
    s, f, rc = cur[k]
    ok = rc == "0"
    bad += not ok
    line = f"{k[0] + '/' + str(k[1]):<12}{s:>10.1f}{f:>10.1f}{s / f:>9.3f}  {'PASS' if ok else 'FAIL'}  "
    if k in base:
        d = f / base[k][1] - 1
        slow += d > tol
        line += f"{base[k][1]:>9.1f}{d * 100:>8.1f}%" + ("  <-- slower" if d > tol else "")
    print(line)
print(f"{len(cur)} cells, {bad} failed" + (f", {slow} slower than ref by > {tol:.0%}" if base else ""))
print(f"results: {out}")
sys.exit(1 if bad else 0)
EOF
