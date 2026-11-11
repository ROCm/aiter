#!/usr/bin/env bash
# Submit every bench_champions point as an independent gpuq job, wait, then merge.
# Usage: scripts/run_bench_champions.sh [extra bench_champions.py args, e.g. --perf-layout both --causal both
#   --group all --out-dtype both --tag <label>; all args pass through to every job]
# Env knobs: MODE (default both), KERNELS (default "exact noslack flydsl gluon triton"), SHAPES (default: all)
# Compare two runs: python3 scripts/bench_champions.py --compare <dirA> <dirB>
set -u
W=/home/jograner/projects/aiter/flydsl-unified-attention
TS=$(date +%Y%m%d-%H%M%S)
OUT=$W/bench_out/champions/$TS
mkdir -p "$OUT"
MODE=${MODE:-both}
KERNELS=${KERNELS:-exact noslack flydsl gluon triton}
SHAPES=${SHAPES:-$(cd "$W" && ENABLE_CK=0 PYTHONPATH=$W python3 scripts/bench_champions.py --list | awk '{print $2}')}
pids=()
for s in $SHAPES; do
  for k in $KERNELS; do
    # one job per (kernel, shape); FlyDSL cache off per the README (cached kernels would hide source edits)
    ( cd "$W" && ENABLE_CK=0 FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=$W \
        timeout 900 gpuq --num-gpus=1 python3 -u scripts/bench_champions.py --kernel "$k" --shape "$s" --mode "$MODE" \
        --out "$OUT/${k}__${s}.json" "$@" > "$OUT/${k}__${s}.log" 2>&1 ) &
    pids+=($!)
  done
done
echo "submitted ${#pids[@]} jobs -> $OUT"
for p in "${pids[@]}"; do wait "$p" || echo "job pid $p failed (see logs)"; done
cd "$W" && ENABLE_CK=0 PYTHONPATH=$W python3 scripts/bench_champions.py --merge "$OUT"
echo "summary: $OUT/summary.md"
