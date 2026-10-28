#!/bin/bash
# Interleaved kernel timing of the FlyDSL grouped MoE gemm1/gemm2 (Yan's DSR1 prefill benchmark, run_dsr1.sh).
#
# usage: op_tests/flydsl_tests/moe_exp/bench.sh [ROUNDS] [VARIANT ...]
#   ROUNDS   timed rounds (default 3); round 0 is an extra warm-up/compile round, not timed
#   VARIANT  name[:ENV=VAL,ENV=VAL,...]   (default: base). Each variant adds its env on top of run_dsr1.sh's env.
#            Variants run interleaved (r0: a b c, r1: a b c, ...), one process per (variant, round).
# env:
#   PY           python with torch (default: python3)
#   SITE         directory with AITER (this branch) and FlyDSL installed with pip --target; prepended to PYTHONPATH
#   LOCK         command prefix for every GPU process, e.g. LOCK=gpu-lock on a shared node (default: none)
#   IDLE_CHECK=1 before each process, wait up to 20 s for rocm-smi to show 0% GPU use, else stop
#   OUT          output directory (default: moe_exp_runs/<UTC stamp> in the current directory)
#   MOE_ARGS     test arguments (default: run_dsr1.sh's shape, below)
#   AMD_SMI      amd-smi binary for the firmware/clock lines in meta.txt (default: amd-smi; skipped if missing)
#   MOE_EXP_GRAPH_ITERS / MOE_EXP_GRAPH_REPS   graph timer (bench_driver.py; default 128 launches x 10 replays)
# Writes OUT/meta.txt (node, versions, firmware), OUT/<variant>_r<round>.log, OUT/out/*.pt (outputs),
# OUT/dump-<variant>/ (FlyDSL IR dumps, named after the kernels), and OUT/summary.txt (summarize.py).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
ROUNDS=${1:-3}; shift || true
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(base)
PY=${PY:-python3}; LOCK=${LOCK:-}
OUT=${OUT:-moe_exp_runs/$(date -u +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/out" && OUT=$(cd "$OUT" && pwd)
MOE_ARGS=${MOE_ARGS:-"--scenario kernel --data-format a4w4 --experts 64 --tokens 1536 --topk 8 --model-dim 7168
  --inter-dim 2048 --act silu --no-bias --iters 128 --no-check-aot-cache --data-init uniform --scale-init uniform"}
# run_dsr1.sh's environment
BASE_ENV=(AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_A4W4_A_PRESHUFFLE=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0
          FLYDSL_DUMP_IR=1 AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
          AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}   # FlyDSL 0.3.3.dev882 finds ld.lld only through ROCM_PATH
PYPATH=${SITE:+$SITE:}${PYTHONPATH:-}
LOG=$OUT/run.log
log() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }

idle_wait() {
  local i use
  for i in $(seq 1 10); do
    use=$(rocm-smi --showuse 2>/dev/null | grep -o 'GPU use (%): [0-9]*' | grep -o '[0-9]*$' | head -1)
    [ "${use:-x}" = 0 ] && { echo "idle (GPU use 0%)"; return 0; }
    sleep 2
  done
  echo "busy (GPU use ${use:-?}%)"; return 1
}

{
  echo "start      $(date -u '+%F %T UTC')"
  echo "host       $(hostname)   boot $(uptime -s)"
  echo "repo       $(git -C "$REPO" log -1 --format='%h %cd %s' --date=iso | cut -c1-120)"
  echo "dirty      $(git -C "$REPO" status --short -- aiter op_tests | grep -v '^??' | wc -l) modified tracked files"
  # package metadata only: importing aiter would JIT-build its host modules (and touch the GPU) outside LOCK;
  # run from $HERE so the repository's own aiter/ source tree is not on sys.path
  echo "python     $(cd "$HERE" && PYTHONPATH=$PYPATH $PY -c 'import sys,importlib.metadata as m,importlib.util as u
print(sys.version.split()[0], "| torch", m.version("torch"), "| flydsl", m.version("flydsl"),
      "| aiter", m.version("amd-aiter"), "@", u.find_spec("aiter").origin)' 2>&1 | tail -1)"
  echo "site       ${SITE:-<none>}"
  AMD_SMI=${AMD_SMI:-amd-smi}
  if command -v "$AMD_SMI" >/dev/null; then
    echo "firmware   $($AMD_SMI firmware 2>/dev/null | grep -A1 -E 'FW_ID: (CP_MEC1|PM)$' | grep -oE '(CP_MEC1|PM|FW_VERSION: .*)' | paste -sd' ')"
    echo "clock      $($AMD_SMI metric -g 0 --clock 2>/dev/null | grep -m1 MAX_CLK | tr -s ' ')"
  fi
  echo "variants   ${VARIANTS[*]}"
  echo "rounds     $ROUNDS timed (+ warm-up round 0)"
  echo "args       $(echo $MOE_ARGS)"
  echo "base env   ${BASE_ENV[*]}"
} > "$OUT/meta.txt"
cat "$OUT/meta.txt" | tee -a "$LOG"

for r in $(seq 0 "$ROUNDS"); do
  for spec in "${VARIANTS[@]}"; do
    name=${spec%%:*}; extra=()
    [ "$spec" != "$name" ] && IFS=',' read -r -a extra <<< "${spec#*:}"
    if [ "${IDLE_CHECK:-0}" = 1 ]; then
      s=$(idle_wait) || { log "${name}_r$r: GPU $s, stopping"; exit 1; }
    else s="no idle check"; fi
    log "${name}_r$r start ($s) ${extra[*]}"
    f=$OUT/${name}_r$r.log
    (cd "$REPO" && $LOCK env "${BASE_ENV[@]}" "${extra[@]}" PYTHONPATH="$PYPATH" \
        FLYDSL_RUNTIME_CACHE_DIR="$OUT/cache-$name" FLYDSL_DUMP_DIR="$OUT/dump-$name" \
        MOE_EXP_SAVE_OUT="$OUT/out/${name}_r$r.pt" \
        $PY -u "$HERE/bench_driver.py" $MOE_ARGS > "$f" 2>&1)
    rc=$?
    log "${name}_r$r rc=$rc $(grep -hoE 'gemm[12]: us = [0-9.]+' "$f" | tr '\n' ' ')$(grep -hoE 'gemm[12]: graph_us = [0-9.]+' "$f" | tr '\n' ' ')"
  done
done
log DONE
PYTHONPATH=$PYPATH $PY "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
