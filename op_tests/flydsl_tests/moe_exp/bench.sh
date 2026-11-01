#!/bin/bash
# Interleaved kernel timing of the FlyDSL grouped MoE quant_a1/gemm1/gemm2 (Yan's DSR1 conc6144 prefill benchmark,
# run_a4w4_conc6144.sh).
#
# usage: op_tests/flydsl_tests/moe_exp/bench.sh [ROUNDS] [VARIANT ...]
#   ROUNDS   timed rounds (default 3); round 0 is an extra warm-up/compile round, not timed
#   VARIANT  name[:ENV=VAL,ENV=VAL,...]   (default: base). Each variant adds its env on top of
#            run_a4w4_conc6144.sh's env.
#            Variants run interleaved (r0: a b c, r1: a b c, ...), one process per (variant, round).
# env:
#   PY           python with torch (default: python3)
#   SITE         directory with AITER (this branch) and FlyDSL installed with pip --target; prepended to PYTHONPATH
#   LOCK         command prefix for every GPU process, e.g. LOCK=gpu-lock on a shared node (default: none)
#   IDLE_CHECK=1 inside LOCK, before each process: wait up to 20 s for rocm-smi to show 0% GPU use and less than
#                IDLE_VRAM_MB (default 1024) of VRAM in use, else stop the series (a loaded but idle model server
#                shows 0% use, so the VRAM check matters on a shared node)
#   OUT          output directory (default: moe_exp_runs/<UTC stamp> in the current directory)
#   MOE_ARGS     test arguments (default: run_a4w4_conc6144.sh's shape, below)
#   AMD_SMI      amd-smi binary for the firmware/clock lines in meta.txt (default: amd-smi; skipped if missing)
#   MOE_EXP_GRAPH_ITERS / MOE_EXP_GRAPH_REPS / MOE_EXP_GRAPH_GAP_S   graph timer (bench_driver.py; default 128
#                launches x 10 replays, 1 s idle before each kernel's graph timing; ITERS=0 turns it off)
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
# run_a4w4_conc6144.sh's environment (the A layout comes from the config CSV; run_dsr1.sh forced
# AITER_GROUPED_A4W4_A_PRESHUFFLE=1 and had no MMA-group / DS-first settings)
BASE_ENV=(AITER_FLYDSL_MMA_FIRST_GROUP=4 AITER_FLYDSL_MMA_GROUP=4 AITER_FLYDSL_DS_FIRST_N=8
          AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0
          FLYDSL_DUMP_IR=1 AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
          AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}   # FlyDSL 0.3.3.dev882 finds ld.lld only through ROCM_PATH
PYPATH=${SITE:+$SITE:}${PYTHONPATH:-}
LOG=$OUT/run.log
log() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }

# Idle check, run inside LOCK so nothing of ours can start between the check and the run. Exit 99 = busy.
IDLE_VRAM_MB=${IDLE_VRAM_MB:-1024}
IDLE_SH='for i in $(seq 1 10); do
  use=$(rocm-smi --showuse 2>/dev/null | grep -o "GPU use (%): [0-9]*" | grep -o "[0-9]*$" | head -1)
  vb=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -o "Used Memory (B): [0-9]*" | grep -o "[0-9]*$" | head -1)
  vram=$(( ${vb:-0} / 1048576 ))
  [ "${use:-x}" = 0 ] && [ -n "$vb" ] && [ "$vram" -lt '"$IDLE_VRAM_MB"' ] && {
    echo "[moe-exp] idle: GPU use 0%, VRAM ${vram} MB"; exec "$@"; }
  sleep 2
done
echo "[moe-exp] busy: GPU use ${use:-?}%, VRAM ${vram:-?} MB in use; not running"; exit 99'

{
  echo "start      $(date -u '+%F %T UTC')"
  echo "host       $(hostname)   boot $(uptime -s) (system clock, $(date +%Z))"
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
    echo "power cap  $($AMD_SMI static -g 0 --limit 2>/dev/null | grep -E 'SOCKET_POWER_LIMIT' | tr -s ' ' | paste -sd' ')"
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
    pre=(); [ "${IDLE_CHECK:-0}" = 1 ] && pre=(bash -c "$IDLE_SH" _)
    log "${name}_r$r queued ${extra[*]}"
    f=$OUT/${name}_r$r.log
    (cd "$REPO" && $LOCK "${pre[@]}" env "${BASE_ENV[@]}" "${extra[@]}" PYTHONPATH="$PYPATH" \
        FLYDSL_RUNTIME_CACHE_DIR="$OUT/cache-$name" FLYDSL_DUMP_DIR="$OUT/dump-$name" \
        MOE_EXP_SAVE_OUT="$OUT/out/${name}_r$r.pt" \
        $PY -u "$HERE/bench_driver.py" $MOE_ARGS > "$f" 2>&1)
    rc=$?
    log "${name}_r$r rc=$rc $(grep -hoE '\[moe-exp\] (idle|busy):.*' "$f" | head -1 | cut -c10-) |" \
        "$(grep -hoE '(quant_a1|gemm[12]): (profiler|graph)_us = [0-9.]+' "$f" | tr '\n' ' ')"
    [ $rc = 99 ] && { log "GPU busy inside the lock, stopping"; exit 1; }
  done
done
log DONE
PYTHONPATH=$PYPATH $PY "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
