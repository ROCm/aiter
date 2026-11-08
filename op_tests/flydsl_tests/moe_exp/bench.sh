#!/bin/bash
# Kernel timing of the FlyDSL grouped MoE kernels on gfx1250: gemm1 of the DSR1 prefill MoE by default (2048 tokens,
# 64 experts, top 8, model dim 7168, inter dim 2048), weights rotated over 4 copies so every launch reads them from HBM.
#
# usage: op_tests/flydsl_tests/moe_exp/bench.sh [-p PROCS] [-r ROUNDS] [-d DISPATCHES] [--no-profiler] [-o OUT] [VARIANT ...]
#   -p PROCS        timed processes per variant (default 5)
#   -r ROUNDS       rounds per process (default 50)
#   -d DISPATCHES   dispatches per round (default 128; a multiple of MOE_EXP_ROTATE)
#   --no-profiler   run the timed processes with no profiler (default: rocprofv3 --kernel-trace)
#   -o OUT          output directory (default: moe_exp_runs/<UTC stamp> in the current directory)
#   VARIANT         name[:ENV=VAL,ENV=VAL,...] (default: base); the env is added on top of the settings below
#
# Each variant first gets one untimed compile process (p0: no profiler, one round) that fills the variant's FlyDSL
# cache. The timed processes p1..pPROCS then load the kernels from that cache (FLYDSL_RUNTIME_RUN_ONLY=1), so nothing
# compiles under the profiler. Timed processes run interleaved: p1: a b c, p2: a b c, ...
# A process is: build inputs, one MoE run (the accuracy check's output), warm-up, one CUDA graph of DISPATCHES launches,
# 1 s idle, then ROUNDS rounds; a round is one replay of the graph between two CUDA events, then a synchronize
# (bench_driver.py). The graph keeps the host's launch cost (about 34 us per gemm1 launch) out of the rounds.
#
# env:
#   PY             python with torch, AITER and FlyDSL (default: python3)
#   SITE           a pip --target directory holding AITER and FlyDSL, prepended to PYTHONPATH (default: none)
#   ROCM_PATH      the ROCm FlyDSL links with and AITER JIT-builds with (default /opt/rocm); with the ROCm Python wheels
#                  set it to $(rocm-sdk path --root)
#   TRACE_ROCPROF  rocprofv3 to use (default: the one next to PY)
#   LOCK           command prefix for every GPU process, e.g. LOCK=gpu-lock on a shared node (default: none)
#   IDLE_CHECK=1   inside LOCK, before each process, wait up to 20 s for 0% GPU use and less than IDLE_VRAM_MB
#                  (default 1024) of VRAM in use, else stop
#   MOE_EXP_TOKENS (default 2048), MOE_EXP_KERNELS (quant_a1, gemm1, gemm2 or all; default gemm1),
#   MOE_EXP_ROTATE (default 4), MOE_EXP_ROTATE_A (1 = also rotate A; default 0)
#   MOE_ARGS       the test's arguments (default: the shape above)
#   DUMP=1         first run one process per variant that dumps FlyDSL's IR and final ISA to OUT/dump-<variant>/
#   COMPILE=0      skip the compile processes p0; every variant then needs MOE_EXP_FCACHE pointing at a filled cache
# variant env read here:
#   MOE_EXP_FCACHE=<dir>  the variant's FlyDSL cache (default OUT/cache-<variant>), e.g. one holding a patched kernel
#   MOE_EXP_AGT=<ms>      a power-management log at that period during each process (with_agt.sh; agt_internal, sudo)
#
# Writes OUT/meta.txt, OUT/run.log, OUT/<variant>_p<k>.log, OUT/trace-<variant>_p<k>/ (rocprofv3 kernel trace),
# OUT/out/<variant>_p<k>.pt (output) and .pt.rounds.json (host timestamps of every round), OUT/summary.txt.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
PROCS=5; ROUNDS=50; DISPATCHES=128; PROFILE=1; OUT=""
while [ $# -gt 0 ]; do
  case $1 in
    -p) PROCS=$2; shift 2 ;;
    -r) ROUNDS=$2; shift 2 ;;
    -d) DISPATCHES=$2; shift 2 ;;
    -o) OUT=$2; shift 2 ;;
    --no-profiler) PROFILE=0; shift ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
    -*) echo "unknown option $1 (see --help)"; exit 2 ;;
    *) break ;;
  esac
done
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(base)
PY=${PY:-python3}; LOCK=${LOCK:-}
OUT=${OUT:-moe_exp_runs/$(date -u +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/out" && OUT=$(cd "$OUT" && pwd)
export MOE_EXP_KERNELS=${MOE_EXP_KERNELS:-gemm1} MOE_EXP_ROTATE=${MOE_EXP_ROTATE:-4} MOE_EXP_ROTATE_A=${MOE_EXP_ROTATE_A:-0}
MOE_EXP_TOKENS=${MOE_EXP_TOKENS:-2048}
MOE_ARGS=${MOE_ARGS:-"--scenario kernel --data-format a4w4 --experts 64 --tokens $MOE_EXP_TOKENS --topk 8 --model-dim 7168
  --inter-dim 2048 --act silu --no-bias --iters 128 --no-check-aot-cache --data-init uniform --scale-init uniform"}
# run_a4w4_conc6144.sh's environment (the A layout comes from the config CSV)
BASE_ENV=(AITER_FLYDSL_MMA_FIRST_GROUP=4 AITER_FLYDSL_MMA_GROUP=4 AITER_FLYDSL_DS_FIRST_N=8
          AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0
          AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
          AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}   # FlyDSL 0.3.3.dev882 finds ld.lld only through ROCM_PATH
ROCPROF=${TRACE_ROCPROF:-$(dirname "$(command -v "$PY")")/rocprofv3}
PYPATH=${SITE:+$SITE:}${PYTHONPATH:-}
LOG=$OUT/run.log
log() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }

# Idle check, run inside LOCK so nothing can start between the check and the run. Exit 99 = busy.
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
  echo "rocm path  $ROCM_PATH"
  AMD_SMI=${AMD_SMI:-amd-smi}
  if command -v "$AMD_SMI" >/dev/null; then
    echo "firmware   $($AMD_SMI firmware 2>/dev/null | grep -A1 -E 'FW_ID: (CP_MEC1|PM)$' | grep -oE '(CP_MEC1|PM|FW_VERSION: .*)' | paste -sd' ')"
    echo "clock      $($AMD_SMI metric -g 0 --clock 2>/dev/null | grep -m1 MAX_CLK | tr -s ' ')"
    echo "power cap  $($AMD_SMI static -g 0 --limit 2>/dev/null | grep -E 'SOCKET_POWER_LIMIT' | tr -s ' ' | paste -sd' ')"
  fi
  echo "variants   ${VARIANTS[*]}"
  echo "processes  $PROCS per variant $([ "${COMPILE:-1}" = 1 ] && echo "(+ compile process p0)" || echo "(no compile process: caches from MOE_EXP_FCACHE)")"
  echo "rounds     $ROUNDS x $DISPATCHES dispatches"
  echo "profiler   $([ $PROFILE = 1 ] && echo "rocprofv3 --kernel-trace ($ROCPROF)" || echo none)"
  echo "args       $(echo $MOE_ARGS)"
  echo "kernels    $MOE_EXP_KERNELS"
  if [ "$MOE_EXP_ROTATE" -gt 1 ]; then
    echo "rotation   $MOE_EXP_ROTATE copies of W + W scales$([ "$MOE_EXP_ROTATE_A" = 1 ] && echo ' + A + A scales')"
  else echo "rotation   none"; fi
  echo "base env   ${BASE_ENV[*]}"
} > "$OUT/meta.txt"
cat "$OUT/meta.txt" | tee -a "$LOG"

# run <variant spec> <process tag> <mode: dump | compile | timed>
run() {
  local spec=$1 tag=$2 mode=$3 name=${1%%:*} extra=() cache pre=() prof=() agt=() p f rc
  [ "$spec" != "$name" ] && IFS=',' read -r -a extra <<< "${spec#*:}"
  cache=$OUT/cache-$name
  for e in "${extra[@]}"; do case $e in MOE_EXP_FCACHE=*) cache=${e#MOE_EXP_FCACHE=} ;; esac; done
  [ "${IDLE_CHECK:-0}" = 1 ] && pre=(bash -c "$IDLE_SH" _)
  local env=(FLYDSL_DUMP_IR=0 FLYDSL_RUNTIME_CACHE_DIR="$cache" MOE_EXP_ROUNDS=$ROUNDS MOE_EXP_DISPATCHES=$DISPATCHES)
  case $mode in
    dump)    env=(FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$OUT/dump-$name" MOE_EXP_ROUNDS=1 MOE_EXP_DISPATCHES=$DISPATCHES) ;;
    compile) env=(FLYDSL_DUMP_IR=0 FLYDSL_RUNTIME_CACHE_DIR="$cache" MOE_EXP_ROUNDS=1 MOE_EXP_DISPATCHES=$DISPATCHES) ;;
    timed)   env+=(FLYDSL_RUNTIME_RUN_ONLY=1)
             [ $PROFILE = 1 ] && prof=("$ROCPROF" --kernel-trace --output-format csv -d "$OUT/trace-${name}_$tag" -o trace --) ;;
  esac
  case ",${spec#*:}," in *,MOE_EXP_AGT=*)
    p=${spec#*MOE_EXP_AGT=}; agt=("$HERE/with_agt.sh" "$OUT/pm-${name}_$tag.csv" "${p%%,*}") ;;
  esac
  log "${name}_$tag ($mode) queued ${extra[*]}"
  f=$OUT/${name}_$tag.log
  (cd "$REPO" && $LOCK "${pre[@]}" "${agt[@]}" env "${BASE_ENV[@]}" "${extra[@]}" "${env[@]}" PYTHONPATH="$PYPATH" \
      MOE_EXP_SAVE_OUT="$OUT/out/${name}_$tag.pt" \
      "${prof[@]}" $PY -u "$HERE/bench_driver.py" $MOE_ARGS > "$f" 2>&1)
  rc=$?
  log "${name}_$tag rc=$rc $(grep -hoE '\[moe-exp\] (idle|busy):.*' "$f" | head -1 | cut -c10-) |" \
      "$(grep -hoE '(quant_a1|gemm[12]): event_us = [0-9.]+' "$f" | tr '\n' ' ')"
  [ $rc = 99 ] && { log "GPU busy inside the lock, stopping"; exit 1; }
  # an ablated kernel fails the accuracy check (rc 1) after its timing; the timing line is what must be there
  grep -q 'event_us = ' "$f" || { log "${name}_$tag: no event_us line, stopping (see $f)"; exit 1; }
}

if [ "${DUMP:-0}" = 1 ]; then
  for spec in "${VARIANTS[@]}"; do run "$spec" dump dump; done
fi
[ "${COMPILE:-1}" = 1 ] && for spec in "${VARIANTS[@]}"; do run "$spec" p0 compile; done
for k in $(seq 1 "$PROCS"); do
  for spec in "${VARIANTS[@]}"; do run "$spec" "p$k" timed; done
done
log DONE
PYTHONPATH=$PYPATH $PY "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
