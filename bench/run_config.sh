#!/bin/bash
# One config: start server -> cold run1 -> prewarm -> run2..run4 -> stop server.
#
# Usage: [AITER_DIR=..] [ATOM_DIR=..] [MODEL=..] [REQUIRE_LOG=..] [FORBID_LOG=..] run_config.sh <result_dir> <serve_script>
#   result_dir   created if missing; refuses to run if it already holds run1/
#   serve_script a path, or a file name in this directory
#   AITER_DIR    aiter repo put on PYTHONPATH (default: the opt aiter)
#   ATOM_DIR     ATOM repo put in front of it (default: none, i.e. /app/ATOM)
#   REQUIRE_LOG / FORBID_LOG  grep -E pattern that must / must not be in server.log once the server is up
set -u
BENCH=$(cd "$(dirname "$0")" && pwd)
R=$1
SERVE=$2
[ -f "$SERVE" ] || SERVE=$BENCH/$SERVE
AITER_DIR=${AITER_DIR:-/data/binding/moe_1031/opt/aiter}
ATOM_DIR=${ATOM_DIR:-}
MODEL=${MODEL:-/data/models/DeepSeek-V4-Flash}

if [ -e "$R/run1" ]; then
  echo "$R already holds results; pick a new result dir"
  exit 1
fi
if curl -sf http://localhost:8000/health > /dev/null; then
  echo "a server is already listening on :8000; stop it first"
  exit 1
fi
mkdir -p "$R"
R=$(cd "$R" && pwd)
NAME=$(basename "$R")
exec > >(tee -a "$R/run_config.log") 2>&1
log() { echo "[$(date +%T)] $NAME: $*"; }

# The container sets PYTHONPATH=/app/aiter, which would shadow the pip-installed aiter.
export PYTHONPATH=${ATOM_DIR:+$ATOM_DIR:}$AITER_DIR
ATOM_ROOT=$(cd /tmp && python -c 'import atom, os; print(os.path.dirname(os.path.dirname(atom.__file__)))' 2>/dev/null | tail -1)
log "serve_script=$SERVE model=$MODEL"
log "aiter=$(cd /tmp && python -c 'import aiter; print(aiter.__file__)' 2>/dev/null | tail -1) commit=$(git -C "$AITER_DIR" rev-parse --short HEAD 2>/dev/null)"
log "atom=$ATOM_ROOT commit=$(git -C "$ATOM_ROOT" rev-parse --short HEAD 2>/dev/null) moe_py_modified=$(git -C "$ATOM_ROOT" diff --quiet -- atom/model_ops/moe.py 2>/dev/null && echo no || echo yes)"
log "gfx=$(cd /tmp && python -c 'from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())' 2>/dev/null | tail -1)"
log "versions: $(pip list 2>/dev/null | grep -E '^(amd-aiter|flydsl|triton|torch) ' | tr -s ' ' | tr '\n' ';')"

stop_server() {
  pkill -INT -f "^python -m atom.entrypoints.openai_server"
  for _ in $(seq 1 60); do
    sleep 2
    if ! pgrep -f "^ATOM::" > /dev/null && ! pgrep -f "^python -m atom.entrypoints.openai_server" > /dev/null; then break; fi
  done
  log "server stopped, vram_used=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)"
}

cd /app/scripts/dsv4
FLYDSL_RUNTIME_CACHE_DIR=$R/flydsl_cache bash "$SERVE" "$MODEL" > "$R/server.log" 2>&1 &
SPID=$!
log "server launched"
ready=0
for _ in $(seq 1 360); do
  if curl -sf http://localhost:8000/health > /dev/null; then ready=1; break; fi
  if ! kill -0 $SPID 2>/dev/null; then break; fi
  sleep 5
done
if [ $ready -ne 1 ]; then
  log "server failed to start"
  tail -n 40 "$R/server.log"
  stop_server
  exit 1
fi
log "server ready"

if [ -n "${REQUIRE_LOG:-}" ]; then
  if ! grep -Eq "$REQUIRE_LOG" "$R/server.log"; then
    log "required server log '$REQUIRE_LOG' not found, aborting"
    stop_server
    exit 2
  fi
  log "check ok: '$REQUIRE_LOG' found $(grep -Ec "$REQUIRE_LOG" "$R/server.log") times"
fi
if [ -n "${FORBID_LOG:-}" ] && grep -Eq "$FORBID_LOG" "$R/server.log"; then
  log "forbidden server log '$FORBID_LOG' found, aborting"
  stop_server
  exit 2
fi

bench() {
  mkdir -p "$R/$1"
  (cd "$R/$1" && bash "$BENCH/bench_dsv4.sh" "$MODEL" > bench.log 2>&1)
  log "$1: $(grep -E 'Output token throughput|Mean TTFT|Mean TPOT' "$R/$1/bench.log" | tr -s ' ' | tr '\n' ';')"
}

bench run1
(cd "$R" && bash "$BENCH/prewarm.sh" "$MODEL" > prewarm.log 2>&1)
log "prewarm done"
touch "$R/.clean_start"
for run in run2 run3 run4; do bench $run; done
log "new FlyDSL kernels during clean runs: $(find "$R/flydsl_cache" -mindepth 1 -maxdepth 1 -newer "$R/.clean_start" | wc -l)"
log "new Triton cache entries during clean runs: $(find ~/.triton/cache -mindepth 1 -maxdepth 1 -newer "$R/.clean_start" | wc -l)"
stop_server
