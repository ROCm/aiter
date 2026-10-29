#!/usr/bin/env bash
# Wait for 4 idle GPUs (two idle checks 60 s apart), then run the given test
# through gpu_run.sh on them; a run past TIMEOUT is killed (container too).
# Usage: ./gpu_when_free.sh LOG -- torchrun --nproc-per-node=4 -m ...
set -uo pipefail
LOG=$1; shift; [ "$1" = "--" ] && shift
mkdir -p "$(dirname "$LOG")"
HERE=$(cd "$(dirname "$0")" && pwd)
TIMEOUT=${TIMEOUT:-3600}
idle=$("$HERE/wait_gpus.sh" 4 | sed -n 's/^IDLE GPUs (rocm-smi idx): \([0-9,]*\) at.*/\1/p')
cards=$(echo "$idle" | tr ',' '\n' | head -4 | paste -sd,)
echo "$(date) idle=$idle cards=$cards" | tee "$LOG"
rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]' | tee -a "$LOG"
name=vigokhal-dsv4g-$$
CARDS=$cards timeout "$TIMEOUT" "$HERE/gpu_run.sh" "$@" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 124 ]; then
  echo "TIMEOUT after $TIMEOUT s" >> "$LOG"
  docker ps --format '{{.Names}}' | grep '^vigokhal-dsv4g-' | xargs -r docker kill >> "$LOG" 2>&1
fi
echo "$(date) rc=$rc" >> "$LOG"
exit $rc
