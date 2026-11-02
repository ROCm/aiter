#!/usr/bin/env bash
set -euo pipefail
model=${1:?Flash or Pro}; mode=${2:?log prefix}; logfile=${3:?server log}
for attempt in $(seq 1 600); do
 if grep -q 'Server started successfully and ready to accept requests' "$logfile" && curl -sf http://127.0.0.1:8000/v1/models | grep -q "DeepSeek-V4-$model"; then
  docker exec -w /work -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 python /work/gfx1250-integration/client_smoke.py "$model" "$mode"
  exit
 fi
 if grep -Eq 'EXECUTOR_FAILED|ProcessExitedException|EngineCore failed' "$logfile"; then tail -60 "$logfile"; exit 1; fi
 sleep 1
done
echo 'Timed out waiting for server readiness' >&2
exit 1
