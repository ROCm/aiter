#!/bin/bash
# wait_ready.sh PORT [TIMEOUT_S]
PORT=${1:-8243}; T=${2:-1500}; s=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
  (( $(date +%s) - s > T )) && { echo "not ready after ${T}s"; exit 1; }; sleep 5; done
echo "ready after $(( $(date +%s) - s ))s"
