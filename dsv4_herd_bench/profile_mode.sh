#!/bin/bash
# Usage (inside gcy_atom): bash profile_mode.sh <mode> [model]
# Starts the TP1 server with a torch profiler dir, drives 64 concurrent
# ISL=OSL=1024 requests, records PROFILE_S seconds of steady bs=64 decode via
# /start_profile + /stop_profile, then stops the server.
set -uo pipefail
MODE="$1"
MODEL_PATH="${2:-/data/DeepSeek-V4-Flash}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/profiles/$MODE}"
PROFILE_S="${PROFILE_S:-0.3}"
DECODE_WAIT_S="${DECODE_WAIT_S:-4}"
rm -rf "$OUT" && mkdir -p "$OUT"
LOG="$OUT/driver.log"

cd /app/scripts/dsv4
PROFILE_DIR="$OUT/trace" MOE_MODE="$MODE" setsid bash "$HERE/serve_dsv4_tp1_herd.sh" "$MODEL_PATH" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
stop_server() {
  kill -TERM -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null
  echo "[prof] $(date) server stopped" | tee -a "$LOG"
}
trap stop_server EXIT

until curl -sf http://localhost:8000/health > /dev/null; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[prof] server died" | tee -a "$LOG"; exit 1; }
  sleep 10
done
echo "[prof] $(date) server healthy" | tee -a "$LOG"

PREFILLS_BEFORE=$(grep -c "Scheduled prefill batch" "$OUT/server.log")
python -m atom.benchmarks.benchmark_serving --model="$MODEL_PATH/" --backend=vllm \
  --base-url=http://localhost:8000 --dataset-name=random --random-input-len=1024 \
  --random-output-len=1024 --random-range-ratio 1 --num-prompts=64 --max-concurrency=64 \
  --request-rate=inf --ignore-eos --num-warmups 0 > "$OUT/load.log" 2>&1 &
LOAD_PID=$!

# 64 x 1024-token prompts at max_num_batched_tokens=16384 are 4 prefill
# batches; once the 4th is scheduled every step is a bs=64 decode.
until (( $(grep -c "Scheduled prefill batch" "$OUT/server.log") - PREFILLS_BEFORE >= 4 )); do
  kill -0 "$LOAD_PID" 2>/dev/null || { echo "[prof] load exited early" | tee -a "$LOG"; exit 1; }
  sleep 0.2
done
grep "Scheduled prefill batch" "$OUT/server.log" | tail -4 | cut -c1-90 | tee -a "$LOG"
sleep "$DECODE_WAIT_S"
echo "[prof] $(date +%T.%N) start_profile" | tee -a "$LOG"
curl -sf -X POST http://localhost:8000/start_profile | tee -a "$LOG"; echo >> "$LOG"
sleep "$PROFILE_S"
curl -sf -X POST http://localhost:8000/stop_profile | tee -a "$LOG"; echo >> "$LOG"
echo "[prof] $(date +%T.%N) stop_profile" | tee -a "$LOG"
wait "$LOAD_PID"
grep -E "Output token throughput|Mean TPOT" "$OUT/load.log" | tee -a "$LOG"
# Trace export happens asynchronously in the model runner.
for _ in $(seq 120); do ls "$OUT"/trace/*/*.json.gz > /dev/null 2>&1 && break; sleep 2; done
ls -la "$OUT"/trace/*/ | tee -a "$LOG"
