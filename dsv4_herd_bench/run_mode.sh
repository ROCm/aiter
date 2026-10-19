#!/bin/bash
# Usage (inside gcy_atom): [TASKS=bench|eval|bench,eval] bash run_mode.sh <triton_herd|flydsl_herd> [model] [conc]
# Starts the TP1 server for MOE_MODE, waits for /health, runs bench_dsv4.sh
# and/or eval_dsv4.sh, then stops the server.
# Logs go to $LOG_DIR/<mode>_<tasks>.log and $LOG_DIR/<mode>_<tasks>_server.log.
set -uo pipefail
MODE="$1"
MODEL_PATH="${2:-/data/DeepSeek-V4-Flash}"
CONC="${3:-64}"
TASKS="${TASKS:-bench}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-$HERE/logs}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"
mkdir -p "$LOG_DIR"
TAG="${TASKS//,/_}"
SERVER_LOG="$LOG_DIR/${MODE}_${TAG}_server.log"
BENCH_LOG="$LOG_DIR/${MODE}_${TAG}.log"

cd /app/scripts/dsv4
echo "[run] $(date) mode=$MODE model=$MODEL_PATH conc=$CONC" | tee "$BENCH_LOG"
MOE_MODE="$MODE" setsid bash "$HERE/serve_dsv4_tp1_herd.sh" "$MODEL_PATH" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[run] server pgid=$SERVER_PID" | tee -a "$BENCH_LOG"

stop_server() {
  kill -TERM -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2
  done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null
  echo "[run] $(date) server stopped" | tee -a "$BENCH_LOG"
}
trap stop_server EXIT

start=$(date +%s)
until curl -sf "http://localhost:8000/health" > /dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] $(date) server exited before becoming healthy" | tee -a "$BENCH_LOG"
    exit 1
  fi
  if (( $(date +%s) - start > STARTUP_TIMEOUT_S )); then
    echo "[run] $(date) server not healthy after ${STARTUP_TIMEOUT_S}s" | tee -a "$BENCH_LOG"
    exit 1
  fi
  sleep 15
done
echo "[run] $(date) server healthy after $(( $(date +%s) - start ))s" | tee -a "$BENCH_LOG"

# Greedy sanity answers before the timed run; 32 copies so decode runs at a
# batch inside the HERD window (16..128) rather than at bs=1.
python3 - "$MODEL_PATH" <<'EOF' 2>&1 | tee -a "$BENCH_LOG"
import concurrent.futures, json, sys, urllib.request
model = sys.argv[1]
prompts = [
    "What is 17 * 23? Reply with just the number.",
    "Name the capital of France in one word.",
    "Janet has 3 apples and buys 5 more, then eats 2. How many apples does she have? Reply with just the number.",
    "Translate 'good morning' into Spanish.",
] * 8
def ask(p):
    body = {"model": model, "messages": [{"role": "user", "content": p}],
            "max_tokens": 256, "temperature": 0}
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.load(r)["choices"][0]["message"]
    return (msg.get("content") or "").strip().replace("\n", " ")[-120:]
with concurrent.futures.ThreadPoolExecutor(32) as ex:
    answers = list(ex.map(ask, prompts))
for p, a in list(zip(prompts, answers))[:4]:
    print(f"[sanity] {p!r} -> {a!r}")
print(f"[sanity] {len(set(answers))} distinct answers over {len(answers)} requests")
EOF

if [[ ",$TASKS," == *",bench,"* ]]; then
  bash bench_dsv4.sh "${BENCH_MODEL:-$MODEL_PATH/}" "$CONC" 2>&1 | tee -a "$BENCH_LOG"
  echo "[run] $(date) bench exit=${PIPESTATUS[0]} (save_csv.py is absent, so a non-zero exit after the results is expected)" | tee -a "$BENCH_LOG"
fi
if [[ ",$TASKS," == *",eval,"* ]]; then
  bash eval_dsv4.sh "${BENCH_MODEL:-$MODEL_PATH/}" 2>&1 | tee -a "$BENCH_LOG"
  echo "[run] $(date) eval exit=${PIPESTATUS[0]}" | tee -a "$BENCH_LOG"
fi
