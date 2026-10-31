#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=$1
SERVE_ARGS=$2
MODEL_ARGS=$3
PORT=${PORT:-8000}
THRESHOLD=${THRESHOLD:-0.90}
OUT=${OUT:-/tmp/vllm-nightly-accuracy}
mkdir -p "$OUT"

MODEL=$MODEL_ID
if [[ -d "/models/$MODEL_ID" ]]; then
  MODEL="/models/$MODEL_ID"
fi

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" $SERVE_ARGS \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 7200); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$OUT/server.log"
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

lm_eval --model local-completions \
  --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/completions,$MODEL_ARGS" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --log_samples \
  --output_path "$OUT/results" 2>&1 | tee "$OUT/lm_eval.log"

python3 - "$OUT/results" "$THRESHOLD" <<'PY'
import glob
import json
import sys

paths = glob.glob(f"{sys.argv[1]}/**/*.json", recursive=True)
for path in sorted(paths, reverse=True):
    data = json.load(open(path))
    result = data.get("results", {}).get("gsm8k")
    if result and "exact_match,flexible-extract" in result:
        score = result["exact_match,flexible-extract"]
        threshold = float(sys.argv[2])
        print(f"GSM8K_FLEXIBLE_EXTRACT={score}")
        if score < threshold:
            raise SystemExit(f"gsm8k score {score} is below {threshold}")
        break
else:
    raise SystemExit("gsm8k flexible-extract score not found")
PY
