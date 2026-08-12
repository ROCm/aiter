#!/bin/bash
# Self-contained Kimi-K2.5 vLLM (upstream) throughput PERFORMANCE gate.
# Runs inside an official ROCm vLLM image (e.g. rocm/vllm-dev:nightly) with the
# PR's AITER already installed. Launches `vllm serve` (the exact flags that pass
# the accuracy gate), warms up, then runs a throughput sweep over concurrency
# and prints `KIMI_PERF_C64_OUT_TOKS=<output token throughput at concurrency 64>`
# for the workflow to gate on.
set -uo pipefail

MODEL_PATH=${MODEL_PATH:-amd/Kimi-K2.5-MXFP4}
TP=${TP:-8}
PORT=${PORT:-8000}
ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONCURRENCIES=${CONCURRENCIES:-"4 8 16 32 64"}
OUT=${OUT:-/tmp/kimi_vllm_perf}
mkdir -p "$OUT"
SLOG="$OUT/server.log"

export AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}

# Prefer a /models mount if the weights are pre-cached there.
if [ -d "/models/${MODEL_PATH}" ]; then MODEL="/models/${MODEL_PATH}"; else MODEL="${MODEL_PATH}"; fi
echo "== Kimi vLLM perf =="; echo "model=$MODEL tp=$TP isl=$ISL osl=$OSL concurrencies=[$CONCURRENCIES]"
python3 -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null | tail -1
pip show amd-aiter 2>/dev/null | grep -E '^Version' || true

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then echo "ERROR: port $PORT busy"; exit 3; fi

echo "== launching vllm serve =="
# Same launch as the accuracy gate. Do NOT pass `--load-format fastsafetensors`:
# the official rocm/vllm-dev:nightly image does not bundle the package, so the
# flag kills every TP worker during weight load.
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --async-scheduling \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    > "$SLOG" 2>&1 &
SVPID=$!

cleanup() {
  kill "$SVPID" 2>/dev/null || true; sleep 3
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT

echo "== waiting for /health =="
ready=0
for i in $(seq 1 2400); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SVPID 2>/dev/null || { echo "SERVER DIED"; tail -n 80 "$SLOG"; exit 4; }
  sleep 1
done
[ "$ready" -eq 1 ] || { echo "SERVER NOT READY"; tail -n 80 "$SLOG"; exit 5; }
echo "SERVER READY after ${i}s"

BASE_URL="http://127.0.0.1:${PORT}"

run_sweep_point() {
  local C="$1"
  local jf="$OUT/bench_c${C}.json"
  echo "== bench serve concurrency=${C} prompts=$((C*10)) =="
  vllm bench serve \
    --backend vllm \
    --base-url "$BASE_URL" \
    --endpoint /v1/completions \
    --model "$MODEL" \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --max-concurrency "$C" \
    --num-prompts $((C*10)) \
    --num-warmups 8 \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename "$jf" \
    2>&1 | tee "$OUT/bench_c${C}.log"
}

# Warmup: a small dedicated bench pass before measuring.
echo "== warmup =="
vllm bench serve --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions \
  --model "$MODEL" --trust-remote-code --dataset-name random \
  --random-input-len "$ISL" --random-output-len "$OSL" \
  --max-concurrency 8 --num-prompts 16 --num-warmups 8 \
  --request-rate inf --ignore-eos --disable-tqdm \
  > "$OUT/warmup.log" 2>&1 || echo "warmup pass returned non-zero (continuing)"

C64_OUT="NA"
echo ""
printf "%-12s %-22s %-16s %-16s\n" "concurrency" "out_tok/s" "mean_ttft_ms" "mean_tpot_ms"
RESULTS=""
for C in $CONCURRENCIES; do
  run_sweep_point "$C"
  jf="$OUT/bench_c${C}.json"
  if [ -f "$jf" ]; then
    read -r OUTTOK TTFT TPOT < <(python3 - "$jf" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
def g(*keys):
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return float('nan')
print(g('output_throughput'), g('mean_ttft_ms'), g('mean_tpot_ms'))
PY
)
  else
    OUTTOK="NA"; TTFT="NA"; TPOT="NA"
  fi
  printf "%-12s %-22s %-16s %-16s\n" "$C" "$OUTTOK" "$TTFT" "$TPOT"
  RESULTS="${RESULTS}${C} ${OUTTOK} ${TTFT} ${TPOT}\n"
  if [ "$C" = "64" ]; then C64_OUT="$OUTTOK"; fi
done

echo ""
echo "== SWEEP SUMMARY (concurrency out_tok/s mean_ttft_ms mean_tpot_ms) =="
printf "%b" "$RESULTS"
echo ""
echo "KIMI_PERF_C64_OUT_TOKS=${C64_OUT}"

cleanup
trap - EXIT
sleep 2
echo "== done =="
