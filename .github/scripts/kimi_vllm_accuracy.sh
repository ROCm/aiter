#!/bin/bash
# Self-contained Kimi-K2.5 vLLM (upstream) gsm8k accuracy check.
# Runs inside an official ROCm vLLM image (e.g. rocm/vllm-dev:nightly) with the
# PR's AITER already installed. Launches `vllm serve`, runs lm_eval gsm8k 3-shot,
# and prints `KIMI_FLEX_EXTRACT=<value>` for the workflow to gate on.
set -uo pipefail

MODEL_PATH=${MODEL_PATH:-amd/Kimi-K2.5-MXFP4}
TP=${TP:-8}
PORT=${PORT:-8000}
FEWSHOT=${FEWSHOT:-3}
OUT=${OUT:-/tmp/kimi_vllm_acc}
mkdir -p "$OUT"
SLOG="$OUT/server.log"

export AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}

# Prefer a /models mount if the weights are pre-cached there.
if [ -d "/models/${MODEL_PATH}" ]; then MODEL="/models/${MODEL_PATH}"; else MODEL="${MODEL_PATH}"; fi
echo "== Kimi vLLM accuracy =="; echo "model=$MODEL tp=$TP"
python3 -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null | tail -1
pip show amd-aiter 2>/dev/null | grep -E '^Version' || true
command -v lm_eval >/dev/null 2>&1 || pip install -q "lm-eval[api]"

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then echo "ERROR: port $PORT busy"; exit 3; fi

echo "== launching vllm serve =="
# NOTE: do NOT pass `--load-format fastsafetensors` here. The official
# rocm/vllm-dev:nightly image does not bundle the `fastsafetensors` package, so
# that flag makes every TP worker die during weight load with
# `ImportError: Please install vllm[fastsafetensors] for fastsafetensors support`
# (surfaced only as "WorkerProc initialization failed" in the parent). The
# default safetensors loader loads this 521GB checkpoint in ~27s/worker on
# MI355X, so fastsafetensors buys nothing here. The reference 0.9409 run used it
# only because the ATOM image happened to ship the package.
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

echo "== waiting for /health =="
ready=0
for i in $(seq 1 2400); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SVPID 2>/dev/null || { echo "SERVER DIED"; tail -n 80 "$SLOG"; exit 4; }
  sleep 1
done
[ "$ready" -eq 1 ] || { echo "SERVER NOT READY"; tail -n 80 "$SLOG"; exit 5; }
echo "SERVER READY after ${i}s"

echo "== lm_eval gsm8k ${FEWSHOT}-shot =="
lm_eval --model local-completions \
  --model_args model="$MODEL",base_url="http://127.0.0.1:${PORT}/v1/completions",num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k --num_fewshot "$FEWSHOT" \
  --output_path "$OUT/acc" 2>&1 | tee "$OUT/accuracy.txt"

kill $SVPID 2>/dev/null || true; sleep 3
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true

rf=$(ls -1t "$OUT"/acc/**/*.json "$OUT"/acc/*.json 2>/dev/null | head -1)
[ -z "$rf" ] && rf=$(find "$OUT/acc" -name '*.json' 2>/dev/null | head -1)
if [ -z "$rf" ]; then echo "ERROR: no lm_eval result JSON"; exit 6; fi
val=$(python3 -c "import json,sys;d=json.load(open('$rf'));print(d['results']['gsm8k']['exact_match,flexible-extract'])")
echo "KIMI_FLEX_EXTRACT=${val}"
