#!/bin/bash
# Self-contained Kimi-K2.5 SGLang (upstream native kimi_k25) gsm8k accuracy check.
# Runs inside an official ROCm SGLang image (lmsysorg/sglang-rocm:...) with the
# PR's AITER already installed. Launches sglang.launch_server, runs lm_eval
# gsm8k 3-shot, prints `KIMI_FLEX_EXTRACT=<value>`.
#
# NOTE: SGLang currently fails to load AMD Quark MXFP4 per-expert weights
# (loader bug). This gate is wired non-blocking until that is fixed upstream.
set -uo pipefail

MODEL_PATH=${MODEL_PATH:-amd/Kimi-K2.5-MXFP4}
TP=${TP:-8}
PORT=${PORT:-8000}
FEWSHOT=${FEWSHOT:-3}
OUT=${OUT:-/tmp/kimi_sglang_acc}
mkdir -p "$OUT"
SLOG="$OUT/server.log"

export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}
export AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}
export SGLANG_AITER_FP8_PREFILL_ATTN=${SGLANG_AITER_FP8_PREFILL_ATTN:-0}

if [ -d "/models/${MODEL_PATH}" ]; then MODEL="/models/${MODEL_PATH}"; else MODEL="${MODEL_PATH}"; fi
echo "== Kimi SGLang accuracy =="; echo "model=$MODEL tp=$TP"
pip show amd-aiter 2>/dev/null | grep -E '^Version' || pip show aiter 2>/dev/null | grep -E '^Version' || true
command -v lm_eval >/dev/null 2>&1 || pip install -q "lm-eval[api]"

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then echo "ERROR: port $PORT busy"; exit 3; fi

echo "== launching sglang.launch_server =="
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    > "$SLOG" 2>&1 &
SVPID=$!

echo "== waiting for /v1/models =="
ready=0
for i in $(seq 1 2400); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
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
pkill -9 -f "sglang.launch_server" 2>/dev/null || true

rf=$(find "$OUT/acc" -name '*.json' 2>/dev/null | head -1)
if [ -z "$rf" ]; then echo "ERROR: no lm_eval result JSON"; exit 6; fi
val=$(python3 -c "import json;d=json.load(open('$rf'));print(d['results']['gsm8k']['exact_match,flexible-extract'])")
echo "KIMI_FLEX_EXTRACT=${val}"
