#!/usr/bin/env bash
# Validate DSv4-Flash optimization stack: gsm8k + perf sweep c=4..64.
#
# Usage:
#   MODEL_PATH=/path/to/DeepSeek-V4-Flash ./validate.sh [LABEL]
#
# Starts the server with the full env stack, runs smoke + gsm8k + best-of-2
# perf sweep, kills the server, prints summary. Writes artifacts under
# ./out/<LABEL>/.
#
# Assumes patches from apply.sh are applied, and the active python environment
# has atom + aiter installed (editable) along with lm_eval.
set -uo pipefail
ulimit -c 0

LABEL="${1:-dsv4_validate_$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the DeepSeek-V4-Flash checkpoint}"
PORT="${PORT:-8000}"
OUT="./out/$LABEL"
mkdir -p "$OUT/perf_sweep"

# Critical env vars (read HANDOFF.md "Canonical server start" for what each does).
export ATOM_V4_USE_TRITON_FUSION=1
ENV_STACK="ATOM_USE_TRITON_MOE=1 \
ATOM_MOE_BACKEND=a8w4 \
ATOM_A8W4_SWIGLU_FOLD=1 \
AITER_A8W4_DECODE_NUM_STAGES=2 \
ATOM_A8W4_TRITON_ROUTING=1 \
ATOM_A8W4_GEMM1_MX_EMIT=1 \
ATOM_A8W4_FUSE_RESIDUAL=1 \
ATOM_FP8_BLOCKSCALE_USE_MXFP8=1 \
ATOM_A8W4_HASH_FAST_ROUTING=1 \
AITER_LOG_LEVEL=WARNING"

cleanup() {
  pkill -9 -f atom.entrypoints 2>/dev/null || true
  pkill -9 -f spawn_main 2>/dev/null || true
  pkill -9 -f benchmark_serving 2>/dev/null || true
  sleep 15
}

echo "[validate] killing leftover processes..."
cleanup
rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | head -1 || true

echo "[validate] starting server with stack: $ENV_STACK"
eval "$ENV_STACK nohup python -m atom.entrypoints.openai_server \
  --model '$MODEL_PATH' --kv_cache_dtype fp8 -tp 8 \
  --max-num-seqs 64 --gpu-memory-utilization 0.85 --server-port $PORT --max-model-len 4096 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,64,128,256]' --level 0 \
  > '$OUT/server.log' 2>&1 &"
SPID=$!

echo "[validate] waiting for server up..."
for i in $(seq 1 300); do
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then
    echo "[validate] SERVER DIED. Check $OUT/server.log"
    tail -30 "$OUT/server.log"
    exit 1
  fi
  sleep 5
done

echo "[validate] SMOKE test (must say Paris)..."
SMOKE=$(curl -s -X POST "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"The capital of France is\",\"max_tokens\":15,\"temperature\":0}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'])")
echo "[validate] SMOKE: $SMOKE"
if [[ "$SMOKE" != *Paris* ]]; then
  echo "[validate] FAIL: smoke didn't say Paris. ABORT."
  cleanup; exit 1
fi

echo "[validate] running gsm8k (n=100)..."
lm_eval --model local-completions \
  --model_args "model=$MODEL_PATH,base_url=http://localhost:$PORT/v1/completions,num_concurrent=16,max_retries=2,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 3 --limit 100 \
  --output_path "$OUT/lm_eval.json" 2>&1 | tee "$OUT/lm_eval.log" | grep -E "strict-match|flexible|exact_match"

echo "[validate] running perf sweep c=4..64 (best-of-2)..."
# Warmup at each c so subsequent measurements are stable
for c in 4 8 16 32 64; do
  python -m atom.benchmarks.benchmark_serving \
    --model="$MODEL_PATH" --backend=vllm --base-url="http://localhost:$PORT" \
    --dataset-name=random --random-input-len=1024 --random-output-len=1024 \
    --num-prompts=$((c*2)) --max-concurrency=$c --request-rate=inf --ignore-eos \
    > /dev/null 2>&1 || true
done

for c in 4 8 16 32 64; do
  for run in 1 2; do
    RDIR="$OUT/perf_sweep/c${c}_r${run}"; mkdir -p "$RDIR"
    echo "[$(date +%H:%M:%S)] c=$c run=$run"
    python -m atom.benchmarks.benchmark_serving \
      --model="$MODEL_PATH" --backend=vllm --base-url="http://localhost:$PORT" \
      --dataset-name=random --random-input-len=1024 --random-output-len=1024 \
      --num-prompts=$((c*8)) --max-concurrency=$c \
      --request-rate=inf --ignore-eos \
      --save-result --result-dir "$RDIR" > "$RDIR/bench.log" 2>&1
  done
done

echo ""
echo "=== SUMMARY ==="
printf "%4s | %10s | %10s | %8s\n" "c" "r1 TPS" "r2 TPS" "var%"
for c in 4 8 16 32 64; do
  r1=$(ls "$OUT/perf_sweep/c${c}_r1"/vllm-*.json 2>/dev/null | head -1)
  r2=$(ls "$OUT/perf_sweep/c${c}_r2"/vllm-*.json 2>/dev/null | head -1)
  python3 -c "
import json
def g(p):
    try: return json.load(open(p)) if p else None
    except: return None
a,b=g('$r1'),g('$r2')
if a and b:
    print(f'  {$c:>2d} | {a[\"output_throughput\"]:>10.2f} | {b[\"output_throughput\"]:>10.2f} | {(b[\"output_throughput\"]-a[\"output_throughput\"])/a[\"output_throughput\"]*100:>+7.3f}%')
"
done | tee "$OUT/SUMMARY.txt"

cleanup
echo "[validate] DONE — artifacts under $OUT"
