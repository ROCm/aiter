#!/bin/bash
# Self-contained Kimi-K2.5 SGLang (upstream native kimi_k25) throughput PERFORMANCE gate.
# Runs inside an official ROCm SGLang image (lmsysorg/sglang-rocm:...) with the
# PR's AITER already installed. Launches sglang.launch_server with the EXACT flags
# that pass the accuracy gate (kimi_sglang_accuracy.sh), warms up, then runs a
# throughput sweep over concurrency and prints
# `KIMI_PERF_C64_OUT_TOKS=<output token throughput at concurrency 64>` for the
# workflow to gate on.
#
# The launch flags MUST match kimi_sglang_accuracy.sh exactly. Both flags below
# work around upstream-SGLang bugs (not AITER) when serving the Quark MXFP4
# checkpoint on MI350X (gfx950):
#
# 1) --disable-shared-experts-fusion
#    With fusion ON (SGLang default for DeepseekV3-style models) the MoE weight
#    loader crashes during load (w13 shape 3584 vs 7168 mismatch): the fused
#    shared-expert gate/up does not match the per-expert Quark MXFP4 shard shape.
#
# 2) --attention-backend aiter   (and DO NOT use --kv-cache-dtype fp8_e4m3)
#    The default SGLang triton/MLA path is broken on gfx950 (head-dim mismatch at
#    CUDA-graph capture). With fp8 KV cache it instead dies in decode on the
#    triton fp8 dot ("Unsupported lhs dtype fp8e4nv"). AITER MLA + bf16 KV cache
#    serves cleanly.
set -uo pipefail

MODEL_PATH=${MODEL_PATH:-amd/Kimi-K2.5-MXFP4}
TP=${TP:-8}
PORT=${PORT:-8000}
ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONCURRENCIES=${CONCURRENCIES:-"4 8 16 32 64"}
OUT=${OUT:-/tmp/kimi_sglang_perf}
mkdir -p "$OUT"
SLOG="$OUT/server.log"

export SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}
export AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}
export SGLANG_AITER_FP8_PREFILL_ATTN=${SGLANG_AITER_FP8_PREFILL_ATTN:-0}

# Prefer a /models mount if the weights are pre-cached there, else use the path
# as given (e.g. a local /data mount).
if [ -d "/models/${MODEL_PATH}" ]; then MODEL="/models/${MODEL_PATH}"; else MODEL="${MODEL_PATH}"; fi
echo "== Kimi SGLang perf =="; echo "model=$MODEL tp=$TP isl=$ISL osl=$OSL concurrencies=[$CONCURRENCIES]"
pip show amd-aiter 2>/dev/null | grep -E '^Version' || pip show aiter 2>/dev/null | grep -E '^Version' || true

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then echo "ERROR: port $PORT busy"; exit 3; fi

echo "== launching sglang.launch_server =="
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --attention-backend aiter \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    > "$SLOG" 2>&1 &
SVPID=$!

cleanup() {
  kill "$SVPID" 2>/dev/null || true; sleep 3
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "sglang::scheduler" 2>/dev/null || true
}
trap cleanup EXIT

echo "== waiting for /v1/models =="
ready=0
for i in $(seq 1 2400); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SVPID 2>/dev/null || { echo "SERVER DIED"; tail -n 80 "$SLOG"; exit 4; }
  sleep 1
done
[ "$ready" -eq 1 ] || { echo "SERVER NOT READY"; tail -n 80 "$SLOG"; exit 5; }
echo "SERVER READY after ${i}s"

run_sweep_point() {
  local C="$1"
  local jf="$OUT/bench_c${C}.jsonl"
  rm -f "$jf"
  echo "== bench_serving concurrency=${C} prompts=$((C*10)) =="
  python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts $((C*10)) \
    --max-concurrency "$C" \
    --request-rate inf \
    --output-file "$jf" \
    2>&1 | tee "$OUT/bench_c${C}.log"
}

# Warmup: a small dedicated bench pass before measuring.
echo "== warmup =="
python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random \
  --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
  --num-prompts 16 --max-concurrency 8 --request-rate inf \
  > "$OUT/warmup.log" 2>&1 || echo "warmup pass returned non-zero (continuing)"

C64_OUT="NA"
echo ""
printf "%-12s %-22s %-16s %-16s\n" "concurrency" "out_tok/s" "mean_ttft_ms" "mean_tpot_ms"
RESULTS=""
for C in $CONCURRENCIES; do
  run_sweep_point "$C"
  jf="$OUT/bench_c${C}.jsonl"
  # sglang.bench_serving --output-file appends one JSON object per line; take the
  # last line for this run. Console labels map to JSON keys:
  #   "Output token throughput (tok/s)" -> output_throughput
  #   "Mean TTFT (ms)"                  -> mean_ttft_ms
  #   "Mean TPOT (ms)"                  -> mean_tpot_ms
  read -r OUTTOK TTFT TPOT < <(python3 - "$jf" "$OUT/bench_c${C}.log" <<'PY'
import json,sys,re
jf, logf = sys.argv[1], sys.argv[2]
d=None
try:
    lines=[l for l in open(jf) if l.strip()]
    if lines: d=json.loads(lines[-1])
except Exception:
    d=None
def from_json():
    def g(k):
        return d[k] if (d and k in d and d[k] is not None) else None
    return g('output_throughput'), g('mean_ttft_ms'), g('mean_tpot_ms')
ot,tt,tp=from_json()
# Fallback: parse the console log if JSON missing/incomplete.
if ot is None or tt is None or tp is None:
    txt=open(logf,errors='ignore').read()
    def grab(label):
        m=re.search(re.escape(label)+r'\s*:\s*([0-9.]+)', txt)
        return float(m.group(1)) if m else None
    ot = ot if ot is not None else grab('Output token throughput (tok/s)')
    tt = tt if tt is not None else grab('Mean TTFT (ms)')
    tp = tp if tp is not None else grab('Mean TPOT (ms)')
def f(x): return 'NA' if x is None else x
print(f(ot), f(tt), f(tp))
PY
)
  [ -z "$OUTTOK" ] && OUTTOK="NA"
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
