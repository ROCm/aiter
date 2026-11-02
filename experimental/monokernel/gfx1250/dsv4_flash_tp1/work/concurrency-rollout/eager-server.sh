export PYTHONPATH=/app/aiter:/app/ATOM:${PYTHONPATH:-}
 # Preserve compilation caches for paired experiments.
# TRACE=1: collect torch profiler traces. Has to go through the CLI flag:
# EngineArgs.torch_profiler_dir defaults to a hard None that _get_engine_kwargs
# forwards into the Config, overriding the field's ATOM_TORCH_PROFILER_DIR
# default_factory -- so the env var alone silently traces nothing. Kept off the
# / overlay, which fills up; each rank writes its own subdir and traces get
# large fast.
TRACE="${TRACE:-0}"
TRACE_DIR="${TRACE_DIR:-/app/traces}"
TRACE_ARGS=()
FAKE_EPLB="${FAKE_EPLB:-0}"
FAKE_EPLB_ARGS=()
if [ "$FAKE_EPLB" = "1" ]; then
  FAKE_EPLB_ARGS=(--fake-eplb --hf-overrides '{"num_hash_layers": 0}')
fi
if [ "$TRACE" = "1" ]; then
  mkdir -p "$TRACE_DIR"
  TRACE_ARGS=(--torch-profiler-dir "$TRACE_DIR" --mark-trace)
  echo "[trace] torch profiler dir: $TRACE_DIR (client needs --profile)"
fi
 # Preserve compilation caches for paired experiments.
MODEL_PATH="${1:-/data/DeepSeek-V4-Flash}"
SERVER_ENV=(
  HSA_ENABLE_SDMA=1
  HSA_USE_SVM=1
  HSA_XNACK=1
  AITER_DISABLE_KERNARG_PRELOAD=0
  ATOM_USE_TRITON_MOE=1
  ATOM_USE_UNIFIED_ATTN=1
  ATOM_FORCE_ATTN_TRITON=1
  ATOM_LOADER_USE_THREADPOOL=1
  AITER_ROPE_TRITON_BACKEND=1
  ATOM_ENABLE_DS_QKNORM_FUSION=0
  ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0
  ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0
  ATOM_USE_TRITON_GEMM=1
  ENABLE_CK=0
  AITER_USE_OPUS_MOE_SORTING=1
  AITER_FORCE_A8W4=0
  AITER_USE_GROUPED_GEMM=1
  ATOM_USE_AITER_TRITON_ATTN=1
  AITER_TRITON_USE_HERD=1
  ATOM_USE_TRITON_MOE_DECODE=1
  ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
  AITER_CUSTOM_AR_MIN_SIZE=131072
  ATOM_MOE_GU_ITLV=1
  ATOM_GC_THRESHOLD=20000,50,50
)
if [ "$TRACE" = "1" ]; then
  SERVER_ENV+=(
    DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0
    DEBUG_HIP_GRAPH_CLASSIC_PATH=1
  )
fi
env "${SERVER_ENV[@]}" \
  python -m atom.entrypoints.openai_server -tp 1 --model "$MODEL_PATH" --kv_cache_dtype bf16 --index-cache-dtype fp8 --no-enable_prefix_caching --disable_uvicorn_access_log --max-model-len 2048 --max-num-seqs 8 --enforce-eager --level 0 "${FAKE_EPLB_ARGS[@]}" "${TRACE_ARGS[@]}" --port "${PORT:-8001}" --server-port "${HTTP_PORT:-8000}" --gpu-memory-utilization 0.85 2>&1 | tee "${SERVER_LOG:-/work/modelwide/baseline-server.log}"
