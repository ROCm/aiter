# Preserve per-experiment compile cache across launches.
MODEL_PATH="${1:-/data/models/DeepSeek-V4-Pro}"
# TRACE=1: collect torch profiler traces. Has to go through the CLI flag:
# EngineArgs.torch_profiler_dir defaults to a hard None that _get_engine_kwargs
# forwards into the Config, overriding the field's ATOM_TORCH_PROFILER_DIR
# default_factory -- so the env var alone silently traces nothing. Kept off the
# / overlay, which fills up; each rank writes its own subdir and traces get
# large fast.
TRACE="${TRACE:-0}"
TRACE_DIR="${TRACE_DIR:-/app/traces}"
TRACE_ARGS=()
TRITON_MOE="${TRITON_MOE:-1}"
ATOM_USE_TRITON_MOE=$TRITON_MOE
ATOM_USE_TRITON_MOE_DECODE=$TRITON_MOE
ATOM_USE_TRITON_MOE_A4W4=$TRITON_MOE
FAKE_EPLB="${FAKE_EPLB:-1}"
FAKE_EPLB_ARGS=()
if [ "$FAKE_EPLB" = "1" ]; then
  FAKE_EPLB_ARGS=(--fake-eplb --hf-overrides '{"num_hash_layers": 0}')
fi
if [ "$TRACE" = "1" ]; then
  mkdir -p "$TRACE_DIR"
  TRACE_ARGS=(--torch-profiler-dir "$TRACE_DIR" --mark-trace)
  echo "[trace] torch profiler dir: $TRACE_DIR (client needs --profile)"
fi
# Preserve per-experiment compile cache across launches.
MODEL_PATH="${1:-/data/DeepSeek-V4-Pro}"
SERVER_ENV=(
  "MEGA_COMBINE_WIRE=${MEGA_COMBINE_WIRE:-bf16}"
  HSA_ENABLE_SDMA=1
  HSA_USE_SVM=1
  HSA_XNACK=1
  AITER_DISABLE_KERNARG_PRELOAD=0
  "ATOM_USE_TRITON_MOE=$ATOM_USE_TRITON_MOE"
  "ATOM_USE_TRITON_MOE_DECODE=$ATOM_USE_TRITON_MOE_DECODE"
  "ATOM_USE_TRITON_MOE_A4W4=$ATOM_USE_TRITON_MOE_A4W4"
  ATOM_USE_UNIFIED_ATTN=1
  "ATOM_FORCE_ATTN_TRITON=$ATOM_FORCE_ATTN_TRITON"
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
  ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
  AITER_CUSTOM_AR_MIN_SIZE=131072
  ATOM_MOE_GU_ITLV=1
  ATOM_GC_THRESHOLD=20000,50,50
  AITER_CUSTOM_AR_MAX_SIZE=8388608
  # gfx1250: RCCL 2.30.4 cannot capture all_to_all_single into a hipGraph,
  # and the all2all default takes that path. allgather goes through aiter's
  # custom IPC all-gather instead, which captures.
  ATOM_DP_LM_HEAD_MODE=allgather
)
if [ "$TRACE" = "1" ]; then
  SERVER_ENV+=(
    DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0
    DEBUG_HIP_GRAPH_CLASSIC_PATH=1
  )
fi
env "${SERVER_ENV[@]}" \
  python -m atom.entrypoints.openai_server -tp 4 --model "$MODEL_PATH" --kv_cache_dtype bf16  --index-cache-dtype fp4 --no-enable_prefix_caching --disable_uvicorn_access_log --max-num-seqs 128 --cudagraph-capture-sizes "[1,2,4,8,16,32,64,128]" "${FAKE_EPLB_ARGS[@]}" "${TRACE_ARGS[@]}" --enforce-eager 2>&1 | tee server.log
