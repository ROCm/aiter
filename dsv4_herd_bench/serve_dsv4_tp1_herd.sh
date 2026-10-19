#!/bin/bash
# serve_dsv4.sh at -tp 1, with the routed-expert backend picked by MOE_MODE:
#   triton_herd: FlyDSL prefill + Triton/gluon decode, Triton HERD routing
#   flydsl_herd: FlyDSL prefill + decode, FlyDSL HERD routing (decode only)
#   triton:      FlyDSL prefill + Triton/gluon decode, stock routing
#   flydsl:      FlyDSL prefill + decode, stock routing
# ATOM_USE_TRITON_MOE=1 is required for ATOM_USE_TRITON_MOE_DECODE to take
# effect at all (serve_dsv4.sh sets it to 0, which leaves decode on FlyDSL).
# ATOM_USE_TRITON_MOE=1 with ATOM_USE_TRITON_MOE_DECODE=0 would run Triton on
# both phases, so the FlyDSL mode turns ATOM_USE_TRITON_MOE off instead.
MODEL_PATH="${1:-/data/DeepSeek-V4-Flash}"
MOE_MODE="${MOE_MODE:-triton_herd}"
PORT="${PORT:-8000}"

case "$MOE_MODE" in
  triton_herd)
    MODE_ENV=(ATOM_USE_TRITON_MOE=1 ATOM_USE_TRITON_MOE_DECODE=1 AITER_TRITON_USE_HERD=1 AITER_FLYDSL_USE_HERD=0)
    ;;
  triton)
    MODE_ENV=(ATOM_USE_TRITON_MOE=1 ATOM_USE_TRITON_MOE_DECODE=1 AITER_TRITON_USE_HERD=0 AITER_FLYDSL_USE_HERD=0)
    ;;
  flydsl_herd)
    MODE_ENV=(ATOM_USE_TRITON_MOE=0 ATOM_USE_TRITON_MOE_DECODE=0 AITER_TRITON_USE_HERD=0 AITER_FLYDSL_USE_HERD=1)
    ;;
  flydsl)
    MODE_ENV=(ATOM_USE_TRITON_MOE=0 ATOM_USE_TRITON_MOE_DECODE=0 AITER_TRITON_USE_HERD=0 AITER_FLYDSL_USE_HERD=0)
    ;;
  *)
    echo "unknown MOE_MODE=$MOE_MODE" >&2
    exit 1
    ;;
esac

PROFILE_ARGS=()
if [ -n "${PROFILE_DIR:-}" ]; then
  mkdir -p "$PROFILE_DIR"
  PROFILE_ARGS=(--torch-profiler-dir "$PROFILE_DIR")
  # Segment-scheduled graph replays report spread-out kernel times; the classic
  # path launches graph nodes one by one so each kernel gets its own timestamps.
  MODE_ENV+=(DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0 DEBUG_HIP_GRAPH_CLASSIC_PATH=1)
fi

rm -rf ~/.cache/atom
echo "[serve] MOE_MODE=$MOE_MODE ${MODE_ENV[*]} ${PROFILE_ARGS[*]}"
env \
  HSA_ENABLE_SDMA=1 \
  HSA_USE_SVM=1 \
  HSA_XNACK=1 \
  AITER_DISABLE_KERNARG_PRELOAD=0 \
  ATOM_USE_UNIFIED_ATTN=1 \
  ATOM_FORCE_ATTN_TRITON=1 \
  ATOM_LOADER_USE_THREADPOOL=1 \
  AITER_ROPE_TRITON_BACKEND=1 \
  ATOM_ENABLE_DS_QKNORM_FUSION=0 \
  ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0 \
  ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0 \
  ATOM_USE_TRITON_GEMM=1 \
  ENABLE_CK=0 \
  AITER_USE_OPUS_MOE_SORTING=1 \
  AITER_FORCE_A8W4=1 \
  AITER_USE_GROUPED_GEMM=1 \
  ATOM_USE_AITER_TRITON_ATTN=1 \
  ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 \
  AITER_CUSTOM_AR_MIN_SIZE=131072 \
  ATOM_MOE_GU_ITLV=1 \
  ATOM_GC_THRESHOLD=20000,50,50 \
  "${MODE_ENV[@]}" \
  python -m atom.entrypoints.openai_server -tp 1 --model "$MODEL_PATH" --kv_cache_dtype bf16 --no-enable_prefix_caching --server-port "$PORT" "${PROFILE_ARGS[@]}" ${SERVE_EXTRA_ARGS:-}
