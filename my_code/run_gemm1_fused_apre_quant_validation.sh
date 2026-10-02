#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HEAD="c0d98474c1a2bc937b07930d35221d7bfc8dd713"
MODE="${1:-compare-random}"
SEED="${SEED:-0}"

if [[ ! -f my_code/verify_gemm1_fused_apre_quant.py ]]; then
  echo "run this script from the aiter repository root" >&2
  exit 2
fi

actual_head="$(git rev-parse HEAD)"
if [[ "${actual_head}" != "${EXPECTED_HEAD}" ]]; then
  echo "HEAD mismatch: expected=${EXPECTED_HEAD}, actual=${actual_head}" >&2
  exit 2
fi

export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1

shape=(
  --experts 64
  --tokens 1536
  --topk 8
  --model-dim 7168
  --inter-dim 2048
)

case "${MODE}" in
  compare-random)
    python3 -u my_code/verify_gemm1_fused_apre_quant.py \
      "${shape[@]}" --seed "${SEED}"
    ;;
  compare-const0)
    python3 -u my_code/verify_gemm1_fused_apre_quant.py \
      "${shape[@]}" --const-init 0
    ;;
  e2e-random)
    python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
      --scenario bench \
      --data-format a4w4 \
      "${shape[@]}" \
      --act silu \
      --no-bias \
      --no-check-aot-cache \
      --iters 1
    ;;
  perf-const0)
    # Run this mode only after the host-level GPU/KFD idle check passes.
    python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
      --scenario bench \
      --data-format a4w4 \
      "${shape[@]}" \
      --act silu \
      --no-bias \
      --no-check-aot-cache \
      --iters "${ITERS:-20}" \
      --const-init 0
    ;;
  baseline-const0)
    # Same working tree and timing path, but force the original GEMM1 plus
    # standalone quant pipeline for an interleaved hardware-state control.
    AITER_FLYDSL_GEMM1_FUSED_QUANT=0 \
      python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
        --scenario bench \
        --data-format a4w4 \
        "${shape[@]}" \
        --act silu \
        --no-bias \
        --no-check-aot-cache \
        --iters "${ITERS:-20}" \
        --const-init 0
    ;;
  ab-const0)
    python3 -u my_code/benchmark_gemm1_fused_apre_quant_ab.py \
      --rounds "${ROUNDS:-3}" \
      --iters "${ITERS:-100}"
    ;;
  *)
    echo "usage: $0 {compare-random|compare-const0|e2e-random|perf-const0|baseline-const0|ab-const0}" >&2
    exit 2
    ;;
esac
