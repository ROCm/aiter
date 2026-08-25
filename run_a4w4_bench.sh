#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_SCRIPT="${SCRIPT_DIR}/op_tests/test_flydsl_grouped_gemm_gfx1250.py"

# Add test cases as: "model_dim inter_dim experts tokens".
readonly TEST_CASES=(
  "16384 3072 96 128"
  "7168 3072 96 128"
  "7168 3072 96 512"
  # "32768 3072 96 512"
)

run_test() {
  local model_dim="$1"
  local inter_dim="$2"
  local experts="$3"
  local tokens="$4"

  printf \
    '\nRunning model_dim=%s inter_dim=%s experts=%s tokens=%s\n' \
    "${model_dim}" "${inter_dim}" "${experts}" "${tokens}"

  local -a test_env=(
    "AITER_USE_GROUPED_GEMM=1"
    "AITER_GROUPED_DEBUG=0"
    "ENABLE_CK=0"
    "AITER_LOG_MORE=1"
    "AITER_MOE_EXPERT_BALANCE=true"
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1"
  )
  env "${test_env[@]}" python3 -u "${TEST_SCRIPT}" \
    --scenario kernel \
    --data-format a4w4 \
    --experts "${experts}" \
    --tokens "${tokens}" \
    --topk 6 \
    --model-dim "${model_dim}" \
    --inter-dim "${inter_dim}" \
    --act silu \
    --no-bias \
    --no-check-aot-cache
}

main() {
  local test_case model_dim inter_dim experts tokens

  for test_case in "${TEST_CASES[@]}"; do
    read -r model_dim inter_dim experts tokens <<<"${test_case}"
    run_test "${model_dim}" "${inter_dim}" "${experts}" "${tokens}"
  done
}

main "$@"
