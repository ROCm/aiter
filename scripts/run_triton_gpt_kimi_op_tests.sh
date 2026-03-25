#!/usr/bin/env bash
# Run Triton op tests for GPT OSS 120B and Kimi K2.5 / DeepSeek R1 style kernels.
# Intended to be launched from the repo root, e.g.:
#   cd ~/&& ./scripts/run_triton_gpt_kimi_op_tests.sh

set -uo pipefail

LOG_DIR="$(pwd)/pytest_logs"
mkdir -p "$LOG_DIR"

failed=0

run_one() {
  local idx="$1"
  local label="$2"
  local test_path="$3"
  local stem
  stem="$(basename "$test_path" .py)"

  echo -e "\n\n========== [$idx/15] Running: $label ==========\n"
  if ! pytest -s -v "$test_path" 2>&1 | tee "$LOG_DIR/${stem}.output"; then
    failed=$((failed + 1))
  fi
}

# ── GPT OSS 120B ──
run_one 1 "test_moe_gemm_a8w4 (GPT OSS 120B)" "op_tests/triton_tests/moe/test_moe_gemm_a8w4.py"
run_one 2 "test_unified_attention (GPT OSS 120B)" "op_tests/triton_tests/attention/test_unified_attention.py"
run_one 3 "test_gemm_a16w16 (GPT OSS 120B)" "op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py"
run_one 4 "test_fused_add_rmsnorm_pad (GPT OSS 120B)" "op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py"
run_one 5 "test_fused_kv_cache (GPT OSS 120B)" "op_tests/triton_tests/fusions/test_fused_kv_cache.py"
run_one 6 "test_rmsnorm (GPT OSS 120B)" "op_tests/triton_tests/normalization/test_rmsnorm.py"
run_one 7 "test_rope (GPT OSS 120B)" "op_tests/triton_tests/rope/test_rope.py"

# ── Kimi K2.5 / DeepSeek R1 (NVFP4 → MXFP4) ──
run_one 8 "test_moe_mx (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/moe/test_moe_mx.py"
run_one 9 "test_mha (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/attention/test_mha.py"
run_one 10 "test_unified_attention_sparse_mla (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py"
run_one 11 "test_gemm_afp4wfp4 (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py"
run_one 12 "test_batched_gemm_afp4wfp4 (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py"
run_one 13 "test_fused_mxfp4_quant (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/quant/test_fused_mxfp4_quant.py"
run_one 14 "test_fused_gemm_afp4wfp4_a16w16 (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py"
run_one 15 "test_quant_mxfp4 (Kimi K2.5 / DeepSeek R1)" "op_tests/triton_tests/quant/test_quant_mxfp4.py"

echo -e "\033[32m=== All tests finished ===\033[0m"
echo "Test logs directory (cwd): $LOG_DIR"
echo

if [[ "$failed" -gt 0 ]]; then
  echo -e "\033[31m$failed test suite(s) failed.\033[0m" >&2
  exit 1
fi
