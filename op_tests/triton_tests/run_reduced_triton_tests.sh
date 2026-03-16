#!/usr/bin/env bash
# Run this script inside satya_container after: source /ffm/ffmlite_env.sh
# Usage: bash run_reduced_triton_tests.sh [aiter_root]
# Example: cd /path/to/aiter && bash op_tests/triton_tests/run_reduced_triton_tests.sh /path/to/aiter

set -e
AITER_ROOT="${1:-$(pwd)}"
cd "$AITER_ROOT"
export PYTHONPATH="$AITER_ROOT:$PYTHONPATH"

LEDGER="triton_reduced_test_ledger.txt"
echo "Pytest command ledger (reduced triton_tests)" > "$LEDGER"
echo "============================================" >> "$LEDGER"
n=0
run_pytest() {
  n=$((n+1))
  cmd="$1"
  targets="$2"
  echo "" >> "$LEDGER"
  echo "$n. Command: $cmd" >> "$LEDGER"
  echo "   Targets: $targets" >> "$LEDGER"
  if eval "$cmd" >> "$LEDGER" 2>&1; then
    echo "   Result: PASS" >> "$LEDGER"
    echo "   Exit code: 0" >> "$LEDGER"
    echo "   Summary: passed" >> "$LEDGER"
  else
    ec=$?
    echo "   Result: FAIL" >> "$LEDGER"
    echo "   Exit code: $ec" >> "$LEDGER"
    echo "   Summary: failed (exit $ec)" >> "$LEDGER"
  fi
}

run_pytest "python3 -m pytest op_tests/triton_tests/moe/test_moe_gemm_a4w4.py -v --tb=line -q" "test_moe_gemm_a4w4.py"
run_pytest "python3 -m pytest op_tests/triton_tests/attention/test_unified_attention.py -v --tb=line -q" "test_unified_attention.py"
run_pytest "python3 -m pytest op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py -v --tb=line -q" "test_gemm_a16w16.py"
run_pytest "python3 -m pytest op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py -v --tb=line -q" "test_fused_add_rmsnorm_pad.py"
run_pytest "python3 -m pytest op_tests/triton_tests/fusions/test_fused_kv_cache.py -v --tb=line -q" "test_fused_kv_cache.py"
run_pytest "python3 -m pytest op_tests/triton_tests/normalization/test_rmsnorm.py -v --tb=line -q" "test_rmsnorm.py"
run_pytest "python3 -m pytest op_tests/triton_tests/rope/test_rope.py -v --tb=line -q" "test_rope.py"
run_pytest "python3 -m pytest op_tests/triton_tests/moe/test_moe_mx.py -v --tb=line -q" "test_moe_mx.py"
run_pytest "python3 -m pytest op_tests/triton_tests/attention/test_mha.py -v --tb=line -q" "test_mha.py"
run_pytest "python3 -m pytest op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py -v --tb=line -q" "test_unified_attention_sparse_mla.py"
run_pytest "python3 -m pytest op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py -v --tb=line -q" "test_gemm_afp4wfp4.py"
run_pytest "python3 -m pytest op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py -v --tb=line -q" "test_batched_gemm_afp4wfp4.py"
run_pytest "python3 -m pytest op_tests/triton_tests/quant/test_fused_mxfp4_quant.py -v --tb=line -q" "test_fused_mxfp4_quant.py"
run_pytest "python3 -m pytest op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py -v --tb=line -q" "test_fused_gemm_afp4wfp4_a16w16.py"
run_pytest "python3 -m pytest op_tests/triton_tests/quant/test_quant_mxfp4.py -v --tb=line -q" "test_quant_mxfp4.py"
run_pytest "python3 -m pytest op_tests/triton_tests/attention/test_fp8_mqa_logits.py -v --tb=line -q" "test_fp8_mqa_logits.py"
run_pytest "python3 -m pytest op_tests/triton_tests/attention/test_la_paged.py -v --tb=line -q" "test_la_paged.py"
run_pytest "python3 -m pytest op_tests/triton_tests/attention/test_mla_decode_rope.py -v --tb=line -q" "test_mla_decode_rope.py"
run_pytest "python3 -m pytest op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py -v --tb=line -q" "test_fused_bmm_rope_kv_cache.py"

echo "" >> "$LEDGER"
echo "Total pytest commands run: $n" >> "$LEDGER"
echo "Ledger written to $AITER_ROOT/$LEDGER"
cat "$LEDGER"
