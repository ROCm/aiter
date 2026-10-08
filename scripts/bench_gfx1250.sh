#!/usr/bin/env bash
# gfx1250 kernel benchmark suite
# Runs the full set of Gluon gfx1250 benchmarks from op_benchmarks/triton and
# collects stdout/stderr per test into $RESULTS_DIR.  After all tests finish a
# summary report is printed (and written to $RESULTS_DIR/report.txt).
#
# Usage:
#   ./scripts/bench_gfx1250.sh              # run everything
#   ./scripts/bench_gfx1250.sh --dry-run    # print commands without executing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="op_tests/op_benchmarks/triton"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/bench_results_gfx1250_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$RESULTS_DIR"

PASS=0
FAIL=0
SKIP=0
declare -a SUMMARY_LINES=()

run_bench() {
    local tag="$1"
    local script="$2"
    shift 2
    local args=("$@")

    local logfile="${RESULTS_DIR}/${tag}.log"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] python -m ${BENCH_DIR//\//.}.${script%.py} ${args[*]}"
        return
    fi

    local module="${BENCH_DIR//\//.}.${script%.py}"
    echo "────────────────────────────────────────────────────────────"
    echo "[$tag] python -m $module ${args[*]}"
    echo "────────────────────────────────────────────────────────────"

    local start end elapsed status
    start=$(date +%s)
    if (cd "$REPO_ROOT" && python -m "$module" "${args[@]}") > "$logfile" 2>&1; then
        status="PASS"
        ((PASS++)) || true
    else
        status="FAIL"
        ((FAIL++)) || true
    fi
    end=$(date +%s)
    elapsed=$((end - start))

    echo "  -> $status (${elapsed}s)  log: $logfile"
    SUMMARY_LINES+=("$(printf '%-60s %-6s %4ds  %s' "$tag" "$status" "$elapsed" "$logfile")")
}

skip_bench() {
    local tag="$1"
    local reason="$2"
    echo "[SKIP] $tag — $reason"
    ((SKIP++)) || true
    SUMMARY_LINES+=("$(printf '%-60s %-6s %4s  %s' "$tag" "SKIP" "-" "$reason")")
}

# ---------------------------------------------------------------------------
# Gluon warp-pipeline GEMM (gemm_warp_pipeline_cdna5.py)
# This benchmark is not shipped in the aiter repo; it ships with the Gluon
# compiler / Triton test suite.  If available on PYTHONPATH it will run.
# ---------------------------------------------------------------------------
GLUON_WP="gemm_warp_pipeline_cdna5"
if python -c "import importlib; importlib.import_module('$GLUON_WP')" 2>/dev/null; then
    HAS_GLUON_WP=1
else
    HAS_GLUON_WP=0
fi

if [[ $HAS_GLUON_WP -eq 1 ]]; then
    run_bench "gemm_wp_bf16_4096x4096x65536" \
        "$GLUON_WP.py" \
        --dtype bf16 -M 4096 -N 4096 -K 65536 --check --benchmark

    run_bench "gemm_wp_mxfp8_4096x4096x65536" \
        "$GLUON_WP.py" \
        --dtype mxfp8 -M 4096 -N 4096 -K 65536 --mxfp8-tile 256x256x256 --mxfp8-cluster 2x2 --check --benchmark

    run_bench "gemm_wp_mxfp4_4096x4096x65536" \
        "$GLUON_WP.py" \
        --dtype mxfp4 -M 4096 -N 4096 -K 65536 --check --benchmark

    run_bench "gemm_wp_fp8_mxfp4_4096x4096x65536" \
        "$GLUON_WP.py" \
        --dtype fp8_mxfp4 -M 4096 -N 4096 -K 65536 --check --benchmark
else
    skip_bench "gemm_wp_bf16_4096x4096x65536"          "gemm_warp_pipeline_cdna5 not found"
    skip_bench "gemm_wp_mxfp8_4096x4096x65536"         "gemm_warp_pipeline_cdna5 not found"
    skip_bench "gemm_wp_mxfp4_4096x4096x65536"         "gemm_warp_pipeline_cdna5 not found"
    skip_bench "gemm_wp_fp8_mxfp4_4096x4096x65536"     "gemm_warp_pipeline_cdna5 not found"
fi

# ---------------------------------------------------------------------------
# GEMM — bench_gemm_a16w16
# ---------------------------------------------------------------------------
run_bench "gemm_a16w16_8192x6144x6144" \
    bench_gemm_a16w16.py \
    --shape 8192 6144 6144

run_bench "gemm_a16w16_8192x5120x4096" \
    bench_gemm_a16w16.py \
    --shape 8192 5120 4096

run_bench "gemm_a16w16_256x5120x4096" \
    bench_gemm_a16w16.py \
    --shape 256 5120 4096

run_bench "gemm_a16w16_persistent_8192x5120x2880" \
    bench_gemm_a16w16.py \
    --persistent --shape 8192 5120 2880

# ---------------------------------------------------------------------------
# GEMM — bench_gemm_a8w8_blockscale
# ---------------------------------------------------------------------------
run_bench "gemm_a8w8_bs_preshuffle_4096x2048x7168" \
    bench_gemm_a8w8_blockscale.py \
    -preshuffle --shape 4096 2048 7168

run_bench "gemm_a8w8_bs_preshuffle_16x2048x7168" \
    bench_gemm_a8w8_blockscale.py \
    -preshuffle --shape 16 2048 7168

# ---------------------------------------------------------------------------
# GEMM — bench_gemm_afp4wfp4 (mxfp4 preshuffled)
# ---------------------------------------------------------------------------
run_bench "gemm_afp4wfp4_preshuffle_4096x7168x16384" \
    bench_gemm_afp4wfp4.py \
    --preshuffle --shape 4096 7168 16384

# ---------------------------------------------------------------------------
# GEMM — bench_gemm_afp8wfp8 (mxfp8 preshuffled)
# ---------------------------------------------------------------------------
run_bench "gemm_afp8wfp8_preshuffle_4096x7168x16384" \
    bench_gemm_afp8wfp8.py \
    --preshuffle --shape 4096 7168 16384

# ---------------------------------------------------------------------------
# Batched GEMM — bench_batched_gemm_bf16
# ---------------------------------------------------------------------------
run_bench "batched_gemm_bf16_B16_2048x1024x4096" \
    bench_batched_gemm_bf16.py \
    --shape 16 2048 1024 4096

# ---------------------------------------------------------------------------
# Batched GEMM — bench_batched_gemm_a8w8 (MLA absorbed BMM)
# ---------------------------------------------------------------------------
run_bench "batched_gemm_a8w8_B128_1024x512x128" \
    bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py \
    --no-bias --backend gluon --shape 128 1024 512 128

# ---------------------------------------------------------------------------
# MoE — bench_moe_gemm_a8w4_cudagraph
# ---------------------------------------------------------------------------
run_bench "moe_a8w4_7168x4096_E256T8_M2048" \
    bench_moe_gemm_a8w4_cudagraph.py \
    --shape 7168 4096 --experts 256 8 --M 2048

run_bench "moe_a8w4_preshuffle_7168x4096_E256T8_M2048" \
    bench_moe_gemm_a8w4_cudagraph.py \
    --shape 7168 4096 --experts 256 8 --M 2048 --preshuffle

# ---------------------------------------------------------------------------
# MoE — bench_moe_gemm_a4w4_cudagraph
# ---------------------------------------------------------------------------
run_bench "moe_a4w4_7168x4096_E256T8_M2048" \
    bench_moe_gemm_a4w4_cudagraph.py \
    --shape 7168 4096 --experts 256 8 --M 2048

# ---------------------------------------------------------------------------
# Attention — bench_unified_attention (bf16 + fp8)
# ---------------------------------------------------------------------------
run_bench "attn_unified_bf16_b4_hq64_hk8_sq1024_sk8192" \
    bench_unified_attention.py \
    -b 4 -hq 64 -hk 8 -sq 1024 -sk 8192 -d 128 -equal_seqlens -block_size 256

run_bench "attn_unified_fp8_b4_hq64_hk8_sq1024_sk8192" \
    bench_unified_attention.py \
    -b 4 -hq 64 -hk 8 -sq 1024 -sk 8192 -d 128 -equal_seqlens -fp8

# ---------------------------------------------------------------------------
# Attention — bench_mla (MLA decode)
# ---------------------------------------------------------------------------
run_bench "attn_mla_decode_varlen" \
    bench_mla.py \
    --varlen ''

# ---------------------------------------------------------------------------
# Attention — bench_kda (KDA decode)
# ---------------------------------------------------------------------------
run_bench "attn_kda_decode_h24_b64_s1" \
    bench_kda.py \
    --num_heads 24 --batch_sizes 64 --seq_lens 1

# ---------------------------------------------------------------------------
# Attention — bench_chunk_kda (chunked KDA prefill)
# ---------------------------------------------------------------------------
run_bench "attn_chunk_kda_h24_b1_s4096" \
    bench_chunk_kda.py \
    --num_heads 24 --batch_sizes 1 --seq_lens 4096 --backends gluon --walk_configs 64,4

# ---------------------------------------------------------------------------
# Attention — bench_fp8_mqa_logits
# ---------------------------------------------------------------------------
run_bench "attn_fp8_mqa_logits" \
    bench_fp8_mqa_logits.py

# ---------------------------------------------------------------------------
# Norm / fusion — bench_fused_add_rmsnorm_pad
# ---------------------------------------------------------------------------
run_bench "norm_fused_add_rmsnorm_pad_8192x7168" \
    bench_fused_add_rmsnorm_pad.py \
    --shape 8192 7168 --add-residual

# ---------------------------------------------------------------------------
# Norm / fusion — bench_fused_rmsnorm_add
# ---------------------------------------------------------------------------
run_bench "norm_fused_rmsnorm_add_8192x7168" \
    bench_fused_rmsnorm_add.py \
    --shape 8192 7168 --add-residual

# ---------------------------------------------------------------------------
# Norm / fusion — bench_rmsnorm (+ mxfp4 quant)
# ---------------------------------------------------------------------------
run_bench "norm_rmsnorm_mxfp4_4096x7168" \
    bench_rmsnorm.py \
    --quant mxfp4 --shape 4096 7168

# ---------------------------------------------------------------------------
# Quant / other — bench_fused_clamp_act_mul
# ---------------------------------------------------------------------------
run_bench "fused_clamp_act_mul_8192x3584" \
    bench_fused_clamp_act_mul.py \
    --shape 8192 3584

# ---------------------------------------------------------------------------
# Quant / other — bench_quant_mxfp4_fp8 (mxfp4)
# ---------------------------------------------------------------------------
run_bench "quant_mxfp4_4096x7168" \
    bench_quant_mxfp4_fp8.py \
    --provider gluon --format mxfp4 --shape 4096 7168

# ---------------------------------------------------------------------------
# Quant / other — bench_quant_mxfp4_fp8 (mxfp8)
# ---------------------------------------------------------------------------
run_bench "quant_mxfp8_4096x7168" \
    bench_quant_mxfp4_fp8.py \
    --provider gluon --format mxfp8 --shape 4096 7168

# ═══════════════════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════════════════
REPORT="$RESULTS_DIR/report.txt"

{
    echo "═══════════════════════════════════════════════════════════════"
    echo " gfx1250 Benchmark Report"
    echo " Date : $(date -Iseconds)"
    echo " Host : $(hostname)"
    echo " GPU  : $(rocm-smi --showproductname 2>/dev/null | grep -m1 'GPU' || echo 'N/A')"
    echo " Commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    printf "%-60s %-6s %5s  %s\n" "Test" "Status" "Time" "Log / Note"
    printf "%-60s %-6s %5s  %s\n" "----" "------" "-----" "----------"
    for line in "${SUMMARY_LINES[@]}"; do
        echo "$line"
    done
    echo ""
    echo "─────────────────────────────────"
    echo " PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
    echo "─────────────────────────────────"
    echo ""
    echo "Results directory: $RESULTS_DIR"
} | tee "$REPORT"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "FAILED tests — see individual logs above for details."
    exit 1
fi
