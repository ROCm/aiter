#!/usr/bin/env bash
# Capture torch.profiler chrome traces for a small set of representative
# Qwen3-Next per_1x128 shapes across all three SplitK zero-init modes,
# then print a per-kernel summary so we can verify the producer kernel
# does NOT slow down materially when it absorbs the GEMM Y zero-fill.
set -euo pipefail

AITER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-${AITER_ROOT}/op_tests/zero_init_demo_results/producer_overhead_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

NOSPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_nosplitk_gfx950.csv"
SPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv"
SHAPES_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_untuned.csv"

# Shapes chosen to span splitK depth and Y zero-fill cost (~Y bytes):
#   1, 12288, 2048  : splitK=2, M=1   -> 24 KB  (big-Y, tiny producer)
#   8, 12288, 2048  : splitK=2, M=8   -> 192 KB (big-Y, mid producer)
#   1,  2048, 4096  : splitK=3, M=1   -> 4 KB   (small-Y, lots of K work)
#   32, 2048, 4096  : splitK=3, M=32  -> 128 KB (small-Y, big producer)
#   64, 12288, 2048 : splitK=0, M=64  -> 1.5 MB (CSV picks no splitK; should bypass fuse)
SHAPES=(
    "1,12288,2048"
    "8,12288,2048"
    "1,2048,4096"
    "32,2048,4096"
    "64,12288,2048"
)

ITERS=120
WARMUP=20
TRACE_ITERS=100

JIT_DIR="${AITER_ROOT}/aiter/jit"
SO_FILE="${JIT_DIR}/module_gemm_a8w8_blockscale_bpreshuffle_cktile.so"
BUILD_DIR="${JIT_DIR}/build/module_gemm_a8w8_blockscale_bpreshuffle_cktile"

nuke_module() {
    echo "# nuking bpreshuffle CKTile module so it rebuilds against the active CSV"
    rm -f "${SO_FILE}"
    rm -rf "${BUILD_DIR}"
}

run_shape() {
    local mode="$1"
    local csv="$2"
    local shape="$3"
    local tag="${mode}_${shape//,/x}"
    PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="${csv}" \
        python "${AITER_ROOT}/op_tests/bench_zero_init_splitk_demo.py" \
        --mode "${mode}" \
        --tuned-csv "${csv}" \
        --shapes-csv "${SHAPES_CSV}" \
        --shape "${shape}" \
        --iters "${ITERS}" --warmup "${WARMUP}" \
        --trace-dir "${RESULTS_DIR}/traces" \
        --trace-iters "${TRACE_ITERS}" \
        2>&1 | tee -a "${RESULTS_DIR}/${tag}.log"
}

# Configs splitk and splitk_fused share one CSV-driven .so build.
# Build once at the with-SplitK CSV first.
nuke_module
for shape in "${SHAPES[@]}"; do
    run_shape splitk "${SPLITK_CSV}" "${shape}"
done
for shape in "${SHAPES[@]}"; do
    run_shape splitk_fused "${SPLITK_CSV}" "${shape}"
done

# Then rebuild against the no-SplitK CSV for the baseline.
nuke_module
for shape in "${SHAPES[@]}"; do
    run_shape none "${NOSPLITK_CSV}" "${shape}"
done

echo
echo "=========================================================="
echo "# Per-kernel GPU duration summary (mean us, n iterations):"
echo "=========================================================="
python "${AITER_ROOT}/op_tests/analyze_producer_overhead.py" \
    --trace-dir "${RESULTS_DIR}/traces" \
    --shapes "${SHAPES[@]}"

echo
echo "# Traces and logs saved under: ${RESULTS_DIR}"
