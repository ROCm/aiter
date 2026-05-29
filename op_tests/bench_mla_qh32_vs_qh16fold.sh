#!/usr/bin/env bash
# Benchmark qh32 native kernel vs qh16-fold fallback for nhead=32 qSeqLen=1 fp8/fp8 on gfx950.
#
# Usage:
#   ./bench_mla_qh32_vs_qh16fold.sh [output_file]
#
# Kernel selection:
#   MLA_FORCE_QH16_FOLD=1  -> forces the old qh16 fold path (set in env before launching)
#   MLA_FORCE_QH16_FOLD=0  -> uses the new native qh32 kernel (default)
#
# Sweep structure:
#   Part 1 – performance sweep  : large ctx (2048-32768), all batch sizes
#   Part 2 – precision sweep    : small ctx (32-1024) where tail-masking fires, batch=1
#   Part 3 – split-count sweep  : ctx=2048, batch=1, max_split in {1,4,8,16,32,64}

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(command -v python3 || command -v python)"
OUTPUT="${1:-${SCRIPT_DIR}/bench_qh32_vs_qh16fold_results.txt}"

# Part 1 – performance (original coverage)
PERF_BATCH_SIZES=(1 4 16 32 64 128 256 512 1024)
PERF_CTX_LENS=(2048 8192 32768)

# Part 2 – precision / tail-mask stress (small ctx, batch=1)
PREC_CTX_LENS=(32 64 128 256 512 1024)

# Part 3 – split-count precision (ctx=2048, batch=1)
SPLIT_COUNTS=(1 4 8 16 32 64)

run_perf_sweep() {
    local label="$1"
    local force_fold="$2"

    echo "===== ${label} — PERFORMANCE SWEEP =====" | tee -a "${OUTPUT}"
    echo "  MLA_FORCE_QH16_FOLD=${force_fold}" | tee -a "${OUTPUT}"
    echo "" | tee -a "${OUTPUT}"

    for batch in "${PERF_BATCH_SIZES[@]}"; do
        for ctx in "${PERF_CTX_LENS[@]}"; do
            echo "  batch=${batch}  ctxLen=${ctx}" | tee -a "${OUTPUT}"
            MLA_FORCE_QH16_FOLD="${force_fold}" "${PYTHON}" "${SCRIPT_DIR}/test_mla_persistent.py" \
                --nhead 32,1 \
                --dtype fp8 \
                --kv_dtype fp8 \
                --batchSize "${batch}" \
                --ctxLen "${ctx}" \
                2>&1 | tee -a "${OUTPUT}"
            echo "" | tee -a "${OUTPUT}"
        done
    done
    echo "" | tee -a "${OUTPUT}"
}

run_precision_sweep() {
    local label="$1"
    local force_fold="$2"

    echo "===== ${label} — PRECISION SWEEP (small ctx, tail-mask path) =====" | tee -a "${OUTPUT}"
    echo "  MLA_FORCE_QH16_FOLD=${force_fold}" | tee -a "${OUTPUT}"
    echo "" | tee -a "${OUTPUT}"

    for ctx in "${PREC_CTX_LENS[@]}"; do
        echo "  batch=1  ctxLen=${ctx}" | tee -a "${OUTPUT}"
        MLA_FORCE_QH16_FOLD="${force_fold}" "${PYTHON}" "${SCRIPT_DIR}/test_mla_persistent.py" \
            --nhead 32,1 \
            --dtype fp8 \
            --kv_dtype fp8 \
            --batchSize 1 \
            --ctxLen "${ctx}" \
            2>&1 | tee -a "${OUTPUT}"
        echo "" | tee -a "${OUTPUT}"
    done
    echo "" | tee -a "${OUTPUT}"
}

run_split_sweep() {
    local label="$1"
    local force_fold="$2"

    echo "===== ${label} — SPLIT-COUNT SWEEP (ctx=2048, batch=1) =====" | tee -a "${OUTPUT}"
    echo "  MLA_FORCE_QH16_FOLD=${force_fold}" | tee -a "${OUTPUT}"
    echo "" | tee -a "${OUTPUT}"

    for splits in "${SPLIT_COUNTS[@]}"; do
        echo "  max_split=${splits}" | tee -a "${OUTPUT}"
        MLA_FORCE_QH16_FOLD="${force_fold}" "${PYTHON}" "${SCRIPT_DIR}/test_mla_persistent.py" \
            --nhead 32,1 \
            --dtype fp8 \
            --kv_dtype fp8 \
            --batchSize 1 \
            --ctxLen 2048 \
            --max_split_per_batch "${splits}" \
            2>&1 | tee -a "${OUTPUT}"
        echo "" | tee -a "${OUTPUT}"
    done
    echo "" | tee -a "${OUTPUT}"
}

echo "MLA qh32 native vs qh16-fold benchmark" | tee "${OUTPUT}"
echo "Date: $(date)" | tee -a "${OUTPUT}"
echo "GPU: $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: //' || echo 'unknown')" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"

run_perf_sweep      "qh32 NATIVE kernel (new)" 0
run_precision_sweep "qh32 NATIVE kernel (new)" 0
run_split_sweep     "qh32 NATIVE kernel (new)" 0

run_perf_sweep      "qh16 FOLD fallback (old)" 1
run_precision_sweep "qh16 FOLD fallback (old)" 1
run_split_sweep     "qh16 FOLD fallback (old)" 1

echo "Done. Results saved to: ${OUTPUT}"
