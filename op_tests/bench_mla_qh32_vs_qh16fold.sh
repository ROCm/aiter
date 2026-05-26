#!/usr/bin/env bash
# Benchmark qh32 native kernel vs qh16-fold fallback for nhead=32 qSeqLen=1 fp8/fp8 on gfx950.
#
# Usage:
#   ./bench_mla_qh32_vs_qh16fold.sh [output_file]
#
# Kernel selection:
#   MLA_FORCE_QH16_FOLD=1  -> forces the old qh16 fold path (set in env before launching)
#   MLA_FORCE_QH16_FOLD=0  -> uses the new native qh32 kernel (default)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-${SCRIPT_DIR}/bench_qh32_vs_qh16fold_results.txt}"

BATCH_SIZES=(1 4 16)
CTX_LENS=(2048 8192 32768)

run_sweep() {
    local label="$1"
    local force_fold="$2"

    echo "===== ${label} =====" | tee -a "${OUTPUT}"
    echo "  MLA_FORCE_QH16_FOLD=${force_fold}" | tee -a "${OUTPUT}"
    echo "" | tee -a "${OUTPUT}"

    for batch in "${BATCH_SIZES[@]}"; do
        for ctx in "${CTX_LENS[@]}"; do
            echo "  batch=${batch}  ctxLen=${ctx}" | tee -a "${OUTPUT}"
            MLA_FORCE_QH16_FOLD="${force_fold}" python "${SCRIPT_DIR}/test_mla_persistent.py" \
                --nhead 32,1 \
                --dtype fp8 \
                --kv_dtype fp8 \
                --batchSize "${batch}" \
                --ctxLen "${ctx}" \
                2>&1 | grep -E "us\.\.\.\.\.\.|summary|batch|ctx|error|Error" \
                | tee -a "${OUTPUT}"
            echo "" | tee -a "${OUTPUT}"
        done
    done
    echo "" | tee -a "${OUTPUT}"
}

echo "MLA qh32 native vs qh16-fold benchmark" | tee "${OUTPUT}"
echo "Date: $(date)" | tee -a "${OUTPUT}"
echo "GPU: $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: //' || echo 'unknown')" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"

run_sweep "qh32 NATIVE kernel (new)" 0
run_sweep "qh16 FOLD fallback (old)" 1

echo "Done. Results saved to: ${OUTPUT}"
