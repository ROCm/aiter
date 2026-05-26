#!/usr/bin/env bash
# 3-way benchmark: qh32 v2 native vs qh16-fold vs qh8-fold
# for nhead=32 qSeqLen=1 fp8/fp8 on gfx950.
#
# Usage:
#   ./bench_mla_3way.sh [output_file]
#
# Kernel selection via env vars:
#   (default)              -> qh32 v2 native kernel
#   MLA_FORCE_QH16_FOLD=1  -> qh16 fold (2x work items, 16 heads/WG)
#   MLA_FORCE_QH8_FOLD=1   -> qh8 fold  (4x work items, 8 heads/WG)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(command -v python3 || command -v python)"
OUTPUT="${1:-${SCRIPT_DIR}/bench_3way_results.txt}"

BATCH_SIZES=(1 4 16 32 64 128 256 512 1024)
CTX_LENS=(2048 8192 32768)

run_sweep() {
    local label="$1"
    shift
    local -a env_vars=("$@")

    echo "===== ${label} =====" | tee -a "${OUTPUT}"
    for ev in "${env_vars[@]}"; do
        echo "  ${ev}" | tee -a "${OUTPUT}"
    done
    echo "" | tee -a "${OUTPUT}"

    for batch in "${BATCH_SIZES[@]}"; do
        for ctx in "${CTX_LENS[@]}"; do
            echo "  batch=${batch}  ctxLen=${ctx}" | tee -a "${OUTPUT}"
            env "${env_vars[@]}" "${PYTHON}" "${SCRIPT_DIR}/test_mla_persistent.py" \
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

echo "MLA 3-way benchmark: qh32-v2 vs qh16-fold vs qh8-fold" | tee "${OUTPUT}"
echo "Date: $(date)" | tee -a "${OUTPUT}"
echo "GPU: $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: //' || echo 'unknown')" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"

run_sweep "qh32 v2 NATIVE kernel" \
    "MLA_FORCE_QH16_FOLD=0" "MLA_FORCE_QH8_FOLD=0"

run_sweep "qh16 FOLD (16 heads/WG, 2x work items)" \
    "MLA_FORCE_QH16_FOLD=1" "MLA_FORCE_QH8_FOLD=0"

run_sweep "qh8 FOLD (8 heads/WG, 4x work items)" \
    "MLA_FORCE_QH16_FOLD=0" "MLA_FORCE_QH8_FOLD=1"

echo "Done. Results saved to: ${OUTPUT}"
