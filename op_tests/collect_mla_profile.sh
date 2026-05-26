#!/usr/bin/env bash
# Collect rocprof hardware counters + work-distribution metadata for qh32 vs fold.
#
# Usage:
#   ./op_tests/collect_mla_profile.sh
#   ./op_tests/collect_mla_profile.sh --batch 1,16,256 --ctx 8192 --outdir ./results
#
# Outputs per experiment (label = qh32 or fold, b = batch, c = ctx):
#   <outdir>/<label>_b<b>_ctx<c>_counters.csv  — rocprof hardware counters (all passes)
#   <outdir>/<label>_b<b>_ctx<c>_metadata.txt  — work_indptr / CU assignment dump

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(command -v python3 || command -v python)"
COUNTERS="${SCRIPT_DIR}/mla_counters.txt"

# Defaults
BATCH_SIZES=(1 16)
CTX=8192
OUTDIR="${SCRIPT_DIR}/../mla_profile_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)  IFS=',' read -ra BATCH_SIZES <<< "$2"; shift 2 ;;
        --ctx)    CTX="$2";   shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTDIR}"

run_experiment() {
    local label="$1"   # qh32 | fold
    local fold="$2"    # 0    | 1
    local batch="$3"
    local ctx="$4"
    local prefix="${OUTDIR}/${label}_b${batch}_ctx${ctx}"

    echo ""
    echo "=== ${label}  batch=${batch}  ctx=${ctx} ==="

    # --- Pass 1: hardware counters via rocprof ---
    echo "  [1/2] rocprof hardware counters -> ${prefix}_counters.csv"
    MLA_FORCE_QH16_FOLD="${fold}" rocprof \
        -i "${COUNTERS}" \
        -o "${prefix}_counters.csv" \
        "${PYTHON}" "${SCRIPT_DIR}/test_mla_persistent.py" \
            --nhead 32,1 --dtype fp8 --kv_dtype fp8 \
            --batchSize "${batch}" --ctxLen "${ctx}" \
        2>&1 | grep -E "(ROCProfiler|error|Error)" || true

    # --- Pass 2: metadata dump ---
    echo "  [2/2] metadata dump           -> ${prefix}_metadata.txt"
    MLA_FORCE_QH16_FOLD="${fold}" \
    DUMP_MLA_METADATA=1 \
    MLA_METADATA_DUMP_PATH="${prefix}_metadata.txt" \
        "${PYTHON}" "${SCRIPT_DIR}/test_mla_persistent.py" \
            --nhead 32,1 --dtype fp8 --kv_dtype fp8 \
            --batchSize "${batch}" --ctxLen "${ctx}" \
        > /dev/null 2>&1

    echo "  done."
}

echo "MLA profile collection: qh32 vs fold"
echo "  batches : ${BATCH_SIZES[*]}"
echo "  ctx     : ${CTX}"
echo "  outdir  : ${OUTDIR}"
echo "  counters: ${COUNTERS}"

for batch in "${BATCH_SIZES[@]}"; do
    run_experiment "qh32" 0 "${batch}" "${CTX}"
    run_experiment "fold" 1 "${batch}" "${CTX}"
done

echo ""
echo "All results saved to: ${OUTDIR}"
