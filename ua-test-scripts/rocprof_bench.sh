#!/bin/bash
# ROCProfiler v3 benchmark wrapper for CK vs Triton unified attention

set -e

WARMUP=20
ITERS=50
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="rocprofv3_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

PYTHON_OUTPUT="$OUTPUT_DIR/python_output.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

echo "============================================"
echo "ROCProfiler Benchmark for UA Kernels"
echo "============================================"
echo "Warmup iterations: $WARMUP"
echo "Bench iterations:  $ITERS"
echo "Extra arguments:   $EXTRA_ARGS"
echo "Output directory:  $OUTPUT_DIR"
echo ""

cd "$OUTPUT_DIR"
rocprofv3 \
    --sys-trace \
    --kernel-trace \
    --stats \
    -d . \
    -o results \
    -f csv json \
    -- python3 ../test_single_shape.py \
        $EXTRA_ARGS \
        --warmup $WARMUP \
        --iters $ITERS \
    2>&1 | tee python_output.txt
cd ..

echo ""
echo "============================================"
echo "Processing results..."
echo "============================================"

PYTHON_CK_MS=$(grep "CK time:" "$PYTHON_OUTPUT" | awk '{print $3}')
PYTHON_TRITON_MS=$(grep "Triton time:" "$PYTHON_OUTPUT" | awk '{print $3}')
PYTHON_SPEEDUP=$(grep "Speedup:" "$PYTHON_OUTPUT" | awk '{print $2}' | sed 's/x//')

echo "" > "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"
echo "BENCHMARK SUMMARY" >> "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Test Configuration:" >> "$SUMMARY_FILE"
grep -A 20 "Shape Configuration:" "$PYTHON_OUTPUT" | head -20 >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "============================================" >> "$SUMMARY_FILE"
echo "Python-Level Timing (median of $ITERS iters)" >> "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"
if [ -n "$PYTHON_CK_MS" ]; then
    printf "  %-20s %10s ms\n" "CK:" "$PYTHON_CK_MS" >> "$SUMMARY_FILE"
fi
if [ -n "$PYTHON_TRITON_MS" ]; then
    printf "  %-20s %10s ms\n" "Triton:" "$PYTHON_TRITON_MS" >> "$SUMMARY_FILE"
fi
if [ -n "$PYTHON_SPEEDUP" ]; then
    printf "  %-20s %10sx\n" "Speedup:" "$PYTHON_SPEEDUP" >> "$SUMMARY_FILE"
fi
echo "" >> "$SUMMARY_FILE"

echo "============================================" >> "$SUMMARY_FILE"
echo "ROCProfiler Kernel Statistics" >> "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

KERNEL_TRACE="$OUTPUT_DIR/results_kernel_trace.csv"
if [ -f "$KERNEL_TRACE" ] && [ -s "$KERNEL_TRACE" ]; then
    if [ -f "parse_kernel_trace.py" ]; then
        python3 parse_kernel_trace.py "$KERNEL_TRACE" --warmup $WARMUP >> "$SUMMARY_FILE" 2>&1
    fi
fi

echo "" >> "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"
echo "Files generated:" >> "$SUMMARY_FILE"
echo "  - $PYTHON_OUTPUT" >> "$SUMMARY_FILE"
echo "  - $SUMMARY_FILE" >> "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Full summary: $SUMMARY_FILE"
echo ""
echo "Cleaning up trace files..."
rm -rf "$OUTPUT_DIR"
echo "Done (results deleted after summary)."
