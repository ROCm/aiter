#!/usr/bin/env bash
# Profile prefill_d128 (FP8 or BF16) on a single representative shape and
# capture the four-phase rocprofv3 result tree the prior runs in
# `rocprof_analysis/runs/` follow:
#   phase_a_trace     — kernel dispatch trace + per-kernel timing stats
#   phase_b1_compute  — instruction-mix counters (MFMA / VALU / VMEM / LDS / etc)
#   phase_b2_stalls   — wait + memory-pipeline busy counters
#   phase_c_pcsamp    — PC sampling (which instructions inside the kernel are hottest)
#
# Each phase reruns the kernel from a clean Python process to avoid PMC
# contamination across phases; rocprofv3 only collects what fits in one PMC pass.
#
# Usage:
#   ua-test-scripts/rocprof_prefill_d128.sh fp8  16 10000 10000
#   ua-test-scripts/rocprof_prefill_d128.sh bf16 16 10000 10000

set -euo pipefail

DTYPE="${1:?dtype: fp8 or bf16}"
BATCH="${2:?batch}"
SQ="${3:?sq}"
SK="${4:?sk}"
GPU="${GPU:-2}"
HQ="${HQ:-12}"
HK="${HK:-2}"
D="${D:-128}"
PS="${PS:-64}"
TAG="${TAG:-prefill_d${D}_${DTYPE}_b${BATCH}_sq${SQ}_sk${SK}}"
ITERS="${ITERS:-50}"

HERE="$(cd "$(dirname "$0")" && pwd)"
AITER_ROOT="$(dirname "$HERE")"
RUN_DIR="$HERE/rocprof_analysis/runs/$TAG"
mkdir -p "$RUN_DIR"

# Match both the mangled symbol (rocprof emits this for some PMC paths) and the
# demangled `void ck_tile::kentry<...UnifiedAttentionKernel...>` form (used by
# the trace+stats path). Going through UnifiedAttentionKernel is enough.
KERNEL_FILTER='.*UnifiedAttentionKernel.*'

# The test_unified_attention_ck.py @perftest decorator runs warmup + 101 iters
# already, plus runs the Triton kernel — we want a CK-only minimal driver so
# rocprof captures only the UA kernel dispatches. Use --no-triton + --no-reference
# to keep the workload to exactly the CK forward dispatch.
RUN_CMD=(
    python3 "$AITER_ROOT/op_tests/test_unified_attention_ck.py"
    -b "$BATCH" -sq "$SQ" -sk "$SK"
    --num-heads "${HQ},${HK}"
    --head-size "$D"
    --block-size "$PS"
    --dtype "$DTYPE"
    --num-blocks auto
    --no-triton
    --no-reference
    --seed 42
)

export HIP_VISIBLE_DEVICES="$GPU"

echo "============================================================"
echo " rocprof prefill_d${D} ${DTYPE}  b=${BATCH} sq=${SQ} sk=${SK}"
echo "   GPU=${GPU}  HQ,HK=${HQ},${HK}  page_size=${PS}"
echo "   out: ${RUN_DIR}"
echo "============================================================"

# Phase A: kernel dispatch trace + per-kernel stats
echo "[phase_a_trace] ..."
rm -rf "$RUN_DIR/phase_a_trace"; mkdir -p "$RUN_DIR/phase_a_trace"
/opt/rocm/bin/rocprofv3 \
    --kernel-trace --stats \
    --kernel-include-regex "$KERNEL_FILTER" \
    -o "trace" -d "$RUN_DIR/phase_a_trace" \
    --output-format csv \
    -- "${RUN_CMD[@]}" > "$RUN_DIR/phase_a_trace.log" 2>&1 || \
    { tail -20 "$RUN_DIR/phase_a_trace.log"; exit 1; }

# Phase B1: compute / instruction-mix counters
echo "[phase_b1_compute] ..."
rm -rf "$RUN_DIR/phase_b1_compute"; mkdir -p "$RUN_DIR/phase_b1_compute"
/opt/rocm/bin/rocprofv3 \
    --pmc GRBM_GUI_ACTIVE SQ_WAVES \
          SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_SALU \
          SQ_INSTS_VMEM SQ_INSTS_LDS SQ_INSTS_VALU_CVT \
    --kernel-include-regex "$KERNEL_FILTER" \
    -o "pmc" -d "$RUN_DIR/phase_b1_compute" \
    --output-format csv \
    -- "${RUN_CMD[@]}" > "$RUN_DIR/phase_b1_compute.log" 2>&1 || \
    { tail -20 "$RUN_DIR/phase_b1_compute.log"; exit 1; }

# Phase B2: stall + memory-pipeline busy counters
echo "[phase_b2_stalls] ..."
rm -rf "$RUN_DIR/phase_b2_stalls"; mkdir -p "$RUN_DIR/phase_b2_stalls"
/opt/rocm/bin/rocprofv3 \
    --pmc GRBM_GUI_ACTIVE SQ_WAVES \
          SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS \
          SQC_TC_STALL TA_BUSY_avr TCC_BUSY_avr TCP_PENDING_STALL_CYCLES_sum \
    --kernel-include-regex "$KERNEL_FILTER" \
    -o "pmc" -d "$RUN_DIR/phase_b2_stalls" \
    --output-format csv \
    -- "${RUN_CMD[@]}" > "$RUN_DIR/phase_b2_stalls.log" 2>&1 || \
    { tail -20 "$RUN_DIR/phase_b2_stalls.log"; exit 1; }

# Phase C: PC sampling (stochastic, cycle-based interval). gfx950 host-trap is
# not supported; stochastic samples are the right choice anyway because they
# don't need a debugger-style breakpoint per sample.
echo "[phase_c_pcsamp] ..."
rm -rf "$RUN_DIR/phase_c_pcsamp"; mkdir -p "$RUN_DIR/phase_c_pcsamp"
/opt/rocm/bin/rocprofv3 \
    --pc-sampling-beta-enabled \
    --pc-sampling-unit cycles \
    --pc-sampling-method stochastic \
    --pc-sampling-interval 1048576 \
    --kernel-include-regex "$KERNEL_FILTER" \
    -o "pcsamp" -d "$RUN_DIR/phase_c_pcsamp" \
    --output-format csv json \
    -- "${RUN_CMD[@]}" > "$RUN_DIR/phase_c_pcsamp.log" 2>&1 || \
    { tail -20 "$RUN_DIR/phase_c_pcsamp.log"; exit 1; }

echo "[done] $RUN_DIR"
