#!/usr/bin/env bash
# Focused rocprofv3 comparison: bf16 decode b=128 sk=128000 at d=64 vs d=128.
#
# Goal: explain the BW gap (CK 4502 vs 4818 GB/s for d=64/128; Triton 3970 vs
# 5662). All d=128 instances use kBlockN=32 while d=64 uses kBlockN=64 — so
# d=128 issues 2x more main-loop iterations, 2x s_barrier count, 2x page-table
# lookups. We want to see if that pushes us into LDS-bank-conflict / LDS-
# stall territory or if it's mostly extra s_barrier / VALU.
#
# Three single-pass PMC bundles per (d, dtype):
#   p1 = compute mix         (VALU / MFMA / LDS / SALU / VMEM inst counts)
#   p2 = LDS stalls          (bank/addr conflicts, IDX_ACTIVE, INSTS_LDS)
#   p3 = cache + texture     (TCC hit/miss, TA busy, TCP stall)
set -euo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_ROOT="${OUT_ROOT:-rocprof_analysis/runs/d64_vs_d128_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

# 6 iters total (3 warmup discarded later) per PMC pass — heavier counters need
# multi-iter stability. Use --no-causal to match the sweep_fp8.sh decode shape.
COMMON_ARGS=(-sq 1 -sk 128000 -hq 64 -hk 8 --block-size 32
             --warmup 3 --iters 6 --seed 42 --only-ck --dtype bf16)

run_one() {
    local label="$1" d="$2" b="$3"
    local out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"

    local args=("${COMMON_ARGS[@]}" -b "$b" -d "$d")
    echo "=== $label (d=$d b=$b) ==="

    echo "  > p1 compute mix"
    rocprofv3 --pmc \
        SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_LDS SQ_INSTS_SALU SQ_INSTS_VMEM \
        SQ_INSTS_VALU_CVT GRBM_GUI_ACTIVE SQ_WAVES \
      -d "$out_dir/p1_compute" -o pmc -f csv \
      -- python3 test_single_shape.py "${args[@]}" 2>&1 | tail -2 >/dev/null

    echo "  > p2 LDS stalls"
    rocprofv3 --pmc \
        SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_LDS_IDX_ACTIVE \
        SQ_INSTS_LDS SQ_INST_LEVEL_LDS GRBM_GUI_ACTIVE SQ_WAVES \
      -d "$out_dir/p2_lds" -o pmc -f csv \
      -- python3 test_single_shape.py "${args[@]}" 2>&1 | tail -2 >/dev/null

    echo "  > p3 cache+TA"
    rocprofv3 --pmc \
        TCC_HIT_sum TCC_MISS_sum TA_BUSY_avr TCP_PENDING_STALL_CYCLES_sum \
        TCC_BUSY_avr GRBM_GUI_ACTIVE SQ_WAVES \
      -d "$out_dir/p3_cache" -o pmc -f csv \
      -- python3 test_single_shape.py "${args[@]}" 2>&1 | tail -2 >/dev/null
}

run_one "d64_b128"  64  128
run_one "d128_b128" 128 128

# Also profile prefill (b=1 sq=sk=75600) — the worst d=128 row in the sweep.
COMMON_ARGS=(-sq 75600 -sk 75600 -hq 64 -hk 8 --block-size 32
             --warmup 3 --iters 6 --seed 42 --only-ck --dtype bf16)

run_one "d64_prefill"  64  1
run_one "d128_prefill" 128 1

echo ""
echo "Saved to $OUT_ROOT"
echo "Compare with: python3 rocprof_analysis/compare_d64_d128.py $OUT_ROOT"
