#!/usr/bin/env bash
# Three-phase rocprofv3 profile of the CK Unified Attention kernel.
#
#   Phase A: kernel-trace + occupancy/resource info (workgroup size, VGPR, LDS).
#   Phase B: PMC counters for compute mix and stalls.
#   Phase C: PC sampling for per-instruction hotspots inside the kernel.
#
# Usage:
#   ./run_profile.sh -b 4 -sq 8 -sk 4096 -hq 64 -hk 8 -d 128 \
#                    --block-size 32 --dtype fp8 --no-splitkv
# Add `--only-ck` or `--only-triton` to restrict to one backend.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT=${OUT:-runs/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

# Forward all args to test_single_shape.py.
APP_ARGS=("$@")
SHAPE_ARGS=("${APP_ARGS[@]}" --warmup 5 --iters 50 --seed 42)
PMC_ARGS=("${APP_ARGS[@]}" --warmup 3 --iters 6 --seed 42)     # PMC is heavy; few iters
PC_ARGS=("${APP_ARGS[@]}"  --warmup 2 --iters 200 --seed 42)  # need enough samples

echo "=== Phase A: kernel-trace ==="
rocprofv3 --kernel-trace --stats -d "$OUT/phase_a_trace" -o trace -f csv \
  -- python3 ../test_single_shape.py "${SHAPE_ARGS[@]}" 2>&1 | tail -3

echo "=== Phase B1: PMC compute mix ==="
rocprofv3 --pmc \
    SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_LDS SQ_INSTS_SALU SQ_INSTS_VMEM \
    SQ_INSTS_VALU_CVT GRBM_GUI_ACTIVE SQ_WAVES \
  -d "$OUT/phase_b1_compute" -o pmc -f csv \
  -- python3 ../test_single_shape.py "${PMC_ARGS[@]}" 2>&1 | tail -3

echo "=== Phase B2: PMC stalls + memory ==="
rocprofv3 --pmc \
    SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY SQ_WAIT_ANY \
    GRBM_GUI_ACTIVE SQ_WAVES \
    TCP_PENDING_STALL_CYCLES_sum TA_BUSY_avr TCC_BUSY_avr SQC_TC_STALL \
  -d "$OUT/phase_b2_stalls" -o pmc -f csv \
  -- python3 ../test_single_shape.py "${PMC_ARGS[@]}" 2>&1 | tail -3

echo "=== Phase C: PC sampling (host_trap, 100us) ==="
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-unit time --pc-sampling-interval 100 \
    --pc-sampling-method host_trap --kernel-trace \
  -d "$OUT/phase_c_pcsamp" -o pcsamp -f rocpd csv json \
  -- python3 ../test_single_shape.py "${PC_ARGS[@]}" 2>&1 | tail -3

echo ""
echo "=== Aggregated PMC tables ==="
python3 aggregate_pmc.py "$OUT/phase_b1_compute/pmc_counter_collection.csv" --warmup 3
python3 aggregate_pmc.py "$OUT/phase_b2_stalls/pmc_counter_collection.csv"  --warmup 3

echo "=== PC-sample hotspots (CK) ==="
python3 pc_hotspots.py "$OUT/phase_c_pcsamp/pcsamp_results.json" --top 10 --context 4

echo "Saved to $OUT"
