#!/usr/bin/env bash
# ============================================================================
# profile_attention_grids.sh
#
# Profiles attention kernels with rocprofv3 and prints a summary table showing
# grid dimensions, workgroup sizes, register usage, LDS, and duration for each
# kernel dispatch.
#
# Usage:
#   ./profile_attention_grids.sh                        # default shape
#   PROF_SEQS=128 PROF_MAXK=8192 ./profile_attention_grids.sh   # custom shape
#
# Environment variables (all optional):
#   PROF_SEQS   number of sequences        (default: 248)
#   PROF_NQH    number of query heads       (default: 8)
#   PROF_NKH    number of kv heads          (default: 1)
#   PROF_HDIM   head dimension              (default: 64)
#   PROF_MAXK   max kv sequence length      (default: 4096)
#   PROF_BLK    page block size             (default: 64)
#   PROF_ITERS  iterations per backend      (default: 3)
#
# Requirements:
#   - rocprofv3 (ships with ROCm >= 6.0)
#   - Python with aiter installed
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD="${SCRIPT_DIR}/profile_attention_grids.py"
OUTDIR="/tmp/attn_profile_$$"

if ! command -v rocprofv3 &>/dev/null; then
    echo "ERROR: rocprofv3 not found. Make sure ROCm is installed and in PATH." >&2
    exit 1
fi

if [[ ! -f "$WORKLOAD" ]]; then
    echo "ERROR: workload script not found at $WORKLOAD" >&2
    exit 1
fi

echo "============================================================"
echo " rocprofv3 Attention Kernel Profiler"
echo "============================================================"
echo "Output directory: $OUTDIR"
echo ""

# Run with kernel trace + marker trace (roctx) + truncated kernel names
rocprofv3 \
    --kernel-trace \
    --marker-trace \
    -T \
    -d "$OUTDIR" \
    -f csv \
    -- python3 "$WORKLOAD" 2>&1

echo ""

# Find the kernel trace CSV (rocprofv3 nests under hostname/pid)
KERNEL_CSV=$(find "$OUTDIR" -name '*kernel_trace.csv' -type f | head -1)

if [[ -z "$KERNEL_CSV" || ! -f "$KERNEL_CSV" ]]; then
    echo "ERROR: No kernel_trace.csv found in $OUTDIR" >&2
    echo "Files present:" >&2
    find "$OUTDIR" -type f >&2
    exit 1
fi

echo "============================================================"
echo " Kernel Dispatch Summary"
echo "============================================================"
echo ""
echo "Source: $KERNEL_CSV"
echo ""

# Parse and display the CSV with Python for clean formatting
python3 - "$KERNEL_CSV" <<'PYEOF'
import csv
import sys
import re
from collections import defaultdict

csv_path = sys.argv[1]

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("Kind") != "KERNEL_DISPATCH":
            continue
        rows.append(row)

if not rows:
    print("No kernel dispatches found.")
    sys.exit(0)

# Filter to attention-related kernels (skip setup/memset/etc)
ATTN_PATTERNS = [
    r"unified_attention",
    r"kernel_unified_attention",
    r"fmha",
    r"Fmha",
    r"FMHA",
    r"attention",
    r"reduce_segments",
    r"splitkv",
    r"split_kv",
    r"batch_prefill",
    r"pagedkv",
    r"kentry",        # CK tile generic kernel entry point
]

def is_attention_kernel(name):
    for pat in ATTN_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return True
    return False

def short_name(name):
    """Truncate long kernel names to something readable."""
    # Already truncated by -T flag, but clean up further
    name = name.strip('"')
    if len(name) > 80:
        # Try to get just the function name
        # Look for the last :: before template args
        match = re.match(r'([^<(]+)', name)
        if match:
            base = match.group(1).strip()
            if len(base) > 80:
                base = base[:77] + "..."
            return base
    return name

# Group kernels by name and aggregate
grouped = defaultdict(lambda: {
    "count": 0,
    "grid_x": 0, "grid_y": 0, "grid_z": 0,
    "wg_x": 0, "wg_y": 0, "wg_z": 0,
    "vgpr": 0, "agpr": 0, "sgpr": 0,
    "lds": 0, "scratch": 0,
    "durations_us": [],
})

for row in rows:
    name = row["Kernel_Name"]
    g = grouped[name]
    g["count"] += 1
    g["grid_x"]  = int(row["Grid_Size_X"])
    g["grid_y"]  = int(row["Grid_Size_Y"])
    g["grid_z"]  = int(row["Grid_Size_Z"])
    g["wg_x"]    = int(row["Workgroup_Size_X"])
    g["wg_y"]    = int(row["Workgroup_Size_Y"])
    g["wg_z"]    = int(row["Workgroup_Size_Z"])
    g["vgpr"]    = int(row["VGPR_Count"])
    g["agpr"]    = int(row["Accum_VGPR_Count"])
    g["sgpr"]    = int(row["SGPR_Count"])
    g["lds"]     = int(row["LDS_Block_Size"])
    g["scratch"]  = int(row["Scratch_Size"])
    start = int(row["Start_Timestamp"])
    end   = int(row["End_Timestamp"])
    g["durations_us"].append((end - start) / 1000.0)

# ---- Print ALL kernels summary ----
print(f"Total kernel dispatches: {len(rows)}")
print(f"Unique kernel names:     {len(grouped)}")
print()

# ---- Print attention kernels in detail ----
attn_kernels = {k: v for k, v in grouped.items() if is_attention_kernel(k)}
other_kernels = {k: v for k, v in grouped.items() if not is_attention_kernel(k)}

if not attn_kernels:
    print("WARNING: No attention kernels detected. Showing ALL kernels.\n")
    attn_kernels = grouped
    other_kernels = {}

print("=" * 120)
print(" ATTENTION KERNELS")
print("=" * 120)

# MI300X hardware capacity (for thread-budget context)
MI300X_CUS         = 256
MI300X_SIMDS_PER_CU = 4
MI300X_WAVES_PER_SIMD = 8   # max in-flight wavefronts per SIMD
MI300X_WAVE_SIZE   = 64
GPU_MAX_THREADS    = (MI300X_CUS * MI300X_SIMDS_PER_CU
                      * MI300X_WAVES_PER_SIMD * MI300X_WAVE_SIZE)  # 524288

for name, g in sorted(attn_kernels.items(), key=lambda x: -x[1]["count"]):
    sname = short_name(name)
    wg_total = g["wg_x"] * g["wg_y"] * g["wg_z"]
    num_warps = wg_total // 64
    total_threads = g["grid_x"] * g["grid_y"] * g["grid_z"]
    total_wgs = total_threads // max(wg_total, 1)
    total_waves = total_threads // MI300X_WAVE_SIZE
    wgs_per_cu = total_wgs / MI300X_CUS
    pct_of_gpu = 100.0 * total_threads / GPU_MAX_THREADS
    avg_us = sum(g["durations_us"]) / len(g["durations_us"])
    min_us = min(g["durations_us"])
    max_us = max(g["durations_us"])

    print(f"\n  Kernel: {sname}")
    print(f"  {'─' * (len(sname) + 8)}")
    print(f"  Dispatches:     {g['count']}")
    print(f"  Grid:           ({g['grid_x']}, {g['grid_y']}, {g['grid_z']})   "
          f"= {total_wgs} workgroups")
    print(f"  Workgroup:      ({g['wg_x']}, {g['wg_y']}, {g['wg_z']})   "
          f"= {wg_total} threads = {num_warps} warps")
    print(f"  Total threads:  {total_threads:,} ({total_waves:,} wavefronts)   "
          f"WGs/CU: {wgs_per_cu:.2f}   "
          f"= {pct_of_gpu:.1f}% of GPU max ({GPU_MAX_THREADS:,} threads @ peak occupancy)")
    print(f"  VGPRs:          {g['vgpr']}   "
          f"AccVGPRs: {g['agpr']}   "
          f"SGPRs: {g['sgpr']}")
    print(f"  LDS:            {g['lds']} bytes ({g['lds'] / 1024:.1f} KB)")
    if g["scratch"] > 0:
        print(f"  Scratch:        {g['scratch']} bytes")
    print(f"  Duration:       avg={avg_us:.1f} us   min={min_us:.1f} us   max={max_us:.1f} us")

    # Occupancy estimate (MI300X: 256 CUs, 64KB LDS, 512 VGPRs per SIMD, 4 SIMDs)
    if g["vgpr"] > 0:
        waves_by_vgpr = 512 // max(g["vgpr"], 1)  # per SIMD
    else:
        waves_by_vgpr = 8
    if g["lds"] > 0:
        waves_by_lds = (65536 // g["lds"]) * num_warps  # LDS is per workgroup
    else:
        waves_by_lds = 999
    max_waves = min(waves_by_vgpr, waves_by_lds, 8)
    print(f"  Occupancy est:  {max_waves} waves/SIMD "
          f"(limited by {'VGPRs' if waves_by_vgpr <= waves_by_lds else 'LDS'}; "
          f"vgpr_limit={waves_by_vgpr}, lds_limit={waves_by_lds})")

# ---- Print non-attention kernels as a compact summary ----
if other_kernels:
    print(f"\n{'=' * 120}")
    print(f" OTHER KERNELS (non-attention)")
    print(f"{'=' * 120}")
    print(f"\n  {'Kernel':<65} {'#':>4} {'Grid':>25} {'WG':>15} {'Dur(us)':>10}")
    print(f"  {'─' * 65} {'─' * 4} {'─' * 25} {'─' * 15} {'─' * 10}")
    for name, g in sorted(other_kernels.items(), key=lambda x: -x[1]["count"]):
        sname = short_name(name)
        if len(sname) > 65:
            sname = sname[:62] + "..."
        grid_str = f"({g['grid_x']},{g['grid_y']},{g['grid_z']})"
        wg_str = f"({g['wg_x']},{g['wg_y']},{g['wg_z']})"
        avg_us = sum(g["durations_us"]) / len(g["durations_us"])
        print(f"  {sname:<65} {g['count']:>4} {grid_str:>25} {wg_str:>15} {avg_us:>10.1f}")

print()
print("=" * 120)
print(" QUICK REFERENCE")
print("=" * 120)
print("""
  Grid = (Grid_Size_X, Grid_Size_Y, Grid_Size_Z)
    Total workgroups = Grid_X * Grid_Y * Grid_Z / Workgroup_Size

  Workgroup = (WG_X, WG_Y, WG_Z)
    Total threads = WG_X * WG_Y * WG_Z
    Warps = Total threads / 64

  VGPRs = vector general-purpose registers per thread
  AccVGPRs = accumulation VGPRs (used by MFMA instructions)
  SGPRs = scalar general-purpose registers per wavefront
  LDS = local data share (shared memory) per workgroup

  MI300X: 256 CUs, 4 SIMDs/CU, 512 VGPRs/SIMD, 64KB LDS/CU
          Max in-flight: 8 waves/SIMD * 4 SIMDs * 256 CUs * 64 = 524,288 threads
""")

PYEOF

echo ""
echo "Raw CSV at: $KERNEL_CSV"
echo "To explore manually:  column -s, -t $KERNEL_CSV | less -S"
