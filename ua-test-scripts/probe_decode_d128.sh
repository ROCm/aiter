#!/usr/bin/env bash
# A/B harness: bf16 vs fp8 VGPR/LDS/scratch/spill on the decode variants.
#
# Rationale: attention decode is memory-bound, so peak MFMA throughput is
# not the deciding factor. What matters is per-block resource pressure
# (VGPR + LDS) because that gates how many blocks the scheduler can pack
# per CU — and therefore the latency-hiding the kernel has against HBM.
# Same launch shape, same code path, only the QDataType differs → if
# the LDS / VGPR numbers diverge, we know we're tile-budgeting one
# dtype more aggressively than the other.
#
# Probes both decode_d128_m128 (kBlockM=128, 4 warps) and
# decode_d128_m32 (kBlockM=32, 1 warp), since those are the variants
# the dispatcher actually hits on the bf16-loses-to-Triton shapes.
#
# Outputs a single line per (variant, dtype):
#   CK ms | VGPR | V_spill | AGPR | SGPR | Scratch B | LDS KiB
#
# rocprofv3 under-reports VGPR_Count by ~2x on gfx950 wave64 and reports
# a spurious 64 B scratch, so we go straight to the device ELF kernel
# descriptor (`.note YAML` from llvm-readelf) for authoritative numbers.

set -euo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
cd "$(dirname "$0")/.."

TAG="${1:-decode}"

# Pick a shape that exercises both variants:
#   * sq=1  → forces decode_d{128,64}_m{16,32,128} dispatch (small Q tile)
#   * sk    → long enough that VMEM dominates the kernel runtime
#   * b * num_qpkv: tweak so the decode_d128_m128 path is selected
#                   (kBlockM = num_qpkv * q_tiles → 4 warps tier).
#
# decode_d128_m128 dispatch: b=1, hq=64, hk=8 → num_qpkv=8, max_q*=8 ≥ kBlockQ_tiny
# decode_d128_m32 dispatch:  b=1, hq=4,  hk=4 → MHA, max_q*=1, falls to m32 tier
declare -A SHAPES=(
    # variant_label : "b sq sk hq hk d"
    [decode_d128_m128_GQA8]="1 1 65536 64 8 128"
    [decode_d128_m32_MHA]="1 1 65536 4 4 128"
    [decode_d64_m128_GQA8]="1 1 65536 64 8 64"
)

# Clear the JIT cache so each rebuild picks up any kernel source change.
rm -f aiter/jit/module_unified_attention.so \
      aiter/jit/build/module_unified_attention/build/*.o 2>/dev/null || true

echo
printf '  %-26s %-4s  %8s  %5s  %7s  %5s  %5s  %8s  %5s\n' \
    "shape" "dt" "CK_us" "VGPR" "V_spill" "AGPR" "SGPR" "Scratch" "LDS"
echo "  $(printf '%.0s-' {1..86})"

for label in "${!SHAPES[@]}"; do
    read b sq sk hq hk d <<< "${SHAPES[$label]}"

    for dt in bf16 fp8; do
        # Timed run (no rocprof — rocprofv3's tracing distorts timing on small kernels).
        t=$(python3 ua-test-scripts/test_single_shape.py \
              -b "$b" -sq "$sq" -sk "$sk" -hq "$hq" -hk "$hk" -d "$d" \
              --dtype "$dt" --only-ck --warmup 5 --iters 50 --seed 42 2>/dev/null \
            | grep -E "^\s*CK time:" | awk '{print $3}' || echo "—")

        # Authoritative VGPR/LDS extraction from the device ELF kernel descriptor.
        CLANGBNDL=/opt/rocm/lib/llvm/bin/clang-offload-bundler
        LLVMREADELF=/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/llvm-readelf
        OBJ=aiter/jit/build/module_unified_attention/build/unified_attention_d${d}_${dt}_nmask.cuda.o
        if [[ ! -f "$OBJ" ]]; then
            printf '  %-26s %-4s  %8s  (object %s not built yet)\n' "$label" "$dt" "$t" "$OBJ"
            continue
        fi

        TMPDIR_=$(mktemp -d)
        objcopy --dump-section .hip_fatbin="$TMPDIR_/fatbin.bin" "$OBJ" 2>/dev/null
        "$CLANGBNDL" --type=o --unbundle --input="$TMPDIR_/fatbin.bin" \
            --output="$TMPDIR_/gpu.o" --output="$TMPDIR_/host.o" \
            --targets=hipv4-amdgcn-amd-amdhsa--gfx950,host-x86_64-unknown-linux-gnu- \
            2>/dev/null

        "$LLVMREADELF" --notes "$TMPDIR_/gpu.o" 2>&1 \
            | LABEL="$label" DT="$dt" CK_T="$t" python3 -c "
import sys, re, os
text = sys.stdin.read()
# Group kernels by variant name keyword so we attribute the table row
# to the right kernel descriptor (each .o may pack multiple variants).
needle = os.environ['LABEL'].split('_')[1] + '_' + os.environ['LABEL'].split('_')[2]
entries = re.findall(r'- \.agpr_count:\s+(\d+).*?\.group_segment_fixed_size:\s+(\d+).*?\.name:\s+(\S+).*?\.private_segment_fixed_size:\s+(\d+).*?\.sgpr_count:\s+(\d+).*?\.sgpr_spill_count:\s+(\d+).*?\.vgpr_count:\s+(\d+).*?\.vgpr_spill_count:\s+(\d+)', text, re.DOTALL)
matched = None
for agpr, lds, name, scr, sgpr, ss, vgpr, vs in entries:
    if 'kentry' not in name: continue
    if needle in name:
        matched = (vgpr, vs, agpr, sgpr, scr, lds)
        break
if matched is None:
    # Fallback: take the first kentry that has the right page-size suffix or just the first.
    for agpr, lds, name, scr, sgpr, ss, vgpr, vs in entries:
        if 'kentry' in name:
            matched = (vgpr, vs, agpr, sgpr, scr, lds)
            break
if matched:
    vgpr, vs, agpr, sgpr, scr, lds = matched
    label, dt, t = os.environ['LABEL'], os.environ['DT'], os.environ['CK_T']
    print(f'  {label:<26s} {dt:<4s}  {t:>8s}  {vgpr:>5s}  {vs:>7s}  {agpr:>5s}  {sgpr:>5s}  {scr:>6s} B  {int(lds)//1024:>3d} KiB')
"
        rm -rf "$TMPDIR_"
    done
done

echo
echo "Notes:"
echo "  - CK_us is mean kernel time across 50 iters (excludes 5 warmup)."
echo "  - VGPR / V_spill / AGPR / SGPR / Scratch / LDS come from the device ELF"
echo "    kernel descriptor (.note YAML). These are the numbers the GPU scheduler"
echo "    actually uses to size waves per SIMD; rocprofv3 mis-reports them on wave64."
