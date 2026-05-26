#!/usr/bin/env bash
# Quick A/B harness for the prefill_d128 VGPR-pressure experiments. Run after
# you patch the kernel; clears the JIT cache, rebuilds, then measures
# prefill_d128 bf16 + fp8 at sk=75600. Captures CK ms, VGPR count, scratch
# size from rocprofv3 kernel-trace.
set -euo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
cd "$(dirname "$0")/.."

TAG="${1:-untagged}"

rm -f aiter/jit/module_unified_attention.so aiter/jit/build/module_unified_attention/build/*.o

for dt in bf16 fp8; do
    OUT=/tmp/probe_d128_${TAG}_${dt}
    rm -rf "$OUT"
    mkdir -p "$OUT"

    # First a 1-iter trace run to capture resource info.
    rocprofv3 --kernel-trace -d "$OUT" -o trace -f csv -- \
      python3 ua-test-scripts/test_single_shape.py \
          -b 1 -sq 75600 -sk 75600 -hq 64 -hk 8 -d 128 \
          --dtype "$dt" --only-ck --warmup 2 --iters 3 --seed 42 \
      > "$OUT/run.log" 2>&1

    # Then the timed run (no rocprof, so timings aren't tainted).
    t=$(python3 ua-test-scripts/test_single_shape.py \
          -b 1 -sq 75600 -sk 75600 -hq 64 -hk 8 -d 128 \
          --dtype "$dt" --only-ck --warmup 5 --iters 30 --seed 42 \
        | grep -E "^\s*CK time:" | awk '{print $3}')

    # Extract resources from the trace CSV (rocprof is unreliable on gfx950 wave64;
    # it reports half the real VGPR count and a spurious 64 B scratch). Fall back
    # to the device ELF kernel descriptor for authoritative numbers.
    CLANGBNDL=/opt/rocm/lib/llvm/bin/clang-offload-bundler
    LLVMREADELF=/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/llvm-readelf
    OBJ=aiter/jit/build/module_unified_attention/build/unified_attention_d128_${dt}_nmask.cuda.o
    TMPDIR=$(mktemp -d) && trap "rm -rf $TMPDIR" EXIT
    objcopy --dump-section .hip_fatbin=$TMPDIR/fatbin.bin "$OBJ" 2>/dev/null
    $CLANGBNDL --type=o --unbundle --input=$TMPDIR/fatbin.bin \
        --output=$TMPDIR/gpu.o --output=$TMPDIR/host.o \
        --targets=hipv4-amdgcn-amd-amdhsa--gfx950,host-x86_64-unknown-linux-gnu- 2>/dev/null

    $LLVMREADELF --notes $TMPDIR/gpu.o 2>&1 | TAG="$TAG" DT="$dt" CK_T="$t" python3 -c "
import sys, re, os
text = sys.stdin.read()
entries = re.findall(r'- \.agpr_count:\s+(\d+).*?\.group_segment_fixed_size:\s+(\d+).*?\.name:\s+(\S+).*?\.private_segment_fixed_size:\s+(\d+).*?\.sgpr_count:\s+(\d+).*?\.sgpr_spill_count:\s+(\d+).*?\.vgpr_count:\s+(\d+).*?\.vgpr_spill_count:\s+(\d+)', text, re.DOTALL)
for agpr, lds, name, scr, sgpr, ss, vgpr, vs in entries:
    if 'kentry' not in name: continue
    tag, dt, t = os.environ['TAG'], os.environ['DT'], os.environ['CK_T']
    print(f'  [{tag:<26s} | {dt:>4s}]  CK={t:>8s} ms  VGPR={vgpr:>3s}  V_spill={vs:>3s}  AGPR={agpr:>2s}  SGPR={sgpr:>3s}  Scratch={scr:>4s} B  LDS={int(lds)//1024:>3d} KiB')
    break
"
done
