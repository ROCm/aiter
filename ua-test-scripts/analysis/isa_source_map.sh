#!/usr/bin/env bash
# Compile a SINGLE unified-attention instance with DWARF line tables and
# disassemble its gfx950 code object with interleaved C++ source, so we can
# map hot ISA (e.g. `v_readfirstlane_b32 s2, v0`) back to the .hpp line that
# generated it.
#
# Why single-instance: the full module_unified_attention .so is ~100 TUs and
# takes tens of minutes to build. The prefill-fp8 hot dispatch is ONE instance
# (dynamic page size, masked): unified_attention_d128_fp8_mask.cpp. Building
# just that TU gives the exact ISA for the kernel we profile.
#
# -gline-tables-only (not full -g): keeps -O3 codegen identical to the JIT
# build (so the ISA is representative) and only adds .debug_line, which is all
# llvm-objdump --source / rocprofv3's source-snapshot need.
#
# Output:
#   $OUT/instance.o            fat object
#   $OUT/*gfx950*.co           extracted device code object
#   $OUT/isa_source.txt        llvm-objdump -d --source (ISA interleaved w/ C++)
#   $OUT/readfirstlane.txt     just the v_readfirstlane sites + their source

set -euo pipefail

INSTANCE="${INSTANCE:-unified_attention_d128_fp8_mask}"
ARCH="${ARCH:-gfx950}"
HERE="$(cd "$(dirname "$0")" && pwd)"
AITER_ROOT="$(dirname "$(dirname "$HERE")")"   # analysis/ -> ua-test-scripts -> repo root
CK="$AITER_ROOT/3rdparty/composable_kernel"
SRC="$CK/example/ck_tile/42_unified_attention/instances/${INSTANCE}.cpp"
OUT="${OUT:-$HERE/isa_analysis/$INSTANCE}"
mkdir -p "$OUT"

[[ -f "$SRC" ]] || { echo "instance src not found: $SRC" >&2; exit 1; }

# Exact JIT cuda_cflags (copied from build.ninja for module_unified_attention),
# with --offload-arch=native pinned to $ARCH and -gline-tables-only added.
CFLAGS=(
  -DWITH_HIP -D_GLIBCXX_USE_CXX11_ABI=1 -DTORCH_EXTENSION_NAME=module_unified_attention
  -I"$CK/../ck_helper"
  -I"$CK/include"
  -I"$CK/library/include"
  -I"$AITER_ROOT/csrc/include"
  -I"$AITER_ROOT/aiter/jit/build/module_unified_attention/blob"
  -I"$CK/example/ck_tile/42_unified_attention"
  -I"$AITER_ROOT/csrc/include/torch"
  -I/venv/lib/python3.12/site-packages/pybind11/include
  -isystem /venv/lib/python3.12/site-packages/torch/include/TH
  -isystem /venv/lib/python3.12/site-packages/torch/include/THC
  -isystem /venv/lib/python3.12/site-packages/torch/include
  -isystem /venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
  -isystem /venv/lib/python3.12/site-packages/torch/include/THH
  -isystem /opt/rocm/include
  -isystem /usr/include/python3.12
  -fPIC -std=c++20 -O3 -Wno-unknown-warning-option
  -DENABLE_CK=1 -DENABLE_ROPE_POSITIONS_INT32=0
  -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1
  -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1
  -mcmodel=large -fno-unique-section-names -ffunction-sections -fdata-sections
  -fvisibility=hidden -fvisibility-inlines-hidden
  --offload-arch="$ARCH"
  -DDLLVM_MAIN_REVISION=554785 -DLEGACY_HIPBLAS_DIRECT -DTORCH_Float4_e2m1fn_x2
  -DUSE_PROF_API=1 -D__Float4_e2m1fn_x2 -D__HIP_PLATFORM_HCC__=1
  -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__
  -Wno-macro-redefined -Wno-missing-template-arg-list-after-template-kw
  -Wno-switch-bool -Wno-undefined-func-template -Wno-unused-result
  -Wno-vla-cxx-extension
  -fgpu-flush-denormals-to-zero -fno-offload-uniform-block
  -mllvm --amdgpu-kernarg-preload-count=16
  -mllvm --lsr-drop-solution=1
  -mllvm -amdgpu-early-inline-all=true
  -mllvm -amdgpu-function-calls=false
  -mllvm -enable-post-misched=0
  -fno-gpu-rdc
  -gline-tables-only
)

echo "[1/4] compiling $INSTANCE ($ARCH) with line tables ..."
echo "      src: $SRC"
time /opt/rocm/bin/hipcc "${CFLAGS[@]}" -c "$SRC" -o "$OUT/instance.o"

echo "[2/4] extracting $ARCH device code object ..."
# roc-obj-ls needs a Perl module that isn't installed here; extract the
# embedded fatbin section and unbundle the gfx950 code object directly.
/opt/rocm/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin="$OUT/fat.bin" "$OUT/instance.o"
/opt/rocm/llvm/bin/clang-offload-bundler --type=o --input="$OUT/fat.bin" \
    --unbundle --targets=hipv4-amdgcn-amd-amdhsa--${ARCH} --output="$OUT/device_${ARCH}.co"
CO="$OUT/device_${ARCH}.co"
[[ -s "$CO" ]] || { echo "could not extract $ARCH code object" >&2; exit 1; }
echo "      code object: $CO"

echo "[3/4] disassembling with interleaved source ..."
/opt/rocm/llvm/bin/llvm-objdump -d --source --no-show-raw-insn "$CO" > "$OUT/isa_source.txt" 2>/dev/null
echo "      wrote $OUT/isa_source.txt ($(wc -l < "$OUT/isa_source.txt") lines)"

echo "[4/4] extracting v_readfirstlane sites with source context ..."
grep -n "v_readfirstlane" "$OUT/isa_source.txt" | head -50 > "$OUT/readfirstlane_lines.txt" || true
echo "      readfirstlane occurrences: $(grep -c v_readfirstlane "$OUT/isa_source.txt" || echo 0)"
echo "[done] $OUT"
