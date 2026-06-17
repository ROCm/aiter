#!/usr/bin/env bash
# Fast VGPR/spill probe for a SINGLE unified-attention instance.
# Emits device GCN assembly (-S, device-only) and scrapes the per-kernel
# resource metadata footer (.vgpr_count / .vgpr_spill_count / .sgpr_count /
# .agpr_count / .group_segment_fixed_size). Far faster than the full JIT build.
#
# Usage:
#   ua-test-scripts/measure_vgpr.sh                       # baseline prefill fp8
#   XCFLAGS="-DUA_FOO=1" ua-test-scripts/measure_vgpr.sh  # with extra defines
#   INSTANCE=unified_attention_d128_fp8_mask LABEL=kv128 ... 
set -euo pipefail

INSTANCE="${INSTANCE:-unified_attention_d128_fp8_mask}"
ARCH="${ARCH:-gfx950}"
LABEL="${LABEL:-baseline}"
HERE="$(cd "$(dirname "$0")" && pwd)"
AITER_ROOT="$(dirname "$(dirname "$HERE")")"   # analysis/ -> ua-test-scripts -> repo root
CK="$AITER_ROOT/3rdparty/composable_kernel"
SRC="$CK/example/ck_tile/42_unified_attention/instances/${INSTANCE}.cpp"
OUT="${OUT:-$HERE/vgpr_probe/$LABEL}"
mkdir -p "$OUT"

[[ -f "$SRC" ]] || { echo "instance src not found: $SRC" >&2; exit 1; }

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
  ${XCFLAGS:-}
)

echo "[build:$LABEL] $INSTANCE ($ARCH)  XCFLAGS='${XCFLAGS:-}'"
t0=$(date +%s)
/opt/rocm/bin/hipcc "${CFLAGS[@]}" --cuda-device-only -S "$SRC" -o "$OUT/dev.s" 2> "$OUT/build.err" || {
  echo "  BUILD FAILED — tail of build.err:"; tail -20 "$OUT/build.err"; exit 2; }
t1=$(date +%s)
echo "  compiled in $((t1-t0))s -> $OUT/dev.s"

echo "[metadata:$LABEL]"
# The AMDGPU metadata YAML footer lists per-kernel resource usage (one list
# entry per kernel). Keys are alphabetical; .vgpr_spill_count is last, so print
# on that. Skip the trivial flush_cache helper.
awk '
  /^  - \.agpr_count:/        {a=$3; have=1}
  have && /\.sgpr_count:/     {s=$2}
  have && /\.vgpr_count:/     {v=$2}
  have && /\.group_segment_fixed_size:/ {lds=$2}
  have && /\.private_segment_fixed_size:/ {priv=$2}
  have && /\.name:/           {name=$2}
  have && /\.vgpr_spill_count:/ {vs=$2;
      if (name !~ /flush_cache/)
        printf "  vgpr=%-4s spill=%-4s sgpr=%-4s agpr=%-4s lds=%-7s scratch=%-6s\n", v,vs,s,a,lds,priv;
      have=0}
' "$OUT/dev.s"
