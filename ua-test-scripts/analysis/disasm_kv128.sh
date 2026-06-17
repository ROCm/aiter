#!/usr/bin/env bash
# Device-only compile + interleaved-source disasm for a single UA instance,
# bypassing the broken roc-obj-ls (missing File::Which). Buckets scratch_load /
# scratch_store by the C++ source line they map to, so we can see exactly where
# the register allocator spills/reloads (e.g. refresh_{k,v}_offsets).
set -euo pipefail
INSTANCE="${INSTANCE:-unified_attention_d128_bf16_mask_ps64}"
ARCH="${ARCH:-gfx950}"
HERE="$(cd "$(dirname "$0")" && pwd)"
AITER_ROOT="$(dirname "$(dirname "$HERE")")"   # analysis/ -> ua-test-scripts -> repo root
CK="$AITER_ROOT/3rdparty/composable_kernel"
SRC="$CK/example/ck_tile/42_unified_attention/instances/${INSTANCE}.cpp"
OUT="${OUT:-$HERE/isa_analysis_g/$INSTANCE}"
mkdir -p "$OUT"
[[ -f "$SRC" ]] || { echo "instance src not found: $SRC" >&2; exit 1; }

CFLAGS=(
  -DWITH_HIP -D_GLIBCXX_USE_CXX11_ABI=1 -DTORCH_EXTENSION_NAME=module_unified_attention
  -I"$CK/../ck_helper" -I"$CK/include" -I"$CK/library/include"
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
  -isystem /opt/rocm/include -isystem /usr/include/python3.12
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
  -Wno-switch-bool -Wno-undefined-func-template -Wno-unused-result -Wno-vla-cxx-extension
  -fgpu-flush-denormals-to-zero -fno-offload-uniform-block
  -mllvm --amdgpu-kernarg-preload-count=16 -mllvm --lsr-drop-solution=1
  -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false
  -mllvm -enable-post-misched=0 -fno-gpu-rdc -g
  --cuda-device-only
  ${XCFLAGS:-}
)
echo "[1/3] device-only -S compile $INSTANCE  XCFLAGS='${XCFLAGS:-}'"
/opt/rocm/bin/hipcc "${CFLAGS[@]}" -S "$SRC" -o "$OUT/dev.s" 2> "$OUT/build.err" || {
  echo "BUILD FAILED"; tail -20 "$OUT/build.err"; exit 2; }
echo "[2/3] resource metadata (vgpr/spill)"
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
echo "[3/3] spill buckets by source line (.loc/.file)"
python3 - "$OUT/dev.s" <<'PY'
import re, sys
from collections import Counter
path=sys.argv[1]
files={}
file_re=re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
loc_re=re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)')
ld=Counter(); st=Counter(); cur=None
for line in open(path):
    m=file_re.match(line)
    if m:
        idx=int(m.group(1)); a=m.group(2); b=m.group(3)
        files[idx]=(b if b else a)
        continue
    m=loc_re.match(line)
    if m:
        fi=int(m.group(1)); ln=m.group(2)
        fn=files.get(fi,str(fi)).rsplit('/',1)[-1]
        cur=f"{fn}:{ln}"
        continue
    s=line.strip()
    if s.startswith('scratch_load'):  ld[cur]+=1
    elif s.startswith('scratch_store'): st[cur]+=1
print(f"scratch_load total: {sum(ld.values())}   scratch_store total: {sum(st.values())}")
print("-- scratch_load by src (top 15) --")
for k,v in ld.most_common(15): print(f"  {v:>4}  {k}")
print("-- scratch_store by src (top 15) --")
for k,v in st.most_common(15): print(f"  {v:>4}  {k}")
PY
