#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# FlyDSL vs Triton ISA-comparison harness for the MXFP4 sage attention kernel.
#
# Usage:
#   bash op_tests/flydsl_tests/profile_flydsl_vs_triton_mxfp4.sh
#
# Outputs:
#   /tmp/profiling/flydsl_fwd.s    -- FlyDSL final ISA
#   /tmp/profiling/triton_fwd.s    -- Triton .amdgcn for the same shape
#   /tmp/profiling/flydsl_dump/    -- All FlyDSL pipeline IR stages + LLVM IR
#
# Prints inner-loop per-iter instruction-mix diff (the actionable signal).

set -euo pipefail

WORKDIR=/tmp/profiling
mkdir -p "$WORKDIR"
rm -rf /root/.flydsl/debug ~/.triton/cache "$WORKDIR/flydsl_dump" 2>/dev/null || true

# Shape: B1 S8192 H8 D128 forward, q_smooth=False (representative of fwd gap).
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$WORKDIR/flydsl_dump" HIP_VISIBLE_DEVICES=0 python -c "
import torch
from aiter.ops.flydsl.sage_mxfp4_kernels import fav3_sage_mxfp4_flydsl_wrapper
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import fav3_sage_mxfp4_wrapper
torch.manual_seed(42)
q,k,v = [torch.randn(1,8192,8,128, dtype=torch.bfloat16, device='cuda') for _ in range(3)]
for _ in range(3):
    _ = fav3_sage_mxfp4_flydsl_wrapper(q,k,v, causal=False, layout='bshd', q_smooth=False)
    _ = fav3_sage_mxfp4_wrapper(q,k,v, causal=False, layout='bshd', hadamard_rotation=True, q_smooth=False)
torch.cuda.synchronize()
" >/dev/null

cp "$WORKDIR/flydsl_dump/sage_attn_kernel_0/21_final_isa.s" "$WORKDIR/flydsl_fwd.s"
cp ~/.triton/cache/*/sage_fwd_mxfp4.amdgcn "$WORKDIR/triton_fwd.s"

echo "=== ISA sizes ==="
wc -l "$WORKDIR/flydsl_fwd.s" "$WORKDIR/triton_fwd.s"

echo ""
echo "=== MFMA intrinsic variant (look for v4i32 vs v8i32) ==="
echo "Triton LLVM IR:"
grep -oE "@llvm.amdgcn.mfma.scale[a-zA-Z0-9_.]+" \
  ~/.triton/cache/*/sage_fwd_mxfp4.llir 2>/dev/null | sort -u | head -3
echo "FlyDSL LLVM IR:"
grep -oE "@llvm.amdgcn.mfma.scale[a-zA-Z0-9_.]+" \
  "$WORKDIR/flydsl_dump/sage_attn_kernel_0/20_llvm_ir.ll" | sort -u | head -3

echo ""
echo "=== FlyDSL inner-loop body (LBB0_5, body loop, lines 375-907) ==="
sed -n '375,907p' "$WORKDIR/flydsl_fwd.s" \
  | grep -oE "v_(mfma|mov|cmp|cndmask|add_f32|mul_f32|cvt|max)_[a-z0-9_]+|ds_[a-z0-9_]+|s_(waitcnt|barrier)[a-z0-9_]*" \
  | sort | uniq -c | sort -rn | head -20

echo ""
echo "=== Triton inner-loop body (LBB0_1, lines 429-1005) ==="
sed -n '429,1005p' "$WORKDIR/triton_fwd.s" \
  | grep -oE "v_(mfma|mov|cmp|cndmask|add_f32|mul_f32|cvt|max)_[a-z0-9_]+|ds_[a-z0-9_]+|s_(waitcnt|barrier)[a-z0-9_]*" \
  | sort | uniq -c | sort -rn | head -20

echo ""
echo "Done. Inspect ${WORKDIR}/flydsl_fwd.s and ${WORKDIR}/triton_fwd.s manually."
