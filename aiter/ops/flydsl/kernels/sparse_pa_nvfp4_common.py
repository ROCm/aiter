# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared pieces of the MiniMax-M3 NVFP4 sparse paged-attention kernels (FlyDSL, gfx950).

The page layout, the MFMA mapping and the conversion / softmax helpers the kernels share
(the decode kernel today; a prefill kernel reads the same page).

Page layout (128 tokens, ``2 * Hkv * 9216`` bytes; the global fp32 k / v scales are
NOT in the page). ``[K side | V side]``; a side is ``Hkv x 8192 B`` of e2m1 data
followed by ``Hkv x 1024 B`` of e4m3 scales. Inside one head, token t, dim d, scale
group j = d // 16, r = t % 32, n = 4 * (r // 8) + r % 4:

  K data   t * 64 + d // 2                    (low nibble = even dim)
  K scale  ((((t // 32) * 4 + j // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + j % 2
  V data   ((t // 8) * 128 + d) * 4 + (t % 8) // 2, nibble t % 2
  V scale  j * 128 + t

MFMA mapping (v_mfma_f32_16x16x32_f16, 4 waves, wave w owns keys 32w .. 32w+31):

  S^T = K Q^T    A = K tile, row r of tile st is key 8 (r // 4) + 4 st + r % 4,
                 B = Q^T (column = query head).  Lane (n = lane % 16, g = lane // 16)
                 of the accumulator holds keys 8 g + 4 st + e (e < 4) of head n, so
                 its 8 scores are 8 consecutive keys.
  O^T += V^T P^T A = V^T tile dt (row n is dim 8 n + dt), B = P^T taken unchanged from
                 the S^T accumulators: lane (n, g) supplies keys 8 g .. 8 g + 7.

K and V elements are the exact products e2m1 * e4m3 scale, which f16 holds exactly
(at most 6 significant bits, 2^-10 .. 2688), so nothing is rounded before the MFMA.
"""

import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD

BLK = 128  # tokens per page == sparse block
D = 128
NW = 4
KPW = BLK // NW  # keys per wave and block
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # dwords one lane loads per page
DATA_B = BLK * D // 2  # e2m1 bytes per head and side
SCALE_B = BLK * D // 16  # e4m3 scale bytes per head and side
HEAD_B = DATA_B + SCALE_B
LOG2E = 1.4426950409
# Dword index of a masked buffer store (byte 0x7FFFFFFC): past any buffer, so the write is dropped.
OOB_DWORD = 0x1FFFFFFF
# Cache-policy bits of buffer loads / stores: sc1 is coherent across the XCDs' L2s.
SC1 = 0x10
# The first 16 dwords of kernel arguments arrive in SGPRs
# (AITER_FLYDSL_KERNARG_PRELOAD=0 turns that off, as for the other FlyDSL kernels).
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def require(cond, msg):
    """Host-side input check that survives ``python -O`` (an ``assert`` does not): the
    kernels take raw pointers, so a refused argument must never reach the launch."""
    if not cond:
        raise ValueError(msg)


def global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def global_f32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.f32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def as_f32(x):
    return fx.Int32(x).bitcast(fx.Float32)


def maxf_nn(a, b):
    """v_max_f32 with nnan (without it LLVM canonicalizes both inputs first)."""
    return fx.arith.maxnumf(
        fx.Float32(a), fx.Float32(b), fastmath=fx.arith.FastMathFlags.nnan
    )


def exp2(x):
    """v_exp_f32 without the denormal range reduction (``afn``)."""
    return fx.math.exp2(fx.Float32(x), fastmath=fx.arith.FastMathFlags.afn)


def _permlane_swap(op, a, b):
    # Raw rocdl op + llvm.extractvalue: FlyDSL 0.5.0 has no typed permlane*_swap, and the
    # typed fx.gpu.shuffle_xor (xor 32, then 16) compiles to more code: 22 more instructions
    # in the decode kernel (1249 -> 1271 unsplit: v_cmp_eq / v_cndmask / v_mov), same bits;
    # timing on the op-test perf shapes was 1.000 and 1.015 of this version (geomean, two
    # interleaved runs of 5 and 7), so it is not faster either.
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(a), fx.as_ir_value(b), False, False)
    return (
        fx.Int32(_llvm.extractvalue(T.i32, res, [0])),
        fx.Int32(_llvm.extractvalue(T.i32, res, [1])),
    )


def xlane4(x, op):
    """``op`` over lanes L, L ^ 16, L ^ 32, L ^ 48, broadcast to all four."""
    a, b = _permlane_swap(fx.rocdl.permlane32_swap, bits(x), bits(x))
    m = op(as_f32(a), as_f32(b))
    a, b = _permlane_swap(fx.rocdl.permlane16_swap, bits(m), bits(m))
    return op(as_f32(a), as_f32(b))


def pk_f16(a, b):
    """Two fp32 -> one dword of two f16 (RNE), ``a`` low."""
    return (
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
    )


def pk_bf16(a, b):
    """Two fp32 -> one dword of two bf16 (RNE), ``a`` low (v_cvt_pk_bf16_f32)."""
    return (
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def bf16x8_to_f16(v):
    """4 dwords of 8 bf16 -> 4 dwords of 8 f16 (exact unless |x| > 65504)."""
    return Vec(v).bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)


def cvt_fp4x2_f16(word, sel):
    """The two e2m1 codes of byte ``sel`` of ``word`` as two f16 (exact)."""
    # Raw rocdl op: FlyDSL 0.5.0 has no working typed e2m1 -> f16 conversion;
    # Vec.bitcast(fx.Float4E2M1FN).to(fx.Float16) fails to compile ("LLVM Translation
    # failed for operation: builtin.unrealized_conversion_cast", then a segfault).
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp4(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def cvt_fp8x2_f16(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as two f16 (exact)."""
    # Raw rocdl op, for the same reason as cvt_fp4x2_f16: the typed
    # Vec.bitcast(fx.Float8E4M3FN).to(fx.Float16) fails with the same LLVM translation error.
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def mfma_f16(a4, b4, acc):
    """v_mfma_f32_16x16x32_f16 on 4-dword operands: ``acc + A B`` with A = ``a4``, B = ``b4``."""
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float16))
    a = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float16)
    a.store(Vec(a4).bitcast(fx.Float16))
    b = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float16)
    b.store(Vec(b4).bitcast(fx.Float16))
    c = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
    c.store(acc)
    fx.gemm(atom, c, a, b, c)
    return Vec(c.load())
