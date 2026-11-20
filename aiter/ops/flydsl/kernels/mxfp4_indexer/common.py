# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Constants and FlyDSL helpers of the MXFP4 indexer kernels: the scaled MFMA, a token's logit from its accumulator,
DPP / permlane lane exchange, the wave scan and the order-preserving fp32 keys."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

H = 32  # index heads
D = 128  # index head dim
ENTRY_BYTES = D // 2 + D // 32  # e2m1 codes, then the e8m0 scales
BLOCK = 8  # tokens per candidate block
TOPK = 512
CANDIDATES = 2048  # candidate blocks of the two-level indexer
NBINS = 4096  # buckets of the select's histogram

F, I, I64, U = fx.Float32, fx.Int32, fx.Int64, fx.Uint32
NEG_INF = float("-inf")
INF = float("inf")


def bi(x):
    return x.bitcast(I)


def bf(x):
    return x.bitcast(F)


def ushr(a, n):
    return (a.bitcast(U) >> U(n)).bitcast(I)


def dpp(src, ctrl, old=None, bank_mask=0xF, row_mask=0xF, bound=True):
    """``v_mov_b32_dpp`` of an i32. FlyDSL 0.5.0 has no typed DPP operation
    (``shuffle_xor`` lowers to ``ds_bpermute`` through LDS), so this emits
    ``rocdl.update_dpp``; ``old`` defaults to an undefined value (raw
    ``llvm.mlir_undef``: there is no fx undef)."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    return I(
        fx.rocdl.update_dpp(
            T.i32, o, fx.as_ir_value(src), ctrl, row_mask, bank_mask, bound
        )
    )


def _swap_pair(swap, a, b):
    """``v_permlane16/32_swap`` of two i32 (no typed FlyDSL 0.5.0 API): the two
    results, unpacked from the op's ``{i32, i32}`` struct."""
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = swap(pair, fx.as_ir_value(a), fx.as_ir_value(b), False, False)
    return I(_llvm.extractvalue(T.i32, res, [0])), I(
        _llvm.extractvalue(T.i32, res, [1])
    )


def xor_lane(v, m, lane):
    """The i32 ``v`` of lane ``lane ^ m`` (m a power of two below 64)."""
    if m == 1:
        return dpp(v, 0xB1)
    if m == 2:
        return dpp(v, 0x4E)
    if m == 4:
        return dpp(v, 0x104, old=dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return dpp(v, 0x128)
    op = fx.rocdl.permlane16_swap if m == 16 else fx.rocdl.permlane32_swap
    a, b = _swap_pair(op, v, v)
    return ((lane & m) != 0).select(a, b)


def wave_scan_incl(v):
    """Inclusive prefix sum of an i32 across the 64 lanes."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        v = v + dpp(v, ctrl)
    v = v + dpp(v, 0x142, old=I(0), row_mask=0xA)
    return v + dpp(v, 0x143, old=I(0), row_mask=0xC)


def maxf(a, b):
    """``v_max_f32`` (``nnan``: no NaN canonicalisation around it)."""
    fm = ir.Attribute.parse("#arith.fastmath<nnan>")
    return F(fx.arith.maxnumf(a, b, fastmath=fm))


def minf(a, b):
    fm = ir.Attribute.parse("#arith.fastmath<nnan>")
    return F(fx.arith.minnumf(a, b, fastmath=fm))


def not_nan(x):
    """False for a NaN fp32 (a traced IR compare, not a Python one)."""
    return x == x  # noqa: PLR0124


def key_of(v):
    """Order-preserving u32 key of an fp32 (as i32 bits); 0 for NaN and -inf, which no real logit maps to."""
    b = v.bitcast(I)
    k = (b < I(0)).select(~b, b ^ I(-2147483648))
    return (not_nan(v) & (v > F(NEG_INF))).select(k, I(0))


def gptr(addr, align=16, ty=None):
    """Global pointer at a per-lane device address (``fx.generic_load``)."""
    ptr_ty = fx.PointerType.get(
        ty or T.i32, address_space=fx.AddressSpace.Global, alignment=align
    )
    return fx.inttoptr(ptr_ty, I64(addr))


def mfma_fp4(a4, b4, acc, scale_a, scale_b, opsel_b=0):
    """``v_mfma_scale_f32_32x32x64_f8f6f4`` on e2m1 x e2m1: lane (r, h) of ``a4`` / ``b4`` holds 16 bytes (32 codes) of
    row / column r, K elements 32 h .. 32 h + 31 of the 64 of the step; ``scale_a`` is that lane's e8m0 scale,
    ``scale_b`` a dword whose byte ``opsel_b`` is the column's. Lane (c, h) of the result holds rows
    8 (i // 4) + 4 h + i % 4, column c."""
    fp4 = 4  # matrix format fp4_e2m1, for A (cbsz) and B (blgp)

    def pad(v):
        v, z = Vec(v), I(0)
        return Vec.from_elements([v[0], v[1], v[2], v[3], z, z, z, z], I)

    return Vec(
        fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            Vec.make_type(16, F),
            [pad(a4), pad(b4), acc, fp4, fp4, 0, I(scale_a), opsel_b, I(scale_b)],
        )
    )


def token_logit(acc, w):
    """sum_h w_h relu(acc_h) of a lane's 16 head rows (four fma chains, summed in a fixed order) plus the other half
    wave's (lane ^ 32): every lane ends with its token's logit. The same arithmetic in every kernel, so a token scores
    to the same bits wherever it is computed."""
    s = [None] * 4
    for i in range_constexpr(16):
        r = maxf(acc[i], F(0.0))
        s[i % 4] = w[i] * r if i < 4 else fx.math.fma(w[i], r, s[i % 4])
    p = (s[0] + s[1]) + (s[2] + s[3])
    return _add_halves(p)


def _add_halves(p):
    a, b = _swap_pair(fx.rocdl.permlane32_swap, bi(p), bi(p))
    return bf(a) + bf(b)


def pair_logit(acc_a, acc_b, w):
    """The logits of two tiles' tokens: lanes 0-31 of the result the tile A's (as ``token_logit``, same bits), lanes
    32-63 the tile B's, with one lane swap for both."""

    def partial(acc):
        s = [None] * 4
        for i in range_constexpr(16):
            r = maxf(acc[i], F(0.0))
            s[i % 4] = w[i] * r if i < 4 else fx.math.fma(w[i], r, s[i % 4])
        return (s[0] + s[1]) + (s[2] + s[3])

    pa, pb = partial(acc_a), partial(acc_b)
    a, b = _swap_pair(fx.rocdl.permlane32_swap, bi(pa), bi(pb))
    return bf(a) + bf(b)
