# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared pieces of the MiniMax-M3 NVFP4 sparse paged-attention kernels (FlyDSL, gfx950).

Both kernels (decode and prefill) read the same single-buffer K/V page and share the
MFMA mapping, so the page loader, the exact dequantization and the softmax helpers
live here.

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
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

BLK = 128  # tokens per page == sparse block
D = 128
HG = 16  # MFMA N: query heads of one kv head (group 8 runs with half the lanes masked)
NW = 4
KPW = BLK // NW  # keys per wave and block
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # dwords one lane loads per page
DATA_B = BLK * D // 2  # e2m1 bytes per head and side
SCALE_B = BLK * D // 16  # e4m3 scale bytes per head and side
HEAD_B = DATA_B + SCALE_B
LOG2E = 1.4426950409
# Cache-policy bits of buffer loads / stores: sc1 is coherent across the XCDs' L2s.
SC1 = 0x10
# The first 16 dwords of kernel arguments arrive in SGPRs.
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


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


def _nnan():
    return _ir.Attribute.parse("#arith.fastmath<nnan>")


def maxf_nn(a, b):
    """v_max_f32 with nnan (without it LLVM canonicalizes both inputs first)."""
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=_nnan()).result
    )


def exp2(x):
    """v_exp_f32 without the denormal range reduction."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _permlane_swap(op, a, b):
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
    """Two fp32 -> one dword of two bf16 (RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def bf16x8_to_f16(v):
    """4 dwords of 8 bf16 -> 4 dwords of 8 f16 (exact unless |x| > 65504)."""
    return Vec(v).bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)


def cvt_fp4x2_f16(word, sel):
    """The two e2m1 codes of byte ``sel`` of ``word`` as two f16 (exact)."""
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
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def mfma_f16(a4, b4, acc):
    """v_mfma_f32_16x16x32_f16 on 4-dword operands."""
    a = Vec(a4).bitcast(fx.Float16)
    b = Vec(b4).bitcast(fx.Float16)
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_f16(
            Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]
        )
    )


def page_loader(arg_cache, page_bytes, kvh, hkv, w, n, g):
    """Returns ``load(page) -> RAW dwords`` of this lane's share of a page.

    raw[0:4], raw[4:8]  K data of key rows st = 0, 1 (dims 32 g .. 32 g + 31)
    raw[8]              K scale bytes [st][group] of those rows
    raw[9:17]           V data, dword dt = 8 keys (8 g .. 8 g + 7) of dim 8 n + dt
    raw[17:19]          V scale bytes of keys 8 g .. 8 g + 7, dim group n // 2
    """
    side_b = hkv * HEAD_B
    k_data = kvh * DATA_B
    k_scl = hkv * DATA_B + kvh * SCALE_B
    v_data = side_b + kvh * DATA_B
    v_scl = side_b + hkv * DATA_B + kvh * SCALE_B
    k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
    k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
    k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
    v_load = (v_data + ((w * 4 + g) * BLK + 8 * n) * 4) // 4
    v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

    def load(page):
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_cache + fx.Int64(page) * fx.Int64(page_bytes),
            num_records_bytes=page_bytes,
        )
        raw = []
        for st in range_constexpr(2):
            kd = fx.Vector(
                buffer_ops.buffer_load(pr, k_load[st], vec_width=4, dtype=fx.Int32)
            )
            raw += [kd[i] for i in range_constexpr(4)]
        raw.append(
            fx.Int32(
                buffer_ops.buffer_load(pr, k_scale_load, vec_width=1, dtype=fx.Int32)
            )
        )
        for h in range_constexpr(2):
            vd = fx.Vector(
                buffer_ops.buffer_load(pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32)
            )
            raw += [vd[i] for i in range_constexpr(4)]
        vs = fx.Vector(
            buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
        )
        raw += [vs[i] for i in range_constexpr(2)]
        return raw

    return load


def dequant_page(raw):
    """K frags ``kf[st][ks]`` (A operands of S^T) and V^T frags ``vf[dt]`` in f16.

    Each is cvt(e2m1 pair) * scale pair: one convert and one packed multiply per two
    elements, and the product is exact in f16."""
    kf = []
    for st in range_constexpr(2):
        ksc = cvt_fp8x2_f16(raw[8], st == 1)  # [group 2g, group 2g + 1] of row st
        row = []
        for ks in range_constexpr(NKS):
            sv = Vec.from_elements([ksc[ks // 2], ksc[ks // 2]], fx.Float16)
            row.append(
                Vec.from_elements(
                    [
                        (cvt_fp4x2_f16(raw[st * 4 + ks], i) * sv).bitcast(fx.Int32)[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
            )
        kf.append(row)
    vs = []
    for h in range_constexpr(2):
        vs += [cvt_fp8x2_f16(raw[17 + h], False), cvt_fp8x2_f16(raw[17 + h], True)]
    # A V byte holds two consecutive keys: already the operand's f16 pair.
    vf = [
        Vec.from_elements(
            [
                (cvt_fp4x2_f16(raw[9 + dt], i) * vs[i]).bitcast(fx.Int32)[0]
                for i in range_constexpr(4)
            ],
            fx.Int32,
        )
        for dt in range_constexpr(NDT)
    ]
    return kf, vf
