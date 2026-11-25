# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Technique source (credit only, no import): amd-kernels attention/norm_rope_nvfp4_insert_d128_rot64_p128/kernel.py
# (Apache-2.0, Copyright contributors to the amd-kernels project).

"""Device helpers shared by the MiniMax-M3.1 QKV front kernels (``qkv_chain_cooperative``, ``qkv_reduce_insert``):
bf16 / e4m3 / e2m1 conversions, lane butterflies, the NVFP4 group quantizer and the NVFP4 scale-region addressing.
"""

import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

D = 128


BLK = 128  # tokens per NVFP4 page


# kernel arguments preloaded into SGPRs (the first 56 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


RCP6 = 0x3E2AAAAB  # RN(1 / 6), the Triton insert's constant


F448 = 0x43E00000  # 448.0


F448N = 0xC3E00000 - (1 << 32)  # -448.0


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _bf16x4(v):
    """4 fp32 of a dword pair of bf16 (low half first)."""
    w0, w1 = fx.Int32(v[0]), fx.Int32(v[1])
    return [
        (w0 << 16).bitcast(fx.Float32),
        (w0 & -65536).bitcast(fx.Float32),
        (w1 << 16).bitcast(fx.Float32),
        (w1 & -65536).bitcast(fx.Float32),
    ]


def _pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    """v_mov_b32_dpp of an i32 (lanes a bank mask leaves out keep ``old``; an undef
    ``old`` lets the move fold into its consumer)."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(
        _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full)
    )


def _swap16(v):
    """v_permlane16_swap of ``v`` with itself: (row 0 / 2 value, row 1 / 3 value)
    of each 32-lane half, i.e. (own, lane ^ 16) in rows 0 / 2 and swapped in 1 / 3."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl_ops.permlane16_swap(
        pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False
    )
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )


def _xor_lane(v, m, hi16):
    """Lane ``l ^ m`` (m = 1, 2, 4, 8, 16) of the 32-lane half, as an i32. ``hi16``:
    this lane's bit 4."""
    if m == 1:
        return _dpp(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    if m == 2:
        return _dpp(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    if m == 4:  # row_shr:4 into banks 1, 3, then row_shl:4 into banks 0, 2
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)  # row_ror:8
    assert m == 16
    a, b = _swap16(v)
    return hi16.select(a, b)


def _xsum(x, m):
    """Adds lane l ^ m to ``x`` (a commutative fold: the xor-16 pair is unordered)."""
    if m == 16:
        a, b = _swap16(_bits(x))
        return a.bitcast(fx.Float32) + b.bitcast(fx.Float32)
    return x + _xor_lane(_bits(x), m, None).bitcast(fx.Float32)


def _xmax(x, m):
    return _maxnum(x, _xor_lane(_bits(x), m, None).bitcast(fx.Float32))


def _rsq(x):
    """The rsqrt of ocml (v_rsq_f32): its input scaling below 2^-126 never applies,
    as x = fma(sum of squares, 1/128, eps) >= eps >= 2^-126 (the host checks eps)."""
    return fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(x)))


def _e4m3_byte(x):
    """RNE e4m3 of ``x`` (v_cvt_scalef32_pk_fp8_f32, scale 1), as an i32 byte."""
    pk = _rocdl_ops.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        fx.as_ir_value(Vec.filled(2, 0, fx.Int16)),
        fx.as_ir_value(x),
        fx.as_ir_value(x),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return Vec(pk).bitcast(fx.Int32)[0] & 0xFF


def _e4m3_value(byte):
    pk = _rocdl_ops.cvt_scalef32_pk_f32_fp8(
        T.vec(2, T.f32),
        fx.as_ir_value(fx.Int32(byte)),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return fx.Float32(Vec(pk)[0])


def _iq_fp8(a, b, old, hi):
    """Two fp32 -> two e4m3 bytes (v_cvt_pk_fp8_f32) in the low (``hi``: high) word
    of ``old``, ``a`` the lower byte."""
    return fx.Int32(
        _rocdl_ops.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            hi,
        )
    )


def _e2m1x2(a, b, old, sel):
    """E2M1 codes of ``a``, ``b`` (v_cvt_scalef32_pk_fp4_f32, scale 1) into byte ``sel``
    of ``old``, ``a`` the low nibble: bit for bit nvfp4_kv._f32_to_e2m1 on every fp32
    bit pattern."""
    return fx.Int32(
        _rocdl_ops.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _quant4(xs, gs, inv_gs):
    """E2M1 codes of this lane's 4 values (16 bits, xs[0] lowest) and the e4m3 byte
    of their 16-value group (lanes l ^ 0..3): nvfp4_kv._quantize_nvfp4."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    sf = _minnum((m * _f32(RCP6)) * inv_gs, _f32(F448))
    byte = _e4m3_byte(sf)
    sv = _e4m3_value(byte)
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte


def _gptr(addr, ty, align):
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


def _k_scale_off(tok, grp):
    """Byte of (token, group) in a head's main-K scale region (NVFP4_LAYOUT_MAIN_K)."""
    r = tok % 32
    n = (r // 8) * 4 + r % 4
    return ((((tok // 32) * 4 + grp // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + grp % 2
