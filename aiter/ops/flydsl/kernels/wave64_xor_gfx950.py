# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wave64 xor-lane exchange and butterfly reductions for gfx950.

``xor_lane_i32(v, m, lane)`` returns the value of lane ``lane ^ m`` for any power
of two ``m < 64``: DPP within a 16-lane row (xor 1, 2, 4, 8) and
``v_permlane16_swap`` / ``v_permlane32_swap`` across rows (xor 16, 32, gfx950).
No LDS and no ``ds_swizzle``; a butterfly built on it reduces in a fixed order,
so its result does not depend on the launch.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

# DPP controls: quad_perm [1, 0, 3, 2], quad_perm [2, 3, 0, 1], row_shl:4,
# row_shr:4 and row_ror:8.
_DPP_XOR1 = 0xB1
_DPP_XOR2 = 0x4E
_DPP_ROW_SHL4 = 0x104
_DPP_ROW_SHR4 = 0x114
_DPP_ROW_ROR8 = 0x128


def dpp_i32(src, ctrl: int, old=None, bank_mask: int = 0xF):
    """``v_mov_b32_dpp`` of an i32.

    ``dpp_utils.update_dpp_i32`` takes a defined ``old`` operand. The full-mask
    exchanges here write every lane, so ``old`` is left undef (raw
    ``llvm.mlir_undef``: fx has no undef constructor) with ``bound_ctrl`` set.
    """
    o = llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(
        fx.rocdl.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full)
    )


def _permlane_swap(op, v):
    # The op returns an llvm struct of both halves; fx has no typed wrapper.
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return (
        fx.Int32(llvm.extractvalue(T.i32, res, [0])),
        fx.Int32(llvm.extractvalue(T.i32, res, [1])),
    )


def xor_lane_i32(v, m: int, lane):
    """The i32 ``v`` of lane ``lane ^ m`` (``m`` a power of two below 64)."""
    if m == 1:
        return dpp_i32(v, _DPP_XOR1)
    if m == 2:
        return dpp_i32(v, _DPP_XOR2)
    if m == 4:
        return dpp_i32(
            v,
            _DPP_ROW_SHL4,
            old=dpp_i32(v, _DPP_ROW_SHR4, bank_mask=0xA),
            bank_mask=0x5,
        )
    if m == 8:
        return dpp_i32(v, _DPP_ROW_ROR8)
    if m not in (16, 32):
        raise ValueError(f"xor_lane_i32: m must be a power of two < 64, got {m}")
    swap = fx.rocdl.permlane16_swap if m == 16 else fx.rocdl.permlane32_swap
    a, b = _permlane_swap(swap, v)
    return ((lane & m) != 0).select(a, b)


def xor_lane(v, m: int, lane):
    """``xor_lane_i32`` of an fp32 or i32 value, same type out."""
    if v.dtype == fx.Float32:
        return xor_lane_i32(v.bitcast(fx.Int32), m, lane).bitcast(fx.Float32)
    return xor_lane_i32(v, m, lane)
