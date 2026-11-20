# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors


import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T


def update_dpp_i32(
    old,
    src,
    dpp_ctrl: int,
    row_mask: int = 0xF,
    bank_mask: int = 0xF,
    bound_ctrl: bool = False,
    **kw,
):
    """Wrapper for ``llvm.amdgcn.update.dpp.i32``.

    DPP controls are immediate operands. Common CDNA values:
    280/264 for row xor-8, 276/260 for row xor-4, 78 for xor-2,
    and 177 for xor-1 within a 16-lane row.
    """
    return fx.rocdl.update_dpp(
        fx.Int32.ir_type,
        fx.Int32(old).ir_value(),
        fx.Int32(src).ir_value(),
        dpp_ctrl,
        row_mask,
        bank_mask,
        bound_ctrl,
        **kw,
    )


def dpp_xor_f32(src, offset: int, **kw):
    """Return ``src`` from the lane selected by a 16-lane XOR DPP pattern."""
    from flydsl.expr.typing import T

    src_i32 = fx.Float32(src).bitcast(fx.Int32)
    if offset == 8:
        out_i32 = update_dpp_i32(src_i32, src_i32, 280, 0xF, 0xC, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 264, 0xF, 0x3, False, **kw)
    elif offset == 4:
        out_i32 = update_dpp_i32(src_i32, src_i32, 276, 0xF, 0xA, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 260, 0xF, 0x5, False, **kw)
    elif offset == 2:
        out_i32 = update_dpp_i32(src_i32, src_i32, 78, 0xF, 0xF, False, **kw)
    elif offset == 1:
        out_i32 = update_dpp_i32(src_i32, src_i32, 177, 0xF, 0xF, False, **kw)
    else:
        raise ValueError(
            f"dpp_xor_f32 only supports 16-lane offsets 1, 2, 4, 8; got {offset}"
        )
    return out_i32.bitcast(T.f32)


def permlane_swap_i32(swap, a, b):
    """``v_permlane16_swap_b32`` / ``v_permlane32_swap_b32`` of two i32 values.

    ``swap`` is ``fx.rocdl.permlane16_swap`` or ``fx.rocdl.permlane32_swap``;
    returns the op's two results (its ``{i32, i32}`` struct) as ``fx.Int32``.
    Raw rocdl op + ``llvm.extractvalue``: FlyDSL 0.5.0 has no typed permlane
    swap, and ``fx.gpu.shuffle_xor`` lowers to ``ds_bpermute_b32`` (an LDS
    round trip) or a longer select sequence instead of one VALU op.
    """
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = swap(pair, fx.as_ir_value(a), fx.as_ir_value(b), False, False)
    return (
        fx.Int32(llvm.extractvalue(T.i32, res, [0])),
        fx.Int32(llvm.extractvalue(T.i32, res, [1])),
    )
