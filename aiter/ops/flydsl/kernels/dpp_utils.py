# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors


import flydsl.expr as fx


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


def dpp_xor_i32(src, offset: int, **kw):
    """Return the i32 ``src`` of lane ``lane ^ offset`` within a 16-lane row
    (``offset`` 1, 2, 4 or 8)."""
    src_i32 = fx.Int32(src)
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
            f"dpp_xor only supports 16-lane offsets 1, 2, 4, 8; got {offset}"
        )
    return fx.Int32(out_i32)


def dpp_xor_f32(src, offset: int, **kw):
    """Return ``src`` from the lane selected by a 16-lane XOR DPP pattern."""
    from flydsl.expr.typing import T

    src_i32 = fx.Float32(src).bitcast(fx.Int32)
    return dpp_xor_i32(src_i32, offset, **kw).bitcast(T.f32)


def xor_lane_i32(src, offset: int):
    """Return the i32 ``src`` of lane ``lane ^ offset`` (wave64) for any power
    of two ``offset`` below 64.

    Offsets 1, 2, 4 and 8 use the DPP moves of ``dpp_xor_i32``: on gfx950 with
    FlyDSL 0.5.0, ``fx.gpu.shuffle_xor`` lowers them to ``ds_swizzle_b32``
    with an ``s_waitcnt lgkmcnt(0)`` per step. Offsets 16 and 32 cross the
    16-lane DPP rows and use ``fx.gpu.shuffle_xor``, which lowers to
    ``v_permlane16_swap`` / ``v_permlane32_swap`` there. A butterfly built on
    it reduces in a fixed order.
    """
    if offset in (16, 32):
        return fx.Int32(fx.gpu.shuffle_xor(fx.Int32(src), offset, 64))
    return dpp_xor_i32(src, offset)


def xor_lane(src, offset: int):
    """``xor_lane_i32`` of an fp32 or i32 value; the result has ``src``'s type."""
    if src.dtype == fx.Float32:
        return xor_lane_i32(src.bitcast(fx.Int32), offset).bitcast(fx.Float32)
    return xor_lane_i32(src, offset)
