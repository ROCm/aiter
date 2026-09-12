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
    from flydsl._mlir.dialects import llvm as _llvm

    return _llvm.call_intrinsic(
        fx.Int32.ir_type,
        "llvm.amdgcn.update.dpp.i32",
        [
            fx.Int32(old).ir_value(),
            fx.Int32(src).ir_value(),
            fx.Int32(dpp_ctrl).ir_value(),
            fx.Int32(row_mask).ir_value(),
            fx.Int32(bank_mask).ir_value(),
            fx.Boolean(bound_ctrl).ir_value(),
        ],
        [],
        [],
        **kw,
    )
