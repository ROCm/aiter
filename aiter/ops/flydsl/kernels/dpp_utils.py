# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""DPP helpers used by paged-attention kernels."""


def _to_ir(v):
    """Coerce DSL Numeric values to raw MLIR values."""
    from flydsl._mlir import ir as _ir
    from flydsl.expr import arith as _arith_ext

    if isinstance(v, int):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=_ir.IntegerType.get_signless(32)))
    if isinstance(v, float):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=_ir.F32Type.get()))
    if not isinstance(v, _ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


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
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith as _arith_ext
    from flydsl.expr.typing import T

    return _llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.update.dpp.i32",
        [
            _to_ir(old),
            _to_ir(src),
            _arith_ext.unwrap(_arith_ext.constant(dpp_ctrl, type=T.i32)),
            _arith_ext.unwrap(_arith_ext.constant(row_mask, type=T.i32)),
            _arith_ext.unwrap(_arith_ext.constant(bank_mask, type=T.i32)),
            _arith_ext.unwrap(_arith_ext.constant(bound_ctrl, type=_ir.IntegerType.get_signless(1))),
        ],
        [],
        [],
        **kw,
    )


def dpp_xor_f32(src, offset: int, **kw):
    """Return ``src`` from the lane selected by a 16-lane XOR DPP pattern."""
    from flydsl._mlir.dialects import arith as _arith_dialect
    from flydsl.expr.typing import T

    src_i32 = _to_ir(src).bitcast(T.i32)
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
        raise ValueError(f"dpp_xor_f32 only supports 16-lane offsets 1, 2, 4, 8; got {offset}")
    return _arith_dialect.BitcastOp(T.f32, out_i32).result
