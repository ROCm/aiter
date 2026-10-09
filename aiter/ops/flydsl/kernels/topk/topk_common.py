# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared scalar, load, and wave helpers for FlyDSL TopK kernels."""

import flydsl.expr as fx
from flydsl.expr import Float32, Int32, arith, as_ir_value, const_expr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom

_DPP_ROW_MASK = 0xF
_DPP_BANK_MASK = 0xF


def _f32_to_ord(val):
    bits = val.bitcast(Int32)
    ordered = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    abs_bits = bits & fx.Int32(0x7FFFFFFF)
    is_nan = arith.cmpi(arith.CmpIPredicate.ugt, abs_bits, fx.Int32(0x7F800000))
    return is_nan.select(fx.Int32(0x7FFFFFFF), ordered)


def _row_length(row, row_ends, width, next_n):
    request = row // next_n
    offset = row % next_n
    row_len = row_ends[request] - next_n + offset + 1
    row_len = (row_len < 0).select(fx.Int32(0), row_len)
    return (row_len > width).select(width, row_len)


def _load_f32x4(tensor, vec_idx):
    src = fx.slice(tensor, (None, vec_idx))
    fragment = fx.make_fragment_like(src)
    fx.copy(buf_copy_atom(16, Float32), src, fragment)
    return fx.Vector(fx.memref_load_vec(fragment))


def _warp_inclusive_prefix_i32(val, lane, wave_size):
    val_raw = as_ir_value(val)
    zero_raw = as_ir_value(fx.Int32(0))
    for dpp_op, threshold in (
        (0x111, 1),
        (0x112, 2),
        (0x114, 4),
        (0x118, 8),
    ):
        remote = update_dpp_i32(
            zero_raw,
            val_raw,
            dpp_op,
            _DPP_ROW_MASK,
            _DPP_BANK_MASK,
            True,
        )
        val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
        val_raw = as_ir_value(val)

    remote = fly_rocdl.ds_bpermute(T.i32, ((lane & 0x30) - 1) * 4, val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote), val)
    if const_expr(wave_size == 64):
        remote = fly_rocdl.ds_bpermute(T.i32, ((lane & 0x30) - 17) * 4, val)
        val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote), val)
    return val
