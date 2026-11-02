# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from aiter.ops.flydsl.kernels import dpp_utils

kStages = 2
kBS_stride_k0_dw = 64


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr3(base_i32, byte_off_i32):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(base_i32 + byte_off_i32)))


def _buffer_rsrc(addr_i64, num_records_bytes):
    return buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(addr_i64), num_records_bytes=num_records_bytes
    )


def _lds_swizzle_mask(row, row_bytes=128):
    """XOR16 swizzle for an FP4 LDS row of `row_bytes`; permutes its 16-byte columns."""
    assert row_bytes in (64, 128), f"unsupported FP4 LDS row width {row_bytes}"
    return (row & fx.Int32(2 * (row_bytes // 16) - 2)) << fx.Int32(3)


def global_typed_ptr(arg, elem_ty, align=4):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


def lds_typed_ptr(base_i32, elem_ty, align=4):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def _fabs_f32(x):
    return fmath.absf(x)


def _e8m0_roundup(amax_f32):
    wi = (amax_f32 * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = fx.Uint32(bexp) < fx.Uint32(254)
    return lt.select(bexp, fx.Int32(254))


def _e8m0_from_amax(amax_f32):
    e8m0 = _e8m0_roundup(amax_f32)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale


def _umax_i32(a, b):
    is_gt = fx.Uint32(a) > fx.Uint32(b)
    return is_gt.select(a, b)


def _dpp_umax_step(a32, dpp_ctrl):
    swapped = dpp_utils.update_dpp_i32(a32, a32, dpp_ctrl, 0xF, 0xF, True)
    return _umax_i32(a32, fx.Int32(swapped))


def _inline_dpp_quad_amax(a32):
    return _dpp_umax_step(_dpp_umax_step(a32, 0xB1), 0x4E)


def k_half_for(k):
    return k // 2


def k_tiles_total_for(k, BK):
    return k // BK


def kas_c_k1_for(k):
    # Host preshuffle pads the number of 1x32 scale groups to a multiple
    # of eight.  Round up here as well: for K=384 there are 12 logical
    # groups but 16 stored groups (two 256-K scale chunks).  Truncating to
    # one chunk makes ``(row, chunk=1)`` alias ``(row+1, chunk=0)``.
    return ((k // 32) + 7) // 8


def kbs_c_k1_for(k):
    # Match shuffle_scale_a16w4/e8m0_shuffle's eight-group K padding.
    return ((k // 32) + 7) // 8


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64


def kas_per_chunk_dw_for(k):
    return kas_c_k1_for(k) * 64


def num_n_blocks_for(n, BN):
    return n // BN


def kbs_c_n1_for(n):
    return n // 16 // 2


def kbs_per_expert_dw_for(n, k):
    return kbs_c_n1_for(n) * kbs_stride_n0_dw_for(k)


def bq_bytes_for(ne, n, k):
    return ne * n * k_half_for(k)


def bscale_bytes_for(ne, n, k):
    return ne * kbs_per_expert_dw_for(n, k) * 4


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4


