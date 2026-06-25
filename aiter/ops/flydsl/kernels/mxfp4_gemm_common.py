# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops
from flydsl.expr.typing import T

from . import dpp_utils


_PTR3 = "!llvm.ptr<3>"
kStages = 2
kBS_stride_k0_dw = 64


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _lds_base_ptr3(lds_view):
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _buffer_rsrc(addr_i64, num_records_bytes):
    return buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(addr_i64)), num_records_bytes=num_records_bytes
    )


def _lds_swizzle_mask(row):
    return (row & fx.Int32(14)) << fx.Int32(3)


def _fabs_f32(x):
    return fx.Float32(llvm.call_intrinsic(T.f32, "llvm.fabs.f32", [_raw(x)], [], []))


def _e8m0_roundup(amax_f32):
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / 6.0)).bitcast(T.i32))
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = arith.cmpi(arith.CmpIPredicate.ult, _raw(bexp), _raw(fx.Int32(254)))
    return fx.Int32(arith.select(lt, _raw(bexp), _raw(fx.Int32(254))))


def _e8m0_from_amax(amax_f32):
    e8m0 = _e8m0_roundup(amax_f32)
    qscale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
    return e8m0, qscale


def _umax_i32(a, b):
    is_gt = arith.cmpi(arith.CmpIPredicate.ugt, _raw(a), _raw(b))
    return fx.Int32(arith.select(is_gt, _raw(a), _raw(b)))


def _inline_dpp_quad_amax(a32):
    a32 = fx.Int32(_raw(a32))
    s1 = fx.Int32(dpp_utils.update_dpp_i32(_raw(a32), _raw(a32), 0xB1, 0xF, 0xF, True))
    a32 = _umax_i32(a32, s1)
    s2 = fx.Int32(dpp_utils.update_dpp_i32(_raw(a32), _raw(a32), 0x4E, 0xF, 0xF, True))
    return _umax_i32(a32, s2)


def k_half_for(k):
    return k // 2


def k_tiles_total_for(k, BK):
    return k // BK


def kunroll_for(k, BK):
    return k_tiles_total_for(k, BK) - kStages


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2


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
