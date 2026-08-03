# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, rocdl
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, T

from aiter.ops.flydsl.kernels import buffer_ops

from . import dpp_utils
from .layout_utils import crd2idx

_PTR3 = "!llvm.ptr<3>"
kStages = 2
kBS_stride_k0_dw = 64
LOG2E = 1.4426950408889634


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


_A_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype):
    """Build scaled 16x16x128 MFMA atoms for every scale-byte selection."""
    elem_a = _A_ELEM[a_dtype]
    return {
        (opsel_a, opsel_b): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16,
                16,
                128,
                elem_a,
                Float4E2M1FN,
                opsel_a=opsel_a,
                opsel_b=opsel_b,
            )
        )
        for opsel_a in range(4)
        for opsel_b in range(4)
    }


def _global_i32_buffer_view(addr_i64, num_bytes):
    num_bytes_i64 = fx.Int64(num_bytes)
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=False, num_records_bytes=num_bytes_i64
    )


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(
        _global_i32_buffer_view(addr_i64, num_bytes),
        fx.make_layout(tile_elems, 1),
    )


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


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


def _global_i32_load(tiles, idx):
    atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy_atom_call(atom, fx.slice(tiles, (None, idx)), r)
    return r.load()[0]


def _global_scalar_tiles(addr_i64, numeric_cls, num_elems):
    ptr_ty = fx.PointerType.get(
        numeric_cls.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=numeric_cls.width // 8,
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    flat = fx.make_view(ptr, fx.make_layout(num_elems, 1))
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def _scalar_store(tiles, idx, value, numeric_cls):
    atom = fx.make_copy_atom(fx.UniversalCopy(numeric_cls.width), numeric_cls)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), numeric_cls)
    r.store(fx.Vector.from_elements([numeric_cls(value)], numeric_cls))
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _layout_idx(layout, *coords):
    return fx.Int32(crd2idx([fx.Int64(coord) for coord in coords], layout))


def _buffer_rsrc(addr_i64, num_records_bytes):
    return buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(addr_i64)), num_records_bytes=num_records_bytes
    )


def _lds_swizzle_mask(row):
    return (row & fx.Int32(14)) << fx.Int32(3)


def _fabs_f32(x):
    return fx.Float32(llvm.call_intrinsic(T.f32, "llvm.fabs.f32", [_raw(x)], [], []))


def _e8m0_roundup(amax_f32, max_norm=6.0):
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / float(max_norm))).bitcast(T.i32))
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = arith.cmpi(arith.CmpIPredicate.ult, _raw(bexp), _raw(fx.Int32(254)))
    return fx.Int32(arith.select(lt, _raw(bexp), _raw(fx.Int32(254))))


def _e8m0_from_amax(amax_f32, max_norm=6.0):
    e8m0 = _e8m0_roundup(amax_f32, max_norm=max_norm)
    qscale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
    return e8m0, qscale


def _inline_e8m0(amax_u16_i32, max_norm=6.0):
    f32 = fx.Float32(
        _raw((fx.Int32(_raw(amax_u16_i32)) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(
            T.f32
        )
    )
    return _e8m0_roundup(f32, max_norm=max_norm)


def _pkmax_u16(a_i32, b_i32):
    v2i16 = T.vec(2, T.i16)
    va = llvm.BitcastOp(v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    return fx.Int32(llvm.BitcastOp(T.i32, vm).result)


def _silu_mul_batch(gate_values, up_values):
    exp_values = [
        fx.Float32(rocdl.exp2(T.f32, _raw(gate * fx.Float32(-LOG2E))))
        for gate in gate_values
    ]
    sigmoid_values = [
        fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + exp_value)))
        for exp_value in exp_values
    ]
    return [
        gate_values[i] * sigmoid_values[i] * up_values[i]
        for i in range(len(gate_values))
    ]


def _situ_mul_batch(gate_values, up_values, beta=1.0, linear_beta=1.0):
    one = fx.Float32(1.0)
    zero = fx.Float32(0.0)
    beta_f32 = fx.Float32(float(beta))
    beta_rcp = fx.Float32(1.0 / float(beta))
    linear_beta_f32 = fx.Float32(float(linear_beta))
    linear_beta_rcp = fx.Float32(1.0 / float(linear_beta))

    def tanh_elem(x):
        abs_x = x.maximumf(-x)
        e = fx.Float32(rocdl.exp2(T.f32, _raw(abs_x * fx.Float32(-2.0 * LOG2E))))
        tanh_abs = (one - e) * fx.Float32(rocdl.rcp(T.f32, _raw(one + e)))
        return (x > zero).select(tanh_abs, -tanh_abs)

    result = []
    for gate, up in zip(gate_values, up_values):
        situ = (
            beta_f32
            * tanh_elem(gate * beta_rcp)
            * fx.Float32(
                rocdl.rcp(
                    T.f32,
                    _raw(
                        one
                        + fx.Float32(rocdl.exp2(T.f32, _raw(gate * fx.Float32(-LOG2E))))
                    ),
                )
            )
        )
        result.append(situ * linear_beta_f32 * tanh_elem(up * linear_beta_rcp))
    return result


def _activation_mul_batch(
    gate_values, up_values, act="silu", situ_beta=1.0, situ_linear_beta=1.0
):
    if act == "situv2":
        return _situ_mul_batch(
            gate_values,
            up_values,
            beta=situ_beta,
            linear_beta=situ_linear_beta,
        )
    return _silu_mul_batch(gate_values, up_values)


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
