# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared low-level helpers for the a16w4/a16wi4/a16w16 fused MoE kernels.

Leaf helpers (pointer casts, byte GEPs, groupwise-scale unpack, int4->bf16
upconvert, index math) used by both stage1 (:mod:`gemm1`) and stage2
(:mod:`gemm2`).
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

# a16wi4 (int4 W) groupwise scale: group_size = 32 == one MFMA K32 step (one ku per
# K-group). Scale packed bf16 pairs (E, N, G//2, 2); even/odd ku selects lo/hi half.
A16WI4_GROUP_SIZE = 32


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


def _global_i32_buffer_view(addr_i64, num_bytes):
    # fx.copy BufferCopy atoms take soffset as an element count (not bytes); the
    # make_layout dynamic-shape leaf must be i32/i64, not fx.Index.
    num_bytes_i64 = fx.Int64(num_bytes)
    view = fx.Tensor(
        fx.make_view(
            _global_i32_ptr(addr_i64), fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)
        )
    )
    return fx.rocdl.make_buffer_tensor(
        view, max_size=False, num_records_bytes=num_bytes_i64
    )


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(
        _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
    )


def _buffer_i32_scalar_read(tiles1, idx, atom):
    """Read one i32 dword at element ``idx`` from a ``_global_i32_buffer_tiles(..., 1)``
    view via the layout-API BufferCopy atom (buffer_load_dword; OOB-clamped by the
    buffer resource). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])


def _int_to_llvm_ptr(addr, address_space):
    # int addr -> raw !llvm.ptr; to_llvm_ptr maps the semantic AS to the backend AS.
    ptr_ty = fx.PointerType.get(T.i8, address_space=address_space)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(addr)))


def _lds_ptr3(base_i32, byte_off_i32):
    return _int_to_llvm_ptr(base_i32 + byte_off_i32, fx.AddressSpace.Shared)


def _global_base_ptr1(addr_i64):
    return _int_to_llvm_ptr(addr_i64, fx.AddressSpace.Global)


def _gep(base_ptr, byte_off_i32):
    # Byte GEP; polymorphic in the base ptr's address space (global ptr<1> / LDS ptr<3>).
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _cvt_pk_bf16_f32_se(src_a_f32, src_b_f32):
    # Side-effecting v_cvt_pk_bf16_f32 (pack 2 f32 -> 2xbf16 in i32). LOAD-BEARING:
    # the stateless rocdl.cvt_pk_bf16_f32 gets CSE-merged/reordered across K steps in
    # the a16wi4 gemm1 hot loop (garbage output); side_effects pins each call.
    return llvm.inline_asm(
        T.i32,
        [_raw(src_a_f32), _raw(src_b_f32)],
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=True,
    )


def _int4_nibble_to_bf16x8(raw_i32, scale_f32, *, use_k16=False):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 signed-int4 nibbles in bits[4n+3:4n] (same K order as the
    mxfp4 sel 0..3 path). ``v_cvt_off_f32_i4`` reads the nibble unsigned, subtracts 8,
    and scales the mantissa by 16, so the x16 is folded into eff = scale*16.
    ``use_k16`` (gfx942): v_cvt_pk_bf16_f32 is gfx950-only -> scalar .to(BFloat16).
    """
    eff = scale_f32 * fx.Float32(16.0)
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        # gfx942 fallback: scalar f32 -> bf16 truncation (no v_cvt_pk_bf16_f32).
        bf16s = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
            bf16s.append(f_lo.to(fx.BFloat16))
            bf16s.append(f_hi.to(fx.BFloat16))
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    # byte_sel loads (1 shift total); side-effecting pk-convert.
    i32s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
        i32s.append(fx.Int32(_cvt_pk_bf16_f32_se(_raw(f_lo), _raw(f_hi))))
    v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
    return v4i32.bitcast(fx.BFloat16)  # v8bf16


def _e8m0_byte_to_f32(packed_i32, byte_pos):
    shift = byte_pos * fx.Int32(8)
    b = packed_i32.shrui(shift) & fx.Int32(0xFF)
    return (b << fx.Int32(23)).bitcast(fx.Float32)


def _a16w4_swizzle_xor16(row, col_bytes, k_blocks16, *, enable=False):
    """A-LDS bank-conflict XOR swizzle (aiter swizzle_xor16: col ^ ((row&(kb16-1))*16)).

    Both the DMA write and the LDS read go through this helper so the physical layout
    stays consistent. gemm1 keeps linear (enable=False); gemm2 enables it.
    """
    if not enable:
        return col_bytes
    rem = row & fx.Int32(k_blocks16 - 1)
    return col_bytes ^ (rem * fx.Int32(16))
