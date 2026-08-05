# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared low-level helpers for the a16w4/a16wi4/a16w16 fused MoE kernels.

Leaf utilities used by both stage1 (:mod:`gemm1`) and stage2 (:mod:`gemm2`).
Dependency-free of the kernel bodies to avoid a stage1<->stage2 import cycle.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

_PTR3 = "!llvm.ptr<3>"

# a16wi4 int4 groupwise scale: group_size = 32 == one MFMA K32 step. Scale is bf16
# pairs (E, N, G//2, 2); even/odd ku selects lo/hi half.
A16WI4_GROUP_SIZE = 32


def s_waitcnt(vmcnt=None, lgkmcnt=None, expcnt=None):
    """CDNA3/gfx950-encoded s_waitcnt shim (aiter's FlyDSL has the bitfield form only)."""
    v = 63 if vmcnt is None else int(vmcnt)
    e = 7 if expcnt is None else int(expcnt)
    lk = 15 if lgkmcnt is None else int(lgkmcnt)
    encoded_vmcnt = (v & 0xF) | ((v & 0x30) << 10)
    return rocdl.s_waitcnt(encoded_vmcnt | (lk << 8) | (e << 4))


def a16wmix_use_k16(arch):
    """True for the gfx942 (CDNA3) codepath: K=16 MFMA + scalar int4 dequant.

    gfx950 (CDNA4) has K=32 mfma_f32_16x16x32_bf16 + v_cvt_pk_bf16_f32; gfx942 has
    neither and falls back to K=16 MFMA + scalar-trunc dequant.
    """
    return "gfx95" not in str(arch)


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
    # BufferCopy soffset is an element count (not bytes); make_layout dynamic-shape
    # leaf must be i32/i64, not fx.Index.
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
    """Read one i32 dword at element ``idx`` via a BufferCopy atom (buffer_load_dword,
    OOB-clamped). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep(base_ptr, byte_off_i32):
    # Byte GEP; polymorphic in the base ptr's address space (global ptr<1> / LDS ptr<3>).
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _a16w4_swizzle_xor16(row, col_bytes, k_blocks16, *, enable=False):
    """A-LDS bank-conflict XOR swizzle (col ^ ((row&(kb16-1))*16)).

    Both the DMA write and the LDS read use this so the physical layout stays
    consistent. gemm1 keeps linear (enable=False); gemm2 enables it.
    """
    if not enable:
        return col_bytes
    rem = row & fx.Int32(k_blocks16 - 1)
    return col_bytes ^ (rem * fx.Int32(16))


def _e8m0_byte_to_f32(packed_i32, byte_pos):
    shift = byte_pos * fx.Int32(8)
    b = packed_i32.shrui(shift) & fx.Int32(0xFF)
    return (b << fx.Int32(23)).bitcast(fx.Float32)


def _int4_nibble_to_bf16x8(raw_i32, scale_f32, *, use_k16=False):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 signed-int4 nibbles in bits[4n+3:4n] (same K order as the mxfp4
    sel 0..3 path). ``v_cvt_off_f32_i4`` reads the nibble unsigned, subtracts 8, and
    x16-scales the mantissa, so x16 folds into eff = scale*16. Scalar ``.to(BFloat16)``
    (arith.truncf), NOT side-effecting ``v_cvt_pk_bf16_f32``: truncf is schedulable so
    the compiler packs the per-nibble ``* eff``, and it sidesteps the stateless-cvt_pk
    mis-CSE that forced the side-effecting pin.
    """
    eff = scale_f32 * fx.Float32(16.0)
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    bf16s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
        bf16s.append(f_lo.to(fx.BFloat16))
        bf16s.append(f_hi.to(fx.BFloat16))
    return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
