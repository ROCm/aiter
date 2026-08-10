# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared low-level helpers for the a16w4/a16wi4/a16w16 fused MoE kernels.

Leaf helpers (pointer casts, byte GEPs, groupwise-scale unpack, int4->bf16
upconvert, index math) used by both stage1 (:mod:`gemm1`) and stage2
(:mod:`gemm2`).
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
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


def _int4_nibble_to_bf16x8(raw_i32, scale_f32, *, use_k16=False, old_pack=False):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 signed-int4 nibbles. ``v_cvt_off_f32_i4`` reads the nibble
    unsigned, subtracts 8, and scales the mantissa by 16, so the x16 is folded into
    eff = scale*16. ``use_k16`` (gfx942): v_cvt_pk_bf16_f32 is gfx950-only -> scalar.

    ``old_pack`` (a16wi4 consuming the OLD FlyDSL kernel's weight preshuffle,
    ``pack_int8_to_packed_int4``): byte j packs K_j (low nibble) and K_{j+4} (high
    nibble) -- the "interleaved-by-4" packing. So the SAME two cvt loads (byte_sel j on
    raw_even/raw_odd) give f_lo[j]=K_j and f_hi[j]=K_{j+4}, and the correct v8bf16 order
    K0..K7 is all-lows-then-all-highs (a pure output REORDER, zero extra instructions).
    ``old_pack=False`` (contiguous {2j,2j+1} packing, mxfp4 fp4 upconvert order):
    interleaved lo_j,hi_j == K_{2j},K_{2j+1}.
    """
    eff = scale_f32 * fx.Float32(16.0)
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        # gfx942 fallback: scalar f32 -> bf16 truncation (no v_cvt_pk_bf16_f32).
        los = []
        his = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
            los.append(f_lo.to(fx.BFloat16))
            his.append(f_hi.to(fx.BFloat16))
        if old_pack:
            bf16s = los + his  # K0..K3 (los), K4..K7 (his)
        else:
            bf16s = [x for pair in zip(los, his) for x in pair]  # K0,K1,...,K7
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    # byte_sel loads (1 shift total); side-effecting pk-convert.
    los = []
    his = []
    for j in range_constexpr(4):
        los.append(fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff)
        his.append(fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff)
    if old_pack:
        # v8bf16 = [K0,K1,K2,K3, K4,K5,K6,K7]; pk pairs (K0,K1),(K2,K3),(K4,K5),(K6,K7).
        i32s = [
            fx.Int32(_cvt_pk_bf16_f32_se(_raw(los[0]), _raw(los[1]))),
            fx.Int32(_cvt_pk_bf16_f32_se(_raw(los[2]), _raw(los[3]))),
            fx.Int32(_cvt_pk_bf16_f32_se(_raw(his[0]), _raw(his[1]))),
            fx.Int32(_cvt_pk_bf16_f32_se(_raw(his[2]), _raw(his[3]))),
        ]
    else:
        i32s = [
            fx.Int32(_cvt_pk_bf16_f32_se(_raw(los[j]), _raw(his[j])))
            for j in range_constexpr(4)
        ]
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


def _bf16_frag8(v8):
    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
    t.store(v8)
    return t


def _bf16_frag4(v8, half):
    t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    t.store(
        fx.Vector.from_elements(
            [_raw(v8[half * 4 + j]) for j in range_constexpr(4)], fx.BFloat16
        )
    )
    return t


def mma_bf16(mma_atom, use_k16, acc, a8, b8):
    """One K32 MFMA from v8bf16 A/B fragments; gfx942 (use_k16) splits it into 2x K16."""
    if const_expr(use_k16):
        for h in range_constexpr(2):
            fx.gemm(mma_atom, acc, _bf16_frag4(a8, h), _bf16_frag4(b8, h), acc)
    else:
        fx.gemm(mma_atom, acc, _bf16_frag8(a8), _bf16_frag8(b8), acc)
