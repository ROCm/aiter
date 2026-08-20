# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared operand paths and low-level helpers for the a16w4/a16wi4/a16w16 MoE kernels.

Used by both stage1 (:mod:`gemm1`) and stage2 (:mod:`gemm2`):

  * :func:`make_a_loader`  the A (activation) path -- global->LDS staging plus the
                           v8bf16 MMA-fragment reads that must match its LDS layout.
  * :func:`make_b_loader`  the B (weight) path -- N addressing, raw W + scale gathers
                           and the upconvert, all specialized for ``w_dtype`` so the
                           stages carry no weight-dtype branches.
  * leaf helpers           pointer casts, byte GEPs, groupwise-scale unpack,
                           int4->bf16 upconvert, index math.
"""

from collections.abc import Callable
from typing import NamedTuple

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.layout_utils import crd2idx
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

# a16wi4 (int4 W) groupwise scale: group_size = 32 == one MFMA K32 step (one ku per
# K-group). Scale packed bf16 pairs (E, G//2, N, 2); even/odd ku selects lo/hi half.
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


def _mma_bf16(mma_atom, use_k16, acc, a8, b8):
    """One K32 MFMA from v8bf16 A/B fragments; gfx942 (use_k16) splits it into 2x K16."""
    if const_expr(use_k16):
        for h in range_constexpr(2):
            fx.gemm(mma_atom, acc, _bf16_frag4(a8, h), _bf16_frag4(b8, h), acc)
    else:
        fx.gemm(mma_atom, acc, _bf16_frag8(a8), _bf16_frag8(b8), acc)


def _a_col_bytes_for_ku(col_base_bytes_L, ku):
    """A-LDS column byte offset for K micro-step ``ku`` (pre-swizzle)."""
    _k0_blk = ku // 4
    _ku_in = ku % 4
    return col_base_bytes_L + fx.Int32(_ku_in * 16 + _k0_blk * 256)


class _ALoader(NamedTuple):
    """The A (activation) LDS path, built by :func:`make_a_loader`."""

    load: Callable  # load(mi, ku, slot=0) -> v8bf16 A MMA fragment
    store_tile: Callable  # store_tile(base_k, slot=0): global -> A-LDS, one K tile


def make_a_loader(
    lds_raw_ptr,
    *,
    num_i32,
    BM,
    TILE_K,
    KH_TILE_BYTES,
    k_blocks16,
    lane_div_16,
    lane_mod_16,
    swizzle,
    a_ptr,
    a_num_bytes,
    a_load_threads,
    row_base_dwords,
    dma_cache_mod,
    dma_via_vgpr,
    k_grp_base_bytes=None,
    A_SLOT_BYTES=0,
):
    """Build the shared A (activation) LDS path: global->LDS staging + fragment reads.

    Both stages stage A into LDS as BM x TILE_K bf16 (``store_tile``) and read it back
    as v8bf16 MMA fragments through the CK sub-lane map (lane L covers
    K[L*32 .. L*32+31], ``load``), so the two live together: they must agree on the
    physical LDS layout, and both go through the same 16 B tile view.

    Stage-specific parameters:
      * ``row_base_dwords``  ``(row_local) -> global A row base in dwords``. stage1
                             gathers the sorted token id out of ``m_indices``; stage2
                             indexes the intermediate by sorted position directly.
      * ``a_num_bytes``      buffer-resource extent. LOAD-BEARING in stage1: sizing it
                             to the REAL [n_tokens, K] alloc HW-clamps padding-row loads
                             (sentinel token id >= n_tokens) to 0. A ~4 GB resource
                             would instead fault on unmapped memory.
      * ``swizzle``          gemm2 XOR-swizzles A to kill LDS bank conflicts: the GMEM
                             source column is swizzled on the write (buffer_load_lds
                             ignores an arbitrary swizzled per-lane LDS dest -> NaN) and
                             ``load`` applies the SAME swizzle on the read. gemm1 is
                             linear -- its DMA source is already conflict-free.
      * ``k_grp_base_bytes`` gemm1 only: base of this wave's k_wave group. Combined with
                             ``A_SLOT_BYTES`` and the per-call ``slot`` it selects the
                             double-buffer ping/pong slot. None (gemm2) omits the term
                             entirely -- stage2 has a single, unslotted A region.
      * ``a_load_threads``   threads cooperating on one tile (< 256 when k_wave > 1
                             splits the block into per-k-group loader sets).
      * ``dma_via_vgpr``     gfx942 (use_k16) staging fallback, see below.
    """
    elem_bytes = 2  # bf16
    # Per-thread A gather: each thread moves 16 B (v8bf16) per pass.
    bytes_per_thread = (BM * TILE_K * elem_bytes) // a_load_threads
    x_load_bytes = 16
    num_x_loads = bytes_per_thread // x_load_bytes
    tile_k_dwords = (TILE_K * elem_bytes) // 4
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    chunk_i32 = x_load_bytes // 4  # 4
    if const_expr(a_load_threads < 256):
        x_load_tid = tx_i32 % fx.Int32(a_load_threads)
    else:
        x_load_tid = tx_i32
    tx_base = x_load_tid * fx.Int32(chunk_i32)

    x_row_local = []
    x_col_dw = []
    x_row_base_div4 = []
    for i in range_constexpr(num_x_loads):
        tile_idx = tx_base + fx.Int32(i * a_load_threads * chunk_i32)
        row_local = tile_idx // fx.Int32(tile_k_dwords)
        x_row_local.append(row_local)
        x_col_dw.append(tile_idx % fx.Int32(tile_k_dwords))
        x_row_base_div4.append(row_base_dwords(row_local))

    x_buf = _global_i32_buffer_view(a_ptr, a_num_bytes)
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(4, 1))
    # gfx950 (K=32): one BufferCopyLDS128b direct-to-LDS async copy (16 B / 8 bf16,
    # VGPR-bypassing). gfx942 (dma_via_vgpr): CDNA3 direct-to-LDS moves only 4 B/lane;
    # the 16 B dwordx4-to-LDS form does not exist and LLVM cannot legalize the s128 LDS
    # store (ISA-lowering crash). A narrow 4x4 B direct-to-LDS split is also wrong
    # because buffer_load_dword...lds writes lane L to M0+L*width -- four 4 B copies
    # cannot reproduce the 16 B/lane layout the 128b copy (and the ds_read_b128 A-read)
    # expect. So stage via VGPRs like the legacy kernel: buffer_load 16 B gmem->regs
    # then ds_write 16 B regs->LDS (both b128, valid on CDNA3), same LDS layout.
    if const_expr(dma_via_vgpr):
        x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(dma_cache_mod), fx.Int32)
    else:
        x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr), fx.make_layout(num_i32, 1)
    )
    tiles = fx.logical_divide(s_x_i32_flat, fx.make_layout(4, 1))  # 16 B A-LDS tiles
    lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)  # ds_write/ds_read
    row_a_lds = lane_mod_16
    col_base_bytes_L = lane_div_16 * fx.Int32(64)  # 32 bf16 * 2 B

    def store_a_tile(base_k, slot=0):
        base_k_div4 = (base_k * fx.Int32(elem_bytes)) // fx.Int32(4)
        if k_grp_base_bytes is None:
            slot_byte = None
        else:
            slot_byte = k_grp_base_bytes + fx.Int32(slot * A_SLOT_BYTES)
        for i in range_constexpr(num_x_loads):
            col_bytes = x_col_dw[i] * fx.Int32(4)
            # Swizzle the GMEM source column; the LDS destination stays linear.
            col_sw = _a16w4_swizzle_xor16(
                x_row_local[i], col_bytes, fx.Int32(k_blocks16), enable=swizzle
            )
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte = row_k_dw * fx.Int32(4) + col_sw
            if const_expr(slot_byte is None):
                lds_byte = x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
            else:
                lds_byte = (
                    slot_byte + x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
                )
            src = fx.slice(x_dma_tiles4, (None, global_byte // fx.Int32(16)))
            dst = fx.slice(tiles, (None, lds_byte // fx.Int32(16)))
            if const_expr(dma_via_vgpr):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(x_dma_atom, src, r)
                fx.copy(lds_atom, r, dst)
            else:
                fx.copy(x_dma_atom, src, dst)

    def lds_load_a(mi, ku, slot=0):
        row = row_a_lds + fx.Int32(mi * 16)
        # Same XOR swizzle as the DMA write (16 B-multiple cols/mask keep alignment).
        col_swz_bytes = _a16w4_swizzle_xor16(
            row,
            _a_col_bytes_for_ku(col_base_bytes_L, ku),
            fx.Int32(k_blocks16),
            enable=swizzle,
        )
        if k_grp_base_bytes is None:
            byte_off = row * fx.Int32(KH_TILE_BYTES) + col_swz_bytes
        else:
            # byte offset within this k-group's A-LDS slot -> 16-byte tile index.
            byte_off = (
                k_grp_base_bytes
                + fx.Int32(slot * A_SLOT_BYTES)
                + row * fx.Int32(KH_TILE_BYTES)
                + col_swz_bytes
            )
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy_atom_call(
            lds_atom, fx.slice(tiles, (None, byte_off // fx.Int32(16))), r
        )
        return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16)  # v8bf16

    return _ALoader(load=lds_load_a, store_tile=store_a_tile)


class _BCol(NamedTuple):
    """This lane's addressing for one 16-wide N block of W, built by ``_BLoader.col``.

    Only the scale fields of the active ``w_dtype`` are filled; the others stay None.
    """

    n_blk: object  # 16-row weight block index (expert-indexed for mxfp4/int4)
    n_intra: object  # row within that block
    sc_blk: object = None  # mxfp4: 32-col e8m0 scale block index
    sc_pack: object = None  # mxfp4: N_Pack half selector (0/1) within that block
    sc_n: object = None  # int4: N index WITHIN the expert


class _BLoader(NamedTuple):
    """The B (weight) operand path, built by :func:`make_b_loader`.

    All three callables are already specialized for ``w_dtype``, so the stages carry no
    weight-dtype branches: they build ``col`` descriptors once outside the K loop, then
    ``load_raw`` + ``load_scale`` + ``upconvert`` per tile.
    """

    col: Callable  # col(*n_terms) -> _BCol for one 16-wide N block
    col_pair: Callable  # col_pair(*n_terms, shift=) -> (_BCol, _BCol) gate|up pair
    load_raw: Callable  # load_raw(base_k, col) -> per-K-step raw fragments
    load_scale: Callable  # load_scale(base_k, col) -> per-K-step f32 scales
    upconvert: Callable  # upconvert(raw, ku, scale) -> v8bf16 MMA operand


def make_b_loader(
    arg_bq,
    arg_bscale,
    *,
    N_OUT,
    K,
    NE,
    e,
    lane_div_16,
    lane_mod_16,
    TILE_K,
    w_dtype,
    b_cache_mod,
    use_k16,
):
    """Build the shared B (weight) operand path for gemm1 and gemm2.

    stage1 and stage2 contract the same way -- N-major preshuffled W, per-lane scalar
    scale gather, upconvert to v8bf16 MMA fragments -- and differ only in what N and K
    mean (stage1: N=2*inter_dim gate|up, K=model_dim; stage2: N=model_dim, K=inter_dim).
    So everything below is a pure function of (N_OUT, K, NE, e, TILE_K, w_dtype) and is
    shared verbatim; keeping two copies only let their comments drift.

    ``w_dtype``:
      ``"fp4"``  mxfp4 W + e8m0 per-1x32 scale
      ``"int4"`` OLD-kernel packed int4 W + (E, G//2, N, 2) bf16 groupwise scale
      ``"bf16"`` raw bf16 W (no scale, no upconvert; ``load_scale`` yields all-None and
                 ``upconvert`` returns the loaded fragment unchanged)

    ``expert_off``/``inter``-style N arithmetic stays with the caller: it hands ``col``
    the 16-wide N block base and gets back everything the loaders need.

    Returns a :class:`_BLoader`. The closures emit no IR until called, so building them
    here rather than inline in the kernel body does not perturb instruction order.
    """
    _is_int4 = w_dtype == "int4"
    _is_bf16 = w_dtype == "bf16"
    # Emitted before the layouts/resources below: this is where the stages used to
    # compute it, and the ISA is sensitive to the order operands are materialized in.
    expert_off = e * fx.Int32(N_OUT)
    elem_bytes = 2
    K_HALF = K // 2
    k_unroll = (TILE_K * elem_bytes) // 64
    _k0_count = TILE_K // 128

    # W (mxfp4) preshuffle layout (aiter make_preshuffle_b_layout, N-major, fp4 bytes):
    #   shape (N_OUT/16, (K/2)/64, klane=4, nlane=16, kpack=16), strides below.
    bl_k0 = K_HALF // 64
    bl_stride_klane = 256
    bl_stride_k0 = 1024
    bl_stride_n0 = bl_k0 * bl_stride_k0
    layout_b = fx.make_layout(
        (N_OUT // 16, bl_k0, 4, 16, 16),
        (bl_stride_n0, bl_stride_k0, bl_stride_klane, 16, 1),
    )
    # int4 W: OLD-kernel preshuffle, kpack=8 (K16/klane slot), strides (n0,512,128,8,1);
    # one MFMA-K32 = two klane slots 128B apart, bytes interleaved-by-4
    # (byte j = K_j | K_{j+4}<<4).
    if const_expr(_is_int4):
        i4l_k0 = K // 64
        i4l_stride_klane = 128
        i4l_stride_k0 = 512
        i4l_stride_n0 = i4l_k0 * i4l_stride_k0
        layout_b_int4 = fx.make_layout(
            (N_OUT // 16, i4l_k0, 4, 16, 8),
            (i4l_stride_n0, i4l_stride_k0, i4l_stride_klane, 8, 1),
        )
    # W (raw bf16) preshuffle layout (N-major == shuffle_weight (16,16)), bf16-elem
    # units: shape (N_OUT/16, K/32, 4, 16, 8). One kpack = 8 bf16 = ONE MFMA K32 B
    # fragment (no upconvert). K reindexed to the fp4 (klane_hw, ku)->K order (see
    # load_b_raw_bf16).
    bfl_k0 = K // 32
    bfl_stride_klane = 128
    bfl_stride_k0 = 512
    bfl_stride_n0 = bfl_k0 * bfl_stride_k0
    layout_b_bf16 = fx.make_layout(
        (N_OUT // 16, bfl_k0, 4, 16, 8),
        (bfl_stride_n0, bfl_stride_k0, bfl_stride_klane, 8, 1),
    )
    # B-scale preshuffle layout (make_preshuffle_scale_layout, e8m0 u8, per-1x32):
    #   K padded to 256 mult. shape (N_OUT/32, c_k1, 4, 16), strides below.
    scale_k_padded = ((K + 255) // 256) * 256
    sc_k1 = ((scale_k_padded // 32) // 4) // 2
    sc_stride_klane = 16
    sc_stride_k0 = 64
    sc_stride_n0 = sc_k1 * sc_stride_k0
    # a16wi4 groupwise scale: bf16 pairs, layout (E, G//2, N, 2).
    _g_half = (K // A16WI4_GROUP_SIZE) // 2

    # ---- buffer resources -----------------------------------------------------
    # bf16 W [E, N_OUT, K] whole-tensor extent overflows the 32-bit num_records/i32
    # byte-offset at large E (E896: 6.6GB): fold the per-expert base into the i64
    # resource addr and index within the expert. mxfp4/int4 keep the whole-tensor path.
    if const_expr(_is_bf16):
        _w_per_expert_bytes = N_OUT * (K * 2)
        w_base_i64 = fx.Int64(arg_bq) + fx.Int64(e) * fx.Int64(_w_per_expert_bytes)
        w_tiles = _global_i32_buffer_tiles(
            w_base_i64, min(_w_per_expert_bytes, 0xFFFFFFFF), 4
        )
    else:
        _w_bytes = NE * N_OUT * K_HALF
        w_tiles = _global_i32_buffer_tiles(arg_bq, min(_w_bytes, 0xFFFFFFFF), 4)
    # W dwordx4 load via BufferCopy128b atom (cache modifier in the aux field).
    w_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
    w_reg_lay = fx.make_layout(4, 1)
    if const_expr(_is_int4):
        # OLD int4 layout: one klane slot = 8 bytes = 2 i32 -> dwordx2 (64b) loads. A
        # port MFMA-K32 block is two such slots 128 B apart (see load_b_raw_int4).
        w_copy_atom64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(b_cache_mod), fx.Int32)
        w_reg_lay2 = fx.make_layout(2, 1)
        w_tiles8 = _global_i32_buffer_tiles(arg_bq, min(_w_bytes, 0xFFFFFFFF), 2)
    if _is_int4:
        # int4 groupwise scale buffer (E, G//2, N_OUT, 2) bf16 -> G//2 dwords per N.
        _sw_bytes = NE * N_OUT * _g_half * 4
    else:
        _sw_bytes = NE * N_OUT * (scale_k_padded // 32)
    # bf16 W has no scale buffer; sw_tiles unused (arg_bscale is a dummy pointer). Scale
    # is a per-lane scalar e8m0/bf16-pair gather: a make_buffer_tensor 1-dword tiles view
    # + BufferCopy32b scalar read (buffer_load_dword, OOB-clamped) replaces the raw
    # create_buffer_resource_from_addr + buffer_load.
    sw_tiles = (
        None
        if _is_bf16
        else _global_i32_buffer_tiles(arg_bscale, min(_sw_bytes, 0xFFFFFFFF), 1)
    )
    sw_read_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

    # ---- per-lane N addressing -------------------------------------------------
    # int4 groupwise scale is the OLD-kernel (E, G//2, N, 2) layout: its N index is
    # WITHIN the expert, so the expert term is a separate G//2*N_OUT stride. Materialized
    # on first use (not here): it is loop-invariant, and emitting it up front instead
    # perturbs downstream scheduling in the int4 upconvert loop.
    _scale_base = []

    def scale_expert_base():
        if not _scale_base:
            _scale_base.append(
                rocdl.readfirstlane(T.i32, _raw(e * fx.Int32(_g_half * N_OUT)))
            )
        return _scale_base[0]

    def _sum(first, terms):
        v = first
        for t in terms:
            v = v + t
        return v

    def _scale_fields(ng, col_g):
        if const_expr(_is_bf16):
            return {}
        if const_expr(_is_int4):
            return {"sc_n": col_g}  # int4 indexes N WITHIN the expert
        return {
            "sc_blk": ng // fx.Int32(32),
            "sc_pack": (ng // fx.Int32(16)) % fx.Int32(2),
        }

    # The N block base is passed as its SUMMANDS (tile base, wave base, ni*16) and the
    # gate->up ``shift`` is applied last, so these build the same address expression
    # trees -- and emit them in the same order -- as the inline code they replaced.
    # Instruction selection is sensitive to both.
    def col(*n_terms):
        """Address one 16-wide N block; this lane owns row ``sum(n_terms) + lane``."""
        col_g = _sum(n_terms[0], n_terms[1:]) + lane_mod_16
        # bf16 W folds the expert into the resource base (above); mxfp4/int4 index it.
        _row_expert_off = fx.Int32(0) if const_expr(_is_bf16) else expert_off
        row = _row_expert_off + col_g
        n_blk = row // fx.Int32(16)
        n_intra = row % fx.Int32(16)
        ng = _sum(expert_off, n_terms)
        return _BCol(n_blk, n_intra, **_scale_fields(ng, col_g))

    def col_pair(*n_terms, shift):
        """Address a gate|up pair: one block, and the one ``shift`` further along N."""
        col_g = _sum(n_terms[0], n_terms[1:]) + lane_mod_16
        _row_expert_off = fx.Int32(0) if const_expr(_is_bf16) else expert_off
        row_g = _row_expert_off + col_g
        row_u = row_g + shift
        blk_g, intra_g = row_g // fx.Int32(16), row_g % fx.Int32(16)
        blk_u, intra_u = row_u // fx.Int32(16), row_u % fx.Int32(16)
        ng = _sum(expert_off, n_terms)
        nu = ng + shift
        return (
            _BCol(blk_g, intra_g, **_scale_fields(ng, col_g)),
            _BCol(blk_u, intra_u, **_scale_fields(nu, col_g + shift)),
        )

    # ---- B (mxfp4 W) raw load: dwordx4 -> v4i32 (8 fp4 per i32) ----------------
    def load_b_raw(base_k, n_blk, n_intra):
        # raw[k0][j] = i32 holding 8 fp4 for K micro-step (k0*4 + j).
        raw = []
        for k0i in range_constexpr(_k0_count):
            k0 = (base_k + fx.Int32(k0i * 128)) // fx.Int32(128)
            idx_pack = fx.Int32(
                crd2idx(
                    [
                        fx.Int64(n_blk),
                        fx.Int64(k0),
                        fx.Int64(lane_div_16),
                        fx.Int64(n_intra),
                        fx.Int64(0),
                    ],
                    layout_b,
                )
            )
            # idx_pack is a fp4-byte offset; dwordx4 tile index = (idx_pack/4 dwords)/4.
            r = fx.make_rmem_tensor(w_reg_lay, fx.Int32)
            fx.copy(w_copy_atom, fx.slice(w_tiles, (None, idx_pack // fx.Int32(16))), r)
            v4 = fx.Vector(fx.memref_load_vec(r))
            raw.append([fx.Int32(v4[j]) for j in range(4)])
        return raw

    def load_b_raw_int4(base_k, n_blk, n_intra):
        # int4 OLD layout (kpack=8): port MFMA-K32 block c = k0*4 + lane_div_16 is the
        # concat of two adjacent klane slots (2*(c%2), +1) 128 B apart -> two dwordx2
        # loads. raw[k0i] = [slot_a i32#0, i32#1, slot_b i32#0, i32#1]; each i32 is one
        # MFMA K-step, "interleaved-by-4" packed (upconvert_b old_pack=True).
        raw = []
        for k0i in range_constexpr(_k0_count):
            k0 = (base_k + fx.Int32(k0i * 128)) // fx.Int32(128)
            c = k0 * fx.Int32(4) + lane_div_16
            old_k0 = c // fx.Int32(2)
            old_klane_a = (c % fx.Int32(2)) * fx.Int32(2)
            four = []
            for slot in range_constexpr(2):
                idx = fx.Int32(
                    crd2idx(
                        [
                            fx.Int64(n_blk),
                            fx.Int64(old_k0),
                            fx.Int64(old_klane_a + fx.Int32(slot)),
                            fx.Int64(n_intra),
                            fx.Int64(0),
                        ],
                        layout_b_int4,
                    )
                )
                # idx is an int4-nibble byte offset; dwordx2 tile index = (idx bytes)/8.
                r = fx.make_rmem_tensor(w_reg_lay2, fx.Int32)
                fx.copy(
                    w_copy_atom64, fx.slice(w_tiles8, (None, idx // fx.Int32(8))), r
                )
                v2 = fx.Vector(fx.memref_load_vec(r))
                four.append(fx.Int32(v2[0]))
                four.append(fx.Int32(v2[1]))
            raw.append(four)
        return raw

    def load_b_raw_bf16(base_k, n_blk, n_intra):
        # Raw bf16 W: one dwordx4 (8 bf16) per ku = one MFMA K32 B fragment (v8bf16,
        # the MMA operand directly). K map matches fp4: bf_k0 = base_k//32 + (ku//4)*4
        # + klane_hw, bf_klane = ku%4.
        raw = []
        base_k0 = base_k // fx.Int32(32)
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            bf_k0 = base_k0 + fx.Int32(_k0_blk * 4) + lane_div_16
            bf_klane = fx.Int32(ku % 4)
            elem_idx = fx.Int32(
                crd2idx(
                    [
                        fx.Int64(n_blk),
                        fx.Int64(bf_k0),
                        fx.Int64(bf_klane),
                        fx.Int64(n_intra),
                        fx.Int64(0),
                    ],
                    layout_b_bf16,
                )
            )
            # elem_idx is a bf16-elem offset; dword index = elem_idx*2/4, tile idx = /4.
            r = fx.make_rmem_tensor(w_reg_lay, fx.Int32)
            fx.copy(w_copy_atom, fx.slice(w_tiles, (None, elem_idx // fx.Int32(8))), r)
            raw.append(fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16))  # v8bf16
        return raw

    def load_b_scale(base_k, mni, n_pack):
        # aiter _get_scale_f32: adj_ku = base_k//32 + (ku//4)*4 + lane_div_16. Per-lane
        # scalar e8m0 load on buffer_ops (no layout form, dict-cached across ku).
        scales = []
        cache = {}
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            k_pack_sub = (adj_ku // fx.Int32(4)) % fx.Int32(2)
            s_ku = adj_ku // fx.Int32(8)
            if _k0_blk not in cache:
                idx = (
                    mni * fx.Int32(sc_stride_n0)
                    + s_ku * fx.Int32(sc_stride_k0)
                    + lane_div_16 * fx.Int32(sc_stride_klane)
                    + lane_mod_16
                )
                cache[_k0_blk] = _buffer_i32_scalar_read(sw_tiles, idx, sw_read_atom)
            packed = cache[_k0_blk]
            byte_even = k_pack_sub * fx.Int32(2)
            byte_odd = byte_even + fx.Int32(1)
            se = _e8m0_byte_to_f32(packed, byte_even)
            so = _e8m0_byte_to_f32(packed, byte_odd)
            scales.append((n_pack == fx.Int32(0)).select(se, so))
        return scales

    def load_b_scale_int4(base_k, n_full):
        # int4 groupwise (bf16-pair) scale, OLD-kernel (E, G//2, N, 2) layout: dword =
        # e*(G//2*N) + (adj_ku//2)*N + n_full, half by parity. N-major, so 16 consecutive
        # N-lanes share one 64 B line (vs a G//2-strided gather).
        scales = []
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            pair_idx = adj_ku // fx.Int32(2)
            dword = scale_expert_base() + pair_idx * fx.Int32(N_OUT) + n_full
            packed = _buffer_i32_scalar_read(sw_tiles, dword, sw_read_atom)
            # even adj_ku -> low bf16, odd -> high.
            lo = (packed << fx.Int32(16)).bitcast(fx.Float32)
            hi = (packed & fx.Int32(0xFFFF0000)).bitcast(fx.Float32)
            scales.append((adj_ku % fx.Int32(2) == fx.Int32(0)).select(lo, hi))
        return scales

    vec2_bf16 = T.vec(2, T.bf16)

    def upconvert_b(raw, ku, scale_f32):
        if const_expr(_is_bf16):
            # raw[ku] is already the v8bf16 MMA operand (no scale, no upconvert).
            return raw[ku]
        i32_val = _raw(raw[ku // 4][ku % 4])
        if const_expr(_is_int4):
            return _int4_nibble_to_bf16x8(
                fx.Int32(i32_val), scale_f32, use_k16=use_k16, old_pack=True
            )
        # raw[ku//4][ku%4] i32 holds 8 fp4 -> 4x cvt (v2bf16, sel 0..3) -> v8bf16.
        s_raw = _raw(scale_f32)
        i32s = []
        for sel in range_constexpr(4):
            p = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
            i32s.append(fx.Int32(fx.Vector(p).bitcast(fx.Int32)[0]))
        v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
        return v4i32.bitcast(fx.BFloat16)  # v8bf16

    # ---- dtype-selected surface: the stages only ever see these ----------------
    if const_expr(_is_bf16):
        _load_raw = load_b_raw_bf16
    elif const_expr(_is_int4):
        _load_raw = load_b_raw_int4
    else:
        _load_raw = load_b_raw

    def load_raw(base_k, col):
        return _load_raw(base_k, col.n_blk, col.n_intra)

    def load_scale(base_k, col):
        if const_expr(_is_bf16):
            # No scale: one None per K micro-step keeps the per-ku indexing uniform.
            return [None] * k_unroll
        if const_expr(_is_int4):
            return load_b_scale_int4(base_k, col.sc_n)
        return load_b_scale(base_k, col.sc_blk, col.sc_pack)

    return _BLoader(
        col=col,
        col_pair=col_pair,
        load_raw=load_raw,
        load_scale=load_scale,
        upconvert=upconvert_b,
    )
