# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.layout_utils import crd2idx

from .common import (
    _buffer_rsrc,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_view,
    _int4_nibble_to_bf16x8,
    _lds_ptr3,
    _raw,
    _udiv,
    _umod,
    kmchunks_for,
    lds_acc_bytes_for,
)
from .gemm1 import A16WI4_GROUP_SIZE, _a16w4_swizzle_xor16, _e8m0_byte_to_f32

# gfx950 (MI350/MI355X) CU count. Used to cap the persistent gemm2 grid so
# high-expert-count launches (E896) do not over-launch ~max_m_blocks empty CTAs.
NUM_CU = 256


# @flyc.jit is load-bearing: it AST-rewrites the ``if token_id < i32_M`` bound
# check into an scf.if. Without it the guard runs as a plain Python if (dropped
# at trace time), so the atomic-fadd scatter fires unconditionally on padded/OOB
# sorted rows -- ~10x extra atomic HBM traffic (gemm2 s2 39us -> ~490us at E896).
@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
):
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    # 4 waves split the BN(=TILE_N) tile: each wave owns _n_per_wave cols (generic over
    # BN, so the int4 tile_n=128 geometry works, not just tile_n=256).
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # readback: each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
        weight.append(
            llvm.load(
                T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


def _gemm2_body_a16w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    NE,
    b_cache_mod=2,
    w_dtype="mxfp4",
):
    """a16w4/a16wi4/a16w16 stage2 body. K=inter_dim (contraction), N=model_dim (N_OUT).

    A = bf16 stage1 intermediate indexed by SORTED position: A[sorted_row, k]. W2 =
    mxfp4/int4/bf16 (see gemm1 for the W variants). Output = bf16 scattered to token
    rows (atomic-fadd, routing-weighted) at [tokens, model_dim].
    """
    _is_int4 = w_dtype == "int4"
    _is_bf16 = (
        w_dtype == "bf16"
    )  # a16w16: raw bf16 W (unpacked, no scale, no upconvert)
    elem_bytes = 2
    KH_TILE_BYTES = TILE_K * elem_bytes
    LDS_STRIDE = TILE_K
    K = INTER
    K_HALF = K // 2
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    k_unroll = KH_TILE_BYTES // 64
    _k0_count = TILE_K // 128
    # 4 waves split the TILE_N tile (matches the atomic-epilog wave-split).
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KH_TILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N

    # W2 (mxfp4) preshuffle layout (make_preshuffle_b_layout, N-major, fp4).
    bl_k0 = K_HALF // 64
    bl_stride_klane = 256
    bl_stride_k0 = 1024
    bl_stride_n0 = bl_k0 * bl_stride_k0
    layout_b = fx.make_layout(
        (N_OUT // 16, bl_k0, 4, 16, 16),
        (bl_stride_n0, bl_stride_k0, bl_stride_klane, 16, 1),
    )
    # W2 (raw bf16) preshuffle layout (elem_bytes=2, N-major == shuffle_weight (16,16)):
    #   shape (N_OUT/16, K/32, 4, 16, 8), bf16-elem strides. One kpack=8 bf16=one MFMA
    #   K32 fragment; K reindexed to match the fp4 (klane_hw, ku)->K order (see load_b_raw_bf16).
    bfl_k0 = K // 32
    bfl_stride_klane = 128
    bfl_stride_k0 = 512
    bfl_stride_n0 = bfl_k0 * bfl_stride_k0
    layout_b_bf16 = fx.make_layout(
        (N_OUT // 16, bfl_k0, 4, 16, 8),
        (bfl_stride_n0, bfl_stride_k0, bfl_stride_klane, 8, 1),
    )
    scale_k_padded = ((K + 255) // 256) * 256
    sc_k1 = ((scale_k_padded // 32) // 4) // 2
    sc_stride_klane = 16
    sc_stride_k0 = 64
    sc_stride_n0 = sc_k1 * sc_stride_k0

    # a16wi4 groupwise scale: bf16 pairs (E, N_OUT, num_groups//2, 2), K = inter_dim.
    _num_groups = K // A16WI4_GROUP_SIZE
    _g_half = _num_groups // 2

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first sorted row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)
    expert_off = e * fx.Int32(N_OUT)

    # bf16 W (2 B/elem) overflows the 32-bit buffer num_records / i32 byte-offset at
    # large E; fold the per-expert base into the i64 resource address and index W
    # within the expert. mxfp4/int4 keep the whole-tensor path (byte-identical).
    if const_expr(_is_bf16):
        _w_per_expert_bytes = N_OUT * (K * 2)
        w_base_i64 = fx.Int64(arg_bq) + fx.Int64(e) * fx.Int64(_w_per_expert_bytes)
        w_rsrc = _buffer_rsrc(
            w_base_i64, num_records_bytes=min(_w_per_expert_bytes, 0xFFFFFFFF)
        )
    else:
        _w_bytes = NE * N_OUT * K_HALF
        w_rsrc = _buffer_rsrc(arg_bq, num_records_bytes=min(_w_bytes, 0xFFFFFFFF))
    if _is_int4:
        _sw_bytes = NE * N_OUT * _g_half * 4
    else:
        _sw_bytes = NE * N_OUT * (scale_k_padded // 32)
    sw_rsrc = (
        None
        if _is_bf16
        else _buffer_rsrc(arg_bscale, num_records_bytes=min(_sw_bytes, 0xFFFFFFFF))
    )

    # ---- A gather (per-thread) -> LDS. A row = SORTED position m_row + row_local.
    total_threads = 256
    bytes_per_thread = (BM * TILE_K * elem_bytes) // total_threads
    x_load_bytes = 16
    num_x_loads = bytes_per_thread // x_load_bytes
    tile_k_dwords = (TILE_K * elem_bytes) // 4
    c_k_div4 = (K * elem_bytes) // 4
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    chunk_i32 = x_load_bytes // 4
    tx_base = tx_i32 * fx.Int32(chunk_i32)

    x_row_local = []
    x_col_dw = []
    x_row_base_div4 = []
    for i in range_constexpr(num_x_loads):
        tile_idx = tx_base + fx.Int32(i * total_threads * chunk_i32)
        row_local = tile_idx // fx.Int32(tile_k_dwords)
        col_dw = tile_idx % fx.Int32(tile_k_dwords)
        x_row_local.append(row_local)
        x_col_dw.append(col_dw)
        sorted_row = m_row + row_local
        x_row_base_div4.append(sorted_row * fx.Int32(c_k_div4))

    x_buf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF))
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(4, 1))
    x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(BM * LDS_STRIDE // 2, 1),
    )
    s_x_i32x4_tiles = fx.logical_divide(s_x_i32_flat, fx.make_layout(4, 1))
    a_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def dma_a_tile_to_lds(base_k):
        base_k_div4 = (base_k * fx.Int32(elem_bytes)) // fx.Int32(4)
        for i in range_constexpr(num_x_loads):
            col_bytes = x_col_dw[i] * fx.Int32(4)
            # A-LDS bank-conflict XOR swizzle: the LDS dest stays LINEAR (buffer_load_lds
            # does NOT honor an arbitrary swizzled per-lane LDS dest -> NaN); instead
            # swizzle the GMEM source column so linear LDS slot [row][col] holds
            # A[row][swz(row,col)], and lds_load_a applies the SAME swizzle on read.
            col_sw = _a16w4_swizzle_xor16(
                x_row_local[i], col_bytes, fx.Int32(k_blocks16), enable=True
            )
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte = row_k_dw * fx.Int32(4) + col_sw
            lds_byte = x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
            fx.copy(
                x_dma_atom,
                fx.slice(x_dma_tiles4, (None, global_byte // fx.Int32(16))),
                fx.slice(s_x_i32x4_tiles, (None, lds_byte // fx.Int32(16))),
            )

    row_a_lds = lane_mod_16
    col_base_bytes_L = lane_div_16 * fx.Int32(64)

    def _a_col_bytes_for_ku(ku):
        _k0_blk = ku // 4
        _ku_in = ku % 4
        return col_base_bytes_L + fx.Int32(_ku_in * 16 + _k0_blk * 256)

    def lds_load_a(mi, ku):
        row = row_a_lds + fx.Int32(mi * 16)
        # Same XOR swizzle as the DMA write (read cols and the mask are 16 B multiples,
        # so 16 B alignment is preserved).
        col_swz_bytes = _a16w4_swizzle_xor16(
            row, _a_col_bytes_for_ku(ku), fx.Int32(k_blocks16), enable=True
        )
        byte_off = row * fx.Int32(KH_TILE_BYTES) + col_swz_bytes
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy_atom_call(
            a_copy_atom, fx.slice(s_x_i32x4_tiles, (None, byte_off // fx.Int32(16))), r
        )
        return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16)

    def load_b_raw(base_k, n_blk, n_intra):
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
            v4 = buffer_ops.buffer_load(
                _raw(w_rsrc),
                _raw(idx_pack // fx.Int32(4)),
                vec_width=4,
                dtype=T.i32,
                cache_modifier=b_cache_mod,
            )
            v4 = fx.Vector(v4)
            raw.append([fx.Int32(v4[j]) for j in range(4)])
        return raw

    def load_b_raw_bf16(base_k, n_blk, n_intra):
        # Raw bf16 W: one dwordx4 (8 bf16) per ku = one MFMA K32 fragment. Index to
        # match the fp4 (klane_hw=lane_div_16, ku)->K order (see gemm1 counterpart).
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
            v4 = buffer_ops.buffer_load(
                _raw(w_rsrc),
                _raw(elem_idx // fx.Int32(2)),
                vec_width=4,
                dtype=T.i32,
                cache_modifier=b_cache_mod,
            )
            raw.append(fx.Vector(v4).bitcast(fx.BFloat16))  # v8bf16
        return raw

    def load_b_scale(base_k, mni, n_pack):
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
                cache[_k0_blk] = fx.Int32(
                    buffer_ops.buffer_load(
                        _raw(sw_rsrc), _raw(idx), vec_width=1, dtype=T.i32
                    )
                )
            packed = cache[_k0_blk]
            byte_even = k_pack_sub * fx.Int32(2)
            byte_odd = byte_even + fx.Int32(1)
            se = _e8m0_byte_to_f32(packed, byte_even)
            so = _e8m0_byte_to_f32(packed, byte_odd)
            is_even = arith.cmpi(
                arith.CmpIPredicate.eq, _raw(n_pack), _raw(fx.Int32(0))
            )
            scales.append(fx.Float32(arith.select(is_even, _raw(se), _raw(so))))
        return scales

    def load_b_scale_int4(base_k, col_g):
        # int4 groupwise (bf16-pair) scale, per-lane N = col_g. See gemm1 counterpart.
        scales = []
        base_dword = col_g * fx.Int32(_g_half)
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            pair_idx = adj_ku // fx.Int32(2)
            packed = fx.Int32(
                buffer_ops.buffer_load(
                    _raw(sw_rsrc), _raw(base_dword + pair_idx), vec_width=1, dtype=T.i32
                )
            )
            lo = fx.Float32(_raw(packed << fx.Int32(16)).bitcast(T.f32))
            hi = fx.Float32(_raw(packed & fx.Int32(0xFFFF0000)).bitcast(T.f32))
            is_even = arith.cmpi(
                arith.CmpIPredicate.eq, _raw(adj_ku % fx.Int32(2)), _raw(fx.Int32(0))
            )
            scales.append(fx.Float32(arith.select(is_even, _raw(lo), _raw(hi))))
        return scales

    vec2_bf16 = ir.Type.parse("vector<2xbf16>")

    def upconvert_b(raw, ku, scale_f32):
        if const_expr(_is_bf16):
            return raw[ku]  # already v8bf16 (no scale, no upconvert)
        i32_val = _raw(raw[ku // 4][ku % 4])
        if const_expr(_is_int4):
            return _int4_nibble_to_bf16x8(fx.Int32(i32_val), scale_f32)
        s_raw = _raw(scale_f32)
        i32s = []
        for sel in range_constexpr(4):
            pp = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
            i32s.append(fx.Int32(fx.Vector(pp).bitcast(fx.Int32)[0]))
        v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
        return v4i32.bitcast(fx.BFloat16)

    # ---- N-column addressing (W2 cols of model_dim; wave owns _n_per_wave) ------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    col_g_list = []
    n_blk_list, n_intra_list, scale_mni_list, scale_np_list = [], [], [], []
    for ni in range_constexpr(num_acc_n):
        col_g = by_n + n_tile_base + fx.Int32(ni * 16) + lane_mod_16
        col_g_list.append(col_g)
        # bf16 W folds expert_off into the resource base (see w_rsrc); mxfp4/int4 index it.
        _row_expert_off = fx.Int32(0) if const_expr(_is_bf16) else expert_off
        row_w = _row_expert_off + col_g
        n_blk_list.append(row_w // fx.Int32(16))
        n_intra_list.append(row_w % fx.Int32(16))
        ng = expert_off + by_n + n_tile_base + fx.Int32(ni * 16)
        scale_mni_list.append(ng // fx.Int32(32))
        scale_np_list.append((ng // fx.Int32(16)) % fx.Int32(2))
    # int4 groupwise scale: per-lane N = expert_off + col_g (row_w already computed).
    if const_expr(_is_int4):
        scale_n_list = [
            expert_off + col_g_list[ni] for ni in range_constexpr(num_acc_n)
        ]

    # ---- accumulators: accm[mi][ni] f32[4] (layout the atomic epilog expects) --
    acc_layout = fx.make_layout(4, 1)
    accm = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

    def _bf16_frag(v8):
        t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        t.store(v8)
        return t

    def _mma(acc, a8, b8):
        fx.gemm(mma_atom, acc, _bf16_frag(a8), _bf16_frag(b8), acc)

    for kt in range_constexpr(K_TILES_TOTAL):
        base_k = fx.Int32(kt * TILE_K)
        dma_a_tile_to_lds(base_k)
        if const_expr(_is_bf16):
            b_raw = [
                load_b_raw_bf16(base_k, n_blk_list[ni], n_intra_list[ni])
                for ni in range_constexpr(num_acc_n)
            ]
            b_sc = None
        else:
            b_raw = [
                load_b_raw(base_k, n_blk_list[ni], n_intra_list[ni])
                for ni in range_constexpr(num_acc_n)
            ]
            if const_expr(_is_int4):
                b_sc = [
                    load_b_scale_int4(base_k, scale_n_list[ni])
                    for ni in range_constexpr(num_acc_n)
                ]
            else:
                b_sc = [
                    load_b_scale(base_k, scale_mni_list[ni], scale_np_list[ni])
                    for ni in range_constexpr(num_acc_n)
                ]
        gpu.barrier()
        for ni in range_constexpr(num_acc_n):
            for ku in range_constexpr(k_unroll):
                _bsc = None if const_expr(_is_bf16) else b_sc[ni][ku]
                bb = upconvert_b(b_raw[ni], ku, _bsc)
                for mi in range_constexpr(m_repeat):
                    a8 = lds_load_a(mi, ku)
                    _mma(accm[mi][ni], a8, bb)
        gpu.barrier()

    # ---- epilogue: atomic bf16 scatter (routing-weighted). The K-loop is done, so
    # the A-LDS region (offset 0) is reused for the epilog's f32 acc staging.
    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    accm_v = [
        [accm[i][J].load().ir_value() for J in range(num_acc_n)]
        for i in range(m_repeat)
    ]
    _atomic_bf16_epilog(
        lds_acc_base_i32,
        accm_v,
        arg_out,
        arg_stids,
        arg_sweights,
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT,
        TILE_N,
    )


def gemm2_a16w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks, persist=False):
    """Flattened launch grid for a16w4 gemm2.

    Non-persistent (default): one CTA per (m-block x n-block) tile over the padded
    ``max_m_blocks``. Persistent (``persist=True``): cap the launch to
    ``min(total_work, NUM_CU)`` CTAs (only when the padded work exceeds ``NUM_CU*4``)
    and let each CTA loop over its real work-tiles (bounded via the cumsum).
    """
    total_work = int(max_m_blocks) * (N_OUT // TILE_N)
    if persist and total_work > NUM_CU * 4:
        return min(total_work, NUM_CU)
    return total_work


def compile_gemm2_a16w4_port(
    BM=32,
    *,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    xcd_swizzle=1,
    b_cache_mod=2,
    waves_per_eu=None,
    w_dtype="mxfp4",
    persist=False,
):
    """a16w4/a16wi4/a16w16 (bf16 intermediate A x mxfp4/int4/bf16 W2) stage2 builder.

    N_OUT = model_dim (down-proj output width). D_INTER = inter_dim (contraction).
    Output is bf16 [tokens, model_dim] via the atomic (routing-weighted) scatter.

    ``xcd_swizzle`` (>0) round-robins the launch index bijectively across the 8 XCDs
    to balance per-XCD/HBM-channel traffic (gemm2 is HBM-bandwidth-bound), and enables
    an optional M-group swizzle for per-XCD L2 locality (group size = xcd_swizzle
    m-blocks).
    """
    assert w_dtype in (
        "mxfp4",
        "int4",
        "bf16",
    ), f"w_dtype must be 'mxfp4', 'int4' or 'bf16', got {w_dtype!r}"
    _K = D_INTER
    assert _K % TILE_K == 0, f"D_INTER (K) must be a multiple of {TILE_K}, got {_K}"
    assert (
        N_OUT % TILE_N == 0
    ), f"model_dim (N_OUT) must be a multiple of {TILE_N}, got {N_OUT}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    _num_n_blocks = N_OUT // TILE_N
    KH_TILE_BYTES = TILE_K * 2

    # LDS: A tile (BM x TILE_K bf16) then f32 accumulator region (BM x TILE_N f32).
    _a_bytes = BM * KH_TILE_BYTES
    _acc_bytes = lds_acc_bytes_for(BM, TILE_N)
    _lds_bytes = _a_bytes + _acc_bytes

    _wd_tag = "" if w_dtype == "mxfp4" else f"_{w_dtype}"
    _name = f"gemm2_a16w4{_wd_tag}_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_tn{TILE_N}"
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"
    if persist:
        _name += "_persist"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        # Bijective XCD round-robin over the valid tiles [0, bound) to balance the
        # per-XCD/HBM-channel weight-load traffic. With xcd_swizzle>0, additionally
        # group-swizzle along M for per-XCD L2 locality.
        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        def _run_tile(tile):
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                tile,
                lane,
                wave,
                i32_M,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                INTER=_K,
                NE=NE,
                b_cache_mod=b_cache_mod,
                w_dtype=w_dtype,
            )

        if const_expr(persist):
            # Persistent CU-limited grid: the launch is capped to ~NUM_CU CTAs. Each
            # CTA processes tile bx_i32 then strides by the grid size over the real
            # work-tiles [0, bound). _xcd_np is applied to every visited index, so each
            # real tile is computed exactly once with the same mapping as the
            # non-persistent grid. The loop-top barrier separates the previous tile's
            # atomic-epilog LDS use from the next tile's A-DMA (shared offset-0 region).
            grid_nb = fx.Int32(gpu.grid_dim.x)
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                gpu.barrier()
                _run_tile(_xcd_np(fx.Int32(iv)))
        else:
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
