# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T, as_ir_value

from .kernels_common import LOG2E
from .mxfp4_gemm1 import _bm_constants
from .mxfp4_gemm1 import gemm1_grid as gemm1_grid
from .mxfp4_gemm2_ep32 import (
    _bm160_a_fragment,
    _bm160_epilogue_ids,
    _bm160_mma_loop,
    lds_slot_scopes,
)
from .mxfp4_gemm_common import (
    _activation_mul_batch,
    _buffer_rsrc,
    _dpp_umax_step,
    _e8m0_from_amax,
    _fabs_f32,
    _global_f32_at,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _global_i32_load,
    _global_scalar_tiles,
    _inline_dpp_quad_amax,
    _layout_idx,
    _lds_ptr3,
    _lds_swizzle_mask,
    _scalar_store,
    _scale_mma_atoms,
    _udiv,
    _umod,
    bq_bytes_for,
    bscale_bytes_for,
    global_typed_ptr,
    k_half_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kStages,
    lds_swizzle_mask_f8,
)
from .tensor_shim import _to_raw as _raw

ACC_LDS_PAD_DW = 4


# EP32 A8W4 GEMM1 uses interleaved gate/up weights and sorted FP8 output.
# BM64 compacts active tiles and folds a <=16-route tail with the existing extra 16 rows;
# BM96 compute a full sorted block. Contiguous tile runs per XCD keep
# gathered activations local to L2. NT weights avoid evicting those activations.
# The fixed pipeline consumes one 16-row A fragment at a time and stages
# 32 C rows per epilogue region, keeping the original activation arithmetic.
EP32_G1_VARIANTS = {
    (64, True, False), (64, False, False),
    (96, True, False), (96, False, False),
}


@flyc.jit
def _gemm1_body(
    lds_raw_ptr,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    arg_bias,
    bx_i32,
    lane,
    wave,
    use_nt,
    i32_ntok,
    i32_total_m_blocks,
    *,
    BM,
    BN,
    BK,
    inline_quant=False,
    prefetch_hidden=False,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    enable_bias=False,
    K,
    N_OUT,
    NE,
    interleave=False,
    native_scale_layout=False,
    num_waves=4,
    k_wave=1,
    epi_splits=1,
    k_stages=kStages,
    active_chunks=None,
    khalf_loop=False,
    epi32=False,
    fast_situ=False,
):
    # active_chunks: compile-time number of leading 16-row chunks that hold real
    # rows (None = all). Chunks past it are padding: their A loads, LDS reads,
    # MFMAs and activation are skipped and they emit zeros.
    # A-code tile bytes/row: fp4 packs 2 codes/byte (BK/2); fp8 is 1 B/elem (BK).
    KH_TILE = BK if a_dtype == "fp8" else BK // 2
    K_HALF = k_half_for(K)
    # A row bytes: fp4 = K/2, fp8 = K (B is always mxfp4 -> keeps K_HALF).
    A_ROW_BYTES = K if a_dtype == "fp8" else K_HALF
    K_TILES_TOTAL = K // BK
    K_TILES_PER_WAVE = K_TILES_TOTAL // k_wave
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    ASCALE_CHUNK_BYTES = kAS_per_chunk_dw * 4
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = N_OUT // BN
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = kas_per_chunk_dw_for(inter)
    OUT_ROW_BYTES = inter if out_dtype == "fp8" else k_half_for(inter)
    kAStages, kSubBlocks, kMChunks, _ = _bm_constants(
        BM, BN, KH_TILE, K_TILES_TOTAL, k_wave, epi_splits, k_stages
    )
    kUnroll = K_TILES_PER_WAVE - k_stages
    ACT = kMChunks if active_chunks is None else active_chunks
    ACT_SUBS = (ACT + 1) // 2
    # NCH > kMChunks: the tile also covers the first 16 rows of the next sorted
    # M block (a padding-heavy tail block of the same expert), 16*NCH rows total.
    NCH = max(kMChunks, ACT)
    A_ROWS = 16 * NCH
    AS_K_SLOTS = 2 if BM == 96 else K_TILES_TOTAL
    if const_expr(ACT != kMChunks):
        assert BM == 64 and a_dtype == "fp8" and num_waves == 4
        assert ACT in (1, 2, 5)

    k_split = (
        a_dtype == "fp4"
        and not inline_quant
        and BM == 128
        and BN == 256
        and num_waves == 4
        and kMChunks > 1
        and kUnroll >= 1
    )
    a_reg_depth = min(2, kUnroll) if k_split else 0

    BN_INT = BN // 2
    b_aux = 2 if use_nt else 0
    M_REPS = BM // 16
    N_REPS = BN // (num_waves * 16)
    EPILOGUE_M_REPS = 1 if num_waves == 2 else M_REPS
    B_SCALE_REPS = max(1, N_REPS // 2) if interleave else 2
    wave_n = wave % fx.Int32(num_waves)
    wave_k = wave // fx.Int32(num_waves)
    aq_group_base = wave_k * fx.Int32(kAStages * A_ROWS * KH_TILE)

    mem_1x1 = fx.make_layout(1, 1)
    mem_4x1 = fx.make_layout(4, 1)
    mem_8x1 = fx.make_layout(8, 1)

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, as_ir_value(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = fx.Int64(i32_ntok * fx.Int32(A_ROW_BYTES))
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = fx.Int64(i32_total_m_blocks) * fx.Int64(_asc_per_mb)

    bq_tiles = _global_i32_buffer_tiles(arg_bq, BQ_BYTES, 4)
    bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_aux), fx.Int32)

    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

    # aq/ascale: global->LDS async DMA (no register fragment), via BufferCopyLDS.
    aq_buf = _global_i32_buffer_view(arg_aq, aq_num_records)
    aq_dma_tiles4 = fx.logical_divide(aq_buf, mem_4x1)
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    aq_reg_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)

    ascale_buf = _global_i32_buffer_view(arg_ascale, ascale_num)
    ascale_dma_tiles4 = fx.logical_divide(ascale_buf, mem_4x1)
    ascale_dma_tiles1 = fx.logical_divide(ascale_buf, mem_1x1)
    ascale_dma_atom16 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    ascale_dma_atom4 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), fx.Int32)


    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in the
    # epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    cached_row_inline = None
    for sub in range_constexpr(kSubBlocks):
        idx = m_row + wave_n * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
        actual_row = _global_i32_at(arg_mind, idx)
        cached_actual_row.append(
            [
                fx.Int32(
                    rocdl.ds_bpermute(
                        T.i32,
                        (lane_div_16 + fx.Int32(row_group * 4)) * fx.Int32(32),
                        as_ir_value(actual_row),
                    )
                )
                for row_group in range_constexpr(2)
            ]
        )
    tail_row = None
    if const_expr(NCH > kMChunks):
        # Tail chunk rows BM..BM+15: wave w DMAs rows BM+4w..BM+4w+3.
        tail_row = _global_i32_at(
            arg_mind, m_row + fx.Int32(BM) + wave_n * fx.Int32(4) + lane_div_16
        )

    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(N_REPS):
        col = (
            n_block_idx * fx.Int32(BN)
            + wave_n * fx.Int32(N_REPS * 16)
            + fx.Int32(j * 16)
        )
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # Each scale word covers two consecutive 16-column weight tiles.
    mni_base = n_block_idx * fx.Int32(BN // 32) + wave_n * fx.Int32(
        N_REPS
    ) // fx.Int32(2)
    np_list = [mni_base, mni_base + fx.Int32(1)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(B_SCALE_REPS):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    scale_atoms = _scale_mma_atoms(a_dtype)
    accm = [
        [fx.make_rmem_tensor(mem_4x1, fx.Float32) for _ in range(N_REPS)]
        for _ in range(NCH)
    ]
    for i in range_constexpr(NCH):
        for J in range_constexpr(N_REPS):
            accm[i][J].store(fx.full(4, 0.0, fx.Float32))
    b = [[[None, None] for _ in range(N_REPS)] for _ in range(k_stages)]
    b_scale_v = [[None, None] for _ in range(k_stages)]

    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(k_wave * kAStages * A_ROWS * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, mem_4x1)
    s_aq_i32x1_tiles = fx.logical_divide(s_aq_i32_flat, mem_1x1)
    i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def issue_a_load_lds(slot, kt):
        # A wave-wide BufferCopyLDS128b writes 1024 contiguous bytes. That is
        # eight FP4 rows or four FP8 rows. Larger FP8 tiles use two four-row
        # passes; BM32 keeps register staging, which is faster at low occupancy.
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave_n * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            if const_expr(a_dtype == "fp8" and BM >= 64 and ACT < kMChunks):
                # BM64: wave w owns rows 16w..16w+15, i.e. chunk w.
                if wave_n < fx.Int32(ACT):
                    for row_group in range_constexpr(2):
                        dst_row_base = lds_row + fx.Int32(row_group * 4)
                        mask = _lds_swizzle_mask(dst_row_base + lane_div_16)
                        voffset = ((lane_mod_16 * fx.Int32(16)) ^ mask) + cached_actual_row[
                            sub
                        ][row_group] * fx.Int32(A_ROW_BYTES)
                        off = (
                            aq_group_base
                            + fx.Int32(slot * (A_ROWS * KH_TILE))
                            + dst_row_base * fx.Int32(KH_TILE)
                        )
                        fx.copy(
                            aq_dma_atom,
                            fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                            fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                            soffset=fx.Int32(kt * KH_TILE) // 4,
                        )
            elif const_expr(a_dtype == "fp8" and BM >= 64):
                for row_group in range_constexpr(2):
                    dst_row_base = lds_row + fx.Int32(row_group * 4)
                    if const_expr(BM == 96):
                        mask = lds_swizzle_mask_f8(dst_row_base + lane_div_16, KH_TILE)
                    else:
                        mask = _lds_swizzle_mask(dst_row_base + lane_div_16)
                    voffset = ((lane_mod_16 * fx.Int32(16)) ^ mask) + cached_actual_row[
                        sub
                    ][row_group] * fx.Int32(A_ROW_BYTES)
                    off = (
                        aq_group_base
                        + fx.Int32(slot * (A_ROWS * KH_TILE))
                        + dst_row_base * fx.Int32(KH_TILE)
                    )
                    fx.copy(
                        aq_dma_atom,
                        fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                        fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                        soffset=fx.Int32(kt * KH_TILE) // 4,
                    )
            elif const_expr(a_dtype == "fp8"):
                actual_row = cached_actual_row[sub]
                dst_row = lds_row + lane_div_8
                mask = _lds_swizzle_mask(dst_row)
                for half in range_constexpr(2):
                    half_off = fx.Int32(half * 128)
                    src_off = (
                        actual_row * fx.Int32(A_ROW_BYTES)
                        + half_off
                        + lane_mod_8 * fx.Int32(16)
                    )
                    dst_off = (
                        aq_group_base
                        + fx.Int32(slot * (A_ROWS * KH_TILE))
                        + dst_row * fx.Int32(KH_TILE)
                        + half_off
                        + ((lane_mod_8 * fx.Int32(16)) ^ mask)
                    )
                    r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
                    fx.copy(
                        aq_reg_atom,
                        fx.slice(aq_dma_tiles4, (None, src_off // fx.Int32(16))),
                        r,
                        soffset=fx.Int32(kt * KH_TILE) // 4,
                    )
                    fx.copy(
                        i32x4_copy_atom,
                        r,
                        fx.slice(s_aq_i32x4_tiles, (None, dst_off // fx.Int32(16))),
                    )
            else:
                mask = _lds_swizzle_mask(lds_row + lane_div_8)
                voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                    sub
                ] * fx.Int32(A_ROW_BYTES)
                off = (
                    aq_group_base
                    + fx.Int32(slot * (A_ROWS * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                )
                fx.copy(
                    aq_dma_atom,
                    fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                    fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                    soffset=fx.Int32(kt * KH_TILE) // 4,
                )

    def issue_a_load_lds_tail(slot, kt):
        dst_row_base = fx.Int32(BM) + wave_n * fx.Int32(4)
        mask = _lds_swizzle_mask(dst_row_base + lane_div_16)
        voffset = ((lane_mod_16 * fx.Int32(16)) ^ mask) + tail_row * fx.Int32(A_ROW_BYTES)
        off = (
            aq_group_base
            + fx.Int32(slot * (A_ROWS * KH_TILE))
            + dst_row_base * fx.Int32(KH_TILE)
        )
        fx.copy(
            aq_dma_atom,
            fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
            fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
            soffset=fx.Int32(kt * KH_TILE) // 4,
        )


    def _lds_i32x4_frag(tile_idx):
        # ds_read_b128 straight into an i32[4] register fragment (kept as a tensor
        # so it can feed fx.gemm directly).
        r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
        fx.copy(i32x4_copy_atom, fx.slice(s_aq_i32x4_tiles, (None, tile_idx)), r)
        return r

    def issue_a_ds_read(slot, khalf=None, chunks=None):
        if const_expr(BM == 96 and a_dtype == "fp8"):
            mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE)
        else:
            mask = _lds_swizzle_mask(lane_mod_16)
        a = [[None, None] for _ in range(NCH)]
        halves = range_constexpr(2) if khalf is None else (khalf,)
        order = [(i, k) for k in halves for i in range_constexpr(NCH)]
        order = [(i, k) for i, k in order if i < ACT and (chunks is None or i in chunks)]
        for i, k in order:
            # fp8 128-K operand (v8i32) = two 16 B halves 64 B apart in the row
            # (f8f6f4 ABI), packed lo++hi. k selects the 128-K half (upper = +128 B).
            kbase = fx.Int32(k * 128)
            lo_col = (lane_div_16 * fx.Int32(16) + kbase) ^ mask
            hi_col = (lane_div_16 * fx.Int32(16) + kbase + fx.Int32(64)) ^ mask
            lds_row = lane_mod_16 + fx.Int32(i * 16)
            boff = (
                aq_group_base
                + fx.Int32(slot * (A_ROWS * KH_TILE))
                + lds_row * fx.Int32(KH_TILE)
            )
            lo = fx.Vector(
                fx.memref_load_vec(_lds_i32x4_frag((boff + lo_col) // fx.Int32(16)))
            )
            hi = fx.Vector(
                fx.memref_load_vec(_lds_i32x4_frag((boff + hi_col) // fx.Int32(16)))
            )
            t = fx.make_rmem_tensor(mem_8x1, fx.Int32)
            t.store(lo.shuffle(hi, list(range(8))))
            a[i][k] = t
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave_n * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave_n * fx.Int32(64) + lane) * fx.Int32(4)
        thread_linear = wave_n * fx.Int32(64) + lane
        for sub in range_constexpr(ACT_SUBS):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            for block4k in range_constexpr(ASCALE_CHUNK_BYTES // 4096):
                byte_off = block4k * 4096
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom16,
                    fx.slice(ascale_dma_tiles4, (None, v16 // fx.Int32(16))),
                    fx.slice(
                        asc_i32x4_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave_n * fx.Int32(1024))
                            // fx.Int32(16),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )
            full4k_bytes = (ASCALE_CHUNK_BYTES // 4096) * 4096
            remainder_bytes = ASCALE_CHUNK_BYTES - full4k_bytes
            for block1k in range_constexpr(remainder_bytes // 1024):
                byte_off = full4k_bytes + block1k * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom4,
                    fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                    fx.slice(
                        asc_i32_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave_n * fx.Int32(256))
                            // fx.Int32(4),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )
            tail_bytes = remainder_bytes % 1024
            if const_expr(tail_bytes > 0):
                byte_off = full4k_bytes + (remainder_bytes // 1024) * 1024
                if thread_linear < fx.Int32(tail_bytes // 4):
                    s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                    fx.copy(
                        ascale_dma_atom4,
                        fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                        fx.slice(
                            asc_i32_tiles,
                            (
                                None,
                                (lds_sub + fx.Int32(byte_off) + wave_n * fx.Int32(256))
                                // fx.Int32(4),
                            ),
                        ),
                        soffset=s_off // fx.Int32(4),
                    )

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(
            fx.Int32,
            fx.add_offset(lds_raw_ptr, k_wave * kAStages * A_ROWS * KH_TILE),
        ),
        fx.make_layout(max(kSubBlocks, (NCH + 1) // 2) * AS_K_SLOTS * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, mem_1x1)
    asc_i32x4_tiles = fx.logical_divide(s_asc_i32_flat, mem_4x1)

    def issue_a_scale_tile(slot, kt):
        # One wave copies each 32-row scale group. Three slots let the next
        # scale tile enter LDS without overwriting another wave's current read.
        if wave_n < fx.Int32(ACT_SUBS):
            src_dw = rocdl.readfirstlane(
                T.i32,
                (m_row // fx.Int32(32) + wave_n) * fx.Int32(kAS_per_chunk_dw)
                + fx.Int32(kt * 64),
            )
            dst_dw = fx.Int32(slot * ACT_SUBS * 64) + wave_n * fx.Int32(64)
            fx.copy(
                ascale_dma_atom4,
                fx.slice(ascale_dma_tiles1, (None, lane)),
                fx.slice(asc_i32_tiles, (None, dst_dw)),
                soffset=src_dw,
            )

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(ACT_SUBS):
            if const_expr(BM == 96):
                lds_dw = fx.Int32((kt % AS_K_SLOTS * ACT_SUBS + sub) * 64)
            else:
                lds_dw = fx.Int32(sub * kAS_per_chunk_dw) + fx.Int32(kt * 64)
            lds_dw = lds_dw + lane_div_16 * fx.Int32(16) + lane_mod_16
            out.append(as_ir_value(_global_i32_load(asc_i32_tiles, lds_dw)))
        return out

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave_n * fx.Int32(4) + lane_div_16


    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
            r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
            fx.copy(
                bq_copy_atom,
                fx.slice(bq_tiles, (None, tile_idx)),
                r,
                soffset=b_load_s_base[j] // fx.Int32(4),
            )
            b_slot[j][half] = r

    def issue_b_load_half(b1, K_C, j, khalf):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048 + khalf * 1024)
        )
        r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
        fx.copy(
            bq_copy_atom,
            fx.slice(bq_tiles, (None, v // fx.Int32(16))),
            r,
            soffset=b_load_s_base[j] // fx.Int32(4),
        )
        b1[j] = r

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        # Interleaved BN64/BN128 have only one scale slot.
        for mw in range_constexpr(B_SCALE_REPS):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            idx = (v + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(mem_1x1, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=s_off // fx.Int32(4),
            )
            bs_slot[mw] = r.load()[0]

    def _mma(ci, opsel_a, opsel_b, a_frag, b_frag, sa, sb):
        # opsel_a/opsel_b select the e8m0 scale byte in the shared 256-K word and are
        # baked into the atom.
        fx.gemm(
            scale_atoms[(opsel_a, opsel_b)],
            ci,
            a_frag,
            b_frag,
            ci,
            scale_a=sa,
            scale_b=sb,
        )


    def kh_chunk_major(slot, slot_b, asc_cur, h, b_half=None, bs_cur=None):
        # One K half, one 16-row chunk of A fragments live at a time.
        assert interleave and BN == 256
        for ci in range_constexpr(ACT):
            a_ci = issue_a_ds_read(slot, khalf=h, chunks=(ci,))
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            for J in range_constexpr(N_REPS):
                b_frag = b[slot_b][J][h] if const_expr(b_half is None) else b_half[J]
                _mma(
                    accm[ci][J],
                    ci % 2 + 2 * h,
                    2 * h + J % 2,
                    a_ci[ci][h],
                    b_frag,
                    asc_cur[ci // 2],
                    b_scale_v[slot_b][J // 2] if const_expr(bs_cur is None) else bs_cur[J // 2],
                )
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)

    def run_k_slice_bpf2(k_tile_base_const):
        # BM96 keeps a current and a next 128-K B half. Prefetch the next
        # half before the current MFMAs; A DMA stays after the last LDS read.
        assert BM == 96 and not inline_quant and k_stages == 1
        b_ring = [[None] * N_REPS for _ in range_constexpr(2)]
        bs_ring = [[None] * B_SCALE_REPS for _ in range_constexpr(2)]
        for s in range_constexpr(2):
            issue_a_scale_tile(s, k_tile_base_const + s)
        issue_a_load_lds(0, k_tile_base_const)
        for J in range_constexpr(N_REPS):
            issue_b_load_half(b_ring[0], k_tile_base_const, J, 0)
        issue_b_scale_load(bs_ring[0], k_tile_base_const)
        for kt in range_constexpr(K_TILES_PER_WAVE):
            gpu.barrier()
            asc_cur = issue_a_scale_ds_read(k_tile_base_const + kt)
            for h in range_constexpr(2):
                next_half = 2 * kt + h + 1
                if const_expr(next_half < 2 * K_TILES_PER_WAVE):
                    next_kt = k_tile_base_const + kt + (1 if h == 1 else 0)
                    for J in range_constexpr(N_REPS):
                        issue_b_load_half(b_ring[1 - h], next_kt, J, 1 - h)
                    if const_expr(h == 1):
                        issue_b_scale_load(bs_ring[(kt + 1) % 2], next_kt)
                rocdl.sched_barrier(0)
                kh_chunk_major(
                    kt % kAStages, 0, asc_cur, h, b_ring[h],
                    bs_cur=bs_ring[kt % 2],
                )
                if const_expr(h == 1 and kt + 1 < K_TILES_PER_WAVE):
                    rocdl.sched_barrier(0)
                    issue_a_load_lds((kt + 1) % kAStages, k_tile_base_const + kt + 1)
                    if const_expr(kt + 2 < K_TILES_PER_WAVE):
                        future = k_tile_base_const + kt + 2
                        issue_a_scale_tile(future % AS_K_SLOTS, future)
                    rocdl.sched_barrier(0)

    def run_k_slice(k_tile_base_const):
        if const_expr(BM == 96):
            run_k_slice_bpf2(k_tile_base_const)
            return
        def global_k_tile(local_tile):
            return k_tile_base_const + local_tile

        b_half = [None] * N_REPS if const_expr(BM == 96) else None

        _relax_prologue = (BM == 128) and not inline_quant
        if const_expr(BM == 96):
            for s in range_constexpr(2):
                issue_a_scale_tile(s, global_k_tile(s))
        else:
            issue_a_scale_load()
        hidden_prefetch = None
        for K_C_LOCAL in range_constexpr(k_stages):
            K_C = global_k_tile(K_C_LOCAL)
            issue_a_load_lds(K_C_LOCAL, K_C)
            if const_expr(NCH > kMChunks):
                issue_a_load_lds_tail(K_C_LOCAL, K_C)
            if const_expr(not _relax_prologue):
                for j in range_constexpr(N_REPS):
                    if const_expr(BM == 96):
                        issue_b_load_half(b_half, K_C, j, 0)
                    else:
                        issue_b_load_j(b[K_C_LOCAL], K_C, j)
            if const_expr(not _relax_prologue):
                issue_b_scale_load(b_scale_v[K_C_LOCAL], K_C)
        if const_expr(_relax_prologue):
            rocdl.sched_barrier(0)
            for K_C_LOCAL in range_constexpr(k_stages):
                K_C = global_k_tile(K_C_LOCAL)
                for j in range_constexpr(N_REPS):
                    issue_b_load_j(b[K_C_LOCAL], K_C, j)
                issue_b_scale_load(b_scale_v[K_C_LOCAL], K_C)

        for OFFSET in range_constexpr(kUnroll):
            K_C_LOCAL = k_stages + OFFSET
            K_C = global_k_tile(K_C_LOCAL)
            read_slot = OFFSET % kAStages
            write_slot = K_C_LOCAL % kAStages
            slot_b = OFFSET % k_stages
            gpu.barrier()
            # K-half major: hold only one half of the A fragments, and refill
            # each B half for tile K_C right after its last use.
            asc_cur = issue_a_scale_ds_read(K_C - k_stages)
            for h in range_constexpr(2):
                kh_chunk_major(read_slot, slot_b, asc_cur, h, b_half)
                if const_expr(h == 1):
                    rocdl.sched_barrier(0)
                    issue_a_load_lds(write_slot, K_C)
                    if const_expr(NCH > kMChunks):
                        issue_a_load_lds_tail(write_slot, K_C)
                    if const_expr(BM == 96 and K_C_LOCAL + 1 < K_TILES_PER_WAVE):
                        future = global_k_tile(K_C_LOCAL + 1)
                        issue_a_scale_tile(future % AS_K_SLOTS, future)
                    rocdl.sched_barrier(0)
                if const_expr(BM == 96):
                    load_tile = K_C - k_stages if h == 0 else K_C
                    load_half = 1 if h == 0 else 0
                    for J in range_constexpr(N_REPS):
                        issue_b_load_half(b_half, load_tile, J, load_half)
                else:
                    bh = [b[slot_b][j][h] for j in range(N_REPS)]
                    for J in range_constexpr(N_REPS):
                        issue_b_load_half(bh, K_C, J, h)
                        b[slot_b][J][h] = bh[J]
                continue
            issue_b_scale_load(b_scale_v[slot_b], K_C)
            continue

        for S in range_constexpr(k_stages):
            kt_local = K_TILES_PER_WAVE - k_stages + S
            kt = global_k_tile(kt_local)
            gpu.barrier()
            asc_cur = issue_a_scale_ds_read(kt)
            for h in range_constexpr(2):
                kh_chunk_major(kt_local % kAStages, kt_local % k_stages, asc_cur, h, b_half)
                if const_expr(BM == 96 and h == 0):
                    for J in range_constexpr(N_REPS):
                        issue_b_load_half(b_half, kt, J, 1)
                continue
            continue

    run_k_slice(0)

    gpu.barrier()

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    # Four dwords of row padding rotate each four-row store group by 16 banks
    # on gfx950, avoiding the 4-way conflict from a BN-aligned row stride.
    ACC_LDS_STRIDE = BN + ACC_LDS_PAD_DW
    ACC_ROWS = BM // epi_splits
    acc_layout = fx.make_layout((ACC_ROWS, BN), (ACC_LDS_STRIDE, 1))
    ACC_GROUP_STRIDE = ACC_ROWS * ACC_LDS_STRIDE

    def acc_idx(group, row, col):
        return group * fx.Int32(ACC_GROUP_STRIDE) + _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_flat_view = fx.make_view(
        fx.recast_iter(fx.Float32, lds_raw_ptr),
        fx.make_layout(k_wave * ACC_GROUP_STRIDE, 1),
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, mem_1x1)

    def acc_store(idx, value):
        r = fx.make_rmem_tensor(mem_1x1, fx.Float32)
        r.store(fx.Vector.from_elements([value], fx.Float32))
        fx.copy(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(mem_1x1, fx.Float32)
        fx.copy(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    def acc_load_sum(row, col):
        value = acc_load(acc_idx(fx.Int32(0), row, col))
        for group in range_constexpr(1, k_wave):
            value = value + acc_load(acc_idx(fx.Int32(group), row, col))
        return value

    tx_i32 = wave_n * fx.Int32(64) + lane
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    CHUNKS_PER_SPLIT = kMChunks // epi_splits

    def store_acc_split(split, chunks=None):
        if const_expr(chunks is None):
            chunks = [split * CHUNKS_PER_SPLIT + j for j in range(CHUNKS_PER_SPLIT)]
        for i_local in range_constexpr(len(chunks)):
            i = chunks[i_local]
            if const_expr(i < ACT):
                row_base = fx.Int32(i_local * 16) + lane_div_16 * fx.Int32(4)
                for J in range_constexpr(N_REPS):
                    # Both weight layouts assign alternating gate/up tiles to the
                    # same accumulators, including BN64's pairs of adjacent waves.
                    is_up = (J % 2) == 1
                    J_local = J // 2
                    col_local = (
                        wave_n * fx.Int32(BN // 8)
                        + fx.Int32(J_local * 16)
                        + lane_mod_16
                    )
                    lds_col = fx.Int32(BN_INT) + col_local if is_up else col_local
                    vec = fx.Vector(fx.memref_load_vec(accm[i][J]))
                    for v in range_constexpr(4):
                        idx = acc_idx(wave_k, row_base + fx.Int32(v), lds_col)
                        acc_store(idx, vec[v])

    MR_PER_SPLIT = EPILOGUE_M_REPS // epi_splits
    scales_per_mr = [None] * (max(EPILOGUE_M_REPS, NCH) + 1)
    if const_expr(NCH % 2 == 1):
        # Odd chunk count: the last 32-row scale group pairs with an all-padding
        # 16-row half, which gets the scale of an all-zero row.
        scales_per_mr[NCH] = _e8m0_from_amax(
            fx.Float32(0.0), max_norm=448.0 if out_dtype == "fp8" else 6.0
        )[0]

    def run_epilogue(split, mrs=None):
        if const_expr(mrs is None):
            mrs = list(range(split * MR_PER_SPLIT, (split + 1) * MR_PER_SPLIT))
        aqout_layout = fx.make_layout((BM, OUT_ROW_BYTES), (OUT_ROW_BYTES, 1))
        aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)

        def store_payload(payload_row, packed0, packed1):
            byte_pos = (
                n_block_idx * fx.Int32(BN_INT)
                + wave_grp * fx.Int32(32)
                + kk * fx.Int32(8)
            )
            store_off = _layout_idx(aqout_layout, payload_row, byte_pos)
            _scalar_store(
                aqout_tiles,
                store_off // fx.Int32(4),
                packed0,
                fx.Int32,
            )
            _scalar_store(
                aqout_tiles,
                (store_off + fx.Int32(4)) // fx.Int32(4),
                packed1,
                fx.Int32,
            )

        if const_expr(enable_bias):
            bias_gate = [None] * 8
            bias_up = [None] * 8
            bias_base = e * fx.Int32(N_OUT)
            for ee in range_constexpr(8):
                out_col = n_block_idx * fx.Int32(BN_INT) + (
                    wave_grp * fx.Int32(32) + fx.Int32(8) * kk + fx.Int32(ee)
                )
                bias_gate[ee] = _global_f32_at(arg_bias, bias_base + out_col)
                bias_up[ee] = _global_f32_at(
                    arg_bias, bias_base + fx.Int32(inter) + out_col
                )

        for mr in mrs:
            row_lds = fx.Int32((mr - mrs[0]) * 16) + m_lane
            row_local = fx.Int32(mr * 16) + m_lane

            gate_vs = [None] * 8
            up_vs = [None] * 8
            for ee in range_constexpr(8 if mr < ACT else 0):
                col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
                gate_col = wave_grp * fx.Int32(32) + col_in_grp
                up_col = fx.Int32(BN_INT) + gate_col
                gate_vs[ee] = acc_load_sum(row_lds, gate_col)
                up_vs[ee] = acc_load_sum(row_lds, up_col)
                if const_expr(enable_bias):
                    gate_vs[ee] = gate_vs[ee] + bias_gate[ee]
                    up_vs[ee] = up_vs[ee] + bias_up[ee]
            if const_expr(mr >= ACT):
                result = [fx.Float32(0.0)] * 8
            else:
                result = _activation_mul_batch(
                    gate_vs,
                    up_vs,
                    act=act,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                    swiglu_limit=swiglu_limit,
                )
            # Match the existing two-stage path: activation is materialized
            # as BF16 before the inter-stage MXFP8 quantization.
            result = [value.to(fx.BFloat16).to(fx.Float32) for value in result]

            local_max = _fabs_f32(result[0])
            for ee in range_constexpr(1, 8):
                local_max = local_max.maximumf(_fabs_f32(result[ee]))
            lm_i = _inline_dpp_quad_amax(local_max.bitcast(fx.Int32))
            local_max = lm_i.bitcast(fx.Float32)

            e8m0, qscale = _e8m0_from_amax(
                local_max, max_norm=448.0 if out_dtype == "fp8" else 6.0
            )
            scales_per_mr[mr] = e8m0

            qscale_raw = as_ir_value(qscale)
            sorted_out_row = m_row + row_local
            i16x2 = fx.Vector.make_type(2, fx.Int16)
            zero16 = fx.full(2, 0, fx.Int16)
            packed0 = rocdl.cvt_scalef32_pk_fp8_f32(
                i16x2,
                zero16,
                as_ir_value(result[0]),
                as_ir_value(result[1]),
                qscale_raw,
                0,
            )
            packed0 = rocdl.cvt_scalef32_pk_fp8_f32(
                i16x2,
                packed0,
                as_ir_value(result[2]),
                as_ir_value(result[3]),
                qscale_raw,
                1,
            )
            packed1 = rocdl.cvt_scalef32_pk_fp8_f32(
                i16x2,
                zero16,
                as_ir_value(result[4]),
                as_ir_value(result[5]),
                qscale_raw,
                0,
            )
            packed1 = rocdl.cvt_scalef32_pk_fp8_f32(
                i16x2,
                packed1,
                as_ir_value(result[6]),
                as_ir_value(result[7]),
                qscale_raw,
                1,
            )
            store_payload(
                sorted_out_row,
                fx.Vector(packed0).bitcast(fx.Int32)[0],
                fx.Vector(packed1).bitcast(fx.Int32)[0],
            )

    def store_scales(subs=None):
        # Generic scale chunks cover 32 sorted rows; native BM16 GEMM2 consumes
        # one padded chunk per M block. Derive the active runtime extent.
        num_scale_chunks = (
            i32_total_m_blocks
            if native_scale_layout
            else (i32_total_m_blocks * fx.Int32(BM) + fx.Int32(31)) // fx.Int32(32)
        )
        ascaleout_layout = fx.make_layout(
            (num_scale_chunks, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
        )
        ascaleout_i8_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int8, 1 << 26)
        ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
        if kk == fx.Int32(0):
            ku = n_block_idx >> fx.Int32(1)
            scale_wave_grp = wave_grp
            ikxdl = n_block_idx & fx.Int32(1)
            if const_expr(num_waves == 2):
                chunk = m_block_idx
                row_half = m_lane >> fx.Int32(4)
                row_in_16 = m_lane & fx.Int32(15)
                dword_off = _layout_idx(
                    ascaleout_layout, chunk, ku, scale_wave_grp, row_in_16
                )
                byte_pos = ikxdl * fx.Int32(2) + row_half
                _scalar_store(
                    ascaleout_i8_tiles,
                    dword_off * fx.Int32(4) + byte_pos,
                    scales_per_mr[0],
                    fx.Int8,
                )
            else:
                for sub in (range((NCH + 1) // 2) if subs is None else subs):
                    chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                    dword_off = _layout_idx(
                        ascaleout_layout, chunk, ku, scale_wave_grp, m_lane
                    )
                    pair_i32 = scales_per_mr[sub * 2 + 0] | (
                        scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                    )
                    addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                    _scalar_store(
                        ascaleout_i16_tiles,
                        addr // fx.Int32(2),
                        pair_i32,
                        fx.Int16,
                    )

    # Stage the accumulators 32 rows (one scale group) at a time so the
    # epilogue LDS stays below the A buffers (3 WGs/CU).
    def _epi32(fn):
        if wave_k == fx.Int32(0):
            fn()

    regions = [list(range(c, min(c + 2, NCH))) for c in range(0, NCH, 2)]
    for r_i in range_constexpr(len(regions)):
        reg = regions[r_i]
        if const_expr(r_i > 0):
            gpu.barrier()
        store_acc_split(0, reg)
        gpu.barrier()

        def _region(reg=reg):
            run_epilogue(0, reg)
            store_scales([reg[0] // 2])

        _epi32(_region)


def compile_gemm1_ep32(
    BM=64,
    use_nt=True,
    inline_quant=False,
    *,
    prefetch_hidden=False,
    D_HIDDEN,
    D_INTER,
    NE,
    BN=256,
    BK=256,
    interleave=True,
    xcd_swizzle=0,
    a_dtype="fp8",
    out_dtype="fp8",
    act="situv2",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    enable_bias=False,
    native_scale_layout=False,
    num_waves=4,
    k_wave=1,
):
    """Compile the dedicated EP32 A8W4 GEMM1 (same arguments and outputs as the
    generic port; only the configuration below is supported)."""
    if BM == 160:
        if BN != 256:
            raise ValueError("M8192 GEMM1 requires BN256")
        if num_waves != 12:
            raise ValueError("M8192 GEMM1 requires twelve waves")
        if (D_HIDDEN, D_INTER, NE, BK, interleave, a_dtype, out_dtype, k_wave,
            inline_quant, prefetch_hidden, xcd_swizzle, native_scale_layout) != (
            3584, 3072, 28, 256, True, "fp8", "fp8", 1, False, False, 0, False):
            raise ValueError("BM160 GEMM1 requires Kimi-K3 EP32 A8W4, BN256 BK256")
        if act != "situv2" or situ_beta <= 0 or situ_linear_beta <= 0:
            raise ValueError("BM160 GEMM1 requires SiTUv2 with positive beta values")
        if enable_bias:
            raise ValueError("M8192 BM160 GEMM1 does not support bias")
        return _compile_gemm1_bm160(BN=BN, use_nt=use_nt,
            situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)
    if (BM, use_nt, inline_quant) not in EP32_G1_VARIANTS or (
        a_dtype,
        out_dtype,
        BN,
        BK,
        interleave,
        num_waves,
        k_wave,
        prefetch_hidden,
        native_scale_layout,
        xcd_swizzle,
    ) != ("fp8", "fp8", 256, 256, True, 4, 1, False, False, 0):
        raise AssertionError(
            "gemm1 ep32 supports BM64/BM96 BN256 BK256 A8 FP8-out interleaved gate/up, "
            "4 waves, k_wave 1, no inline quant / xcd swizzle; got "
            f"BM={BM} use_nt={use_nt} inline_quant={inline_quant} a={a_dtype} "
            f"out={out_dtype} BN={BN} BK={BK} interleave={interleave} "
            f"num_waves={num_waves} k_wave={k_wave} xcd_swizzle={xcd_swizzle}"
        )
    sort_block_m = BM
    if act not in ("silu", "swiglu", "situv2"):
        raise AssertionError(f"act must be 'silu', 'swiglu', or 'situv2', got {act!r}")
    if act == "situv2" and (situ_beta <= 0.0 or situ_linear_beta <= 0.0):
        raise AssertionError("SiTUv2 beta values must be positive")
    if act == "swiglu" and swiglu_limit <= 0.0:
        raise AssertionError("Swiglu limit must be positive")
    KH_TILE = BK
    assert D_HIDDEN % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {D_HIDDEN}"
    assert D_HIDDEN // BK <= 32, "B-scale addressing covers at most 32 K tiles"
    N_OUT = 2 * D_INTER
    assert N_OUT % BN == 0, f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {N_OUT}"
    K_TILES_TOTAL = D_HIDDEN // BK
    assert K_TILES_TOTAL >= 2, f"need >= 2 K tiles, got {K_TILES_TOTAL}"
    NUM_N_BLOCKS = N_OUT // BN

    # Fixed pipeline (see module docstring): one B stage in registers, two A
    # stages in LDS, K-half-major MFMA order, 32-row epilogue staging and
    # padding-aware tiles. BM96 uses six 16-row chunks and three scale groups.
    epi_splits = 1
    k_stages = 1
    skip_pad = True
    khalf_loop = 2
    epi32 = True
    compact = True  # real tiles first in the grid (see the unit ballot in the kernel)
    xcd_chunk = True  # contiguous tile runs per XCD over the compact list
    fast_situ = False  # Preserve the original activation rounding.
    kAStages_ = k_stages + 1
    pipeline_rows = 96 if sort_block_m == 96 else BM + 16
    scale_k_slots = 2 if sort_block_m == 96 else K_TILES_TOTAL
    lds_bytes = max(
        kAStages_ * pipeline_rows * KH_TILE + ((pipeline_rows + 31) // 32) * scale_k_slots * 256,
        32 * (BN + ACC_LDS_PAD_DW) * 4,
    )

    variant_tag = "nt" if use_nt else "cached"
    name_suffix = f"h{D_HIDDEN}_i{D_INTER}_ne{NE}_bm{sort_block_m}_{variant_tag}_{act}{'_fs' if fast_situ else ''}"
    if sort_block_m == 96:
        name_suffix += "_ask2_bpf2_swf8"
    if enable_bias:
        name_suffix += "_bias"

    body_kwargs = dict(
        BM=BM,
        BN=BN,
        BK=BK,
        inline_quant=inline_quant,
        prefetch_hidden=prefetch_hidden,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        enable_bias=enable_bias,
        K=D_HIDDEN,
        N_OUT=N_OUT,
        NE=NE,
        interleave=interleave,
        native_scale_layout=native_scale_layout,
        num_waves=num_waves,
        k_wave=k_wave,
        epi_splits=epi_splits,
        k_stages=k_stages,
        khalf_loop=khalf_loop,
        epi32=epi32,
        fast_situ=fast_situ,
    )

    block_threads = num_waves * k_wave * 64

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(
        name=f"gemm1_ep32_a8w4_{name_suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def gemm1_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        arg_bias: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % 64
        wave = rocdl.readfirstlane(T.i32, tx_i32 // 64)
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(NUM_N_BLOCKS)
        one, zero = fx.Int32(1), fx.Int32(0)
        pad_tok = _global_i32_at(arg_cumsum, one)
        # Workgroups past every M block's tiles (the host grid is an upper
        # bound, ~20x the real tiles) exit before any per-block loads.
        if bx_i32 < bound:
            # Units = M blocks not folded into the previous block's tile, found
            # with a wave ballot over the first 128 blocks (later blocks are
            # appended unchanged). Workgroups index units compactly so the real
            # tiles take the first grid slots; with folded blocks interleaved
            # the dispatcher refills their freed slots and stacks up to 4 real
            # tiles on some CUs while others get one.
            u_masks = []
            for c in range_constexpr(2):
                b = lane + fx.Int32(c * 64)
                b_ok = b < total_m_blocks
                bc = b_ok.select(b, zero)
                bp = (bc > zero).select(bc - one, bc)
                t16 = _global_i32_at(arg_mind, bc * fx.Int32(BM) + fx.Int32(16))
                folded = (bc > zero).select(
                    (_global_i32_at(arg_eids, bc) == _global_i32_at(arg_eids, bp)).select(
                        (t16 == pad_tok).select(one, zero), zero
                    ),
                    zero,
                )
                is_unit = b_ok.select((folded == zero).select(one, zero), zero)
                u_masks.append(fx.Int64(rocdl.ballot(T.i64, as_ir_value(is_unit == one))))
            u_cnt0 = fx.Int32(fx.math.ctpop(u_masks[0]))
            u_cnt1 = fx.Int32(fx.math.ctpop(u_masks[1]))
            extra = (total_m_blocks > fx.Int32(128)).select(total_m_blocks - fx.Int32(128), zero)
            n_units = u_cnt0 + u_cnt1 + extra
            gate = bx_i32 < n_units * fx.Int32(NUM_N_BLOCKS)

            NXCD = 8
            xq = _udiv(bound, NXCD)
            xr = _umod(bound, NXCD)


            if gate:
                tid = bx_i32
                # Workgroups go round-robin over the 8 XCDs; give each XCD a
                # contiguous run of the (unit, N block) list so a unit's
                # gathered A rows are read into one or two XCD L2s, not all 8.
                n_tiles = n_units * fx.Int32(NUM_N_BLOCKS)
                cq = _udiv(n_tiles, 8)
                cr = _umod(n_tiles, 8)
                xc = _umod(bx_i32, 8)
                tid = xc * cq + (xc < cr).select(xc, cr) + _udiv(bx_i32, 8)
                u = tid // fx.Int32(NUM_N_BLOCKS)
                n_blk = tid - u * fx.Int32(NUM_N_BLOCKS)
                in0 = u < u_cnt0
                k = in0.select(u, u - u_cnt0)
                msk = in0.select(u_masks[0], u_masks[1])
                lane64 = fx.Int64(lane)
                below = msk & ((fx.Int64(1) << lane64) - fx.Int64(1))
                bit = (msk >> lane64) & fx.Int64(1)
                rank = fx.Int32(fx.math.ctpop(below))
                hit = (bit == fx.Int64(1)).select((rank == k).select(one, zero), zero)
                sel = fx.Int64(rocdl.ballot(T.i64, as_ir_value(hit == one)))
                pos = fx.Int32(fx.math.cttz(sel))
                m_u = (u < u_cnt0 + u_cnt1).select(
                    in0.select(pos, pos + fx.Int32(64)), fx.Int32(128) + u - u_cnt0 - u_cnt1
                )
                m_u = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(m_u)))
                tile = m_u * fx.Int32(NUM_N_BLOCKS) + n_blk
                # Per M block: absorb a <=16-row tail block of the same expert
                # into the preceding full block's tile (80 rows, one weight
                # stream); blocks with <=16 real rows otherwise take the light
                # path. Padding sits at the end of each expert run.
                m_blk = tile // fx.Int32(NUM_N_BLOCKS)
                m_prev = (m_blk > fx.Int32(0)).select(m_blk - one, m_blk)
                has_next = m_blk + one < total_m_blocks
                m_next = has_next.select(m_blk + one, m_blk)
                e_cur = _global_i32_at(arg_eids, m_blk)
                e_prev = _global_i32_at(arg_eids, m_prev)
                e_next = _global_i32_at(arg_eids, m_next)
                tok16 = _global_i32_at(arg_mind, m_blk * fx.Int32(BM) + fx.Int32(16))
                tokn16 = _global_i32_at(arg_mind, m_next * fx.Int32(BM) + fx.Int32(16))
                light = (tok16 == pad_tok).select(one, fx.Int32(0))
                prev_same = (m_blk > fx.Int32(0)).select((e_prev == e_cur).select(one, fx.Int32(0)), fx.Int32(0))
                next_light = has_next.select((e_next == e_cur).select((tokn16 == pad_tok).select(one, fx.Int32(0)), fx.Int32(0)), fx.Int32(0))
                mode = rocdl.readfirstlane(
                    T.i32,
                    as_ir_value((light * prev_same == one).select(fx.Int32(3), (next_light * (one - light) == one).select(fx.Int32(2), light))),
                )
                mode = fx.Int32(mode)
                if mode == fx.Int32(2):
                    _gemm1_body(
                        lds_raw_ptr,
                        arg_aq,
                        arg_ascale,
                        arg_bq,
                        arg_bscale,
                        arg_eids,
                        arg_mind,
                        arg_aqout,
                        arg_ascaleout,
                        arg_hidden,
                        arg_bias,
                        tile,
                        lane,
                        wave,
                        use_nt,
                        i32_ntok,
                        total_m_blocks,
                        **body_kwargs,
                        active_chunks=5,
                    )
                elif mode == fx.Int32(1):
                    _gemm1_body(
                        lds_raw_ptr,
                        arg_aq,
                        arg_ascale,
                        arg_bq,
                        arg_bscale,
                        arg_eids,
                        arg_mind,
                        arg_aqout,
                        arg_ascaleout,
                        arg_hidden,
                        arg_bias,
                        tile,
                        lane,
                        wave,
                        use_nt,
                        i32_ntok,
                        total_m_blocks,
                        **body_kwargs,
                        active_chunks=1,
                    )
                elif mode == fx.Int32(0):
                    _gemm1_body(
                        lds_raw_ptr,
                        arg_aq,
                        arg_ascale,
                        arg_bq,
                        arg_bscale,
                        arg_eids,
                        arg_mind,
                        arg_aqout,
                        arg_ascaleout,
                        arg_hidden,
                        arg_bias,
                        tile,
                        lane,
                        wave,
                        use_nt,
                        i32_ntok,
                        total_m_blocks,
                        **body_kwargs,
                        active_chunks=None,
                    )

    @flyc.kernel(
        name=f"gemm1_ep32_a8w4_{name_suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def gemm1_kernel_block(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        arg_bias: fx.Int64,
    ):
        # BM96 sort blocks: one full tile per block; the (block, N) list
        # is cut into one contiguous run per XCD (as in the BM64 kernel).
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.thread_idx.x
        bx_i32 = fx.block_idx.x
        lane = tx_i32 % 64
        wave = rocdl.readfirstlane(T.i32, tx_i32 // 64)
        total_m_blocks = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(sort_block_m)
        bound = total_m_blocks * fx.Int32(NUM_N_BLOCKS)
        if bx_i32 < bound:
            cq = _udiv(bound, 8)
            cr = _umod(bound, 8)
            xc = _umod(bx_i32, 8)
            tile = xc * cq + (xc < cr).select(xc, cr) + _udiv(bx_i32, 8)
            _gemm1_body(
                lds_raw_ptr,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                arg_bias,
                tile,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                **body_kwargs,
                active_chunks=sort_block_m // 16,
            )

    @flyc.jit
    def launch_gemm1(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        arg_bias: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        (gemm1_kernel_block if const_expr(sort_block_m == 96) else gemm1_kernel)(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
            arg_bias,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    if k_stages == 1:
        launch_gemm1.compile_hints = {
            "llvm_options": {"amdgpu-sched-strategy": "iterative-minreg"},
        }
    return launch_gemm1


def _dpp_row16_amax(a32):
    # Quad xor-1/xor-2, then row_half_mirror and row_mirror span one 16-lane row.
    return _dpp_umax_step(_dpp_umax_step(_inline_dpp_quad_amax(a32), 0x141), 0x140)


def _situv2_mul_batch(gates, ups, beta, linear_beta):
    """SiTUv2 gate * up with c * tanh(x / c) = c - 2c / (1 + exp(2x / c)).

    Equal to the shared helper up to rounding, without the |x| and sign select;
    +-inf exponentials give the exact +-c limits.
    """
    beta, linear_beta = float(beta), float(linear_beta)

    def rcp(x):
        return fx.Float32(rocdl.rcp(T.f32, _raw(x)))

    one = fx.Float32(1.0)
    out = []
    for g, u in zip(gates, ups):
        eg = fx.exp2(g * fx.Float32(2.0 * LOG2E / beta), fastmath="afn")
        es = fx.exp2(g * fx.Float32(-LOG2E), fastmath="afn")
        eu = fx.exp2(u * fx.Float32(2.0 * LOG2E / linear_beta), fastmath="afn")
        tg = fx.Float32(beta) - rcp(eg + one) * fx.Float32(2.0 * beta)
        tu = fx.Float32(linear_beta) - rcp(eu + one) * fx.Float32(2.0 * linear_beta)
        out.append(tg * rcp(es + one) * tu)
    return out


def _compile_gemm1_bm160(*, BN=256, use_nt=True, situ_beta=4.0, situ_linear_beta=25.0):
    if BN != 256:
        raise ValueError("M8192 W12 GEMM1 requires BN256")
    BM, BK, H, INTER, NE = 160, 128, 3584, 3072, 28
    N_OUT, N_BLOCKS, M_CHUNKS, N_REPS = 2 * INTER, 2 * INTER // BN, 4, BN // 64
    A_STAGES = 2
    B_STAGES = 3
    GRID = 256
    AS_DW = kas_per_chunk_dw_for(H)
    OUT_AS_DW = kas_per_chunk_dw_for(INTER)
    BS_DW = kbs_per_expert_dw_for(N_OUT, H)
    BS_N_DW = kbs_stride_n0_dw_for(H)
    A_SLOT, B_SLOT, S_SLOT = BM * BK, BN * BK // 2, (5 + BN // 32) * 256
    B_BASE, S_BASE = A_STAGES * A_SLOT, A_STAGES * A_SLOT + B_STAGES * B_SLOT
    LDS_BYTES = S_BASE + 3 * S_SLOT
    assert LDS_BYTES <= 160 * 1024
    name = f"gemm1_ep8192_a8w4_bm160_bn{BN}_w12_m64x2_m32_as{A_STAGES}_bs{B_STAGES}_ak128_{'nt' if use_nt else 'cached'}_situv2_runtimek_regepi3_earlypro_fastsitu"

    @fx.struct
    class Storage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[768, 1, 1])
    def gemm1(
        aq: fx.Int64,
        asc: fx.Int64,
        bq: fx.Int64,
        bsc: fx.Int64,
        eids: fx.Int64,
        counts: fx.Int64,
        token_rows: fx.Int64,
        tokens: fx.Int32,
        out: fx.Int64,
        out_scale: fx.Int64,
        hidden: fx.Int64,
        bias: fx.Int64,
    ):
        tx = fx.thread_idx.x
        lane = tx % fx.Int32(64)
        # Uniform wave IDs preserve scalar resource-address arithmetic.
        wave = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(tx // fx.Int32(64))))
        wn, wm = wave % fx.Int32(4), wave // fx.Int32(4)
        # Three M groups cover 64 + 64 + 32 complete rows.
        m_chunk_base = wm * fx.Int32(4)
        loader = wm == fx.Int32(0)
        ld16, lm16 = lane // fx.Int32(16), lane % fx.Int32(16)
        ld8, lm8 = lane // fx.Int32(8), lane % fx.Int32(8)
        lds_ptr = fx.SharedAllocator().allocate(Storage).peek().raw.ptr
        lds = fx.Int32(fx.ptrtoint(lds_ptr))
        scopes = lds_slot_scopes(A_STAGES + B_STAGES + 3, "ep8192_g1")
        ac, bc, sc = (
            scopes[:A_STAGES],
            scopes[A_STAGES : A_STAGES + B_STAGES],
            scopes[A_STAGES + B_STAGES :],
        )
        m_blocks = _global_i32_at(counts, fx.Int32(0)) // fx.Int32(BM)
        bound = m_blocks * fx.Int32(N_BLOCKS)
        ares = _buffer_rsrc(aq, fx.Int64(tokens) * H)
        asres = _buffer_rsrc(asc, fx.Int64(m_blocks) * 5 * AS_DW * 4)
        bres = _buffer_rsrc(bq, bq_bytes_for(NE, N_OUT, H))
        bsres = _buffer_rsrc(bsc, bscale_bytes_for(NE, N_OUT, H))
        mma = _scale_mma_atoms("fp8")
        mem4 = fx.make_layout(4, 1)
        # Constant GEP displacements fold into DS offsets instead of
        # keeping one VGPR address for each K-slot and M chunk.
        a_mask = lds_swizzle_mask_f8(lm16, BK)
        a_row_base = (m_chunk_base * 16 + lm16) * BK
        a_ptr_lo = _lds_ptr3(lds, a_row_base + ((ld16 * 16) ^ a_mask))
        a_ptr_hi = _lds_ptr3(lds, a_row_base + ((ld16 * 16 + 64) ^ a_mask))
        b_ptr = _lds_ptr3(lds, wn * (BN * 16) + ld16 * 256 + lm16 * 16)
        s_ptr0 = _lds_ptr3(lds, (m_chunk_base // 2) * 256 + (ld16 * 16 + lm16) * 4)
        s_ptr1 = _lds_ptr3(
            lds, ((m_chunk_base + 1) // 2) * 256 + (ld16 * 16 + lm16) * 4
        )
        # Byte buffer views keep each epilogue store to one VGPR offset.
        qbuf, sbuf = [
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        global_typed_ptr(addr, T.i8, align=1),
                        fx.make_layout((1, 1), (1, 1)),
                    )
                ),
                max_size=True,
            )
            for addr in (out, out_scale)
        ]
        byte_copy = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)

        def store_byte(buf, offset, value):
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int8)
            r.store(fx.Vector.from_elements([value], fx.Int8))
            fx.copy(byte_copy, r, buf[None, offset])

        def tile_setup(pid):
            # Returns [ni, mrow, bbase x4, scale_base, a_voff x5] for one tile.
            xq, xr, xc = _udiv(bound, 8), _umod(bound, 8), _umod(pid, 8)
            tile = xc * xq + fx.min(xc, xr) + _udiv(pid, 8)
            mi, ni = tile // fx.Int32(N_BLOCKS), tile % fx.Int32(N_BLOCKS)
            mrow = mi * fx.Int32(BM)
            expert = fx.Int32(
                rocdl.readfirstlane(T.i32, as_ir_value(_global_i32_at(eids, mi)))
            )
            bbase = []
            for j in range_constexpr(N_REPS):
                col = ni * BN + wn * (BN // 4) + j * 16
                # fx.Int32 has no scalar/uniform annotation for buffer bases.
                bbase.append(
                    fx.Int32(
                        rocdl.readfirstlane(
                            T.i32, as_ir_value((expert * N_OUT + col) * (H // 2))
                        )
                    )
                )
            scale_base = (
                expert * BS_DW + (ni * (BN // 32) + wn * (BN // 128)) * BS_N_DW
            ) * 4
            scale_base = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(scale_base)))
            # Reuse route addresses for every K step. Loading row IDs inside
            # load_a forces vmcnt(0) waits between the future A/B DMAs.
            a_voff = []
            for sub in range_constexpr(5):
                row = wn * 40 + sub * 8
                actual = _global_i32_at(token_rows, mrow + row + ld8)
                mask = lds_swizzle_mask_f8(row + ld8, BK)
                a_voff.append(actual * H + ((lm8 * 16) ^ mask))
            return [ni, mrow] + bbase + [scale_base] + a_voff

        def load_a(t, slot, step):
            if loader:
                for sub in range_constexpr(5):
                    row = wn * 40 + sub * 8
                    # Scoped DMA has no equivalent fx.copy alias-scope API.
                    rocdl.raw_ptr_buffer_load_lds(
                        ares,
                        _lds_ptr3(lds, fx.Int32(slot * A_SLOT) + row * BK),
                        fx.Int32(16),
                        t[7 + sub],
                        step * fx.Int32(BK),
                        fx.Int32(0),
                        fx.Int32(0),
                        alias_scopes=ac[slot][0],
                        noalias_scopes=ac[slot][1],
                    )

        def load_b(t, slot, step):
            if loader:
                for j in range_constexpr(N_REPS):
                    dst = fx.Int32(B_BASE + slot * B_SLOT + j * 1024) + wn * (BN * 16)
                    rocdl.raw_ptr_buffer_load_lds(
                        bres,
                        _lds_ptr3(lds, dst),
                        fx.Int32(16),
                        ld16 * 256 + lm16 * 16 + step * fx.Int32(1024),
                        t[2 + j],
                        fx.Int32(0),
                        2 if use_nt else 0,
                        alias_scopes=bc[slot][0],
                        noalias_scopes=bc[slot][1],
                    )

        def load_s(t, slot, word):
            if loader:
                for group in range_constexpr(2):
                    sub = wn + group * 4
                    if sub < fx.Int32(5):
                        src = (t[1] // 32 + sub) * (AS_DW * 4)
                        dst = fx.Int32(S_BASE + slot * S_SLOT) + sub * 256
                        rocdl.raw_ptr_buffer_load_lds(
                            asres,
                            _lds_ptr3(lds, dst),
                            fx.Int32(4),
                            lane * 4,
                            src + word * fx.Int32(256),
                            fx.Int32(0),
                            fx.Int32(0),
                            alias_scopes=sc[slot][0],
                            noalias_scopes=sc[slot][1],
                        )

        def load_bs(t, slot, word):
            if loader:
                for pair in range_constexpr(2):
                    dst = (
                        fx.Int32(S_BASE + slot * S_SLOT + 5 * 256 + pair * 256)
                        + wn * 512
                    )
                    src = t[6] + pair * BS_N_DW * 4
                    rocdl.raw_ptr_buffer_load_lds(
                        bsres,
                        _lds_ptr3(lds, dst),
                        fx.Int32(4),
                        lane * 4,
                        src + word * fx.Int32(256),
                        fx.Int32(0),
                        fx.Int32(0),
                        alias_scopes=sc[slot][0],
                        noalias_scopes=sc[slot][1],
                    )

        def read_a(slot, ci):
            return _bm160_a_fragment(a_ptr_lo, a_ptr_hi, slot * A_SLOT + ci * 16 * BK, ac[slot])

        def prologue(t):
            for p in range_constexpr(A_STAGES - 1):
                load_a(t, p, fx.Int32(p))
            for p in range_constexpr(B_STAGES - 1):
                load_b(t, p, fx.Int32(p))
            load_s(t, 0, fx.Int32(0))
            load_s(t, 1, fx.Int32(1))
            load_bs(t, 0, fx.Int32(0))
            load_bs(t, 1, fx.Int32(1))

        # Each tile's setup and prologue DMAs are issued before the previous
        # tile's epilogue, whose VALU work then hides the pipeline refill.
        wg = fx.Int32(fx.block_idx.x)
        first = tile_setup(fx.min(wg, fx.max(bound - fx.Int32(1), fx.Int32(0))))
        if wg < bound:
            prologue(first)
        for iv, cur in range(wg, bound, fx.Int32(gpu.grid_dim.x), init=first):
            ni, mrow = cur[0], cur[1]
            acc = [
                [fx.make_rmem_tensor(mem4, fx.Float32) for _ in range_constexpr(N_REPS)]
                for _ in range_constexpr(M_CHUNKS)
            ]
            for i in range_constexpr(M_CHUNKS):
                for j in range_constexpr(N_REPS):
                    acc[i][j].store(fx.full(4, 0.0, fx.Float32))
            _bm160_mma_loop(acc, H, wm,
                (B_BASE, B_SLOT, S_BASE, S_SLOT),
                (lds, b_ptr, s_ptr0, s_ptr1, wn, ld16, lm16),
                (bc, sc), mma, mem4, (lambda slot, step, cur=cur: load_a(cur, slot, step), lambda slot, step, cur=cur: load_b(cur, slot, step),
                 lambda slot, word, cur=cur: load_s(cur, slot, word), lambda slot, word, cur=cur: load_bs(cur, slot, word), read_a))

            # Every wave has finished reading this tile's LDS slots.
            gpu.barrier()
            nxt_pid = fx.Int32(iv) + fx.Int32(gpu.grid_dim.x)
            nxt = tile_setup(fx.min(nxt_pid, bound - fx.Int32(1)))
            if nxt_pid < bound:
                prologue(nxt)

            epi_lane, epi_wave = _bm160_epilogue_ids(tx)
            epi_wn, epi_wm = epi_wave % fx.Int32(4), epi_wave // fx.Int32(4)
            epi_ld16, epi_lm16 = epi_lane // fx.Int32(16), epi_lane % fx.Int32(16)
            z = fx.full(2, 0, fx.Int16)
            # Row mrow + 64 * wm + 16 * ci + 4 * ld16 + v, column
            # 128 * ni + 32 * wn + 16 * half + lm16 of the FP8 output.
            q_base = (
                mrow + epi_wm * fx.Int32(64) + epi_ld16 * fx.Int32(4)
            ) * fx.Int32(INTER) + ni * fx.Int32(BN // 2) + epi_wn * fx.Int32(32) + epi_lm16
            # The shuffled E8M0 offset of the same row and 32-column group,
            # with the non-negative divisions written as shifts and masks.
            s_base = (
                (mrow.shrui(fx.Int32(5)) + epi_wm * fx.Int32(2))
                * fx.Int32(OUT_AS_DW * 4)
                + ni.shrui(fx.Int32(1)) * fx.Int32(256)
                + epi_wn * fx.Int32(64)
                + epi_ld16 * fx.Int32(16)
                + (ni & fx.Int32(1)) * fx.Int32(2)
            )
            # The interleaved weights put gate (even j) and up (odd j) of one
            # output column in the same lane, and one 16-lane row holds a full
            # 32-column E8M0 group, so activation and quantization stay in
            # registers. All 160 rows, including padding, are written.
            for ci in range_constexpr(4):
                if (epi_wm < fx.Int32(2)) | (fx.Int32(ci) < fx.Int32(2)):
                    gates, ups = [], []
                    for half in range_constexpr(2):
                        g = fx.Vector(acc[ci][2 * half].load())
                        u = fx.Vector(acc[ci][2 * half + 1].load())
                        for v in range_constexpr(4):
                            gates.append(g[v])
                            ups.append(u[v])
                    result = _situv2_mul_batch(gates, ups, situ_beta, situ_linear_beta)
                    result = [x.to(fx.BFloat16).to(fx.Float32) for x in result]
                    for v in range_constexpr(4):
                        lo, hi = result[v], result[4 + v]
                        amax = _fabs_f32(lo).maximumf(_fabs_f32(hi))
                        amax = _dpp_row16_amax(amax.bitcast(fx.Int32)).bitcast(
                            fx.Float32
                        )
                        code, scale = _e8m0_from_amax(amax, max_norm=448.0)
                        p = rocdl.cvt_scalef32_pk_fp8_f32(
                            fx.Vector.make_type(2, fx.Int16),
                            z,
                            as_ir_value(lo),
                            as_ir_value(hi),
                            as_ir_value(scale),
                            0,
                        )
                        word = fx.Vector(p).bitcast(fx.Int32)[0]
                        q_off = q_base + fx.Int32((ci * 16 + v) * INTER)
                        store_byte(qbuf, q_off, word)
                        store_byte(qbuf, q_off + fx.Int32(16), word.shrui(fx.Int32(8)))
                        if epi_lm16 == fx.Int32(0):
                            s_off = s_base + fx.Int32(
                                (ci // 2) * OUT_AS_DW * 4 + v * 4 + ci % 2
                            )
                            store_byte(sbuf, s_off, code)

            results = yield nxt

    @flyc.jit
    def launch(
        aq: fx.Int64,
        asc: fx.Int64,
        bq: fx.Int64,
        bsc: fx.Int64,
        eids: fx.Int64,
        counts: fx.Int64,
        rows: fx.Int64,
        tokens: fx.Int32,
        grid: fx.Int32,
        out: fx.Int64,
        out_scale: fx.Int64,
        hidden: fx.Int64,
        bias: fx.Int64,
        stream: fx.Stream,
    ):
        gemm1(
            aq, asc, bq, bsc, eids, counts, rows, tokens, out, out_scale, hidden, bias
        ).launch(grid=(GRID, 1, 1), block=(768, 1, 1), stream=stream)

    launch.compile_hints = {
        "waves_per_eu": 3,
        "llvm_options": {"amdgpu-sched-strategy": "iterative-minreg"},
    }
    return launch
