# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from . import dpp_utils
from .mxfp4_gemm_common import (
    _activation_mul_batch,
    _e8m0_from_amax,
    _fabs_f32,
    _global_f32_at,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _global_i32_load,
    _global_scalar_tiles,
    _inline_dpp_quad_amax,
    _inline_e8m0,
    _layout_idx,
    _lds_swizzle_mask,
    _pkmax_u16,
    _scalar_store,
    _scale_mma_atoms,
    _udiv,
    _umax_i32,
    _umod,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kmchunks_for,
    kStages,
    kunroll_for,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def n_out_for(inter):
    return 2 * inter


def k_g2_half_for(inter):
    return inter // 2


def out_as_per_chunk_dw_for(inter):
    scale_cols = inter // 32
    # Match fused_dynamic_mx_quant_moe_sort: scale-N is padded to eight bytes.
    scale_cols_padded = ((scale_cols + 7) // 8) * 8
    return scale_cols_padded * 8


def gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, BN=256):
    num_n_blocks = num_n_blocks_for(n_out_for(INTER), BN)
    if BM == 128:
        max_m_blocks = (n_tokens * TOPK + NE * (BM - 1) + BM - 1) // BM
    else:
        active = min(n_tokens * TOPK, NE)
        max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * num_n_blocks


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
):
    # A-code tile bytes/row: fp4 packs 2 codes/byte (BK/2); fp8 is 1 B/elem (BK).
    KH_TILE = BK if a_dtype == "fp8" else BK // 2
    K_HALF = k_half_for(K)
    # A row bytes: fp4 = K/2, fp8 = K (B is always mxfp4 -> keeps K_HALF).
    A_ROW_BYTES = K if a_dtype == "fp8" else K_HALF
    K_TILES_TOTAL = k_tiles_total_for(K, BK)
    kUnroll = kunroll_for(K, BK)
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    ASCALE_CHUNK_BYTES = kAS_per_chunk_dw * 4
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = num_n_blocks_for(N_OUT, BN)
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(inter)
    OUT_ROW_BYTES = inter if out_dtype == "fp8" else k_g2_half_for(inter)
    kAStages, kSubBlocks, kMChunks, _ = _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL)

    BN_INT = BN // 2
    b_aux = 2 if use_nt else 0
    M_REPS = BM // 16
    N_REPS = BN // 64

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
    bq_reg_lay = fx.make_layout(4, 1)

    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    bscale_reg_lay = fx.make_layout(1, 1)

    # aq/ascale: global->LDS async DMA (no register fragment), via BufferCopyLDS.
    aq_buf = _global_i32_buffer_view(arg_aq, aq_num_records)
    aq_dma_tiles4 = fx.logical_divide(aq_buf, fx.make_layout(4, 1))
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    aq_reg_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
    aq_reg_lay = fx.make_layout(4, 1)

    ascale_buf = _global_i32_buffer_view(arg_ascale, ascale_num)
    ascale_dma_tiles4 = fx.logical_divide(ascale_buf, fx.make_layout(4, 1))
    ascale_dma_tiles1 = fx.logical_divide(ascale_buf, fx.make_layout(1, 1))
    ascale_dma_atom16 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    ascale_dma_atom4 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), fx.Int32)

    if const_expr(inline_quant):
        hidden_num = fx.Int64(i32_ntok * fx.Int32(K * 2))
        hidden_tiles = _global_i32_buffer_tiles(arg_hidden, hidden_num, 4)
        hidden_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        hidden_reg_lay = fx.make_layout(4, 1)

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in the
    # epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    cached_row_inline = None
    if const_expr(inline_quant):
        rcls = wave * fx.Int32(4) + lane_div_16
        cached_row_inline = _global_i32_at(arg_mind, m_row + rcls)
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            actual_row = _global_i32_at(arg_mind, idx)
            if const_expr(a_dtype == "fp8" and BM >= 64):
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
            else:
                cached_actual_row.append(actual_row)

    # -- b_load_s_base[j], readfirstlane'd uniform per wave --------------------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(N_REPS):
        if const_expr(interleave):
            col = (
                n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
            )
        else:
            tile_il = (
                n_block_idx * fx.Int32(BN // 16) + wave * fx.Int32(N_REPS) + fx.Int32(j)
            )
            g = tile_il & fx.Int32(1)
            n0 = tile_il >> fx.Int32(1)
            col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi --------------------------------------------------
    if const_expr(interleave):
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        if const_expr(BN == 128):
            np_gate = n_block_idx * fx.Int32(2) + wave // fx.Int32(2)
        else:
            np_gate = n_block_idx * fx.Int32(BN // 64) + wave
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    # f32 accumulator fragments (one i32[4]->f32[4] per (mchunk, J)); seeded to 0
    # so the first K-iter's fx.gemm computes A*B+0 (matches the raw init path).
    scale_atoms = _scale_mma_atoms(a_dtype)
    accm = [
        [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_REPS)]
        for _ in range(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(N_REPS):
            accm[i][J].store(fx.Vector.filled(4, 0.0, fx.Float32))
    b = [[[None, None] for _ in range(N_REPS)] for _ in range(kStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(kAStages * BM * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))
    s_aq_i32x1_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(1, 1))
    i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    i32x4_reg_lay = fx.make_layout(4, 1)

    def issue_a_load_lds(slot, kt):
        # A wave-wide BufferCopyLDS128b writes 1024 contiguous bytes. That is
        # eight FP4 rows or four FP8 rows. Larger FP8 tiles use two four-row
        # passes; BM32 keeps register staging, which is faster at low occupancy.
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            if const_expr(a_dtype == "fp8" and BM >= 64):
                for row_group in range_constexpr(2):
                    dst_row_base = lds_row + fx.Int32(row_group * 4)
                    mask = _lds_swizzle_mask(dst_row_base + lane_div_16)
                    voffset = ((lane_mod_16 * fx.Int32(16)) ^ mask) + cached_actual_row[
                        sub
                    ][row_group] * fx.Int32(A_ROW_BYTES)
                    off = fx.Int32(slot * (BM * KH_TILE)) + dst_row_base * fx.Int32(
                        KH_TILE
                    )
                    fx.copy(
                        aq_dma_atom,
                        fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                        fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                        soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
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
                        fx.Int32(slot * (BM * KH_TILE))
                        + dst_row * fx.Int32(KH_TILE)
                        + half_off
                        + ((lane_mod_8 * fx.Int32(16)) ^ mask)
                    )
                    r = fx.make_rmem_tensor(aq_reg_lay, fx.Int32)
                    fx.copy(
                        aq_reg_atom,
                        fx.slice(aq_dma_tiles4, (None, src_off // fx.Int32(16))),
                        r,
                        soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                    )
                    fx.copy_atom_call(
                        i32x4_copy_atom,
                        r,
                        fx.slice(s_aq_i32x4_tiles, (None, dst_off // fx.Int32(16))),
                    )
            else:
                mask = _lds_swizzle_mask(lds_row + lane_div_8)
                voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                    sub
                ] * fx.Int32(A_ROW_BYTES)
                off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
                fx.copy(
                    aq_dma_atom,
                    fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                    fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                    soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                )

    def _lds_i32x4_frag(tile_idx):
        # ds_read_b128 straight into an i32[4] register fragment (kept as a tensor
        # so it can feed fx.gemm directly).
        r = fx.make_rmem_tensor(i32x4_reg_lay, fx.Int32)
        fx.copy_atom_call(
            i32x4_copy_atom, fx.slice(s_aq_i32x4_tiles, (None, tile_idx)), r
        )
        return r

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            if const_expr(a_dtype == "fp8"):
                # fp8 128-K operand (v8i32) = two 16 B halves 64 B apart in the row
                # (f8f6f4 ABI), packed lo++hi. k selects the 128-K half (upper = +128 B).
                kbase = fx.Int32(k * 128)
                lo_col = (lane_div_16 * fx.Int32(16) + kbase) ^ mask
                hi_col = (lane_div_16 * fx.Int32(16) + kbase + fx.Int32(64)) ^ mask
                for i in range_constexpr(kMChunks):
                    lds_row = lane_mod_16 + fx.Int32(i * 16)
                    boff = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
                    lo = fx.Vector(
                        fx.memref_load_vec(
                            _lds_i32x4_frag((boff + lo_col) // fx.Int32(16))
                        )
                    )
                    hi = fx.Vector(
                        fx.memref_load_vec(
                            _lds_i32x4_frag((boff + hi_col) // fx.Int32(16))
                        )
                    )
                    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Int32)
                    t.store(lo.shuffle(hi, list(range(8))))
                    a[i][k] = t
            else:
                lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
                for i in range_constexpr(kMChunks):
                    lds_row = lane_mod_16 + fx.Int32(i * 16)
                    off = (
                        fx.Int32(slot * (BM * KH_TILE))
                        + lds_row * fx.Int32(KH_TILE)
                        + lds_col
                    )
                    a[i][k] = _lds_i32x4_frag(off // fx.Int32(16))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        thread_linear = wave * fx.Int32(64) + lane
        for sub in range_constexpr(kSubBlocks):
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
                            (lds_sub + fx.Int32(byte_off) + wave * fx.Int32(1024))
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
                            (lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256))
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
                                (lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256))
                                // fx.Int32(4),
                            ),
                        ),
                        soffset=s_off // fx.Int32(4),
                    )

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(lds_raw_ptr, kAStages * BM * KH_TILE)),
        fx.make_layout(kSubBlocks * K_TILES_TOTAL * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(1, 1))
    asc_i32x4_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(4, 1))

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(as_ir_value(_global_i32_load(asc_i32_tiles, lds_dw)))
        return out

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave * fx.Int32(4) + lane_div_16

    def inline_quant_load_kt(B128_IDX, kt, row_token):
        v_voff = (
            row_token * fx.Int32(K * 2)
            + lane_shr2_and3 * fx.Int32(64)
            + lib * fx.Int32(16)
        )
        s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * (BK * 2) + B128_IDX * 256))
        r = fx.make_rmem_tensor(hidden_reg_lay, fx.Int32)
        fx.copy(
            hidden_copy_atom,
            fx.slice(hidden_tiles, (None, v_voff // fx.Int32(16))),
            r,
            soffset=s_soff // fx.Int32(4),
        )
        return r.load()

    def _bf16x2(dw):
        return as_ir_value(fx.Vector.from_elements([dw], fx.Int32).bitcast(fx.BFloat16))

    def _iq_block_amax(h_dw_i):
        # Per-32 amax over the 8 bf16 in this lane's block (4 bf16x2 dwords).
        hm = [h_dw_i[j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
        m01 = _pkmax_u16(hm[0], hm[1])
        m23 = _pkmax_u16(hm[2], hm[3])
        m0123 = _pkmax_u16(m01, m23)
        lo = m0123 & fx.Int32(0xFFFF)
        hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
        return _umax_i32(lo, hi)

    def _iq_pack_store(h_dw_i, qs_raw, B128_IDX, SUB, slot):
        # Quantize this lane's 8 elements (4 bf16x2 dwords) and store into the A-LDS
        # slot at the position issue_a_ds_read expects. fp4: 8 fp4 -> one i32 in a
        # 16 B swizzled block. fp8: 8 fp8 -> two i32; the 16 B block is
        # B128_IDX*8 + lsa3*2 + lib//2 and the byte within it is (lib%2)*8 (matches
        # the split-64 fp8 read).
        r = fx.Int32(SUB * 16) + r_in_chunk
        mask_r = _lds_swizzle_mask(r)
        row_off = fx.Int32(slot * (BM * KH_TILE)) + r * fx.Int32(KH_TILE)
        if const_expr(a_dtype == "fp8"):
            blk_byte = (
                fx.Int32(B128_IDX * 128)
                + lane_shr2_and3 * fx.Int32(32)
                + (lib >> fx.Int32(1)) * fx.Int32(16)
            )
            off = row_off + (blk_byte ^ mask_r) + (lib & fx.Int32(1)) * fx.Int32(8)
            # cvt.pk.fp8 packs 2 fp8 into a 16b lane of a vector<2xi16> accumulator
            # (dstLoHiSel = lo/hi); two lanes -> 4 fp8 = one i32 stored to LDS.
            i16x2 = fx.Vector.make_type(2, fx.Int16)
            zero16 = fx.Vector.filled(2, 0, fx.Int16)
            pk0 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, zero16, _bf16x2(h_dw_i[0]), qs_raw, 0
            )
            pk0 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, pk0, _bf16x2(h_dw_i[1]), qs_raw, 1
            )
            pk1 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, zero16, _bf16x2(h_dw_i[2]), qs_raw, 0
            )
            pk1 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, pk1, _bf16x2(h_dw_i[3]), qs_raw, 1
            )
            _scalar_store(
                s_aq_i32x1_tiles,
                off // fx.Int32(4),
                fx.Vector(pk0).bitcast(fx.Int32)[0],
                fx.Int32,
            )
            _scalar_store(
                s_aq_i32x1_tiles,
                (off + fx.Int32(4)) // fx.Int32(4),
                fx.Vector(pk1).bitcast(fx.Int32)[0],
                fx.Int32,
            )
        else:
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            off = row_off + ((kb_in_kt * fx.Int32(16)) ^ mask_r) + lib * fx.Int32(4)
            pk = as_ir_value(fx.Int32(0))
            for j in range_constexpr(4):
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(
                    T.i32, pk, _bf16x2(h_dw_i[j]), qs_raw, j
                )
            _scalar_store(s_aq_i32x1_tiles, off // fx.Int32(4), fx.Int32(pk), fx.Int32)

    def _inline_quant_core_batch(specs, slot, scale_accum):
        n = len(specs)
        h_dw = [
            [fx.Int32(as_ir_value(h_v[j])) for j in range_constexpr(4)]
            for (_b, _s, h_v) in specs
        ]
        a = [_iq_block_amax(h_dw[i]) for i in range_constexpr(n)]
        s1 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(
                    as_ir_value(a[i]),
                    as_ir_value(a[i]),
                    0xB1,
                    0xF,
                    0xF,
                    True,
                )
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s1[i]) for i in range_constexpr(n)]
        s2 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(
                    as_ir_value(a[i]),
                    as_ir_value(a[i]),
                    0x4E,
                    0xF,
                    0xF,
                    True,
                )
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s2[i]) for i in range_constexpr(n)]
        e8 = [
            _inline_e8m0(a[i], max_norm=448.0 if a_dtype == "fp8" else 6.0)
            for i in range_constexpr(n)
        ]
        for i in range_constexpr(n):
            B128_IDX, SUB, _hv = specs[i]
            qs_raw = as_ir_value(
                fx.Float32(as_ir_value(e8[i] << fx.Int32(23)).bitcast(T.f32))
            )
            _iq_pack_store(h_dw[i], qs_raw, B128_IDX, SUB, slot)
            pack_byte = B128_IDX * 2 + SUB
            scale_accum = scale_accum | (e8[i] << fx.Int32(pack_byte * 8))
        return scale_accum

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        return _inline_quant_core_batch([(B128_IDX, SUB, h_v)], slot, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        _scalar_store(asc_i32_tiles, off // fx.Int32(4), scale_accum, fx.Int32)

    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
            r = fx.make_rmem_tensor(bq_reg_lay, fx.Int32)
            fx.copy(
                bq_copy_atom,
                fx.slice(bq_tiles, (None, tile_idx)),
                r,
                soffset=b_load_s_base[j] // fx.Int32(4),
            )
            b_slot[j][half] = r

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            idx = (v + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(bscale_reg_lay, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=s_off // fx.Int32(4),
            )
            bs_slot[mw] = r.load()[0]

    def _mma(ci, opsel_a, opsel_b, a_frag, b_frag, sa, sb):
        # Scaled 16x16x128 MFMA via fx.gemm; accumulate in place (d == c == ci).
        # opsel_a/opsel_b select the e8m0 scale byte in the shared 256-K word and are
        # baked into the atom, exactly mirroring the raw intrinsic's opsel operands.
        fx.gemm(
            scale_atoms[(opsel_a, opsel_b)],
            ci,
            a_frag,
            b_frag,
            ci,
            scale_a=sa,
            scale_b=sb,
        )

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J):
        # Accumulators are pre-seeded to 0, so every issue accumulates (A*B + C).
        if const_expr(interleave):
            mni = J // 2
        else:
            mni = J % 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]

        def issue_cluster(in_b):
            if const_expr(kMChunks == 1):
                sa = a_scale[0]
                _mma(accm[0][J], 0, 0 + in_b, a[0][0], bJ0, sa, sb)
                _mma(accm[0][J], 2, 2 + in_b, a[0][1], bJ1, sa, sb)
            else:
                for sub in range_constexpr(kSubBlocks):
                    i0 = sub * 2 + 0
                    i1 = sub * 2 + 1
                    sa = a_scale[sub]
                    _mma(accm[i0][J], 0, 0 + in_b, a[i0][0], bJ0, sa, sb)
                    _mma(accm[i1][J], 1, 0 + in_b, a[i1][0], bJ0, sa, sb)
                    _mma(accm[i0][J], 2, 2 + in_b, a[i0][1], bJ1, sa, sb)
                    _mma(accm[i1][J], 3, 2 + in_b, a[i1][1], bJ1, sa, sb)

        if const_expr(BN == 128 and not interleave):

            @flyc.jit
            def issue_bn128_cluster():
                if (wave & fx.Int32(1)) == fx.Int32(0):
                    issue_cluster(0)
                else:
                    issue_cluster(1)

            issue_bn128_cluster()
        else:
            issue_cluster(J % 2 if const_expr(interleave) else J // 2)

    _relax_prologue = (BM == 128) and not inline_quant
    if const_expr(not inline_quant):
        issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        if const_expr(inline_quant):
            scale_accum = fx.Int32(0)
            scale_accum = inline_quant_kt(
                0, 0, K_C, K_C, cached_row_inline, scale_accum
            )
            issue_b_load_j(b[K_C], K_C, 0)
            issue_b_load_j(b[K_C], K_C, 1)
            scale_accum = inline_quant_kt(
                1, 0, K_C, K_C, cached_row_inline, scale_accum
            )
            if const_expr(N_REPS > 2):
                issue_b_load_j(b[K_C], K_C, 2)
                issue_b_load_j(b[K_C], K_C, 3)
            inline_quant_pack_write(K_C, scale_accum)
        else:
            issue_a_load_lds(K_C, K_C)
            if const_expr(not _relax_prologue):
                for j in range_constexpr(N_REPS):
                    issue_b_load_j(b[K_C], K_C, j)
        if const_expr(not _relax_prologue):
            issue_b_scale_load(b_scale_v[K_C], K_C)
    if const_expr(_relax_prologue):
        rocdl.sched_barrier(0)
        for K_C in range_constexpr(kStages):
            for j in range_constexpr(N_REPS):
                issue_b_load_j(b[K_C], K_C, j)
            issue_b_scale_load(b_scale_v[K_C], K_C)

    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
            a_cur = issue_a_ds_read(read_slot)
        else:
            a_cur = issue_a_ds_read(read_slot)
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
        if const_expr(not inline_quant):
            issue_a_load_lds(write_slot, K_C)
        if const_expr(inline_quant):
            h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline)
            h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline)
            rocdl.sched_barrier(0)
        for J in range_constexpr(N_REPS):
            if const_expr(BM != 128):
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
            mfma_cluster(b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J)
            if const_expr(BM != 128):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(b[slot_b], K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(b_scale_v[slot_b], K_C)
        if const_expr(inline_quant):
            scale_accum = _inline_quant_core_batch(
                [(0, 0, h_v0), (1, 0, h_v1)], write_slot, fx.Int32(0)
            )
            inline_quant_pack_write(K_C, scale_accum)

    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(kt)
            a_cur = issue_a_ds_read(kt % kAStages)
        else:
            a_cur = issue_a_ds_read(kt % kAStages)
            asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(N_REPS):
            mfma_cluster(b[kt % kStages], a_cur, asc_cur, b_scale_v[kt % kStages], J)

    gpu.barrier()

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    acc_layout = fx.make_layout((BM, BN), (BN, 1))

    def acc_idx(row, col):
        return _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_reg_lay = fx.make_layout(1, 1)
    acc_flat_view = fx.make_view(
        fx.recast_iter(fx.Float32, lds_raw_ptr), fx.make_layout(BM * BN, 1)
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, fx.make_layout(1, 1))

    def acc_store(idx, value):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        r.store(fx.Vector.from_elements([fx.Float32(value)], fx.Float32))
        fx.copy_atom_call(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        fx.copy_atom_call(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(N_REPS):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave * fx.Int32(BN // 8) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(BN_INT) + col_local) if is_up else col_local
            vec = fx.Vector(fx.memref_load_vec(accm[i][J]))
            for v in range_constexpr(4):
                idx = acc_idx(row_base + fx.Int32(v), lds_col)
                acc_store(idx, vec[v])

    gpu.barrier()

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    def run_epilogue():
        aqout_layout = fx.make_layout((BM, OUT_ROW_BYTES), (OUT_ROW_BYTES, 1))
        # UniversalCopy has no nontemporal/cache-hint knob; dropped (perf-neutral).
        aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
        scales_per_mr = [None] * M_REPS

        for mr in range_constexpr(M_REPS):
            row_local = fx.Int32(mr * 16) + m_lane

            gate_vs = [None] * 8
            up_vs = [None] * 8
            for ee in range_constexpr(8):
                col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
                gate_col = wave_grp * fx.Int32(32) + col_in_grp
                up_col = fx.Int32(BN_INT) + gate_col
                gate_vs[ee] = acc_load(acc_idx(row_local, gate_col))
                up_vs[ee] = acc_load(acc_idx(row_local, up_col))
                if const_expr(enable_bias):
                    out_col = n_block_idx * fx.Int32(BN_INT) + gate_col
                    bias_base = e * fx.Int32(N_OUT)
                    gate_vs[ee] = gate_vs[ee] + _global_f32_at(
                        arg_bias, bias_base + out_col
                    )
                    up_vs[ee] = up_vs[ee] + _global_f32_at(
                        arg_bias, bias_base + fx.Int32(inter) + out_col
                    )
            result = _activation_mul_batch(
                gate_vs,
                up_vs,
                act=act,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                swiglu_limit=swiglu_limit,
            )
            if const_expr(out_dtype == "fp8"):
                # Match the existing two-stage path: activation is materialized
                # as BF16 before the inter-stage MXFP8 quantization.
                result = [value.to(fx.BFloat16).to(fx.Float32) for value in result]

            local_max = _fabs_f32(result[0])
            for ee in range_constexpr(1, 8):
                local_max = local_max.maximumf(_fabs_f32(result[ee]))
            lm_i = _inline_dpp_quad_amax(
                fx.Int32(as_ir_value(local_max).bitcast(T.i32))
            )
            local_max = fx.Float32(as_ir_value(lm_i).bitcast(T.f32))

            e8m0, qscale = _e8m0_from_amax(
                local_max, max_norm=448.0 if out_dtype == "fp8" else 6.0
            )
            scales_per_mr[mr] = e8m0

            qscale_raw = as_ir_value(qscale)
            out_row = m_row + row_local
            if const_expr(out_dtype == "fp8"):
                i16x2 = fx.Vector.make_type(2, fx.Int16)
                zero16 = fx.Vector.filled(2, 0, fx.Int16)
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
                byte_pos = (
                    n_block_idx * fx.Int32(BN_INT)
                    + wave_grp * fx.Int32(32)
                    + kk * fx.Int32(8)
                )
                store_off = _layout_idx(aqout_layout, out_row, byte_pos)
                _scalar_store(
                    aqout_tiles,
                    store_off // fx.Int32(4),
                    fx.Vector(packed0).bitcast(fx.Int32)[0],
                    fx.Int32,
                )
                _scalar_store(
                    aqout_tiles,
                    (store_off + fx.Int32(4)) // fx.Int32(4),
                    fx.Vector(packed1).bitcast(fx.Int32)[0],
                    fx.Int32,
                )
            else:
                packed_i32 = as_ir_value(fx.Int32(0))
                for w in range_constexpr(4):
                    packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        packed_i32,
                        as_ir_value(result[2 * w]),
                        as_ir_value(result[2 * w + 1]),
                        qscale_raw,
                        w,
                    )
                byte_pos = (
                    n_block_idx * fx.Int32(BN_INT // 2)
                    + wave_grp * fx.Int32(16)
                    + kk * fx.Int32(4)
                )
                store_off = _layout_idx(aqout_layout, out_row, byte_pos)
                _scalar_store(
                    aqout_tiles,
                    store_off // fx.Int32(4),
                    fx.Int32(packed_i32),
                    fx.Int32,
                )

        # (chunk, ku, wave_grp, m_lane) -> dword index; shape is a placeholder.
        ascaleout_layout = fx.make_layout(
            (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
        )
        ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
        ascaleout_i32_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int32, 1 << 24)
        atomic_add_atom = fx.make_copy_atom(
            fx.UniversalAtomicAdd(fx.Int32, syncscope=fx.rocdl.SyncScope.Agent),
            fx.Int32,
        )
        atomic_reg_lay = fx.make_layout(1, 1)

        def atomic_add_scale(idx, value):
            r = fx.make_rmem_tensor(atomic_reg_lay, fx.Int32)
            r.store(fx.Vector.from_elements([value], fx.Int32))
            fx.copy_atom_call(
                atomic_add_atom,
                r,
                fx.slice(ascaleout_i32_tiles, (None, idx)),
            )

        if kk == fx.Int32(0):
            if const_expr(BN == 128):
                # Match shuffle_scale's physical layout for logical scale column
                # c = 2*n_block_idx + wave_grp:
                #   (ku, n4, n2) = (c//8, c%4, (c//4)%2).
                ku = n_block_idx >> fx.Int32(2)
                scale_wave_grp = (n_block_idx * fx.Int32(2) + wave_grp) & fx.Int32(3)
                ikxdl = (n_block_idx >> fx.Int32(1)) & fx.Int32(1)
            else:
                ku = n_block_idx >> fx.Int32(1)
                scale_wave_grp = wave_grp
                ikxdl = n_block_idx & fx.Int32(1)
            if const_expr(BM == 16):
                chunk = m_block_idx >> fx.Int32(1)
                row_half = m_block_idx & fx.Int32(1)
                dword_off = _layout_idx(
                    ascaleout_layout, chunk, ku, scale_wave_grp, m_lane
                )
                byte_pos = ikxdl * fx.Int32(2) + row_half
                packed_scale = scales_per_mr[0] << (byte_pos * fx.Int32(8))
                atomic_add_scale(dword_off, packed_scale)
            else:
                for sub in range_constexpr(kSubBlocks):
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

    if const_expr(BN == 128):

        @flyc.jit
        def run_bn128_epilogue():
            # BN128 produces 64 post-activation columns: two 32-column scale groups.
            if wave_grp < fx.Int32(2):
                run_epilogue()

        run_bn128_epilogue()
    else:
        run_epilogue()


def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL):
    kAStages = 2 if BM == 128 else 3
    kSubBlocks = 1 if BM < 32 else BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = lds_acc_bytes_for(BM, BN)
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    return kAStages, kSubBlocks, kMChunks, lds_bytes


_G1_VARIANTS = {
    "fp4": {
        (32, True, False),
        (32, False, False),
        (64, True, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    },
    # a8w4: fp8 (e4m3) A x mxfp4 W1. Same tiling as fp4; A is 1 B/elem so the
    # A-LDS tile doubles. inline_quant (bf16 hidden -> fp8) supported at (16,True).
    "fp8": {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    },
}


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BN=256,
    BK=256,
    interleave=False,
    xcd_swizzle=0,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    enable_bias=False,
):
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if (BM, use_nt, inline_quant) not in _G1_VARIANTS[a_dtype]:
        raise AssertionError(
            f"unsupported gemm1 variant (a_dtype={a_dtype}, BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
        )
    if out_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"out_dtype must be 'fp4' or 'fp8', got {out_dtype!r}")
    if act not in ("silu", "swiglu", "situv2"):
        raise AssertionError(f"act must be 'silu', 'swiglu', or 'situv2', got {act!r}")
    if act == "situv2" and (situ_beta <= 0.0 or situ_linear_beta <= 0.0):
        raise AssertionError("SiTUv2 beta values must be positive")
    if act == "swiglu" and swiglu_limit <= 0.0:
        raise AssertionError("Swiglu limit must be positive")

    assert (
        BN in (128, 256) and BK == 256
    ), f"only BN in (128, 256) and BK==256 supported, got BN={BN} BK={BK}"
    assert BN == 256 or not interleave, "BN=128 only supports separated gate/up layout"
    KH_TILE = BK if a_dtype == "fp8" else BK // 2
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _NUM_N_BLOCKS = num_n_blocks_for(_N_OUT, BN)
    _OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(_INTER)

    _, _, _, lds_bytes = _bm_constants(BM, BN, KH_TILE, _K_TILES_TOTAL)

    variant_tag = "iq" if inline_quant else ("nt" if use_nt else "cached")
    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    gu_tag = "il" if interleave else "sep"
    name_suffix = f"{a_dtype}_h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{variant_tag}_{gu_tag}"
    if out_dtype != "fp4":
        name_suffix += f"_o{out_dtype}"
    if act == "situv2":
        beta_tag = (
            str(float(situ_beta)).replace("-", "m").replace("+", "p").replace(".", "p")
        )
        linear_tag = (
            str(float(situ_linear_beta))
            .replace("-", "m")
            .replace("+", "p")
            .replace(".", "p")
        )
        name_suffix += f"_{act}_sb{beta_tag}_slb{linear_tag}"
    elif act == "swiglu":
        limit_tag = (
            str(float(swiglu_limit))
            .replace("-", "m")
            .replace("+", "p")
            .replace(".", "p")
        )
        name_suffix += f"_swiglu_slim{limit_tag}"
    if enable_bias:
        name_suffix += "_bias"
    if BN != 256:
        name_suffix += f"_bn{BN}"
    if xcd_swizzle > 0:
        name_suffix += f"_xcd{xcd_swizzle}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=f"gemm1_scale_zero_{name_suffix}", known_block_size=[256, 1, 1])
    def scale_zero_kernel(
        arg_ascaleout: fx.Int64,
        i32_num_vec4: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        vec_idx = bx * fx.Int32(256) + tx
        scale_ptr_ty = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )
        scale_ptr = fx.inttoptr(scale_ptr_ty, arg_ascaleout)
        scale_flat = fx.make_view(
            scale_ptr,
            fx.make_layout(fx.Int64(i32_num_vec4) * fx.Int64(4), 1),
        )
        scale_tiles = fx.logical_divide(scale_flat, fx.make_layout(4, 1))
        zero_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        zero_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        zero_reg.store(fx.Vector.filled(4, 0, fx.Int32))
        if vec_idx < i32_num_vec4:
            fx.copy_atom_call(
                zero_atom,
                zero_reg,
                fx.slice(scale_tiles, (None, vec_idx)),
            )

    @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
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
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NUM_N_BLOCKS)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(as_ir_value(xc), as_ir_value(_xr)))
                + _udiv(pid, _NXCD)
            )
            _ng = fx.Int32(_SW * _NUM_N_BLOCKS)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(
                arith.minsi(as_ir_value(remaining_m), as_ir_value(fx.Int32(_SW)))
            )
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_NUM_N_BLOCKS) + n_block

        if bx_i32 < bound:
            if const_expr(_SW > 0):
                _tile = _xcd(bx_i32)
            else:
                _tile = bx_i32
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
                _tile,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=inline_quant,
                a_dtype=a_dtype,
                out_dtype=out_dtype,
                act=act,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                swiglu_limit=swiglu_limit,
                enable_bias=enable_bias,
                K=_K,
                N_OUT=_N_OUT,
                NE=_NE,
                interleave=interleave,
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
        if const_expr(BM == 16):
            # Two M blocks atomically pack scale bytes into each dword. A prior
            # dispatch is required because in-kernel cross-workgroup zeroing races.
            max_m_blocks = i32_grid // fx.Int32(_NUM_N_BLOCKS)
            scale_chunks = (max_m_blocks + fx.Int32(1)) // fx.Int32(2)
            num_vec4 = scale_chunks * fx.Int32(_OUT_AS_PER_CHUNK_DW // 4)
            zero_grid_x = fx.Int64((num_vec4 + fx.Int32(255)) // fx.Int32(256))
            scale_zero_kernel(arg_ascaleout, num_vec4).launch(
                grid=(zero_grid_x, 1, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        gemm1_kernel(
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
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1
