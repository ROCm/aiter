# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device body (BM32): gemm2 down."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as _arith
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int8,
    Int16,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from .mxfp4_gemm_common import _fabs_f32 as fabs_f32
from .mxfp4_gemm_common import (
    _inline_dpp_pair_amax,
    _inline_dpp_quad_amax,
    _udiv,
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_typed_ptr,
    lds_vec_load,
)
from .mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask

STORE_CACHE_MODIFIER = 2

_FP8_E8M0_SHIFT = 7

_G2_EPI_LANES = 32


def bq_view(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
    num_records_bytes=None,
):
    """Layout view over preshuffled B for one N-row tile; slice -> i32<4:1> (16B=32 fp4). num_records_bytes (has_pad pad-skip) sizes to REAL K; None -> max_size=False byte-identical default."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane[0,4)->64, nlane[0,16)->4,
    # K_tile->K_HALVES*256, half->256, kpack4->1.
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 4)
    view = fx.Tensor(
        fx.make_view(
            base_iter,
            fx.make_layout(shape, (64, 4, K_HALVES * 256, 256, 1)),
        )
    )
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_view_fp8(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
    num_records_bytes=None,
):
    """Layout view over preshuffled FP8 B; pair selects two 16B cells per MFMA."""
    base = bq_view(
        arg_bq,
        row_elems,
        KH4,
        K_TILES_TOTAL * 2,
        K_HALVES,
        num_records_bytes=num_records_bytes,
    )
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 2, 4)
    stride = (64, 4, K_HALVES * 2 * 256, 2 * 256, 256, 1)
    return fx.Tensor(fx.make_view(fx.get_iter(base), fx.make_layout(shape, stride)))


def scale_view(
    arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """Layout view over an e8m0 scale buffer (A-scale per 32-row chunk / B-scale per n-pack); slice -> i32<1:1> scale word. num_records_bytes (has_pad pad-skip) sizes to real extent; None -> max_size=False byte-identical default."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_scale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def scale_mma_atoms(a_dtype, b_dtype):
    """16 (opselA,opselB) scaled-MFMA atoms for FP4/FP8 operands."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    elem_b = Float8E4M3FN if b_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, elem_b, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def mma_one_j(
    J,
    in_b,
    sa,
    sb,
    bq_frags_kt,
    a_frags,
    c_frags,
    atoms,
    i0=0,
    single_rg=False,
    rg_off=0,
    k_start=0,
    k_halves=2,
):
    """One J-cluster of scaled MFMAs over a 32-row A-scale group (row-groups i0, i0+1); each is
    an fx.gemm on i32 A/B frags (fp8 A = i32<8:1>, fp4 A = i32<4:1>), e8m0 words on scale_a/scale_b.
    sa: 32-row A-scale reg. single_rg (BM16): one 16-row group, rg_off picks its byte.
    """
    row_groups = (rg_off,) if const_expr(single_rg) else range(2)
    for k in range(k_start, k_start + k_halves):
        for im in row_groups:
            i = i0 if const_expr(single_rg) else i0 + im
            fx.gemm(
                atoms[(2 * k + im, 2 * k + in_b)],
                c_frags[i][J],
                a_frags[i][k],
                bq_frags_kt[J][k],
                c_frags[i][J],
                scale_a=sa,
                scale_b=sb,
            )


def issue_a_load_lds_dt(
    arg_aq,
    aq_num_records,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    is_f8,
    KH_TILE_A,
    K_BYTES,
    BM=32,
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via the flat buffer view bounds."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4  # rows each wave loads (BM32: 8, BM64: 16)
    # BM16 fp4: partial-wave round-robin (waves 2,3 re-load, harmless); BM>=32 byte-identical per-wave blocks.
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        n_gather_calls = BM // rows_per_call
        gather_base_row = (wave % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    atom = lds_dma_atom_128()
    src = flat_buffer_view(
        arg_aq,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=aq_num_records,
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            if const_expr(is_f8)
            else lds_swizzle_mask(lds_row + a_lane_row, KH_TILE_A)
        )
        car = m_row + lds_row + a_lane_row  # direct sorted row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        # The byte offset is non-negative and 4-byte aligned; avoid signed-division fixup VGPRs.
        v_e = (voffset + kt * KH_TILE_A).shrui(fx.Int32(2))
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(s_aq_base, off, elem_ty=T.i32, align=16)
        )


@flyc.jit
def gemm2_body_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_bias,
    i32_M,
    i32_max_m_blocks,
    arg_out,
    bx_i32,
    lane,
    wave,
    arg_aq,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    g2_kstatic=False,
    aStages,
    a_slot_alias=False,
    a_dtype,
    b_dtype,
    use_reduce=False,
    topk=1,
    has_pad=False,
    SBM=None,
    mn_idx=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    g2_bf16_lds=False,
    route_out_fp8=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_epi_lanes=None,
    g2_apre=False,
    enable_bias=False,
    k_valid_halves=None,
    has_kpad=False,
    has_npad=False,
):
    # GEMM2 double-buffers B weight and scale one tile ahead. bhoist issues that
    # prefetch above the LDS barrier; ascale_pf prefetches A-scale one tile ahead.
    # SBM (sort padding unit) >= BM (compute tile); SBM==BM default byte-identical.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    kScaleSubBlocks = max(1, kMChunks // 2)
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    is_f8_b = b_dtype == "fp8"
    B_NDW = 8 if is_f8_b else 4
    B_PAIR = 2 if is_f8_b else 1
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = _udiv(K_rt, fx.Int32(a_pack))
    kc_rt = _udiv(K_rt + fx.Int32(255), fx.Int32(256))
    K_TILES_RT = _udiv(K_rt, fx.Int32(BK))
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = _udiv(N_OUT_rt, fx.Int32(32)) * kBS_stride_n0_dw
    num_n_blocks = _udiv(N_OUT_rt, fx.Int32(BN))
    KH4 = _udiv(K_rt, fx.Int32(4 if is_f8_b else 8))
    K_TILES_MAX = INTER_MAX // BK
    K_SCALE_CHUNKS_MAX = (INTER_MAX + 255) // 256
    total_k_halves = K_TILES_MAX * kHalves
    if k_valid_halves is None:
        k_valid_halves = total_k_halves

    # has_pad OOB pad-skip (const_expr-gated): K-skip sizes 16N B-weight buffer to REAL K; N-skip zeros fully-pad-N w2 tiles (col >= N_real=N_OUT-npad; PERF-ONLY). B-scale NOT shrunk.
    bq_num_records = None
    N_real = None
    if const_expr(has_kpad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = _udiv(K_real + fx.Int32(127), fx.Int32(128))
        bq_num_records = halves_real * fx.Int32(1024 * B_PAIR)
    if const_expr(has_npad):
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # block -> (m_block_idx, n_block_idx); e = sorted_expert_ids[SBM-padded sort block] (SBM==BM: sort_block==m_block_idx).
    if const_expr(mn_idx is not None):
        m_block_idx, n_block_idx = mn_idx
    else:
        m_block_idx = _udiv(bx_i32, num_n_blocks)
        n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    m_row = m_block_idx * BM
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
    else:
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[_udiv(m_row, fx.Int32(SBM))]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    lds_acc_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype, b_dtype)

    aq_num_records = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * K_BYTES)
    A_NDW = 8 if is_f8_a else 4
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
            aq_num_records,
            s_aq_base,
            slot,
            kt,
            m_row,
            wave,
            lane,
            is_f8_a,
            KH_TILE_A,
            K_BYTES,
            BM=BM,
        )

    def issue_a_ds_read(slot):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                if const_expr(is_f8_a):
                    mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
                    col0 = lane_div_16 * 16 + k * 128
                    col_lo = col0 ^ mask
                    col_hi = (col0 + 64) ^ mask
                    lo = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_lo,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    hi = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_hi,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                    a_frags[i][k].store(a64.bitcast(fx.Int32))
                else:
                    mask = lds_swizzle_mask(lane_mod_16, KH_TILE_A)
                    lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                    vec = lds_vec_load(
                        s_aq_base,
                        row_off + lds_col,
                        Vec.make_type(4, fx.Int32),
                        fx.Int32,
                        align=16,
                    )
                    a_frags[i][k].store(Vec(vec))

    # Scale words (e8m0): shared scale_view / copy atom for both A and B. A-scale is one
    # word per 32-row chunk, each view bounded to bytes remaining after its baked base.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * fx.Int64(4)
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def scale_chunk_tile(kt):
        return (
            kt
            if const_expr(tilesPerScaleChunk == 1)
            else _udiv(kt, fx.Int32(tilesPerScaleChunk))
        )

    def load_a_scale_tile(kt):
        chunk_kt = scale_chunk_tile(kt)
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(Vec(saf.load())[0])
        return out

    # B-weight + B-scale: global->register, streamed per K-tile (not LDS-staged).
    # b128 weight copy atom; cache modifier 2=nontemporal, 0=default.
    b_catom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if use_nt else 0), 32)

    def make_bq_view(j):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        nrec = bq_num_records
        if const_expr(has_npad and has_kpad):
            # N-skip: fully-pad-N tile (col >= 16-aligned N_real) -> 0 records so weight loads OOB -> 0.
            nrec = (col < N_real).select(bq_num_records, fx.Int32(0))
        if const_expr(is_f8_b):
            return bq_view_fp8(
                arg_bq,
                e * N_OUT_rt + col,
                KH4,
                K_TILES_MAX,
                kHalves,
                num_records_bytes=nrec,
            )
        return bq_view(
            arg_bq,
            e * N_OUT_rt + col,
            KH4,
            K_TILES_MAX,
            kHalves,
            num_records_bytes=nrec,
        )

    bq_views = [make_bq_view(j) for j in range_constexpr(numAccN)]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = (
        None
        if const_expr(is_f8_b)
        else bq_views[0][0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    )
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_value_load(dst, j, half, kt_rt):
        if const_expr(is_f8_b):
            lo = fx.make_rmem_tensor(4, Int32)
            hi = fx.make_rmem_tensor(4, Int32)
            fx.copy(
                b_catom,
                bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, 0, None],
                lo,
            )
            fx.copy(
                b_catom,
                bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, 1, None],
                hi,
            )
            lo_v = Vec(fx.memref_load_vec(lo))
            hi_v = Vec(fx.memref_load_vec(hi))
            dst.store(lo_v.shuffle(hi_v, list(range(B_NDW))))
        else:
            fx.copy(
                b_catom,
                bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
                dst,
            )

    def issue_bscale_into(bsf, chunk_kt):
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def issue_b_load_into(bqf, bsf, kt_rt, valid_halves=None):
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                if const_expr(valid_halves is None or half < valid_halves):
                    issue_b_value_load(bqf[j][half], j, half, kt_rt)
        if const_expr(bsf is not None):
            issue_bscale_into(bsf, scale_chunk_tile(kt_rt))

    def make_bq_fragments():
        if const_expr(is_f8_b):
            return [
                [fx.make_rmem_tensor(B_NDW, Int32) for _ in range_constexpr(kHalves)]
                for _ in range_constexpr(numAccN)
            ]
        return [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]

    def make_scale_fragments(count):
        return [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(count)]

    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return scale.shrui(scale_shift)

    K_REAL_RT = K_rt - fx.Int32(i32_kpad)

    def mfma_cluster(bqf, bsf, sa, kt_rt, interleave=None, valid_halves=None):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sa = [
            shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)
        ]
        sb_words = [
            shift_scale_word(Vec(bsf[mni].load())[0], kt_rt)
            for mni in range_constexpr(nPairs)
        ]
        for J in range_constexpr(numAccN):
            mni, in_b = J // 2, J % 2
            sb = sb_words[mni]

            def emit_mma(k_start, k_count, J=J, in_b=in_b, sb=sb):
                if const_expr(is_bm16):
                    mma_one_j(
                        J,
                        in_b,
                        sa[0],
                        sb,
                        bqf,
                        a_frags,
                        c_frags,
                        mma_atoms,
                        i0=0,
                        single_rg=True,
                        k_start=k_start,
                        k_halves=k_count,
                    )
                else:
                    for sub in range_constexpr(kScaleSubBlocks):
                        mma_one_j(
                            J,
                            in_b,
                            sa[sub],
                            sb,
                            bqf,
                            a_frags,
                            c_frags,
                            mma_atoms,
                            i0=2 * sub,
                            k_start=k_start,
                            k_halves=k_count,
                        )

            if const_expr(valid_halves is not None):
                if const_expr(valid_halves > 0):
                    emit_mma(0, valid_halves)
            else:
                for k in range_constexpr(kHalves):
                    if kt_rt * fx.Int32(BK) + fx.Int32(k * 128) < K_REAL_RT:
                        emit_mma(k, 1)
            if const_expr(interleave is not None and J > 0):
                interleave[J - 1]()
        if const_expr(interleave is not None):
            interleave[numAccN - 1]()

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = Vec.filled(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(numAccN)]

    def init_c_carry():
        return load_c_carry()

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    def _epilog(accm, **kw):
        atomic_bf16_epilog(
            lds_acc_base,
            accm,
            arg_out,
            arg_stids,
            arg_sweights,
            arg_bias,
            e,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT_rt,
            BN=BN,
            use_reduce=use_reduce,
            topk=topk,
            SBM=SBM,
            g2_bf16_lds=g2_bf16_lds,
            route_out_fp8=route_out_fp8,
            g2_defer_weight=g2_defer_weight,
            g2_out_pitch_align=g2_out_pitch_align,
            g2_scale_blk=g2_scale_blk,
            g2_epi_lanes=g2_epi_lanes,
            enable_bias=enable_bias,
            **kw,
        )

    g2_interleave = const_expr(g2_kstatic and g2_bf16_lds)
    epi_thunks = [] if const_expr(g2_interleave) else None
    if const_expr(g2_interleave):
        _epilog(c_frags, emit_thunks=epi_thunks)

    if const_expr(g2_kstatic):
        KT = K_TILES_MAX
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(zero4)
        cur_bqf = make_bq_fragments()
        nxt_bqf = make_bq_fragments()
        chunk_of = [kt // tilesPerScaleChunk for kt in range(KT)]
        n_slots = min(2, chunk_of[-1] + 1)
        bsf_slots = [make_scale_fragments(nPairs) for _ in range_constexpr(n_slots)]
        saf_slots = None
        if const_expr(g2_ascale_pf):
            saf_slots = [
                make_scale_fragments(kScaleSubBlocks) for _ in range_constexpr(n_slots)
            ]

        def _ks_issue_ascale(saf, kt_rt):
            sa_t = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(Vec.from_elements([sa_t[sub]], Int32))

        def _ks_issue_scales(kt):
            slot = chunk_of[kt] % n_slots
            issue_bscale_into(bsf_slots[slot], scale_chunk_tile(fx.Int32(kt)))
            if const_expr(g2_ascale_pf):
                _ks_issue_ascale(saf_slots[slot], fx.Int32(kt))

        def _ks_prefetch(kt):
            valid_halves = min(kHalves, max(0, k_valid_halves - kt * kHalves))
            issue_b_load_into(
                nxt_bqf,
                None,
                fx.Int32(kt),
                valid_halves=valid_halves,
            )
            if const_expr(kt == 0 or chunk_of[kt] != chunk_of[kt - 1]):
                _ks_issue_scales(kt)

        issue_b_load_into(
            cur_bqf,
            None,
            fx.Int32(0),
            valid_halves=min(kHalves, k_valid_halves),
        )
        _ks_issue_scales(0)
        rocdl.sched_barrier(0)

        a_all_resident = const_expr((aStages if g2_apre else kStages) >= KT)
        if const_expr(a_all_resident):
            gpu.barrier()

        for kt in range_constexpr(KT):
            kt_rt = fx.Int32(kt)
            valid_halves_kt = min(kHalves, max(0, k_valid_halves - kt * kHalves))
            cur_bsf = bsf_slots[chunk_of[kt] % n_slots]
            if const_expr(g2_bhoist) and const_expr(kt + 1 < KT):
                _ks_prefetch(kt + 1)
            if const_expr(not a_all_resident):
                gpu.barrier()
            issue_a_ds_read(fx.Int32(kt % aStages))
            if const_expr(not a_all_resident and kt + kStages < KT):
                if const_expr(a_slot_alias):
                    gpu.barrier()  # prefetch rewrites the slot just ds_read
                issue_a_load_lds(
                    fx.Int32((kt + kStages) % aStages), fx.Int32(kt + kStages)
                )
            if const_expr(g2_ascale_pf):
                cur_saf = saf_slots[chunk_of[kt] % n_slots]
                sa = [
                    Vec(cur_saf[sub].load())[0]
                    for sub in range_constexpr(kScaleSubBlocks)
                ]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist) and const_expr(kt + 1 < KT):
                _ks_prefetch(kt + 1)
            _il = epi_thunks if const_expr(kt == KT - 1) else None
            if const_expr(g2_interleave and kt == KT - 1):
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                gpu.barrier()
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(
                cur_bqf,
                cur_bsf,
                sa,
                kt_rt,
                interleave=_il,
                valid_halves=valid_halves_kt,
            )
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            cur_bqf, nxt_bqf = nxt_bqf, cur_bqf
    else:
        # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
        cur_bqf = make_bq_fragments()
        cur_bsf = make_scale_fragments(nPairs)
        nxt_bqf = make_bq_fragments()
        nxt_bsf = make_scale_fragments(nPairs)
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = make_scale_fragments(kScaleSubBlocks)
            nxt_saf = make_scale_fragments(kScaleSubBlocks)

        def load_b_fragments(bqf, bsf, saf):
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(saf[sub].load())
            return out

        def store_b_carry(state, base):
            n = base
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    cur_bqf[j][half].store(state[n])
                    n += 1
            for mw in range_constexpr(nPairs):
                cur_bsf[mw].store(state[n])
                n += 1
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    cur_saf[sub].store(state[n])
                    n += 1
            return n

        def issue_a_scale_load_into(saf, kt_rt):
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(Vec.from_elements([sa[sub]], Int32))

        def load_carry():
            return init_c_carry() + load_b_fragments(cur_bqf, cur_bsf, cur_saf)

        def store_carry(state):
            base = store_c_carry(state)
            store_b_carry(state, base)

        def yield_carry():
            return load_c_carry() + load_b_fragments(nxt_bqf, nxt_bsf, nxt_saf)

        # Prologue: prefetch tile 0's B/B-scale into "current" (VALUES enter via init=load_carry()).
        issue_b_load_into(cur_bqf, cur_bsf, fx.Int32(0))
        if const_expr(g2_ascale_pf):
            issue_a_scale_load_into(cur_saf, fx.Int32(0))
        rocdl.sched_barrier(0)

        def prefetch_next_b(kt_rt):
            # Prefetch NEXT tile's B; if none, copy current through (rotate_b_carry state, unused after loop).
            nxt_b = kt_rt + fx.Int32(1)
            if nxt_b < K_TILES_RT:
                issue_b_load_into(nxt_bqf, nxt_bsf, nxt_b)
                if const_expr(g2_ascale_pf):
                    issue_a_scale_load_into(nxt_saf, nxt_b)
            else:
                for j in range_constexpr(numAccN):
                    for half in range_constexpr(kHalves):
                        nxt_bqf[j][half].store(cur_bqf[j][half].load())
                for mw in range_constexpr(nPairs):
                    nxt_bsf[mw].store(cur_bsf[mw].load())
                if const_expr(g2_ascale_pf):
                    for sub in range_constexpr(kScaleSubBlocks):
                        nxt_saf[sub].store(cur_saf[sub].load())

        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_carry(),
        ):
            store_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(g2_bhoist):
                prefetch_next_b(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt_a = kt_rt + fx.Int32(kStages)
            if const_expr(a_slot_alias):
                gpu.barrier()  # outside the runtime if: barriers must be uniform
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            if const_expr(g2_ascale_pf):
                sa = [
                    Vec(cur_saf[sub].load())[0]
                    for sub in range_constexpr(kScaleSubBlocks)
                ]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist):
                prefetch_next_b(kt_rt)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    if const_expr(g2_interleave):
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        _epilog(None, lds_ready=True)
    else:
        _epilog(
            [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)]
        )


# ---- Atomic bf16 epilogue (shared store path; gemm2 down-proj) ----
def atomic_bf16_epilog(
    lds_acc_base,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    arg_bias,
    expert_id,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    *,
    BN=256,
    use_reduce=False,
    topk=1,
    SBM=None,
    g2_bf16_lds=False,
    route_out_fp8=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_epi_lanes=None,
    emit_thunks=None,
    lds_ready=False,
    enable_bias=False,
):
    if SBM is None:
        SBM = BM
    EPI_LANES = _G2_EPI_LANES if g2_epi_lanes is None else int(g2_epi_lanes)
    EPI_ROWS = 256 // EPI_LANES
    M_REPS = BM // EPI_ROWS
    ROUTE_VEC = BN // EPI_LANES
    if const_expr(use_reduce and route_out_fp8):
        assert BM % EPI_ROWS == 0, (EPI_LANES, EPI_ROWS, BM)
        assert ROUTE_VEC % 4 == 0, (EPI_LANES, BN, ROUTE_VEC)
        assert g2_scale_blk in (ROUTE_VEC, 2 * ROUTE_VEC, 4 * ROUTE_VEC), (
            EPI_LANES,
            ROUTE_VEC,
            g2_scale_blk,
        )
    bf16_src = const_expr(bool(g2_bf16_lds) and bool(route_out_fp8))
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // EPI_LANES
    n_lane = tx_i32 % EPI_LANES
    store_vec = 2
    store_group_n = EPI_LANES * store_vec
    col_start = n_lane * store_vec
    wave_n = BN // 4

    def flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = flat_buffer(arg_stids, T.i32, 4)
    sweights = flat_buffer(arg_sweights, T.f32, 4)
    bias_f32 = None
    if const_expr(enable_bias):
        bias_f32 = flat_buffer(arg_bias, T.f32, 4)
    out_bf16 = flat_buffer(arg_out, T.bf16, 4)
    out_bf16_ptr = global_typed_ptr(arg_out, T.bf16, align=2)
    out_i8 = flat_buffer(arg_out, T.i8, 4)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)
    store_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(STORE_CACHE_MODIFIER), Int32)
    store_i8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(STORE_CACHE_MODIFIER), Int8)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    def load_bias(col):
        return load_scalar(
            load_f32,
            bias_f32,
            expert_id * N_OUT + col,
            Float32,
        )

    defer_w = bool(g2_defer_weight)

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * EPI_ROWS + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, Int32))
        if const_expr(not defer_w):
            weight.append(load_scalar(load_f32, sweights, sorted_pos, Float32))

    if const_expr(not lds_ready):
        if const_expr(emit_thunks is None):
            gpu.barrier()
        if const_expr(g2_bf16_lds):

            def _write_i(i, only_j=None):
                row_base = fx.Int32(i * 16) + lane_div_16 * 4
                w_row = (
                    None
                    if const_expr(defer_w)
                    else [
                        load_scalar(load_f32, sweights, m_row + row_base + v, Float32)
                        for v in range_constexpr(4)
                    ]
                )
                for J in range_constexpr(numAccN):
                    if const_expr(only_j is not None and J != only_j):
                        continue
                    col = wave * wave_n + J * 16 + lane_mod_16
                    vec = (
                        Vec(accm[i][J].load())
                        if const_expr(emit_thunks is not None)
                        else Vec(accm[i][J])
                    )

                    def _scaled(v, vec=vec):
                        f = fx.Float32(vec[v])
                        return f if const_expr(defer_w) else f * fx.Float32(w_row[v])

                    for v0 in range_constexpr(0, 4, 2):
                        pk = Vec.from_elements(
                            [_scaled(v0), _scaled(v0 + 1)], Float32
                        ).to(BFloat16)
                        for h in range_constexpr(2):
                            lds_base_bf16[(row_base + v0 + h) * BN + col] = pk[h]

            if const_expr(emit_thunks is not None):
                for J in range_constexpr(numAccN):
                    emit_thunks.append(
                        lambda J=J: [_write_i(i, J) for i in range_constexpr(BM // 16)]
                    )
            else:
                for i in range_constexpr(BM // 16):
                    _write_i(i)
        else:
            for i in range_constexpr(BM // 16):
                row_base = fx.Int32(i * 16) + lane_div_16 * 4
                for J in range_constexpr(numAccN):
                    col = wave * wave_n + J * 16 + lane_mod_16
                    vec = Vec(accm[i][J])
                    for v in range_constexpr(4):
                        idx = (row_base + v) * BN + col
                        lds_base_fptr[idx] = fx.Float32(vec[v])
        if const_expr(emit_thunks is not None):
            return
        gpu.barrier()

    def store_one_mr(mr):
        row_in_block = fx.Int32(mr * EPI_ROWS) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(use_reduce):
            # reduce out_row can reach tokens*topk (large-M) so compute the element base in i64 (atomic i32 path byte-identical).
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            if const_expr(route_out_fp8):
                row_pitch = N_OUT + _udiv(N_OUT, fx.Int32(g2_scale_blk))
                if const_expr(g2_out_pitch_align > 0):
                    al = fx.Int32(g2_out_pitch_align)
                    row_pitch = ((row_pitch + al - fx.Int32(1)) // al) * al
                row_base_addr = out_row * fx.Int64(row_pitch)
            else:
                row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(
                    n_block_idx * BN + col_start
                )
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        if const_expr(use_reduce and route_out_fp8):
            route_vec = ROUTE_VEC
            route_group_n = EPI_LANES * route_vec
            n_rg = (BN + route_group_n - 1) // route_group_n
            for rg in range_constexpr(n_rg):
                col_lane8 = rg * route_group_n + n_lane * fx.Int32(route_vec)

                def store_route_group(col_lane8, rg=rg):
                    col_g0 = n_block_idx * BN + col_lane8
                    bvals = []
                    vals = []
                    for q in range_constexpr(route_vec):
                        idx_q = row_in_block * BN + col_lane8 + fx.Int32(q)
                        if const_expr(bf16_src):
                            bval = lds_base_bf16[idx_q]
                            if const_expr(enable_bias):
                                bias_val = load_bias(col_g0 + q)
                                if const_expr(not defer_w):
                                    bias_val = bias_val * weight[mr]
                                bval = (fx.Float32(bval) + bias_val).to(BFloat16)
                            bvals.append(bval)
                        elif const_expr(g2_bf16_lds):
                            val = fx.Float32(lds_base_bf16[idx_q])
                            if const_expr(enable_bias):
                                bias_val = load_bias(col_g0 + q)
                                if const_expr(not defer_w):
                                    bias_val = bias_val * weight[mr]
                                val = val + bias_val
                            vals.append(val)
                        elif const_expr(defer_w):
                            val = fx.Float32(lds_base_fptr[idx_q])
                            if const_expr(enable_bias):
                                val = val + load_bias(col_g0 + q)
                            vals.append(val)
                        else:
                            val = fx.Float32(lds_base_fptr[idx_q])
                            if const_expr(enable_bias):
                                val = val + load_bias(col_g0 + q)
                            vals.append(val * weight[mr])
                    if const_expr(bf16_src):
                        msk = Vec.filled([2], 0x7FFF, Int16)
                        acc = None
                        for h in range_constexpr(route_vec // 2):
                            p = (
                                Vec.from_elements(
                                    [bvals[2 * h], bvals[2 * h + 1]], BFloat16
                                ).bitcast(Int16)
                                & msk
                            )
                            acc = p if h == 0 else Vec(_arith.maxui(_raw(acc), _raw(p)))
                        a0, a1 = fx.Int32(acc[0]), fx.Int32(acc[1])
                        amax_bits = (a0 > a1).select(a0, a1) << fx.Int32(16)
                    else:
                        local_max = fabs_f32(vals[0])
                        for q in range_constexpr(1, route_vec):
                            local_max = local_max.maximumf(fabs_f32(vals[q]))
                        amax_bits = fx.Int32(_raw(local_max).bitcast(T.i32))
                    if const_expr(g2_scale_blk == route_vec):
                        pass
                    elif const_expr(g2_scale_blk == 2 * route_vec):
                        amax_bits = _inline_dpp_pair_amax(amax_bits)
                    elif const_expr(g2_scale_blk == 4 * route_vec):
                        amax_bits = _inline_dpp_quad_amax(amax_bits)
                    ax_e = (amax_bits >> fx.Int32(23)) & fx.Int32(0xFF)
                    e8m0 = ax_e - fx.Int32(_FP8_E8M0_SHIFT)
                    e8m0 = (e8m0 < fx.Int32(1)).select(fx.Int32(1), e8m0)
                    e8m0 = (amax_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
                    block_scale = (amax_bits == fx.Int32(0)).select(
                        fx.Float32(1.0),
                        fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32)),
                    )
                    bs_raw = _raw(block_scale)
                    pk_ty = T.vec(2, T.i16)

                    def pk_seed():
                        return _raw(Vec.filled([2], 0, fx.Int16))

                    words = []
                    for d in range_constexpr(route_vec // 4):
                        w = pk_seed()
                        for h in range_constexpr(2):
                            e = 4 * d + 2 * h
                            if const_expr(bf16_src):
                                src2 = Vec.from_elements(
                                    [bvals[e], bvals[e + 1]], BFloat16
                                )
                                w = rocdl.cvt_scalef32_pk_fp8_bf16(
                                    pk_ty, w, _raw(src2), bs_raw, h
                                )
                            else:
                                w = rocdl.cvt_scalef32_pk_fp8_f32(
                                    pk_ty,
                                    w,
                                    _raw(vals[e]),
                                    _raw(vals[e + 1]),
                                    bs_raw,
                                    h,
                                )
                        words.append(w)
                    emit_stores(col_g0, words, e8m0)

                def emit_stores(col_g0, words, e8m0, rg=rg):
                    row_val_off = row_base_addr + fx.Int64(col_g0)
                    packed_frag = fx.make_rmem_tensor(1, Int32)
                    for d in range_constexpr(len(words)):
                        packed_frag.store(Vec(words[d]).bitcast(Int32))
                        fx.copy(
                            store_i32,
                            packed_frag,
                            out_i8[None, row_val_off + fx.Int64(4 * d)],
                        )
                    scale_off = (
                        row_base_addr
                        + fx.Int64(N_OUT)
                        + fx.Int64(_udiv(col_g0, fx.Int32(g2_scale_blk)))
                    )
                    scale_frag = fx.make_rmem_tensor(1, Int8)
                    scale_frag.store(Vec.from_elements([e8m0.to(Int8)], Int8))
                    fx.copy(store_i8, scale_frag, out_i8[None, scale_off])

                @flyc.jit
                def store_route_group_if_valid(col_lane8):
                    if col_lane8 < fx.Int32(BN):
                        store_route_group(col_lane8)

                store_route_group_if_valid(col_lane8)
        else:
            for s in range_constexpr(BN // store_group_n):
                # adjacent ee=0,1 contiguous -> one 2-wide load.
                idx0 = row_in_block * BN + col_start + s * store_group_n
                if const_expr(g2_bf16_lds):
                    pk = Vec(
                        lds_vec_load(
                            lds_acc_base,
                            idx0 * 2,
                            Vec.make_type(store_vec, BFloat16),
                            BFloat16,
                            align=4,
                        )
                    )
                    if const_expr(enable_bias):
                        bias_col = n_block_idx * BN + col_start + s * store_group_n
                        bias0 = load_bias(bias_col)
                        bias1 = load_bias(bias_col + 1)
                        if const_expr(not defer_w):
                            bias0 = bias0 * weight[mr]
                            bias1 = bias1 * weight[mr]
                        pk = Vec.from_elements(
                            [
                                fx.Float32(pk[0]) + bias0,
                                fx.Float32(pk[1]) + bias1,
                            ],
                            Float32,
                        ).to(BFloat16)
                else:
                    v2 = Vec(
                        lds_vec_load(
                            lds_acc_base,
                            idx0 * 4,
                            Vec.make_type(store_vec, Float32),
                            Float32,
                            align=8,
                        )
                    )
                    v0 = fx.Float32(v2[0])
                    v1 = fx.Float32(v2[1])
                    if const_expr(enable_bias):
                        bias_col = n_block_idx * BN + col_start + s * store_group_n
                        v0 = v0 + load_bias(bias_col)
                        v1 = v1 + load_bias(bias_col + 1)
                    if const_expr(defer_w):
                        pk = Vec.from_elements([v0, v1], Float32).to(BFloat16)
                    else:
                        pk = Vec.from_elements(
                            [v0 * weight[mr], v1 * weight[mr]], Float32
                        ).to(BFloat16)
                out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                out_frag.store(pk)
                out_off = row_base_addr + fx.Int64(s * store_group_n)
                if const_expr(use_reduce):
                    fx.ptr_store(pk, out_bf16_ptr + out_off)
                else:
                    fx.copy(atomic_bf16x2, out_frag, out_bf16[None, out_off])

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)

        @flyc.jit
        def store_if_valid(token_id, mr):
            if token_id < i32_M:
                store_one_mr(mr)

        store_if_valid(token_id, mr)
