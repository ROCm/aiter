# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device body (BM32): gemm2 down."""

import flydsl.compiler as flyc
import os

import flydsl.expr as fx
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
from flydsl._mlir.dialects import arith as _arith_d

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


def scale_mma_atoms(a_dtype, b_dtype, swap_ab=False):
    """16 (opselA,opselB) scaled-MFMA atoms for FP4/FP8 operands.

    swap_ab builds the MIRRORED atom (B as the A operand and vice versa), which
    computes C^T = B^T A^T. The point is the ACCUMULATOR LAYOUT, not the math:
    with the operands swapped, lane L ends up holding C[row, 4 consecutive cols]
    instead of C[4 consecutive rows, col]. That is the layout the route-out store
    wants, so the whole LDS transpose in the epilogue (128 narrow ds_write_b16 +
    16 ds_read_b128 + a barrier + a 64 KiB C slab) becomes unnecessary.

    Ported from FlyDSL 741f3d6 (fp4_gemm_4wave), which does this with hand-written
    inline asm. It does not need asm here: MFMA_Scale takes both element types as
    parameters, so the mirrored form is just the atom with (elem_b, elem_a) and
    the opsel pair exchanged. Keyed by the ORIGINAL (osa, osb) so mma_one_j's
    indexing is unchanged.
    """
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    elem_b = Float8E4M3FN if b_dtype == "fp8" else Float4E2M1FN
    if swap_ab:
        return {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, elem_b, elem_a, opsel_a=osb, opsel_b=osa
                )
            )
            for osa in range(4)
            for osb in range(4)
        }
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
    k_halves=2,
    swap_ab=False,
):
    """One J-cluster of scaled MFMAs over a 32-row A-scale group (row-groups i0, i0+1); each is
    an fx.gemm on i32 A/B frags (fp8 A = i32<8:1>, fp4 A = i32<4:1>), e8m0 words on scale_a/scale_b.
    sa: 32-row A-scale reg. single_rg (BM16): one 16-row group, rg_off picks its byte.
    """
    row_groups = (rg_off,) if const_expr(single_rg) else range(2)
    for k in range(k_halves):
        for im in row_groups:
            i = i0 if const_expr(single_rg) else i0 + im
            if const_expr(swap_ab):
                fx.gemm(
                    atoms[(2 * k + im, 2 * k + in_b)],
                    c_frags[i][J],
                    bq_frags_kt[J][k],
                    a_frags[i][k],
                    c_frags[i][J],
                    scale_a=sb,
                    scale_b=sa,
                )
            else:
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
    nwaves=4,
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via the flat buffer view bounds."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // nwaves  # rows each wave loads (BM32: 8, BM64: 16)
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
    g2_cpad=0,
    g2_swz=0,
    g2_pk8=0,
    g2_tr=0,
    g2_serp=0,
    route_out_fp8=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_off32=False,
    g2_epi_lanes=None,
    g2_bfull=False,
    g2_split_lds=False,
    g2_wave_epi=False,
    g2_noepi=False,
    g2_sched=True,
    g2_schedmask=0,
    g2_mfmarep=1,
    g2_prio=1,
    g2_apf2=0,
    g2_bsingle=False,
    g2_epirep=1,
    g2_swapab=False,
    g2_cbase=0,
    g2_earlybar=False,
    g2_interleave=False,
    g2_noppad=0,
    g2_barpad=0,
    g2_dspad=0,
    g2_klnop=0,
    g2_epinop=0,
    g2_illag=1,
    g2_asplit=False,
    g2_bspread=False,
    g2_nwaves=4,
    g2_nrep=1,
    g2_apre=False,
    g2_kmerge=1,
    g2_brstore=True,
):
    # GEMM2 double-buffers B weight and scale one tile ahead. bhoist issues that
    # prefetch above the LDS barrier; ascale_pf prefetches A-scale one tile ahead.
    # SBM (sort padding unit) >= BM (compute tile); SBM==BM default byte-identical.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // g2_nwaves) // 16  # 16-column MFMA subblocks per wave
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

    # has_pad OOB pad-skip (const_expr-gated): K-skip sizes 16N B-weight buffer to REAL K; N-skip zeros fully-pad-N w2 tiles (col >= N_real=N_OUT-npad; PERF-ONLY). B-scale NOT shrunk.
    bq_num_records = None
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = _udiv(K_real + fx.Int32(127), fx.Int32(128))
        bq_num_records = halves_real * fx.Int32(1024 * B_PAIR)
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

    c_lds_bytes_for_a = const_expr(BM * (BN + g2_cpad) * 2 if g2_bf16_lds else BM * BN * 4)
    s_aq_base = lds_base_i32 + (const_expr(c_lds_bytes_for_a) if const_expr(g2_asplit) else 0)
    lds_acc_base = lds_base_i32 + (
        aStages * slot_bytes if const_expr(g2_split_lds) else const_expr(g2_cbase)
    )
    mma_atoms = scale_mma_atoms(a_dtype, b_dtype, swap_ab=g2_swapab)

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
            nwaves=g2_nwaves,
        )

    def issue_a_ds_read(slot, dst=None, sel=None):
        dst = a_frags if dst is None else dst
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                if const_expr(sel is None or sel(i, k)):
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
                        dst[i][k].store(a64.bitcast(fx.Int32))
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
                        dst[i][k].store(Vec(vec))

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

    def make_bq_view(j, nblk=None):
        nblk = n_block_idx if nblk is None else nblk
        col = nblk * BN + wave * (BN // g2_nwaves) + j * 16
        nrec = bq_num_records
        if const_expr(has_pad):
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

    def make_b_views(nblk):
        """B weight + B-scale views for one n-block. These are the ONLY pieces of
        per-tile state: the expert, m_row, the A staging and the A-scale views are
        all shared across n-blocks of the same m rows, which is what makes the
        n-block pipeline cheap."""
        bqv = [make_bq_view(j, nblk) for j in range_constexpr(numAccN)]
        mnib = nblk * (BN // 16 // 2) + wave * (BN // g2_nwaves // 16 // 2)
        bsv = [
            scale_view(
                arg_bscale,
                e * kbs_per_expert_dw + (mnib + mw) * kBS_stride_n0_dw,
                K_SCALE_CHUNKS_MAX,
                k0_stride_dw=kBS_stride_k0_dw,
            )
            for mw in range_constexpr(nPairs)
        ]
        return bqv, bsv

    bq_views, bscale_views = make_b_views(n_block_idx)

    frag_tmpl = (
        None
        if const_expr(is_f8_b)
        else bq_views[0][0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    )
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_value_load(dst, j, half, kt_rt, views=None):
        views = bq_views if views is None else views
        if const_expr(is_f8_b):
            lo = fx.make_rmem_tensor(4, Int32)
            hi = fx.make_rmem_tensor(4, Int32)
            fx.copy(
                b_catom,
                views[j][lane_div_16, lane_mod_16, kt_rt, half, 0, None],
                lo,
            )
            fx.copy(
                b_catom,
                views[j][lane_div_16, lane_mod_16, kt_rt, half, 1, None],
                hi,
            )
            lo_v = Vec(fx.memref_load_vec(lo))
            hi_v = Vec(fx.memref_load_vec(hi))
            dst.store(lo_v.shuffle(hi_v, list(range(B_NDW))))
        else:
            fx.copy(
                b_catom,
                views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
                dst,
            )

    def issue_bscale_into(bsf, chunk_kt, views=None):
        views = bscale_views if views is None else views
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def issue_b_load_into(bqf, bsf, kt_rt, bqv=None, bsv=None, only_j=None):
        bqv = bq_views if bqv is None else bqv
        for j in range_constexpr(numAccN):
            if const_expr(only_j is None or j == only_j):
                for half in range_constexpr(kHalves):
                    issue_b_value_load(bqf[j][half], j, half, kt_rt, views=bqv)
        if const_expr(bsf is not None):
            issue_bscale_into(bsf, scale_chunk_tile(kt_rt), views=bsv)

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

    def mfma_cluster(bqf, bsf, sa, kt_rt, af=None, interleave=None):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        af = a_frags if af is None else af
        _thunks = list(interleave) if interleave else []
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
            if const_expr(is_bm16):
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    af,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    k_halves=kHalves,
                    swap_ab=g2_swapab,
                )
                continue
            for sub in range_constexpr(kScaleSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    af,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
                    swap_ab=g2_swapab,
                )
            _t = J - const_expr(g2_illag)
            if const_expr(0 <= _t < len(_thunks)):
                _thunks[_t]()
        for _t in range_constexpr(max(0, numAccN - const_expr(g2_illag)), len(_thunks)):
            _thunks[_t]()

    def mfma_cluster_serp(bqf, bsf, sa, kt_rt):
        sa = [
            shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)
        ]
        sb_words = [
            shift_scale_word(Vec(bsf[mni].load())[0], kt_rt)
            for mni in range_constexpr(nPairs)
        ]
        order = []
        j0s = list(range(0, numAccN, 2))
        for _n, i0 in enumerate(range(0, kMChunks, 2)):
            for j0 in (reversed(j0s) if _n % 2 else j0s):
                order += [(i0 + di, j0 + dj) for di in range(2) for dj in range(2)]
        for k in range_constexpr(kHalves):
            for i, J in order:
                _atom = mma_atoms[(2 * k + (i % 2), 2 * k + (J % 2))]
                if const_expr(g2_swapab):
                    fx.gemm(
                        _atom,
                        c_frags[i][J],
                        bqf[J][k],
                        a_frags[i][k],
                        c_frags[i][J],
                        scale_a=sb_words[J // 2],
                        scale_b=sa[i // 2],
                    )
                else:
                    fx.gemm(
                        _atom,
                        c_frags[i][J],
                        a_frags[i][k],
                        bqf[J][k],
                        c_frags[i][J],
                        scale_a=sa[i // 2],
                        scale_b=sb_words[J // 2],
                    )

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

    _NREP = abs(g2_nrep)
    if const_expr(g2_nrep > 1 or g2_nrep < 0):
        KT = K_TILES_MAX
        gpu.barrier()  # A DMA (issued by the caller) has landed

        chunk_of = [kt // tilesPerScaleChunk for kt in range(KT)]

        _a_res = const_expr(kStages >= KT)

        def run_kloop(nblk, first):
            bqv, bsv = make_b_views(nblk)
            for i in range_constexpr(kMChunks):
                for J in range_constexpr(numAccN):
                    c_frags[i][J].store(zero4)
            bqf_cur = make_bq_fragments()
            bqf_nxt = make_bq_fragments()
            n_chunks = chunk_of[-1] + 1
            bsf = [make_scale_fragments(nPairs) for _ in range_constexpr(n_chunks)]
            issue_b_load_into(bqf_cur, bsf[chunk_of[0]], fx.Int32(0), bqv, bsv)
            for kt in range_constexpr(KT):
                if const_expr(kt + 1 < KT):
                    _nb = (
                        bsf[chunk_of[kt + 1]]
                        if const_expr(chunk_of[kt + 1] != chunk_of[kt])
                        else None
                    )
                    issue_b_load_into(bqf_nxt, _nb, fx.Int32(kt + 1), bqv, bsv)
                if const_expr(first and not _a_res):
                    gpu.barrier()
                issue_a_ds_read(fx.Int32(kt % aStages))
                if const_expr(first and kt + kStages < KT):
                    if const_expr(a_slot_alias):
                        gpu.barrier()
                    issue_a_load_lds(
                        fx.Int32((kt + kStages) % aStages), fx.Int32(kt + kStages)
                    )
                sa_kt = load_a_scale_tile(fx.Int32(kt))
                if const_expr(g2_sched):
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                mfma_cluster(bqf_cur, bsf[chunk_of[kt]], sa_kt, fx.Int32(kt))
                if const_expr(g2_sched):
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                bqf_cur, bqf_nxt = bqf_nxt, bqf_cur

        _uid = [0]

        def epi(nblk, phase):
            _uid[0] += 1
            accm_v = [
                [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
            ]
            atomic_bf16_epilog(
                lds_acc_base, accm_v, arg_out, arg_stids, arg_sweights, m_row, nblk,
                wave, lane, i32_M, BM, N_OUT_rt, BN=BN, use_reduce=use_reduce,
                topk=topk, SBM=SBM, g2_bf16_lds=g2_bf16_lds,
                g2_cpad=g2_cpad,
                g2_swz=g2_swz,
                g2_pk8=g2_pk8,
                g2_tr=g2_tr,
                g2_serp=g2_serp,
                route_out_fp8=route_out_fp8, g2_defer_weight=g2_defer_weight,
                g2_out_pitch_align=g2_out_pitch_align, g2_scale_blk=g2_scale_blk,
                g2_off32=g2_off32, g2_epi_lanes=g2_epi_lanes,
                g2_split_lds=g2_split_lds, g2_wave_epi=g2_wave_epi,
                g2_brstore=g2_brstore, phase=phase,
                g2_swapab=g2_swapab,
                g2_nwaves=g2_nwaves,
                epi_uid=_uid[0],
            )

        nb0 = n_block_idx * fx.Int32(_NREP)
        run_kloop(nb0, True)
        if const_expr(_NREP == 1):
            epi(nb0, "all")
            return
        epi(nb0, "write")
        for t in range_constexpr(1, _NREP):
            nbt = nb0 + fx.Int32(t)
            gpu.barrier()
            run_kloop(nbt, False)                      # matrix pipe
            epi(nb0 + fx.Int32(t - 1), "read")  # LDS + stores, independent
            gpu.barrier()
            epi(nbt, "write")
        gpu.barrier()
        epi(nb0 + fx.Int32(_NREP - 1), "read")
        return

    _epi_thunks = [] if const_expr(g2_interleave) else None
    if const_expr(g2_interleave):
        atomic_bf16_epilog(
            lds_acc_base,
            c_frags,
            arg_out,
            arg_stids,
            arg_sweights,
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
            g2_cpad=g2_cpad,
            g2_swz=g2_swz,
            g2_pk8=g2_pk8,
            g2_tr=g2_tr,
            route_out_fp8=route_out_fp8,
            g2_defer_weight=g2_defer_weight,
            g2_out_pitch_align=g2_out_pitch_align,
            g2_scale_blk=g2_scale_blk,
            g2_off32=g2_off32,
            g2_epi_lanes=g2_epi_lanes,
            g2_split_lds=g2_split_lds,
            g2_wave_epi=g2_wave_epi,
            g2_noepi=g2_noepi,
            g2_brstore=g2_brstore,
        g2_swapab=g2_swapab,
            epi_thunks=_epi_thunks,
            c_frags_lazy=c_frags,
            phase="write",
        )

    if const_expr(g2_noppad):
        for _n in range_constexpr(g2_noppad):
            rocdl.s_nop(0)

    if const_expr(g2_barpad):
        for _n in range_constexpr(g2_barpad):
            gpu.barrier()

    if const_expr(g2_kstatic):
        KT = K_TILES_MAX
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(zero4)
        cur_bqf = make_bq_fragments()
        nxt_bqf = cur_bqf if const_expr(g2_bsingle) else make_bq_fragments()
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

        def _ks_prefetch(kt, only_j=None):
            issue_b_load_into(nxt_bqf, None, fx.Int32(kt), only_j=only_j)
            if const_expr(only_j in (None, 0)) and const_expr(
                kt == 0 or chunk_of[kt] != chunk_of[kt - 1]
            ):
                _ks_issue_scales(kt)

        a_all_resident = const_expr((aStages if g2_apre else kStages) >= KT)

        if const_expr(g2_bfull):
            bqf_all = [make_bq_fragments() for _ in range_constexpr(KT)]
            n_chunks = chunk_of[-1] + 1
            bsf_all = [
                make_scale_fragments(nPairs) for _ in range_constexpr(n_chunks)
            ]
            saf_all = (
                [
                    make_scale_fragments(kScaleSubBlocks)
                    for _ in range_constexpr(n_chunks)
                ]
                if const_expr(g2_ascale_pf)
                else None
            )
            for kt in range_constexpr(KT):
                issue_b_load_into(bqf_all[kt], None, fx.Int32(kt))
                if const_expr(kt == 0 or chunk_of[kt] != chunk_of[kt - 1]):
                    issue_bscale_into(
                        bsf_all[chunk_of[kt]], scale_chunk_tile(fx.Int32(kt))
                    )
                    if const_expr(g2_ascale_pf):
                        _ks_issue_ascale(saf_all[chunk_of[kt]], fx.Int32(kt))
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.sched_barrier(0)
            if const_expr(a_all_resident):
                gpu.barrier()

            if const_expr(g2_kmerge > 1 and a_all_resident):
                af_all = [
                    [
                        [
                            fx.make_rmem_tensor(A_NDW, Int32)
                            for _ in range_constexpr(kHalves)
                        ]
                        for _ in range_constexpr(kMChunks)
                    ]
                    for _ in range_constexpr(KT)
                ]
                for g0 in range_constexpr(0, KT, g2_kmerge):
                    gn = min(g2_kmerge, KT - g0)
                    sa_g = []
                    for gi in range_constexpr(gn):
                        kt = g0 + gi
                        issue_a_ds_read(fx.Int32(kt % aStages), af_all[kt])
                        if const_expr(g2_ascale_pf):
                            sa_g.append(
                                [
                                    Vec(saf_all[chunk_of[kt]][sub].load())[0]
                                    for sub in range_constexpr(kScaleSubBlocks)
                                ]
                            )
                        else:
                            sa_g.append(load_a_scale_tile(fx.Int32(kt)))
                    if const_expr(g2_sched):
                        rocdl.sched_barrier(0)
                        rocdl.s_setprio(1)
                    for gi in range_constexpr(gn):
                        kt = g0 + gi
                        mfma_cluster(
                            bqf_all[kt],
                            bsf_all[chunk_of[kt]],
                            sa_g[gi],
                            fx.Int32(kt),
                            af=af_all[kt],
                        )
                    if const_expr(g2_sched):
                        rocdl.s_setprio(0)
                        rocdl.sched_barrier(0)
            else:
                for kt in range_constexpr(KT):
                    kt_rt = fx.Int32(kt)
                    if const_expr(not a_all_resident):
                        gpu.barrier()
                    issue_a_ds_read(fx.Int32(kt % aStages))
                    if const_expr(not a_all_resident and kt + kStages < KT):
                        if const_expr(a_slot_alias):
                            gpu.barrier()
                        issue_a_load_lds(
                            fx.Int32((kt + kStages) % aStages), fx.Int32(kt + kStages)
                        )
                    if const_expr(g2_ascale_pf):
                        sa = [
                            Vec(saf_all[chunk_of[kt]][sub].load())[0]
                            for sub in range_constexpr(kScaleSubBlocks)
                        ]
                    else:
                        sa = load_a_scale_tile(kt_rt)
                    if const_expr(g2_sched):
                        rocdl.sched_barrier(0)
                        rocdl.s_setprio(1)
                    mfma_cluster(bqf_all[kt], bsf_all[chunk_of[kt]], sa, kt_rt)
                    if const_expr(g2_sched):
                        rocdl.s_setprio(0)
                        rocdl.sched_barrier(0)
        else:
            issue_b_load_into(cur_bqf, None, fx.Int32(0))
            _ks_issue_scales(0)
            rocdl.sched_barrier(0)

            if const_expr(a_all_resident):
                gpu.barrier()

            _npf = min(int(g2_apf2), kMChunks * kHalves)

            def _in_pf(i, k):
                return k * kMChunks + i < _npf

            def _not_pf(i, k):
                return not _in_pf(i, k)

            _ds_scratch = (
                [
                    [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
                    for _ in range_constexpr(kMChunks)
                ]
                if const_expr(g2_dspad)
                else None
            )
            a_pf_cur = a_pf_nxt = None
            if const_expr(_npf):
                a_pf_cur = [
                    [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
                    for _ in range_constexpr(kMChunks)
                ]
                a_pf_nxt = [
                    [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
                    for _ in range_constexpr(kMChunks)
                ]
                issue_a_ds_read(fx.Int32(0), a_pf_cur, _in_pf)

            for kt in range_constexpr(KT):
                kt_rt = fx.Int32(kt)
                cur_bsf = bsf_slots[chunk_of[kt] % n_slots]
                _bspread = const_expr(g2_bspread and not g2_bsingle and g2_bhoist)
                if const_expr(g2_bsingle):
                    if const_expr(kt > 0):
                        _ks_prefetch(kt)
                elif const_expr(g2_bhoist) and const_expr(kt + 1 < KT):
                    if const_expr(not _bspread):
                        _ks_prefetch(kt + 1)
                if const_expr(not a_all_resident):
                    gpu.barrier()
                if const_expr(g2_dspad):
                    for _d in range_constexpr(g2_dspad):
                        issue_a_ds_read(fx.Int32(kt % aStages), _ds_scratch)
                if const_expr(_npf):
                    if const_expr(kt + 1 < KT):
                        issue_a_ds_read(
                            fx.Int32((kt + 1) % aStages), a_pf_nxt, _in_pf
                        )
                    issue_a_ds_read(fx.Int32(kt % aStages), a_frags, _not_pf)
                else:
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
                if const_expr(not g2_bsingle) and const_expr(not g2_bhoist) and const_expr(kt + 1 < KT):
                    _ks_prefetch(kt + 1)
                if const_expr(g2_sched):
                    rocdl.sched_barrier(g2_schedmask)
                    if const_expr(g2_prio):
                        rocdl.s_setprio(1)
                if const_expr(_npf):
                    _af = [
                        [
                            a_pf_cur[i][k] if _in_pf(i, k) else a_frags[i][k]
                            for k in range_constexpr(kHalves)
                        ]
                        for i in range_constexpr(kMChunks)
                    ]
                _il = _epi_thunks if const_expr(kt == KT - 1) else None
                if const_expr(_bspread) and const_expr(kt + 1 < KT):
                    _bt = [
                        (lambda kt=kt, _j=_j: _ks_prefetch(kt + 1, only_j=_j))
                        for _j in range_constexpr(numAccN)
                    ]
                    _il = (list(_il) + _bt) if _il else _bt
                if const_expr(g2_klnop):
                    for _n in range_constexpr(g2_klnop):
                        rocdl.s_nop(0)
                if const_expr(g2_earlybar and kt == KT - 1):
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    gpu.barrier()
                for _rep in range_constexpr(g2_mfmarep):
                    if const_expr(_npf):
                        mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt, _af, interleave=_il)
                    elif const_expr(_il is not None):
                        mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt, interleave=_il)
                    else:
                        (mfma_cluster_serp if const_expr(g2_serp) else mfma_cluster)(
                            cur_bqf, cur_bsf, sa, kt_rt
                        )
                if const_expr(g2_sched):
                    if const_expr(g2_prio):
                        rocdl.s_setprio(0)
                    rocdl.sched_barrier(g2_schedmask)
                if const_expr(not g2_bsingle):
                    cur_bqf, nxt_bqf = nxt_bqf, cur_bqf
                if const_expr(_npf):
                    a_pf_cur, a_pf_nxt = a_pf_nxt, a_pf_cur
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
    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
    ]
    for _erep in range_constexpr(g2_epirep):
        atomic_bf16_epilog(
            lds_acc_base,
            accm_vecs,
            arg_out,
            arg_stids,
            arg_sweights,
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
            g2_cpad=g2_cpad,
            g2_swz=g2_swz,
            g2_pk8=g2_pk8,
            g2_tr=g2_tr,
            route_out_fp8=route_out_fp8,
            g2_defer_weight=g2_defer_weight,
            g2_out_pitch_align=g2_out_pitch_align,
            g2_scale_blk=g2_scale_blk,
            g2_off32=g2_off32,
            g2_epi_lanes=g2_epi_lanes,
            g2_split_lds=g2_split_lds,
            g2_wave_epi=g2_wave_epi,
            g2_noepi=g2_noepi,
            g2_brstore=g2_brstore,
            g2_swapab=g2_swapab,
            g2_earlybar=g2_earlybar,
            g2_nwaves=g2_nwaves,
            g2_epinop=g2_epinop,
            phase=("read" if g2_interleave else "all"),
        )


# ---- Atomic bf16 epilogue (shared store path; gemm2 down-proj) ----
def atomic_bf16_epilog(
    lds_acc_base,
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
    *,
    BN=256,
    use_reduce=False,
    topk=1,
    SBM=None,
    g2_bf16_lds=False,
    g2_cpad=0,
    g2_swz=0,
    g2_pk8=0,
    g2_tr=0,
    g2_serp=0,
    route_out_fp8=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_off32=False,
    g2_epi_lanes=None,
    g2_split_lds=False,
    g2_wave_epi=False,
    g2_noepi=False,
    g2_brstore=True,
    g2_swapab=False,
    g2_earlybar=False,
    g2_interleave=False,
    g2_nwaves=4,
    g2_epinop=0,
    epi_thunks=None,
    c_frags_lazy=None,
    phase="all",
    epi_uid=0,
):
    """phase: 'all' (default, byte-identical) | 'write' (accm -> LDS, stops before
    the read barrier) | 'read' (readback + quantise + store). Splitting them lets
    the MREP driver emit the NEXT m-tile's K loop between the two halves, so the
    matrix pipe runs while this tile's epilogue and stores drain."""
    if SBM is None:
        SBM = BM
    EPI_LANES = _G2_EPI_LANES if g2_epi_lanes is None else int(g2_epi_lanes)
    EPI_ROWS = (g2_nwaves * 64) // EPI_LANES
    M_REPS = BM // EPI_ROWS
    ROUTE_VEC = BN // EPI_LANES
    if const_expr(use_reduce and route_out_fp8):
        if EPI_ROWS < 1 or BM % EPI_ROWS != 0:
            raise AssertionError(
                f"g2_epi_lanes={EPI_LANES} gives EPI_ROWS={EPI_ROWS}, which does "
                f"not tile BM={BM}"
            )
        if g2_scale_blk not in (ROUTE_VEC, 2 * ROUTE_VEC, 4 * ROUTE_VEC):
            raise AssertionError(
                f"g2_epi_lanes={EPI_LANES} gives route_vec={ROUTE_VEC}, but "
                f"g2_scale_blk={g2_scale_blk} is not 1x/2x/4x of it; the amax "
                f"cross-lane fold only covers those ratios"
            )
    numAccN = (BN // g2_nwaves) // 16  # 16-column MFMA subblocks per wave
    if const_expr(g2_wave_epi):
        WAVE_N = BN // g2_nwaves
        EPI_NLANE = WAVE_N // ROUTE_VEC       # lanes spanning one row's slice
        EPI_ROWS = 64 // EPI_NLANE            # rows covered per pass by one wave
        M_REPS = BM // EPI_ROWS
        if EPI_NLANE < 1 or 64 % EPI_NLANE or BM % EPI_ROWS:
            raise AssertionError(
                f"g2_wave_epi: WAVE_N={WAVE_N} / route_vec={ROUTE_VEC} does not "
                f"tile a 64-lane wave over BM={BM}"
            )
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    C_PAD = const_expr(g2_cpad if (g2_bf16_lds and not g2_wave_epi) else 0)
    C_PITCH = const_expr(BN + C_PAD)
    TR_PAD = const_expr(4)
    TR_M = const_expr(BM + TR_PAD)
    USE_TR = const_expr(
        bool(g2_tr) and g2_bf16_lds and not g2_wave_epi and route_out_fp8
    )
    USE_PK8 = const_expr(bool(g2_pk8) and g2_bf16_lds and route_out_fp8)
    SWZ = const_expr(bool(g2_swz) and g2_bf16_lds and not g2_wave_epi)
    BLK_STRIDE = const_expr(18)
    ROW_STRIDE = const_expr((BN // 16) * 18 + 4)
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    wave_n = BN // g2_nwaves
    row_pitch_c = const_expr(wave_n if g2_wave_epi else (BN + C_PAD))
    if const_expr(g2_wave_epi):
        m_lane = lane // EPI_NLANE
        n_lane = lane % EPI_NLANE
        lds_wave_off = wave * fx.Int32(BM * wave_n)   # in bf16 elements
        col_local0 = n_lane * ROUTE_VEC
    else:
        if const_expr(bool(g2_tr)):
            m_lane = tx_i32 % EPI_LANES
            n_lane = tx_i32 // EPI_LANES
        else:
            m_lane = tx_i32 // EPI_LANES
            n_lane = tx_i32 % EPI_LANES
        lds_wave_off = fx.Int32(0)
        col_local0 = None
    store_vec = 2
    store_group_n = EPI_LANES * store_vec
    col_start = n_lane * store_vec

    def flat_buffer(arg, elem_ty, align, nrec=None):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        if nrec is not None:
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=nrec)
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    _out_nrec = None
    if const_expr(use_reduce and g2_brstore):
        if const_expr(route_out_fp8):
            _rp = N_OUT + _udiv(N_OUT, fx.Int32(g2_scale_blk))
            if const_expr(g2_out_pitch_align > 0):
                _al = fx.Int32(g2_out_pitch_align)
                _rp = ((_rp + _al - fx.Int32(1)) // _al) * _al
        else:
            _rp = N_OUT * fx.Int32(2)
        _out_nrec = fx.Int64(i32_M) * fx.Int64(topk) * fx.Int64(_rp)

    stids = flat_buffer(arg_stids, T.i32, 4)
    sweights = flat_buffer(arg_sweights, T.f32, 4)
    out_bf16 = flat_buffer(arg_out, T.bf16, 4, _out_nrec)
    out_i8 = flat_buffer(arg_out, T.i8, 4, _out_nrec)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
    store_bf16x2 = fx.make_copy_atom(
        fx.rocdl.BufferCopy32b(STORE_CACHE_MODIFIER), BFloat16
    )
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)
    store_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(STORE_CACHE_MODIFIER), Int32)
    store_i8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(STORE_CACHE_MODIFIER), Int8)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    defer_w = bool(g2_defer_weight)

    def _off(v):
        """Widen an output-buffer byte offset to the addressing width in use."""
        return fx.Int32(v) if const_expr(g2_off32) else fx.Int64(v)

    if const_expr(g2_noepi):
        acc_sum = Vec(accm[0][0])[0]
        for i in range_constexpr(BM // 16):
            for J in range_constexpr(numAccN):
                if const_expr(i or J):
                    acc_sum = acc_sum + Vec(accm[i][J])[0]
        probe = fx.make_rmem_tensor(1, Int32)
        probe.store(Vec.from_elements([fx.Int32(_raw(acc_sum).bitcast(T.i32))], Int32))
        fx.copy(store_i32, probe, out_i8[None, _off(wave * 256 + lane * 4)])
        return

    if const_expr(g2_epinop):
        for _n in range_constexpr(g2_epinop):
            rocdl.s_nop(0)

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * EPI_ROWS + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, Int32))
        if const_expr(not defer_w):
            weight.append(load_scalar(load_f32, sweights, sorted_pos, Float32))

    if const_expr(phase != "read"):
      if const_expr(not g2_split_lds and not g2_wave_epi and not g2_earlybar):
        if const_expr(g2_swapab or g2_nwaves != 4):
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        gpu.barrier()  # C slab overlays the A slots; wait for every ds_read A
      if const_expr(g2_bf16_lds):
        def _write_i(i, _only_j=None):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            col_sw = fx.Int32(wave * wave_n) if const_expr(not g2_wave_epi) else fx.Int32(0)
            w_sw = (
                None
                if const_expr(defer_w or not g2_swapab)
                else load_scalar(
                    load_f32, sweights, m_row + fx.Int32(i * 16) + lane_mod_16, Float32
                )
            )
            w_row = (
                None
                if const_expr(defer_w or g2_swapab)
                else [
                    load_scalar(load_f32, sweights, m_row + row_base + v, Float32)
                    for v in range_constexpr(4)
                ]
            )
            for J in range_constexpr(numAccN):
                if const_expr(_only_j is not None) and const_expr(J != _only_j):
                    continue
                if const_expr(g2_wave_epi):
                    col = J * 16 + lane_mod_16          # wave-local [0, wave_n)
                    row_pitch_lds = wave_n
                else:
                    col = wave * wave_n + J * 16 + lane_mod_16
                    row_pitch_lds = C_PITCH
                vec = (
                    Vec(c_frags_lazy[i][J].load())
                    if const_expr(epi_thunks is not None)
                    else Vec(accm[i][J])
                )
                if const_expr(USE_TR):
                    if const_expr(defer_w):
                        _f4 = [fx.Float32(vec[_v]) for _v in range_constexpr(4)]
                    else:
                        _f4 = [
                            fx.Float32(vec[_v]) * fx.Float32(w_row[_v])
                            for _v in range_constexpr(4)
                        ]
                    _b4 = Vec.from_elements(_f4, Float32).to(BFloat16)
                    _tidx = col * fx.Int32(TR_M) + row_base
                    fx.ptr_store(_b4, lds_base_bf16 + _tidx)
                    continue
                if const_expr(g2_swapab):
                    _srow = fx.Int32(i * 16) + lane_mod_16
                    _scol = col_sw + fx.Int32(J * 16) + lane_div_16 * 4
                    for v0 in range_constexpr(0, 4, 2):
                        if const_expr(defer_w):
                            pk = Vec.from_elements(
                                [fx.Float32(vec[v0]), fx.Float32(vec[v0 + 1])], Float32
                            ).to(BFloat16)
                        else:
                            pk = Vec.from_elements(
                                [
                                    fx.Float32(vec[v0]) * fx.Float32(w_sw),
                                    fx.Float32(vec[v0 + 1]) * fx.Float32(w_sw),
                                ],
                                Float32,
                            ).to(BFloat16)
                        _sidx = lds_wave_off + _srow * row_pitch_lds + _scol + v0
                        if const_expr(int(g2_swapab) >= 2):
                            for _h in range_constexpr(2):
                                lds_base_bf16[_sidx + _h] = pk[_h]
                        else:
                            fx.ptr_store(pk, lds_base_bf16 + _sidx)
                    continue
                for v0 in range_constexpr(0, 4, 2):
                    if const_expr(defer_w):
                        pk = Vec.from_elements(
                            [fx.Float32(vec[v0]), fx.Float32(vec[v0 + 1])], Float32
                        ).to(BFloat16)
                    else:
                        pk = Vec.from_elements(
                            [
                                fx.Float32(vec[v0]) * fx.Float32(w_row[v0]),
                                fx.Float32(vec[v0 + 1]) * fx.Float32(w_row[v0 + 1]),
                            ],
                            Float32,
                        ).to(BFloat16)
                    for h in range_constexpr(2):
                        if const_expr(SWZ):
                            idx = (
                                (row_base + v0 + h) * fx.Int32(ROW_STRIDE)
                                + (
                                    wave * fx.Int32(wave_n // 16)
                                    + fx.Int32(J)
                                )
                                * fx.Int32(BLK_STRIDE)
                                + lane_mod_16
                            )
                        elif const_expr(epi_thunks is not None):
                            idx = _wbase[i] + (v0 + h) * row_pitch_lds + J * 16
                        else:
                            idx = (
                                lds_wave_off
                                + (row_base + v0 + h) * row_pitch_lds
                                + col
                            )
                        lds_base_bf16[idx] = pk[h]

        def _write_j(J):
            for i in range_constexpr(BM // 16):
                _write_i(i, J)

        if const_expr(epi_thunks is not None):
            _wbase = [
                fx.Int32(lds_wave_off)
                + (fx.Int32(i * 16) + lane_div_16 * 4) * fx.Int32(row_pitch_c)
                + (fx.Int32(wave * wave_n) if const_expr(not g2_wave_epi) else fx.Int32(0))
                + lane_mod_16
                for i in range_constexpr(BM // 16)
            ]
            for J in range_constexpr(numAccN):
                epi_thunks.append(lambda J=J: _write_j(J))
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
    if const_expr(phase == "all" and not g2_wave_epi):
        gpu.barrier()  # waves read columns their siblings wrote

    # read back + weighted store (atomic: fadd out[token_id]; reduce: store out[token_id*topk+slot]);
    # token_id<i32_M gates padding at runtime. At small/mid M the block-sparse sort floor is
    # ~97% padding rows (M64: 12288 pad vs 384 real); ungated, every padding row (token_id==M) issues
    # a weighted atomic-fadd into an OOB out-row -> 77x wasted atomics + L2 RMW serialization
    # (rocprof M64: TCC_ATOMIC 7.3M->95K, 932us->107us). Always gate; reduce already gated via
    # use_reduce. (fp4-atomic families are gated too -> strictly correct OOB-skip, kernel IR changes.)

    if const_expr(phase != "write"):
      def store_one_mr(mr):
          row_in_block = fx.Int32(mr * EPI_ROWS) + m_lane
          token_id = packed[mr] & fx.Int32(0x00FFFFFF)
          if const_expr(use_reduce):
              out_row_i32 = token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24))
              if const_expr(route_out_fp8):
                  row_pitch = N_OUT + _udiv(N_OUT, fx.Int32(g2_scale_blk))
                  if const_expr(g2_out_pitch_align > 0):
                      al = fx.Int32(g2_out_pitch_align)
                      row_pitch = ((row_pitch + al - fx.Int32(1)) // al) * al
                  if const_expr(g2_off32):
                      row_base_addr = out_row_i32 * row_pitch
                  else:
                      row_base_addr = fx.Int64(out_row_i32) * fx.Int64(row_pitch)
              elif const_expr(g2_off32):
                  row_base_addr = out_row_i32 * N_OUT + (n_block_idx * BN + col_start)
              else:
                  row_base_addr = fx.Int64(out_row_i32) * fx.Int64(N_OUT) + fx.Int64(
                      n_block_idx * BN + col_start
                  )
          else:
              out_row = token_id
              row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
          if const_expr(use_reduce and route_out_fp8):
              route_vec = ROUTE_VEC
              if const_expr(g2_wave_epi):
                  route_span, route_group_n = wave_n, EPI_NLANE * route_vec
              else:
                  route_span, route_group_n = BN, EPI_LANES * route_vec
              n_rg = (route_span + route_group_n - 1) // route_group_n
              for rg in range_constexpr(n_rg):
                  col_lane8 = rg * route_group_n + n_lane * fx.Int32(route_vec)

                  def store_route_group(col_lane8, rg=rg):
                      if const_expr(g2_wave_epi):
                          col_g0 = n_block_idx * BN + wave * wave_n + col_lane8
                          lds_row_pitch = wave_n
                      else:
                          col_g0 = n_block_idx * BN + col_lane8
                          lds_row_pitch = C_PITCH
                      vals = []
                      bvals = []
                      if const_expr(USE_TR):
                          _v4t = T.vec(4, T.bf16)
                          _lp = lds_base_bf16.type
                          _mbase = row_in_block - m_lane
                          for _r in range_constexpr(route_vec // 4):
                              _ncol = col_lane8 + fx.Int32(4 * _r)
                              _lrow = _ncol + (lane % fx.Int32(16)) // fx.Int32(4)
                              _lcol = _mbase + (lane % fx.Int32(4)) * fx.Int32(4)
                              _byte = (
                                  (_lrow * fx.Int32(TR_M) + _lcol) * fx.Int32(2)
                                  + fx.Int32(lds_acc_base)
                              )
                              _raw4 = rocdl.ds_read_tr16_b64(
                                  _v4t,
                                  fx.to_llvm_ptr(fx.inttoptr(_lp, fx.Int64(_byte))),
                              ).result
                              _vv = fx.Vector(_raw4, (4,), BFloat16)
                              for _e in range_constexpr(4):
                                  bvals.append(_vv[_e])
                                  vals.append(fx.Float32(_vv[_e]))
                      else:
                        for q in range_constexpr(route_vec):
                          if const_expr(SWZ):
                              _blk_base = col_lane8 >> fx.Int32(4)
                              idx_q = (
                                  row_in_block * fx.Int32(ROW_STRIDE)
                                  + (_blk_base + fx.Int32(q // 16))
                                  * fx.Int32(BLK_STRIDE)
                                  + fx.Int32(q % 16)
                              )
                          else:
                              idx_q = (
                                  lds_wave_off
                                  + row_in_block * lds_row_pitch
                                  + col_lane8
                                  + fx.Int32(q)
                              )
                          if const_expr(g2_bf16_lds):
                              _b = lds_base_bf16[idx_q]
                              bvals.append(_b)
                              vals.append(fx.Float32(_b))
                          elif const_expr(defer_w):
                              vals.append(fx.Float32(lds_base_fptr[idx_q]))
                          else:
                              vals.append(fx.Float32(lds_base_fptr[idx_q]) * weight[mr])
                      if const_expr(USE_PK8):
                          _msk = Vec.filled([2], 0x7FFF, Int16)
                          _acc = None
                          for _k in range_constexpr(route_vec // 2):
                              _p = Vec.from_elements(
                                  [bvals[2 * _k], bvals[2 * _k + 1]], BFloat16
                              ).bitcast(Int16)
                              _p = _p & _msk
                              _acc = (
                                  _p
                                  if _k == 0
                                  else Vec(
                                      _arith_d.MaxUIOp(
                                          _raw(_acc), _raw(_p)
                                      ).result
                                  )
                              )
                          _a0 = fx.Int32(_acc[0])
                          _a1 = fx.Int32(_acc[1])
                          _am = (_a0 > _a1).select(_a0, _a1)
                          amax_bits = _am << fx.Int32(16)
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
                      block_scale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
                      bs_raw = _raw(block_scale)
                      pk_ty = T.vec(2, T.i16)

                      def pk_seed():
                          return _raw(Vec.filled([2], 0, fx.Int16))

                      packed = []
                      if const_expr(USE_PK8):
                          for d in range_constexpr(route_vec // 4):
                              w = pk_seed()
                              for h in range_constexpr(2):
                                  e = 4 * d + 2 * h
                                  src2 = Vec.from_elements(
                                      [bvals[e], bvals[e + 1]], BFloat16
                                  )
                                  w = rocdl.cvt_scalef32_pk_fp8_bf16(
                                      pk_ty, w, _raw(src2), bs_raw, h
                                  )
                              packed.append(w)
                      else:
                          for d in range_constexpr(route_vec // 4):
                              w = pk_seed()
                              for h in range_constexpr(2):
                                  e = 4 * d + 2 * h
                                  w = rocdl.cvt_scalef32_pk_fp8_f32(
                                      pk_ty, w, _raw(vals[e]), _raw(vals[e + 1]), bs_raw, h
                                  )
                              packed.append(w)
                      emit_stores(col_g0, packed, e8m0)

                  def emit_stores(col_g0, packed, e8m0, rg=rg):
                      row_val_off = row_base_addr + _off(col_g0)
                      packed_frag = fx.make_rmem_tensor(1, Int32)
                      for d in range_constexpr(len(packed)):
                          packed_frag.store(Vec(packed[d]).bitcast(Int32))
                          fx.copy(
                              store_i32,
                              packed_frag,
                              out_i8[None, row_val_off + _off(4 * d)],
                          )
                      scale_off = (
                          row_base_addr
                          + _off(N_OUT)
                          + _off(_udiv(col_g0, fx.Int32(g2_scale_blk)))
                      )
                      scale_frag = fx.make_rmem_tensor(1, Int8)
                      scale_frag.store(Vec.from_elements([e8m0.to(Int8)], Int8))
                      fx.copy(store_i8, scale_frag, out_i8[None, scale_off])

                  _lanes = EPI_NLANE if const_expr(g2_wave_epi) else EPI_LANES
                  _max_col = (n_rg - 1) * route_group_n + (_lanes - 1) * route_vec
                  if const_expr(_max_col < route_span):
                      store_route_group(col_lane8)   # guard provably dead
                  else:
                      @flyc.jit
                      def store_route_group_if_valid(col_lane8, uid):
                          if col_lane8 < fx.Int32(route_span):
                              store_route_group(col_lane8)

                      store_route_group_if_valid(col_lane8, epi_uid)
          else:
              for s in range_constexpr(BN // store_group_n):
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
                      if const_expr(defer_w):
                          pk = Vec.from_elements([v2[0], v2[1]], Float32).to(BFloat16)
                      else:
                          pk = Vec.from_elements(
                              [v2[0] * weight[mr], v2[1] * weight[mr]], Float32
                          ).to(BFloat16)
                  out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                  out_frag.store(pk)
                  out_off = row_base_addr + _off(s * store_group_n)
                  if const_expr(use_reduce):
                      fx.copy(store_bf16x2, out_frag, out_bf16[None, out_off])
                  else:
                      fx.copy(atomic_bf16x2, out_frag, out_bf16[None, out_off])

      for mr in range_constexpr(M_REPS):
          token_id = packed[mr] & fx.Int32(0x00FFFFFF)

          if const_expr(use_reduce and g2_brstore):
              store_one_mr(mr)
          else:
              @flyc.jit
              def store_if_valid(token_id, mr, uid):
                  if token_id < i32_M:
                      store_one_mr(mr)

              store_if_valid(token_id, mr, epi_uid)
