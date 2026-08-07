# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device body (BM32): gemm2 down."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
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

from . import dpp_utils
from .mxfp4_gemm_common import _fabs_f32 as fabs_f32
from .mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from .mxfp4_gemm_common import (
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


def scale_mma_atoms(a_dtype):
    """16 (opselA,opselB) scaled-MFMA atoms; A elem is fp8/fp4, B is fp4."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb
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
):
    """One J-cluster of scaled MFMAs over a 32-row A-scale group (row-groups i0, i0+1); each is
    an fx.gemm on i32 A/B frags (fp8 A = i32<8:1>, fp4 A = i32<4:1>), e8m0 words on scale_a/scale_b.
    sa: 32-row A-scale reg. single_rg (BM16): one 16-row group, rg_off picks its byte.
    """
    row_groups = (rg_off,) if const_expr(single_rg) else range(2)
    for k in range(k_halves):
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
    aStages,
    a_dtype,
    use_reduce=False,
    topk=1,
    has_pad=False,
    SBM=None,
    g2_kstages=2,
    g2_bhoist=True,
    g2_ascale_pf=True,
    g2_bf16_lds=False,
    g2_diag=0,
    g2_kunroll=False,
    g2_epi=0,
    g2_wcpl=0,
    route_out_fp8=False,
    c_lds_off=0,
):
    # g2_diag: PERF-ATTRIBUTION ONLY (wrong results); each bit drops one phase.
    diag_no_barrier = bool(g2_diag & 1)
    diag_no_epilog = bool(g2_diag & 2)
    diag_no_ads = bool(g2_diag & 4)   # kunroll path only: skip the A ds-reads
    diag_no_bld = bool(g2_diag & 8)   # kunroll path only: skip the B global loads
    diag_no_mfma = bool(g2_diag & 16)  # kunroll path only: skip the MFMA cluster
    # gemm2 K-loop perf knobs (default ON, no-op unless g2_kstages==2): kstages=2 double-buffers B weight+scale one tile ahead; bhoist issues that prefetch above the LDS barrier; ascale_pf prefetches A-scale one tile ahead.
    if g2_kstages not in (1, 2):
        raise AssertionError(f"g2_kstages must be 1 or 2, got {g2_kstages}")
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
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = (K_rt + fx.Int32(255)) // fx.Int32(256)
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // fx.Int32(32)) * kBS_stride_n0_dw
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_TILES_MAX = INTER_MAX // BK
    K_SCALE_CHUNKS_MAX = (INTER_MAX + 255) // 256

    # has_pad OOB pad-skip (const_expr-gated): K-skip sizes 16N B-weight buffer to REAL K; N-skip zeros fully-pad-N w2 tiles (col >= N_real=N_OUT-npad; PERF-ONLY). B-scale NOT shrunk.
    bq_num_records = None
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        bq_num_records = halves_real * fx.Int32(1024)
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # block -> (m_block_idx, n_block_idx); e = sorted_expert_ids[SBM-padded sort block] (SBM==BM: sort_block==m_block_idx).
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    # The f32 C slab normally unions the A-tile LDS region.  Under the N-loop schedule
    # A must survive every n-tile's epilogue, so the caller passes a disjoint offset.
    lds_acc_base = lds_base_i32 + fx.Int32(c_lds_off) if c_lds_off else lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

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

    def issue_a_ds_read(slot, frags=None):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        if frags is None:
            frags = a_frags
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
                    frags[i][k].store(a64.bitcast(fx.Int32))
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
                    frags[i][k].store(Vec(vec))

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
            else kt // fx.Int32(tilesPerScaleChunk)
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
        if const_expr(has_pad):
            # N-skip: fully-pad-N tile (col >= 16-aligned N_real) -> 0 records so weight loads OOB -> 0.
            nrec = (col < N_real).select(bq_num_records, fx.Int32(0))
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

    frag_tmpl = bq_views[0][0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                fx.copy(
                    b_catom,
                    bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
                    bqf[j][half],
                )
        chunk_kt = scale_chunk_tile(kt_rt)
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def make_bq_fragments():
        return [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]

    def make_scale_fragments(count):
        return [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(count)]

    def stream_b_tile(kt_rt):
        bqf = make_bq_fragments()
        bsf = make_scale_fragments(nPairs)
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return scale.shrui(scale_shift)

    def mfma_cluster(bqf, bsf, sa, kt_rt, afrg=None):
        if afrg is None:
            afrg = a_frags
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
            if const_expr(is_bm16):
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    afrg,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    k_halves=kHalves,
                )
                continue
            for sub in range_constexpr(kScaleSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    afrg,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
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

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    if const_expr(g2_kunroll):
        # Fully-unrolled K (inter_dim == INTER_MAX, K_TILES compile-time).  All
        # K_TILES A-tiles were DMA'd to their own LDS slot by the prologue, so the
        # whole contraction needs ONE barrier and carries no scf.for state.
        KT = INTER_MAX // BK
        # g2_kunroll==2: reuse ONE register set across all K-tiles (A frags and the
        # B double-buffer) instead of KT private sets, trading some ds_read/MFMA
        # overlap for ~128 VGPRs of occupancy headroom.
        nA = 1 if g2_kunroll == 2 else KT
        nB = 2 if g2_kunroll == 2 else KT
        a_sets = [
            [
                [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
                for _ in range_constexpr(kMChunks)
            ]
            for _ in range_constexpr(nA)
        ]
        bqfs = [make_bq_fragments() for _ in range_constexpr(nB)]
        bsfs = [make_scale_fragments(nPairs) for _ in range_constexpr(nB)]
        safs = [make_scale_fragments(kScaleSubBlocks) for _ in range_constexpr(nB)]

        def load_a_scale_into(saf, kt_ct):
            sa = load_a_scale_tile(fx.Int32(kt_ct))
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(Vec.from_elements([sa[sub]], Int32))

        # Issue tile 0's B before the barrier so its vmem latency hides behind the
        # A-DMA wait that the barrier resolves.
        if const_expr(not diag_no_bld):
            issue_b_load_into(bqfs[0], bsfs[0], fx.Int32(0))
        load_a_scale_into(safs[0], 0)
        if const_expr(not diag_no_barrier):
            gpu.barrier()
        for kt_ct in range_constexpr(KT):
            cur_b, nxt_b = kt_ct % nB, (kt_ct + 1) % nB
            if const_expr(kt_ct + 1 < KT):
                if const_expr(not diag_no_bld):
                    issue_b_load_into(
                        bqfs[nxt_b], bsfs[nxt_b], fx.Int32(kt_ct + 1)
                    )
                load_a_scale_into(safs[nxt_b], kt_ct + 1)
            if const_expr(not diag_no_ads):
                issue_a_ds_read(kt_ct, frags=a_sets[kt_ct % nA])
            sa = [
                Vec(safs[cur_b][sub].load())[0]
                for sub in range_constexpr(kScaleSubBlocks)
            ]
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            if const_expr(not diag_no_mfma):
                mfma_cluster(
                    bqfs[cur_b],
                    bsfs[cur_b],
                    sa,
                    fx.Int32(kt_ct),
                    afrg=a_sets[kt_ct % nA],
                )
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
    elif const_expr(g2_kstages == 1):
        # 1-deep pipe: synchronous B load per K-tile.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(not diag_no_barrier):
                gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt = kt_rt + fx.Int32(kStages)
            if nxt < K_TILES_RT:
                issue_a_load_lds(nxt % fx.Int32(aStages), nxt)
            bqf, bsf = stream_b_tile(kt_rt)
            sa = load_a_scale_tile(kt_rt)
            mfma_cluster(bqf, bsf, sa, kt_rt)
            results = yield load_c_carry()
        store_c_carry(results)
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
            return load_c_carry() + load_b_fragments(cur_bqf, cur_bsf, cur_saf)

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
            if const_expr(not diag_no_barrier):
                gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt_a = kt_rt + fx.Int32(kStages)
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
            # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
    ]
    if const_expr(diag_no_epilog):
        # Attribution build: one dependent store keeps the whole MFMA chain live.
        acc = fx.Float32(0.0)
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                v = Vec(accm_vecs[i][J])
                for q in range_constexpr(4):
                    acc = acc + fx.Float32(v[q])
        out_i8 = global_typed_ptr(arg_out, T.i8)
        if acc > fx.Float32(1.0e30):
            out_i8[lane] = fx.Int8(1)
        return
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
        g2_epi=g2_epi,
        g2_wcpl=g2_wcpl,
        g2_diag=g2_diag,
        route_out_fp8=route_out_fp8,
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
    g2_epi=0,
    g2_wcpl=0,
    g2_diag=0,
    route_out_fp8=False,
):
    # Epilogue attribution bits (WRONG results): 64 = skip the C-slab cshuffle write,
    # 128 = skip the readback/quantise/store loop, 256 = skip the stids/sweights
    # loads, 512 = skip the e8m0 scale stores.
    diag_no_cwrite = bool(g2_diag & 64)
    diag_no_cread = bool(g2_diag & 128)
    diag_no_meta = bool(g2_diag & 256)
    diag_no_scale = bool(g2_diag & 512)
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )
    # C-slab XOR swizzle (g2_epi>=1).  Row stride BN is a multiple of 32 banks, so the
    # four lane_div_16 write groups (4 rows apart) all land on the same banks -> 4-way
    # conflict on every cshuffle ds_write.  XOR the column by 32 bytes per 4-row group
    # to spread them over banks 0/8/16/24.  32 bytes is >= every readback run, so
    # contiguous runs survive the XOR intact and stay addressable.
    C_ELEM_BYTES = 2 if g2_bf16_lds else 4
    SWZ = (32 // C_ELEM_BYTES) if g2_epi >= 1 else 0
    # g2_epi>=2: stage the cshuffle in CSPLIT row-slices instead of the whole BM x BN
    # slab at once.  The slab is the LDS high-water mark, so splitting it buys
    # workgroups/CU.  Purely a staging change: the same values are written and read
    # back in the same order, so it is bit-exact (unlike g2_bf16_lds).
    _want_split = 4 if (g2_epi & 4) else (2 if (g2_epi & 2) else 1)
    CSPLIT = _want_split
    while CSPLIT > 1 and (kMChunks % CSPLIT or M_REPS % CSPLIT):
        CSPLIT //= 2
    C_CHUNKS = kMChunks // CSPLIT  # cshuffle-write chunks per slice
    C_MREPS = M_REPS // CSPLIT  # readback rows per thread per slice
    C_SLICE_ROWS = BM // CSPLIT

    # g2_epi bit 5: bank-conflict swizzle for the slab READBACK.  cswz's row_grp4 XOR
    # only permutes within a 16-element block, so the 16 lanes of a readback row-group
    # (columns 16 apart) start in just 2 of the 8 bank quads -> 8-way conflict.  XOR
    # bits 2-4 of the column with bits 5-7 of the same column: bits 5-7 survive, so it
    # is an involution for every fixed row_grp4, and it moves the 16 lanes to 2 per
    # quad -- the 32-bank minimum for a 64-lane b128 read.  Granularity drops from 8
    # elements to 4, so readbacks swizzle each 4-float chunk instead of adding 4 to a
    # swizzled base.
    CSWZ4 = bool(g2_epi & 32) and SWZ != 0 and C_ELEM_BYTES == 4
    # ...and the WRITE side needs a wider row XOR.  LDS retires a b32 wave access in
    # two 32-lane halves, so lane groups 0/1 must cover 32 distinct banks; a 32-byte
    # XOR leaves both on the same 16.  A 64-byte XOR on the row-group parity puts them
    # on opposite halves (groups 2/3 reuse those banks, but are the other half-wave).
    SWZ_R = (64 // C_ELEM_BYTES) if CSWZ4 else SWZ

    def cswz(col, row_grp4):
        # row_grp4 = (row >> 2) & 3, the write-group / readback-row selector.
        if const_expr(SWZ == 0):
            return col
        if const_expr(CSWZ4):
            col = col ^ ((row_grp4 & fx.Int32(1)) * fx.Int32(SWZ_R))
            return col ^ (((col >> fx.Int32(5)) & fx.Int32(7)) << fx.Int32(2))
        return col ^ (row_grp4 * fx.Int32(SWZ))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    store_vec = 2
    store_group_n = 32 * store_vec
    col_start = n_lane * store_vec
    wave_n = BN // 4

    # g2_epi bit 3: "wide" route-out store.  The default readback gives each lane 8
    # output columns of 8 different rows, so per thread the epilogue issues 8 dwordx2
    # value stores + 8 single-byte e8m0 scale stores and redoes the i64 row-address
    # math and the token_id<M padding guard eight times.  Give each lane one whole
    # row of the current C slice instead (BN/W_NL contiguous columns): the values go
    # out as dwordx4, the lane's W_CPL/8 scale bytes are contiguous (one dword/short),
    # and the row metadata is computed once per slice.  Same 8-element scale groups,
    # same multiply order -> bit-identical output.
    WIDE = bool(g2_epi & 8) and use_reduce and route_out_fp8 and not g2_bf16_lds
    # Columns per lane.  The natural choice keeps all 256 threads busy
    # (C_SLICE_ROWS * BN / 256), but that can leave W_CPL/8 == 1 e8m0 byte per lane,
    # and single-byte global stores dominate the epilogue.  g2_wcpl forces a wider
    # slice so the scales go out as i16/i32, at the price of idling some threads.
    W_CPL = (BN * C_SLICE_ROWS) // 256 if WIDE else 0
    if WIDE and g2_wcpl:
        W_CPL = max(W_CPL, g2_wcpl)
    W_CPL = min(W_CPL, BN) if WIDE else 0
    if WIDE and (
        W_CPL < 8
        or W_CPL % 8
        or BN % W_CPL
        or (W_CPL // 8) not in (1, 2, 4)
        or C_SLICE_ROWS % 4
        or (C_SLICE_ROWS * (BN // W_CPL)) > 256
    ):
        WIDE = False
    W_NL = (BN // W_CPL) if WIDE else 0
    W_NS = (W_CPL // 8) if WIDE else 0
    W_ACTIVE = (C_SLICE_ROWS * W_NL) if WIDE else 0
    # g2_epi bit 4: DPP-combine the e8m0 bytes of W_NPACK adjacent lanes into one
    # dword before storing.  Those lanes hold the same output row and consecutive
    # scale indices, so the merge is a pure quad_perm shuffle.  Unlike widening the
    # per-lane column slice, this drops the sub-dword stores with no idle threads.
    W_NPACK = (4 // W_NS) if (WIDE and (g2_epi & 16)) else 1
    if WIDE and (W_NPACK > 1) and (W_NL % W_NPACK or W_NL < W_NPACK):
        W_NPACK = 1
    if const_expr(WIDE):
        w_mlane = tx_i32 // fx.Int32(W_NL)
        w_nlane = tx_i32 - w_mlane * fx.Int32(W_NL)
        # Rows within a wave must stride by 4 so that the 4-row cswz groups differ and
        # the readback keeps the bank spread the 8-col mapping had (a 4 x C_SLICE_ROWS/4
        # transpose; bijective on [0, C_SLICE_ROWS)).
        w_m4 = w_mlane * fx.Int32(4)
        w_row_local = (w_m4 % fx.Int32(C_SLICE_ROWS)) + (
            w_m4 // fx.Int32(C_SLICE_ROWS)
        )
        w_col_base = w_nlane * fx.Int32(W_CPL)

    def flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = flat_buffer(arg_stids, T.i32, 4)
    sweights = flat_buffer(arg_sweights, T.f32, 4)
    out_bf16 = flat_buffer(arg_out, T.bf16, 4)
    out_i8 = flat_buffer(arg_out, T.i8, 4)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
    store_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(2), BFloat16)
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)
    store_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(2), Int32)
    store_i32x2 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(2), Int32)
    store_i8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(2), Int8)
    store_i32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2), Int32)
    store_i16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(2), Int16)

    def issue_scalar(atom, src, index, elem_ty):
        # Issue only: the register read is deferred to collect_scalar so that ONE
        # s_waitcnt can cover the whole batch instead of one stall per load.
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return frag

    def collect_scalar(frag):
        return Vec(frag.load())[0]

    def load_scalar(atom, src, index, elem_ty):
        return collect_scalar(issue_scalar(atom, src, index, elem_ty))

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    # Issue every load back to back and only then read the destination registers,
    # otherwise each `frag.load()` forces its own waitcnt and the batch costs one
    # serialised L2 round-trip per row right before the barrier.
    packed = []
    weight = []
    _pf = []
    # WIDE needs one (token_id, weight) pair per C slice, not one per M_REPS row.
    _meta_reps = range_constexpr(CSPLIT) if const_expr(WIDE) else range_constexpr(M_REPS)
    # NOTE: these loads are invariant across the n-tiles of an MXFP4_G2_NLOOP group,
    # but caching them across tiles measures slower -- the extra values live across
    # the next n-tile's whole K loop, and the K loop is more sensitive to register
    # pressure than to a handful of L2-resident scalar loads.
    for mr in _meta_reps:
        if const_expr(WIDE):
            sorted_pos = m_row + fx.Int32(mr * C_SLICE_ROWS) + w_row_local
        else:
            sorted_pos = m_row + mr * 8 + m_lane
        if const_expr(diag_no_meta):
            _pf.append(None)
        else:
            _pf.append(
                (
                    issue_scalar(load_i32, stids, sorted_pos, Int32),
                    issue_scalar(load_f32, sweights, sorted_pos, Float32),
                )
            )

    # bf16-LDS path bakes the routing weight in at cshuffle-write time; prefetch all
    # kMChunks*4 of those sweights rows ABOVE the barrier so the vmem latency overlaps
    # the barrier + the tail of the K loop instead of stalling each chunk in turn.
    w_rows = None
    _wpf = None
    if const_expr(g2_bf16_lds and diag_no_meta):
        w_rows = [
            [fx.Float32(1.0) for _ in range_constexpr(4)]
            for _ in range_constexpr(kMChunks)
        ]
    elif const_expr(g2_bf16_lds):
        _wpf = [
            [
                issue_scalar(
                    load_f32,
                    sweights,
                    m_row + fx.Int32(i * 16) + lane_div_16 * 4 + v,
                    Float32,
                )
                for v in range_constexpr(4)
            ]
            for i in range_constexpr(kMChunks)
        ]

    # All meta loads are in flight; now read the destination registers.
    for mr in _meta_reps:
        if const_expr(diag_no_meta):
            packed.append(fx.Int32(0))
            weight.append(fx.Float32(1.0))
        else:
            packed.append(collect_scalar(_pf[mr][0]))
            weight.append(collect_scalar(_pf[mr][1]))

    if _wpf is not None:
        w_rows = [
            [collect_scalar(_wpf[i][v]) for v in range_constexpr(4)]
            for i in range_constexpr(kMChunks)
        ]

    def cshuffle_write(sp):
        # accm -> lds_acc for row slice `sp`.  f32 path stores raw accumulators (the
        # routing weight is applied on readback); bf16 path bakes the weight in.
        if const_expr(diag_no_cwrite):
            return
        for i in range_constexpr(sp * C_CHUNKS, (sp + 1) * C_CHUNKS):
            # LDS row is slice-relative; the global row is i*16 + ... as before.
            row_base = fx.Int32((i - sp * C_CHUNKS) * 16) + lane_div_16 * 4
            for J in range_constexpr(numAccN):
                col = wave * wave_n + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                col_s = cswz(col, lane_div_16)
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col_s
                    if const_expr(g2_bf16_lds):
                        lds_base_bf16[idx] = fx.BFloat16(
                            fx.Float32(vec[v]) * fx.Float32(w_rows[i][v])
                        )
                    else:
                        lds_base_fptr[idx] = fx.Float32(vec[v])

    # read back + weighted store (atomic: fadd out[token_id]; reduce: store out[token_id*topk+slot]);
    # token_id<i32_M gates padding at runtime. At small/mid M the block-sparse sort floor is
    # ~97% padding rows (M64: 12288 pad vs 384 real); ungated, every padding row (token_id==M) issues
    # a weighted atomic-fadd into an OOB out-row -> 77x wasted atomics + L2 RMW serialization
    # (rocprof M64: TCC_ATOMIC 7.3M->95K, 932us->107us). Always gate; reduce already gated via
    # use_reduce. (fp4-atomic families are gated too -> strictly correct OOB-skip, kernel IR changes.)

    def _quant8(vals):
        """8 weighted f32 -> (lo_i32, hi_i32, e8m0), as in store_one_mr's route path."""
        local_max = fabs_f32(vals[0])
        for q in range_constexpr(1, 8):
            local_max = local_max.maximumf(fabs_f32(vals[q]))
        amax_bits = fx.Int32(_raw(local_max).bitcast(T.i32))
        ax_e = (amax_bits >> fx.Int32(23)) & fx.Int32(0xFF)
        e8m0 = ax_e - fx.Int32(7)
        e8m0 = (e8m0 < fx.Int32(1)).select(fx.Int32(1), e8m0)
        e8m0 = (amax_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
        block_scale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
        bs_raw = _raw(block_scale)
        pk_ty = T.vec(2, T.i16)
        lo = _raw(Vec.filled([2], 0, fx.Int16))
        lo = rocdl.cvt_scalef32_pk_fp8_f32(
            pk_ty, lo, _raw(vals[0]), _raw(vals[1]), bs_raw, 0
        )
        lo = rocdl.cvt_scalef32_pk_fp8_f32(
            pk_ty, lo, _raw(vals[2]), _raw(vals[3]), bs_raw, 1
        )
        hi = _raw(Vec.filled([2], 0, fx.Int16))
        hi = rocdl.cvt_scalef32_pk_fp8_f32(
            pk_ty, hi, _raw(vals[4]), _raw(vals[5]), bs_raw, 0
        )
        hi = rocdl.cvt_scalef32_pk_fp8_f32(
            pk_ty, hi, _raw(vals[6]), _raw(vals[7]), bs_raw, 1
        )
        return (
            Vec(Vec(lo).bitcast(Int32))[0],
            Vec(Vec(hi).bitcast(Int32))[0],
            e8m0,
        )

    def store_wide(sp):
        # One C-slice row per lane, W_CPL contiguous output columns.
        grow = fx.Int32(sp * C_SLICE_ROWS) + w_row_local
        row_grp4 = (grow >> fx.Int32(2)) & fx.Int32(3)
        pk = packed[sp]
        wt = weight[sp]
        out_row = fx.Int64(
            (pk & fx.Int32(0x00FFFFFF)) * fx.Int32(topk) + (pk >> fx.Int32(24))
        )
        row_base_addr = out_row * fx.Int64(N_OUT + (N_OUT // fx.Int32(8)))
        col_g0_base = n_block_idx * BN + w_col_base
        words = []
        scales = []
        for g in range_constexpr(W_NS):
            row_e = w_row_local * BN
            vals = []
            for h in range_constexpr(2):
                base_e = row_e + cswz(
                    w_col_base + fx.Int32(g * 8 + h * 4), row_grp4
                )
                v4 = Vec(
                    lds_vec_load(
                        lds_acc_base,
                        base_e * 4,
                        Vec.make_type(4, Float32),
                        Float32,
                        align=16,
                    )
                )
                for q in range_constexpr(4):
                    vals.append(fx.Float32(v4[q]) * wt)
            lo, hi, e8m0 = _quant8(vals)
            words.append(lo)
            words.append(hi)
            scales.append(e8m0)
        # W_CPL values = 2*W_NS dwords: dwordx4 while four remain, then a dwordx2 for
        # the odd pair (W_NS==1, i.e. BN/W_NL == 8).  row_base_addr (4032 B/row) and
        # col_g0_base are both 16 B aligned, so every store is naturally aligned.
        _nw = len(words)
        for w in range_constexpr(_nw // 4):
            f4 = fx.make_rmem_tensor(4, Int32)
            f4.store(Vec.from_elements(words[w * 4 : w * 4 + 4], Int32))
            fx.copy(
                store_i32x4,
                f4,
                out_i8[
                    None,
                    row_base_addr + fx.Int64(col_g0_base + fx.Int32(w * 16)),
                ],
            )
        if const_expr(_nw % 4 == 2):
            f2 = fx.make_rmem_tensor(2, Int32)
            f2.store(Vec.from_elements(words[_nw - 2 :], Int32))
            fx.copy(
                store_i32x2,
                f2,
                out_i8[
                    None,
                    row_base_addr
                    + fx.Int64(col_g0_base + fx.Int32((_nw // 4) * 16)),
                ],
            )
        scale_off = (
            row_base_addr
            + fx.Int64(N_OUT)
            + fx.Int64(col_g0_base // fx.Int32(8))
        )
        acc = scales[0] & fx.Int32(0xFF)
        for q in range_constexpr(1, W_NS):
            acc = acc | ((scales[q] & fx.Int32(0xFF)) << fx.Int32(8 * q))
        if const_expr(diag_no_scale):
            pass
        elif const_expr(W_NPACK > 1):
            # quad_perm broadcasts: [i,i,i,i] for a 4-lane merge, [0,0,2,2]/[1,1,3,3]
            # for a 2-lane one.  Every lane ends up with the full dword; only the
            # group leader stores it, at its own (lowest) scale offset.
            ctrls = (
                (0x00, 0x55, 0xAA, 0xFF) if W_NPACK == 4 else (0xA0, 0xF5)
            )
            merged = None
            for j in range_constexpr(W_NPACK):
                part = fx.Int32(
                    dpp_utils.update_dpp_i32(
                        _raw(acc), _raw(acc), ctrls[j], 0xF, 0xF, True
                    )
                )
                part = part << fx.Int32(8 * W_NS * j)
                merged = part if merged is None else (merged | part)
            sf = fx.make_rmem_tensor(1, Int32)
            sf.store(Vec.from_elements([merged], Int32))

            @flyc.jit
            def store_scale_leader(sf, scale_off, tx_i32):
                if (tx_i32 & fx.Int32(W_NPACK - 1)) == fx.Int32(0):
                    fx.copy(store_i32, sf, out_i8[None, scale_off])

            store_scale_leader(sf, scale_off, tx_i32)
        elif const_expr(W_NS == 4):
            sf = fx.make_rmem_tensor(1, Int32)
            sf.store(Vec.from_elements([acc], Int32))
            fx.copy(store_i32, sf, out_i8[None, scale_off])
        elif const_expr(W_NS == 2):
            sf = fx.make_rmem_tensor(1, Int16)
            sf.store(Vec.from_elements([acc.to(Int16)], Int16))
            fx.copy(store_i16, sf, out_i8[None, scale_off])
        else:
            sf = fx.make_rmem_tensor(1, Int8)
            sf.store(Vec.from_elements([acc.to(Int8)], Int8))
            fx.copy(store_i8, sf, out_i8[None, scale_off])

    def store_one_mr(mr, sp=0):
        # Global row within the BM tile drives the swizzle selector (it must match the
        # write side, which indexes by the global chunk); the LDS row is slice-relative.
        row_in_block = fx.Int32((mr - sp * C_MREPS) * 8) + m_lane
        row_grp4 = (
            (fx.Int32(mr * 8) + m_lane) >> fx.Int32(2)
        ) & fx.Int32(3)
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(use_reduce):
            # reduce out_row can reach tokens*topk (large-M) so compute the element base in i64 (atomic i32 path byte-identical).
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            if const_expr(route_out_fp8):
                row_base_addr = out_row * fx.Int64(N_OUT + (N_OUT // fx.Int32(8)))
            else:
                row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(
                    n_block_idx * BN + col_start
                )
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        if const_expr(use_reduce and route_out_fp8):
            route_vec = 8
            route_group_n = 32 * route_vec
            for rg in range_constexpr((BN + route_group_n - 1) // route_group_n):
                col_lane8 = rg * route_group_n + n_lane * fx.Int32(route_vec)

                def store_route_group(col_lane8):
                    col_g0 = n_block_idx * BN + col_lane8
                    vals = []
                    if const_expr(g2_epi >= 1):
                        # route_vec cshuffle slots are contiguous in the C slab (the
                        # 32-byte XOR swizzle preserves 8-element runs), so pull them
                        # in as b128 vector ds_reads instead of route_vec scalar ones.
                        base_e = row_in_block * BN + cswz(col_lane8, row_grp4)
                        row_e = row_in_block * BN
                        if const_expr(g2_bf16_lds):
                            v8 = Vec(
                                lds_vec_load(
                                    lds_acc_base,
                                    base_e * 2,
                                    Vec.make_type(route_vec, BFloat16),
                                    BFloat16,
                                    align=16,
                                )
                            )
                            for q in range_constexpr(route_vec):
                                vals.append(fx.Float32(v8[q]))
                        else:
                            for h in range_constexpr(route_vec // 4):
                                e_h = row_e + cswz(
                                    col_lane8 + fx.Int32(h * 4), row_grp4
                                )
                                v4 = Vec(
                                    lds_vec_load(
                                        lds_acc_base,
                                        e_h * 4,
                                        Vec.make_type(4, Float32),
                                        Float32,
                                        align=16,
                                    )
                                )
                                for q in range_constexpr(4):
                                    vals.append(fx.Float32(v4[q]) * weight[mr])
                    else:
                        for q in range_constexpr(route_vec):
                            idx_q = row_in_block * BN + col_lane8 + fx.Int32(q)
                            if const_expr(g2_bf16_lds):
                                # bf16 LDS already has the routing weight baked in.
                                vals.append(fx.Float32(lds_base_bf16[idx_q]))
                            else:
                                vals.append(
                                    fx.Float32(lds_base_fptr[idx_q]) * weight[mr]
                                )
                    local_max = fabs_f32(vals[0])
                    for q in range_constexpr(1, route_vec):
                        local_max = local_max.maximumf(fabs_f32(vals[q]))
                    amax_bits = fx.Int32(_raw(local_max).bitcast(T.i32))
                    ax_e = (amax_bits >> fx.Int32(23)) & fx.Int32(0xFF)
                    e8m0 = ax_e - fx.Int32(7)
                    e8m0 = (e8m0 < fx.Int32(1)).select(fx.Int32(1), e8m0)
                    e8m0 = (amax_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
                    block_scale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
                    bs_raw = _raw(block_scale)
                    pk_ty = T.vec(2, T.i16)
                    packed_lo = _raw(Vec.filled([2], 0, fx.Int16))
                    packed_lo = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty, packed_lo, _raw(vals[0]), _raw(vals[1]), bs_raw, 0
                    )
                    packed_lo = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty, packed_lo, _raw(vals[2]), _raw(vals[3]), bs_raw, 1
                    )
                    packed_hi = _raw(Vec.filled([2], 0, fx.Int16))
                    packed_hi = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty, packed_hi, _raw(vals[4]), _raw(vals[5]), bs_raw, 0
                    )
                    packed_hi = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty, packed_hi, _raw(vals[6]), _raw(vals[7]), bs_raw, 1
                    )
                    row_val_off = row_base_addr + fx.Int64(col_g0)
                    if const_expr(g2_epi >= 1):
                        # The two fp8 dwords are adjacent (col_g0 is 8-aligned) -> one
                        # dwordx2 store instead of two dwordx1.
                        pf2 = fx.make_rmem_tensor(2, Int32)
                        pf2.store(
                            Vec.from_elements(
                                [
                                    Vec(Vec(packed_lo).bitcast(Int32))[0],
                                    Vec(Vec(packed_hi).bitcast(Int32))[0],
                                ],
                                Int32,
                            )
                        )
                        fx.copy(store_i32x2, pf2, out_i8[None, row_val_off])
                    else:
                        packed_frag = fx.make_rmem_tensor(1, Int32)
                        packed_frag.store(Vec(packed_lo).bitcast(Int32))
                        fx.copy(store_i32, packed_frag, out_i8[None, row_val_off])
                        packed_frag.store(Vec(packed_hi).bitcast(Int32))
                        fx.copy(
                            store_i32,
                            packed_frag,
                            out_i8[None, row_val_off + fx.Int64(4)],
                        )
                    scale_off = (
                        row_base_addr
                        + fx.Int64(N_OUT)
                        + fx.Int64(col_g0 // fx.Int32(route_vec))
                    )
                    if const_expr(not diag_no_scale):
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
                idx0 = row_in_block * BN + cswz(
                    col_start + fx.Int32(s * store_group_n), row_grp4
                )
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
                    pk = Vec.from_elements(
                        [v2[0] * weight[mr], v2[1] * weight[mr]], Float32
                    ).to(BFloat16)
                out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                out_frag.store(pk)
                out_off = row_base_addr + fx.Int64(s * store_group_n)
                if const_expr(use_reduce):
                    fx.copy(store_bf16x2, out_frag, out_bf16[None, out_off])
                else:
                    fx.copy(atomic_bf16x2, out_frag, out_bf16[None, out_off])

    for sp in range_constexpr(CSPLIT):
        # Barrier before each slice: for sp>0 it also fences the previous slice's
        # readback against this slice's overwrite of the same LDS bytes.  The sp==0
        # barrier is needed too: when the C slab unions the A LDS region
        # (c_lds_off == 0), the first slice's write otherwise races other waves'
        # last-K-tile A ds_reads and the route-out differs run to run.
        gpu.barrier()
        cshuffle_write(sp)
        gpu.barrier()
        if const_expr(diag_no_cread):
            continue
        if const_expr(WIDE):
            token_id = packed[sp] & fx.Int32(0x00FFFFFF)

            @flyc.jit
            def store_wide_if_valid(token_id, sp, tx_i32):
                # tx >= W_ACTIVE only when g2_wcpl widened the lane slice past the
                # thread count; those lanes have no row and must not store.
                if token_id < i32_M:
                    if tx_i32 < fx.Int32(W_ACTIVE):
                        store_wide(sp)

            store_wide_if_valid(token_id, sp, tx_i32)
            continue
        for mr in range_constexpr(sp * C_MREPS, (sp + 1) * C_MREPS):
            token_id = packed[mr] & fx.Int32(0x00FFFFFF)

            @flyc.jit
            def store_if_valid(token_id, mr, sp):
                if token_id < i32_M:
                    store_one_mr(mr, sp)

            store_if_valid(token_id, mr, sp)
