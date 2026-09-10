# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 K-loop instruction scheduling (layout API)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float32, Int32, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from .mxfp4_gemm_common import (
    _udiv,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
)
from .mxmoe_g2_atoms import (
    bq_view,
    bq_view_fp8,
    issue_a_ds_read_slot,
    issue_a_load_lds_dt,
    issue_b_load_into as _issue_b_load_into,
    issue_bscale_into as _issue_bscale_into,
    load_a_scale_tile as _load_a_scale_tile,
    make_b_copy_atom,
    make_bq_fragments as _make_bq_fragments,
    make_scale_copy_atom,
    make_scale_fragments as _make_scale_fragments,
    mma_one_j,
    scale_chunk_tile as _scale_chunk_tile,
    scale_mma_atoms,
    scale_view,
    shift_scale_word as _shift_scale_word,
)
from .mxmoe_g2_epilog import atomic_bf16_epilog, nonatomic_bf16_epilog


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
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    D_INTER,
    D_HIDDEN,
    g2_kstatic=False,
    aStages,
    a_slot_alias=False,
    a_dtype,
    b_dtype,
    use_reduce=False,
    topk=1,
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
    nonatomic=False,
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
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K / output N are compile-time D_INTER / D_HIDDEN.
    K_BYTES = D_INTER // a_pack
    K_TILES = D_INTER // BK
    K_SCALE_CHUNKS = (D_INTER + 255) // 256
    kAS_per_chunk_dw = K_SCALE_CHUNKS * 64
    kBS_stride_n0_dw = K_SCALE_CHUNKS * 64
    N_OUT = D_HIDDEN
    num_n_blocks = D_HIDDEN // BN
    kbs_per_expert_dw = (D_HIDDEN // 32) * kBS_stride_n0_dw
    KH4 = D_INTER // (4 if is_f8_b else 8)

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
        issue_a_ds_read_slot(
            a_frags,
            s_aq_base,
            slot,
            slot_bytes,
            KH_TILE_A,
            kHalves,
            kMChunks,
            lane_mod_16,
            lane_div_16,
            is_f8_a,
        )

    # Scale words (e8m0): shared scale_view / copy atom for both A and B. A-scale is one
    # word per 32-row chunk, each view bounded to bytes remaining after its baked base.
    sc_copy_atom = make_scale_copy_atom()

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * fx.Int64(4)
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def scale_chunk_tile(kt):
        return _scale_chunk_tile(kt, tilesPerScaleChunk)

    def load_a_scale_tile(kt):
        return _load_a_scale_tile(
            kt,
            sc_copy_atom,
            ascale_views,
            sc_frag_tmpl,
            kScaleSubBlocks,
            lane_div_16,
            lane_mod_16,
            tilesPerScaleChunk,
        )

    # B-weight + B-scale: global->register, streamed per K-tile (not LDS-staged).
    # b128 weight copy atom; cache modifier 2=nontemporal, 0=default.
    b_catom = make_b_copy_atom(use_nt)

    def make_bq_view(j):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        if const_expr(is_f8_b):
            return bq_view_fp8(
                arg_bq,
                e * N_OUT + col,
                KH4,
                K_TILES,
                kHalves,
            )
        return bq_view(
            arg_bq,
            e * N_OUT + col,
            KH4,
            K_TILES,
            kHalves,
        )

    bq_views = [make_bq_view(j) for j in range_constexpr(numAccN)]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS,
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

    def issue_bscale_into(bsf, chunk_kt):
        _issue_bscale_into(
            bsf,
            chunk_kt,
            sc_copy_atom,
            bscale_views,
            nPairs,
            lane_div_16,
            lane_mod_16,
        )

    def issue_b_load_into(bqf, bsf, kt_rt):
        _issue_b_load_into(
            bqf,
            bsf,
            kt_rt,
            b_catom,
            bq_views,
            sc_copy_atom,
            bscale_views,
            numAccN,
            kHalves,
            nPairs,
            lane_div_16,
            lane_mod_16,
            is_f8_b,
            B_NDW,
            tilesPerScaleChunk,
        )

    def make_bq_fragments():
        return _make_bq_fragments(is_f8_b, B_NDW, kHalves, numAccN, frag_tmpl)

    def make_scale_fragments(count):
        return _make_scale_fragments(count, sc_frag_tmpl)

    def shift_scale_word(scale, kt_rt):
        return _shift_scale_word(scale, kt_rt, tilesPerScaleChunk)

    def mfma_cluster(bqf, bsf, sa, kt_rt, interleave=None):
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
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    k_halves=kHalves,
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
                        k_halves=kHalves,
                    )
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
            N_OUT,
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

    g2_interleave = const_expr(g2_kstatic and g2_bf16_lds and not nonatomic)
    epi_thunks = [] if const_expr(g2_interleave) else None
    if const_expr(g2_interleave):
        _epilog(c_frags, emit_thunks=epi_thunks)

    if const_expr(g2_kstatic):
        KT = K_TILES
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
            issue_b_load_into(nxt_bqf, None, fx.Int32(kt))
            if const_expr(kt == 0 or chunk_of[kt] != chunk_of[kt - 1]):
                _ks_issue_scales(kt)

        issue_b_load_into(cur_bqf, None, fx.Int32(0))
        _ks_issue_scales(0)
        rocdl.sched_barrier(0)

        a_all_resident = const_expr((aStages if g2_apre else kStages) >= KT)
        if const_expr(a_all_resident):
            gpu.barrier()

        for kt in range_constexpr(KT):
            kt_rt = fx.Int32(kt)
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
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt, interleave=_il)
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
            if nxt_b < K_TILES:
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
            fx.Int32(K_TILES),
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
            if nxt_a < K_TILES:
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

    if const_expr(nonatomic):
        nonatomic_bf16_epilog(
            [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)],
            arg_out,
            m_row,
            n_block_idx,
            wave,
            lane,
            N_OUT,
            BN,
            kMChunks,
        )
    elif const_expr(g2_interleave):
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        _epilog(None, lds_ready=True)
    else:
        _epilog(
            [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)]
        )
