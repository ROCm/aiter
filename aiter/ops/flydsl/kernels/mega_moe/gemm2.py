# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 compute for fused MegaMoE v2 stage2."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import _to_raw as _raw
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from ..mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from ..mxfp4_gemm_common import (
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
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
    """View one preshuffled B N-tile as i32<4:1>, optionally bounded to real K."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane 64, nlane 4, K-tile K_HALVES*256, half 256, kpack4 1.
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


def scale_view(arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None):
    """View one e8m0 scale word, optionally bounded to the real buffer extent."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
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
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb)
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
    """Run one scaled-MFMA cluster over a 32-row A-scale group."""
    if const_expr(single_rg):
        steps = tuple((2 * k + rg_off, k, i0, bq_frags_kt[J][k]) for k in range(k_halves))
    else:
        steps = tuple((2 * k + im, k, i0 + im, bq_frags_kt[J][k]) for k in range(k_halves) for im in range(2))
    for osa, k, i, bJ in steps:
        osb = 2 * k + in_b
        fx.gemm(
            atoms[(osa, osb)],
            c_frags[i][J],
            a_frags[i][k],
            bJ,
            c_frags[i][J],
            scale_a=sa,
            scale_b=sb,
        )


def issue_a_load_lds_dt(
    arg_aq,
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
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI."""
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
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    tile_base = fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES)
    src = flat_buffer_view(
        tile_base,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(BM) * fx.Int64(K_BYTES),
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row) if const_expr(is_f8) else lds_swizzle_mask(lds_row + a_lane_row)
        )
        tile_row = lds_row + a_lane_row
        voffset = (lane_col ^ mask) + tile_row * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16))


@flyc.jit
def gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
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
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    expert_offset=0,
):
    """Run the GEMM2 K-loop and return accumulators for the selected epilogue."""
    # SBM is the sort padding unit; BM is the compute tile and must divide SBM.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = max(1, kMChunks // 2)
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // fx.Int32(32)) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_TILES_MAX = INTER_MAX // BK
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256

    # Padded shapes bound B to real K and zero weight tiles beyond real N.
    bq_num_records = None
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        bq_num_records = halves_real * fx.Int32(1024)
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # Map each compute block to its SBM-padded expert metadata row.
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = 8 if is_f8_a else 4  # fp8 packs two 128-K halves -> i32<8:1>; fp4 -> i32<4:1>
    a_frags = [[fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)] for _ in range_constexpr(kMChunks)]

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
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
                    mask = lds_swizzle_mask_f8(lane_mod_16)
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
                    mask = lds_swizzle_mask(lane_mod_16)
                    lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                    vec = lds_vec_load(
                        s_aq_base,
                        row_off + lds_col,
                        Vec.make_type(4, fx.Int32),
                        fx.Int32,
                        align=16,
                    )
                    a_frags[i][k].store(Vec(vec))

    # The shared e8m0 scale layout bounds each A-scale view to its remaining bytes.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

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

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks).
        chunk_kt = kt if const_expr(tilesPerScaleChunk == 1) else kt // fx.Int32(tilesPerScaleChunk)
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(_raw(Vec(saf.load())[0]))
        return out

    # Stream B weights and scales through registers so use_nt reaches the ISA cache policy.
    bq_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_bq)

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

    bq_base_dw = [
        rocdl.readfirstlane(
            T.i32,
            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // 4) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

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

    frag_tmpl = make_bq_view(0)[0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
                load_mask = None
                if const_expr(has_pad):
                    col = n_block_idx * BN + wave * (BN // 4) + j * 16
                    load_mask = (col < N_real) & (kt_rt * fx.Int32(kHalves) + fx.Int32(half) < halves_real)
                bq_vec = buffer_ops.buffer_load(
                    bq_rsrc,
                    bq_off_dw,
                    vec_width=4,
                    dtype=T.i32,
                    mask=load_mask,
                    cache_modifier=2 if use_nt else 0,
                )
                bqf[j][half].store(Vec(bq_vec))
        chunk_kt = kt_rt if const_expr(tilesPerScaleChunk == 1) else kt_rt // fx.Int32(tilesPerScaleChunk)
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)] for _ in range_constexpr(numAccN)]
        bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    # Scaled-MFMA clusters over the loaded A / B / scale fragments.
    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return _raw(fx.Int32(scale).shrui(scale_shift))

    def mfma_cluster(bqf, bsf, sa, kt_rt):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sa = [shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)]
        sb_words = [shift_scale_word(_raw(Vec(bsf[mni].load())[0]), kt_rt) for mni in range_constexpr(nPairs)]
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
                    rg_off=rg_off,
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
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
                )

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = Vec.filled(4, 0.0, Float32)
    c_frags = [[fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)] for _ in range_constexpr(kMChunks)]
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

    if const_expr(BM == 64 and BN == 256):
        # BM64/BN256 uses the 1-stage B path unconditionally; do not depend on env knobs.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
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
        cur_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)] for _ in range_constexpr(numAccN)
        ]
        cur_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        nxt_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)] for _ in range_constexpr(numAccN)
        ]
        nxt_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]
            nxt_saf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]

        def load_b_carry():
            # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(cur_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(cur_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(cur_saf[sub].load())
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

        def rotate_b_carry():
            # Yield the PREFETCHED (next-tile) values -> become "current" next iteration.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(nxt_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(nxt_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(nxt_saf[sub].load())
            return out

        def issue_a_scale_load_into(saf, kt_rt):
            # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(sa[sub])

        def load_carry():
            return load_c_carry() + load_b_carry()

        def store_carry(state):
            base = store_c_carry(state)
            store_b_carry(state, base)

        def yield_carry():
            return load_c_carry() + rotate_b_carry()

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
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
            if const_expr(g2_ascale_pf):
                sa = [_raw(Vec(cur_saf[sub].load())[0]) for sub in range_constexpr(kScaleSubBlocks)]
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

    # Load the C fragments (fp8/fp4 unified onto the same fx.gemm path) and hand them to the epilog.
    accm_vecs = [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt


def get_gemm2_autotune_configs(a_dtype="fp8"):
    """Return the pruned Stage2 autotune space."""
    cfgs = [
        # (default) 2-stage B + A-scale prefetch + spatial-partition remap, f32 cshuffle, non-persist.
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=128, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=0, persist=False),
        dict(BN=128, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=0, persist=False),
        dict(BN=256, BK=256, use_nt=False, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=False, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=False, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=202, persist=False),
    ]
    bn_values = (128, 256)
    bn_env = os.environ.get("MEGA_G2_BN_VALUES")
    if bn_env:
        bn_values = tuple(int(value) for value in bn_env.split(","))
        if not bn_values or any(value not in (128, 256) for value in bn_values):
            raise ValueError("MEGA_G2_BN_VALUES must contain comma-separated values from {128, 256}")
    persist_cu_values = (8, 16, 32, 64, 128, 240, 256)
    persist_cu_env = os.environ.get("MEGA_G2_PERSIST_CU_VALUES")
    if persist_cu_env:
        persist_cu_values = tuple(int(value) for value in persist_cu_env.split(","))
        if not persist_cu_values or any(value <= 0 or value > 256 for value in persist_cu_values):
            raise ValueError("MEGA_G2_PERSIST_CU_VALUES must contain comma-separated values in [1, 256]")
    persist_strided_values = (False,)
    persist_strided_env = os.environ.get("MEGA_G2_PERSIST_STRIDED")
    if persist_strided_env is not None:
        if persist_strided_env not in ("0", "1"):
            raise ValueError("MEGA_G2_PERSIST_STRIDED must be '0' or '1'")
        persist_strided_values = (persist_strided_env == "1",)
    bf16_lds_values = (False,)
    bf16_lds_env = os.environ.get("MEGA_G2_BF16_LDS")
    if bf16_lds_env is not None:
        if bf16_lds_env not in ("0", "1"):
            raise ValueError("MEGA_G2_BF16_LDS must be '0' or '1'")
        bf16_lds_values = (bf16_lds_env == "1",)
    cfgs.extend(
        dict(
            BN=bn,
            BK=256,
            use_nt=use_nt,
            g2_bhoist=True,
            g2_ascale_pf=True,
            g2_spart=402,
            persist=True,
            persist_cu=slots,
            **({"persist_strided": True} if persist_strided else {}),
            **({"g2_bf16_lds": True} if g2_bf16_lds else {}),
        )
        for bn in bn_values
        for use_nt in (True, False)
        for slots in persist_cu_values
        for persist_strided in persist_strided_values
        for g2_bf16_lds in bf16_lds_values
    )
    if persist_strided_env is None and bf16_lds_env is None and 240 in persist_cu_values:
        cfgs.extend(
            dict(
                BN=bn,
                BK=256,
                use_nt=use_nt,
                g2_bhoist=True,
                g2_ascale_pf=True,
                g2_spart=402,
                persist=True,
                persist_cu=240,
                persist_strided=True,
            )
            for bn in bn_values
            for use_nt in (True, False)
        )
    if bf16_lds_env is None and 128 in bn_values:
        cfgs.extend(
            dict(
                BN=128,
                BK=256,
                use_nt=use_nt,
                g2_bhoist=True,
                g2_ascale_pf=True,
                g2_spart=402,
                persist=True,
                persist_cu=slots,
                persist_strided=persist_strided_values[0],
                g2_bf16_lds=True,
            )
            for use_nt in (True, False)
            for slots in persist_cu_values
            if slots in (64, 128, 240, 256)
        )
    no_persist = os.environ.get("MEGA_G2_NO_PERSIST") == "1"
    force_persist = os.environ.get("MEGA_G2_FORCE_PERSIST") == "1"
    if no_persist and force_persist:
        raise ValueError("MEGA_G2_NO_PERSIST and MEGA_G2_FORCE_PERSIST cannot both be enabled")
    if no_persist:
        cfgs = [config for config in cfgs if not config["persist"]]
    elif force_persist:
        cfgs = [config for config in cfgs if config["persist"]]
    force_use_nt = os.environ.get("MEGA_G2_FORCE_USE_NT")
    if force_use_nt is not None:
        if force_use_nt not in ("0", "1"):
            raise ValueError("MEGA_G2_FORCE_USE_NT must be '0' or '1'")
        use_nt = force_use_nt == "1"
        cfgs = [config for config in cfgs if config["use_nt"] == use_nt]
    # BK128 is inaccurate here; BM128 at BN256 fits gfx950's 160 KiB block LDS.
    return [dict(config, BM=BM) for BM in (16, 32, 64, 128) for config in cfgs]


def _spart_output_tile_index(block_1d_id, M0, N0, group_num, m01):
    """Map a 1D block ID to a spatially local output tile."""
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = (mn + gn - fx.Int32(1)) // gn
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = block_1d_id // gn
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x <= big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)

    idx_M0 = remap // n0
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = M0 // m01c
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = idx_M0 // m01c
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = idx_local // M01_adapt
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def _resolve_g2_knobs(g2_bhoist, g2_ascale_pf, g2_spart, g2_bf16_lds, use_reduce):
    """Env-default the gemm2 perf knobs (explicit arg wins), matching aiter compile_gemm2_a4w4_port."""
    if g2_bhoist is None:
        g2_bhoist = os.environ.get("MXFP4_G2_BHOIST", "1") == "1"
    g2_bhoist = bool(g2_bhoist)
    if g2_ascale_pf is None:
        g2_ascale_pf = os.environ.get("MXFP4_G2_ASCALE_PF", "1") == "1"
    g2_ascale_pf = bool(g2_ascale_pf)
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0
    if g2_spart > 0 and (g2_group_num < 1 or g2_m01 < 1):
        raise AssertionError(f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1 as GroupNum*100+M01 (e.g. 402)")
    if g2_bf16_lds is None:
        g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", "1") == "1" and use_reduce
    g2_bf16_lds = bool(g2_bf16_lds) and use_reduce
    return g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, g2_bf16_lds
