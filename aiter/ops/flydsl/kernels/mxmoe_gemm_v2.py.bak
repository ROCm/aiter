# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device body (BM32): gemm2 down."""

import types

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
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

from .mxfp4_gemm_common import (
    _e8m0_from_amax,
    _global_i32_buffer_tiles,
    _inline_dpp_pair_amax,
    _inline_dpp_quad_amax,
    _udiv,
    _umax_i32,
    _umod,
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
from .mxfp4_gemm_common import _fabs_f32 as fabs_f32
from .mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask

_FP8_E8M0_SHIFT = 7
_FP4_E8M0_SHIFT = 2

_G2_EPI_LANES = 32


def bq_view(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
):
    """Layout view over preshuffled B for one N-row tile."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * 4)
    # i32 strides: klane[0,4)->64, nlane[0,16)->4,
    # K_tile->K_HALVES*256, half->256, kpack4->1.
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 4)
    view = fx.Tensor(
        fx.make_view(
            base_iter,
            fx.make_layout(shape, (64, 4, K_HALVES * 256, 256, 1)),
        )
    )
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_view_fp8(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
):
    """Layout view over preshuffled FP8 B; pair selects two 16B cells per MFMA."""
    base = bq_view(
        arg_bq,
        row_elems,
        KH4,
        K_TILES_TOTAL * 2,
        K_HALVES,
    )
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 2, 4)
    stride = (64, 4, K_HALVES * 2 * 256, 2 * 256, 256, 1)
    return fx.Tensor(fx.make_view(fx.get_iter(base), fx.make_layout(shape, stride)))


def scale_view(
    arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """Layout view over an e8m0 scale buffer with an optional byte bound."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_scale) + off_i64 * 4)
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
    k_halves=2,
    swap_ab=False,
):
    """One J-cluster of scaled MFMAs over a 32-row A-scale group (row-groups i0, i0+1); each is
    an fx.gemm on i32 A/B frags (fp8 A = i32<8:1>, fp4 A = i32<4:1>), e8m0 words on scale_a/scale_b.
    sa: 32-row A-scale reg. single_rg (BM16): one 16-row group, rg_off picks its byte.
    swap_ab: B is the first operand, so a lane accumulates 4 consecutive columns of one row.
    """
    row_groups = (rg_off,) if const_expr(single_rg) else range(2)
    for k in range(k_halves):
        for im in row_groups:
            i = i0 if const_expr(single_rg) else i0 + im
            if const_expr(swap_ab):
                fx.gemm(
                    atoms[(2 * k + in_b, 2 * k + im)],
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
    resolved_rows=(),
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via the flat buffer view bounds."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4
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
        sorted_row = m_row + lds_row + a_lane_row
        car = resolved_rows[g] if const_expr(len(resolved_rows) > 0) else sorted_row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        # The byte offset is non-negative and 4-byte aligned; avoid signed-division fixup VGPRs.
        v_e = (voffset + kt * KH_TILE_A).shrui(fx.Int32(2))
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(s_aq_base, off, elem_ty=T.i32, align=16)
        )


def _route_e8m0(amax_bits, shift):
    """Route-row block scale: the amax exponent minus shift (at least 1; 0 for an all-zero
    block) as e8m0, and the f32 scale that quantizes the block."""
    ax_e = (amax_bits >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = ax_e - fx.Int32(shift)
    e8m0 = (e8m0 < fx.Int32(1)).select(fx.Int32(1), e8m0)
    e8m0 = (amax_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
    block_scale = (amax_bits == fx.Int32(0)).select(
        fx.Float32(1.0),
        fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32)),
    )
    return e8m0, block_scale


def _swap_umax(m, swap):
    """Unsigned max of m with the partner lane's m of a permlane swap.
    Raw: FlyDSL 0.5 has only the ROCDL permlane16/32_swap op (struct result), and DPP row
    ops cannot reach the lanes 16 and 32 away."""
    r = swap(
        _llvm.StructType.get_literal([T.i32, T.i32]), _raw(m), _raw(m), False, False
    )
    return _umax_i32(
        fx.Int32(_llvm.ExtractValueOp(T.i32, r, [0]).res),
        fx.Int32(_llvm.ExtractValueOp(T.i32, r, [1]).res),
    )


def es_default_peel(rows_per_expert):
    """persist-es: (chunks unrolled ahead of a panel's chunk loop, slices of the next panel's
    20 W2 loads they issue, whether chunk 0 issues its slice before staging chunk 1). Panels
    with more rows spread the next panel's W2 over more chunks."""
    if rows_per_expert <= 48:
        return 2, (0, 20, 20), False
    if rows_per_expert <= 96:
        return 3, (0, 8, 14, 20), False
    return 5, (0, 8, 11, 14, 17, 20), True


def make_es_state(BN, BK, INTER_MAX):
    """Register fragments a persist-es workgroup carries from panel to panel: W2 of this
    panel and the next (16B units [kt][J][half], e8m0 words [kt][pair]), three staged A
    chunks (2 x 16B of A, an e8m0 word, a route id and weight per lane) and the expert-table rows
    [first sorted row, rows, expert, 0] of this panel and the next two."""
    KT, numAccN, kHalves = INTER_MAX // BK, BN // 64, BK // 128
    nPairs = max(1, numAccN // 2)

    def w2():
        return types.SimpleNamespace(
            b=[
                [
                    [fx.make_rmem_tensor(4, Int32) for _ in range(kHalves)]
                    for _ in range(numAccN)
                ]
                for _ in range(KT)
            ],
            s=[
                [fx.make_rmem_tensor(1, Int32) for _ in range(nPairs)]
                for _ in range(KT)
            ],
        )

    def chunk():
        return [fx.make_rmem_tensor(n, Int32) for n in (4, 4, 1, 1, 1)]

    return types.SimpleNamespace(
        w=w2(),
        w_n=w2(),
        a=[chunk() for _ in range(3)],
        ein=fx.make_rmem_tensor(4, Int32),
        ein_n=fx.make_rmem_tensor(4, Int32),
        ein_nn=fx.make_rmem_tensor(4, Int32),
    )


def es_clamp_panel(p, n_panels):
    p = (p < n_panels).select(p, n_panels - 1)
    return (p > fx.Int32(0)).select(p, fx.Int32(0))


def load_es_einfo(arg_einfo, num_experts, p, n_panels, num_n_blocks, dst):
    """dst <- the expert-table row of panel p (clamped into the panels)."""
    tiles = _global_i32_buffer_tiles(arg_einfo, fx.Int64((num_experts + 1) * 16), 4)
    row = _udiv(es_clamp_panel(p, n_panels), num_n_blocks)
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Int32),
        fx.slice(tiles, (None, row)),
        dst,
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
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
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
    route_out_fp4=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_epi_lanes=None,
    g2_apre=False,
    enable_bias=False,
    nonatomic=False,
    scatter_out_fp4=False,
    arg_out_scale=None,
    g2_prefetch_ids=False,
    reduce_store_cache_modifier=None,
    resolved_input_rows=(),
    output_n_base=0,
    output_width=None,
    write_limit=None,
    es=None,
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
    # Contraction K = inter_dim (i32_inter); the K-loop is unrolled over INTER_MAX // BK tiles, so i32_inter must equal INTER_MAX.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = _udiv(K_rt, fx.Int32(a_pack))
    kc_rt = _udiv(K_rt + 255, fx.Int32(256))
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = _udiv(N_OUT_rt, fx.Int32(32)) * kBS_stride_n0_dw
    num_n_blocks = _udiv(N_OUT_rt, fx.Int32(BN))
    KH4 = _udiv(K_rt, fx.Int32(4 if is_f8_b else 8))
    K_TILES_MAX = INTER_MAX // BK
    K_SCALE_CHUNKS_MAX = (INTER_MAX + 255) // 256

    # block -> (m_block_idx, n_block_idx); e = sorted_expert_ids[SBM-padded sort block] (SBM==BM: sort_block==m_block_idx).
    if const_expr(mn_idx is not None):
        m_block_idx, n_block_idx = mn_idx
    elif const_expr(es is not None):
        # persist-es panel es.p: N block es.p % num_n_blocks of expert-table row es.state.ein.
        m_block_idx = fx.Int32(0)
        n_block_idx = _umod(es.p, num_n_blocks)
    else:
        m_block_idx = _udiv(bx_i32, num_n_blocks)
        n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    m_row = m_block_idx * BM
    if const_expr(es is not None):
        e = rocdl.readfirstlane(T.i32, _raw(Vec(es.state.ein.load())[2]))
    elif const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
    else:
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[_udiv(m_row, fx.Int32(SBM))]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    lds_acc_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype, b_dtype)

    if const_expr(len(resolved_input_rows) > 0):
        aq_num_records = fx.Int64(i32_M) * fx.Int64(topk) * fx.Int64(K_BYTES)
    else:
        aq_num_records = fx.Int64(i32_max_m_blocks) * fx.Int64(BM) * fx.Int64(K_BYTES)
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
            resolved_rows=resolved_input_rows,
        )

    def issue_a_ds_read(slot, row_groups=None):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                if const_expr(row_groups is not None and i not in row_groups):
                    continue
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
    scale_chunk0 = (
        m_block_idx if const_expr(is_bm16 and SBM == BM) else m_row // fx.Int32(32)
    )

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * 4
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
            scale = Vec(saf.load())[0]
            if const_expr(is_bm16 and SBM != BM):
                scale_shift = ((m_row // 16) & fx.Int32(1)) * fx.Int32(8)
                scale = scale.shrui(scale_shift) & fx.Int32(0x00FF00FF)
            out.append(scale)
        return out

    # B-weight + B-scale: global->register, streamed per K-tile (not LDS-staged).
    # b128 weight copy atom; cache modifier 2=nontemporal, 0=default.
    b_catom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if use_nt else 0), 32)

    def make_bq_view(j, e=e, n_block_idx=n_block_idx):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        if const_expr(is_f8_b):
            return bq_view_fp8(
                arg_bq,
                e * N_OUT_rt + col,
                KH4,
                K_TILES_MAX,
                kHalves,
            )
        return bq_view(
            arg_bq,
            e * N_OUT_rt + col,
            KH4,
            K_TILES_MAX,
            kHalves,
        )

    bq_views = [make_bq_view(j) for j in range_constexpr(numAccN)]

    def make_bscale_views(e=e, n_block_idx=n_block_idx):
        mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
        return [
            scale_view(
                arg_bscale,
                e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
                K_SCALE_CHUNKS_MAX,
                k0_stride_dw=kBS_stride_k0_dw,
            )
            for mw in range_constexpr(nPairs)
        ]

    bscale_views = make_bscale_views()

    frag_tmpl = (
        None
        if const_expr(is_f8_b)
        else bq_views[0][0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    )
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_value_load(dst, j, half, kt_rt, views=None):
        if const_expr(views is None):
            views = bq_views
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
            lo_v = fx.memref_load_vec(lo)
            hi_v = fx.memref_load_vec(hi)
            dst.store(lo_v.shuffle(hi_v, list(range(B_NDW))))
        else:
            fx.copy(
                b_catom,
                views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
                dst,
            )

    def issue_bscale_into(bsf, chunk_kt):
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def issue_b_load_into(bqf, bsf, kt_rt):
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
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

    def mfma_cluster(bqf, bsf, sa, kt_rt, interleave=None, swap_ab=False, rg=None):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        # rg: only that 16-row group of a 32-row tile.
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
            if const_expr(is_bm16 or rg is not None):
                rg_i = 0 if rg is None else rg
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=rg_i,
                    single_rg=True,
                    rg_off=rg_i,
                    k_halves=kHalves,
                    swap_ab=swap_ab,
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
                        swap_ab=swap_ab,
                    )
            if const_expr(interleave is not None and J > 0):
                interleave[J - 1]()
        if const_expr(interleave is not None):
            interleave[numAccN - 1]()

    # C accumulator: register fragments, zeroed then accumulated in place.
    zero4 = fx.full(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

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
            route_out_fp4=route_out_fp4,
            g2_defer_weight=g2_defer_weight,
            g2_out_pitch_align=g2_out_pitch_align,
            g2_scale_blk=g2_scale_blk,
            g2_epi_lanes=g2_epi_lanes,
            enable_bias=enable_bias,
            output_n_base=output_n_base,
            output_width=output_width,
            reduce_store_cache_modifier=reduce_store_cache_modifier,
            write_limit=write_limit,
            **kw,
        )

    def load_epilog_ids():
        EPI_LANES = _G2_EPI_LANES if g2_epi_lanes is None else int(g2_epi_lanes)
        EPI_ROWS = 256 // EPI_LANES
        M_REPS = BM // EPI_ROWS
        m_lane = fx.Int32(gpu.thread_id("x")) // EPI_LANES
        stids = flat_buffer_view(
            arg_stids, None, T.i32, align=4, elem_bytes=4, fold=False
        )
        load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
        packed = []
        for mr in range_constexpr(M_REPS):
            sorted_pos = m_row + mr * EPI_ROWS + m_lane
            frag = fx.make_rmem_tensor(1, Int32)
            fx.copy(load_i32, stids[None, sorted_pos], frag)
            packed.append(frag.load()[0])
        return packed

    g2_interleave = const_expr(
        g2_bf16_lds and not nonatomic and output_width is None and es is None
    )
    g2_prefetch_ids = const_expr(
        g2_prefetch_ids
        and g2_interleave
        and use_reduce
        and route_out_fp8
        and bool(g2_defer_weight)
    )
    prefetched_ids = None
    epi_thunks = [] if const_expr(g2_interleave) else None
    if const_expr(g2_interleave):
        _epilog(c_frags, emit_thunks=epi_thunks)

    if const_expr(es is not None):
        # persist-es: this panel is one expert's rows x BN columns over the whole K. Its W2
        # is in registers (st.w; the first panel loads its own, each panel prefetches the
        # next one's into st.w_n over its first PEEL chunks). A streams through LDS BM rows
        # at a time, global -> VGPR -> LDS two chunks ahead: an LDS DMA wait would also wait
        # for that W2 in flight. W2 is the MFMA's first operand, so a lane holds 4 columns of
        # one row: route rows are quantized in registers (or weighted to bf16 for a bf16
        # reduce), staged in LDS and stored 16 bytes a lane. 16-row halves past the expert's
        # rows are skipped.
        st = es.state
        PEEL, SL, EARLY = es.peel
        KT = K_TILES_MAX
        G = es.n_wg
        A_UNITS = INTER_MAX // 32  # 16-byte units of an fp4 A row
        es_bf16 = not route_out_fp4  # weighted bf16 route rows [tokens * topk, N]
        assert BM == 32 and BK == 256 and A_UNITS == 16
        assert route_out_fp4 or (use_reduce and not g2_defer_weight and not enable_bias)
        assert SL[-1] == KT * (nPairs + numAccN * kHalves) and len(SL) == PEEL + 1
        # Bytes of a route row's BN columns: packed e2m1 (then SB e8m0 bytes), or bf16 with
        # the LDS rows padded 16 bytes off the banks.
        OUT_RB = BN * 2 if es_bf16 else BN // 2
        RB = OUT_RB + 16 if es_bf16 else OUT_RB
        SB = 0 if es_bf16 else BN // g2_scale_blk
        assert es_bf16 or (RB == 128 and SB == 8 and g2_scale_blk == 32)
        # A chunk's META: its e8m0 words, route ids, then (bf16) route weights.
        META_IDS = BM * INTER_MAX // 32
        META_W = META_IDS + BM * 4
        META = META_W + (BM * 4 if es_bf16 else 0)
        OBUF = BM * RB + BM * SB
        # LDS: the A slots of two chunks | their META | two OBUF.
        META0 = 2 * KT * slot_bytes
        OB0 = META0 + 2 * META
        tid = fx.Int32(gpu.thread_id("x"))
        ROWS_PER_PASS = 256 // (OUT_RB // 16)  # route rows stored per pass, 16B a lane
        orow = tid // (OUT_RB // 16)
        oseg = tid % (OUT_RB // 16)
        if const_expr(es_bf16):
            pitch = N_OUT_rt * fx.Int32(2)
        else:
            pitch = _udiv(N_OUT_rt, fx.Int32(2)) + _udiv(
                N_OUT_rt, fx.Int32(g2_scale_blk)
            )
            if const_expr(g2_out_pitch_align > 0):
                al = fx.Int32(g2_out_pitch_align)
                pitch = ((pitch + al - fx.Int32(1)) // al) * al
        nrec = i32_M * fx.Int32(topk) * pitch
        a_tiles = _global_i32_buffer_tiles(arg_aq, aq_num_records, 4)
        as_tiles = _global_i32_buffer_tiles(arg_ascale, asc_num, 1)
        id_tiles = _global_i32_buffer_tiles(
            arg_stids, fx.Int64(i32_max_m_blocks) * fx.Int64(BM * 4), 1
        )
        w_tiles = _global_i32_buffer_tiles(
            arg_sweights, fx.Int64(i32_max_m_blocks) * fx.Int64(BM * 4), 1
        )
        out16 = _global_i32_buffer_tiles(arg_out, fx.Int64(nrec), 4)
        out4 = _global_i32_buffer_tiles(arg_out, fx.Int64(nrec), 1)
        g16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Int32)
        g4 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
        l16 = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        l8 = fx.make_copy_atom(fx.UniversalCopy64b(), Int32)
        l4 = fx.make_copy_atom(fx.UniversalCopy32b(), Int32)

        def lds_view(byte_off, n):
            ptr = lds_typed_ptr(lds_base_i32, T.i32, align=4 * n, byte_offset=byte_off)
            return fx.make_view(ptr, fx.make_layout(n, 1))

        def lds_i32(byte_off):
            return fx.Int32(lds_vec_load(lds_base_i32, byte_off, T.i32, Int32))

        SLOT_UNITS = KH_TILE_A // 16

        def a_unit(h):
            # (row, K tile, 16B unit) of a lane's h-th unit of a chunk. 16 lanes write two
            # rows of one slot: 256 bytes on distinct banks.
            u = h * 256 + tid
            row = (u // (2 * SLOT_UNITS * KT)) * 2 + (u // SLOT_UNITS) % 2
            return row, (u // (2 * SLOT_UNITS)) % KT, u % SLOT_UNITS

        def load_chunk(dst, base, nch, ci):
            # Chunk ci (clamped into the panel) of the rows from base: two 16B units of A,
            # a word of the chunk's e8m0 scales, a route id and (bf16) weight per lane.
            cc = (ci < nch).select(ci, nch - fx.Int32(1))
            cc = (cc < fx.Int32(0)).select(fx.Int32(0), cc)
            r = base + cc * BM
            for h in range_constexpr(2):
                row, kt, c = a_unit(h)
                unit = (r + row) * A_UNITS + kt * SLOT_UNITS + c
                fx.copy(g16, fx.slice(a_tiles, (None, unit)), dst[h])
            sc = _udiv(r, fx.Int32(32)) * kAS_per_chunk_dw
            sc = sc + (tid & (BM * INTER_MAX // 128 - 1))
            fx.copy(g4, fx.slice(as_tiles, (None, sc)), dst[2])
            fx.copy(g4, fx.slice(id_tiles, (None, r + (tid & (BM - 1)))), dst[3])
            if const_expr(es_bf16):
                fx.copy(g4, fx.slice(w_tiles, (None, r + (tid & (BM - 1)))), dst[4])

        def to_lds(src, buf):
            # A to the K-tile slots issue_a_ds_read reads; scales and route ids to META.
            for h in range_constexpr(2):
                row, kt, c = a_unit(h)
                col = c * 16
                off = (buf * KT + kt) * slot_bytes + row * KH_TILE_A
                off = off + (col ^ lds_swizzle_mask(row, KH_TILE_A))
                fx.copy(l16, src[h], lds_view(off, 4))
            meta = META0 + buf * META
            if tid < fx.Int32(BM * INTER_MAX // 128):
                fx.copy(l4, src[2], lds_view(meta + tid * 4, 1))
            if tid < fx.Int32(BM):
                fx.copy(l4, src[3], lds_view(meta + META_IDS + tid * 4, 1))
                if const_expr(es_bf16):
                    fx.copy(l4, src[4], lds_view(meta + META_W + tid * 4, 1))

        ein = Vec(st.ein.load())
        ein_n = Vec(st.ein_n.load())
        base = fx.Int32(rocdl.readfirstlane(T.i32, _raw(ein[0])))
        cnt = fx.Int32(rocdl.readfirstlane(T.i32, _raw(ein[1])))
        nch = (cnt + fx.Int32(BM - 1)) // BM
        base_n = fx.Int32(rocdl.readfirstlane(T.i32, _raw(ein_n[0])))
        cnt_n = fx.Int32(rocdl.readfirstlane(T.i32, _raw(ein_n[1])))
        e_n = rocdl.readfirstlane(T.i32, _raw(ein_n[2]))
        nch_n = (cnt_n + fx.Int32(BM - 1)) // BM
        nb_n = _umod(es_clamp_panel(es.p + G, es.n_panels), num_n_blocks)

        def chunk(ci, src, ahead, early=False):
            """Chunk ci: chunk ci + 1 (src) to LDS, ahead's loads, then chunk ci's MFMAs up
            to the expert's last row, quantization and stores."""
            if const_expr(early):
                ahead()
            if ci + 1 < nch:
                to_lds(src, (ci + 1) & 1)
            if const_expr(not early):
                ahead()
            buf = ci & 1
            meta = META0 + buf * META
            sa = [
                lds_i32(meta + kt * 256 + lane_div_16 * 64 + lane_mod_16 * 4)
                for kt in range_constexpr(KT)
            ]
            fos = [
                lds_i32(meta + META_IDS + (orow + q * ROWS_PER_PASS) * 4)
                for q in range_constexpr(BM // ROWS_PER_PASS)
            ]
            live = ci < nch
            ob = OB0 + buf * OBUF

            def quant(rg):
                row = rg * 16 + lane_mod_16
                for b in range_constexpr(numAccN // 2):
                    vals = [
                        fx.Float32(Vec(c_frags[rg][2 * b + h].load())[v])
                        for h in range_constexpr(2)
                        for v in range_constexpr(4)
                    ]
                    amax = fabs_f32(vals[0])
                    for q in range_constexpr(1, 8):
                        amax = amax.maximumf(fabs_f32(vals[q]))
                    bits = fx.Int32(_raw(amax).bitcast(T.i32))
                    bits = _swap_umax(bits, rocdl.permlane16_swap)
                    bits = _swap_umax(bits, rocdl.permlane32_swap)
                    e8m0, scale = _route_e8m0(bits, _FP4_E8M0_SHIFT)
                    w = _raw(fx.Int32(0))
                    for h in range_constexpr(4):
                        w = rocdl.cvt_scalef32_pk_fp4_f32(
                            T.i32,
                            w,
                            _raw(vals[2 * h]),
                            _raw(vals[2 * h + 1]),
                            _raw(scale),
                            h,
                        )
                    w = fx.Int32(w)
                    o = ob + row * RB + wave * (BN // 8) + 16 * b + 2 * lane_div_16
                    for h in range_constexpr(2):
                        fx.ptr_store(
                            (w >> fx.Int32(16 * h)).to(Int16),
                            lds_typed_ptr(
                                lds_base_i32, T.i16, align=2, byte_offset=o + 8 * h
                            ),
                        )
                    if lane_div_16 == fx.Int32(0):
                        so = ob + BM * RB + row * SB + wave * (numAccN // 2) + b
                        fx.ptr_store(
                            e8m0.to(Int8),
                            lds_typed_ptr(lds_base_i32, T.i8, align=1, byte_offset=so),
                        )

            def to_bf16(rg):
                row = rg * 16 + lane_mod_16
                wt = fx.Float32(
                    lds_vec_load(lds_base_i32, meta + META_W + row * 4, T.f32, Float32)
                )
                for J in range_constexpr(numAccN):
                    acc = Vec(c_frags[rg][J].load())
                    pk = Vec.from_elements(
                        [fx.Float32(acc[v]) * wt for v in range_constexpr(4)], Float32
                    ).to(BFloat16)
                    r2 = fx.make_rmem_tensor(2, Int32)
                    r2.store(pk.bitcast(Int32))
                    col = wave * (BN // 4) + J * 16 + 4 * lane_div_16
                    fx.copy(l8, r2, lds_view(ob + row * RB + col * 2, 2))

            def half(rg):
                # One 16-row group at a time keeps half the accumulators live.
                for J in range_constexpr(numAccN):
                    c_frags[rg][J].store(zero4)
                rocdl.s_setprio(1)
                for kt in range_constexpr(KT):
                    issue_a_ds_read(buf * KT + kt, row_groups=(rg,))
                    mfma_cluster(
                        st.w.b[kt],
                        st.w.s[kt],
                        [sa[kt]],
                        fx.Int32(kt),
                        swap_ab=True,
                        rg=rg,
                    )
                rocdl.s_setprio(0)
                if const_expr(es_bf16):
                    to_bf16(rg)
                else:
                    quant(rg)

            if ci * BM + 16 < cnt:
                half(0)
                half(1)
            if (ci * BM + 16 >= cnt) & live:
                half(0)
            gpu.barrier()
            # Padding rows carry token M, slot topk: their route rows lie past nrec.
            for q in range_constexpr(BM // ROWS_PER_PASS):
                fo = fos[q]
                dest = ((fo & fx.Int32(0xFFFFFF)) * topk + (fo >> fx.Int32(24))) * pitch
                dest = live.select(dest, nrec)
                v = fx.make_rmem_tensor(4, Int32)
                orow_q = orow + q * ROWS_PER_PASS
                fx.copy(l16, lds_view(ob + orow_q * RB + oseg * 16, 4), v)
                dest_n = dest + n_block_idx * OUT_RB + oseg * 16
                fx.copy(g16, v, fx.slice(out16, (None, dest_n >> 4)))
            if const_expr(es_bf16):
                return
            # Unconditional, so wait counts stay exact: lanes past a row's scale words aim past nrec.
            sseg = oseg % (SB // 4)
            sv = fx.make_rmem_tensor(1, Int32)
            fx.copy(l4, lds_view(ob + BM * RB + orow * SB + sseg * 4, 1), sv)
            so = dest + _udiv(N_OUT_rt, fx.Int32(2)) + n_block_idx * SB + sseg * 4
            so = (oseg < fx.Int32(SB // 4)).select(so, nrec)
            fx.copy(g4, sv, fx.slice(out4, (None, so >> 2)))

        if es.j == fx.Int32(0):
            # The first panel loads its own W2 and chunks; later ones come prefetched.
            for kt in range_constexpr(KT):
                issue_b_load_into(st.w.b[kt], st.w.s[kt], fx.Int32(kt))
            load_chunk(st.a[0], base, nch, fx.Int32(0))
            to_lds(st.a[0], 0)
            load_chunk(st.a[0], base, nch, fx.Int32(1))
            gpu.barrier()

        # Chunk 2 ahead of the next panel's W2: waiting for it does not wait for that W2.
        load_chunk(st.a[1], base, nch, fx.Int32(2))
        nxt_bq = [make_bq_view(J, e_n, nb_n) for J in range_constexpr(numAccN)]
        nxt_bs = make_bscale_views(e_n, nb_n)
        w2_next = [
            lambda kt=kt, mw=mw: fx.copy(
                sc_copy_atom,
                nxt_bs[mw][
                    lane_div_16, lane_mod_16, scale_chunk_tile(fx.Int32(kt)), None
                ],
                st.w_n.s[kt][mw],
            )
            for mw in range_constexpr(nPairs)
            for kt in range_constexpr(KT)
        ] + [
            lambda J=J, kt=kt, half=half: issue_b_value_load(
                st.w_n.b[kt][J][half], J, half, fx.Int32(kt), views=nxt_bq
            )
            for J in range_constexpr(numAccN)
            for kt in range_constexpr(KT)
            for half in range_constexpr(kHalves)
        ]
        for k in range_constexpr(PEEL):

            def ahead(k=k):
                if const_expr(k >= 1):
                    load_chunk(st.a[1], base, nch, fx.Int32(k + 2))
                for t in range_constexpr(SL[k], SL[k + 1]):
                    w2_next[t]()
                if const_expr(k == PEEL - 1):
                    load_es_einfo(
                        arg_out_scale,
                        es.num_experts,
                        es.p + 2 * G,
                        es.n_panels,
                        num_n_blocks,
                        st.ein_nn,
                    )
                    load_chunk(st.a[2], base_n, nch_n, fx.Int32(0))

            chunk(
                fx.Int32(k),
                st.a[0] if k == 0 else st.a[1],
                ahead,
                early=EARLY and k == 0,
            )
        for c in range(fx.Int32(PEEL), nch, fx.Int32(1)):
            ci = fx.Int32(c)
            chunk(
                ci,
                st.a[1],
                lambda ci=ci: load_chunk(st.a[1], base, nch, ci + fx.Int32(2)),
            )
        # Every read of both A buffers is behind the last chunk's barrier.
        to_lds(st.a[2], 0)
        load_chunk(st.a[0], base_n, nch_n, fx.Int32(1))
        st.ein.store(
            Vec.from_elements([base_n, cnt_n, fx.Int32(e_n), fx.Int32(0)], Int32)
        )
        st.ein_n.store(st.ein_nn.load())
        for kt in range_constexpr(KT):
            for mw in range_constexpr(nPairs):
                st.w.s[kt][mw].store(st.w_n.s[kt][mw].load())
            for J in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    st.w.b[kt][J][half].store(st.w_n.b[kt][J][half].load())
        gpu.barrier()
    elif const_expr(nonatomic):
        KT = K_TILES_MAX
        a_all_resident = const_expr((aStages if g2_apre else kStages) >= KT)

        if const_expr(a_all_resident):
            sa = [load_a_scale_tile(fx.Int32(S)) for S in range_constexpr(KT)]
            b = [make_bq_fragments() for _ in range_constexpr(KT)]
            bs = []
            for S in range_constexpr(KT):
                bsf = make_scale_fragments(nPairs)
                issue_bscale_into(bsf, scale_chunk_tile(fx.Int32(S)))
                bs.append(bsf)
            issue_b_load_into(b[0], None, fx.Int32(0))
            gpu.barrier()
            nDsPerTile = kMChunks * kHalves
            nBPerTile = numAccN * kHalves
            nMfmaPerTile = numAccN * kScaleSubBlocks * 2 * kHalves
            nNextB = (KT - 1) * nBPerTile
            for n in range_constexpr(nDsPerTile):
                rocdl.sched_dsrd(2)
                if const_expr(n < nNextB):
                    rocdl.sched_vmem(1)
            for _ in range_constexpr(KT * nMfmaPerTile):
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
            for S in range_constexpr(KT):
                issue_a_ds_read(fx.Int32(S % aStages))
                if const_expr(S + 1 < KT):
                    issue_b_load_into(b[S + 1], None, fx.Int32(S + 1))
                mfma_cluster(b[S], bs[S], sa[S], fx.Int32(S))
        else:
            sa = [load_a_scale_tile(fx.Int32(S)) for S in range_constexpr(KT)]
            bs = []
            for S in range_constexpr(KT):
                bsf = make_scale_fragments(nPairs)
                issue_bscale_into(bsf, scale_chunk_tile(fx.Int32(S)))
                bs.append(bsf)
            cur_bqf = make_bq_fragments()
            nxt_bqf = make_bq_fragments()
            issue_b_load_into(cur_bqf, None, fx.Int32(0))
            nDsPerTile = kMChunks * kHalves
            nBPerTile = numAccN * kHalves
            nMfmaPerJ = kHalves if is_bm16 else kScaleSubBlocks * 2 * kHalves
            nMfmaPerTile = numAccN * nMfmaPerJ
            rows_per_call = 64 // (KH_TILE_A // 16)
            rows_per_wave = BM // 4
            nA = 1 if rows_per_wave < rows_per_call else rows_per_wave // rows_per_call
            nDsPerGroup = max(1, (nDsPerTile + numAccN - 1) // numAccN)
            nDsGroups = (nDsPerTile + nDsPerGroup - 1) // nDsPerGroup
            nMfmaPerVmem = 4
            nPrologueVmem = kStages * nA + KT * (kScaleSubBlocks + nPairs) + nBPerTile
            nVmem0 = nPrologueVmem + (nBPerTile if const_expr(KT > 1) else 0)
            nVmemPerDs0 = max(1, (nVmem0 + nDsPerTile - 1) // nDsPerTile)
            fenced = -1
            issued = kStages - 1
            for kt in range_constexpr(KT):
                prefetch_a = const_expr(kt + kStages < KT)
                if const_expr(kt > fenced):
                    gpu.barrier()
                    fenced = issued
                if const_expr(kt == 0):
                    for _ in range_constexpr(nDsPerTile):
                        rocdl.sched_vmem(nVmemPerDs0)
                        rocdl.sched_dsrd(1)
                else:
                    nAVmem = nA if const_expr(kt - 1 + kStages < KT) else 0
                    nBVmem = nBPerTile if const_expr(kt + 1 < KT) else 0
                    for _ in range_constexpr(nAVmem + nBVmem):
                        rocdl.sched_mfma(nMfmaPerVmem)
                        rocdl.sched_vmem(1)
                    nMfmaLeft = max(
                        1,
                        (nMfmaPerTile - nMfmaPerVmem * (nAVmem + nBVmem)) // nDsGroups,
                    )
                    for _ in range_constexpr(nDsGroups):
                        rocdl.sched_mfma(nMfmaLeft)
                        rocdl.sched_dsrd(nDsPerGroup)
                if const_expr(kt + 1 < KT):
                    issue_b_load_into(nxt_bqf, None, fx.Int32(kt + 1))
                issue_a_ds_read(fx.Int32(kt % aStages))
                if prefetch_a:
                    if const_expr(a_slot_alias):
                        gpu.barrier()
                        fenced = issued
                    issue_a_load_lds(
                        fx.Int32((kt + kStages) % aStages),
                        fx.Int32(kt + kStages),
                    )
                    issued = kt + kStages
                mfma_cluster(cur_bqf, bs[kt], sa[kt], fx.Int32(kt))
                cur_bqf, nxt_bqf = nxt_bqf, cur_bqf
    else:
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
            if const_expr(g2_prefetch_ids and kt == KT - 1):
                prefetched_ids = load_epilog_ids()
                rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt, interleave=_il)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            cur_bqf, nxt_bqf = nxt_bqf, cur_bqf
    if const_expr(nonatomic and scatter_out_fp4):
        nonatomic_mxfp4_epilog(
            lds_acc_base,
            [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)],
            arg_out,
            arg_out_scale,
            m_row,
            n_block_idx,
            wave,
            lane,
            N_OUT_rt,
            BN,
            kMChunks,
        )
    elif const_expr(nonatomic):
        nonatomic_bf16_epilog(
            [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)],
            arg_out,
            m_row,
            n_block_idx,
            wave,
            lane,
            N_OUT_rt,
            BN,
            kMChunks,
            arg_bias=arg_bias,
            expert_id=e,
            enable_bias=enable_bias,
        )
    elif const_expr(g2_interleave):
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        _epilog(None, lds_ready=True, prefetched_ids=prefetched_ids)
    elif const_expr(es is None):
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
    route_out_fp4=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_epi_lanes=None,
    emit_thunks=None,
    lds_ready=False,
    enable_bias=False,
    prefetched_ids=None,
    output_n_base=0,
    output_width=None,
    reduce_store_cache_modifier=None,
    write_limit=None,
):
    if SBM is None:
        SBM = BM
    EPI_LANES = _G2_EPI_LANES if g2_epi_lanes is None else int(g2_epi_lanes)
    EPI_ROWS = 256 // EPI_LANES
    M_REPS = BM // EPI_ROWS
    ROUTE_VEC = BN // EPI_LANES
    # fp4 route rows reuse the fp8 route-out path: packed e2m1 values, same e8m0 blocks.
    route_out_q = route_out_fp8 or route_out_fp4
    if const_expr(use_reduce and route_out_q):
        assert BM % EPI_ROWS == 0, (EPI_LANES, EPI_ROWS, BM)
        assert ROUTE_VEC % 4 == 0, (EPI_LANES, BN, ROUTE_VEC)
        assert not route_out_fp4 or ROUTE_VEC % 8 == 0, (EPI_LANES, BN, ROUTE_VEC)
        assert g2_scale_blk in (ROUTE_VEC, 2 * ROUTE_VEC, 4 * ROUTE_VEC), (
            EPI_LANES,
            ROUTE_VEC,
            g2_scale_blk,
        )
    bf16_src = const_expr(bool(g2_bf16_lds) and bool(route_out_q))
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
    if reduce_store_cache_modifier is not None and (
        not use_reduce or route_out_q or enable_bias or g2_defer_weight
    ):
        raise ValueError("custom BF16 store policies require unweighted reduce output")
    output_width = N_OUT if const_expr(output_width is None) else fx.Int32(output_width)
    val_width = (
        _udiv(output_width, fx.Int32(2)) if const_expr(route_out_fp4) else output_width
    )
    output_n_base = fx.Int32(output_n_base)
    store_vec = 8 if reduce_store_cache_modifier is not None else 2
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
    out_i8_ptr = global_typed_ptr(arg_out, T.i8, align=1)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)
    reduce_bf16x8 = (
        fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(reduce_store_cache_modifier), BFloat16
        )
        if reduce_store_cache_modifier is not None
        else None
    )

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return frag.load()[0]

    def load_bias(col):
        return load_scalar(
            load_f32,
            bias_f32,
            expert_id * N_OUT + col,
            Float32,
        )

    defer_w = bool(g2_defer_weight)

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    packed = [] if const_expr(prefetched_ids is None) else prefetched_ids
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * EPI_ROWS + m_lane
        if const_expr(prefetched_ids is None):
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
            if const_expr(route_out_q):
                row_pitch = val_width + _udiv(output_width, fx.Int32(g2_scale_blk))
                if const_expr(g2_out_pitch_align > 0):
                    al = fx.Int32(g2_out_pitch_align)
                    row_pitch = ((row_pitch + al - fx.Int32(1)) // al) * al
                row_base_addr = out_row * fx.Int64(row_pitch)
            else:
                row_base_addr = out_row * fx.Int64(output_width) + fx.Int64(
                    n_block_idx * BN + col_start - output_n_base
                )
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        if const_expr(use_reduce and route_out_q):
            route_vec = ROUTE_VEC
            route_group_n = EPI_LANES * route_vec
            n_rg = (BN + route_group_n - 1) // route_group_n
            for rg in range_constexpr(n_rg):
                col_lane8 = rg * route_group_n + n_lane * fx.Int32(route_vec)

                def store_route_group(col_lane8, rg=rg):
                    global_col = n_block_idx * BN + col_lane8
                    output_col = global_col - output_n_base
                    bvals = []
                    vals = []
                    for q in range_constexpr(route_vec):
                        idx_q = row_in_block * BN + col_lane8 + fx.Int32(q)
                        if const_expr(bf16_src):
                            bval = lds_base_bf16[idx_q]
                            if const_expr(enable_bias):
                                bias_val = load_bias(global_col + q)
                                if const_expr(not defer_w):
                                    bias_val = bias_val * weight[mr]
                                bval = (fx.Float32(bval) + bias_val).to(BFloat16)
                            bvals.append(bval)
                        elif const_expr(g2_bf16_lds):
                            val = fx.Float32(lds_base_bf16[idx_q])
                            if const_expr(enable_bias):
                                bias_val = load_bias(global_col + q)
                                if const_expr(not defer_w):
                                    bias_val = bias_val * weight[mr]
                                val = val + bias_val
                            vals.append(val)
                        elif const_expr(defer_w):
                            val = fx.Float32(lds_base_fptr[idx_q])
                            if const_expr(enable_bias):
                                val = val + load_bias(global_col + q)
                            vals.append(val)
                        else:
                            val = fx.Float32(lds_base_fptr[idx_q])
                            if const_expr(enable_bias):
                                val = val + load_bias(global_col + q)
                            vals.append(val * weight[mr])
                    if const_expr(bf16_src):
                        msk = fx.full([2], 0x7FFF, Int16)
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
                    e8m0, block_scale = _route_e8m0(
                        amax_bits,
                        (
                            _FP4_E8M0_SHIFT
                            if const_expr(route_out_fp4)
                            else _FP8_E8M0_SHIFT
                        ),
                    )
                    bs_raw = _raw(block_scale)
                    pk_ty = T.vec(2, T.i16)

                    def pk_seed():
                        return _raw(fx.full([2], 0, fx.Int16))

                    words = []
                    if const_expr(route_out_fp4):
                        for d in range_constexpr(route_vec // 8):
                            w = _raw(fx.Int32(0))
                            for h in range_constexpr(4):
                                e = 8 * d + 2 * h
                                if const_expr(bf16_src):
                                    src2 = Vec.from_elements(
                                        [bvals[e], bvals[e + 1]], BFloat16
                                    )
                                    w = rocdl.cvt_scalef32_pk_fp4_bf16(
                                        T.i32, w, _raw(src2), bs_raw, h
                                    )
                                else:
                                    w = rocdl.cvt_scalef32_pk_fp4_f32(
                                        T.i32,
                                        w,
                                        _raw(vals[e]),
                                        _raw(vals[e + 1]),
                                        bs_raw,
                                        h,
                                    )
                            words.append(Vec.from_elements([fx.Int32(w)], Int32))
                    for d in range_constexpr(0 if route_out_fp4 else route_vec // 4):
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
                    emit_stores(output_col, words, e8m0)

                def emit_stores(output_col, words, e8m0, rg=rg):
                    row_val_off = row_base_addr + fx.Int64(
                        _udiv(output_col, fx.Int32(2))
                        if const_expr(route_out_fp4)
                        else output_col
                    )
                    for d in range_constexpr(len(words)):
                        fx.ptr_store(
                            Vec(words[d]).bitcast(Int8),
                            out_i8_ptr + (row_val_off + fx.Int64(4 * d)),
                        )
                    scale_off = (
                        row_base_addr
                        + fx.Int64(val_width)
                        + fx.Int64(_udiv(output_col, fx.Int32(g2_scale_blk)))
                    )
                    fx.ptr_store(e8m0.to(Int8), out_i8_ptr + scale_off)

                @flyc.jit
                def store_route_group_if_valid(col_lane8):
                    if col_lane8 < fx.Int32(BN):
                        store_route_group(col_lane8)

                store_route_group_if_valid(col_lane8)
        elif const_expr(reduce_store_cache_modifier is not None):
            for s in range_constexpr(BN // store_group_n):
                idx0 = row_in_block * BN + col_start + s * store_group_n
                if const_expr(g2_bf16_lds):
                    pk = Vec(
                        lds_vec_load(
                            lds_acc_base,
                            idx0 * 2,
                            Vec.make_type(store_vec, BFloat16),
                            BFloat16,
                            align=16,
                        )
                    )
                else:
                    values = Vec(
                        lds_vec_load(
                            lds_acc_base,
                            idx0 * 4,
                            Vec.make_type(store_vec, Float32),
                            Float32,
                            align=16,
                        )
                    )
                    pk = Vec.from_elements(
                        [
                            fx.Float32(values[i]) * weight[mr]
                            for i in range_constexpr(8)
                        ],
                        Float32,
                    ).to(BFloat16)
                out_off = row_base_addr + fx.Int64(s * store_group_n)
                out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                out_frag.store(pk)
                fx.copy(reduce_bf16x8, out_frag, out_bf16[None, out_off])
        else:
            for s in range_constexpr(BN // store_group_n):
                if const_expr(g2_bf16_lds):
                    pk = lds_pre[mr][s]
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
                    v2 = lds_pre[mr][s]
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

    def lds_pk_load(mr, s):
        # adjacent ee=0,1 contiguous -> one 2-wide load.
        idx0 = (fx.Int32(mr * EPI_ROWS) + m_lane) * BN + col_start + s * store_group_n
        if const_expr(g2_bf16_lds):
            return Vec(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * 2,
                    Vec.make_type(store_vec, BFloat16),
                    BFloat16,
                    align=4,
                )
            )
        return Vec(
            lds_vec_load(
                lds_acc_base,
                idx0 * 4,
                Vec.make_type(store_vec, Float32),
                Float32,
                align=8,
            )
        )

    # Read every row group's C from LDS before the per-row validity branches, so
    # the LDS latency overlaps instead of serializing read -> wait -> store per row.
    lds_pre = None
    if const_expr(
        not (use_reduce and route_out_q) and reduce_store_cache_modifier is None
    ):
        lds_pre = [
            [lds_pk_load(mr, s) for s in range_constexpr(BN // store_group_n)]
            for mr in range_constexpr(M_REPS)
        ]

    if const_expr(prefetched_ids is not None or (use_reduce and route_out_q)):
        rocdl.s_waitcnt(vmcnt=0)

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        route_slot = packed[mr] >> fx.Int32(24)

        @flyc.jit
        def store_if_valid(token_id, route_slot, mr):
            token_limit = i32_M if const_expr(write_limit is None) else write_limit
            valid = token_id < token_limit
            if const_expr(use_reduce):
                valid = valid & (route_slot < fx.Int32(topk))
            if valid:
                store_one_mr(mr)

        store_if_valid(token_id, route_slot, mr)


def nonatomic_bf16_epilog(
    accm,
    arg_out,
    m_row,
    n_block_idx,
    wave,
    lane,
    N_OUT,
    BN,
    kMChunks,
    arg_bias=None,
    expert_id=None,
    enable_bias=False,
):
    numAccN = (BN // 4) // 16
    row_base = m_row + (lane // 16) * 4
    gn_base = n_block_idx * BN + wave * (BN // 4) + (lane % 16)
    out_ptr = global_typed_ptr(arg_out, T.bf16, align=2) + (
        fx.Int64(row_base) * fx.Int64(N_OUT) + fx.Int64(gn_base)
    )
    bias = None
    if const_expr(enable_bias):
        bias_ptr = global_typed_ptr(arg_bias, T.f32) + (
            fx.Int64(expert_id) * fx.Int64(N_OUT) + fx.Int64(gn_base)
        )
        bias = [
            fx.Float32(bias_ptr[fx.Int32(J * 16)]) for J in range_constexpr(numAccN)
        ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                val = vec[v]
                if const_expr(enable_bias):
                    val = fx.Float32(val) + bias[J]
                bf = Vec.from_elements([val], Float32).to(BFloat16)
                out_ptr[fx.Int32(i * 16 + v) * N_OUT + J * 16] = bf[0]


def nonatomic_mxfp4_epilog(
    lds_acc_base,
    accm,
    arg_out_q,
    arg_out_scale,
    m_row,
    n_block_idx,
    wave,
    lane,
    N_OUT,
    BN,
    kMChunks,
):
    """Scatter epilog with MXFP4 output: q [max_sorted, N/2] + e8m0 [max_sorted, N/32], unweighted.

    fp32 C is staged through LDS (aliases the A slots) so each lane reads 8 contiguous
    columns; a lane quad owns one 32-column scale block.
    """
    assert BN % 128 == 0, f"mxfp4 scatter epilog needs BN % 128 == 0, got {BN}"
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    # C aliases the A slots: every wave must be done ds_reading A before C lands.
    gpu.barrier()
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + (lane // 16) * 4
        for J in range_constexpr(numAccN):
            col = wave * wave_n + J * 16 + lane % 16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                lds_base_fptr[(row_base + v) * BN + col] = fx.Float32(vec[v])
    gpu.barrier()

    NBLK = BN // 32
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 16
    n_lane = tx_i32 % 16
    wave_grp = n_lane // 4
    kk = n_lane % 4
    out_q = flat_buffer_view(arg_out_q, None, T.i32, align=4, elem_bytes=4, fold=False)
    store_q_nt = fx.make_copy_atom(fx.rocdl.BufferCopy32b(2), Int32)
    out_scale_ptr = global_typed_ptr(arg_out_scale, T.i8, align=1)
    q_row0 = fx.Int64(m_row + m_lane) * fx.Int64(_udiv(N_OUT, fx.Int32(8)))
    s_row0 = fx.Int64(m_row + m_lane) * fx.Int64(_udiv(N_OUT, fx.Int32(32)))
    blocks = [(mr, half) for mr in range(kMChunks) for half in range(NBLK // 4)]

    def issue_load(mr, half):
        group = wave_grp + half * 4
        col0 = group * 32 + kk * 8
        idx = (fx.Int32(mr * 16) + m_lane) * BN + col0
        v0 = Vec(lds_vec_load(lds_acc_base, idx * 4, T.vec(4, T.f32), T.f32, align=16))
        v1 = Vec(
            lds_vec_load(lds_acc_base, (idx + 4) * 4, T.vec(4, T.f32), T.f32, align=16)
        )
        return (
            [fx.Float32(v0[e]) for e in range(4)]
            + [fx.Float32(v1[e]) for e in range(4)],
            group,
            col0,
        )

    r_next = issue_load(*blocks[0])
    for bi in range_constexpr(len(blocks)):
        mr, _half = blocks[bi]
        r, group, col0 = r_next
        if const_expr(bi + 1 < len(blocks)):
            r_next = issue_load(*blocks[bi + 1])
        local_max = fabs_f32(r[0])
        for e in range_constexpr(1, 8):
            local_max = local_max.maximumf(fabs_f32(r[e]))
        amax_bits = _inline_dpp_quad_amax(fx.Int32(_raw(local_max).bitcast(T.i32)))
        e8m0, qscale = _e8m0_from_amax(amax_bits.bitcast(Float32))
        packed = _raw(fx.Int32(0))
        for h in range_constexpr(4):
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[2 * h]), _raw(r[2 * h + 1]), _raw(qscale), h
            )
        global_col = n_block_idx * BN + col0
        q_idx = (
            q_row0
            + fx.Int64(mr * 16) * fx.Int64(_udiv(N_OUT, fx.Int32(8)))
            + fx.Int64(global_col // 8)
        )
        q_frag = fx.make_rmem_tensor(1, Int32)
        q_frag.store(Vec.from_elements([packed], Int32))
        fx.copy(store_q_nt, q_frag, out_q[None, q_idx])
        s_off = (
            s_row0
            + fx.Int64(mr * 16) * fx.Int64(_udiv(N_OUT, fx.Int32(32)))
            + fx.Int64(n_block_idx * NBLK + group)
        )

        @flyc.jit
        def store_scale_if_leader(kk, e8m0, s_off):
            if kk == fx.Int32(0):
                fx.ptr_store(e8m0.to(Int8), out_scale_ptr + s_off)

        store_scale_if_leader(kk, e8m0, s_off)
