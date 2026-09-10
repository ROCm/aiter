# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 copy / scaled-MFMA instruction definitions (layout API)."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import (
    Float4E2M1FN,
    Float8E4M3FN,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from ..mxfp4_gemm_common import (
    _udiv,
    flat_buffer_view,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_vec_load,
)
from ..mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask


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


def make_scale_copy_atom():
    """32-bit buffer copy atom for e8m0 scale words."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def make_b_copy_atom(use_nt):
    """128-bit B-weight copy atom; cache modifier 2=nontemporal, 0=default."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if use_nt else 0), 32)


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
    resolved_rows=(),
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
        sorted_row = m_row + lds_row + a_lane_row
        car = resolved_rows[g] if const_expr(len(resolved_rows) > 0) else sorted_row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        # The byte offset is non-negative and 4-byte aligned; avoid signed-division fixup VGPRs.
        v_e = (voffset + kt * KH_TILE_A).shrui(fx.Int32(2))
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(s_aq_base, off, elem_ty=T.i32, align=16)
        )


def issue_a_ds_read_slot(
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
):
    """A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>."""
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


def scale_chunk_tile(kt, tilesPerScaleChunk):
    return (
        kt
        if const_expr(tilesPerScaleChunk == 1)
        else _udiv(kt, fx.Int32(tilesPerScaleChunk))
    )


def load_a_scale_tile(
    kt,
    sc_copy_atom,
    ascale_views,
    sc_frag_tmpl,
    kScaleSubBlocks,
    lane_div_16,
    lane_mod_16,
    tilesPerScaleChunk,
):
    chunk_kt = scale_chunk_tile(kt, tilesPerScaleChunk)
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


def issue_b_value_load(
    dst,
    j,
    half,
    kt_rt,
    b_catom,
    bq_views,
    lane_div_16,
    lane_mod_16,
    is_f8_b,
    B_NDW,
):
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


def issue_bscale_into(
    bsf,
    chunk_kt,
    sc_copy_atom,
    bscale_views,
    nPairs,
    lane_div_16,
    lane_mod_16,
):
    for mw in range_constexpr(nPairs):
        fx.copy(
            sc_copy_atom,
            bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
            bsf[mw],
        )


def issue_b_load_into(
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
):
    for j in range_constexpr(numAccN):
        for half in range_constexpr(kHalves):
            issue_b_value_load(
                bqf[j][half],
                j,
                half,
                kt_rt,
                b_catom,
                bq_views,
                lane_div_16,
                lane_mod_16,
                is_f8_b,
                B_NDW,
            )
    if const_expr(bsf is not None):
        issue_bscale_into(
            bsf,
            scale_chunk_tile(kt_rt, tilesPerScaleChunk),
            sc_copy_atom,
            bscale_views,
            nPairs,
            lane_div_16,
            lane_mod_16,
        )


def make_bq_fragments(is_f8_b, B_NDW, kHalves, numAccN, frag_tmpl):
    if const_expr(is_f8_b):
        return [
            [fx.make_rmem_tensor(B_NDW, Int32) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
    return [
        [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(numAccN)
    ]


def make_scale_fragments(count, sc_frag_tmpl):
    return [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(count)]


def shift_scale_word(scale, kt_rt, tilesPerScaleChunk):
    if const_expr(tilesPerScaleChunk == 1):
        return scale
    scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
    return scale.shrui(scale_shift)
