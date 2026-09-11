# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 epilogue instruction scheduling (layout API)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as _arith
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float32,
    Int8,
    Int16,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import (
    _inline_dpp_pair_amax,
    _inline_dpp_quad_amax,
    _udiv,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)

STORE_CACHE_MODIFIER = 2

_FP8_E8M0_SHIFT = 7

_G2_EPI_LANES = 32


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
    output_n_base=0,
    output_width=None,
    reduce_store_cache_modifier=None,
    route_guard=False,
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
    if reduce_store_cache_modifier is not None and (
        not use_reduce or route_out_fp8 or enable_bias or g2_defer_weight
    ):
        raise ValueError("custom BF16 store policies require unweighted reduce output")
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
    out_i8 = flat_buffer(arg_out, T.i8, 4)

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
                if const_expr(output_width is None):
                    row_pitch = N_OUT + _udiv(N_OUT, fx.Int32(g2_scale_blk))
                else:
                    ow = fx.Int32(output_width)
                    row_pitch = ow + _udiv(ow, fx.Int32(g2_scale_blk))
                if const_expr(g2_out_pitch_align > 0):
                    al = fx.Int32(g2_out_pitch_align)
                    row_pitch = ((row_pitch + al - fx.Int32(1)) // al) * al
                row_base_addr = out_row * fx.Int64(row_pitch)
            elif const_expr(output_width is None):
                row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(
                    n_block_idx * BN + col_start
                )
            else:
                row_base_addr = out_row * fx.Int64(fx.Int32(output_width)) + fx.Int64(
                    n_block_idx * BN + col_start - fx.Int32(output_n_base)
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
                    store_col = (
                        col_g0
                        if const_expr(output_width is None)
                        else col_g0 - fx.Int32(output_n_base)
                    )
                    row_extent = (
                        N_OUT
                        if const_expr(output_width is None)
                        else fx.Int32(output_width)
                    )
                    row_val_off = row_base_addr + fx.Int64(store_col)
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
                        + fx.Int64(row_extent)
                        + fx.Int64(_udiv(store_col, fx.Int32(g2_scale_blk)))
                    )
                    scale_frag = fx.make_rmem_tensor(1, Int8)
                    scale_frag.store(Vec.from_elements([e8m0.to(Int8)], Int8))
                    fx.copy(store_i8, scale_frag, out_i8[None, scale_off])

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
            if const_expr(route_guard):
                route_slot = packed[mr] >> fx.Int32(24)
                if token_id < i32_M and route_slot < fx.Int32(topk):
                    store_one_mr(mr)
            else:
                if token_id < i32_M:
                    store_one_mr(mr)

        store_if_valid(token_id, mr)


def nonatomic_bf16_epilog(
    accm, arg_out, m_row, n_block_idx, wave, lane, N_OUT, BN, kMChunks
):
    """Unweighted store into ``flat_out[sorted_row, hidden]`` (epilog=scatter); host scatter_reduce applies weights."""
    numAccN = (BN // 4) // 16
    row_base = m_row + (lane // 16) * 4
    gn_base = n_block_idx * BN + wave * (BN // 4) + (lane % 16)
    out_ptr = global_typed_ptr(arg_out, T.bf16, align=2)
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                bf = Vec.from_elements([vec[v]], Float32).to(BFloat16)
                out_ptr[(row_base + i * 16 + v) * N_OUT + gn_base + J * 16] = bf[0]
