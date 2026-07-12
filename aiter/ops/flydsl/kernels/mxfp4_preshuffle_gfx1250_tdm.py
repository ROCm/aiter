# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Strided-batched A8W4 (MXFP8 E4M3 A x MXFP4 B) preshuffle GEMM for gfx1250"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from .gemm_common_gfx1250 import (
    lds_load_b32_raw,
    lds_load_b128_raw,
    lds_store_b128_raw,
    pipeline_fence,
    workgroup_barrier,
)

@flyc.jit
def launch_gemm_a8w4_tdm(
    arg_c: fx.Tensor,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    warp_tile_m: Constexpr[int],
    warp_tile_n: Constexpr[int],
    out_is_f16: Constexpr[int],
    batch: Constexpr[int],
    layout_mbn: Constexpr[int],
    num_buffers: Constexpr[int],
    waves_per_eu: Constexpr[int],
):
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    Kp = K // 2
    PACK_TK = tile_k // 2
    K_TILES = K // tile_k
    KWS = tile_k // WMMA_K
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    A_TILE_B = tile_m * tile_k
    B_TILE_B = (tile_n // 16) * (PACK_TK * 16)
    A_LDS_ROW = tile_k
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = ((A_TILE_B + 15) // 16) * 16
    STAGE_B = ((B_TILE_B + 15) // 16) * 16

    SC_INNER = tile_k // 4
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    SA_TILE_W = SA_SUPERS * SC_INNER
    SB_TILE_W = SB_SUPERS * SC_INNER
    STAGE_SA = ((SA_TILE_W * 4 + 15) // 16) * 16
    STAGE_SB = ((SB_TILE_W * 4 + 15) // 16) * 16
    SA_OFF = STAGE_A + STAGE_B
    SB_OFF = STAGE_A + STAGE_B + STAGE_SA
    PITCH = ((STAGE_A + STAGE_B + STAGE_SA + STAGE_SB + 127) // 128) * 128

    B_BATCH_ROWS = N // 16
    N_SUPERS = (N + 31) // 32
    if const_expr(out_is_f16):
        out_elem = T.f16
        out_elem_cls = fx.Float16
    else:
        out_elem = T.bf16
        out_elem_cls = fx.BFloat16

    C_STORE_B = ((tile_m * tile_n * 2 + 127) // 128) * 128
    ARENA_B = max(num_buffers * PITCH, C_STORE_B)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
    ):
        if const_expr(waves_per_eu <= 0):
            rocdl.disable_xdl_arb_stall()

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        # wave is uniform across the wavefront (readfirstlane -> SGPR); values
        # derived from it (wave_m/n, wmb/wnb) stay uniform without extra hints.
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        blk_m = bid_x * tile_m             # tile origin (M rows)
        blk_n = bid_y * tile_n
        # Global tile offsets are i64 element offsets (add_offset takes any scalar);
        # this keeps the address math off the index type -> no fx.index churn.
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        m64 = fx.Int64(i32_m)
        bz64 = fx.Int64(bid_z) if const_expr(batch > 1) else 0

        # ── scale (n32k4, i32 words) per-batch base offset + super stride. ──
        if const_expr(layout_mbn):
            SA_OUTER_STRIDE = batch * (K // 4)
            sa_batch_off = bz64 * (K // 4)
        else:
            SA_OUTER_STRIDE = K // 4
            sa_batch_off = bz64 * ((m64 + 31) // 32) * (K // 4)
        SB_OUTER_STRIDE = K // 4
        sb_batch_off = bz64 * (N_SUPERS * (K // 4))
        # OOB (tile-start-relative): valid M rows (A load / C store) and M-supers
        mn_oob = i32_m - blk_m
        sa_oob = (i32_m + 31) // 32 - blk_m // 32

        if const_expr(layout_mbn):
            C_OUTER_STRIDE = batch * N
            c_off = blk_m64 * (batch * N) + bz64 * N + blk_n64
        else:
            C_OUTER_STRIDE = N
            c_off = (bz64 * m64 + blk_m64) * N + blk_n64

        base_ptr = fx.SharedAllocator(static=False).allocate(ARENA_B)._ptr  # uint8 shared
        pA = [fx.add_offset(base_ptr, i * PITCH) for i in range_constexpr(num_buffers)]
        pB = [fx.add_offset(base_ptr, i * PITCH + STAGE_A) for i in range_constexpr(num_buffers)]
        pSA = [fx.add_offset(base_ptr, i * PITCH + SA_OFF) for i in range_constexpr(num_buffers)]
        pSB = [fx.add_offset(base_ptr, i * PITCH + SB_OFF) for i in range_constexpr(num_buffers)]

        def _bidx(p):
            return fx.index_cast(T.index, fx.ptrtoint(p))

        stA_idx, stB_idx = [_bidx(p) for p in pA], [_bidx(p) for p in pB]
        stSA_idx, stSB_idx = [_bidx(p) for p in pSA], [_bidx(p) for p in pSB]
        stC_idx = _bidx(base_ptr)

        def _view(it, shape, stride):
            return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))

        def _goff(base, off):
            return fx.add_offset(base, off)

        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)
        gSA_base, gSB_base = fx.get_iter(arg_scale_a), fx.get_iter(arg_scale_b)
        gC_base = fx.get_iter(arg_c)
        A_OUTER_STRIDE = (batch * K) if layout_mbn else K
        b_outer_row = bz64 * B_BATCH_ROWS + blk_n64 // 16
        sa_super_off = blk_m64 // 32
        sb_super_off = blk_n64 // 32

        tdm_ab = fx.make_copy_atom(fx.rocdl.TDM2D(num_waves), fx.Int8)
        tdm_sc = fx.make_copy_atom(fx.rocdl.TDM2D(num_waves), fx.Int32)
        tdm_c = fx.make_copy_atom(fx.rocdl.TDM2D(num_waves), out_elem_cls)

        def issue(s, kt):
            if const_expr(layout_mbn):
                a_off = blk_m64 * (batch * K) + bz64 * K + kt * tile_k
            else:
                a_off = (bz64 * m64 + blk_m64) * K + kt * tile_k
            b_off = b_outer_row * (Kp * 16) + kt * PACK_TK * 16
            sa_off = sa_super_off * SA_OUTER_STRIDE + kt * SC_INNER + sa_batch_off
            sb_off = sb_super_off * SB_OUTER_STRIDE + kt * SC_INNER + sb_batch_off
            AT, BT = (tile_m, tile_k), (tile_n // 16, PACK_TK * 16)
            SAT, SBT = (SA_SUPERS, SC_INNER), (SB_SUPERS, SC_INNER)
            fx.copy(tdm_ab, _view(_goff(gA_base, a_off), AT, (A_OUTER_STRIDE, 1)),
                    _view(pA[s], AT, (A_LDS_ROW, 1)), oob_outer=mn_oob)
            fx.copy(tdm_ab, _view(_goff(gB_base, b_off), BT, (Kp * 16, 1)),
                    _view(pB[s], (tile_n // 16, B_LDS_ROW), (B_LDS_ROW, 1)))
            fx.copy(tdm_sc, _view(_goff(gSA_base, sa_off), SAT, (SA_OUTER_STRIDE, 1)),
                    _view(fx.recast_iter(fx.Int32, pSA[s]), SAT, (SC_INNER, 1)), oob_outer=sa_oob)
            fx.copy(tdm_sc, _view(_goff(gSB_base, sb_off), SBT, (SB_OUTER_STRIDE, 1)),
                    _view(fx.recast_iter(fx.Int32, pSB[s]), SBT, (SC_INNER, 1)))

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        def load_a(s, wm, ksl):
            row = wmb + wm * 16 + lane16
            b0 = row * A_LDS_ROW + ksl * 128 + kgrp * 16
            v = [Vec(lds_load_b128_raw(stA_idx[s], b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b(s, wn, ksl):
            nbl = wnb // 16 + wn
            b0 = nbl * B_LDS_ROW + ksl * 1024 + kgrp * 256 + lane16 * 16
            v0 = Vec(lds_load_b128_raw(stB_idx[s], b0))
            v1 = Vec(lds_load_b128_raw(stB_idx[s], b0 + 512))
            return v0.shuffle(v1, list(range(8)))

        def load_sa(s, wm, ksl):
            row_rel = wmb + wm * 16 + lane16
            word = (row_rel // 32) * SC_INNER + ksl * 32 + (row_rel % 32)
            return lds_load_b32_raw(stSA_idx[s], word * 4)

        def load_sb(s, wn, ksl):
            col_rel = wnb + wn * 16 + lane16
            word = (col_rel // 32) * SC_INNER + ksl * 32 + (col_rel % 32)
            return lds_load_b32_raw(stSB_idx[s], word * 4)

        # Scaled WMMA via the flydsl MX-scale atom (A=fp4 weight, B=fp8 act, E8M0).
        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32)
        )
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(fx.constant_vector(0.0, T.vec(8, T.f32)))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        TDM_PER = 4
        prologue = min(num_buffers - 1, K_TILES)
        issued = 0
        for i in range_constexpr(prologue):
            issue(i % num_buffers, i)
            issued += 1
        for kt in range_constexpr(K_TILES):
            s = kt % num_buffers
            pipeline_fence(outstanding=TDM_PER * (issued - (kt + 1)))
            a_fr = [[load_a(s, wm, ksl) for ksl in range_constexpr(KWS)] for wm in range_constexpr(wmma_m_rep)]
            b_fr = [[load_b(s, wn, ksl) for ksl in range_constexpr(KWS)] for wn in range_constexpr(wmma_n_rep)]
            sa_fr = [[load_sa(s, wm, ksl) for ksl in range_constexpr(KWS)] for wm in range_constexpr(wmma_m_rep)]
            sb_fr = [[load_sb(s, wn, ksl) for ksl in range_constexpr(KWS)] for wn in range_constexpr(wmma_n_rep)]
            nxt = kt + num_buffers - 1
            if const_expr(nxt < K_TILES):
                issue(nxt % num_buffers, nxt)
                issued += 1
            for ksl in range_constexpr(KWS):
                act = [_rmem(16, a_fr[wm][ksl]) for wm in range_constexpr(wmma_m_rep)]
                wt = [_rmem(8, b_fr[wn][ksl]) for wn in range_constexpr(wmma_n_rep)]
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx],
                                scale_a=sb_fr[wn][ksl], scale_b=sa_fr[wm][ksl])

        accs = [c_frags[idx].load().ir_value() for idx in range_constexpr(n_acc)]

        # ── Epilogue: stage the WMMA tile into LDS
        pipeline_fence(outstanding=0)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                h = fx.trunc_f(T.vec(8, out_elem), accs[wm * wmma_n_rep + wn])
                i32v = fx.vector.bitcast(T.vec(4, T.i32), h)
                lds_store_b128_raw(stC_idx, (row_rel * tile_n + col_rel) * 2, i32v)
        workgroup_barrier()
        fx.copy(tdm_c, _view(fx.recast_iter(out_elem_cls, base_ptr), (tile_m, tile_n), (tile_n, 1)),
                _view(_goff(gC_base, c_off), (tile_m, tile_n), (C_OUTER_STRIDE, 1)), oob_outer=mn_oob)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = N // tile_n
    kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m).launch(
        grid=(gx, gy, batch), block=(block, 1, 1), stream=stream
    )
