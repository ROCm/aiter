# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Strided-batched A8W4 (MXFP8 E4M3 A x MXFP4 B) preshuffle GEMM for gfx1250, written
in the idiomatic flydsl CuTe style: fx.copy over the gfx1250 TDM 2D copy atom for every
global<->LDS transfer, and fx.gemm over the MX-scaled WMMA atom for the matmul.

Per K-tile, four cooperative fx.copy(TDM2D) DMAs (A fp8, B preshuffled fp4, and both
E8M0 n32k4 scale planes) stream into a shared ``num_buffers``-stage LDS ring
(fx.SharedAllocator), overlapping DMA with the WMMA compute of earlier tiles. Because 4
streams share the ring, a deeper ring (see the launcher's ``num_buffers``) is needed to
stay above the pipeline cliff. WMMA-input fragments and the i32-packed scales are read
from the same LDS with raw ds_load (``ds_load_b128`` / ``ds_load_b32``; layouts verified
in the a8w4 WMMA probe) off the ring base pointer. The MX-scaled matmul goes through the
``WMMAScale`` MMA atom via ``fx.gemm`` (E8M0 block scales on ``scale_a``/``scale_b``).
The epilogue stages the WMMA tile back into LDS (reusing the ring at offset 0) and DMAs
it to C with fx.copy(TDM2D). Ragged / partial M tiles use the TDM copy atom's
``oob_outer`` state (A / scale-A load fault-guard, C store drop). Keeps the batched /
bmn+mbn design; requires ``tile_m % 32 == 0`` (whole scale supers) and int32-typed
scale operands (n32k4 words).

WMMA atom lowers to V_WMMA_SCALE_F32_16X16X128_F8F6F4 (wave32): A=fp4 weight, B=fp8 act.
"""

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

WMMA_M = WMMA_N = 16
WMMA_K = 128
WAVE = 32


@flyc.jit
def launch_gemm_a8w4_tdm(
    arg_c: fx.Tensor,
    arg_a: fx.Tensor,
    arg_b: fx.Tensor,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
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
    Kp = K // 2
    PACK_TK = tile_k // 2                 # packed fp4 bytes per B row per K-tile
    K_TILES = K // tile_k
    KWS = tile_k // WMMA_K                # WMMA-K steps per K-tile
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    # LDS tile geometry (bytes): A = tile_m x tile_k fp8; B = (tile_n//16) x (PACK_TK*16).
    A_TILE_B = tile_m * tile_k
    B_TILE_B = (tile_n // 16) * (PACK_TK * 16)
    A_LDS_ROW = tile_k                    # A LDS row stride (bytes)
    B_LDS_ROW = PACK_TK * 16              # B LDS n-block row stride (bytes)
    STAGE_A = ((A_TILE_B + 15) // 16) * 16
    STAGE_B = ((B_TILE_B + 15) // 16) * 16

    # ── E8M0 scale tiles (n32k4 preshuffle): i32 words, [super, K/128, 32]. ──
    SC_INNER = tile_k // 4                # i32 words per super per K-tile
    SA_SUPERS = tile_m // 32              # requires tile_m % 32 == 0
    SB_SUPERS = tile_n // 32
    SA_TILE_W = SA_SUPERS * SC_INNER
    SB_TILE_W = SB_SUPERS * SC_INNER
    STAGE_SA = ((SA_TILE_W * 4 + 15) // 16) * 16
    STAGE_SB = ((SB_TILE_W * 4 + 15) // 16) * 16
    SA_OFF = STAGE_A + STAGE_B
    SB_OFF = STAGE_A + STAGE_B + STAGE_SA
    PITCH = ((STAGE_A + STAGE_B + STAGE_SA + STAGE_SB + 127) // 128) * 128

    B_BATCH_ROWS = N // 16                # preshuffled B outer rows per batch
    N_SUPERS = (N + 31) // 32
    if const_expr(out_is_f16):
        out_elem = T.f16
        out_elem_cls = fx.Float16
    else:
        out_elem = T.bf16
        out_elem_cls = fx.BFloat16

    # TDM store staging tile (out_elem, row-major [tile_m, tile_n]) reuses the ring at 0.
    C_STORE_B = ((tile_m * tile_n * 2 + 127) // 128) * 128
    ARENA_B = max(num_buffers * PITCH, C_STORE_B)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        if const_expr(waves_per_eu <= 0):
            rocdl.disable_xdl_arb_stall()

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = rocdl.readfirstlane(T.i32, wave // n_warp)
        wave_n = rocdl.readfirstlane(T.i32, wave % n_warp)
        blk_m = bid_x * tile_m             # tile origin (M rows)
        blk_n = bid_y * tile_n
        blk_m_i = fx.index_cast(T.index, blk_m)
        blk_n_i = fx.index_cast(T.index, blk_n)
        m_idx = fx.index_cast(T.index, i32_m)
        if const_expr(batch > 1):
            bz_i = fx.index_cast(T.index, bid_z)
        else:
            bz_i = fx.index(0)

        # ── scale (n32k4, i32 words) per-batch base offset + super stride. ──
        if const_expr(layout_mbn):
            SA_OUTER_STRIDE = batch * (K // 4)
            sa_batch_off = bz_i * fx.index(K // 4)
        else:
            SA_OUTER_STRIDE = K // 4
            m_supers_i = (m_idx + fx.index(31)) // fx.index(32)
            sa_batch_off = bz_i * m_supers_i * fx.index(K // 4)
        SB_OUTER_STRIDE = K // 4
        sb_batch_off = bz_i * fx.index(N_SUPERS * (K // 4))
        # OOB (tile-start-relative): valid M rows / M-supers from the tile origin.
        a_oob = i32_m - blk_m
        sa_oob = (i32_m + fx.Int32(31)) // fx.Int32(32) - blk_m // fx.Int32(32)

        # ── C global geometry (row-major, per-batch column/row block). ──
        # mbn: C physical [M, batch*N]; bmn: C physical [batch*M, N].
        if const_expr(layout_mbn):
            C_OUTER_STRIDE = batch * N
            c_outer_off = blk_m_i
            c_inner_off = bz_i * fx.index(N) + blk_n_i
        else:
            C_OUTER_STRIDE = N
            c_outer_off = bz_i * m_idx + blk_m_i
            c_inner_off = blk_n_i
        c_oob = i32_m - blk_m

        # ── LDS ring: one SharedAllocator base; sub-ptrs + raw base indices. ──
        base_ptr = fx.SharedAllocator(static=False).allocate(ARENA_B)._ptr  # uint8 shared

        def _sub(off):
            return fx.add_offset(base_ptr, off)

        def _bidx(p):
            return fx.index_cast(T.index, fx.ptrtoint(p))

        pA = [_sub(i * PITCH) for i in range_constexpr(num_buffers)]
        pB = [_sub(i * PITCH + STAGE_A) for i in range_constexpr(num_buffers)]
        pSA = [_sub(i * PITCH + SA_OFF) for i in range_constexpr(num_buffers)]
        pSB = [_sub(i * PITCH + SB_OFF) for i in range_constexpr(num_buffers)]
        stA_idx = [_bidx(pA[i]) for i in range_constexpr(num_buffers)]
        stB_idx = [_bidx(pB[i]) for i in range_constexpr(num_buffers)]
        stSA_idx = [_bidx(pSA[i]) for i in range_constexpr(num_buffers)]
        stSB_idx = [_bidx(pSB[i]) for i in range_constexpr(num_buffers)]
        stC_idx = _bidx(base_ptr)

        def vA(s):
            return fx.Tensor(fx.make_view(pA[s], fx.make_layout((tile_m, tile_k), (A_LDS_ROW, 1))))

        def vB(s):
            return fx.Tensor(fx.make_view(pB[s], fx.make_layout((tile_n // 16, B_LDS_ROW), (B_LDS_ROW, 1))))

        def vSA(s):
            return fx.Tensor(fx.make_view(fx.recast_iter(fx.Int32, pSA[s]), fx.make_layout((SA_SUPERS, SC_INNER), (SC_INNER, 1))))

        def vSB(s):
            return fx.Tensor(fx.make_view(fx.recast_iter(fx.Int32, pSB[s]), fx.make_layout((SB_SUPERS, SC_INNER), (SC_INNER, 1))))

        def vC():
            return fx.Tensor(fx.make_view(fx.recast_iter(out_elem_cls, base_ptr), fx.make_layout((tile_m, tile_n), (tile_n, 1))))

        # ── Global tile views (per K-tile) for fx.copy(TDM2D). ──
        gA_base = fx.get_iter(arg_a)
        gB_base = fx.get_iter(arg_b)
        gSA_base = fx.get_iter(arg_scale_a)
        gSB_base = fx.get_iter(arg_scale_b)
        gC_base = fx.get_iter(arg_c)
        A_OUTER_STRIDE = (batch * K) if layout_mbn else K
        b_outer_row = bz_i * fx.index(B_BATCH_ROWS) + blk_n_i // fx.index(16)
        sa_super_off = blk_m_i // fx.index(32)
        sb_super_off = blk_n_i // fx.index(32)

        def gA(kt):
            if const_expr(layout_mbn):
                off = blk_m_i * fx.index(batch * K) + bz_i * fx.index(K) + fx.index(kt * tile_k)
            else:
                off = (bz_i * m_idx + blk_m_i) * fx.index(K) + fx.index(kt * tile_k)
            return fx.Tensor(fx.make_view(fx.add_offset(gA_base, off), fx.make_layout((tile_m, tile_k), (A_OUTER_STRIDE, 1))))

        def gB(kt):
            off = b_outer_row * fx.index(Kp * 16) + fx.index(kt * PACK_TK * 16)
            return fx.Tensor(fx.make_view(fx.add_offset(gB_base, off), fx.make_layout((tile_n // 16, PACK_TK * 16), (Kp * 16, 1))))

        def gSA(kt):
            off = sa_super_off * fx.index(SA_OUTER_STRIDE) + fx.index(kt * SC_INNER) + sa_batch_off
            return fx.Tensor(fx.make_view(fx.add_offset(gSA_base, off), fx.make_layout((SA_SUPERS, SC_INNER), (SA_OUTER_STRIDE, 1))))

        def gSB(kt):
            off = sb_super_off * fx.index(SB_OUTER_STRIDE) + fx.index(kt * SC_INNER) + sb_batch_off
            return fx.Tensor(fx.make_view(fx.add_offset(gSB_base, off), fx.make_layout((SB_SUPERS, SC_INNER), (SB_OUTER_STRIDE, 1))))

        def gC():
            off = c_outer_off * fx.index(C_OUTER_STRIDE) + c_inner_off
            return fx.Tensor(fx.make_view(fx.add_offset(gC_base, off), fx.make_layout((tile_m, tile_n), (C_OUTER_STRIDE, 1))))

        tdm_ab = fx.make_copy_atom(fx.rocdl.TDM2D(num_waves), fx.Int8)
        tdm_sc = fx.make_copy_atom(fx.rocdl.TDM2D(num_waves), fx.Int32)
        tdm_c = fx.make_copy_atom(fx.rocdl.TDM2D(num_waves), out_elem_cls)

        # Cooperative load: all waves split each tile via num_warps. A/scale-A carry
        # the runtime OOB row/super bound; B/scale-B are N-tile-aligned (no clamp).
        def issue(s, kt):
            fx.copy(tdm_ab, gA(kt), vA(s), oob_outer=a_oob)
            fx.copy(tdm_ab, gB(kt), vB(s))
            fx.copy(tdm_sc, gSA(kt), vSA(s), oob_outer=sa_oob)
            fx.copy(tdm_sc, gSB(kt), vSB(s))

        # ── LDS fragment loads (raw ds_load off the ring base index). ──
        wmb = rocdl.readfirstlane(T.i32, wave_m * warp_tile_m)
        wnb = rocdl.readfirstlane(T.i32, wave_n * warp_tile_n)

        def load_a(s, wm, ksl):
            row = wmb + wm * 16 + lane16
            b0 = row * A_LDS_ROW + ksl * 128 + kgrp * 16
            v = [Vec(lds_load_b128_raw(stA_idx[s], fx.index_cast(T.index, b0 + 32 * j))) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b(s, wn, ksl):
            nbl = wnb // 16 + wn
            b0 = nbl * B_LDS_ROW + ksl * 1024 + kgrp * 256 + lane16 * 16
            v0 = Vec(lds_load_b128_raw(stB_idx[s], fx.index_cast(T.index, b0)))
            v1 = Vec(lds_load_b128_raw(stB_idx[s], fx.index_cast(T.index, b0 + 512)))
            return v0.shuffle(v1, list(range(8)))

        def load_sa(s, wm, ksl):
            row_rel = wmb + wm * 16 + lane16
            word = (row_rel // 32) * SC_INNER + ksl * 32 + (row_rel % 32)
            return lds_load_b32_raw(stSA_idx[s], fx.index_cast(T.index, word * 4))

        def load_sb(s, wn, ksl):
            col_rel = wnb + wn * 16 + lane16
            word = (col_rel // 32) * SC_INNER + ksl * 32 + (col_rel % 32)
            return lds_load_b32_raw(stSB_idx[s], fx.index_cast(T.index, word * 4))

        # Scaled WMMA via the flydsl MX-scale atom (A=fp4 weight, B=fp8 act, E8M0).
        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32)
        )
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(fx.constant_vector(0.0, T.vec(8, T.f32)))

        TDM_PER = 4  # A + B + scale-A + scale-B cooperative tensor loads per K-tile
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
                act_frs = []
                for wm in range_constexpr(wmma_m_rep):
                    t = fx.make_rmem_tensor(16, fx.Int32)   # fp8 activation, 16 i32/lane
                    t.store(a_fr[wm][ksl])
                    act_frs.append(t)
                w_frs = []
                for wn in range_constexpr(wmma_n_rep):
                    t = fx.make_rmem_tensor(8, fx.Int32)    # fp4 weight, 8 i32/lane
                    t.store(b_fr[wn][ksl])
                    w_frs.append(t)
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        fx.gemm(
                            wmma_atom, c_frags[idx], w_frs[wn], act_frs[wm], c_frags[idx],
                            scale_a=sb_fr[wn][ksl], scale_b=sa_fr[wm][ksl],
                        )

        accs = [c_frags[idx].load().ir_value() for idx in range_constexpr(n_acc)]

        # ── Epilogue: stage the WMMA tile into LDS (raw ds_store, reusing the ring
        # at offset 0) then DMA it to C with fx.copy(TDM2D). Drain + sync first. ──
        pipeline_fence(outstanding=0)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                byte_off = fx.index_cast(T.index, (row_rel * tile_n + col_rel) * 2)
                h = fx.trunc_f(T.vec(8, out_elem), accs[wm * wmma_n_rep + wn])
                i32v = fx.vector.bitcast(T.vec(4, T.i32), h)
                lds_store_b128_raw(stC_idx, byte_off, i32v)
        workgroup_barrier()
        fx.copy(tdm_c, vC(), gC(), oob_outer=c_oob)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = i32_n // tile_n
    kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n).launch(
        grid=(gx, gy, batch), block=(block, 1, 1), stream=stream
    )
