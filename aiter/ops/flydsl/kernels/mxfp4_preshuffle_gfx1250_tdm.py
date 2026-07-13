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
    a_is_fp4: Constexpr[int],
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

    A_PACK = 2 if a_is_fp4 else 1
    A_ROW_B = tile_k // A_PACK      # A tile-row data bytes
    A_KROW = K // A_PACK            # A global full-row bytes
    A_KSTEP = WMMA_K // A_PACK      # A LDS bytes per WMMA-K step
    ACT_ELEM = fx.Float4E2M1FN if a_is_fp4 else fx.Float8E4M3FN
    ACT_NDW = 8 if a_is_fp4 else 16  # act fragment i32 count (fp4 8, fp8 16)

    # Pad the A LDS row stride (mirrors gemm_mxscale's LDS_PAD_A_BYTES=16): 16 lanes
    # read rows at this stride in load_a, so an unpadded power-of-two stride collides
    # all 16 on the same LDS banks. +16B (a ds_load_b128 width) shifts each row off.
    LDS_PAD_A = 16
    A_LDS_ROW = A_ROW_B + LDS_PAD_A
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = ((tile_m * A_LDS_ROW + 15) // 16) * 16
    STAGE_B = (((tile_n // 16) * B_LDS_ROW + 15) // 16) * 16

    SC_INNER = tile_k // 4
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    STAGE_SA = ((SA_SUPERS * SC_INNER * 4 + 15) // 16) * 16
    STAGE_SB = ((SB_SUPERS * SC_INNER * 4 + 15) // 16) * 16
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

        # ── scale (n32k4 i32-word) super stride + per-batch base, and C geometry;
        # mbn interleaves batch into the super/row stride, bmn stacks per-batch. ──
        if const_expr(layout_mbn):
            SA_OUTER_STRIDE, C_OUTER_STRIDE = batch * (K // 4), batch * N
            sa_batch_off = bz64 * (K // 4)
            c_off = blk_m64 * (batch * N) + bz64 * N + blk_n64
        else:
            SA_OUTER_STRIDE, C_OUTER_STRIDE = K // 4, N
            sa_batch_off = bz64 * ((m64 + 31) // 32) * (K // 4)
            c_off = (bz64 * m64 + blk_m64) * N + blk_n64
        SB_OUTER_STRIDE = K // 4
        sb_batch_off = bz64 * (N_SUPERS * (K // 4))
        # OOB (tile-start-relative): valid M rows (A load / C store), M-supers (scale-A).
        mn_oob = i32_m - blk_m
        sa_oob = (i32_m + 31) // 32 - blk_m // 32

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

        def _gv(base, off, shape, stride):  # offset a global base -> tile view
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        def _lv(ptr, shape, stride):  # LDS ring-slot view
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _tdm(gt, outer=None):  # 2-D TDM copy atom; `outer` clamps dim0 (None = no OOB clamp)
            return fx.rocdl.make_tdm_atom(gt, [outer, None], num_warps=num_waves)

        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)
        gSA_base, gSB_base = fx.get_iter(arg_scale_a), fx.get_iter(arg_scale_b)
        gC_base = fx.get_iter(arg_c)
        A_OUTER_STRIDE = (batch * A_KROW) if layout_mbn else A_KROW
        b_outer_row = bz64 * B_BATCH_ROWS + blk_n64 // 16
        sa_super_off = blk_m64 // 32
        sb_super_off = blk_n64 // 32

        # A row bytes / K-tile advance follow the A dtype (fp8 = tile_k, fp4 = tile_k/2).
        AT, BT = (tile_m, A_ROW_B), (tile_n // 16, PACK_TK * 16)
        SAT, SBT = (SA_SUPERS, SC_INNER), (SB_SUPERS, SC_INNER)
        if const_expr(layout_mbn):
            a_off0 = blk_m64 * (batch * A_KROW) + bz64 * A_KROW
        else:
            a_off0 = (bz64 * m64 + blk_m64) * A_KROW
        b_off0 = b_outer_row * (Kp * 16)
        sa_off0 = sa_super_off * SA_OUTER_STRIDE + sa_batch_off
        sb_off0 = sb_super_off * SB_OUTER_STRIDE + sb_batch_off
        gtA0 = _gv(gA_base, a_off0, AT, (A_OUTER_STRIDE, 1))
        gtB0 = _gv(gB_base, b_off0, BT, (Kp * 16, 1))
        gtSA0 = _gv(gSA_base, sa_off0, SAT, (SA_OUTER_STRIDE, 1))
        gtSB0 = _gv(gSB_base, sb_off0, SBT, (SB_OUTER_STRIDE, 1))
        atomA = fx.rocdl.make_tdm_atom(
            gtA0, [mn_oob, None], num_warps=num_waves,
            pad_interval=A_ROW_B, pad_amount=LDS_PAD_A,
        )
        atomB = _tdm(gtB0)
        atomSA, atomSB = _tdm(gtSA0, sa_oob), _tdm(gtSB0)
        _adv = fx.rocdl.advance_tdm_atom

        def issue(s, kt):  # bump imm_offset by the K-tile byte delta (base fixed)
            fx.copy(_adv(atomA, kt * A_ROW_B), gtA0, _lv(pA[s], AT, (A_LDS_ROW, 1)))
            fx.copy(_adv(atomB, kt * PACK_TK * 16), gtB0, _lv(pB[s], BT, (B_LDS_ROW, 1)))
            fx.copy(_adv(atomSA, kt * SC_INNER * 4), gtSA0,
                    _lv(fx.recast_iter(fx.Int32, pSA[s]), SAT, (SC_INNER, 1)))
            fx.copy(_adv(atomSB, kt * SC_INNER * 4), gtSB0,
                    _lv(fx.recast_iter(fx.Int32, pSB[s]), SBT, (SC_INNER, 1)))

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        def load_a(s, wm, ksl):
            row = wmb + wm * 16 + lane16
            b0 = row * A_LDS_ROW + ksl * A_KSTEP + kgrp * 16
            if const_expr(a_is_fp4):
                v0 = Vec(lds_load_b128_raw(stA_idx[s], b0))
                v1 = Vec(lds_load_b128_raw(stA_idx[s], b0 + 32))
                return v0.shuffle(v1, list(range(8)))
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

        # Scaled WMMA via the flydsl MX-scale atom (A=fp4 weight, B=fp4/fp8 act, E8M0).
        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, fx.Float4E2M1FN, ACT_ELEM, fx.Float32)
        )
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(fx.constant_vector(0.0, T.vec(8, T.f32)))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        front_wm = (wmma_m_rep + 1) // 2
        _WM_GROUPS = [list(range(front_wm)), list(range(front_wm, wmma_m_rep))]

        def _emit_group(s, ksl, wm_list, wt, sa_fr, sb_fr):
            if const_expr(len(wm_list) == 0):
                return
            act = [_rmem(ACT_NDW, load_a(s, wm, ksl)) for wm in wm_list]
            rocdl.s_wait_dscnt(0)
            for i in range_constexpr(len(wm_list)):
                wm = wm_list[i]
                for wn_raw in range_constexpr(wmma_n_rep):
                    wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                    idx = wm * wmma_n_rep + wn
                    fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[i], c_frags[idx],
                            scale_a=sb_fr[wn], scale_b=sa_fr[wm])

        TDM_PER = 4
        prologue = min(num_buffers - 1, K_TILES)
        issued = 0
        for i in range_constexpr(prologue):
            issue(i % num_buffers, i)
            issued += 1
        for kt in range_constexpr(K_TILES):
            s = kt % num_buffers
            pipeline_fence(outstanding=TDM_PER * (issued - (kt + 1)))
            # e8m0 scales up front (small ds_load_b32); A and B weight are streamed
            # per WMMA-K step so only one k-step's fragments are live at a time.
            sa_fr = [[load_sa(s, wm, ksl) for ksl in range_constexpr(KWS)] for wm in range_constexpr(wmma_m_rep)]
            sb_fr = [[load_sb(s, wn, ksl) for ksl in range_constexpr(KWS)] for wn in range_constexpr(wmma_n_rep)]
            nxt = kt + num_buffers - 1
            if const_expr(nxt < K_TILES):
                issue(nxt % num_buffers, nxt)
                issued += 1
            for ksl in range_constexpr(KWS):
                wt = [_rmem(8, load_b(s, wn, ksl)) for wn in range_constexpr(wmma_n_rep)]
                sa_k = [sa_fr[wm][ksl] for wm in range_constexpr(wmma_m_rep)]
                sb_k = [sb_fr[wn][ksl] for wn in range_constexpr(wmma_n_rep)]
                for wm_list in _WM_GROUPS:
                    _emit_group(s, ksl, wm_list, wt, sa_k, sb_k)

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
        gtC = _gv(gC_base, c_off, (tile_m, tile_n), (C_OUTER_STRIDE, 1))
        fx.copy(_tdm(gtC, mn_oob),
                _lv(fx.recast_iter(out_elem_cls, base_ptr), (tile_m, tile_n), (tile_n, 1)), gtC)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = N // tile_n
    kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m).launch(
        grid=(gx, gy, batch), block=(block, 1, 1), stream=stream
    )


# Match the tuned no-batch gemm_mxscale_gfx1250: enable the amdgpu expert
# instruction scheduler (denser WMMA/DMA interleave for the compute-bound path).
launch_gemm_a8w4_tdm.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
