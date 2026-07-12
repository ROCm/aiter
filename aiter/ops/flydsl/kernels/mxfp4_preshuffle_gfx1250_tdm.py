# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Strided-batched A8W4 (MXFP8 E4M3 A x MXFP4 B) preshuffle GEMM for gfx1250 with a
fully-TDM (tensor-DMA) + multi-buffer LDS pipeline.

Every global<->LDS transfer goes through TDM. Per K-tile, four cooperative
``tensor_load_2d`` DMAs (A fp8, B preshuffled fp4, and both E8M0 n32k4 scale planes)
stream into a shared ``num_buffers``-stage LDS ring, overlapping DMA with the WMMA
compute of earlier tiles. Because 4 streams share the ring, a deeper ring (see the
launcher's ``num_buffers``) is needed to stay above the pipeline cliff. Fragments and
the i32-packed scales are read from LDS (``ds_load_b128`` / ``ds_load_b32``; layouts
verified in the a8w4 WMMA probe). The epilogue stages the WMMA tile back into LDS
(reusing the ring at offset 0) and DMAs it to C with ``tensor_store_2d``. Keeps the
batched / bmn+mbn / ragged-M design; requires ``tile_m % 32 == 0`` (whole scale supers).

WMMA: V_WMMA_SCALE_F32_16X16X128_F8F6F4 (wave32), SRC0=fp4 weight, SRC1=fp8 activation.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .gemm_common_gfx1250 import (
    extract_lds_base_idx,
    lds_load_b32_raw,
    lds_load_b128_raw,
    pipeline_fence,
    store_acc_vec8_to_lds,
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
    # L2 prefetch measured net-negative here (cooperative num_warps TDM already saturates
    # the DMA queues); left plumbed but disabled (0). >0 = prefetch that many tiles ahead.
    L2_DIST = 0
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

    # ── E8M0 scale tiles (n32k4 preshuffle), streamed via TDM into the ring too. ──
    # Each i32 word packs the 4 group-scales of one 128-K WMMA step for one row/col.
    # Buffer layout (per batch) is [super(32 rows/cols), K/128 steps, 32 in-super],
    # so per super a K-tile is SC_INNER = (tile_k/128)*32 = tile_k/4 contiguous words.
    SC_INNER = tile_k // 4                # i32 words per super per K-tile
    SA_SUPERS = tile_m // 32              # requires tile_m % 32 == 0
    SB_SUPERS = tile_n // 32
    SA_TILE_W = SA_SUPERS * SC_INNER
    SB_TILE_W = SB_SUPERS * SC_INNER
    STAGE_SA = ((SA_TILE_W * 4 + 15) // 16) * 16
    STAGE_SB = ((SB_TILE_W * 4 + 15) // 16) * 16
    PITCH = ((STAGE_A + STAGE_B + STAGE_SA + STAGE_SB + 127) // 128) * 128

    B_BATCH_ROWS = N // 16                # preshuffled B outer rows per batch
    N_SUPERS = (N + 31) // 32
    if const_expr(out_is_f16):
        out_elem = T.f16
    else:
        out_elem = T.bf16

    # TDM store staging tile (bf16/f16, row-major [tile_m, tile_n]). Reuses the ring
    # LDS region (offset 0) after the K-loop drains, so total LDS = max(ring, store).
    C_STORE_B = ((tile_m * tile_n * 2 + 127) // 128) * 128

    arch = str(get_rocm_arch())
    arena = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"a8w4_tdm_{tile_m}x{tile_n}x{tile_k}_{m_warp}x{n_warp}_{num_buffers}b",
    )
    arena.ptr = max(num_buffers * PITCH, C_STORE_B)

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
        blk_m = bid_x * tile_m             # tile origin (for TDM outer offset)
        blk_n = bid_y * tile_n
        m_idx = fx.index_cast(T.index, i32_m)
        if const_expr(batch > 1):
            bz_i = fx.index_cast(T.index, bid_z)
        else:
            bz_i = fx.index(0)

        # ── E8M0 scale TDM geometry (n32k4 preshuffle), i32-word units. ──
        # sa (activation) super-major over M/32; mbn interleaves the batch dim into
        # the super stride, bmn stacks whole per-batch scale planes. sb (weight)
        # is one contiguous per-batch plane of N/32 supers. Per-batch base offsets
        # are folded into the descriptor inner offset (base ptr = whole buffer).
        if const_expr(layout_mbn):
            SA_OUTER_STRIDE = batch * (K // 4)
            sa_batch_off = bz_i * fx.index(K // 4)
        else:
            SA_OUTER_STRIDE = K // 4
            m_supers_i = (m_idx + fx.index(31)) // fx.index(32)
            sa_batch_off = bz_i * m_supers_i * fx.index(K // 4)
        SB_OUTER_STRIDE = K // 4
        sb_batch_off = bz_i * fx.index(N_SUPERS * (K // 4))
        sa_oob = (i32_m + 31) // 32

        # ── C TDM store geometry (row-major, per-batch column/row block). ──
        # mbn: C physical [M, batch*N]; batch bz occupies columns [bz*N, bz*N+N).
        # bmn: C physical [batch*M, N]; batch bz occupies rows [bz*M, bz*M+M).
        if const_expr(layout_mbn):
            C_OUTER_STRIDE = batch * N
            c_outer_off = fx.index_cast(T.index, blk_m)
            c_inner_off = bz_i * fx.index(N) + fx.index_cast(T.index, blk_n)
            c_oob = i32_m
        else:
            C_OUTER_STRIDE = N
            c_outer_off = bz_i * m_idx + fx.index_cast(T.index, blk_m)
            c_inner_off = fx.index_cast(T.index, blk_n)
            c_oob = fx.Int32(bid_z) * i32_m + i32_m

        base = arena.get_base()
        stA = [SmemPtr(base, i * PITCH, T.i8, shape=(A_TILE_B,)) for i in range_constexpr(num_buffers)]
        stB = [SmemPtr(base, i * PITCH + STAGE_A, T.i8, shape=(B_TILE_B,)) for i in range_constexpr(num_buffers)]
        stA_mem = [stA[i].get() for i in range_constexpr(num_buffers)]
        stB_mem = [stB[i].get() for i in range_constexpr(num_buffers)]
        stA_idx = [extract_lds_base_idx(stA[i]) for i in range_constexpr(num_buffers)]
        stB_idx = [extract_lds_base_idx(stB[i]) for i in range_constexpr(num_buffers)]

        SA_OFF = STAGE_A + STAGE_B
        SB_OFF = STAGE_A + STAGE_B + STAGE_SA
        stSA = [SmemPtr(base, i * PITCH + SA_OFF, T.i8, shape=(STAGE_SA,)) for i in range_constexpr(num_buffers)]
        stSB = [SmemPtr(base, i * PITCH + SB_OFF, T.i8, shape=(STAGE_SB,)) for i in range_constexpr(num_buffers)]
        stSA_mem = [stSA[i].get() for i in range_constexpr(num_buffers)]
        stSB_mem = [stSB[i].get() for i in range_constexpr(num_buffers)]
        stSA_idx = [extract_lds_base_idx(stSA[i]) for i in range_constexpr(num_buffers)]
        stSB_idx = [extract_lds_base_idx(stSB[i]) for i in range_constexpr(num_buffers)]

        # Store staging tile overlays the ring at offset 0 (used only post-K-loop).
        stC = SmemPtr(base, 0, out_elem, shape=(tile_m * tile_n,))
        stC_mem = stC.get()

        # ── TDM descriptors: cooperative num_warps DMA (each wave loads a sub-tile). ──
        def _a_offsets(kt):
            if const_expr(layout_mbn):
                return (
                    fx.index_cast(T.index, blk_m),
                    bz_i * fx.index(K) + fx.index(kt * tile_k),
                    batch * K, m_idx,
                )
            return (
                bz_i * m_idx + fx.index_cast(T.index, blk_m),
                fx.index(kt * tile_k),
                K, bz_i * m_idx + m_idx,
            )

        def _b_outer():
            return bz_i * fx.index(B_BATCH_ROWS) + fx.index_cast(
                T.index, blk_n
            ) // fx.index(16)

        def desc_a(s, kt):
            o, i, os, oob = _a_offsets(kt)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=stA_mem[s],
                global_offset=(o, i), tensor_shape=(batch * K, K), strides=(os, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=1, num_warps=num_waves,
                oob_outer_bound=oob,
            )

        def desc_b(s, kt):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=stB_mem[s],
                global_offset=(_b_outer(), fx.index(kt * PACK_TK * 16)),
                tensor_shape=(batch * B_BATCH_ROWS, Kp * 16), strides=(Kp * 16, 1),
                tile_shape=(tile_n // 16, PACK_TK * 16), elem_bytes=1, num_warps=num_waves,
            )

        def desc_sa(s, kt):
            # scale-A: outer = M-supers, inner = i32 words; per-batch base folded in.
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_a, lds_memref=stSA_mem[s],
                global_offset=(
                    fx.index_cast(T.index, blk_m) // fx.index(32),
                    fx.index(kt * SC_INNER) + sa_batch_off,
                ),
                tensor_shape=(SA_SUPERS, SC_INNER), strides=(SA_OUTER_STRIDE, 1),
                tile_shape=(SA_SUPERS, SC_INNER), elem_bytes=4, num_warps=num_waves,
                oob_outer_bound=sa_oob,
            )

        def desc_sb(s, kt):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_b, lds_memref=stSB_mem[s],
                global_offset=(
                    fx.index_cast(T.index, blk_n) // fx.index(32),
                    fx.index(kt * SC_INNER) + sb_batch_off,
                ),
                tensor_shape=(SB_SUPERS, SC_INNER), strides=(SB_OUTER_STRIDE, 1),
                tile_shape=(SB_SUPERS, SC_INNER), elem_bytes=4, num_warps=num_waves,
            )

        # Cooperative load: all waves split each tile via num_warps (more DMA MLP).
        def issue(s, kt):
            tdm_ops.tensor_load_2d(desc_a(s, kt))
            tdm_ops.tensor_load_2d(desc_b(s, kt))
            tdm_ops.tensor_load_2d(desc_sa(s, kt))
            tdm_ops.tensor_load_2d(desc_sb(s, kt))

        def l2_prefetch(kt):
            if const_expr(L2_DIST <= 0):
                return
            o, i, os, _ = _a_offsets(kt)
            tdm_ops.l2_prefetch_tile(
                arg_a, (o, i), (tile_m, tile_k), (os, 1), elem_bytes=1,
                thread_id=tid, block_threads=block,
            )
            tdm_ops.l2_prefetch_tile(
                arg_b, (_b_outer(), fx.index(kt * PACK_TK * 16)),
                (tile_n // 16, PACK_TK * 16), (Kp * 16, 1), elem_bytes=1,
                thread_id=tid, block_threads=block,
            )

        # ── LDS fragment loads (tile-local; layout verified in a8w4 probe). ──
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

        accs = [fx.constant_vector(0.0, T.vec(8, T.f32)) for _ in range_constexpr(n_acc)]

        TDM_PER = 4  # A + B + scale-A + scale-B cooperative tensor loads per K-tile
        prologue = min(num_buffers - 1, K_TILES)
        issued = 0
        for i in range_constexpr(prologue):
            issue(i % num_buffers, i)
            issued += 1
        for kt in range_constexpr(K_TILES):
            s = kt % num_buffers
            pipeline_fence(outstanding=TDM_PER * (issued - (kt + 1)))
            if const_expr(L2_DIST > 0 and kt + L2_DIST < K_TILES):
                l2_prefetch(kt + L2_DIST)
            a_fr = [[load_a(s, wm, ksl) for ksl in range_constexpr(KWS)] for wm in range_constexpr(wmma_m_rep)]
            b_fr = [[load_b(s, wn, ksl) for ksl in range_constexpr(KWS)] for wn in range_constexpr(wmma_n_rep)]
            sa_fr = [[load_sa(s, wm, ksl) for ksl in range_constexpr(KWS)] for wm in range_constexpr(wmma_m_rep)]
            sb_fr = [[load_sb(s, wn, ksl) for ksl in range_constexpr(KWS)] for wn in range_constexpr(wmma_n_rep)]
            nxt = kt + num_buffers - 1
            if const_expr(nxt < K_TILES):
                issue(nxt % num_buffers, nxt)
                issued += 1
            for ksl in range_constexpr(KWS):
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                            T.vec(8, T.f32), b_fr[wn][ksl], a_fr[wm][ksl], accs[idx],
                            sb_fr[wn][ksl], sa_fr[wm][ksl], fmtA=4, fmtB=0,
                        )

        # ── TDM store epilogue: stage the WMMA tile into LDS (row-major
        # [tile_m, tile_n]) then DMA it to C. Reuses the ring LDS at offset 0, so
        # first drain + sync to make sure no in-flight load / LDS read races it. ──
        pipeline_fence(outstanding=0)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                elem_off = fx.index_cast(T.index, row_rel * tile_n + col_rel)
                store_acc_vec8_to_lds(
                    stC_mem, elem_off, 0, accs[wm * wmma_n_rep + wn], out_elem=out_elem
                )
        workgroup_barrier()
        c_desc = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_c, lds_memref=stC_mem,
            global_offset=(c_outer_off, c_inner_off),
            tensor_shape=(tile_m, C_OUTER_STRIDE), strides=(C_OUTER_STRIDE, 1),
            tile_shape=(tile_m, tile_n), elem_bytes=2, num_warps=num_waves,
            for_store=True, oob_outer_bound=c_oob,
        )
        tdm_ops.tensor_store_2d(c_desc)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = i32_n // tile_n
    kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n).launch(
        grid=(gx, gy, batch), block=(block, 1, 1), stream=stream
    )
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        arena.finalized = False
        arena.finalize()
