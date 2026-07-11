# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Strided-batched A8W4 (MXFP8 E4M3 A x MXFP4 B) preshuffle GEMM for gfx1250 (wave32
WMMA). out[b] = dequant(x[b]) @ dequant(w[b]).T with per-1x32 E8M0 scales folded into
V_WMMA_SCALE_F32_16X16X128_F8F6F4.

Operand layouts (prepared once by the caller, off the launch path):
  A (fp8 activation): plain row-major codes [B,M,K] (bmn) / [M,B,K] (mbn), 1 byte/code.
  B (fp4 weight):     shuffle_weight_gfx1250 -> [B, N//16, (K//2)*16].
  A-scale / B-scale:  shuffle_scale_n32k4    -> [B, ceil(rows/32), (K//32)*32] E8M0.

The WMMA hardware fragment/scale/accumulator lane layouts were verified empirically
(see the A8W4 layout probe): the fp8 A operand is read straight from row-major memory,
the fp4 B operand from the 16x16 gfx1250 preshuffle, and each lane's E8M0 scale operand
is one contiguous i32 (4 blocks of 32 = one WMMA-K=128 step) of the n32k4 layout.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

WMMA_M = WMMA_N = 16
WMMA_K = 128
WAVE = 32
SCALE_BLOCK = 32


@flyc.jit
def launch_gemm_a8w4(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    warp_tile_m: Constexpr[int],
    warp_tile_n: Constexpr[int],
    out_is_f16: Constexpr[int],
    batch: Constexpr[int],
    layout_mbn: Constexpr[int],
    waves_per_eu: Constexpr[int],
):
    """Direct @flyc.jit launcher. a8w4 fixed (fp8 A x fp4 B), E8M0 scales, wave32 WMMA.
    layout_mbn selects the deepseek-v4 [M,B,*] physical layout for A / A-scale / C."""
    Kp = K // 2  # packed fp4 bytes per B row
    LDA_I32 = K // 4  # fp8 A row stride (i32), bmn
    B_ROW_I32 = K * 2  # preshuffled B n-block row stride (i32) = (Kp*16)//4
    SCALE_SUPER_I32 = K // 4  # n32k4 super-row stride (i32), per batch
    K_STEPS = K // WMMA_K
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep

    # Per-batch buffer sizes (compile-time for B / B-scale; A / A-scale carry runtime M).
    # Preshuffled B is [N//16, (K//2)*16] bytes = N*(K//2) bytes per batch.
    B_BATCH_I32 = N * Kp // 4
    N_SUPERS = (N + 31) // 32
    BS_BATCH_I32 = N_SUPERS * (K // 32 * 32) // 4

    if const_expr(out_is_f16):
        out_elem = T.f16
    else:
        out_elem = T.bf16

    block = m_warp * n_warp * WAVE

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = rocdl.readfirstlane(T.i32, wave // n_warp)
        wave_n = rocdl.readfirstlane(T.i32, wave % n_warp)

        bx_m = bid_x * fx.Int32(tile_m) + wave_m * fx.Int32(warp_tile_m)
        by_n = bid_y * fx.Int32(tile_n) + wave_n * fx.Int32(warp_tile_n)
        bz = fx.Int64(bid_z)

        m64 = fx.Int64(i32_m)
        m_supers = fx.Int64((i32_m + fx.Int32(31)) // fx.Int32(32))

        # ── Per-batch base addresses + strides (bmn contiguous / mbn interleaved). ──
        if const_expr(layout_mbn):
            lda_i32 = fx.Int32(batch) * fx.Int32(LDA_I32)
            a_base = arg_a + bz * fx.Int64(K)
            sa_super_i32 = fx.Int32(batch) * fx.Int32(SCALE_SUPER_I32)
            sa_base = arg_scale_a + bz * fx.Int64(K)
            ldc = fx.Int32(batch) * i32_n
            c_base = arg_c + bz * (fx.Int64(i32_n) * fx.Int64(2))
        else:
            lda_i32 = fx.Int32(LDA_I32)
            a_base = arg_a + bz * (m64 * fx.Int64(K))
            sa_super_i32 = fx.Int32(SCALE_SUPER_I32)
            sa_base = arg_scale_a + bz * (m_supers * fx.Int64(K // 32 * 32))
            ldc = i32_n
            c_base = arg_c + bz * (m64 * fx.Int64(i32_n) * fx.Int64(2))
        b_base = arg_b + bz * fx.Int64(B_BATCH_I32 * 4)
        sb_base = arg_scale_b + bz * fx.Int64(BS_BATCH_I32 * 4)

        # A / A-scale / C bounded to M (ragged-M OOB -> read 0 / dropped store).
        a_nrec = (m64 * fx.Int64(K)) if not const_expr(layout_mbn) else (
            (m64 - fx.Int64(1)) * fx.Int64(lda_i32 * 4) + fx.Int64(K)
        )
        a_rsrc = buffer_ops.create_buffer_resource_from_addr(
            a_base, num_records_bytes=a_nrec
        )
        sa_nrec = (m_supers * fx.Int64(K // 32 * 32)) if not const_expr(layout_mbn) else (
            (m_supers - fx.Int64(1)) * fx.Int64(sa_super_i32 * 4)
            + fx.Int64(K // 32 * 32)
        )
        sa_rsrc = buffer_ops.create_buffer_resource_from_addr(
            sa_base, num_records_bytes=sa_nrec
        )
        c_nrec = (
            ((m64 - fx.Int64(1)) * fx.Int64(ldc) + fx.Int64(i32_n)) * fx.Int64(2)
        )
        c_rsrc = buffer_ops.create_buffer_resource_from_addr(
            c_base, num_records_bytes=c_nrec
        )
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(b_base)
        sb_rsrc = buffer_ops.create_buffer_resource_from_addr(sb_base)

        def load_a(wm, ks):
            row = bx_m + fx.Int32(wm * 16) + lane16
            base = row * lda_i32 + fx.Int32(ks * 32) + kgrp * fx.Int32(4)
            v = [
                Vec(buffer_ops.buffer_load(a_rsrc, base + fx.Int32(8 * j),
                                           vec_width=4, dtype=T.i32))
                for j in range_constexpr(4)
            ]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b(wn, ks):
            nb = (by_n + fx.Int32(wn * 16)) // fx.Int32(16)
            base = nb * fx.Int32(B_ROW_I32) + fx.Int32(ks * 256) + kgrp * fx.Int32(64) \
                + lane16 * fx.Int32(4)
            v0 = Vec(buffer_ops.buffer_load(b_rsrc, base, vec_width=4, dtype=T.i32))
            v1 = Vec(buffer_ops.buffer_load(b_rsrc, base + fx.Int32(128),
                                            vec_width=4, dtype=T.i32))
            return v0.shuffle(v1, list(range(8)))

        def load_sa(wm, ks):
            row = bx_m + fx.Int32(wm * 16) + lane16
            super_ = row // fx.Int32(32)
            row32 = row % fx.Int32(32)
            off = super_ * sa_super_i32 + fx.Int32(ks * 32) + row32
            return buffer_ops.buffer_load(sa_rsrc, off, vec_width=1, dtype=T.i32)

        def load_sb(wn, ks):
            col = by_n + fx.Int32(wn * 16) + lane16
            super_ = col // fx.Int32(32)
            col32 = col % fx.Int32(32)
            off = super_ * fx.Int32(SCALE_SUPER_I32) + fx.Int32(ks * 32) + col32
            return buffer_ops.buffer_load(sb_rsrc, off, vec_width=1, dtype=T.i32)

        accs = [arith.constant_vector(0.0, T.vec(8, T.f32)) for _ in range_constexpr(n_acc)]
        for ks in range_constexpr(K_STEPS):
            a_fr = [load_a(wm, ks) for wm in range_constexpr(wmma_m_rep)]
            b_fr = [load_b(wn, ks) for wn in range_constexpr(wmma_n_rep)]
            sa_fr = [load_sa(wm, ks) for wm in range_constexpr(wmma_m_rep)]
            sb_fr = [load_sb(wn, ks) for wn in range_constexpr(wmma_n_rep)]
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_fr[wn],
                        a_fr[wm],
                        accs[idx],
                        sb_fr[wn],
                        sa_fr[wm],
                        fmtA=4,
                        fmtB=0,
                    )

        # Epilogue: lane owns row = bx_m + wm*16 + lane16, cols [base : base+8].
        for wm in range_constexpr(wmma_m_rep):
            row = bx_m + fx.Int32(wm * 16) + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_base = by_n + fx.Int32(wn * 16) + kgrp * fx.Int32(8)
                acc = accs[wm * wmma_n_rep + wn]
                h = arith.trunc_f(T.vec(8, out_elem), acc)
                i32v = vector.bitcast(T.vec(4, T.i32), h)
                off = (row * ldc + col_base) // fx.Int32(2)
                buffer_ops.buffer_store(i32v, c_rsrc, off)

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
    gy = i32_n // fx.Int32(tile_n)
    kernel(
        c_addr, a_addr, b_addr, sa_addr, sb_addr, i32_m, i32_n,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=(gx, gy, batch), block=(block, 1, 1), stream=stream)
