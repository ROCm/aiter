# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from the amd-kernels project's gemm/bf16_concat_linear_n1536_k30720 kernel (revision fd233ef;
# SPDX-License-Identifier: Apache-2.0, SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project):
# the idea of reading the five activation taps in place instead of materializing their concat. The kernel body is
# the small-M ring GEMM of gemm_a16w16_ring_gfx950.py with a split-K grid.
"""Split-K concat-then-GEMM on the small-M ring GEMM, gfx950.

``part[ks, M, N]`` (fp32) ``= concat(x0..x4)[M, K] @ w[N, K]^T`` over K split ``ks`` ways. A split never straddles a
tap, so a workgroup reads ONE tap, chosen once from the five kernel-argument base pointers (no host-side pointer
table: the launch is CUDA-graph capturable). The caller sums the ``ks`` partials in ascending order.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .gemm_a16w16_ring_gfx950 import _barrier, _lds_ptr, _sb

NWV = 4
W_CM = 2
NSEG = 5
# Weight layouts (same ids as the ring kernel's): row-major [N, K], ring tiles [N/16, K/64, 16, 64] and AITER's
# shuffle_weight(w, (16, 16)) layout [N/16, K/32, 4, 16, 8], read in place (no second weight copy).
LAYOUT_ROW, LAYOUT_RING, LAYOUT_AITER = 0, 1, 2


@functools.cache
def compile_cat_ring(
    n: int,
    seg_k: int,
    mt: int,
    ks: int,
    r: int,
    device: int = 0,
    nr: int = 4,
    layout: int = LAYOUT_RING,
):
    """``device`` only keys the cache: a compiled handle must not be shared across GPUs."""
    K = NSEG * seg_k
    NB_TOTAL = K // 64
    assert layout in (LAYOUT_ROW, LAYOUT_RING, LAYOUT_AITER), layout
    assert ks % NSEG == 0 and NB_TOTAL % ks == 0 and 1 <= mt <= 8
    STG = NB_TOTAL // ks
    KSPS = ks // NSEG
    assert nr == NWV and n % (16 * nr) == 0 and STG >= r
    NT = n // (16 * nr)
    MT = mt
    NWT = NWV
    XD = MT * 2
    XPW = (XD + NWT - 1) // NWT
    PER = 2 + XPW  # one weight fragment per wave -> 2 DMA (8 rows each) + x DMAs
    WF = nr + 1
    SLOT = (WF + MT) * 2048
    LDS = r * SLOT
    assert LDS <= 160 * 1024 and (r - 1) * PER <= 63, (LDS, r, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"aiter_cat_ring_n{n}_k{K}_mt{MT}_ks{ks}_r{r}_l{layout}"

    @flyc.kernel(name=name, known_block_size=[64 * NWT, 1, 1])
    def kernel(
        x0: fx.Int64,
        x1: fx.Int64,
        x2: fx.Int64,
        x3: fx.Int64,
        x4: fx.Int64,
        arg_w: fx.Int64,
        arg_y: fx.Int64,
        i32_m: fx.Int32,
        i32_xs: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        kidx = bid % ks
        nt = bid // ks
        n0 = nt * (16 * nr)
        seg = kidx // KSPS
        kb0 = (kidx % KSPS) * STG  # first stage within the segment
        wb0 = kidx * STG  # first stage within the full K (weight)
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xa = (seg == 1).select(x1, x0)
        xa = (seg == 2).select(x2, xa)
        xa = (seg == 3).select(x3, xa)
        xa = (seg == 4).select(x4, xa)
        xr = buffer_ops.create_buffer_resource_from_addr(
            xa, num_records_bytes=(i32_m - 1) * i32_xs + seg_k * 2
        )
        wr = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=n * K * 2
        )
        yr = buffer_ops.create_buffer_resource_from_addr(
            arg_y, num_records_bytes=i32_m * (ks * n * 4)
        )
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        w_src, w_dst = [], []
        rn = wave
        for jj in range_constexpr(2):
            pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
            if const_expr(layout == LAYOUT_RING):
                off = (n0 // 16 + rn) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
            elif const_expr(layout == LAYOUT_AITER):
                off = (
                    (n0 // 16 + rn) * (K * 32)
                    + (pc // 4) * 1024
                    + (pc % 4) * 256
                    + (jj * 8 + r8) * 16
                )
            else:
                off = (n0 + rn * 16 + jj * 8 + r8) * (K * 2) + pc * 16
            w_src.append(off)
            w_dst.append((rn * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave + i * NWT
            d = (d < XD).select(d, fx.Int32(XD - 1))
            jx = d % 2
            row = (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * i32_xs + px * 16)
            x_dst.append((WF * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % r) * SLOT + dst))
            lp = fx.to_llvm_ptr(
                fx.inttoptr(
                    fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)
                )
            )
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs,
                lp,
                fx.Int32(16).ir_value(),
                fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(),
                fx.Int32(0).ir_value(),
                aux=aux,
            )

        def dma(b):
            for q in range_constexpr(2):
                issue(
                    wr,
                    w_dst[q],
                    w_src[q],
                    b,
                    (wb0 + b) * (128 if layout == LAYOUT_ROW else 2048),
                    aux_w,
                )
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, (kb0 + b) * 128, aux_x)

        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [
            hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16
            for kk in range_constexpr(2)
        ]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % r) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)]

        def read_ops(b, kk):
            return (rd(b, wave, kk), [rd(b, WF + m, kk) for m in range_constexpr(MT)])

        def mfmas(ops):
            wv, xv = ops
            for m in range_constexpr(MT):
                # Raw MFMA, not fx.make_mma_atom + fx.gemm: the vmcnt / lgkmcnt waits and sched_barrier groups
                # of this ring depend on this exact MFMA issue order (same as gemm_a16w16_ring_gfx950).
                acc[m] = Vec(
                    fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv, xv[m], acc[m], 0, 0, 0]
                    )
                )

        for b in range_constexpr(r):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(r - 1) * PER)
        _barrier()
        _sb()
        ops_a = read_ops(0, 0)
        for b in range_constexpr(STG):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < STG:
                fx.rocdl.s_waitcnt(
                    lgkmcnt=0, vmcnt=(min(b + r - 1, STG - 1) - (b + 1)) * PER
                )
                _barrier()
                _sb()
                ops_a = read_ops(b + 1, 0)
                _sb()
                if b + r < STG:
                    dma(b + r)
                _sb()
            mfmas(ops_b)

        nn = n0 + wave * 16 + g * 4
        for m in range_constexpr(MT):
            row = m * 16 + l16
            buffer_ops.buffer_store(
                acc[m], yr, (kidx * i32_m + row) * n + nn, mask=row < i32_m
            )

    @flyc.jit
    def launch(
        x0: fx.Int64,
        x1: fx.Int64,
        x2: fx.Int64,
        x3: fx.Int64,
        x4: fx.Int64,
        arg_w: fx.Int64,
        arg_y: fx.Int64,
        i32_m: fx.Int32,
        i32_xs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x0, x1, x2, x3, x4, arg_w, arg_y, i32_m, i32_xs).launch(
            grid=(NT * ks, 1, 1), block=(64 * NWT, 1, 1), stream=stream
        )

    return launch
