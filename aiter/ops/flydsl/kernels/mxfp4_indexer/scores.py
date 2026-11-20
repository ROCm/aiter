# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block scores of the MXFP4 index cache: for every row, the maximum logit of each 8-token block of its context,
logit(t) = sum_h w_h relu(q_h . k_t) over 32 heads of 128 dims (e2m1 codes, e8m0 scales per 32).

A page of ``P`` entries is cut into 32-token groups: the groups' K codes are stored as [chunk 0..3][token][16 B] (the
16 bytes of K elements 32 chunk .. + 31), their scales as [scale % 2][token][scale // 2]. A tile is one 32-token group:
the A operand of ``v_mfma_scale_f32_32x32x64_f8f6f4`` holds the row's heads (lane (head, h) the 32 codes 32 (2 kb + h)
of K step kb), the B operand the tile's tokens, so a lane's 16 accumulators are 16 heads of one token (lane ^ 32 has the
other 16): the relu-weighted head sum is one fma chain and a 32-lane swap, and the maximum over a block's 8 tokens
(8 adjacent lanes) three DPP steps. A wave scores ``rpw`` rows of the request, a workgroup (``ceil(rows / rpw)`` waves)
the request's rows, every wave reading each K tile once: the tile's K loads of the next group are in flight during the
current group's MFMAs. Request r's work is split in ``nseg`` equal segments of tiles, so the grid covers any lengths
without a schedule; a segment past a request's length exits at once. ``tokens=True`` (contexts up to 16384 entries,
``tokens``): the walk stores every token's logit (the lane's own, 4 B, coalesced) instead of the block maxima.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .. import buffer_ops
from .common import I64, NEG_INF, F, H, I, bf, bi, dpp, maxf, mfma_fp4, pair_logit

ROWS_PER_WAVE = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4}


@functools.cache
def compile_block_scores(next_n: int, page_entries: int, tokens: bool = False):
    """Block maxima of ``next_n`` rows per request over pages of ``page_entries`` (64 or 128) entries; ``tokens``: the
    logit of every token instead (``scores`` rows hold ``tier + 64`` logits; for short contexts, ``tokens``).
    """
    P, NN = page_entries, next_n
    RPW = ROWS_PER_WAVE[NN]
    NW = -(-NN // RPW)
    ALIGN = 4  # tiles: a pair is half a page at 64 entries
    assert P in (64, 128)
    name = f"indexer_mxfp4_{'token_logits' if tokens else 'block_scores'}_h32_d128_nn{NN}_p{P}"

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        q: fx.Int64,
        q_scale: fx.Int64,
        weights: fx.Int64,
        kv: fx.Int64,
        block_table: fx.Int64,
        row_ends: fx.Int64,
        scores: fx.Int64,
        page_stride: fx.Int64,
        bt_stride: fx.Int32,
        scores_stride: fx.Int32,
        nseg: fx.Int32,
    ):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        col = lane % 32
        h = lane // 32
        req = fx.block_idx.y
        seg = fx.block_idx.x

        def res(addr, n):
            return buffer_ops.create_buffer_resource_from_addr(
                addr, num_records_bytes=n
            )

        er = res(row_ends + I64(req) * (NN * 4), NN * 4)
        ends, mx = [], I(0)
        for r in range_constexpr(NN):
            e = I(buffer_ops.buffer_load(er, r, vec_width=1, dtype=I, is_scalar=True))
            ends.append(e)
            mx = (e > mx).select(e, mx)
        ntiles = (mx + 31) // 32
        per_tile = 32 if tokens else 4  # scores written a tile
        ntiles = (ntiles * per_tile > scores_stride - 64).select(
            (scores_stride - 64) // per_tile, ntiles
        )
        seg_tiles = (((ntiles + nseg - 1) // nseg + ALIGN - 1) // ALIGN) * ALIGN
        t0 = seg * seg_tiles
        t1 = (ntiles < t0 + seg_tiles).select(ntiles, t0 + seg_tiles)
        if t0 < t1:
            qr = res(q + I64(req) * (NN * H * 64), NN * H * 64)
            qsr = res(q_scale + I64(req) * (NN * H * 4), NN * H * 4)
            wr = res(weights + I64(req) * (NN * H * 4), NN * H * 4)
            btr = res(block_table + I64(req) * I64(bt_stride) * 4, I64(bt_stride) * 4)
            sr = res(
                scores + I64(req) * NN * I64(scores_stride) * 4,
                I64(NN) * I64(scores_stride) * 4,
            )
            rows, qv, sa, wv, end = [], [], [], [], []
            for r in range_constexpr(RPW):
                rg = wave * RPW + r
                rg = (rg < NN).select(rg, NN - 1)
                rows.append(rg)
                end.append(
                    I(
                        buffer_ops.buffer_load(
                            er, rg, vec_width=1, dtype=I, is_scalar=True
                        )
                    )
                )
                qsw = I(buffer_ops.buffer_load(qsr, rg * H + col, vec_width=1, dtype=I))
                qv.append(
                    [
                        Vec(
                            buffer_ops.buffer_load(
                                qr,
                                (rg * H * 64 + col * 64 + kb * 32 + h * 16) // 4,
                                vec_width=4,
                                dtype=I,
                            )
                        )
                        for kb in range_constexpr(2)
                    ]
                )
                sa.append(
                    [(qsw >> ((2 * kb + h) * 8)) & 0xFF for kb in range_constexpr(2)]
                )
                wl = []
                for j in range_constexpr(4):
                    v = Vec(
                        buffer_ops.buffer_load(
                            wr, rg * H + 8 * j + 4 * h, vec_width=4, dtype=F
                        )
                    )
                    wl += [F(v[0]), F(v[1]), F(v[2]), F(v[3])]
                wv.append(wl)

            def load_pair(t):
                """K codes and scales of tiles t, t + 1 (one page: t is even)."""
                tt = (t < t1).select(t, t1 - 1)
                page = I(
                    buffer_ops.buffer_load(
                        btr, (tt * 32) // P, vec_width=1, dtype=I, is_scalar=True
                    )
                )
                kr = res(kv + page.to(I64) * page_stride, page_stride)
                out = []
                for u in range_constexpr(2):
                    g = (tt % (P // 32)) + u
                    kd0 = Vec(
                        buffer_ops.buffer_load(
                            kr,
                            (g * 2048 + h * 512 + col * 16) // 4,
                            vec_width=4,
                            dtype=I,
                        )
                    )
                    kd1 = Vec(
                        buffer_ops.buffer_load(
                            kr,
                            (g * 2048 + (2 + h) * 512 + col * 16) // 4,
                            vec_width=4,
                            dtype=I,
                        )
                    )
                    ksw = (
                        I(
                            fx.Int16(
                                buffer_ops.buffer_load(
                                    kr,
                                    (P * 64 + g * 128 + h * 64 + col * 2) // 2,
                                    vec_width=1,
                                    dtype=fx.Int16,
                                )
                            ).to(I)
                        )
                        & 0xFFFF
                    )
                    out += [kd0, kd1, ksw]
                return out

            def score_pair(t, st_a, st_b):
                """Tiles t and t + 1: their head sums share the lane swap, the block maxima, the store."""
                tok = t * 32 + col + (lane // 32) * 32
                accs = []
                for kd0, kd1, ksw in (st_a, st_b):
                    row_accs = []
                    for r in range_constexpr(RPW):
                        acc = mfma_fp4(
                            qv[r][0], kd0, Vec.filled(16, 0.0, F), sa[r][0], ksw, 0
                        )
                        row_accs.append(mfma_fp4(qv[r][1], kd1, acc, sa[r][1], ksw, 1))
                    accs.append(row_accs)
                for r in range_constexpr(RPW):
                    v = (tok < end[r]).select(
                        pair_logit(accs[0][r], accs[1][r], wv[r]), F(NEG_INF)
                    )
                    if const_expr(tokens):
                        buffer_ops.buffer_store(
                            v,
                            sr,
                            rows[r] * scores_stride + tok,
                            mask=(tok < end[r])
                            & (wave * RPW + r < NN)
                            & (t + lane // 32 < t1),
                        )
                        continue
                    x = bi(v)
                    for m in (1, 2):
                        x = bi(maxf(bf(x), bf(dpp(x, 0xB1 if m == 1 else 0x4E))))
                    x = bi(
                        maxf(
                            bf(x),
                            bf(
                                dpp(
                                    x,
                                    0x104,
                                    old=dpp(x, 0x114, bank_mask=0xA),
                                    bank_mask=0x5,
                                )
                            ),
                        )
                    )
                    buffer_ops.buffer_store(
                        bf(x),
                        sr,
                        rows[r] * scores_stride + t * 4 + lane // 8,
                        mask=((lane % 8) == 0)
                        & (wave * RPW + r < NN)
                        & (t + lane // 32 < t1),
                    )

            niter = ((t1 - t0 + 1) // 2 + 1) // 2
            for it, state in range(0, I64(niter), 1, init=load_pair(t0)):
                t = t0 + I(it) * 4
                nxt = load_pair(t + 2)
                score_pair(t, state[0:3], state[3:6])
                last = load_pair(t + 4)
                score_pair(t + 2, nxt[0:3], nxt[3:6])
                _results = yield last

    @flyc.jit
    def launch(
        q: fx.Int64,
        q_scale: fx.Int64,
        weights: fx.Int64,
        kv: fx.Int64,
        block_table: fx.Int64,
        row_ends: fx.Int64,
        scores: fx.Int64,
        page_stride: fx.Int64,
        bt_stride: fx.Int32,
        scores_stride: fx.Int32,
        nseg: fx.Int32,
        reqs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            q_scale,
            weights,
            kv,
            block_table,
            row_ends,
            scores,
            page_stride,
            bt_stride,
            scores_stride,
            nseg,
        ).launch(
            grid=(fx.Int64(nseg), fx.Int64(reqs), 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
