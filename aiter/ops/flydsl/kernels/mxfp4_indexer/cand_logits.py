# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Logits of the candidate tokens of a decode row: the 2048 candidate 8-token blocks of the two-level indexer's consumer
layers are 16384 tokens a row, scored with the select's arithmetic (``token_logit``: the same bits as the select's own
rescoring) and written as a [rows, 16384 + 64] fp32 row (-inf for a padded block or a token past the row's end).

A few rows (decode steps of up to ~190 rows) leave most of the GPU idle when one workgroup scores and selects a row
(the gather select: ~42 us a row, VALU bound on its one CU); here a row's 512 tiles (32 tokens = 4 blocks) are cut into
segments of ``TPS`` tiles, one 4-wave workgroup a (row, segment), so the scoring takes a few us on many CUs and
``tokens`` (gathered mode) selects from the logits. Tile ``s * TPS + 4 i + wave`` is
the wave's i-th of segment s.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import (
    BLOCK,
    CANDIDATES,
    I64,
    NEG_INF,
    F,
    H,
    I,
    gptr,
    load,
    mfma_fp4,
    sload,
    store,
    token_logit,
    view,
    wave_of,
)

NW = 4  # waves a workgroup
FLIGHT = 4  # tiles whose K loads are in flight together


@functools.cache
def compile_candidate_logits(
    next_n: int,
    page_entries: int,
    tps: int,
    min_end: int = 0,
    cand_blocks: int = CANDIDATES,
):
    """Logits of the candidate tokens, ``tps`` tiles (a multiple of ``4 * FLIGHT``, dividing 512) a workgroup; grid
    (512 / tps, rows)."""
    P, NN = page_entries, next_n
    TILES = cand_blocks * BLOCK // 32  # tiles of 32 tokens a row
    assert P in (64, 128) and TILES % tps == 0 and tps % (NW * FLIGHT) == 0
    LOGP = P.bit_length() - 1
    NI = tps // NW  # tiles a wave
    name = (
        f"indexer_mxfp4_candidate_logits_h32_d128_nn{NN}_p{P}_t{tps}"
        + (f"_min{min_end}" if min_end else "")
        + (f"_c{cand_blocks}" if cand_blocks != CANDIDATES else "")
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        q: fx.Int64,
        q_scale: fx.Int64,
        weights: fx.Int64,
        kv: fx.Int64,
        block_table: fx.Int64,
        row_ends: fx.Int64,
        cand: fx.Int64,
        logits: fx.Int64,
        page_stride: fx.Int64,
        bt_stride: fx.Int32,
        logits_stride: fx.Int32,
    ):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = wave_of(tx)
        col = lane % 32
        h = lane // 32
        row = fx.block_idx.y
        seg = fx.block_idx.x
        req = row // NN

        end = sload(view(row_ends + I64(row) * 4, 4), 0)
        go = (
            (end > I(min_end)) if min_end else (I(0) == I(0))
        )  # min_end 0: every row, empty ones write -inf
        if go:
            qr = view(q + I64(row) * (H * 64), H * 64, unit=4)
            qsw = load(view(q_scale + I64(row) * (H * 4), H * 4), col)
            qv = [
                load(qr, (col * 64 + kb * 32 + h * 16) // 4, unit=4)
                for kb in range_constexpr(2)
            ]
            sa = [(qsw >> ((2 * kb + h) * 8)) & 0xFF for kb in range_constexpr(2)]
            wr = view(weights + I64(row) * (H * 4), H * 4, F, 4)
            wv = []
            for j in range_constexpr(4):
                v = load(wr, 8 * j + 4 * h, F, 4)
                wv += [F(v[0]), F(v[1]), F(v[2]), F(v[3])]
            btr = view(block_table + I64(req) * I64(bt_stride) * 4, I64(bt_stride) * 4)
            cr = view(cand + I64(row) * (CANDIDATES * 4), CANDIDATES * 4)
            sr = view(
                logits + I64(row) * I64(logits_stride) * 4, I64(logits_stride) * 4, F
            )

            for g0 in range_constexpr(0, NI, FLIGHT):
                ids = []
                for ti in range_constexpr(g0, g0 + FLIGHT):
                    tile = seg * tps + ti * NW + wave
                    ids.append(load(cr, tile * 4 + (col >> 3)))
                info = []
                for ti, idv in zip(range_constexpr(g0, g0 + FLIGHT), ids):
                    tile = seg * tps + ti * NW + wave
                    pos = (idv << 3) | (col & 7)
                    ok = (idv >= I(0)) & (pos < end)
                    at = ok.select(pos >> LOGP, I(0))
                    info.append(
                        (
                            tile,
                            pos,
                            ok,
                            load(btr, at),
                        )
                    )
                ks = []
                for tile, pos, ok, page in info:
                    within = pos & 31
                    a0 = (
                        kv
                        + page.to(I64) * page_stride
                        + (((pos & (P - 1)) >> 5) << 11).to(I64)
                        + (h * 512 + (within << 4)).to(I64)
                    )
                    a2 = (
                        kv
                        + page.to(I64) * page_stride
                        + (
                            P * 64
                            + ((pos & (P - 1)) >> 5) * 128
                            + h * 64
                            + (within << 1)
                        ).to(I64)
                    )
                    ks.append(
                        (
                            Vec(fx.generic_load(gptr(a0), dtype=I, count=4)),
                            Vec(fx.generic_load(gptr(a0 + 1024), dtype=I, count=4)),
                            I(
                                fx.Int16(
                                    fx.generic_load(gptr(a2, 2, T.i16), dtype=fx.Int16)
                                ).to(I)
                            )
                            & 0xFFFF,
                        )
                    )
                for (tile, pos, ok, page), (kd0, kd1, ksw) in zip(info, ks):
                    acc = mfma_fp4(qv[0], kd0, Vec.filled(16, 0.0, F), sa[0], ksw, 0)
                    acc = mfma_fp4(qv[1], kd1, acc, sa[1], ksw, 1)
                    store(
                        sr,
                        tile * 32 + col,
                        ok.select(token_logit(acc, wv), F(NEG_INF)),
                        F,
                        mask=h == 0,
                    )

    @flyc.jit
    def launch(
        q: fx.Int64,
        q_scale: fx.Int64,
        weights: fx.Int64,
        kv: fx.Int64,
        block_table: fx.Int64,
        row_ends: fx.Int64,
        cand: fx.Int64,
        logits: fx.Int64,
        page_stride: fx.Int64,
        bt_stride: fx.Int32,
        logits_stride: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            q_scale,
            weights,
            kv,
            block_table,
            row_ends,
            cand,
            logits,
            page_stride,
            bt_stride,
            logits_stride,
        ).launch(
            grid=(fx.Int64(TILES // tps), fx.Int64(rows), 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
