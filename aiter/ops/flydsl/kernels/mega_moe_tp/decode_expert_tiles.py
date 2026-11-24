# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape (8-128 row, E128 top4, H6144 / I768, MXFP4 routed experts + MXFP8 shared expert)
MoE GEMM1 / GEMM2 expert tiles and their fused "shared tiles first" launches.

One launch per GEMM: the first ``nsh`` workgroups run the shared expert's unsplit MXFP8 tiles,
the rest run the routed MXFP4 tiles, so the shared expert streams next to the routed experts
and costs no extra launch:

* ``compile_gemm1_shared_first``: routed gate/up GEMM + SwiGLU + MXFP4 quant (each workgroup builds
  its 16-row expert block in-kernel from ``topk_ids``: no sort launch, see ``decode_id_routing``),
  and the shared gate/up GEMM + SwiGLU + MXFP8 quant.
* ``compile_gemm2_shared_first``: routed down GEMM (weighted BF16 slot rows, one writer per
  (token, slot)) and the shared down GEMM (BF16, one writer per token).

K-waves sum their FP32 partials through LDS in rank order, so every output is the same bits for
any (kw, pf, tile) choice that keeps the K order; the tile / prefetch knobs are tuning only.
The per-row-count tiling for the 8-128 row decode path is ``decode_expert_gemms.tiles_for``.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/{gemm1,gemm2}_a4w4_id_rows.py``,
``shared_gemm1_a8w8.py`` and ``shared_gemm2_a8w8.py`` (Apache-2.0, "Copyright contributors to the
amd-kernels project"); the tile bodies are the same, the MFMA / quant / routing primitives come
from ``decode_tile_primitives`` / ``decode_id_routing`` and the buffer ops from aiter's own
``aiter.ops.flydsl.kernels.buffer_ops``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .decode_id_routing import WAVE_BYTES, wave_block_rows, wave_expert
from .decode_tile_primitives import (
    _e8m0_fp8,
    _fp4,
    _fp8,
    _hold,
    _hold8,
    _lds_ptr,
    _load8,
    _mfma,
    _mfma8,
    _raw,
    _swiglu,
)

H, I, E = 6144, 768, 128


def _swiglu_quant_store(vals, mr, row0, col, l16, q16, tok, m, qr, ss):
    """FP32 gate/up sums of one wave's 32 columns -> SwiGLU -> MXFP8 codes + e8m0."""
    for mi in range_constexpr(mr):
        y = [
            [
                _swiglu(
                    fx.Float32(vals[mi * 4 + ni * 2][v]),
                    fx.Float32(vals[mi * 4 + ni * 2 + 1][v]),
                )
                for v in range_constexpr(4)
            ]
            for ni in range_constexpr(2)
        ]
        amax = fx.math.absf(y[0][0])
        for ni in range_constexpr(2):
            for v in range_constexpr(4):
                z = fx.math.absf(y[ni][v])
                amax = (amax > z).select(amax, z)
        for sh in range_constexpr(2):
            z = amax.shuffle_xor(16 << sh, 64)
            amax = (amax > z).select(amax, z)
        e = _e8m0_fp8(amax)
        scale = (e << 23).bitcast(fx.Float32)
        row = row0 + mi * 16 + l16
        for ni in range_constexpr(2):
            pk = _fp8(fx.Int32(0), y[ni][0], y[ni][1], scale, False)
            pk = _fp8(pk, y[ni][2], y[ni][3], scale, True)
            off = row * I + col + ni * 16 + q16 * 4
            buffer_ops.buffer_store(pk, qr, off // 4, mask=tok[mi] < m)
        buffer_ops.buffer_store(
            e.to(fx.Int8),
            ss,
            row * (I // 32) + col // 32,
            offset_is_bytes=True,
            mask=(tok[mi] < m) & (q16 == 0),
        )


@functools.cache
def compile_shared_gemm1(*, kw=1, pf=3, bm=16):
    """The body of a gate/up tile of ``bm`` rows x 128 / ``kw`` columns: ``kw`` K-waves, ``pf`` K-steps in flight."""
    # a K range starts on an even 128-K step: one weight-scale word covers two steps
    assert kw in (1, 2, 4) and bm in (16, 32, 64) and 48 % kw == 0 and 48 // kw % 2 == 0
    nw = 4 // kw
    tn, kt, mr = nw * 32, 48 // kw, bm // 16

    @flyc.jit
    def tile(base, bid, A, AS, W, WS, Q, S, m, rows):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw
        tb, kg = bid, kk
        mb, nb = tb // (I // tn), tb % (I // tn)
        if mb * bm < m:
            col = nb * tn + nk * 32
            ar = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=m * H
            )
            sr = buffer_ops.create_buffer_resource(
                AS, max_size=False, num_records_bytes=m * (H // 32)
            )
            wr = buffer_ops.create_buffer_resource(
                W, max_size=False, num_records_bytes=2 * I * H
            )
            ws = buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=I * (H // 16)
            )
            qr = buffer_ops.create_buffer_resource(
                Q, max_size=False, num_records_bytes=m * I
            )
            ss = buffer_ops.create_buffer_resource(
                S, max_size=False, num_records_bytes=m * (I // 32)
            )
            tok = [mb * bm + mi * 16 + l16 for mi in range_constexpr(mr)]
            ad = [
                (tok[mi] * H + kg * (kt * 128) + q16 * 16) // 4
                for mi in range_constexpr(mr)
            ]
            sd = [tok[mi] * (H // 32) // 4 + kg * kt for mi in range_constexpr(mr)]
            wd = [
                [
                    ((col // 16 + ni + gu * 48) * 48 * 2048 + q16 * 256 + l16 * 16) // 4
                    + kg * kt * 512
                    for gu in range_constexpr(2)
                ]
                for ni in range_constexpr(2)
            ]
            swd = [
                ((col // 32 + gu * 24) * 24) * 64 + q16 * 16 + l16 + kg * (kt // 2) * 64
                for gu in range_constexpr(2)
            ]

            def load(k):
                aa = [_load8(ar, ad[mi], k * 128) for mi in range_constexpr(mr)]
                sa = [
                    (
                        fx.Int32(
                            buffer_ops.buffer_load(
                                sr,
                                sd[mi],
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=k * 4,
                            )
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]
                bb = [
                    [_load8(wr, wd[ni][gu], k * 2048, 256) for gu in range_constexpr(2)]
                    for ni in range_constexpr(2)
                ]
                sb = [
                    buffer_ops.buffer_load(
                        ws,
                        swd[gu],
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(k // 2) * 256,
                    )
                    for gu in range_constexpr(2)
                ]
                return aa, sa, bb, sb

            acc = [[[None] * 2 for _ in range(2)] for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(kt):
                if const_expr(k + pf < kt):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for ni in range_constexpr(2):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = _mfma8(
                                aa[mi],
                                bb[ni][gu],
                                acc[mi][ni][gu],
                                sa[mi],
                                sb[gu],
                                (k % 2) * 2 + ni,
                                True,
                            )
            flat = _hold8(
                [
                    acc[mi][ni][gu]
                    for mi in range(mr)
                    for ni in range(2)
                    for gu in range(2)
                ]
            )
            if const_expr(kw > 1):
                for j in range_constexpr(mr * 4):
                    llvm.StoreOp(
                        _raw(flat[j]),
                        _lds_ptr(base, ((wave * mr * 4 + j) * 64 + lane) * 16),
                        alignment=16,
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            if kk == 0:
                vals = [Vec(v) for v in flat]
                for k in range_constexpr(1, kw):
                    for j in range_constexpr(mr * 4):
                        other = Vec(
                            llvm.load(
                                T.vec(4, T.f32),
                                _lds_ptr(
                                    base,
                                    (((nk + k * nw) * mr * 4 + j) * 64 + lane) * 16,
                                ),
                                alignment=16,
                            )
                        )
                        vals[j] = vals[j] + other
                _swiglu_quant_store(vals, mr, mb * bm, col, l16, q16, tok, m, qr, ss)

    return tile


@functools.cache
def compile_shared_gemm2(*, tn=256, bm=16, pf=2):
    """The body ``tile(base, bid, A, AS, W, WS, OUT, m)`` of a down-GEMM tile of ``bm`` rows x ``tn`` columns with
    ``pf`` K-steps in flight; the output goes through its ``lds_bytes`` of LDS from ``base``.
    """
    # tn 64: one 16-column subtile per wave, so the wave's half of a 32-column scale word is runtime (wave parity);
    # its bytes are shifted into the even slot
    assert tn in (64, 128, 256, 512) and bm in (16, 32, 64)
    ni, mr = tn // 64, bm // 16

    @flyc.jit
    def tile(base, bid, A, AS, W, WS, OUT, m):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        mb, nb = bid // (H // tn), bid % (H // tn)
        if mb * bm < m:
            ar = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=m * I
            )
            sr = buffer_ops.create_buffer_resource(
                AS, max_size=False, num_records_bytes=m * (I // 32)
            )
            wr = buffer_ops.create_buffer_resource(
                W, max_size=False, num_records_bytes=H * I
            )
            sw = buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=H * (I // 32)
            )
            out = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=m * H * 2
            )
            ad = [
                ((mb * bm + mi * 16 + l16) * I + q16 * 16) // 4
                for mi in range_constexpr(mr)
            ]
            sd = [
                (mb * bm + mi * 16 + l16) * (I // 32) // 4 for mi in range_constexpr(mr)
            ]
            wd = [
                ((nb * (tn // 16) + wave * ni + j) * 6 * 2048 + q16 * 256 + l16 * 16)
                // 4
                for j in range_constexpr(ni)
            ]
            swd = [
                ((nb * (tn // 32) + (wave * ni + j) // 2) * 3) * 64 + q16 * 16 + l16
                for j in range_constexpr(ni)
            ]

            def load(k):
                aa = [_load8(ar, ad[mi], k * 128) for mi in range_constexpr(mr)]
                sa = [
                    (
                        fx.Int32(
                            buffer_ops.buffer_load(
                                sr,
                                sd[mi],
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=k * 4,
                            )
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]
                bb = [_load8(wr, wd[j], k * 2048, 256) for j in range_constexpr(ni)]
                sb = [
                    buffer_ops.buffer_load(
                        sw,
                        swd[j],
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(k // 2) * 256,
                    )
                    for j in range_constexpr(ni)
                ]
                if fx.const_expr(ni == 1):
                    sb = [fx.Int32(sb[0]) >> ((wave % 2) * 8)]
                return aa, sa, bb, sb

            acc = [[None] * ni for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(6):
                if fx.const_expr(k + pf < 6):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for j in range_constexpr(ni):
                        acc[mi][j] = _mfma8(
                            aa[mi],
                            bb[j],
                            acc[mi][j],
                            sa[mi],
                            sb[j],
                            (k % 2) * 2 + j % 2,
                        )
            flat = _hold8([acc[mi][j] for mi in range(mr) for j in range(ni)])
            for mi in range_constexpr(mr):
                for j in range_constexpr(ni):
                    v = Vec(flat[mi * ni + j])
                    col = wave * (tn // 4) + j * 16 + l16
                    for ii in range_constexpr(4):
                        off = ((mi * 16 + q16 * 4 + ii) * tn + col) * 4
                        llvm.StoreOp(_raw(v[ii]), _lds_ptr(base, off))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            for rr in range_constexpr(bm // 8):
                row = rr * 8 + tx // 32
                token = mb * bm + row
                for c in range_constexpr(tn // 64):
                    col = tx % 32 * 2 + c * 64
                    v = Vec(
                        llvm.load(
                            T.vec(2, T.f32),
                            _lds_ptr(base, (row * tn + col) * 4),
                            alignment=8,
                        )
                    )
                    pk = v.to(fx.BFloat16).bitcast(fx.Int32)[0]
                    buffer_ops.buffer_store(
                        pk, out, (token * H + nb * tn + col) // 2, mask=token < m
                    )

    tile.lds_bytes = bm * tn * 4
    return tile


@functools.cache
def compile_routed_gemm1_id_rows(
    *, kw=1, pf=3, bm=16, pairs=256, prio=None, w13_split=False
):
    """``w13_split``: the routed w13 values are read in the checkpoint-order preshuffle layout
    ``[E, 2 (gate, up), I/16, 16-row block]`` instead of the gate / up interleaved ``[E, I/16, 2, block]``:
    only the block address changes, the bytes read and every output bit are the same.
    ``prio=(high, low)``: ``s_setprio high`` for the prologue (routing-id load, histogram, first
    weight loads), ``s_setprio low`` once the first weight loads are issued. No data or instruction
    order change: the outputs are the same bits."""
    assert kw in (1, 2, 4) and bm in (16, 32) and pairs in (64, 128, 256, 512)
    nw = 4 // kw
    tn, kt, mr = nw * 32, 48 // kw, bm // 16
    assert bm == 16
    ex = 4 * mr * 4 * 64 * 16
    route = ex if kw > 1 else 16  # the waves' routing slices follow the K-wave exchange

    @flyc.jit
    def routed_tile(base, rb, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw
        mb, nb = rb // (I // tn), rb % (I // tn)
        hb = base + route + wave * WAVE_BYTES
        if const_expr(prio is not None):
            fx.rocdl.s_setprio(prio[0])
        expert, first, blocks, pair_ids = wave_expert(IDS, m, mb, lane, hb, pairs)
        if (tx == 0) & (nb == 0):
            if mb == 0:
                EIDS[rows // 16] = blocks
            if mb < blocks:
                EIDS[mb] = expert
        if mb < blocks:
            col = nb * tn + nk * 32
            ar = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=m * (H // 2)
            )
            sr = buffer_ops.create_buffer_resource(
                AS, max_size=False, num_records_bytes=m * (H // 32)
            )
            wr = buffer_ops.create_buffer_resource(
                W,
                max_size=False,
                num_records_bytes=I * H,
                base_byte_offset=fx.Int64(expert) * (I * H),
            )
            ws = buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=E * I * (H // 16)
            )
            qr = buffer_ops.create_buffer_resource(
                Q, max_size=False, num_records_bytes=rows * (I // 2)
            )
            ss = buffer_ops.create_buffer_resource(
                S, max_size=False, num_records_bytes=rows * (I // 32)
            )
            wd = [
                [
                    (
                        (
                            gu * (I // 16) + col // 16 + ni
                            if w13_split
                            else (col // 16 + ni) * 2 + gu
                        )
                        * 48
                        * 1024
                        + q16 * 256
                        + l16 * 16
                    )
                    // 4
                    + kk * kt * 256
                    for gu in range_constexpr(2)
                ]
                for ni in range_constexpr(2)
            ]
            swd = [
                ((expert * 48 + col // 16 + ni) * 24) * 64
                + q16 * 16
                + l16
                + kk * (kt // 2) * 64
                for ni in range_constexpr(2)
            ]

            bpre = [
                [
                    buffer_ops.buffer_load(
                        wr, wd[ni][gu], vec_width=4, dtype=fx.Int32, cache_modifier=2
                    )
                    for gu in range_constexpr(2)
                ]
                for ni in range_constexpr(2)
            ]
            spre = [
                buffer_ops.buffer_load(ws, swd[ni], vec_width=1, dtype=fx.Int32)
                for ni in range_constexpr(2)
            ]
            if const_expr(prio is not None):
                fx.rocdl.s_setprio(prio[1])
            # the rows after the first weight loads are in flight
            packed = wave_block_rows(pair_ids, expert, first, lane, hb)
            if (tx < 16) & (nb == 0):
                NV[mb * 16 + tx] = packed
            tok = [packed & 0xFFFFFF for mi in range_constexpr(mr)]
            ad = [
                (tok[mi] * (H // 2) + kk * (kt * 64) + q16 * 16) // 4
                for mi in range_constexpr(mr)
            ]
            sd = [tok[mi] * (H // 32) // 4 + kk * kt for mi in range_constexpr(mr)]

            def load(k):
                aa = [
                    buffer_ops.buffer_load(
                        ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 64
                    )
                    for mi in range_constexpr(mr)
                ]
                sa = [
                    (
                        fx.Int32(
                            buffer_ops.buffer_load(
                                sr,
                                sd[mi],
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=k * 4,
                            )
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]
                bb = (
                    bpre
                    if k == 0
                    else [
                        [
                            buffer_ops.buffer_load(
                                wr,
                                wd[ni][gu],
                                vec_width=4,
                                dtype=fx.Int32,
                                soffset_bytes=k * 1024,
                                cache_modifier=2,
                            )
                            for gu in range_constexpr(2)
                        ]
                        for ni in range_constexpr(2)
                    ]
                )
                sb = (
                    spre
                    if k == 0
                    else [
                        buffer_ops.buffer_load(
                            ws,
                            swd[ni],
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(k // 2) * 256,
                        )
                        for ni in range_constexpr(2)
                    ]
                )
                return aa, sa, bb, sb

            acc = [[[None] * 2 for _ in range(2)] for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(kt):
                if const_expr(k + pf < kt):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for ni in range_constexpr(2):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = _mfma(
                                aa[mi],
                                bb[ni][gu],
                                acc[mi][ni][gu],
                                sa[mi],
                                sb[ni],
                                (k % 2) * 2 + gu,
                                True,
                            )
            flat = _hold(
                [
                    acc[mi][ni][gu]
                    for mi in range(mr)
                    for ni in range(2)
                    for gu in range(2)
                ]
            )
            if const_expr(kw > 1):
                for j in range_constexpr(mr * 4):
                    llvm.StoreOp(
                        _raw(flat[j]),
                        _lds_ptr(base, ((wave * mr * 4 + j) * 64 + lane) * 16),
                        alignment=16,
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            if kk == 0:
                vals = [Vec(v) for v in flat]
                for k in range_constexpr(1, kw):
                    for j in range_constexpr(mr * 4):
                        other = Vec(
                            llvm.load(
                                T.vec(4, T.f32),
                                _lds_ptr(
                                    base,
                                    (((nk + k * nw) * mr * 4 + j) * 64 + lane) * 16,
                                ),
                                alignment=16,
                            )
                        )
                        vals[j] = vals[j] + other
                for mi in range_constexpr(mr):
                    y = [
                        [
                            _swiglu(
                                fx.Float32(vals[mi * 4 + ni * 2][v]),
                                fx.Float32(vals[mi * 4 + ni * 2 + 1][v]),
                            )
                            for v in range_constexpr(4)
                        ]
                        for ni in range_constexpr(2)
                    ]
                    amax = fx.math.absf(y[0][0])
                    for ni in range_constexpr(2):
                        for v in range_constexpr(4):
                            z = fx.math.absf(y[ni][v])
                            amax = (amax > z).select(amax, z)
                    for sh in range_constexpr(2):
                        z = amax.shuffle_xor(16 << sh, 64)
                        amax = (amax > z).select(amax, z)
                    e = (amax.bitcast(fx.Int32) - fx.Int32(0xE00000)) >> 23
                    e = (e > 1).select(e, fx.Int32(1))
                    scale = (e << 23).bitcast(fx.Float32)
                    row = mb * bm + mi * 16 + l16
                    for ni in range_constexpr(2):
                        pk = _fp4(fx.Int32(0), y[ni][0], y[ni][1], scale, 0)
                        pk = _fp4(pk, y[ni][2], y[ni][3], scale, 1)
                        off = row * (I // 2) + col // 2 + ni * 8 + q16 * 2
                        buffer_ops.buffer_store(
                            pk.to(fx.Int16),
                            qr,
                            off,
                            offset_is_bytes=True,
                            mask=tok[mi] < m,
                        )
                    buffer_ops.buffer_store(
                        e.to(fx.Int8),
                        ss,
                        row * (I // 32) + col // 32,
                        offset_is_bytes=True,
                        mask=(tok[mi] < m) & (q16 == 0),
                    )

    return routed_tile


@functools.cache
def compile_routed_gemm2_id_rows(*, tn=256, bm=16, pf=2):
    assert tn in (128, 256, 512) and bm == 16
    ni, mr = tn // 64, bm // 16

    @flyc.jit
    def routed_tile(base, rb, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        mb, nb = rb // (H // tn), rb % (H // tn)
        record = fx.Int32(EIDS[mb])
        blocks = fx.Int32(EIDS[rows // 16])
        fx.rocdl.sched_barrier(0)
        if mb < blocks:
            expert = record & 127
            ar = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=rows * (I // 2)
            )
            sr = buffer_ops.create_buffer_resource(
                AS, max_size=False, num_records_bytes=rows * (I // 32)
            )
            wr = buffer_ops.create_buffer_resource(
                W,
                max_size=False,
                num_records_bytes=H * I // 2,
                base_byte_offset=fx.Int64(expert) * (H * I // 2),
            )
            sw = buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=E * H * (I // 32)
            )
            out = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=m * 4 * H * 2
            )
            ad = [
                ((mb * bm + mi * 16 + l16) * (I // 2) + q16 * 16) // 4
                for mi in range_constexpr(mr)
            ]
            sd = [
                (mb * bm + mi * 16 + l16) * (I // 32) // 4 for mi in range_constexpr(mr)
            ]
            wd = [
                ((nb * (tn // 16) + wave * ni + j) * 6 * 1024 + q16 * 256 + l16 * 16)
                // 4
                for j in range_constexpr(ni)
            ]
            swd = [
                ((expert * (H // 32) + nb * (tn // 32) + (wave * ni + j) // 2) * 3) * 64
                + q16 * 16
                + l16
                for j in range_constexpr(ni)
            ]

            def load(k):
                aa = [
                    buffer_ops.buffer_load(
                        ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 64
                    )
                    for mi in range_constexpr(mr)
                ]
                sa = [
                    (
                        fx.Int32(
                            buffer_ops.buffer_load(
                                sr,
                                sd[mi],
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=k * 4,
                            )
                        )
                        >> (q16 * 8)
                    )
                    & 255
                    for mi in range_constexpr(mr)
                ]
                bb = [
                    buffer_ops.buffer_load(
                        wr,
                        wd[j],
                        vec_width=4,
                        dtype=fx.Int32,
                        soffset_bytes=k * 1024,
                        cache_modifier=2,
                    )
                    for j in range_constexpr(ni)
                ]
                sb = [
                    buffer_ops.buffer_load(
                        sw,
                        swd[j],
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(k // 2) * 256,
                    )
                    for j in range_constexpr(ni)
                ]
                return aa, sa, bb, sb

            acc = [[None] * ni for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(6):
                if fx.const_expr(k + pf < 6):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for j in range_constexpr(ni):
                        acc[mi][j] = _mfma(
                            aa[mi],
                            bb[j],
                            acc[mi][j],
                            sa[mi],
                            sb[j],
                            (k % 2) * 2 + j % 2,
                        )
            flat = _hold([acc[mi][j] for mi in range(mr) for j in range(ni)])
            for mi in range_constexpr(mr):
                for j in range_constexpr(ni):
                    v = Vec(flat[mi * ni + j])
                    col = wave * (tn // 4) + j * 16 + l16
                    for ii in range_constexpr(4):
                        off = ((mi * 16 + q16 * 4 + ii) * tn + col) * 4
                        llvm.StoreOp(_raw(v[ii]), _lds_ptr(base, off))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            for rr in range_constexpr(bm // 8):
                row = rr * 8 + tx // 32
                packed = fx.Int32(IDS[mb * 16 + row])
                token, slot = packed & 0xFFFFFF, packed.shrui(fx.Int32(24))
                pair = (token < m).select(token * 4 + slot, fx.Int32(0))
                weight = fx.Float32(WT[pair])
                for c in range_constexpr(tn // 64):
                    col = tx % 32 * 2 + c * 64
                    v = Vec(
                        llvm.load(
                            T.vec(2, T.f32),
                            _lds_ptr(base, (row * tn + col) * 4),
                            alignment=8,
                        )
                    )
                    pk = (v * weight).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    buffer_ops.buffer_store(
                        pk,
                        out,
                        ((token * 4 + slot) * H + nb * tn + col) // 2,
                        mask=token < m,
                    )

    return routed_tile


@functools.cache
def compile_gemm1_shared_first(
    *,
    kw=4,
    pf=3,
    bm=16,
    sh_bm=16,
    sh_kw=4,
    sh_pf=2,
    pairs=256,
    prio=None,
    w13_split=False,
):
    """Routed GEMM1 launch whose first ``nsh`` workgroups run the shared expert's MXFP8 gate/up tiles.

    Launch signature: ``(A, AS, W, WS, Q, S, IDS, EIDS, NV, A8, AS8, W8, WS8, Q8, S8, m, rows, nsh, stream)``
    with ``A/AS`` the MXFP4 activations + e8m0 scales, ``W/WS`` the routed w13 + scales, ``Q/S`` the
    MXFP4 SwiGLU output rows + scales, ``(IDS, EIDS, NV)`` = (``topk_ids``, block->expert map and
    row table, both written here: no sort launch), ``prio`` the prologue issue priority of the routed waves,
    and ``*8`` the shared expert's MXFP8 operands. Grid = ``rows // bm * (I // 32) + nsh``.
    """
    assert kw == 4 and bm == 16 and sh_bm in (16, 32, 64) and sh_kw in (1, 2, 4)
    assert pairs in (64, 128, 256, 512)
    tn = 32
    # shared workgroups use their exchange region; routed ones their exchange + routing slices
    sh_ex = 4 * (max(bm, sh_bm if sh_kw > 1 else 16) // 16) * 4 * 64 * 16
    lds = max(sh_ex, 4 * (bm // 16) * 4 * 64 * 16 + 4 * WAVE_BYTES)
    sh_name = (
        ("" if sh_bm == 16 else f"_bm{sh_bm}")
        + ("" if sh_kw == 4 else f"_kw{sh_kw}")
        + ("" if sh_pf == 2 else f"_pf{sh_pf}")
    )
    tier = f"_id_rows_p{pairs}"
    if prio is not None:
        tier += f"_prio{prio[0]}{prio[1]}"
    if w13_split:
        tier += "_w13split"
    name = f"aiter_moe_gemm1_a4w4_swiglu_quant_h6144_i768_bm{bm}_tn{tn}_kw{kw}_pf{pf}{tier}_shared_a8w8{sh_name}_first"
    sh_tile = compile_shared_gemm1(kw=sh_kw, pf=sh_pf, bm=sh_bm)
    routed_tile = compile_routed_gemm1_id_rows(
        kw=kw, pf=pf, bm=bm, pairs=pairs, prio=prio, w13_split=w13_split
    )

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        IDS: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        A8: fx.Tensor,
        AS8: fx.Tensor,
        W8: fx.Tensor,
        WS8: fx.Tensor,
        Q8: fx.Tensor,
        S8: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nsh: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        if bid < nsh:
            sh_tile(base, bid, A8, AS8, W8, WS8, Q8, S8, m, m)
        if bid >= nsh:
            routed_tile(base, bid - nsh, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows)

    @flyc.jit
    def launch_gemm1_shared_first(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        IDS: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        A8: fx.Tensor,
        AS8: fx.Tensor,
        W8: fx.Tensor,
        WS8: fx.Tensor,
        Q8: fx.Tensor,
        S8: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nsh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            A, AS, W, WS, Q, S, IDS, EIDS, NV, A8, AS8, W8, WS8, Q8, S8, m, rows, nsh
        ).launch(
            grid=(rows // bm * (I // tn) + nsh, 1, 1), block=(256, 1, 1), stream=stream
        )

    launch_gemm1_shared_first.kernel_name = name
    return launch_gemm1_shared_first


@functools.cache
def compile_gemm2_shared_first(*, tn=256, bm=16, pf=2, sh_tn=256, sh_bm=16):
    """Routed GEMM2 launch whose first ``nsh`` workgroups run the shared expert's MXFP8 down tiles.

    Launch signature: ``(A, AS, W, WS, OUT, IDS, WT, EIDS, NV, A8, AS8, W8, WS8, OUT8, m, rows, nsh, stream)``;
    ``OUT`` is (m, 4, H) BF16 slot rows (weighted), ``OUT8`` the shared expert's (m, H) BF16.
    ``IDS`` is GEMM1's published row table (``NV``), ``EIDS`` its block->expert map and ``WT`` the
    flat (token, slot) top-k weights.
    """
    assert bm == 16 and sh_bm in (16, 32, 64)
    sh_name = "" if sh_bm == 16 else f"_bm{sh_bm}"
    name = f"aiter_moe_gemm2_a4w4_partials_h6144_i768_bm{bm}_tn{tn}_pf{pf}_id_rows_shared_a8w8{sh_name}_tn{sh_tn}_first"
    sh_tile = compile_shared_gemm2(tn=sh_tn, bm=sh_bm, pf=2)
    routed_tile = compile_routed_gemm2_id_rows(tn=tn, bm=bm, pf=pf)
    lds = max(bm * tn * 4, sh_tile.lds_bytes)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        OUT: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        A8: fx.Tensor,
        AS8: fx.Tensor,
        W8: fx.Tensor,
        WS8: fx.Tensor,
        OUT8: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nsh: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        if bid < nsh:
            sh_tile(base, bid, A8, AS8, W8, WS8, OUT8, m)
        if bid >= nsh:
            routed_tile(base, bid - nsh, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows)

    @flyc.jit
    def launch_gemm2_shared_first(
        A: fx.Tensor,
        AS: fx.Tensor,
        W: fx.Tensor,
        WS: fx.Tensor,
        OUT: fx.Tensor,
        IDS: fx.Tensor,
        WT: fx.Tensor,
        EIDS: fx.Tensor,
        NV: fx.Tensor,
        A8: fx.Tensor,
        AS8: fx.Tensor,
        W8: fx.Tensor,
        WS8: fx.Tensor,
        OUT8: fx.Tensor,
        m: fx.Int32,
        rows: fx.Int32,
        nsh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            A, AS, W, WS, OUT, IDS, WT, EIDS, NV, A8, AS8, W8, WS8, OUT8, m, rows, nsh
        ).launch(
            grid=(rows // bm * (H // tn) + nsh, 1, 1), block=(256, 1, 1), stream=stream
        )

    launch_gemm2_shared_first.kernel_name = name
    return launch_gemm2_shared_first
