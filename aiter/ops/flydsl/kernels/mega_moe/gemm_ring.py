# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# GEMM core adapted from amd-kernels PR #12 (moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8/kernel.py, b682f71).
"""Ring-core GEMM1 / GEMM2 tiles of the fused small-input MegaMoE kernel.

A tile of ``bm`` rows runs its K loop fully unrolled with the FP4 weights as the
scaled MFMA's operand A, so a lane's accumulator holds four consecutive weight
columns of one tile row. The activation rows reach LDS by async DMA through an
``nbuf``-stage ring of 128-K steps, one 1 KiB slot per 16 rows and 64 K-bytes in
MFMA fragment order; a stage is reissued once the barrier of the step that read it
has passed. The weights of step ``k + pf`` are loaded during step ``k``, one load
every few MFMAs, in the order they are emitted. ``nbuf - 2 >= pf`` keeps the
in-order ``vmcnt`` wait of a weight load from draining the ring.

GEMM1 quantizes clamp-SwiGLU per 32 columns in registers and writes its output in
the fragment order GEMM2's ring copies: per 16-row block and 128-K step two
contiguous 1 KiB slots, and one scale dword per row and step.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import lds_vec_load
from ..tensor_shim import buf_base_i64, buf_copy_atom, buf_copy_store, ptr_buf_tensor
from .gemm_util import _e8m0_scale, _group_absmax, silu_mul_f32

_STEP_K = 128
_SLOT = 1024
# Cache policy (sc0|nt) of GEMM1's weight-scale loads and output stores.
_G1_CM = 3


def _load_w(buf, unit, soff_dw, cm, width=4):
    frag = fx.make_rmem_tensor(width, fx.Int32)
    if width == 1:
        src = fx.slice(fx.logical_divide(buf, fx.make_layout(1, 1)), (None, unit))
    else:
        src = fx.slice(buf, (unit, None))
    fx.copy(
        buf_copy_atom(4 * width, fx.Int32, cm), src, frag, soffset=fx.Int32(soff_dw)
    )
    v = Vec(frag.load())
    return v[0] if width == 1 else v


def _dma_view(addr, nbytes):
    return fx.logical_divide(
        ptr_buf_tensor(addr, fx.Int8, num_records_bytes=nbytes), fx.make_layout(1, 1)
    )


def _dma(atom, src, voff, lds_i8, lds_off, soff):
    dst = fx.make_view(fx.add_offset(lds_i8, lds_off), fx.make_layout(1, 1))
    fx.copy(atom, fx.slice(src, (None, voff)), dst, soffset=fx.Int32(soff))


def _lds_i32(lds_base, off, n):
    if n == 1:
        return fx.Int32(lds_vec_load(lds_base, off, T.i32, T.i32))
    return Vec(
        lds_vec_load(lds_base, off, Vec.make_type(n, fx.Int32), fx.Int32, align=16)
    )


def _mfma(atom, acc, w, a, sw, sa):
    wf = fx.make_rmem_tensor(4, fx.Int32)
    af = fx.make_rmem_tensor(len(a), fx.Int32)
    cf = fx.make_rmem_tensor(4, fx.Float32)
    wf.store(w)
    af.store(a)
    cf.store(acc)
    fx.gemm(atom, cf, wf, af, cf, scale_a=sw, scale_b=sa)
    return cf.load()


def _lane_ids(unit_zero):
    tx = fx.thread_idx.x + unit_zero
    lane = tx % 64
    wave = fx.Int32(rocdl.readfirstlane(T.i32, tx // 64))
    return lane, lane % 16, lane // 16, wave


class _Ring:
    """One ring K loop: ``kt`` 128-K steps over ``mr`` 16-row tiles, ``a_bytes``
    activation bytes a row and step, ``n_mma`` MFMAs a row tile and step."""

    def __init__(self, *, kt, mr, a_bytes, n_mma, nw, pf, nbuf):
        assert nbuf >= pf + 2 and 0 < pf < kt and nbuf <= kt
        self.kt, self.mr, self.a_bytes, self.n_mma = kt, mr, a_bytes, n_mma
        self.pf, self.nbuf = pf, nbuf
        self.slots = mr * a_bytes // 64
        self.pitch = self.slots * _SLOT
        assert self.slots % nw == 0 or nw % self.slots == 0

    def slot_ids(self, wave, nw):
        """The ring slots a wave issues each stage (waves share a slot when there are fewer slots than waves)."""
        if self.slots < nw:
            return [wave & fx.Int32(self.slots - 1)]
        return [wave + fx.Int32(nw * j) for j in range(self.slots // nw)]

    @property
    def atom(self):
        return fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)

    def fragment(self, lds_base, lane, k, mi):
        """Row tile ``mi``'s activation operand at step ``k``: two slots (FP8) or one."""
        st = (k % self.nbuf) * self.pitch
        if self.a_bytes == 64:
            return _lds_i32(lds_base, fx.Int32(st + mi * _SLOT) + lane * 16, 4)
        lo = _lds_i32(lds_base, fx.Int32(st + 2 * mi * _SLOT) + lane * 16, 4)
        hi = _lds_i32(lds_base, fx.Int32(st + (2 * mi + 1) * _SLOT) + lane * 16, 4)
        return Vec.from_elements(
            [lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32
        )

    def run(self, *, issue, load_w, read_a, read_sa, mma, w_ring):
        """The unrolled K loop. ``load_w(k)`` returns step ``k``'s weight loads as
        thunks and a ``finish`` giving its weights; ``w_ring`` holds the first ``pf``
        steps' weights, loaded by the caller."""
        kt, mr, pf, nbuf = self.kt, self.mr, self.pf, self.nbuf
        for t in range(nbuf - 1):
            issue(t)
        w_ring = list(w_ring)
        acc = [
            [fx.full(4, 0.0, fx.Float32) for _ in range(self.n_mma)] for _ in range(mr)
        ]
        rocdl.wait_asyncmark(nbuf - 2)
        fx.gpu.barrier()
        nxt = read_a(0, 0)
        issue(nbuf - 1)
        for k in range(kt):
            w = w_ring.pop(0)
            sa = read_sa(k)
            pieces, finish = load_w(k + pf) if k + pf < kt else ([], None)
            gap = max(1, mr * self.n_mma // max(len(pieces), 1))
            n = 0
            for mi in range(mr):
                cur = nxt
                if mi + 1 < mr:
                    nxt = read_a(k, mi + 1)
                elif k + 1 < kt:
                    rocdl.wait_asyncmark(min(nbuf - 2, kt - 2 - k))
                    rocdl.s_waitcnt(lgkmcnt=0)
                    fx.gpu.barrier()
                    nxt = read_a(k + 1, 0)
                    if k + nbuf < kt:
                        issue(k + nbuf)
                for c in range(self.n_mma):
                    if pieces and n % gap == 0:
                        pieces.pop(0)()
                    n += 1
                    acc[mi][c] = mma(k, mi, c, acc[mi][c], w, cur, sa)
                    rocdl.sched_barrier(0)
            for p in pieces:
                p()
            if finish is not None:
                w_ring.append(finish())
        return acc


def load_all(step):
    pieces, finish = step
    for p in pieces:
        p()
    return finish()


def _w_scale_pairs(kt):
    return [None] * ((kt + 1) // 2)


def _rounds(nbytes, nw):
    return ceildiv(ceildiv(nbytes, _SLOT), nw)


def gemm1_ring_lds_bytes(*, a_fp4, model_dim, bm, nw, nbuf):
    """LDS bytes of make_ring_gemm1's ring and scale tile."""
    return (
        nbuf * bm * _STEP_K // (2 if a_fp4 else 1)
        + _rounds(bm * model_dim // 32, nw) * nw * _SLOT
    )


def gemm2_ring_lds_bytes(*, inter_dim, bm, nw, nbuf):
    """LDS bytes of make_ring_gemm2's ring and scale tile."""
    return nbuf * bm * _STEP_K + _rounds(bm * inter_dim // 32, nw) * nw * _SLOT


def make_ring_gemm1(
    *,
    a_fp4,
    model_dim,
    inter_dim,
    tile_n,
    nw,
    bm,
    pf,
    nbuf,
    w_cm,
    swiglu_limit,
    tile_rows=None
):
    """GEMM1 + SwiGLU + MXFP8 quant of one unit (``bm`` rows x ``tile_n // 2``
    intermediate columns), traced in the kernel by the returned ``tile``; its output
    m-tiles are ``tile_rows`` (default ``bm``) rows apart."""
    tile_rows = tile_rows or bm
    n_wave = tile_n // nw
    ni = n_wave // 32
    assert n_wave % 64 == 0 and model_dim % 512 == 0
    ring = _Ring(
        kt=model_dim // _STEP_K,
        mr=bm // 16,
        a_bytes=_STEP_K // (2 if a_fp4 else 1),
        n_mma=2 * ni,
        nw=nw,
        pf=pf,
        nbuf=nbuf,
    )
    row_bytes = model_dim // (2 if a_fp4 else 1)
    sc_row = model_dim // 32
    sc_chunks = sc_row // 16
    sc_rounds = _rounds(bm * sc_row, nw)
    sc_base = nbuf * ring.pitch
    kt2 = inter_dim // _STEP_K
    act = fx.Float4E2M1FN if a_fp4 else fx.Float8E4M3FN
    atoms = [
        fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, fx.Float4E2M1FN, act, opsel_a=o, opsel_b=0
            )
        )
        for o in range(4)
    ]

    def tile(
        *,
        lds_base,
        lds_i8,
        x,
        sx,
        w,
        sw,
        q,
        qs,
        srcmap,
        x_token,
        scale_token,
        x_bytes,
        expert,
        m_tile,
        n_tile,
        tile_row_base,
        rows,
        row_stride
    ):
        """``x`` / ``sx``: the token rows and their scales (``x_token`` bytes /
        ``scale_token`` 16-byte groups into them, out of bounds from ``x_bytes``);
        ``srcmap``: the slot words; ``q`` / ``qs``: the GEMM1 output and scales. Tile
        row r is slot ``tile_row_base + r * row_stride``, real below ``rows``."""
        unit_zero = (m_tile < fx.Int32(0)).select(m_tile, fx.Int32(0))
        lane, l16, q16, wave = _lane_ids(unit_zero)
        x_view = _dma_view(x, x_bytes)
        sx_view = _dma_view(sx, x_bytes)
        wbuf = ptr_buf_tensor(w, fx.Int32, unit_elems=4)
        swbuf = ptr_buf_tensor(sw, fx.Int32)

        a_rows = []
        for d in ring.slot_ids(wave, nw):
            if a_fp4:
                a_rows.append((d, d * 16 + l16, q16 * 16))
            else:
                a_rows.append(
                    (
                        d,
                        (d >> fx.Int32(1)) * 16 + l16,
                        (d & fx.Int32(1)) * 64 + q16 * 16,
                    )
                )
        sc_rows = []
        for qr in range(sc_rounds):
            chunk = (wave + fx.Int32(nw * qr)) * 64 + lane
            r = chunk // fx.Int32(sc_chunks)
            sc_rows.append(
                (fx.Int32(nw * qr) + wave, r, chunk - r * fx.Int32(sc_chunks))
            )
        words = [
            srcmap[tile_row_base + r * row_stride] & fx.Int32(0xFFFFFF)
            for _d, r, _o in a_rows
        ]
        words += [
            srcmap[
                tile_row_base
                + (r < fx.Int32(bm)).select(r, fx.Int32(bm - 1)) * row_stride
            ]
            & fx.Int32(0xFFFFFF)
            for _c, r, _o in sc_rows
        ]

        blk0 = (
            expert * fx.Int32(2 * inter_dim // 16)
            + n_tile * fx.Int32(tile_n // 16)
            + wave * fx.Int32(n_wave // 16)
        )
        w_unit = [
            [
                (blk0 + fx.Int32(2 * j + gu)) * fx.Int32(model_dim // 2)
                + q16 * 16
                + l16
                for gu in range(2)
            ]
            for j in range(ni)
        ]
        sw_unit = [
            ((blk0 >> fx.Int32(1)) + fx.Int32(j)) * fx.Int32(model_dim // 4) + lane
            for j in range(ni)
        ]
        sw_pairs = _w_scale_pairs(ring.kt)

        def load_w(k):
            box = [[None] * 2 for _ in range(ni)]
            pieces = []
            for j in range(ni):
                for gu in range(2):

                    def piece(j=j, gu=gu):
                        box[j][gu] = _load_w(wbuf, w_unit[j][gu], k * 256, w_cm)

                    pieces.append(piece)
            if k % 2 == 0:

                def scales():
                    sw_pairs[k // 2] = [
                        _load_w(swbuf, sw_unit[j], (k // 2) * 64, _G1_CM, 1)
                        for j in range(ni)
                    ]

                pieces.append(scales)
            return pieces, lambda: box

        w_ring = [load_all(load_w(k)) for k in range(pf)]
        a_src = [
            (
                d,
                (r < rows).select(
                    x_token + word * fx.Int32(row_bytes) + k_off, x_bytes
                ),
            )
            for (d, r, k_off), word in zip(a_rows, words)
        ]
        sc_src = [
            (
                c,
                (r < rows).select(
                    (scale_token + word * fx.Int32(sc_chunks) + col) * 16, x_bytes
                ),
            )
            for (c, r, col), word in zip(sc_rows, words[len(a_rows) :])
        ]

        def issue(t):
            if t >= ring.kt:
                return
            for d, src in a_src:
                _dma(
                    ring.atom,
                    x_view,
                    src,
                    lds_i8,
                    fx.Int32((t % nbuf) * ring.pitch) + d * _SLOT,
                    t * ring.a_bytes,
                )
            if t == 0:
                for c, src in sc_src:
                    _dma(
                        ring.atom,
                        sx_view,
                        src,
                        lds_i8,
                        fx.Int32(sc_base) + c * _SLOT,
                        0,
                    )
            rocdl.asyncmark()

        def read_sa(k):
            return [
                (
                    _lds_i32(
                        lds_base,
                        fx.Int32(sc_base + mi * 16 * sc_row + k * 4) + l16 * sc_row,
                        1,
                    )
                    >> (q16 * 8)
                )
                & fx.Int32(255)
                for mi in range(ring.mr)
            ]

        def mma(k, mi, c, acc, wk, a, sa):
            j, gu = c // 2, c % 2
            return _mfma(
                atoms[(k % 2) * 2 + gu], acc, wk[j][gu], a, sw_pairs[k // 2][j], sa[mi]
            )

        acc = ring.run(
            issue=issue,
            load_w=load_w,
            read_a=lambda k, mi: ring.fragment(lds_base, lane, k, mi),
            read_sa=read_sa,
            mma=mma,
            w_ring=w_ring,
        )

        q_tile = ptr_buf_tensor(
            buf_base_i64(q) + fx.Int64(m_tile) * fx.Int64(tile_rows * inter_dim),
            fx.Int32,
            num_records_bytes=bm * inter_dim,
        )
        qs_tile = ptr_buf_tensor(
            buf_base_i64(qs) + fx.Int64(m_tile) * fx.Int64(tile_rows * inter_dim // 32),
            fx.Int8,
            num_records_bytes=bm * inter_dim // 32,
        )
        c16_0 = n_tile * fx.Int32(tile_n // 32) + wave * fx.Int32(n_wave // 32)
        c64 = fx.Int32(64)
        for mi in range(ring.mr):
            ok = fx.Int32(mi * 16) + l16 < rows
            for g in range(ni // 2):
                y = [
                    [
                        silu_mul_f32(
                            Vec(acc[mi][(2 * g + t) * 2])[v],
                            Vec(acc[mi][(2 * g + t) * 2 + 1])[v],
                            swiglu_limit,
                        )
                        for v in range(4)
                    ]
                    for t in range(2)
                ]
                e8m0, quant = _e8m0_scale(_group_absmax(y[0] + y[1], (16, 32), c64))
                for t in range(2):
                    lo = rocdl.cvt_pk_fp8_f32(
                        T.i32, y[t][0] * quant, y[t][1] * quant, fx.Int32(0), False
                    )
                    pk = fx.Int32(
                        rocdl.cvt_pk_fp8_f32(
                            T.i32, y[t][2] * quant, y[t][3] * quant, lo, True
                        )
                    )
                    c16 = c16_0 + fx.Int32(2 * g + t)
                    off = (
                        (fx.Int32(mi * kt2) + (c16 >> fx.Int32(3))) * 512
                        + ((c16 >> fx.Int32(2)) & fx.Int32(1)) * 256
                        + ((c16 & fx.Int32(3)) * 16 + l16) * 4
                        + q16
                    )
                    buf_copy_store(
                        q_tile,
                        ok.select(off, fx.Int32(bm * inter_dim // 4)),
                        pk,
                        fx.Int32,
                        cache_modifier=_G1_CM,
                    )
                sg = (c16_0 >> fx.Int32(1)) + fx.Int32(g)
                s_off = (
                    fx.Int32(mi * kt2 * 64)
                    + (sg >> fx.Int32(2)) * 64
                    + l16 * 4
                    + (sg & fx.Int32(3))
                )
                buf_copy_store(
                    qs_tile,
                    (ok & (q16 == fx.Int32(0))).select(
                        s_off, fx.Int32(bm * inter_dim // 32)
                    ),
                    e8m0.to(fx.Int8),
                    fx.Int8,
                    cache_modifier=_G1_CM,
                )

    return tile


def make_ring_gemm2(*, model_dim, inter_dim, bm, bn, nw, pf, nbuf, tile_rows=None):
    """GEMM2 of one unit (``bm`` rows x ``bn`` hidden columns of an m-tile of
    ``tile_rows``, default ``bm``), traced in the kernel.

    Returns ``(start, run)``: ``start`` issues the first ``pf`` steps' weight loads
    (they do not depend on GEMM1), ``run`` the rest, each with its weight cache
    policy ``w_cm``. The accumulators are transposed fragments
    (``p2p_scatter_epilog`` ``acc_t``)."""
    ni = bn // 16 // nw
    assert ni % 2 == 0 and inter_dim % 256 == 0
    ring = _Ring(
        kt=inter_dim // _STEP_K,
        mr=bm // 16,
        a_bytes=_STEP_K,
        n_mma=ni,
        nw=nw,
        pf=pf,
        nbuf=nbuf,
    )
    kc = inter_dim // 256
    tile_rows = tile_rows or bm
    sc_bytes = bm * inter_dim // 32
    sc_rounds = _rounds(sc_bytes, nw)
    sc_base = nbuf * ring.pitch
    atoms = [
        fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, fx.Float4E2M1FN, fx.Float8E4M3FN, opsel_a=o, opsel_b=0
            )
        )
        for o in range(4)
    ]

    def start(*, expert, n_block, unit_zero, w, sw, w_cm):
        lane, l16, q16, wave = _lane_ids(unit_zero)
        wbuf = ptr_buf_tensor(w, fx.Int32, unit_elems=4)
        swbuf = ptr_buf_tensor(sw, fx.Int32)
        blk0 = (
            expert * fx.Int32(model_dim // 16)
            + n_block * fx.Int32(bn // 16)
            + wave * fx.Int32(ni)
        )
        w_unit = [
            (blk0 + fx.Int32(j)) * fx.Int32(inter_dim // 2) + q16 * 16 + l16
            for j in range(ni)
        ]
        sw_unit = [
            ((blk0 >> fx.Int32(1)) + fx.Int32(p)) * fx.Int32(kc * 64) + q16 * 16 + l16
            for p in range(ni // 2)
        ]
        sw_pairs = _w_scale_pairs(ring.kt)

        def load_w(k, cm, pairs):
            box = [None] * ni
            pieces = []
            for j in range(ni):

                def piece(j=j):
                    box[j] = _load_w(wbuf, w_unit[j], k * 256, cm)

                pieces.append(piece)
            if k % 2 == 0:

                def scales():
                    pairs[k // 2] = [
                        _load_w(swbuf, sw_unit[p], (k // 2) * 64, 0, 1)
                        for p in range(ni // 2)
                    ]

                pieces.append(scales)
            return pieces, lambda: box

        w_ring = [load_all(load_w(k, w_cm, sw_pairs)) for k in range(pf)]
        return (lane, l16, q16, wave, load_w, sw_pairs), w_ring

    def run(ctx, w_ring, *, lds_base, lds_i8, m_tile, aq, ascale, w_cm):
        lane, l16, q16, wave, load_w, sw_pairs = ctx
        sw_pairs = list(sw_pairs)
        a_view = _dma_view(
            buf_base_i64(aq) + fx.Int64(m_tile) * fx.Int64(tile_rows * inter_dim),
            bm * inter_dim,
        )
        s_view = _dma_view(
            buf_base_i64(ascale)
            + fx.Int64(m_tile) * fx.Int64(tile_rows * inter_dim // 32),
            sc_bytes,
        )
        a_src = []
        for d in ring.slot_ids(wave, nw):
            a_src.append(
                (
                    d,
                    (d >> fx.Int32(1)) * (ring.kt * 2 * _SLOT)
                    + (d & fx.Int32(1)) * _SLOT
                    + lane * 16,
                )
            )

        def issue(t):
            if t >= ring.kt:
                return
            for d, src in a_src:
                _dma(
                    ring.atom,
                    a_view,
                    src,
                    lds_i8,
                    fx.Int32((t % nbuf) * ring.pitch) + d * _SLOT,
                    t * 2 * _SLOT,
                )
            if t == 0:
                for qr in range(sc_rounds):
                    c = wave + fx.Int32(nw * qr)
                    _dma(
                        ring.atom,
                        s_view,
                        c * _SLOT + lane * 16,
                        lds_i8,
                        fx.Int32(sc_base) + c * _SLOT,
                        0,
                    )
            rocdl.asyncmark()

        def read_sa(k):
            return [
                (
                    _lds_i32(
                        lds_base,
                        fx.Int32(sc_base + (mi * ring.kt + k) * 64) + l16 * 4,
                        1,
                    )
                    >> (q16 * 8)
                )
                & fx.Int32(255)
                for mi in range(ring.mr)
            ]

        def mma(k, mi, c, acc, wk, a, sa):
            return _mfma(
                atoms[(k % 2) * 2 + c % 2],
                acc,
                wk[c],
                a,
                sw_pairs[k // 2][c // 2],
                sa[mi],
            )

        return ring.run(
            issue=issue,
            load_w=lambda k: load_w(k, w_cm, sw_pairs),
            read_a=lambda k, mi: ring.fragment(lds_base, lane, k, mi),
            read_sa=read_sa,
            mma=mma,
            w_ring=w_ring,
        )

    return start, run
