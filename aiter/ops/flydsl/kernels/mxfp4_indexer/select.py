# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact top-512 tokens of a row of the MXFP4 index cache, one workgroup per row.

Dense mode starts from the row's block maxima (``scores``). A block's best token is its first token of maximal
logit, and the tokens are ordered (logit descending, position ascending), so the 512 best tokens lie in the 512 best
blocks (a block outside them has its best token below 512 others). The workgroup bins the maxima in 4096 equal
fp32 buckets between the row's extremes (a monotone map), takes the buckets from the top until 512 blocks are
covered (>= 512, normally a few more, at most ``kbmax``; otherwise an exact four-pass radix select of 512), and
scores every token of those blocks again with the same MFMA and the same arithmetic as the block scores, so a token's
logit is the same bits in both. Gather mode takes the blocks from a candidate list (ascending block ids, -1 padded)
instead.

The tokens' logits (4 blocks per 32-token tile, a wave's tiles strided so a short list keeps every wave busy) go to
LDS and
the best 512 are selected the same way: bucket counts, the bucket holding the 512th (``need`` winners among its few
elements, ranked by comparison; a radix select over it if it holds more than ``capm``), then an ordered compaction.
Ties between equal logits go to the lower position. The output is the selected positions ascending, -1 padded.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .. import buffer_ops
from .common import (
    BLOCK,
    CANDIDATES,
    I64,
    INF,
    NBINS,
    NEG_INF,
    TOPK,
    F,
    H,
    I,
    U,
    bf,
    bi,
    gptr,
    key_of,
    maxf,
    mfma_fp4,
    minf,
    not_nan,
    token_logit,
    wave_scan_incl,
    xor_lane,
)


@functools.cache
def compile_select(
    next_n: int,
    page_entries: int,
    blocks: int,
    gather: bool,
    nt: int = 1024,
    tiles_in_flight: int = 4,
    kbmax: int = 768,
    capm: int = 128,
    emit_candidates: bool = False,
    slot_log2: int = 0,
    slot_rows: int = 1024,
    skip_long: bool = False,
):
    """``blocks``: the most 8-token blocks of a row's context (a multiple of 4 * ``nt``); ``gather``: the blocks come
    from a candidate list of 2048, else from the block maxima; ``emit_candidates`` (dense): also write the row's
    candidate list; ``slot_log2`` (> 0): also write the sparse attention's ragged slot lists (entries per page of the
    attention cache = 2 ** slot_log2, for up to ``slot_rows`` rows, a multiple of ``nt``); ``skip_long``: a row of more than
    ``blocks`` * 8 entries is left to the long path (``blocks``), not truncated."""
    P, NN, NT, NB = page_entries, next_n, nt, blocks
    KB = CANDIDATES if gather else kbmax
    NW = NT // 64
    EA = NB // NT
    NTOK = KB * BLOCK
    EB = NTOK // NT
    TPW = NTOK // 32 // NW
    BT = min(tiles_in_flight, TPW)
    BPT = NBINS // NT
    NBT = max(NT, NB * BLOCK // P)
    name = (
        f"indexer_mxfp4_select_h32_d128_top512_nn{NN}_p{P}_nb{NB}_{'gather' if gather else 'dense'}_t{NT}"
        + ("_cand" if emit_candidates else "")
        + (f"_slots{slot_log2}_r{slot_rows}" if slot_log2 else "")
        + ("_skiplong" if skip_long else "")
    )
    MKW = (max(NB, NTOK) + 31) // 32 + 1
    NSG = max(NT, NB * BLOCK >> slot_log2) if slot_log2 else 16
    MAX_ROWS = slot_rows
    assert (
        NT in (512, 1024)
        and TPW % BT == 0
        and (gather or EA % 4 == 0)
        and not (gather and emit_candidates)
    )

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, NBINS]
        red: fx.Array[fx.Int32, 256]
        wsum: fx.Array[fx.Int32, 128]
        bc: fx.Array[fx.Int32, 64]
        bl: fx.Array[fx.Int32, KB + 4]
        kk: fx.Array[fx.Int32, NTOK]
        mk: fx.Array[fx.Int32, MKW]
        cl: fx.Array[fx.Int32, 2 * capm + 4]
        pg: fx.Array[fx.Int32, NBT]
        sg: fx.Array[fx.Int32, NSG]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        scores: fx.Int64,
        candidates: fx.Int64,
        cand_out: fx.Int64,
        q: fx.Int64,
        q_scale: fx.Int64,
        weights: fx.Int64,
        kv: fx.Int64,
        block_table: fx.Int64,
        row_ends: fx.Int64,
        out: fx.Int64,
        page_stride: fx.Int64,
        bt_stride: fx.Int32,
        scores_stride: fx.Int32,
        reqs: fx.Int32,
        slots: fx.Int64,
        indptr: fx.Int64,
        lens: fx.Int64,
        valid: fx.Int64,
        slot_table: fx.Int64,
        slot_stride: fx.Int32,
    ):
        sm = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        lane = tx & 63
        wave = I(fx.rocdl.readfirstlane(T.i32, tx >> 6))
        col = lane & 31
        h = lane >> 5
        # workgroup w runs on XCD w % 8: a request's rows share that XCD's L2 (they read mostly the same keys)
        wg = fx.block_idx.x
        req = (wg // 8 // NN) * 8 + wg % 8
        row = req * NN + (wg // 8) % NN

        def res(addr, n):
            return buffer_ops.create_buffer_resource_from_addr(
                addr, num_records_bytes=n
            )

        # Scratch words (wave sums, block reductions, bucket results) rotate through four regions: a region is not
        # written again before two barriers have passed, so none needs a barrier after its reads.
        turn = [0, 0, 0]

        def region(kind, size):
            turn[kind] += 1
            return (turn[kind] % 4) * size

        def bscan(v):
            """Exclusive prefix across the block's threads, block total."""
            w0 = region(0, 32)
            inc = wave_scan_incl(v)
            sm.wsum.ptr[w0 + (lane == 63).select(wave, I(31))] = inc
            fx.gpu.barrier()
            base, tot = I(0), I(0)
            for ww in range_constexpr(NW):
                s = I(sm.wsum.ptr[w0 + ww])
                base = (wave > I(ww)).select(base + s, base)
                tot = tot + s
            return inc - v + base, tot

        def bminmax(vmax, vmin):
            """Block maximum of ``vmax`` and minimum of ``vmin``."""
            r0 = region(1, 64)
            for m_ in (1, 2, 4, 8, 16, 32):
                vmax = maxf(vmax, bf(xor_lane(bi(vmax), m_, lane)))
                vmin = minf(vmin, bf(xor_lane(bi(vmin), m_, lane)))
            sm.red.ptr[r0 + (lane == 0).select(wave, I(62))] = bi(vmax)
            sm.red.ptr[r0 + (lane == 0).select(wave + 32, I(63))] = bi(vmin)
            fx.gpu.barrier()
            hi, lo = bf(I(sm.red.ptr[r0])), bf(I(sm.red.ptr[r0 + 32]))
            for ww in range_constexpr(1, NW):
                hi = maxf(hi, bf(I(sm.red.ptr[r0 + ww])))
                lo = minf(lo, bf(I(sm.red.ptr[r0 + 32 + ww])))
            return hi, lo

        def radix(keys, E, ksel):
            """Exact select of the ksel best keys (0 = invalid) by four 8-bit passes: (thr, take, has_thr, off, total),
            ties to the lower index."""
            prefix, rem = I(0), I(ksel)
            for ps in range_constexpr(4):
                shift = 24 - 8 * ps
                sm.hist.ptr[tx & 255] = I(0)
                fx.gpu.barrier()
                for j in range_constexpr(E):
                    k = keys[j]
                    if const_expr(ps == 0):
                        fx.atomic_add(sm.hist.ptr + ((k >> shift) & 255), I(1))
                    else:
                        if (k.bitcast(U) >> U(shift + 8)) == (
                            prefix.bitcast(U) >> U(shift + 8)
                        ):
                            fx.atomic_add(sm.hist.ptr + ((k >> shift) & 255), I(1))
                fx.gpu.barrier()
                c = (tx < 256).select(I(sm.hist.ptr[255 - (tx & 255)]), I(0))
                inc = wave_scan_incl(c)
                sm.wsum.ptr[(lane == 63).select(wave, I(31))] = inc
                fx.gpu.barrier()
                base = I(0)
                for ww in range_constexpr(4):
                    s = I(sm.wsum.ptr[ww])
                    base = (wave > I(ww)).select(base + s, base)
                cum = inc + base
                hit = (tx < 256) & (cum >= rem) & ((cum - c) < rem)
                sm.bc.ptr[hit.select(I(0), I(8))] = I(255) - tx
                sm.bc.ptr[hit.select(I(1), I(9))] = rem - (cum - c)
                fx.gpu.barrier()
                sb, rem = I(sm.bc.ptr[0]), I(sm.bc.ptr[1])
                prefix = prefix | (sb << shift)
                fx.gpu.barrier()
            gt, eq = I(0), I(0)
            for j in range_constexpr(E):
                gt = gt + (keys[j].bitcast(U) > prefix.bitcast(U)).select(I(1), I(0))
                eq = eq + (keys[j] == prefix).select(I(1), I(0))
            eq_before, _ = bscan(eq)
            take = rem - eq_before
            take = (take < I(0)).select(I(0), take)
            take = (take > eq).select(eq, take)
            has_thr = prefix != I(0)
            off, total = bscan(gt + has_thr.select(take, I(0)))
            return prefix, take, has_thr, off, total

        def histogram(vals, E):
            """The bucket of every element of ``vals`` (-inf = invalid) between the extremes, counted in ``hist``."""
            for qq in range_constexpr(NBINS // NT):
                sm.hist.ptr[tx + qq * NT] = I(0)
            vmax, vmin = F(NEG_INF), F(INF)
            for j in range_constexpr(E):
                vmax = maxf(vmax, vals[j])
                vmin = minf(vmin, (vals[j] > F(NEG_INF)).select(vals[j], F(INF)))
            vmax, vmin = bminmax(vmax, vmin)
            span = vmax - vmin
            inv = (span > F(0.0)).select(F(NBINS - 1) * F(0.999999) / span, F(0.0))
            bins = []
            for j in range_constexpr(E):
                b = ((vals[j] - vmin) * inv).to(I)
                b = (b < I(0)).select(I(0), b)
                bins.append((b > I(NBINS - 1)).select(I(NBINS - 1), b))
            for j in range_constexpr(E):
                if vals[j] > F(NEG_INF):
                    fx.atomic_add(sm.hist.ptr + bins[j], I(1))
            fx.gpu.barrier()
            return bins

        def find(ksel, nvalid):
            """The bucket b* holding the ksel-th best of the counted elements (NBINS: none), the count above it and in
            it, and ksel clipped to nvalid."""
            w0, c0 = region(0, 32), region(2, 16)
            sm.bc.ptr[c0 + (tx < 3).select(tx, I(8))] = (tx == 0).select(I(NBINS), I(0))
            hv = [
                I(sm.hist.ptr[NBINS - 1 - (tx * BPT + i)]) for i in range_constexpr(BPT)
            ]
            s = I(0)
            for i in range_constexpr(BPT):
                s = s + hv[i]
            inc = wave_scan_incl(s)
            sm.wsum.ptr[w0 + (lane == 63).select(wave, I(31))] = inc
            fx.gpu.barrier()
            base = I(0)
            for ww in range_constexpr(NW):
                base = (wave > I(ww)).select(base + I(sm.wsum.ptr[w0 + ww]), base)
            rem = (nvalid < I(ksel)).select(nvalid, I(ksel))
            run = inc + base - s
            hit = (inc + base >= rem) & (run < rem)
            bstar, cab, mcnt, found = I(0), I(0), I(0), I(0)
            for i in range_constexpr(BPT):
                nxt = run + hv[i]
                take_ = (found == I(0)) & (nxt >= rem) & (run < rem)
                bstar = take_.select(I(NBINS - 1) - (tx * BPT + i), bstar)
                cab = take_.select(run, cab)
                mcnt = take_.select(hv[i], mcnt)
                found = take_.select(I(1), found)
                run = nxt
            wr = hit & (found != I(0))
            sm.bc.ptr[c0 + wr.select(I(0), I(8))] = bstar
            sm.bc.ptr[c0 + wr.select(I(1), I(9))] = cab
            sm.bc.ptr[c0 + wr.select(I(2), I(10))] = mcnt
            fx.gpu.barrier()
            return I(sm.bc.ptr[c0]), I(sm.bc.ptr[c0 + 1]), I(sm.bc.ptr[c0 + 2]), rem

        def resolve(vals, bins, skip, bs, mc, need, E):
            """Marks (bits of ``mk``) the ``need`` best elements of bucket ``bs`` (``mc``, none of ``skip``) when
            they are few enough to rank against each other; returns whether they were (else the caller selects by
            radix)."""
            small = mc <= I(capm)
            inb = [
                (bins[j] == bs) & (vals[j] > F(NEG_INF)) & ~skip[j] & small
                for j in range_constexpr(E)
            ]
            for qq in range_constexpr(-(-MKW // NT)):
                i_ = tx + qq * NT
                sm.mk.ptr[(i_ < MKW).select(i_, I(MKW - 1))] = I(0)
            cnt = I(0)
            for j in range_constexpr(E):
                cnt = cnt + inb[j].select(I(1), I(0))
            off, _t = bscan(cnt)
            c = I(0)
            for j in range_constexpr(E):
                sm.cl.ptr[inb[j].select(off + c, I(2 * capm + 2))] = key_of(vals[j])
                sm.cl.ptr[inb[j].select(I(capm) + off + c, I(2 * capm + 3))] = (
                    tx * E + j
                )
                c = c + inb[j].select(I(1), I(0))
            fx.gpu.barrier()
            mce = small.select(mc, I(0))
            mykey = I(sm.cl.ptr[(tx < mce).select(tx, I(2 * capm))])
            myidx = I(sm.cl.ptr[I(capm) + (tx < mce).select(tx, I(0))])
            for jj, st in range(0, I64(mce), 1, init=[I(0)]):
                kj, ij = I(sm.cl.ptr[I(jj)]), I(sm.cl.ptr[I(capm) + I(jj)])
                better = (kj.bitcast(U) > mykey.bitcast(U)) | (
                    (kj == mykey) & (ij < myidx)
                )
                rank = yield [st[0] + better.select(I(1), I(0))]
            win = (tx < mce) & (rank < need)
            fx.atomic_or(
                sm.mk.ptr + (win.select(myidx, I(MKW * 32 - 1)) >> 5),
                win.select(I(1) << (myidx & 31), I(0)),
            )
            fx.gpu.barrier()
            return small

        def marked(e):
            return ((I(sm.mk.ptr[e >> 5]) >> (e & 31)) & 1) != I(0)

        go = req < reqs
        if const_expr(skip_long):
            raw_end = I(
                buffer_ops.buffer_load(
                    res(row_ends, I64(reqs * NN) * 4), row, vec_width=1, dtype=I
                )
            )
            go = go & (raw_end <= I(NB * BLOCK))
        if go:
            # the first loads: the block maxima or candidate ids, whose latency the rest overlaps
            if const_expr(gather):
                cr = res(candidates + I64(row) * (CANDIDATES * 4), CANDIDATES * 4)
                ids = [
                    I(
                        buffer_ops.buffer_load(
                            cr, tx * (CANDIDATES // NT) + j, vec_width=1, dtype=I
                        )
                    )
                    for j in range_constexpr(CANDIDATES // NT)
                ]
            else:
                wsr = res(
                    scores + I64(row) * I64(scores_stride) * 4, I64(scores_stride) * 4
                )
                raw = [
                    Vec(
                        buffer_ops.buffer_load(
                            wsr, tx * EA + qq * 4, vec_width=4, dtype=F
                        )
                    )
                    for qq in range_constexpr(EA // 4)
                ]
            end = I(
                buffer_ops.buffer_load(
                    res(row_ends + I64(row) * 4, 4),
                    0,
                    vec_width=1,
                    dtype=I,
                    is_scalar=True,
                )
            )
            end = (end < I(0)).select(I(0), end)
            end = (end > I(NB * BLOCK)).select(I(NB * BLOCK), end)
            nb = (end + 7) // 8
            orr = res(out + I64(row) * (TOPK * 4), TOPK * 4)

            sm.bc.ptr[tx & 15] = I(0)
            # q, weights, this request's page ids: independent of the blocks, loaded first
            qr = res(q + I64(row) * (H * 64), H * 64)
            qsw = I(
                buffer_ops.buffer_load(
                    res(q_scale + I64(row) * (H * 4), H * 4), col, vec_width=1, dtype=I
                )
            )
            qv = [
                Vec(
                    buffer_ops.buffer_load(
                        qr, (col * 64 + kb * 32 + h * 16) // 4, vec_width=4, dtype=I
                    )
                )
                for kb in range_constexpr(2)
            ]
            sa = [(qsw >> ((2 * kb + h) * 8)) & 0xFF for kb in range_constexpr(2)]
            wr_ = res(weights + I64(row) * (H * 4), H * 4)
            wv = []
            for j in range_constexpr(4):
                v = Vec(
                    buffer_ops.buffer_load(wr_, 8 * j + 4 * h, vec_width=4, dtype=F)
                )
                wv += [F(v[0]), F(v[1]), F(v[2]), F(v[3])]
            btr = res(block_table + I64(req) * I64(bt_stride) * 4, I64(bt_stride) * 4)
            for qq in range_constexpr(-(-NBT // NT)):
                sm.pg.ptr[tx + qq * NT] = I(
                    buffer_ops.buffer_load(btr, tx + qq * NT, vec_width=1, dtype=I)
                )
            if const_expr(slot_log2 > 0):
                # the ragged lists: this row starts after the rows before it, each min(512, end) long (0 when not valid)
                rows_t = reqs * NN
                vr = res(valid, I64(rows_t) * 4)
                er = res(row_ends, I64(rows_t) * 4)
                before = I(0)
                for qq in range_constexpr(MAX_ROWS // NT):
                    j_ = tx + qq * NT
                    e_j = I(buffer_ops.buffer_load(er, j_, vec_width=1, dtype=I))
                    v_j = I(buffer_ops.buffer_load(vr, j_, vec_width=1, dtype=I))
                    ec = (e_j < I(0)).select(I(0), e_j)
                    l_j = (v_j != I(0)).select((ec > I(TOPK)).select(I(TOPK), ec), I(0))
                    before = before + (j_ < row).select(l_j, I(0))
                _o, start = bscan(before)
                my_valid = I(
                    buffer_ops.buffer_load(
                        vr, row, vec_width=1, dtype=I, is_scalar=True
                    )
                ) != I(0)
                my_len = my_valid.select((end < I(TOPK)).select(end, I(TOPK)), I(0))
                sgr = res(
                    slot_table + I64(req) * I64(slot_stride) * 4, I64(slot_stride) * 4
                )
                for qq in range_constexpr(-(-NSG // NT)):
                    i_ = tx + qq * NT
                    sm.sg.ptr[(i_ < NSG).select(i_, I(NSG - 1))] = I(
                        buffer_ops.buffer_load(sgr, i_, vec_width=1, dtype=I)
                    )
                ipr = res(indptr, I64(rows_t + 1) * 4)
                buffer_ops.buffer_store(start, ipr, row, mask=tx == 0)
                buffer_ops.buffer_store(
                    start + my_len, ipr, rows_t, mask=(tx == 0) & (row == rows_t - 1)
                )
                buffer_ops.buffer_store(
                    my_len, res(lens, I64(rows_t) * 4), row, mask=tx == 0
                )

            # ---- the blocks to score: ascending ids in LDS
            if const_expr(gather):
                ok_ = [(i_ >= I(0)) & (i_ < nb) for i_ in ids]
                cnt0 = I(0)
                for j in range_constexpr(CANDIDATES // NT):
                    cnt0 = cnt0 + ok_[j].select(I(1), I(0))
                off, nsurv = bscan(cnt0)
                c = I(0)
                for j in range_constexpr(CANDIDATES // NT):
                    sm.bl.ptr[ok_[j].select(off + c, I(KB))] = ids[j]
                    c = c + ok_[j].select(I(1), I(0))
            else:
                vals = []
                for qq in range_constexpr(EA // 4):
                    for j in range_constexpr(4):
                        x = F(raw[qq][j])
                        vals.append(
                            ((tx * EA + qq * 4 + j < nb) & not_nan(x)).select(
                                x, F(NEG_INF)
                            )
                        )
                bins = histogram(vals, EA)
                bs, cab, mc, rem = find(TOPK, nb)
                if const_expr(emit_candidates):
                    # the 2048 candidate blocks: the newest key's block, then the best 2047 others
                    pin = end > I(0)
                    pb = (end - 1) >> 3
                    pe = [pin & (tx * EA + j == pb) for j in range_constexpr(EA)]
                    for j in range_constexpr(EA):
                        if pe[j]:
                            fx.atomic_add(sm.hist.ptr + bins[j], I(-1))
                    fx.gpu.barrier()
                    others = pin.select(nb - 1, I(0))
                    bs2, cab2, mc2, rem2 = find(CANDIDATES - 1, others)
                    small2 = resolve(vals, bins, pe, bs2, mc2, rem2 - cab2, EA)
                    if ~small2:
                        bk = [
                            ((bins[j] == bs2) & (vals[j] > F(NEG_INF)) & ~pe[j]).select(
                                key_of(vals[j]), I(0)
                            )
                            for j in range_constexpr(EA)
                        ]
                        thr, take, has_thr, _offr, _totalr = radix(bk, EA, rem2 - cab2)
                        eqs = I(0)
                        for j in range_constexpr(EA):
                            w_ = (bk[j].bitcast(U) > thr.bitcast(U)) | (
                                (bk[j] == thr) & has_thr & (eqs < take)
                            )
                            e_ = tx * EA + j
                            fx.atomic_or(
                                sm.mk.ptr + (w_.select(e_, I(MKW * 32 - 1)) >> 5),
                                w_.select(I(1) << (e_ & 31), I(0)),
                            )
                            eqs = eqs + (bk[j] == thr).select(I(1), I(0))
                        fx.gpu.barrier()
                    chosen = [
                        pe[j]
                        | (
                            (vals[j] > F(NEG_INF))
                            & ~pe[j]
                            & ((bins[j] > bs2) | marked(tx * EA + j))
                        )
                        for j in range_constexpr(EA)
                    ]
                    cnt2 = I(0)
                    for j in range_constexpr(EA):
                        cnt2 = cnt2 + chosen[j].select(I(1), I(0))
                    off3, total3 = bscan(cnt2)
                    cor = res(cand_out + I64(row) * (CANDIDATES * 4), CANDIDATES * 4)
                    c3 = I(0)
                    for j in range_constexpr(EA):
                        buffer_ops.buffer_store(
                            tx * EA + j, cor, off3 + c3, mask=chosen[j]
                        )
                        c3 = c3 + chosen[j].select(I(1), I(0))
                    for qq in range_constexpr(CANDIDATES // NT):
                        p = tx + qq * NT
                        buffer_ops.buffer_store(I(-1), cor, p, mask=p >= total3)
                stotal = cab + mc
                big = stotal > I(KB)
                inset = [
                    (bins[j] >= bs) & (vals[j] > F(NEG_INF))
                    for j in range_constexpr(EA)
                ]
                sel = I(0)
                for j in range_constexpr(EA):
                    sel = sel + inset[j].select(I(1), I(0))
                off, _t = bscan(sel)
                c = I(0)
                for j in range_constexpr(EA):
                    s_ = inset[j] & (~big)
                    sm.bl.ptr[s_.select(off + c, I(KB))] = tx * EA + j
                    c = c + s_.select(I(1), I(0))
                if big:
                    keys = [key_of(vals[j]) for j in range_constexpr(EA)]
                    thr, take, has_thr, off2, _total2 = radix(keys, EA, TOPK)
                    c2, eqs = I(0), I(0)
                    for j in range_constexpr(EA):
                        s2 = (keys[j].bitcast(U) > thr.bitcast(U)) | (
                            (keys[j] == thr) & has_thr & (eqs < take)
                        )
                        sm.bl.ptr[s2.select(off2 + c2, I(KB))] = tx * EA + j
                        c2 = c2 + s2.select(I(1), I(0))
                        eqs = eqs + (keys[j] == thr).select(I(1), I(0))
                nsurv = big.select(rem, stotal)
            for qq in range_constexpr(-(-KB // NT)):
                i_ = tx + qq * NT
                sm.bl.ptr[((i_ >= nsurv) & (i_ < KB)).select(i_, I(KB))] = I(0)
            fx.gpu.barrier()
            ntok = nsurv * BLOCK

            # ---- logits of the blocks' tokens, 4 blocks (32 tokens) a tile
            ntiles = (nsurv + 3) // 4
            LOGP = P.bit_length() - 1
            for t0 in range_constexpr(0, TPW, BT):
                if t0 * NW < ntiles:
                    info = []
                    for ti in range_constexpr(t0, t0 + BT):
                        tile = ti * NW + wave
                        bi_ = tile * 4 + (col >> 3)
                        pos = (I(sm.bl.ptr[bi_]) << 3) | (col & 7)
                        info.append(
                            (
                                tile,
                                pos,
                                (bi_ < nsurv) & (pos < end),
                                I(sm.pg.ptr[pos >> LOGP]),
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
                                        fx.generic_load(
                                            gptr(a2, 2, T.i16), dtype=fx.Int16
                                        )
                                    ).to(I)
                                )
                                & 0xFFFF,
                            )
                        )
                    for (tile, pos, ok, page), (kd0, kd1, ksw) in zip(info, ks):
                        acc = mfma_fp4(
                            qv[0], kd0, Vec.filled(16, 0.0, F), sa[0], ksw, 0
                        )
                        acc = mfma_fp4(qv[1], kd1, acc, sa[1], ksw, 1)
                        sm.kk.ptr[tile * 32 + col] = ok.select(
                            token_logit(acc, wv), F(NEG_INF)
                        ).bitcast(I)
            fx.gpu.barrier()

            # ---- the 512 best tokens
            vals = []
            for j in range_constexpr(EB):
                e = tx * EB + j
                x = I(sm.kk.ptr[e]).bitcast(F)
                vals.append(((e < ntok) & not_nan(x)).select(x, F(NEG_INF)))
            nvalid = I(0)
            for j in range_constexpr(EB):
                nvalid = nvalid + (vals[j] > F(NEG_INF)).select(I(1), I(0))
            _o, nvalid = bscan(nvalid)
            bins = histogram(vals, EB)
            bs, cab, mc, rem = find(TOPK, nvalid)
            small = resolve(vals, bins, [I(0) != I(0)] * EB, bs, mc, rem - cab, EB)
            if ~small:
                bk = [
                    ((bins[j] == bs) & (vals[j] > F(NEG_INF))).select(
                        key_of(vals[j]), I(0)
                    )
                    for j in range_constexpr(EB)
                ]
                thr, take, has_thr, _offr, _totalr = radix(bk, EB, rem - cab)
                eqs = I(0)
                for j in range_constexpr(EB):
                    w_ = (bk[j].bitcast(U) > thr.bitcast(U)) | (
                        (bk[j] == thr) & has_thr & (eqs < take)
                    )
                    e_ = tx * EB + j
                    fx.atomic_or(
                        sm.mk.ptr + (w_.select(e_, I(MKW * 32 - 1)) >> 5),
                        w_.select(I(1) << (e_ & 31), I(0)),
                    )
                    eqs = eqs + (bk[j] == thr).select(I(1), I(0))
                fx.gpu.barrier()
            chosen = [
                (vals[j] > F(NEG_INF)) & ((bins[j] > bs) | marked(tx * EB + j))
                for j in range_constexpr(EB)
            ]
            cnt = I(0)
            for j in range_constexpr(EB):
                cnt = cnt + chosen[j].select(I(1), I(0))
            off, total = bscan(cnt)
            c = I(0)
            for j in range_constexpr(EB):
                e = tx * EB + j
                pos_e = I(sm.bl.ptr[e >> 3]) * BLOCK + (e & 7)
                buffer_ops.buffer_store(pos_e, orr, off + c, mask=chosen[j])
                if const_expr(slot_log2 > 0):
                    slot = (I(sm.sg.ptr[pos_e >> slot_log2]) << slot_log2) | (
                        pos_e & ((1 << slot_log2) - 1)
                    )
                    buffer_ops.buffer_store(
                        slot,
                        res(slots, I64(reqs * NN) * (TOPK * 4)),
                        start + off + c,
                        mask=chosen[j] & my_valid,
                    )
                c = c + chosen[j].select(I(1), I(0))
            for qq in range_constexpr(-(-TOPK // NT)):
                p = tx + qq * NT
                buffer_ops.buffer_store(I(-1), orr, p, mask=(p >= total) & (p < TOPK))

    @flyc.jit
    def launch(
        scores: fx.Int64,
        candidates: fx.Int64,
        cand_out: fx.Int64,
        q: fx.Int64,
        q_scale: fx.Int64,
        weights: fx.Int64,
        kv: fx.Int64,
        block_table: fx.Int64,
        row_ends: fx.Int64,
        out: fx.Int64,
        page_stride: fx.Int64,
        bt_stride: fx.Int32,
        scores_stride: fx.Int32,
        reqs: fx.Int32,
        slots: fx.Int64,
        indptr: fx.Int64,
        lens: fx.Int64,
        valid: fx.Int64,
        slot_table: fx.Int64,
        slot_stride: fx.Int32,
        groups: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            scores,
            candidates,
            cand_out,
            q,
            q_scale,
            weights,
            kv,
            block_table,
            row_ends,
            out,
            page_stride,
            bt_stride,
            scores_stride,
            reqs,
            slots,
            indptr,
            lens,
            valid,
            slot_table,
            slot_stride,
        ).launch(grid=(fx.Int64(groups), 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
