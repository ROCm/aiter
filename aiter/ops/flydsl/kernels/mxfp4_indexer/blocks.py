# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The best blocks of a long row from its block maxima (``scores``): a segment select and a merge.

Contexts above the block path's 262144 entries (up to 1M: 131072 blocks) do not fit the block path's select (a block
maximum a register, the page tables in LDS). A row's 512 best tokens lie in its 512 best blocks by (maximum descending,
id ascending), so these kernels select exactly those blocks (512) and the candidate consumer's scorer and token select
(``cand_logits``, ``tokens`` gathered) pick the tokens: the same arithmetic as the block path, the same bits.
For the candidate source layer they select the block of the row's newest key and the 2047 best others
(``reference.candidates_reference``).

``seg``: one workgroup per (row, segment of ``SEGB`` blocks) takes the segment's ``k`` best blocks (the best blocks of a row
lie among its segments' best) and writes their ids and maxima, ascending, -1 / -inf padded, to the row's slots of a compact
list; ``merge``: one workgroup a row takes the ``k`` best of the compact list (entries in id order: ties to the lower id) and
writes the ids. The source layer's newest block enters its segment's list with maximum +inf (so the merge keeps it) and
the segment's selection leaves it out.

Selection of one workgroup (1024 threads, the maxima in registers by 4-vector groups, no loop): order-preserving u32 keys;
a pass takes the extreme keys, then levels of 4096-bucket histograms over the keys between them (bucket = (max - key) >>
shift: the resolution follows the keys' own scale, an outlier does not crowd the bucket of the k-th key) find the bucket of
the k-th best key; while it holds more than ``CAPM`` blocks a level re-bins that bucket (three levels reach single keys),
equal keys at the end are cut by block id (a level of id buckets: ties to the lower id). The blocks above the final bucket
are chosen at once (bits of a bitmask over the ids), the bucket's few blocks (a list in LDS) are ranked by comparison; a
scan of the bitmask writes the chosen ids ascending.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .common import (
    BLOCK,
    I64,
    INF,
    NEG_INF,
    F,
    I,
    U,
    bf,
    bi,
    load,
    maxf,
    minf,
    sload,
    store,
    view,
    wave_of,
    wave_scan_incl,
    xor_lane,
)

NT = 1024
NW = NT // 64
NBINS = 4096
BPT = NBINS // NT
HR = 2  # copies of the histogram: lanes with the same bucket rarely hit one word
CAPM = 512  # blocks of the k-th bucket ranked by comparison
LEVELS = 3  # key levels after the first: 4096 ** 3 > 2 ** 32
SEGB = 16384  # blocks of a segment (131072 entries)
PLAIN_SLOTS = 512  # blocks a segment (or the merge) keeps for the plain top-512
SOURCE_SLOTS = 2048  # for the candidate source layer


@functools.cache
def compile_block_select(
    mode: str, k: int, newest: bool, min_end: int = 0, merged: int = 4096
):
    """``mode`` ``seg`` or ``merge``; ``k``: blocks taken (512; 2047 for the source layer's segments, 2048 for its
    merge); ``newest``: the source layer's lists; rows of at most ``min_end`` entries are left alone; ``merged``: entries
    of the compact list a merge can read (a multiple of 4096), it reads the first ``used`` (a launch argument).
    """
    assert mode in ("seg", "merge") and k <= SOURCE_SLOTS
    SEG = mode == "seg"
    NB = SEGB if SEG else merged  # elements of a workgroup
    SLOTS = SOURCE_SLOTS if newest else PLAIN_SLOTS
    UN = NB // 4096  # vectors of 4 a thread loads
    MKW = NB // 32  # words of the bitmask over the element ids
    WPT = -(-MKW // NT)
    assert NB % 4096 == 0
    IDSHIFT = max(
        0, (NB // NBINS).bit_length() - 1
    )  # id buckets of the tie level: NB >> IDSHIFT <= NBINS
    name = (
        f"indexer_mxfp4_block_{mode}_h32_d128_top512_k{k}"
        + ("_newest" if newest else "")
        + (f"_min{min_end}" if min_end else "")
        + ("" if SEG else f"_n{NB}")
    )

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, NBINS * HR]
        mk: fx.Array[fx.Int32, MKW + 1]
        cc: fx.Array[fx.Uint64, CAPM + 16]
        wsum: fx.Array[fx.Int32, 64]
        red: fx.Array[fx.Int32, 128]
        bc: fx.Array[fx.Int32, 32]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        src: fx.Int64,  # seg: block maxima [rows, src_stride]; merge: the compact maxima [rows, cmp_stride]
        ids: fx.Int64,  # compact ids [rows, cmp_stride]: seg writes them, merge reads them
        vals: fx.Int64,  # compact maxima (seg writes them)
        row_ends: fx.Int64,
        cand_out: fx.Int64,  # merge: the list [rows, 2048]
        src_stride: fx.Int32,
        cmp_stride: fx.Int32,
        used: fx.Int32,  # merge: entries of the compact list the segments wrote
    ):
        sm = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        lane = tx & 63
        wave = wave_of(tx)
        row = fx.block_idx.x
        seg = fx.block_idx.y if SEG else I(0)

        def u(x):
            return x.bitcast(U)

        end = sload(view(row_ends + I64(row) * 4, 4), 0)
        end = (end < I(0)).select(I(0), end)
        base = seg * SEGB if SEG else I(0)
        nb = (end + (BLOCK - 1)) // BLOCK  # blocks of the row
        pin = end > I(0)
        pb = (end - 1) // BLOCK  # the newest key's block
        nbl = nb - base if SEG else used  # valid elements of this workgroup
        nbl = (nbl < I(0)).select(I(0), (nbl > I(NB)).select(I(NB), nbl))
        nvl = (nbl + 3) // 4
        sstride = src_stride if SEG else cmp_stride
        srow = src + I64(row) * I64(sstride) * 4
        sr = view(srow, I64(sstride) * 4, F)
        sr4 = view(srow, I64(sstride) * 4, F, 4)
        idr = view(ids + I64(row) * I64(cmp_stride) * 4, I64(cmp_stride) * 4)
        valr = view(vals + I64(row) * I64(cmp_stride) * 4, I64(cmp_stride) * 4, F)
        candr = view(cand_out + I64(row) * (SOURCE_SLOTS * 4), SOURCE_SLOTS * 4)
        sm.bc.ptr[tx & 31] = I(0)
        if tx < CAPM + 16:
            sm.cc.ptr[tx] = fx.Uint64(0)

        def key_of(x):
            b = x.bitcast(I)
            return b ^ ((b >> 31) | I(-2147483648))

        def sweep():
            """(element id, maximum, valid) of the 4 UN consecutive elements this thread holds."""
            vs = []
            for uu in range_constexpr(UN):
                v = tx * UN + uu
                vs.append((v, load(sr4, base + v * 4, F, 4, mask=v < nvl)))
            out = []
            for v, x in vs:
                for j in range_constexpr(4):
                    e = v * 4 + j
                    xj = F(x[j])
                    ok = (e < nbl) & (xj > F(NEG_INF))
                    if const_expr(newest and SEG):
                        ok = ok & ~(pin & (base + e == pb))
                    out.append((e, xj, ok))
            return out

        def block_max(vmax, vmin):
            for m_ in (1, 2, 4, 8, 16, 32):
                vmax = maxf(vmax, bf(xor_lane(bi(vmax), m_, lane)))
                vmin = minf(vmin, bf(xor_lane(bi(vmin), m_, lane)))
            sm.red.ptr[(lane == 0).select(wave, I(62))] = bi(vmax)
            sm.red.ptr[(lane == 0).select(wave + 32, I(63))] = bi(vmin)
            fx.gpu.barrier()
            hi, lo = bf(I(sm.red.ptr[0])), bf(I(sm.red.ptr[32]))
            for ww in range_constexpr(1, NW):
                hi = maxf(hi, bf(I(sm.red.ptr[ww])))
                lo = minf(lo, bf(I(sm.red.ptr[32 + ww])))
            fx.gpu.barrier()
            return hi, lo

        def block_scan(v):
            """Exclusive prefix across the block's threads, block total."""
            inc = wave_scan_incl(v)
            sm.wsum.ptr[(lane == 63).select(wave, I(63))] = inc
            fx.gpu.barrier()
            base_, tot = I(0), I(0)
            for ww in range_constexpr(NW):
                s = I(sm.wsum.ptr[ww])
                base_ = (wave > I(ww)).select(base_ + s, base_)
                tot = tot + s
            fx.gpu.barrier()
            return inc - v + base_, tot

        def pad_slots(total):
            """-1 / -inf in the slots past the ``total`` chosen."""
            for qq in range_constexpr(-(-(SLOTS if SEG else PAD) // NT)):
                p = tx + qq * NT
                if const_expr(SEG):
                    at = seg * SLOTS + p
                    m = (p >= total) & (p < SLOTS)
                    store(idr, at, I(-1), mask=m)
                    store(valr, at, F(NEG_INF), F, mask=m)
                else:
                    store(candr, p, I(-1), mask=(p >= total) & (p < PAD))

        PAD = SOURCE_SLOTS if k > PLAIN_SLOTS else PLAIN_SLOTS

        if end > I(min_end):
            if nbl > I(0):
                # ---- the extreme maxima
                vmax_, vmin_ = F(NEG_INF), F(INF)
                for _e, x, ok in sweep():
                    vmax_ = maxf(vmax_, ok.select(x, F(NEG_INF)))
                    vmin_ = minf(vmin_, ok.select(x, F(INF)))
                vmax, vmin = block_max(vmax_, vmin_)
                kmax = key_of(vmax)
                kmin = key_of(vmin)

                def histogram(binf):
                    """Counts in ``hist`` the elements ``binf`` puts in a bucket (-1: none)."""
                    for qq in range_constexpr(NBINS * HR // NT):
                        sm.hist.ptr[tx + qq * NT] = I(0)
                    fx.gpu.barrier()
                    for e, x, ok in sweep():
                        b = binf(e, x, ok)
                        if b >= I(0):
                            fx.atomic_add(
                                sm.hist.ptr + (b * HR + (lane & (HR - 1))), I(1)
                            )
                    fx.gpu.barrier()

                def find(need):
                    """(bucket, elements before it, elements in it, elements counted) of the ``need``-th best."""
                    hv = []
                    for i in range_constexpr(BPT):
                        acc_ = I(0)
                        for c_ in range_constexpr(HR):
                            acc_ = acc_ + I(sm.hist.ptr[(tx * BPT + i) * HR + c_])
                        hv.append(acc_)
                    s = hv[0] + hv[1] + hv[2] + hv[3]
                    pre, tot = block_scan(s)
                    rem = (tot < need).select(tot, need)
                    run = pre
                    bstar, cab, mcnt, found = I(0), I(0), I(0), I(0)
                    for i in range_constexpr(BPT):
                        nxt = run + hv[i]
                        t_ = (found == I(0)) & (nxt >= rem) & (run < rem)
                        bstar = t_.select(tx * BPT + i, bstar)
                        cab = t_.select(run, cab)
                        mcnt = t_.select(hv[i], mcnt)
                        found = t_.select(I(1), found)
                        run = nxt
                    hit = found != I(0)
                    sm.bc.ptr[hit.select(I(0), I(8))] = bstar
                    sm.bc.ptr[hit.select(I(1), I(9))] = cab
                    sm.bc.ptr[hit.select(I(2), I(10))] = mcnt
                    fx.gpu.barrier()
                    r = (I(sm.bc.ptr[0]), I(sm.bc.ptr[1]), I(sm.bc.ptr[2]), tot)
                    fx.gpu.barrier()
                    return r

                def span_shift(w):
                    """The shift whose 4096 buckets cover ``w`` + 1 keys (0: a bucket a key)."""
                    s = I(32) - I(fx.math.ctlz(w)) - I(12)
                    return (s < I(0)).select(I(0), s)

                def key_bin(hi_, lo_, shift_):
                    def binf(e, x, ok):
                        kk = key_of(x)
                        inside = ok & (u(kk) <= u(hi_)) & (u(kk) >= u(lo_))
                        b = I(u(hi_ - kk) >> u(shift_))
                        b = (u(b) > U(NBINS - 1)).select(I(NBINS - 1), b)
                        return inside.select(b, I(-1))

                    return binf

                def first_bin(hi_, shift_):
                    """The first level: every valid element is between the extremes, and (max - key) >> shift < 4096."""

                    def binf(e, x, ok):
                        return ok.select(I(u(hi_ - key_of(x)) >> u(shift_)), I(-1))

                    return binf

                def bucket_keys(hi_, lo_, shift_, bs_):
                    """The keys [lo, hi] of bucket ``bs_`` of the level (the last bucket reaches down to ``lo_``)."""
                    top = hi_ - (bs_ << shift_)
                    end_ = (bs_ + 1) << shift_
                    low = (u(end_) <= u(hi_) + U(1)).select(hi_ + 1 - end_, I(0))
                    low = (u(low) < u(lo_)).select(lo_, low)
                    return top, (bs_ == I(NBINS - 1)).select(lo_, low)

                hi, lo = kmax, kmin
                shift = span_shift(hi - lo)
                histogram(first_bin(hi, shift))
                bs, cab, mc, tot = find(I(k))
                take_all = tot <= I(k)  # every element is chosen
                need = I(k) - cab  # elements still wanted from the bucket
                hi, lo = bucket_keys(hi, lo, shift, bs)
                for _lvl in range_constexpr(LEVELS):
                    crowded = (~take_all) & (mc > I(CAPM)) & (hi != lo)
                    if crowded:
                        shift = span_shift(hi - lo)
                        histogram(key_bin(hi, lo, shift))
                        bs, cab, mc, tot = find(need)
                        need = need - cab
                        hi, lo = bucket_keys(hi, lo, shift, bs)
                tie = I(0)
                tbs = I(0)
                crowded = (~take_all) & (mc > I(CAPM))  # one key a bucket: cut by id
                if crowded:

                    def id_bin(e, x, ok):
                        return (ok & (key_of(x) == hi)).select(e >> IDSHIFT, I(-1))

                    histogram(id_bin)
                    bs, cab, mc, tot = find(need)
                    need = need - cab
                    tie = I(1)
                    tbs = bs

                # ---- the elements above the bucket are chosen, the bucket's are listed and ranked
                for qq in range_constexpr(-(-(MKW + 1) // NT)):
                    i_ = tx + qq * NT
                    sm.mk.ptr[(i_ < MKW + 1).select(i_, I(MKW))] = I(0)
                sm.bc.ptr[16] = I(0)
                fx.gpu.barrier()
                chunk = I(0)
                if tie == I(0):
                    for i_, (e, x, ok) in enumerate(sweep()):
                        kk = key_of(x)
                        sure = ok & (take_all | (u(kk) > u(hi)))
                        cand = ok & ~take_all & (u(kk) <= u(hi)) & (u(kk) >= u(lo))
                        chunk = chunk | sure.select(I(1 << i_), I(0))
                        if cand:
                            slot = fx.atomic_add(sm.bc.ptr + 16, I(1))
                            if slot < I(CAPM):
                                sm.cc.ptr[slot] = (
                                    u(kk).to(fx.Uint64) << fx.Uint64(32)
                                ) | (fx.Uint64(4294967295) - I(e).to(fx.Uint64))
                if tie != I(0):
                    for i_, (e, x, ok) in enumerate(sweep()):
                        kk = key_of(x)
                        ibin = e >> IDSHIFT
                        sure = ok & ((u(kk) > u(hi)) | ((kk == hi) & (ibin < tbs)))
                        cand = ok & (kk == hi) & (ibin == tbs)
                        chunk = chunk | sure.select(I(1 << i_), I(0))
                        if cand:
                            slot = fx.atomic_add(sm.bc.ptr + 16, I(1))
                            if slot < I(CAPM):
                                sm.cc.ptr[slot] = (
                                    u(kk).to(fx.Uint64) << fx.Uint64(32)
                                ) | (fx.Uint64(4294967295) - I(e).to(fx.Uint64))
                fx.atomic_or(
                    sm.mk.ptr + ((tx * 4 * UN) >> 5), chunk << ((tx * 4 * UN) & 31)
                )
                fx.gpu.barrier()
                ncand = I(sm.bc.ptr[16])
                ncand = (ncand > I(CAPM)).select(I(CAPM), ncand)
                mine = (tx < ncand).select(tx, I(CAPM))
                myc = fx.Uint64(sm.cc.ptr[mine])
                myid = I((fx.Uint64(4294967295) - myc).to(I))
                for j0, st in range(
                    0,
                    I64((wave * 64 < ncand).select((ncand + 15) // 16, I(0))),
                    1,
                    init=[I(0)],
                ):
                    cj = [
                        fx.Uint64(sm.cc.ptr[I(j0) * 16 + c_])
                        for c_ in range_constexpr(16)
                    ]
                    r_ = st[0]
                    for c_ in range_constexpr(16):
                        r_ = r_ + (cj[c_] > myc).select(I(1), I(0))
                    rank = yield [r_]
                win = (tx < ncand) & (I(rank) < need)
                if win:
                    fx.atomic_or(sm.mk.ptr + (myid >> 5), I(1) << (myid & 31))
                if const_expr(newest and SEG):
                    first = pin & (tx == 0) & (pb >= base) & (pb < base + I(NB))
                    if first:
                        fx.atomic_or(
                            sm.mk.ptr + ((pb - base) >> 5), I(1) << ((pb - base) & 31)
                        )
                fx.gpu.barrier()

                # ---- the chosen ids ascending
                words = [
                    I(sm.mk.ptr[(tx * WPT + w < MKW).select(tx * WPT + w, I(MKW))])
                    for w in range_constexpr(WPT)
                ]
                cnt = I(0)
                for w in range_constexpr(WPT):
                    cnt = cnt + (tx * WPT + w < MKW).select(
                        I(fx.math.ctpop(words[w])), I(0)
                    )
                off, total = block_scan(cnt)
                run = off
                for w in range_constexpr(WPT):
                    bits = [
                        (((words[w] >> b) & 1) != I(0)) & (tx * WPT + w < MKW)
                        for b in range_constexpr(32)
                    ]
                    lids = [(tx * WPT + w) * 32 + b for b in range_constexpr(32)]
                    if const_expr(SEG):
                        xs = [
                            load(sr, base + lids[b], F, mask=bits[b])
                            for b in range_constexpr(32)
                        ]
                    else:
                        gids = [
                            load(idr, lids[b], mask=bits[b])
                            for b in range_constexpr(32)
                        ]
                    for b in range_constexpr(32):
                        if const_expr(SEG):
                            gid = base + lids[b]
                            at = seg * SLOTS + run
                            store(idr, at, gid, mask=bits[b])
                            xv = xs[b]
                            if const_expr(newest):
                                xv = (pin & (gid == pb)).select(F(INF), xv)
                            store(valr, at, xv, F, mask=bits[b])
                        else:
                            store(candr, run, gids[b], mask=bits[b])
                        run = run + bits[b].select(I(1), I(0))
                pad_slots(total)
            if nbl <= I(0):
                pad_slots(I(0))

    @flyc.jit
    def launch(
        src: fx.Int64,
        ids: fx.Int64,
        vals: fx.Int64,
        row_ends: fx.Int64,
        cand_out: fx.Int64,
        src_stride: fx.Int32,
        cmp_stride: fx.Int32,
        used: fx.Int32,
        rows: fx.Int32,
        segments: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(src, ids, vals, row_ends, cand_out, src_stride, cmp_stride, used).launch(
            grid=(fx.Int64(rows), fx.Int64(segments if SEG else 1), 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
