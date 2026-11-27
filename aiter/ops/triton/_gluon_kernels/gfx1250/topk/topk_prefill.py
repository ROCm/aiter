# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 DSA prefill top-k: sample a threshold, stream x > T into LDS lists, radix select; classify for short rows."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.standard import _sum_combine

VPT = gl.constexpr(8)  # values per lane and chunk
NE = gl.constexpr(16)  # list slots per lane when a warp's list is reduced
HB = gl.constexpr(11)  # histogram bits
_MAXKEY = gl.constexpr(0xFFFFFFFF)
_IDX_MASK = gl.constexpr((1 << 18) - 1)


@gluon.jit
def _to_key(x, MERGE_ZERO: gl.constexpr = True):
    """Order-preserving uint32 key; NaN -> 0, MERGE_ZERO: -0 -> +0."""
    bits = x.to(gl.uint32, bitcast=True)
    if MERGE_ZERO:
        bits = gl.where(bits == 0x80000000, 0, bits)
    key = gl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)
    return gl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, key)


@gluon.jit
def _wred(v, op: gl.constexpr):
    """Warp reduction (max, add or or) of one value per lane, broadcast to the warp."""
    W: gl.constexpr = v.shape[0] // 32
    L2: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [W, 1], [1, 0])
    v2 = gl.convert_layout(gl.reshape(v, [W, 32]), L2, assert_trivial=True)
    if op == "max":
        r = gl.max(v2, axis=1)
    elif op == "add":
        r = gl.sum(v2, axis=1)
    else:
        r = gl.reduce_or(v2, axis=1)
    r2 = gl.reshape(gl.zeros_like(v2) + r[:, None], [32 * W])
    return gl.convert_layout(r2, v.type.layout, assert_trivial=True)


@gluon.jit
def _wlevel(s, ex, lane, B: gl.constexpr):
    """One butterfly level: s -> sums of aligned 2B-lane blocks; ex += the lower half's sum in upper halves."""
    W: gl.constexpr = s.shape[0] // 32
    L4: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, 1], [1, 32 // (2 * B), 2, B], [W, 1, 1, 1], [3, 2, 1, 0]
    )
    s4 = gl.convert_layout(
        gl.reshape(s, [W, 32 // (2 * B), 2, B]), L4, assert_trivial=True
    )
    t = gl.reshape(gl.zeros_like(s4) + gl.sum(s4, axis=2)[:, :, None, :], [32 * W])
    t = gl.convert_layout(t, s.type.layout, assert_trivial=True)
    return t, gl.where((lane & B) != 0, ex + t - s, ex)


@gluon.jit
def _wscan(v):
    """Inclusive warp prefix sum of one value per lane (reductions only: tt.scan defeats the TDM wait counts)."""
    lane = gl.arange(0, v.shape[0], layout=v.type.layout) % 32
    s, ex = _wlevel(v, v * 0, lane, 1)
    s, ex = _wlevel(s, ex, lane, 2)
    s, ex = _wlevel(s, ex, lane, 4)
    s, ex = _wlevel(s, ex, lane, 8)
    s, ex = _wlevel(s, ex, lane, 16)
    return ex + v


@gluon.jit
def _chain(p, V: gl.constexpr, base=0):
    """Warp ranks of p != 0 in (value, lane) order from base; returns (ranks, count after the last value)."""
    LP: gl.constexpr = p.type.layout
    col = gl.arange(0, V, layout=gl.SliceLayout(0, LP))[None, :]
    sh = 8 * (col % 4)
    q = (p != 0).to(gl.int32) << sh
    lo = col < 4
    pa = gl.sum(gl.where(lo, q, 0), axis=1)  # 8-bit counts of values 0-3
    pb = gl.sum(gl.where(lo, 0, q), axis=1)  # and of values 4-7
    ta = _wred(pa, "add")
    tb = _wred(pb, "add")
    na = ((ta * 0x01010101) >> 24) & 0xFF
    pre = gl.where(lo, (ta * 0x01010100)[:, None], (tb * 0x01010100)[:, None])
    ex = gl.where(lo, (_wscan(pa) - pa)[:, None], (_wscan(pb) - pb)[:, None])
    b = gl.zeros(p.shape, gl.int32, LP) + base
    slot = b + ((pre >> sh) & 0xFF) + ((ex >> sh) & 0xFF) + gl.where(lo, 0, na[:, None])
    return slot, gl.max(b, axis=1) + na + (((tb * 0x01010101) >> 24) & 0xFF)


@gluon.jit
def _compact(x, thr, base, idx, lst, flag, wid, fval, CAPW: gl.constexpr):
    """Append x > thr as (x, idx) to the warp's LDS list in (value, lane) order; overflow max-es fval into flag."""
    m = x > thr
    slot, nb = _chain(m.to(gl.int32), x.shape[1], base)
    lane = gl.arange(0, nb.shape[0], layout=nb.type.layout) % 32
    ent = x.to(gl.uint32, bitcast=True).to(gl.int64) | (idx.to(gl.int64) << 32)
    junk = lst.shape[0] // 2 - 64 + lane[:, None]  # a lane's slot for values not kept
    dst = gl.where(m, wid * CAPW + gl.minimum(slot, CAPW - 1), junk)
    lst64 = lst.reinterpret(
        gl.int64, [lst.shape[0] // 2], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    lst64.scatter(gl.reshape(ent, [x.numel]), gl.reshape(dst, [x.numel]), 0)
    ovf = (lane == 0) & (nb > CAPW - 1)
    flag.atomic_scatter_max(gl.zeros_like(nb) + fval, lane * 0, 0, mask=ovf)
    return nb


@gluon.jit
def _hist_add(hist, digit, cand, first, hot=None, cp=None, lst=None):
    """hist[digit] += 1 where cand: one atomic per warp for bin hot, one per lane for its first value's bin."""
    digit = digit.to(gl.int32)
    if hot is not None:
        wh = _wred(gl.sum((cand & (digit == hot)).to(gl.int32), axis=1), "add")
        lane = gl.arange(0, wh.shape[0], layout=wh.type.layout) % 32
        hist.atomic_scatter_add(wh, lane * 0 + hot, 0, mask=(lane == 0) & (wh > 0))
        cand = cand & (digit != hot)
    d0 = gl.max(gl.where(first, digit, -1), axis=1)[:, None]
    n0 = gl.sum(((digit == d0) & cand).to(gl.int32), axis=1)[:, None]
    N: gl.constexpr = digit.numel
    val = gl.reshape(gl.where(first, n0, 1), [N])
    off = gl.where(first, d0, digit)
    msk = gl.where(first, n0 > 0, cand & (digit != d0))
    if cp is None:
        hist.atomic_scatter_add(val, gl.reshape(off, [N]), 0, mask=gl.reshape(msk, [N]))
    else:  # histogram copy cp: hist for 0, lst[(cp - 1) * NBIN:] for the others
        off1 = gl.reshape(off + gl.maximum(cp - 1, 0) * hist.shape[0], [N])
        hist.atomic_scatter_add(
            val, gl.reshape(off, [N]), 0, mask=gl.reshape(msk & (cp == 0), [N])
        )
        lst.atomic_scatter_add(val, off1, 0, mask=gl.reshape(msk & (cp > 0), [N]))


@gluon.jit
def _warp_exchange(scr, val, wid, LV: gl.constexpr, W: gl.constexpr):
    """Warp-uniform val -> (exclusive prefix over warps, total); one barrier, after all reads of scr."""
    scr.scatter(val, gl.convert_layout(wid, LV), 0)
    gl.barrier()
    WL: gl.constexpr = gl.BlockedLayout([1, W], [32, 1], [W, 1], [1, 0])
    cols = gl.arange(0, W, layout=gl.SliceLayout(0, WL))
    gi = gl.zeros([32 * W, W], gl.int32, WL) + cols[None, :]
    cw = gl.reshape(scr.gather(gl.reshape(gi, [32 * W * W]), 0), [32 * W, W])
    myw = gl.convert_layout(wid, gl.SliceLayout(1, WL))
    pre = gl.sum(gl.where(cols[None, :] < myw[:, None], cw, 0), axis=1)
    return gl.convert_layout(pre, LV), gl.sum(
        scr.load(gl.BlockedLayout([W], [32], [W], [0])), axis=0
    )


@gluon.jit
def _pick(hv, need, scr, scr2, W: gl.constexpr, copies=None):
    """Bin b holding the need-th largest entry -> (b, #above b, #in b); zeroes the histogram, two barriers."""
    ROWS: gl.constexpr = 32 * W
    BPT: gl.constexpr = hv.shape[1]
    HL: gl.constexpr = gl.BlockedLayout([1, BPT], [32, 1], [W, 1], [1, 0])
    LV: gl.constexpr = gl.SliceLayout(1, HL)
    h = hv.load(HL)
    if copies is not None:  # more copies of the histogram (added up; left as they are)
        L1: gl.constexpr = gl.BlockedLayout([BPT], [32], [W], [0])
        for i in gl.static_range(len(copies)):
            h = h + gl.convert_layout(
                gl.reshape(copies[i].load(L1), [ROWS, BPT]), HL, assert_trivial=True
            )
    hv.store(gl.zeros([ROWS, BPT], gl.int32, HL))
    t = gl.arange(0, ROWS, layout=LV)
    c = gl.arange(0, BPT, layout=gl.SliceLayout(0, HL))
    tot = gl.sum(h, axis=1)
    wtot = _wred(tot, "add")
    sinc = wtot - _wscan(tot) + tot  # inclusive warp suffix sums
    wpre, call = _warp_exchange(scr, wtot, t // 32, LV, W)
    after = call - wpre - wtot + sinc - tot  # entries in the bins of later threads
    incl = (
        gl.zeros_like(h) + after[:, None]
    )  # + suffix sums of the thread's bins (no tt.scan)
    for k in gl.static_range(BPT):
        incl += gl.where(
            c[None, :] <= k, gl.sum(gl.where(c[None, :] == k, h, 0), axis=1)[:, None], 0
        )
    hit = (incl - h < need) & (incl >= need)
    hb = gl.max(gl.where(hit, t[:, None] * BPT + c[None, :], -1), axis=1)
    scr2.scatter(
        hb | (gl.sum(gl.where(hit, incl - h, 0), axis=1) << 12),
        gl.where(hb >= 0, 0, 2),
        0,
    )
    scr2.scatter(gl.sum(gl.where(hit, h, 0), axis=1), gl.where(hb >= 0, 1, 3), 0)
    gl.barrier()
    LW1: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    v = gl.max(scr2.slice(0, 1).load(LW1), axis=0)
    return v & 4095, v >> 12, gl.max(scr2.slice(1, 1).load(LW1), axis=0)


@gluon.jit
def _list_vals(lst, lg, h, CAPW: gl.constexpr):
    """Values (bits), validity and slots of list half h; lg = (list lengths, e, lane, warp)."""
    lc, eh, lane, wr = lg
    lj = (h * (NE // 2) + eh)[None, :] * 32 + lane[:, None]
    slot = wr[:, None] * CAPW + gl.minimum(lj, CAPW - 1)
    xb = gl.reshape(lst.gather(gl.reshape(2 * slot, [slot.numel]), 0), slot.shape)
    return xb.to(gl.uint32, bitcast=True), lj < lc[:, None], slot


@gluon.jit
def _key_range(kk, lo, hi, scr, scr4, W: gl.constexpr):
    """(min, max) of the list keys in [lo, hi]. One barrier; scr, scr4 must be free."""
    LKV: gl.constexpr = gl.SliceLayout(1, kk[0][0].type.layout)
    kmax = gl.zeros([kk[0][0].shape[0]], gl.uint32, LKV)
    kmin = kmax
    for h in gl.static_range(2):
        key, sel, _ = kk[h]
        m = sel & (key >= lo) & (key <= hi)
        kmax = gl.maximum(kmax, gl.max(gl.where(m, key, 0), axis=1))
        kmin = gl.maximum(kmin, gl.max(gl.where(m, ~key, 0), axis=1))
    wr = gl.arange(0, kmax.shape[0], layout=kmax.type.layout) // 32
    scr.scatter(_wred(kmax, "max").to(gl.int32, bitcast=True), wr, 0)
    scr4.scatter(_wred(kmin, "max").to(gl.int32, bitcast=True), wr, 0)
    gl.barrier()
    return (
        ~gl.max(
            scr4.load(gl.BlockedLayout([W], [32], [W], [0])).to(
                gl.uint32, bitcast=True
            ),
            axis=0,
        ),
        gl.max(
            scr.load(gl.BlockedLayout([W], [32], [W], [0])).to(gl.uint32, bitcast=True),
            axis=0,
        ),
    )


@gluon.jit
def _finite_max(kk, kinf, scr, scr4, W: gl.constexpr):
    """(largest list key below kinf, #keys == kinf). Two barriers; scr, scr4 must be free."""
    LKV: gl.constexpr = gl.SliceLayout(1, kk[0][0].type.layout)
    kmax = gl.zeros([kk[0][0].shape[0]], gl.uint32, LKV)
    n = gl.zeros([kk[0][0].shape[0]], gl.uint32, LKV)
    for h in gl.static_range(2):
        key, sel, _ = kk[h]
        kmax = gl.maximum(kmax, gl.max(gl.where(sel & (key < kinf), key, 0), axis=1))
        n = n + gl.sum((sel & (key == kinf)).to(gl.uint32), axis=1)
    wr = gl.arange(0, kmax.shape[0], layout=kmax.type.layout) // 32
    scr.scatter(_wred(kmax, "max").to(gl.int32, bitcast=True), wr, 0)
    scr4.scatter(_wred(n, "add").to(gl.int32, bitcast=True), wr, 0)
    gl.barrier()
    kmax = gl.max(
        scr.load(gl.BlockedLayout([W], [32], [W], [0])).to(gl.uint32, bitcast=True),
        axis=0,
    )
    n = gl.sum(scr4.load(gl.BlockedLayout([W], [32], [W], [0])), axis=0)
    gl.barrier()  # every warp has read scr, scr4
    return kmax, n


@gluon.jit
def _select(
    sm, kk, lfirst, lo, hi, need, maxkeep, NARROW: gl.constexpr, W: gl.constexpr
):
    """Threshold t for the need largest keys in [lo, hi] -> (t, #keys > t); stops at whole bins of <= maxkeep."""
    _r, _l, hist, hist2d, scr, scr2, _s3, scr4, _f = sm
    above = need * 0
    t = lo
    go = lo < hi
    for lvl in gl.static_range(3):
        if go:
            nbits = ((hi - lo).to(gl.float32).to(gl.uint32, bitcast=True) >> 23) - 126
            s = gl.maximum(nbits, HB) - HB
            for h in gl.static_range(2):
                key, sel, _ = kk[h]
                _hist_add(
                    hist, (key - lo) >> s, sel & (key >= lo) & (key <= hi), lfirst
                )
            gl.barrier()
            b, ab, nb = _pick(hist2d, need, scr, scr2, W)
            lo = lo + (b.to(gl.uint32) << s)
            hi = gl.minimum(hi, lo + ((1 << s) - 1))
            if above + ab + nb <= maxkeep:  # keep the whole bin
                above = above + ab + nb
                t = lo - 1
                go = lo < 0
            else:
                above = above + ab
                need = need - ab
                t = lo
                go = lo < hi
                # a bin of one key or of many (ties): its key range (one key: the search ends)
                if (lvl < 2) & go & ((nb > 128) | (nb == 1)):
                    klo, khi = _key_range(kk, lo, hi, scr, scr4, W)
                    if (klo == khi) | NARROW:
                        lo = klo
                        hi = khi
                        t = lo
                        go = lo < hi
    return t, above


@gluon.jit
def _list_reduce(
    sm,
    cnt,
    xm,
    coarse,
    tkey,
    out_row,
    OUT: gl.constexpr,
    MAXKEEP: gl.constexpr,
    K: gl.constexpr,
    W: gl.constexpr,
    CAPW: gl.constexpr,
):
    """Keep the K (up to MAXKEEP) largest list entries: OUT writes indices, else compacts in place; -> (n, t)."""
    lst, scr, scr3, scr4 = sm[1], sm[4], sm[6], sm[7]
    ROWS: gl.constexpr = 32 * W
    NH: gl.constexpr = NE // 2
    LR: gl.constexpr = gl.BlockedLayout([1, NH], [32, 1], [W, 1], [1, 0])
    LRV: gl.constexpr = gl.SliceLayout(1, LR)
    LV: gl.constexpr = cnt.type.layout
    rr = gl.arange(0, ROWS, layout=LRV)
    eh = gl.arange(0, NH, layout=gl.SliceLayout(0, LR))
    lane = rr % 32
    wr = rr // 32
    lfirst = (eh == 0)[None, :] & (rr >= 0)[:, None]
    lc = gl.convert_layout(cnt, LRV)
    lg = (lc, eh, lane, wr)
    # the list-length exchange also publishes the max streamed value and (coarse rows) the OR of magnitude bits
    scr3.scatter(
        _wred(_to_key(xm), "max").to(gl.int32, bitcast=True),
        gl.convert_layout(wr, LV),
        0,
    )
    if coarse:
        xo = gl.zeros([ROWS], gl.uint32, LRV)
        for h in gl.static_range(2):
            xb, sel, _ = _list_vals(lst, lg, h, CAPW)
            xo = xo | gl.reduce_or(gl.where(sel, xb & 0x7FFFFFFF, 0), axis=1)
        scr4.scatter(
            _wred(xo, "or").to(gl.int32, bitcast=True), gl.convert_layout(wr, LV), 0
        )
    _pre, total = _warp_exchange(scr, lc, wr, LRV, W)
    hi = gl.max(
        scr3.load(gl.BlockedLayout([W], [32], [W], [0])).to(gl.uint32, bitcast=True),
        axis=0,
    )
    tz = (0 * total).to(gl.uint32)
    if coarse:  # all values have tz trailing zero bits: key >> tz keeps them apart
        o = gl.reduce_or(
            scr4.load(gl.BlockedLayout([W], [32], [W], [0])).to(
                gl.uint32, bitcast=True
            ),
            axis=0,
        )
        o = o | (1 << 20)
        tz = ((o & (0 - o)).to(gl.float32).to(gl.uint32, bitcast=True) >> 23) - 127
    # the keys stay in registers for all passes
    xb, sel, slot = _list_vals(lst, lg, 0, CAPW)
    kk0 = (gl.where(sel, _to_key(xb.to(gl.float32, bitcast=True)) >> tz, 0), sel, slot)
    xb, sel, slot = _list_vals(lst, lg, 1, CAPW)
    kk1 = (gl.where(sel, _to_key(xb.to(gl.float32, bitcast=True)) >> tz, 0), sel, slot)
    kk = (kk0, kk1)
    keepall = total <= K
    if keepall:
        t = tkey
        above = total
        need_eq = 0 * total
    else:
        gl.barrier()  # every warp has read scr
        top = hi >> tz
        ninf = 0 * total
        if OUT & (hi == 0xFF800000):
            # +inf keys: the select spans the finite keys only, the +inf keys are above its threshold
            top, ninf = _finite_max(kk, top, scr, scr4, W)
        if ninf >= K:
            t = hi >> tz
            above = 0 * total
        else:
            t, above = _select(
                sm, kk, lfirst, tkey >> tz, top, K - ninf, MAXKEEP - ninf, OUT, W
            )
            above = above + ninf
        need_eq = gl.maximum(K - above, 0)
    # warp counts of the keys > t and == t, then ranks in list order
    ct = gl.zeros([ROWS], gl.int32, LRV)
    for h in gl.static_range(2):
        key, sel, _ = kk[h]
        gt = sel & ((key > t) | keepall)
        eq = sel & (key == t) & (keepall == 0)
        ct += gl.sum(gt.to(gl.int32) + (eq.to(gl.int32) << 13), axis=1)
    cw = _wred(ct, "add")
    gl.barrier()  # every warp has read scr
    wp, _t = _warp_exchange(scr, cw, wr, LRV, W)
    gb = (wp & 0x1FFF)[:, None]
    eb = (wp >> 13)[:, None]
    rg = gb
    re = eb
    lst64 = lst.reinterpret(
        gl.int64, [lst.shape[0] // 2], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    for h in gl.static_range(2):
        key, sel, slot = kk[h]
        gt = sel & ((key > t) | keepall)
        eq = sel & (key == t) & (keepall == 0)
        sg, ng = _chain(gt.to(gl.int32), NH, rg)
        se, nq = _chain(eq.to(gl.int32), NH, re)
        rg = ng[:, None]
        re = nq[:, None]
        keep = gt | (eq & (se < need_eq))
        if OUT:
            lidx = gl.reshape(
                lst.gather(gl.reshape(2 * slot + 1, [ROWS * NH]), 0), [ROWS, NH]
            )
            gl.store(out_row + sg + gl.minimum(se, need_eq), lidx, mask=keep)
        else:  # in place: a warp reads these slots before it writes any at or below them
            ent = gl.reshape(lst64.gather(gl.reshape(slot, [ROWS * NH]), 0), [ROWS, NH])
            dst = (sg - gb) + gl.minimum(se, need_eq) - gl.minimum(eb, need_eq)
            dst = gl.where(
                keep, wr[:, None] * CAPW + dst, lst.shape[0] // 2 - 64 + lane[:, None]
            )
            lst64.scatter(gl.reshape(ent, [ROWS * NH]), gl.reshape(dst, [ROWS * NH]), 0)
    if OUT:
        res = gl.minimum(total, above + need_eq)
    else:
        nw = (cw & 0x1FFF) + gl.minimum((wp >> 13) + (cw >> 13), need_eq)
        res = gl.convert_layout(nw - gl.minimum(wp >> 13, need_eq), LV)
    return res, t


@gluon.jit
def _load_chunk(ring, ch, L, rb, LX: gl.constexpr):
    """Chunk ch from its ring slot (NaN past the row end) and its indices; warps interleave every 64 values."""
    ROWS: gl.constexpr = rb.shape[0]
    W: gl.constexpr = ROWS // 32
    CH: gl.constexpr = ring.shape[1]
    L4: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, VPT], [4, 1, 8, 1], [1, W, 1, 1], [3, 2, 0, 1]
    )
    x4 = ring.index(ch % ring.shape[0]).reshape([4, W, 8, VPT]).load(L4)
    x = gl.convert_layout(
        gl.reshape(gl.permute(x4, [1, 0, 2, 3]), [ROWS, VPT]), LX, assert_trivial=True
    )
    idx = (ch * CH + rb) + gl.arange(0, VPT, layout=gl.SliceLayout(0, LX))[None, :]
    if (ch + 1) * CH > L:
        x = gl.where(idx < L, x, float("nan"))
    return x, idx


@gluon.jit
def _prune_redo(
    ctx,
    cnt,
    prev,
    xm,
    tkey,
    ch,
    LX: gl.constexpr,
    K: gl.constexpr,
    W: gl.constexpr,
    CAPW: gl.constexpr,
    SLACK: gl.constexpr,
):
    """A list overflowed on chunk ch: prune the lists (T rises), the overflowed warps redo ch from global memory."""
    seg_ptr, sm, wid, rb, L, out_row = ctx
    ovw = cnt > CAPW - 1
    cnt, tkey = _list_reduce(
        sm,
        gl.where(ovw, prev, cnt),
        xm,
        False,
        tkey,
        out_row,
        False,
        K + SLACK,
        K,
        W,
        CAPW,
    )
    idx = (ch * (32 * W * VPT) + rb) + gl.arange(0, VPT, layout=gl.SliceLayout(0, LX))[
        None, :
    ]
    x = gl.load(seg_ptr + idx, mask=ovw[:, None] & (idx < L), other=float("nan"))
    return (
        _compact(
            x,
            gl.where(tkey >= 0x80000000, tkey ^ 0x80000000, ~tkey).to(
                gl.float32, bitcast=True
            ),
            cnt[:, None],
            idx,
            sm[1],
            sm[8],
            wid[:, None],
            1 << 30,
            CAPW,
        ),
        tkey,
    )


@gluon.jit
def _stream(
    desc,
    seg_ptr,
    sm,
    NCH,
    L,
    rb,
    wid,
    tkey,
    out_row,
    LX: gl.constexpr,
    K: gl.constexpr,
    W: gl.constexpr,
    CAPW: gl.constexpr,
    SLACK: gl.constexpr,
):
    """Append the row's x > float(tkey) to the warp lists -> (list length, max value, tkey, lossy redo)."""
    ring, lst, _h, _h2, _s, _s2, _s3, _s4, flag = sm
    NBUF: gl.constexpr = ring.shape[0]
    CH: gl.constexpr = ring.shape[1]
    LV: gl.constexpr = gl.SliceLayout(1, LX)
    ctx = (seg_ptr, sm, wid, rb, L, out_row)
    tt = gl.where(tkey >= 0x80000000, tkey ^ 0x80000000, ~tkey).to(
        gl.float32, bitcast=True
    )
    cnt = gl.zeros([32 * W], gl.int32, LV)
    prev = cnt
    xm = gl.full([32 * W], float("-inf"), gl.float32, LV)
    flag.store(gl.full([1], -2, gl.int32, gl.BlockedLayout([1], [32], [W], [0])))
    for ch in range(NCH):
        gl.amd.gfx1250.tdm.async_wait(NBUF - 2)
        gl.barrier()
        nxt = ch + NBUF - 1
        gl.amd.gfx1250.tdm.async_load(
            desc, [nxt * CH], ring.index(nxt % NBUF), pred=nxt < NCH
        )
        fl = gl.max(
            flag.load(gl.BlockedLayout([1], [32], [W], [0])), axis=0
        )  # read with the chunk (one LDS round trip)
        x, idx = _load_chunk(ring, ch, L, rb, LX)
        if fl == ch - 1:  # a list overflowed on the previous chunk
            cnt, tkey = _prune_redo(
                ctx, cnt, prev, xm, tkey, ch - 1, LX, K, W, CAPW, SLACK
            )
            tt = gl.where(tkey >= 0x80000000, tkey ^ 0x80000000, ~tkey).to(
                gl.float32, bitcast=True
            )
            # reloaded: not kept live across the prune
            x, idx = _load_chunk(ring, ch, L, rb, LX)
        prev = cnt
        cnt = _compact(x, tt, cnt[:, None], idx, lst, flag, wid[:, None], ch, CAPW)
        xm = gl.maximum(xm, gl.max(x, axis=1))
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    if gl.max(flag.load(gl.BlockedLayout([1], [32], [W], [0])), axis=0) == NCH - 1:
        cnt, tkey = _prune_redo(
            ctx, cnt, prev, xm, tkey, NCH - 1, LX, K, W, CAPW, SLACK
        )
        gl.barrier()
    return (
        cnt,
        xm,
        tkey,
        gl.max(flag.load(gl.BlockedLayout([1], [32], [W], [0])), axis=0) >= (1 << 30),
    )


@gluon.jit
def _topk_prefill_kernel(
    logits_ptr,
    ks_ptr,
    ke_ptr,
    out_ptr,
    stride0,
    out_stride0,
    K: gl.constexpr,
    NBUF: gl.constexpr,
    CAPW: gl.constexpr,
    TGT_NUM: gl.constexpr,
    RMIN: gl.constexpr,
    SLACK: gl.constexpr,
    FEW_MAX: gl.constexpr,
    RETRY_MAX: gl.constexpr,
    RING_ROW: gl.constexpr,
    CL_CAP: gl.constexpr,
    CL_SAMPLE_MIN: gl.constexpr,
    CL_SAMPLE_BLOCKS: gl.constexpr,
    CL_HI: gl.constexpr,
    CL_LO: gl.constexpr,
    ZERO_AGG: gl.constexpr,
    num_warps: gl.constexpr,
):
    W: gl.constexpr = num_warps
    ROWS: gl.constexpr = 32 * W
    CH: gl.constexpr = ROWS * VPT
    NBIN: gl.constexpr = 1 << HB
    S1: gl.constexpr = 32 - HB  # radix levels: HB / HB / 32 - 2 HB bits
    S2: gl.constexpr = 32 - 2 * HB
    gl.static_assert(32 * NE >= CAPW)
    gl.static_assert(CAPW - 32 >= (K + SLACK + W - 1) // W)
    gl.static_assert(2 * W * CAPW + 5 * W + 8 <= 8192)
    LX: gl.constexpr = gl.BlockedLayout([1, VPT], [32, 1], [W, 1], [1, 0])
    LV: gl.constexpr = gl.SliceLayout(1, LX)
    LS: gl.constexpr = gl.BlockedLayout([1, 8], [32, 1], [W, 1], [1, 0])
    OL: gl.constexpr = gl.BlockedLayout([max(K // ROWS, 1)], [32], [W], [0])
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    row = gl.program_id(0)
    ks = gl.load(ks_ptr + row)
    L = gl.maximum(gl.load(ke_ptr + row), ks) - ks
    out_row = out_ptr + row.to(gl.int64) * out_stride0

    if L <= K:
        i = gl.arange(0, K, layout=OL)
        gl.store(out_row + i, gl.where(i < L, i, -1))
    else:
        seg_ptr = logits_ptr + row.to(gl.int64) * stride0 + ks
        # LDS: ring 32 KB, lists + scratch 32 KB, histogram 8 KB (the slot after the ring; classify's with the last)
        gl.static_assert(NBIN == CH)
        ring_h = gl.allocate_shared_memory(gl.float32, [NBUF + 1, CH], SL)
        ring = ring_h.slice(0, NBUF)
        lst = gl.allocate_shared_memory(gl.int32, [8192], SL)
        hist = ring_h.index(NBUF).reinterpret(gl.int32, [NBIN], SL)
        hist.store(
            gl.zeros([NBIN], gl.int32, gl.BlockedLayout([NBIN // ROWS], [32], [W], [0]))
        )
        hist2d = hist.reshape([ROWS, NBIN // ROWS])
        scr = lst.slice(8192 - 4 * W - 8, W)  # warp totals
        scr3 = lst.slice(8192 - 3 * W - 8, W)  # tie-scan warp totals, list key bound
        scr2 = lst.slice(8192 - 2 * W - 8, 4)  # pick broadcast
        flag = lst.slice(8192 - 2 * W - 4, 1)  # overflow flag
        scr4 = lst.slice(8192 - 5 * W - 8, W)  # bit ORs
        sm = (ring, lst, hist, hist2d, scr, scr2, scr3, scr4, flag)

        desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(seg_ptr, [L], [1], [CH], SL)
        NCH = gl.cdiv(L, CH)

        r = gl.arange(0, ROWS, layout=LV)
        rb = ((r % 32) // 8 * (64 * W) + (r // 32) * 64 + (r % 8) * 8)[:, None]
        wid = r // 32

        # ---- 1. sample: 11 + 11-bit radix select on 2W blocks of 128 -> T ----
        rs = gl.arange(0, ROWS, layout=gl.SliceLayout(1, LS))
        cs = gl.arange(0, 8, layout=gl.SliceLayout(0, LS))
        sfirst = (cs == 0)[None, :] & (rs >= 0)[:, None]
        NBK: gl.constexpr = ROWS // 16
        start = (((rs // 16) * (L - 128)) // (NBK - 1)) & ~3
        sx = gl.load(seg_ptr + (start + (rs % 16) * 8)[:, None] + cs[None, :])
        for i in gl.static_range(ring.shape[0] - 1):
            gl.amd.gfx1250.tdm.async_load(
                desc, [i * ring.shape[1]], ring.index(i), pred=i < NCH
            )
        ZL: gl.constexpr = gl.BlockedLayout([NBIN // ROWS], [32], [W], [0])
        for i in gl.static_range(3):
            lst.slice(i * NBIN, NBIN).store(gl.zeros([NBIN], gl.int32, ZL))
        skey = _to_key(sx)
        # per warp: OR of sample bits 0-7; FEW_MAX: bit 29 a non-NaN sample, bits 8-15 lanes of a repeated max
        bits = gl.reduce_or(sx.to(gl.uint32, bitcast=True) & 0xFF, axis=1)
        swid = rs // 32
        if FEW_MAX > 0:
            mx = gl.max(sx, axis=1)
            rep = gl.sum((sx == mx[:, None]).to(gl.int32), axis=1)
            npair = _wred(((rep >= 2) & (mx != 0)).to(gl.int32), "add")
            bits = bits | gl.where(gl.max(skey, axis=1) > 0, 1 << 29, 0).to(gl.uint32)
            bits = bits | (npair.to(gl.uint32) << 8)
        scr4.scatter(_wred(bits, "or").to(gl.int32, bitcast=True), swid, 0)
        gl.barrier()
        wbits = scr4.load(gl.BlockedLayout([W], [32], [W], [0])).to(
            gl.uint32, bitcast=True
        )
        sbits = gl.reduce_or(wbits, axis=0)
        coarse = (sbits & 0xFF) == 0  # bf16-like values
        # rows of only NaN samples or few distinct values (8+ such lanes) take the classify algorithm
        npair = gl.sum((wbits >> 8) & 0xFF, axis=0)
        few = (L <= FEW_MAX) & ((npair >= 8) | ((sbits >> 29) & 1 == 0))
        if few:
            tk = (0 * L).to(gl.uint32)
        else:
            # level 1: lanes l % 4 > 0 add to histogram copies in the (empty) lists; the pick adds them up
            cp = (
                gl.convert_layout(rs % 4, gl.SliceLayout(1, LS))[:, None]
                + 0 * cs[None, :]
            )
            _hist_add(
                hist,
                skey >> S1,
                sfirst | (cs >= 0)[None, :],
                sfirst,
                1 << (HB - 1),
                cp,
                lst,
            )
            gl.barrier()
            NS: gl.constexpr = NBK * 128
            rank = gl.minimum(gl.maximum((TGT_NUM * K * NS) // (16 * L), RMIN), NS)
            cps = (
                lst.slice(0, NBIN),
                lst.slice(NBIN, NBIN),
                lst.slice(2 * NBIN, NBIN),
            )
            b1, a1, _n1 = _pick(hist2d, rank, scr, scr2, W, cps)
            b1u = b1.to(gl.uint32)
            tlo = b1u << S1  # lowest key of the selected bin
            # shorter rows do better with the coarse threshold (no second level)
            if L > 16384:
                hot2 = gl.where(b1u == (1 << (HB - 1)), 0, -1)
                _hist_add(
                    hist, (skey >> S2) & (NBIN - 1), (skey >> S1) == b1u, sfirst, hot2
                )
                gl.barrier()
                b2, _a2, _n2 = _pick(hist2d, rank - a1, scr, scr2, W)
                tlo = tlo | (b2.to(gl.uint32) << S2)
            # x > T keeps the selected bin (T is NaN on all-NaN samples)
            tk = gl.where(tlo == 0, 0, tlo - 1)
        gl.barrier()

        # ---- 2. stream, 3. resolve, or the classify algorithm ----
        fb = few | (tk == 0)  # tk == 0: all samples NaN, the list would stay empty
        att = 0 * L
        go = fb == 0
        while go:
            if att > 0:
                for i in gl.static_range(ring.shape[0] - 1):
                    gl.amd.gfx1250.tdm.async_load(
                        desc, [i * ring.shape[1]], ring.index(i), pred=i < NCH
                    )
            cnt, xm, tkey, lossy = _stream(
                desc,
                seg_ptr,
                sm,
                NCH,
                L,
                rb,
                wid,
                tk,
                out_row,
                LX,
                K,
                W,
                CAPW,
                SLACK,
            )
            n, tkey = _list_reduce(
                sm,
                cnt,
                xm,
                coarse,
                tkey,
                out_row,
                True,
                K,
                K,
                W,
                CAPW,
            )
            fb = lossy
            go = L < 0
            # the first K - n x == T complete the row (tied rows)
            if (n < K) & (lossy == 0):
                tneed = K - n
                tt = gl.zeros([ROWS, VPT], gl.float32, LX) + gl.where(
                    tkey >= 0x80000000, tkey ^ 0x80000000, ~tkey
                ).to(gl.float32, bitcast=True)
                for i in gl.static_range(ring.shape[0] - 1):
                    gl.amd.gfx1250.tdm.async_load(
                        desc, [i * ring.shape[1]], ring.index(i), pred=i < NCH
                    )
                found = 0 * L
                ch = 0 * L
                while (ch < NCH) & (found < tneed):
                    gl.amd.gfx1250.tdm.async_wait(NBUF - 2)
                    gl.barrier()
                    nxt = ch + NBUF - 1
                    gl.amd.gfx1250.tdm.async_load(
                        desc, [nxt * CH], ring.index(nxt % NBUF), pred=nxt < NCH
                    )
                    x, idx = _load_chunk(ring, ch, L, rb, LX)
                    p = x == tt
                    slot, wt = _chain(p.to(gl.int32), VPT)
                    wp, tot = _warp_exchange(scr3, wt, wid, LV, W)
                    g = found + wp[:, None] + slot
                    gl.store(out_row + n + g, idx, mask=p & (g < tneed))
                    found = found + tot
                    ch = ch + 1
                gl.amd.gfx1250.tdm.async_wait(0)
                gl.barrier()
                fb = found < tneed
                # the sample missed low: shorter rows retry from T = -inf, longer ones take classify
                go = fb & (att == 0) & (L <= RETRY_MAX)
                tk = (0 * L).to(gl.uint32) + 0x007FFFFF  # -inf
            att = att + 1
        if fb:
            # RING_ROW (widths <= NBUF - 1 chunks): classify reads the row from the ring
            if RING_ROW:
                gl.amd.gfx1250.tdm.async_wait(0)
                src = (
                    seg_ptr - ks,
                    ring_h.slice(0, 4).reinterpret(gl.float32, [4 * CH], SL),
                    NCH < NBUF,
                )
            else:
                src = (seg_ptr - ks, None, False)
            gl.barrier()
            _classify_row(
                src,
                ks,
                L,
                out_row,
                ring_h.slice(NBUF - 1, 2).reinterpret(gl.int32, [2 * CH], SL),
                lst,
                K,
                VPT,
                gl.cdiv(L, CH),
                CL_CAP,
                8192,
                CL_SAMPLE_MIN,
                CL_SAMPLE_BLOCKS,
                CL_HI,
                CL_LO,
                W,
                False,
                ZERO_AGG,
            )
            gl.amd.gfx1250.tdm.async_wait(0)  # a routed row's prologue


@gluon.jit
def _gload(row_ptr, ks, L, base, pos, CONTIG: gl.constexpr):
    idx = base + pos
    valid = idx < L
    if CONTIG:
        ptr = row_ptr + gl.multiple_of(ks, 4)
        off = gl.max_contiguous(gl.multiple_of(idx, [1, 4]), [1, 4])
        if base + pos.numel <= L:
            x = gl.load(ptr + off)
        else:
            x = gl.load(ptr + off, mask=valid, other=0.0)
    else:
        x = gl.load(row_ptr + (ks + idx).to(gl.int64), mask=valid, other=0.0)
    return x


@gluon.jit
def _load_keys(src, ks, L, base, pos, CONTIG: gl.constexpr = False):
    """Keys of row[base + pos] (0 past the end), validity; src = (row pointer, smem row or None, in smem)."""
    row_ptr, rsm, in_smem = src
    idx = base + pos
    valid = idx < L
    if rsm is not None:
        if in_smem:
            x = gl.reshape(
                rsm.gather(gl.reshape(gl.where(valid, idx, 0), [idx.numel]), 0),
                idx.shape,
            )
        else:
            x = _gload(row_ptr, ks, L, base, pos, CONTIG)
    else:
        x = _gload(row_ptr, ks, L, base, pos, CONTIG)
    return gl.where(valid, _to_key(x, False), 0), valid


@gluon.jit
def _hist_scatter(
    hist, key, cand, first, SHIFT: gl.constexpr, NB: gl.constexpr, ZAGG: gl.constexpr
):
    """hist[digit] += 1 where cand; one atomic per lane for its first slot's bin (ZAGG: the +0 bin)."""
    digit = ((key >> SHIFT) & (NB - 1)).to(gl.int32)
    pick = gl.where(first, digit, -1)
    if ZAGG:
        ZB: gl.constexpr = (0x80000000 >> SHIFT) & (NB - 1)
        pick = gl.where(cand & (digit == ZB), NB, pick)
    d0 = gl.max(pick, axis=1)[:, None]
    if ZAGG:
        d0 = gl.where(d0 == NB, ZB, d0)
    same = (digit == d0) & cand
    n0 = gl.sum(same.to(gl.int32), axis=1)[:, None]
    if ZAGG:  # the lane's first slot in the bin carries the count
        col = gl.arange(0, key.shape[1], layout=gl.SliceLayout(0, key.type.layout))[
            None, :
        ]
        agg = col == gl.min(gl.where(same, col, key.shape[1]), axis=1)[:, None]
        val = gl.where(agg, n0, 1)
        msk = gl.where(same, agg, cand)
        off = digit
    else:
        val = gl.where(first, n0, 1)
        msk = gl.where(first, n0 > 0, cand & (digit != d0))
        off = gl.where(first, d0, digit)
    hist.atomic_scatter_add(
        gl.reshape(val, [val.numel]),
        gl.reshape(off, [off.numel]),
        0,
        mask=gl.reshape(msk, [msk.numel]),
    )


@gluon.jit
def _pick_bin(hist, need, NB: gl.constexpr, HL: gl.constexpr):
    h = hist.load(HL)
    bins = gl.arange(0, NB, layout=HL)
    incl = gl.associative_scan(h, 0, _sum_combine, reverse=True)
    hit = (incl - h < need) & (incl >= need)
    b = gl.max(gl.where(hit, bins, -1), axis=0)
    above = gl.sum(gl.where(bins > b, h, 0), axis=0)
    inbin = gl.sum(gl.where(bins == b, h, 0), axis=0)
    return b, above, inbin


@gluon.jit
def _ranks(packed, L2D: gl.constexpr, LANES: gl.constexpr):
    lane_incl = gl.associative_scan(packed, 1, _sum_combine)
    lane_tot = gl.sum(packed, axis=1)
    tot_b = gl.convert_layout(lane_tot, LANES)
    lane_excl = gl.convert_layout(
        gl.associative_scan(tot_b, 0, _sum_combine) - tot_b, gl.SliceLayout(1, L2D)
    )
    return lane_excl[:, None] + lane_incl - packed, gl.sum(lane_tot, axis=0), lane_tot


@gluon.jit
def _select_out(
    out_row,
    key,
    idx,
    sel,
    t,
    n_gt,
    need,
    base_slot,
    gt_base,
    eq_base,
    L2D: gl.constexpr,
    LANES: gl.constexpr,
):
    gt = sel & (key > t)
    eq = sel & (key == t)
    pre, tot, _ = _ranks(gt.to(gl.int32) + (eq.to(gl.int32) << 12), L2D, LANES)
    eq_rank = eq_base + (pre >> 12)
    slot = base_slot + gl.where(gt, gt_base + (pre & 0xFFF), n_gt + eq_rank)
    gl.store(out_row + slot, idx, mask=gt | (eq & (eq_rank < need)))
    return gt_base + (tot & 0xFFF), eq_base + (tot >> 12)


@gluon.jit
def _classify(
    key_reg,
    buf,
    row_ptr,
    ks,
    L,
    pos,
    lo,
    hi,
    nkey,
    nvalid,
    K: gl.constexpr,
    CAP: gl.constexpr,
    BUF: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_CHUNKS,
    ONE_TILE: gl.constexpr,
    L2D: gl.constexpr,
    LANES: gl.constexpr,
    TRACK_TIES: gl.constexpr,
):
    """Winners / candidates of digit range [lo, hi] into buf -> (#winners, #candidates, all candidates tied)."""
    kmin = gl.full([pos.shape[0]], _MAXKEY, gl.uint32, gl.SliceLayout(1, L2D))
    kmax = gl.zeros([pos.shape[0]], gl.uint32, gl.SliceLayout(1, L2D))
    nwin = gl.zeros([pos.shape[0]], gl.int32, gl.SliceLayout(1, L2D))
    ncand = gl.zeros([pos.shape[0]], gl.int32, gl.SliceLayout(1, L2D))
    base = 0 * L
    if ONE_TILE:
        nkey = key_reg
        nvalid = pos < L
    else:  # two tiles in flight: chunk ch+1 is already loaded, chunk ch+2 is requested
        n2key, n2valid = _load_keys(row_ptr, ks, L, BLOCK, pos, True)
    for ch in range(NUM_CHUNKS):
        key = nkey
        valid = nvalid
        if not ONE_TILE:
            nkey = n2key
            nvalid = n2valid
            n2key, n2valid = _load_keys(row_ptr, ks, L, (ch + 2) * BLOCK, pos, True)
        d = key >> 20
        win = valid & (d > hi)
        cand = valid & (d >= lo) & (d <= hi)
        packed = win.to(gl.int32) + (cand.to(gl.int32) << 12)
        pre, tot, lane_tot = _ranks(packed, L2D, LANES)
        pre = pre + base
        crank = pre >> 12
        wslot = pre & 0xFFF
        idx = ch * BLOCK + pos
        dst = gl.where(
            win & (wslot < K),
            wslot,
            gl.where(
                cand & (crank < CAP), K + crank, K + CAP + (idx & (BUF - K - CAP - 1))
            ),
        )
        val = gl.where(cand, ((d - lo).to(gl.int32) << 18) | idx, idx)
        buf.scatter(gl.reshape(val, [BLOCK]), gl.reshape(dst, [BLOCK]), 0)
        if TRACK_TIES:
            kmin = gl.minimum(kmin, gl.min(gl.where(cand, key, _MAXKEY), axis=1))
            kmax = gl.maximum(kmax, gl.max(gl.where(cand, key, 0), axis=1))
        nwin = nwin + (lane_tot & 0xFFF)  # per-thread totals of this tile (<= VPT each)
        ncand = ncand + (lane_tot >> 12)
        base = base + tot
    if TRACK_TIES:
        tied = gl.min(kmin, axis=0) == gl.max(kmax, axis=0)
    else:
        tied = L < 0
    return gl.sum(nwin, axis=0), gl.sum(ncand, axis=0), tied


@gluon.jit
def _refine_span(
    buf,
    hist_s,
    row_ptr,
    ks,
    out_row,
    inbin,
    need,
    above,
    blo,
    bhi,
    K: gl.constexpr,
    NC: gl.constexpr,
    CX: gl.constexpr,
    H1: gl.constexpr,
    H3: gl.constexpr,
    LANES: gl.constexpr,
):
    """Top need of the inbin <= NC candidates in buf[K:] spanning digits [blo, blo + 7]: two radix levels."""
    ROWS: gl.constexpr = 32 * CX.warps_per_cta[0]
    lr = gl.arange(0, ROWS, layout=gl.SliceLayout(1, CX))
    lc = gl.arange(0, NC // ROWS, layout=gl.SliceLayout(0, CX))
    lpos = lr[:, None] * (NC // ROWS) + lc[None, :]
    lsel = lpos < inbin
    lval = gl.reshape(buf.gather(K + gl.reshape(lpos, [NC]), 0), [ROWS, NC // ROWS])
    lidx = gl.where(lsel, lval & _IDX_MASK, 0)
    lfirst = (lc == 0)[None, :] & (lpos >= 0)
    span = bhi - blo + 1
    S = (
        8
        + (span > 1).to(gl.uint32)
        + (span > 2).to(gl.uint32)
        + (span > 4).to(gl.uint32)
    )
    rp, rsm, _in = row_ptr
    if rsm is not None:
        kl, _ = _load_keys(row_ptr, ks, 1 << 30, 0, lidx)
    else:
        kl = _to_key(gl.load(rp + ks + lidx, mask=lsel, other=0.0), False)
    rel = gl.where(lsel, kl - (blo << 20), 0)
    _hist_scatter(hist_s, rel >> S, lsel, lfirst, 0, 4096, False)
    gl.barrier()
    c1, a1, _ = _pick_bin(hist_s, need, 4096, H1)
    p1 = c1.to(gl.uint32)
    in1 = lsel & ((rel >> S) == p1)
    hist23 = buf.slice(buf.shape[0] - 2048, 2048, dim=0)
    _hist_scatter(hist23, rel & ((1 << S) - 1), in1, lfirst, 0, 2048, False)
    gl.barrier()
    b2, a2, _ = _pick_bin(hist23, need - a1, 2048, H3)
    t = (p1 << S) + b2.to(gl.uint32)  # relative threshold key
    _select_out(
        out_row,
        rel,
        lidx,
        lsel,
        t,
        a1 + a2,
        need - a1 - a2,
        above,
        0 * inbin,
        0 * inbin,
        CX,
        LANES,
    )


@gluon.jit
def _classify_row(
    row_ptr,
    ks,
    L,
    out_row,
    hist_s,
    buf,
    K: gl.constexpr,
    VPT: gl.constexpr,
    NUM_CHUNKS,
    CAP: gl.constexpr,
    BUF: gl.constexpr,
    SAMPLE_MIN: gl.constexpr,
    SAMPLE_BLOCKS: gl.constexpr,
    HI_NUM: gl.constexpr,
    LO_NUM: gl.constexpr,
    num_warps: gl.constexpr,
    ONE_TILE: gl.constexpr,
    ZERO_AGG: gl.constexpr,
):
    """Top-K of a row with L > K. hist_s: 4096 int32 of shared memory, buf: BUF int32 (any contents)."""
    ROWS: gl.constexpr = 32 * num_warps
    BLOCK: gl.constexpr = ROWS * VPT
    L2D: gl.constexpr = gl.BlockedLayout([1, VPT], [32, 1], [num_warps, 1], [1, 0])
    C2D: gl.constexpr = gl.BlockedLayout(
        [1, CAP // ROWS], [32, 1], [num_warps, 1], [1, 0]
    )
    H1: gl.constexpr = gl.BlockedLayout([max(4096 // ROWS, 1)], [32], [num_warps], [0])
    H2: gl.constexpr = gl.BlockedLayout([max(1024 // ROWS, 1)], [32], [num_warps], [0])
    H3: gl.constexpr = gl.BlockedLayout([max(2048 // ROWS, 1)], [32], [num_warps], [0])
    SPAN_NC: gl.constexpr = CAP // 2  # _refine_span candidates (more costs VGPRs)
    CH: gl.constexpr = gl.BlockedLayout(
        [1, max(SPAN_NC // ROWS, 1)], [32, 1], [num_warps, 1], [1, 0]
    )
    OL: gl.constexpr = gl.BlockedLayout([max(K // ROWS, 1)], [32], [num_warps], [0])
    LANES: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    r = gl.arange(0, ROWS, layout=gl.SliceLayout(1, L2D))
    c = gl.arange(0, VPT, layout=gl.SliceLayout(0, L2D))
    pos = (c[None, :] // 4) * (ROWS * 4) + r[:, None] * 4 + (c[None, :] % 4)
    first = (c == 0)[None, :] & (pos >= 0)
    hist_s.store(gl.zeros([4096], gl.int32, H1))
    # buf = [K winners | CAP candidates | junk]; the refine histograms (end of junk) are zeroed after classify
    gl.static_assert(BUF - K - CAP >= 2048)
    hist2 = buf.slice(BUF - 2048, 1024, dim=0)
    hist3 = buf.slice(BUF - 1024, 1024, dim=0)
    gl.barrier()

    key_reg = gl.zeros([ROWS, VPT], gl.uint32, L2D)
    b1x = (0 * L).to(gl.uint32)
    blo = (0 * L).to(gl.uint32)  # candidates: keys with top-12-bit digit in [blo, bhi]
    bhi = (0 * L).to(gl.uint32)
    above = 0 * L
    inbin = 0 * L
    tied = L < 0
    ok = L < 0
    if L > SAMPLE_MIN:
        sidx = (L // SAMPLE_BLOCKS) * (pos // (BLOCK // SAMPLE_BLOCKS)) + pos % (
            BLOCK // SAMPLE_BLOCKS
        )
        t0key, t0valid = _load_keys(row_ptr, ks, L, 0, pos, True)
        skey, _ = _load_keys(row_ptr, ks, L, 0, sidx)
        _hist_scatter(hist_s, skey, sidx >= 0, first, 20, 4096, False)
        gl.barrier()
        exp = (K * BLOCK) // L
        n_hi = gl.maximum(exp * HI_NUM // 16, 1)
        n_lo = gl.minimum(exp * LO_NUM // 16 + 8, BLOCK)
        h = hist_s.load(H1)
        bins = gl.arange(0, 4096, layout=H1)
        incl = gl.associative_scan(h, 0, _sum_combine, reverse=True)
        excl = incl - h
        hi = gl.max(gl.where((excl < n_hi) & (incl >= n_hi), bins, -1), axis=0)
        lo = gl.max(gl.where((excl < n_lo) & (incl >= n_lo), bins, 0), axis=0)
        gl.barrier()
        hist_s.store(gl.zeros([4096], gl.int32, H1))
        gl.barrier()
        blo = lo.to(gl.uint32)
        bhi = hi.to(gl.uint32)
        above, inbin, tied = _classify(
            key_reg,
            buf,
            row_ptr,
            ks,
            L,
            pos,
            lo.to(gl.uint32),
            hi.to(gl.uint32),
            t0key,
            t0valid,
            K,
            CAP,
            BUF,
            BLOCK,
            NUM_CHUNKS,
            ONE_TILE,
            L2D,
            LANES,
            False,
        )
        ok = (above < K) & (above + inbin >= K) & ((inbin <= CAP) | tied)
    if ok == 0:
        nkey, nvalid = _load_keys(row_ptr, ks, L, 0, pos, True)
        for ch in range(NUM_CHUNKS):
            key = nkey
            valid = nvalid
            nkey, nvalid = _load_keys(row_ptr, ks, L, (ch + 1) * BLOCK, pos, True)
            _hist_scatter(hist_s, key, valid, first, 20, 4096, ZERO_AGG)
            if ONE_TILE:
                key_reg = key
        gl.barrier()
        b1, _, _ = _pick_bin(hist_s, K, 4096, H1)
        b1u = b1.to(gl.uint32)
        b1x = b1u
        blo = b1u
        bhi = b1u
        gl.barrier()
        hist_s.store(gl.zeros([4096], gl.int32, H1))
        t0key, t0valid = _load_keys(row_ptr, ks, L, 0, pos, True)
        above, inbin, tied = _classify(
            key_reg,
            buf,
            row_ptr,
            ks,
            L,
            pos,
            b1u,
            b1u,
            t0key,
            t0valid,
            K,
            CAP,
            BUF,
            BLOCK,
            NUM_CHUNKS,
            ONE_TILE,
            L2D,
            LANES,
            True,
        )
    need = K - above

    gl.barrier()
    hist2.store(gl.zeros([1024], gl.int32, H2))
    hist3.store(gl.zeros([1024], gl.int32, H2))
    gl.barrier()
    copy_fill = (inbin == need) | tied
    j = gl.arange(0, K, layout=OL)
    src = gl.where(j < above, j, K + j - above)
    gl.store(out_row + j, buf.gather(src, 0) & _IDX_MASK, mask=(j < above) | copy_fill)
    if copy_fill == 0:
        if (bhi - blo < 8) & (inbin <= SPAN_NC):
            _refine_span(
                buf,
                hist_s,
                row_ptr,
                ks,
                out_row,
                inbin,
                need,
                above,
                blo,
                bhi,
                K,
                SPAN_NC,
                CH,
                H1,
                H3,
                LANES,
            )
        elif inbin <= CAP:
            lr = gl.arange(0, ROWS, layout=gl.SliceLayout(1, C2D))
            lc = gl.arange(0, CAP // ROWS, layout=gl.SliceLayout(0, C2D))
            lpos = lr[:, None] * (CAP // ROWS) + lc[None, :]
            lflat = gl.reshape(lpos, [CAP])
            lsel = lpos < inbin
            lval = gl.where(
                lsel, gl.reshape(buf.gather(K + lflat, 0), [ROWS, CAP // ROWS]), 0
            )
            lidx = lval & _IDX_MASK
            lfirst = (lc == 0)[None, :] & (lpos >= 0)
            lbin = (lval >> 18).to(gl.uint32) + blo
            _hist_scatter(hist_s, lbin << 20, lsel, lfirst, 20, 4096, False)
            gl.barrier()
            c1, a1, _ = _pick_bin(hist_s, need, 4096, H1)
            p1 = c1.to(gl.uint32)
            in1 = lsel & (lbin == p1)
            k1, _ = _load_keys(row_ptr, ks, L, 0, gl.where(in1, lidx, 0))
            lkey = gl.where(in1, k1, gl.where(lsel & (lbin > p1), _MAXKEY, 0))
            _hist_scatter(hist2, lkey, in1, lfirst, 10, 1024, False)
            gl.barrier()
            b2, a2, _ = _pick_bin(hist2, need - a1, 1024, H2)
            p2 = (p1 << 10) | b2.to(gl.uint32)
            _hist_scatter(
                hist3, lkey, in1 & ((lkey >> 10) == p2), lfirst, 0, 1024, False
            )
            gl.barrier()
            b3, a3, _ = _pick_bin(hist3, need - a1 - a2, 1024, H2)
            t = (p2 << 10) | b3.to(gl.uint32)
            _select_out(
                out_row,
                lkey,
                lidx,
                lsel,
                t,
                a1 + a2 + a3,
                need - a1 - a2 - a3,
                above,
                0 * L,
                0 * L,
                C2D,
                LANES,
            )
        else:
            b1u = b1x
            for ch in range(NUM_CHUNKS):
                if ONE_TILE:
                    key, valid = key_reg, pos < L
                else:
                    key, valid = _load_keys(row_ptr, ks, L, ch * BLOCK, pos, True)
                _hist_scatter(
                    hist2, key, valid & ((key >> 20) == b1u), first, 10, 1024, False
                )
            gl.barrier()
            b2, a2, _ = _pick_bin(hist2, need, 1024, H2)
            p2 = (b1u << 10) | b2.to(gl.uint32)
            for ch in range(NUM_CHUNKS):
                if ONE_TILE:
                    key, valid = key_reg, pos < L
                else:
                    key, valid = _load_keys(row_ptr, ks, L, ch * BLOCK, pos, True)
                _hist_scatter(
                    hist3, key, valid & ((key >> 10) == p2), first, 0, 1024, False
                )
            gl.barrier()
            b3, a3, _ = _pick_bin(hist3, need - a2, 1024, H2)
            t = (p2 << 10) | b3.to(gl.uint32)
            gt_base = 0 * L
            eq_base = 0 * L
            for ch in range(NUM_CHUNKS):
                if ONE_TILE:
                    key, valid = key_reg, pos < L
                else:
                    key, valid = _load_keys(row_ptr, ks, L, ch * BLOCK, pos, True)
                gt_base, eq_base = _select_out(
                    out_row,
                    key,
                    ch * BLOCK + pos,
                    valid & ((key >> 20) == b1u),
                    t,
                    a2 + a3,
                    need - a2 - a3,
                    above,
                    gt_base,
                    eq_base,
                    L2D,
                    LANES,
                )


@gluon.jit
def _topk_classify_kernel(
    logits_ptr,
    ks_ptr,
    ke_ptr,
    out_ptr,
    stride0,
    out_stride0,
    K: gl.constexpr,
    VPT: gl.constexpr,
    NUM_CHUNKS: gl.constexpr,
    CAP: gl.constexpr,
    BUF: gl.constexpr,
    SAMPLE_MIN: gl.constexpr,
    SAMPLE_BLOCKS: gl.constexpr,
    HI_NUM: gl.constexpr,
    LO_NUM: gl.constexpr,
    ZERO_AGG: gl.constexpr,
    num_warps: gl.constexpr,
):
    ROWS: gl.constexpr = 32 * num_warps
    OL: gl.constexpr = gl.BlockedLayout([max(K // ROWS, 1)], [32], [num_warps], [0])
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    row = gl.program_id(0)
    ks = gl.load(ks_ptr + row)
    L = gl.maximum(gl.load(ke_ptr + row), ks) - ks
    out_row = out_ptr + row.to(gl.int64) * out_stride0
    if L <= K:
        i = gl.arange(0, K, layout=OL)
        gl.store(out_row + i, gl.where(i < L, i, -1))
    else:
        _classify_row(
            (logits_ptr + row.to(gl.int64) * stride0, None, False),
            ks,
            L,
            out_row,
            gl.allocate_shared_memory(gl.int32, [4096], SL),
            gl.allocate_shared_memory(gl.int32, [BUF], SL),
            K,
            VPT,
            NUM_CHUNKS,
            CAP,
            BUF,
            SAMPLE_MIN,
            SAMPLE_BLOCKS,
            HI_NUM,
            LO_NUM,
            num_warps,
            NUM_CHUNKS == 1,
            ZERO_AGG,
        )
