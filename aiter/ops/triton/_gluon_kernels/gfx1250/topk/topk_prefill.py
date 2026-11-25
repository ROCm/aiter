# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_MAXKEY = gl.constexpr(0xFFFFFFFF)
_IDX_MASK = gl.constexpr((1 << 18) - 1)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _to_key(x):
    bits = x.to(gl.uint32, bitcast=True)
    key = gl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)
    return gl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, key)


@gluon.jit
def _load_keys(row_ptr, ks, L, base, pos, stride1, CONTIG: gl.constexpr = False):
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
        x = gl.load(row_ptr + (ks + idx).to(gl.int64) * stride1, mask=valid, other=0.0)
    return gl.where(valid, _to_key(x), 0), valid


@gluon.jit
def _tile(
    key_reg,
    row_ptr,
    ks,
    L,
    ch,
    pos,
    stride1,
    BLOCK: gl.constexpr,
    ONE_TILE: gl.constexpr,
    CONTIG: gl.constexpr,
):
    if ONE_TILE:
        return key_reg, pos < L
    else:
        return _load_keys(row_ptr, ks, L, ch * BLOCK, pos, stride1, CONTIG)


@gluon.jit
def _hist_add(hist, key, cand, first, SHIFT: gl.constexpr, NB: gl.constexpr):
    digit = ((key >> SHIFT) & (NB - 1)).to(gl.int32)
    d0 = gl.max(gl.where(first, digit, -1), axis=1)[:, None]
    same = (digit == d0) & cand
    n0 = gl.sum(same.to(gl.int32), axis=1)[:, None]
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
    incl = gl.associative_scan(h, 0, _add, reverse=True)
    hit = (incl - h < need) & (incl >= need)
    b = gl.max(gl.where(hit, bins, -1), axis=0)
    above = gl.sum(gl.where(bins > b, h, 0), axis=0)
    inbin = gl.sum(gl.where(bins == b, h, 0), axis=0)
    return b, above, inbin


@gluon.jit
def _bracket(hist, n_hi, n_lo, NB: gl.constexpr, HL: gl.constexpr):
    h = hist.load(HL)
    bins = gl.arange(0, NB, layout=HL)
    incl = gl.associative_scan(h, 0, _add, reverse=True)
    excl = incl - h
    hi = gl.max(gl.where((excl < n_hi) & (incl >= n_hi), bins, -1), axis=0)
    lo = gl.max(gl.where((excl < n_lo) & (incl >= n_lo), bins, 0), axis=0)
    return lo, hi


@gluon.jit
def _ranks(packed, L2D: gl.constexpr, LANES: gl.constexpr):
    lane_incl = gl.associative_scan(packed, 1, _add)
    lane_tot = gl.sum(packed, axis=1)
    tot_b = gl.convert_layout(lane_tot, LANES)
    lane_excl = gl.convert_layout(
        gl.associative_scan(tot_b, 0, _add) - tot_b, gl.SliceLayout(1, L2D)
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
    stride1,
    lo,
    hi,
    nkey,
    nvalid,
    K: gl.constexpr,
    CAP: gl.constexpr,
    BUF: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_CHUNKS: gl.constexpr,
    ONE_TILE: gl.constexpr,
    CONTIG: gl.constexpr,
    L2D: gl.constexpr,
    LANES: gl.constexpr,
    TRACK_TIES: gl.constexpr,
):
    # TRACK_TIES: also report whether all candidates share one key (lets > CAP tied candidates finish without
    # the exact pass). Off for the sampled classify: those rows take the exact pass, which tracks ties.
    kmin = gl.full([pos.shape[0]], _MAXKEY, gl.uint32, gl.SliceLayout(1, L2D))
    kmax = gl.zeros([pos.shape[0]], gl.uint32, gl.SliceLayout(1, L2D))
    nwin = gl.zeros([pos.shape[0]], gl.int32, gl.SliceLayout(1, L2D))
    ncand = gl.zeros([pos.shape[0]], gl.int32, gl.SliceLayout(1, L2D))
    base = 0 * L
    if ONE_TILE:
        nkey = key_reg
        nvalid = pos < L
    else:  # two tiles in flight: chunk ch+1 is already loaded, chunk ch+2 is requested
        n2key, n2valid = _load_keys(row_ptr, ks, L, BLOCK, pos, stride1, CONTIG)
    for ch in range(NUM_CHUNKS):
        key = nkey
        valid = nvalid
        if not ONE_TILE:
            nkey = n2key
            nvalid = n2valid
            n2key, n2valid = _load_keys(
                row_ptr, ks, L, (ch + 2) * BLOCK, pos, stride1, CONTIG
            )
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
    stride1,
    out_row,
    inbin,
    need,
    above,
    blo,
    bhi,
    K: gl.constexpr,
    CAP: gl.constexpr,
    NC: gl.constexpr,
    CONTIG: gl.constexpr,
    CX: gl.constexpr,
    H1: gl.constexpr,
    H3: gl.constexpr,
    LANES: gl.constexpr,
):
    """Top `need` of the inbin <= NC candidates in buf[K:K+inbin] whose digits span <= 8 bins [blo, bhi]: two
    radix levels over the keys relative to blo << 20 (span <= 2^23). A 12-bit level on hist_s spread over the
    actual span (no hot bins), then the remaining <= 11 low bits of the picked bin on the 2048-slot junk-tail
    histogram. Keys are gathered once (masked: only real candidates load)."""
    ROWS: gl.constexpr = 32 * CX.warps_per_cta[0]
    lr = gl.arange(0, ROWS, layout=gl.SliceLayout(1, CX))
    lc = gl.arange(0, NC // ROWS, layout=gl.SliceLayout(0, CX))
    lpos = lr[:, None] * (NC // ROWS) + lc[None, :]
    lsel = lpos < inbin
    lval = gl.reshape(buf.gather(K + gl.reshape(lpos, [NC]), 0), [ROWS, NC // ROWS])
    lidx = gl.where(lsel, lval & _IDX_MASK, 0)
    lfirst = (lc == 0)[None, :] & (lpos >= 0)
    span = bhi - blo + 1
    S = 8 + (span > 1).to(gl.uint32) + (span > 2).to(gl.uint32) + (span > 4).to(gl.uint32)
    if CONTIG:
        kx = gl.load(row_ptr + ks + lidx, mask=lsel, other=0.0)
    else:
        kx = gl.load(row_ptr + (ks + lidx).to(gl.int64) * stride1, mask=lsel, other=0.0)
    rel = gl.where(lsel, _to_key(kx) - (blo << 20), 0)
    _hist_add(hist_s, rel >> S, lsel, lfirst, 0, 4096)
    gl.barrier()
    c1, a1, _ = _pick_bin(hist_s, need, 4096, H1)
    p1 = c1.to(gl.uint32)
    in1 = lsel & ((rel >> S) == p1)
    hist23 = buf.slice(K + CAP, 2048, dim=0)  # = hist2 | hist3, zeroed after classify
    _hist_add(hist23, rel & ((1 << S) - 1), in1, lfirst, 0, 2048)
    gl.barrier()
    b2, a2, _ = _pick_bin(hist23, need - a1, 2048, H3)
    t = (p1 << S) + b2.to(gl.uint32)  # relative threshold key
    _select_out(out_row, rel, lidx, lsel, t, a1 + a2, need - a1 - a2, above, 0 * inbin, 0 * inbin, CX, LANES)



@gluon.jit
def _topk_prefill_kernel(
    logits_ptr,
    ks_ptr,
    ke_ptr,
    out_ptr,
    stride0,
    stride1,
    out_stride0,
    K: gl.constexpr,
    VPT: gl.constexpr,
    NUM_CHUNKS: gl.constexpr,
    CAP: gl.constexpr,
    BUF: gl.constexpr,
    SAMPLE_MIN: gl.constexpr,
    CONTIG: gl.constexpr,
    SAMPLE_BLOCKS: gl.constexpr,
    HI_NUM: gl.constexpr,
    LO_NUM: gl.constexpr,
    num_warps: gl.constexpr,
    SPAN_NC: gl.constexpr = 2048,
):
    ROWS: gl.constexpr = 32 * num_warps
    BLOCK: gl.constexpr = ROWS * VPT
    ONE_TILE: gl.constexpr = NUM_CHUNKS == 1
    L2D: gl.constexpr = gl.BlockedLayout([1, VPT], [32, 1], [num_warps, 1], [1, 0])
    C2D: gl.constexpr = gl.BlockedLayout(
        [1, CAP // ROWS], [32, 1], [num_warps, 1], [1, 0]
    )
    H1: gl.constexpr = gl.BlockedLayout([max(4096 // ROWS, 1)], [32], [num_warps], [0])
    H2: gl.constexpr = gl.BlockedLayout([max(1024 // ROWS, 1)], [32], [num_warps], [0])
    H3: gl.constexpr = gl.BlockedLayout([max(2048 // ROWS, 1)], [32], [num_warps], [0])
    # _refine_span handles up to SPAN_NC candidates (CAP // 2 on shorter rows keeps the kernel at 112 VGPRs)
    CH: gl.constexpr = gl.BlockedLayout([1, max(SPAN_NC // ROWS, 1)], [32, 1], [num_warps, 1], [1, 0])
    OL: gl.constexpr = gl.BlockedLayout([max(K // ROWS, 1)], [32], [num_warps], [0])
    LANES: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    row = gl.program_id(0)
    ks = gl.load(ks_ptr + row)
    L = gl.maximum(gl.load(ke_ptr + row), ks) - ks
    out_row = out_ptr + row.to(gl.int64) * out_stride0

    if L <= K:
        i = gl.arange(0, K, layout=OL)
        gl.store(out_row + i, gl.where(i < L, i, -1))
    else:
        row_ptr = logits_ptr + row.to(gl.int64) * stride0
        r = gl.arange(0, ROWS, layout=gl.SliceLayout(1, L2D))
        c = gl.arange(0, VPT, layout=gl.SliceLayout(0, L2D))
        pos = (c[None, :] // 4) * (ROWS * 4) + r[:, None] * 4 + (c[None, :] % 4)
        first = (c == 0)[None, :] & (pos >= 0)

        hist_s = gl.allocate_shared_memory(
            gl.int32, [4096], SL, gl.zeros([4096], gl.int32, H1)
        )
        # buf = [K winners | CAP candidates | BUF - K - CAP junk slots]. classify scatters every
        # non-selected key into the junk tail; the refine histograms hist2 / hist3 live there too
        # (only used after classify), so they are zeroed once classify is done.
        buf = gl.allocate_shared_memory(gl.int32, [BUF], SL)
        gl.static_assert(BUF - K - CAP >= 2048)
        hist2 = buf.slice(K + CAP, 1024, dim=0)
        hist3 = buf.slice(K + CAP + 1024, 1024, dim=0)
        gl.barrier()

        key_reg = gl.zeros([ROWS, VPT], gl.uint32, L2D)
        b1x = (0 * L).to(gl.uint32)
        blo = (0 * L).to(gl.uint32)
        bhi = (0 * L).to(gl.uint32)  # candidates are the keys with top-12-bit digit in [blo, bhi]
        above = 0 * L
        inbin = 0 * L
        tied = L < 0
        ok = L < 0
        if L > SAMPLE_MIN:
            sidx = (L // SAMPLE_BLOCKS) * (pos // (BLOCK // SAMPLE_BLOCKS)) + pos % (
                BLOCK // SAMPLE_BLOCKS
            )
            t0key, t0valid = _load_keys(row_ptr, ks, L, 0, pos, stride1, CONTIG)
            skey, _ = _load_keys(row_ptr, ks, L, 0, sidx, stride1)
            _hist_add(hist_s, skey, sidx >= 0, first, 20, 4096)
            gl.barrier()
            exp = (K * BLOCK) // L
            lo, hi = _bracket(
                hist_s,
                gl.maximum(exp * HI_NUM // 16, 1),
                gl.minimum(exp * LO_NUM // 16 + 8, BLOCK),
                4096,
                H1,
            )
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
                stride1,
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
                CONTIG,
                L2D,
                LANES,
                False,
            )
            ok = (above < K) & (above + inbin >= K) & ((inbin <= CAP) | tied)
        if ok == 0:
            nkey, nvalid = _load_keys(row_ptr, ks, L, 0, pos, stride1, CONTIG)
            for ch in range(NUM_CHUNKS):
                key = nkey
                valid = nvalid
                nkey, nvalid = _load_keys(
                    row_ptr, ks, L, (ch + 1) * BLOCK, pos, stride1, CONTIG
                )
                _hist_add(hist_s, key, valid, first, 20, 4096)
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
            t0key, t0valid = _load_keys(row_ptr, ks, L, 0, pos, stride1, CONTIG)
            above, inbin, tied = _classify(
                key_reg,
                buf,
                row_ptr,
                ks,
                L,
                pos,
                stride1,
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
                CONTIG,
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
        gl.store(
            out_row + j, buf.gather(src, 0) & _IDX_MASK, mask=(j < above) | copy_fill
        )
        if copy_fill == 0:
            if (bhi - blo < 8) & (inbin <= SPAN_NC):
                _refine_span(buf, hist_s, row_ptr, ks, stride1, out_row, inbin, need, above, blo, bhi,
                             K, CAP, SPAN_NC, CONTIG, CH, H1, H3, LANES)
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
                _hist_add(hist_s, lbin << 20, lsel, lfirst, 20, 4096)
                gl.barrier()
                c1, a1, _ = _pick_bin(hist_s, need, 4096, H1)
                p1 = c1.to(gl.uint32)
                in1 = lsel & (lbin == p1)
                k1, _ = _load_keys(row_ptr, ks, L, 0, gl.where(in1, lidx, 0), stride1)
                lkey = gl.where(in1, k1, gl.where(lsel & (lbin > p1), _MAXKEY, 0))
                _hist_add(hist2, lkey, in1, lfirst, 10, 1024)
                gl.barrier()
                b2, a2, _ = _pick_bin(hist2, need - a1, 1024, H2)
                p2 = (p1 << 10) | b2.to(gl.uint32)
                _hist_add(hist3, lkey, in1 & ((lkey >> 10) == p2), lfirst, 0, 1024)
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
                    key, valid = _tile(
                        key_reg,
                        row_ptr,
                        ks,
                        L,
                        ch,
                        pos,
                        stride1,
                        BLOCK,
                        ONE_TILE,
                        CONTIG,
                    )
                    _hist_add(hist2, key, valid & ((key >> 20) == b1u), first, 10, 1024)
                gl.barrier()
                b2, a2, _ = _pick_bin(hist2, need, 1024, H2)
                p2 = (b1u << 10) | b2.to(gl.uint32)
                for ch in range(NUM_CHUNKS):
                    key, valid = _tile(
                        key_reg,
                        row_ptr,
                        ks,
                        L,
                        ch,
                        pos,
                        stride1,
                        BLOCK,
                        ONE_TILE,
                        CONTIG,
                    )
                    _hist_add(hist3, key, valid & ((key >> 10) == p2), first, 0, 1024)
                gl.barrier()
                b3, a3, _ = _pick_bin(hist3, need - a2, 1024, H2)
                t = (p2 << 10) | b3.to(gl.uint32)
                gt_base = 0 * L
                eq_base = 0 * L
                for ch in range(NUM_CHUNKS):
                    key, valid = _tile(
                        key_reg,
                        row_ptr,
                        ks,
                        L,
                        ch,
                        pos,
                        stride1,
                        BLOCK,
                        ONE_TILE,
                        CONTIG,
                    )
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
