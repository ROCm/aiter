# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact top-512 tokens of a short row from its token logits (``scores`` with ``tokens=True``), one workgroup
per row.

For contexts up to a few thousand entries the walk writes every token's logit (4 B a token: the same bits as the
select's
rescoring, so the answer equals the block path's) and this kernel selects from them directly: no block buckets, no
rescoring gather. A thread owns ``E`` consecutive tokens (keys: the order-preserving u32 of the logit, 0 for tokens past
the row's end); four 8-bit radix passes find the 512th key, the ties at it go to the lower positions (threads and
elements
ascend with the position), and an ordered compaction writes the selected positions ascending, -1 padded, and the sparse
attention's ragged slot lists. One barrier a radix pass: the histograms rotate through three buffers (the one two passes
ahead is cleared after this pass's barrier, when every wave has left its last scan), every wave scans the 256 buckets
itself (4 a lane), and a bucket has ``HIST_COPIES`` words (lane % copies) so lanes on the same bucket rarely hit one
word.
A workgroup is 256-512 threads, so its barriers are cheap and several rows share a CU (the block path's 1024-thread
workgroup with ~40 barriers is one row a CU).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .common import (
    BLOCK,
    CANDIDATES,
    I64,
    TOPK,
    F,
    I,
    U,
    key_of,
    load,
    sload,
    store,
    view,
    wave_of,
    wave_scan_incl,
    xor_lane,
)

HIST_COPIES = 2
MAX_POSITION = 262144  # entries a row's positions reach by default (the gathered mode's slot table)


@functools.cache
def compile_tokens_select(
    next_n: int,
    tokens: int,
    nt: int,
    slot_log2: int = 0,
    slot_rows: int = 1024,
    emit_candidates: bool = False,
    gathered: bool = False,
    max_position: int = MAX_POSITION,
    min_end: int = 0,
):
    """``tokens``: the most entries of a row (the logits row holds ``tokens + 64`` floats; a multiple of 4 * ``nt``);
    ``slot_log2`` (> 0): also write the sparse attention's ragged slot lists (entries per page of the attention cache =
    2 ** slot_log2) for up to ``slot_rows`` rows (a multiple of ``nt``); ``emit_candidates`` (the candidate source
    layer): also the row's 2048 candidate blocks, which for a row of at most 2048 blocks are all of them (ids ascending,
    -1 padded); ``gathered`` (the candidate consumer): the logits are those of the 2048 candidate blocks' tokens
    (``cand_logits``, ``tokens`` // 8 candidate blocks, -inf: padded), entry ``i`` is the position ``8 * ids[i // 8] + i % 8`` (``cand_out`` is
    then the ids, an input; ``max_position``: the largest entry count of a row, sizes the slot table; ``min_end``: rows of at most that many entries are left to the block path).
    """
    NN, NT, NTOK = next_n, nt, tokens
    NW = NT // 64
    E = NTOK // NT
    R = max(
        HIST_COPIES, NT // 256
    )  # a histogram buffer is 256 * R words: one word a thread to clear
    assert (
        E % 4 == 0
        and NT in (256, 512, 1024)
        and slot_rows % NT == 0
        and (256 * R) % NT == 0
    )
    NSG = (
        max(NT, (max_position if gathered else NTOK) >> slot_log2) if slot_log2 else 16
    )
    assert not emit_candidates or NTOK <= CANDIDATES * BLOCK
    assert not (gathered and emit_candidates) and (
        not gathered or (NTOK % (BLOCK * 512) == 0 and E % BLOCK == 0)
    )
    name = (
        f"indexer_mxfp4_tokens_select_h32_d128_top512_nn{NN}_n{NTOK}_t{NT}"
        + (f"_slots{slot_log2}_r{slot_rows}" if slot_log2 else "")
        + ("_cand" if emit_candidates else "")
        + ("_gathered" if gathered else "")
        + (f"_min{min_end}" if min_end else "")
    )

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, 3 * 256 * R]
        wsum: fx.Array[fx.Int32, 128]
        sg: fx.Array[fx.Int32, NSG]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        logits: fx.Int64,
        row_ends: fx.Int64,
        out: fx.Int64,
        logits_stride: fx.Int32,
        reqs: fx.Int32,
        slots: fx.Int64,
        indptr: fx.Int64,
        lens: fx.Int64,
        valid: fx.Int64,
        slot_table: fx.Int64,
        slot_stride: fx.Int32,
        cand_out: fx.Int64,
    ):
        sm = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        lane = tx & 63
        wave = wave_of(tx)
        row = fx.block_idx.x
        req = row // NN

        # scratch words (wave sums) rotate through four regions: one is not written again before two barriers passed
        turn = [0]

        def bscan(v):
            """Exclusive prefix across the block's threads, block total."""
            turn[0] += 1
            w0 = (turn[0] % 4) * 32
            inc = wave_scan_incl(v)
            sm.wsum.ptr[w0 + (lane == 63).select(wave, I(31))] = inc
            fx.gpu.barrier()
            base, tot = I(0), I(0)
            for ww in range_constexpr(NW):
                s = I(sm.wsum.ptr[w0 + ww])
                base = (wave > I(ww)).select(base + s, base)
                tot = tot + s
            return inc - v + base, tot

        def radix(keys, ksel):
            """Exact select of the ksel best keys (0 = invalid) by four 8-bit passes: (thr, take, has_thr, off, total),
            ties to the lower index."""
            prefix, rem = I(0), I(ksel)
            for qq in range_constexpr(2 * 256 * R // NT):
                sm.hist.ptr[tx + qq * NT] = I(0)
            fx.gpu.barrier()
            cpy = lane & (R - 1)
            for ps in range_constexpr(4):
                shift = 24 - 8 * ps
                hb = (ps % 3) * 256 * R
                for j in range_constexpr(E):
                    k = keys[j]
                    if const_expr(ps == 0):
                        fx.atomic_add(
                            sm.hist.ptr + (hb + ((k >> shift) & 255) * R + cpy), I(1)
                        )
                    else:
                        if (k.bitcast(U) >> U(shift + 8)) == (
                            prefix.bitcast(U) >> U(shift + 8)
                        ):
                            fx.atomic_add(
                                sm.hist.ptr + (hb + ((k >> shift) & 255) * R + cpy),
                                I(1),
                            )
                fx.gpu.barrier()
                for qq in range_constexpr(256 * R // NT):
                    sm.hist.ptr[((ps + 2) % 3) * 256 * R + tx + qq * NT] = I(0)
                hv = []
                for m_ in range_constexpr(4):
                    dg = 255 - 4 * lane - m_
                    acc_ = I(0)
                    for c_ in range_constexpr(R):
                        acc_ = acc_ + I(sm.hist.ptr[hb + dg * R + c_])
                    hv.append(acc_)
                ssum = (hv[0] + hv[1]) + (hv[2] + hv[3])
                inc = wave_scan_incl(ssum)
                run = inc - ssum
                dig, cb = I(0), I(0)
                for m_ in range_constexpr(4):
                    nxt = run + hv[m_]
                    t_ = (rem > run) & (rem <= nxt)
                    dig = t_.select(I(255) - 4 * lane - m_, dig)
                    cb = t_.select(run, cb)
                    run = nxt
                pk = (
                    dig << 16
                ) | cb  # exactly one lane holds the bucket of the rank: broadcast by max
                for m_ in (1, 2, 4, 8, 16, 32):
                    o_ = xor_lane(pk, m_, lane)
                    pk = (o_ > pk).select(o_, pk)
                rem = rem - (pk & 0xFFFF)
                prefix = prefix | ((pk >> 16) << shift)
            gt, eq = I(0), I(0)
            for j in range_constexpr(E):
                gt = gt + (keys[j].bitcast(U) > prefix.bitcast(U)).select(I(1), I(0))
                eq = eq + (keys[j] == prefix).select(I(1), I(0))
            # one scan for both counts (each below 2^16): the ties taken are the first ``rem`` of the row's ties in
            # position
            # order, so before this thread ``min(rem, ties before it)`` of them are taken
            pre, tot = bscan(gt | (eq << 16))
            eq_before = pre >> 16
            take = rem - eq_before
            take = (take < I(0)).select(I(0), take)
            take = (take > eq).select(eq, take)
            has_thr = prefix != I(0)
            tie_before = (eq_before > rem).select(rem, eq_before)
            tie_tot = ((tot >> 16) > rem).select(rem, tot >> 16)
            off = (pre & 0xFFFF) + has_thr.select(tie_before, I(0))
            total = (tot & 0xFFFF) + has_thr.select(tie_tot, I(0))
            return prefix, take, has_thr, off, total

        rows_t = reqs * NN
        go = row < rows_t
        if const_expr(min_end):
            go = go & (load(view(row_ends, I64(rows_t) * 4), row) > I(min_end))
        if go:
            lr = view(
                logits + I64(row) * I64(logits_stride) * 4,
                I64(logits_stride) * 4,
                F,
                4,
            )
            raw = [load(lr, tx * E + qq * 4, F, 4) for qq in range_constexpr(E // 4)]
            end = sload(view(row_ends + I64(row) * 4, 4), 0)
            end = (end < I(0)).select(I(0), end)
            if const_expr(not gathered):
                end = (end > I(NTOK)).select(I(NTOK), end)
            orr = view(out + I64(row) * (TOPK * 4), TOPK * 4)
            if const_expr(emit_candidates):
                # at most 2048 blocks: the candidate list is every block of the row
                cor = view(cand_out + I64(row) * (CANDIDATES * 4), CANDIDATES * 4)
                nb = (end + (BLOCK - 1)) // BLOCK
                for qq in range_constexpr(CANDIDATES // NT):
                    p = tx + qq * NT
                    store(cor, p, (p < nb).select(p, I(-1)))
            if const_expr(slot_log2 > 0):
                # the ragged lists: this row starts after the rows before it, each min(512, end) long (0 when not valid)
                vr = view(valid, I64(rows_t) * 4)
                er = view(row_ends, I64(rows_t) * 4)
                before = I(0)
                for qq in range_constexpr(slot_rows // NT):
                    j_ = tx + qq * NT
                    e_j = load(er, j_)
                    v_j = load(vr, j_)
                    ec = (e_j < I(0)).select(I(0), e_j)
                    l_j = (v_j != I(0)).select((ec > I(TOPK)).select(I(TOPK), ec), I(0))
                    before = before + (j_ < row).select(l_j, I(0))
                _o, start = bscan(before)
                my_valid = sload(vr, row) != I(0)
                my_len = my_valid.select((end < I(TOPK)).select(end, I(TOPK)), I(0))
                sgr = view(
                    slot_table + I64(req) * I64(slot_stride) * 4, I64(slot_stride) * 4
                )
                npages = (
                    end + ((1 << slot_log2) - 1)
                ) >> slot_log2  # the pages the row's positions reach
                for qq in range_constexpr(-(-NSG // NT)):
                    i_ = tx + qq * NT
                    sm.sg.ptr[(i_ < NSG).select(i_, I(NSG - 1))] = load(
                        sgr, i_, mask=i_ < npages
                    )
                ipr = view(indptr, I64(rows_t + 1) * 4)
                store(ipr, row, start, mask=tx == 0)
                store(ipr, rows_t, start + my_len, mask=(tx == 0) & (row == rows_t - 1))
                store(view(lens, I64(rows_t) * 4), row, my_len, mask=tx == 0)

            if const_expr(gathered):
                gr = view(cand_out + I64(row) * (CANDIDATES * 4), CANDIDATES * 4)
                ids = [
                    load(gr, tx * (E // BLOCK) + b_)
                    for b_ in range_constexpr(E // BLOCK)
                ]
            keys = []
            for qq in range_constexpr(E // 4):
                for j in range_constexpr(4):
                    x = F(raw[qq][j])
                    if const_expr(gathered):
                        keys.append(
                            key_of(x)
                        )  # -inf (a padded block or a token past the row's end): invalid
                    else:
                        keys.append((tx * E + qq * 4 + j < end).select(key_of(x), I(0)))
            thr, take, has_thr, off, total = radix(
                keys, I(TOPK) if gathered else (end < I(TOPK)).select(end, I(TOPK))
            )
            c, eqs = I(0), I(0)
            for j in range_constexpr(E):
                s_ = (keys[j].bitcast(U) > thr.bitcast(U)) | (
                    (keys[j] == thr) & has_thr & (eqs < take)
                )
                pos = ((ids[j // BLOCK] << 3) | (j % BLOCK)) if gathered else tx * E + j
                store(orr, off + c, pos, mask=s_)
                if const_expr(slot_log2 > 0):
                    slot = (I(sm.sg.ptr[pos >> slot_log2]) << slot_log2) | (
                        pos & ((1 << slot_log2) - 1)
                    )
                    store(
                        view(slots, I64(rows_t) * (TOPK * 4)),
                        start + off + c,
                        slot,
                        mask=s_ & my_valid,
                    )
                c = c + s_.select(I(1), I(0))
                eqs = eqs + (keys[j] == thr).select(I(1), I(0))
            for qq in range_constexpr(-(-TOPK // NT)):
                p = tx + qq * NT
                store(orr, p, I(-1), mask=(p >= total) & (p < TOPK))

    @flyc.jit
    def launch(
        logits: fx.Int64,
        row_ends: fx.Int64,
        out: fx.Int64,
        logits_stride: fx.Int32,
        reqs: fx.Int32,
        slots: fx.Int64,
        indptr: fx.Int64,
        lens: fx.Int64,
        valid: fx.Int64,
        slot_table: fx.Int64,
        slot_stride: fx.Int32,
        cand_out: fx.Int64,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            logits,
            row_ends,
            out,
            logits_stride,
            reqs,
            slots,
            indptr,
            lens,
            valid,
            slot_table,
            slot_stride,
            cand_out,
        ).launch(grid=(fx.Int64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
