# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# MiniMax-M3 prefill index top-k fused with the MSA page sort_topk.
#
# One program per (group of TQ consecutive query rows, index head):
#   1. each row's top-TOPK blocks by index score, in AITER's
#      pa_sparse_block_topk order (vLLM's m3_index_topk_prefill_indices), and
#   2. for TQ > 1, the group's page sort_topk the Gluon MSA kernel walks
#      (msa_sort_topk_kernel's uniq / count / tokblk / pos), so the attention
#      needs no sort_topk pre-pass.
#
# Top-k order: block i < min(num_valid_pages[row], width) has the key
# bits ^ ((bits >> 31) & 0x7FFFFFFF) of its fp32 score bits (a signed int order
# of the floats: -0 < +0, NaNs folded to -inf); blocks rank by (key, i)
# descending, so ties go to the larger block id; rows with fewer than TOPK
# live blocks pad with -1.
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_m3_index_topk_prefill_indices_sort_topk_kernel_repr = make_kernel_repr(
    "fused_m3_index_topk_prefill_indices_sort_topk_kernel",
    ["TOPK", "TQ", "CHUNK", "TWO_PASS", "G", "SORT_TOPK"],
)


@triton.jit
def _comp(s_row, idx, nlive, KEY_POS_INF: tl.constexpr, KEY_NEG_INF: tl.constexpr,
          DEAD: tl.constexpr):  # fmt: skip
    """(key << 32 | block) int64 of blocks idx ([TQ, N] or [N]) of the rows
    (s_row [TQ] score row pointers); DEAD past each row's live blocks."""
    if len(idx.shape) == 1:
        idx = idx[None, :]
    live = idx < nlive[:, None]
    bits = tl.load(s_row[:, None] + idx, mask=live, other=0.0).to(tl.int32, bitcast=True)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    key = tl.where((key > KEY_POS_INF) | (key < KEY_NEG_INF), KEY_NEG_INF, key)
    return tl.where(live, (key.to(tl.int64) << 32) | idx.to(tl.int64), DEAD)


@triton.jit(repr=_fused_m3_index_topk_prefill_indices_sort_topk_kernel_repr)
def fused_m3_index_topk_prefill_indices_sort_topk_kernel(
    score_ptr,  # fp32 [H, rows, >= width]
    nvp_ptr,  # int32 [rows]: row n's visible blocks
    topk_ptr,  # int32 [H, rows, TOPK] out
    block_table_ptr,  # int32 [reqs, max_blocks]          (SORT_TOPK)
    query_req_ptr,  # int32 [rows]                       (SORT_TOPK)
    query_pos_ptr,  # int32 [rows]                       (SORT_TOPK)
    uniq_ptr,  # int32 [H, groups, C] out: distinct pages   (SORT_TOPK)
    count_ptr,  # int32 [H, groups] out                       (SORT_TOPK)
    tokblk_ptr,  # int32 [H, groups, C, TQ] out, -1 = not picked (SORT_TOPK)
    pos_ptr,  # int32 [groups * TQ] out (head 0)              (SORT_TOPK)
    num_rows,
    width,
    stride_s_h,
    stride_s_t,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    TOPK: tl.constexpr,
    TQ: tl.constexpr,
    CHUNK: tl.constexpr,
    TWO_PASS: tl.constexpr,
    G: tl.constexpr,
    SORT_TOPK: tl.constexpr,
):
    BLK: tl.constexpr = 128
    KEY_POS_INF: tl.constexpr = 0x7F800000
    KEY_NEG_INF: tl.constexpr = -2139095041  # key of -inf (0xFF800000)
    DEAD: tl.constexpr = -9223372036854775808  # below every live (key, i)

    grp = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.arange(0, TQ)
    row = grp * TQ + t
    r_ok = row < num_rows
    nlive = tl.minimum(tl.load(nvp_ptr + row, mask=r_ok, other=0), width)
    n_max = tl.max(nlive, axis=0)
    s_row = score_ptr + h.to(tl.int64) * stride_s_h + row.to(tl.int64) * stride_s_t

    col = tl.arange(0, CHUNK)
    if TWO_PASS:
        # Pass 1: stream the row keeping each group of G blocks' max and the
        # running top TOPK of those maxima. Every top-TOPK block's group is
        # among them (a group max above its own group max is a larger block,
        # and at most TOPK - 1 blocks are larger), so pass 2 takes the exact
        # top TOPK from just those TOPK * G blocks.
        NG: tl.constexpr = CHUNK // G
        gbest = tl.full([TQ, TOPK], DEAD, tl.int64)
        for c0 in range(0, n_max, CHUNK):
            comp = _comp(s_row, c0 + col, nlive, KEY_POS_INF, KEY_NEG_INF, DEAD)
            gmax = tl.max(tl.reshape(comp, [TQ, NG, G]), axis=2)
            gtop = tl.topk(gmax, TOPK, dim=1)
            gbest = tl.topk(tl.reshape(tl.join(gbest, gtop), [TQ, 2 * TOPK]), TOPK, dim=1)
        gid = tl.where(gbest != DEAD, (gbest & 0xFFFFFFFF) // G, -1)
        e = gid[:, :, None] * G + tl.arange(0, G)[None, None, :]
        e = tl.reshape(tl.where(gid[:, :, None] >= 0, e, 1 << 30), [TQ, TOPK * G])
        best = tl.topk(_comp(s_row, e, nlive, KEY_POS_INF, KEY_NEG_INF, DEAD), TOPK, dim=1)
    else:
        # Running top TOPK of each row as (key << 32 | block) int64, descending.
        best = tl.full([TQ, TOPK], DEAD, tl.int64)
        for c0 in range(0, n_max, CHUNK):
            top = tl.topk(
                _comp(s_row, c0 + col, nlive, KEY_POS_INF, KEY_NEG_INF, DEAD), TOPK, dim=1
            )
            best = tl.topk(tl.reshape(tl.join(best, top), [TQ, 2 * TOPK]), TOPK, dim=1)
    blk = tl.where(best != DEAD, (best & 0xFFFFFFFF).to(tl.int32), -1)
    j = tl.arange(0, TOPK)
    tl.store(
        topk_ptr + h.to(tl.int64) * stride_tk_h + row.to(tl.int64)[:, None] * stride_tk_t + j[None, :],
        blk,
        mask=r_ok[:, None],
    )

    if SORT_TOPK:
        # The group's page sort_topk, as msa_sort_topk_kernel builds it from the topk
        # just computed (candidate c = row c // TOPK, pick c % TOPK).
        C: tl.constexpr = TQ * TOPK
        ngrp = tl.num_programs(0)
        c = tl.arange(0, C)
        blk_c = tl.reshape(blk, [C])
        rowc = grp * TQ + c // TOPK
        rc_ok = rowc < num_rows
        req = tl.load(query_req_ptr + rowc, mask=rc_ok, other=0).to(tl.int64)
        pos = tl.load(query_pos_ptr + rowc, mask=rc_ok, other=-1)
        pos = tl.where(rc_ok, pos, -1)
        valid = rc_ok & (blk_c >= 0) & (blk_c * BLK <= pos)
        page = tl.load(
            block_table_ptr + req * stride_bt + tl.where(valid, blk_c, 0), mask=valid, other=-1
        )
        # Dedup by sorting (page, candidate) keys: a run of one page is
        # contiguous, its head starts a new sort_topk slot.
        BIG: tl.constexpr = 0x7FFFFFFFFFFFFFFF
        skey = tl.where(valid, page.to(tl.int64) * 256 + c, BIG)
        srt = tl.sort(skey)
        ok = srt != BIG
        pg = (srt >> 8).to(tl.int32)
        cc = tl.where(ok, (srt & 255).to(tl.int32), 0)
        prev = tl.gather(pg, tl.maximum(c - 1, 0), axis=0)
        first = ok & ((c == 0) | (pg != prev))
        slot = tl.cumsum(first.to(tl.int32), axis=0) - 1
        base = (h * ngrp + grp).to(tl.int64)
        tl.store(uniq_ptr + base * C + slot, pg, mask=first)
        tl.store(count_ptr + base, tl.sum(first.to(tl.int32), axis=0))
        # tokblk[slot, t]: -1, then each candidate's block at (its page's slot, its row)
        tb = tokblk_ptr + base * C * TQ
        tl.store(tb + tl.arange(0, C * TQ), tl.full([C * TQ], -1, tl.int32))
        tl.debug_barrier()
        blk_s = tl.gather(blk_c, cc, axis=0)
        tl.store(tb + slot * TQ + cc // TOPK, blk_s, mask=ok)
        if h == 0:
            tl.store(pos_ptr + rowc, pos, mask=rc_ok & (c % TOPK == 0))
