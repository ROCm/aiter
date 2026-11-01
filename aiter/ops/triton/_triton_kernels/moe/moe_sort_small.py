# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_moe_sort_small_repr = make_kernel_repr(
    "_moe_sort_small_kernel",
    ["TOPK", "E", "BM", "TB", "E_PER", "HCHUNK", "BUF_BLOCK"],
)


@triton.jit(
    repr=_moe_sort_small_repr,
    do_not_specialize=["M", "num_tok_cta", "moe_buf_numel"],
)
def _moe_sort_small_kernel(
    topk_ids_ptr,
    topk_w_ptr,
    sorted_ids_ptr,
    sorted_w_ptr,
    sorted_eids_ptr,
    num_valid_ptr,
    moe_buf_ptr,
    moe_buf_numel,
    M,
    num_tok_cta,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    TB: tl.constexpr,
    E_PER: tl.constexpr,
    HCHUNK: tl.constexpr,
    BUF_BLOCK: tl.constexpr,
):
    # Three CTA roles: token CTAs place their (token, slot) pairs, expert CTAs
    # write block expert ids and block-tail padding, zero CTAs clear moe_buf.
    # Each CTA recomputes the expert histogram, so no CTA waits on another.
    pid = tl.program_id(0)
    NUM_E_CTA: tl.constexpr = E // E_PER
    if pid >= num_tok_cta + NUM_E_CTA:
        offs = (pid - num_tok_cta - NUM_E_CTA) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    is_tok = pid < num_tok_cta
    P = M * TOPK
    p0 = tl.where(is_tok, pid * TB * TOPK, 0)
    offs_e = tl.arange(0, E)
    hist = tl.zeros((E,), tl.int32)
    before = tl.zeros((E,), tl.int32)
    for c in range(0, P, HCHUNK):
        offs = c + tl.arange(0, HCHUNK)
        m = offs < P
        e = tl.load(topk_ids_ptr + offs, mask=m, other=0)
        h = tl.histogram(e, E, mask=m)
        hist += h
        if c + HCHUNK <= p0:
            before += h
        elif c < p0:
            before += tl.histogram(e, E, mask=m & (offs < p0))
    blocks = (hist + BM - 1) // BM
    bb = tl.cumsum(blocks, 0) - blocks

    if is_tok:
        offs_q = p0 + tl.arange(0, TB * TOPK)
        mq = offs_q < P
        eq = tl.load(topk_ids_ptr + offs_q, mask=mq, other=E)
        wq = tl.load(topk_w_ptr + offs_q, mask=mq, other=0.0)
        # Within an expert, pairs keep their flat (token, slot) order.
        same = (eq[:, None] == eq[None, :]) & (offs_q[None, :] < offs_q[:, None])
        start = bb * BM + before
        dest = tl.sum(
            tl.where(eq[:, None] == offs_e[None, :], start[None, :], 0), axis=1
        ) + tl.sum(same.to(tl.int32), axis=1)
        tl.store(
            sorted_ids_ptr + dest, ((offs_q % TOPK) << 24) | (offs_q // TOPK), mask=mq
        )
        tl.store(sorted_w_ptr + dest, wq, mask=mq)
    else:
        offs_x = (pid - num_tok_cta) * E_PER + tl.arange(0, E_PER)
        sel = offs_x[:, None] == offs_e[None, :]
        cnt_x = tl.sum(tl.where(sel, hist[None, :], 0), axis=1)
        blocks_x = tl.sum(tl.where(sel, blocks[None, :], 0), axis=1)
        bb_x = tl.sum(tl.where(sel, bb[None, :], 0), axis=1)
        for j in range(0, tl.max(blocks_x, axis=0)):
            tl.store(sorted_eids_ptr + bb_x + j, offs_x, mask=j < blocks_x)
        offs_b = tl.arange(0, BM)
        pos = (bb_x * BM + cnt_x)[:, None] + offs_b[None, :]
        pad = (cnt_x[:, None] + offs_b[None, :]) < (blocks_x * BM)[:, None]
        pad_id = (TOPK << 24) | M
        tl.store(
            sorted_ids_ptr + pos, tl.zeros((E_PER, BM), tl.int32) + pad_id, mask=pad
        )
        tl.store(sorted_w_ptr + pos, tl.zeros((E_PER, BM), tl.float32), mask=pad)
        if pid == num_tok_cta:
            num_valid = tl.sum(blocks, axis=0) * BM
            two = tl.arange(0, 2)
            tl.store(num_valid_ptr + two, tl.where(two == 0, num_valid, M))
