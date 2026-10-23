# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_softmax_over_topk_bwd_repr = make_kernel_repr(
    "sonicmoe_softmax_over_topk_bwd",
    ["K", "BLOCK_K", "dlogits_is_none"],
)
_topk_over_softmax_bwd_repr = make_kernel_repr(
    "sonicmoe_topk_over_softmax_bwd",
    ["E", "K", "BLOCK_E", "BLOCK_K", "norm_topk_probs"],
)


@triton.jit(repr=_softmax_over_topk_bwd_repr)
def _softmax_over_topk_bwd_kernel(
    dlogits_ptr,
    dlogits_full_ptr,
    score_ptr,
    dscore_ptr,
    idx_ptr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    dlogits_is_none: tl.constexpr,
):
    row = tl.program_id(axis=0)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0
    ).to(tl.int32)
    s_sel = tl.load(
        score_ptr + row * stride_sm + k_offs * stride_sn, mask=k_mask, other=0
    ).to(tl.float32)
    g_sel = tl.load(
        dscore_ptr + row * stride_gm + k_offs * stride_gk, mask=k_mask, other=0
    ).to(tl.float32)

    dot = tl.sum(g_sel * s_sel, axis=0)
    add_vals = s_sel * (g_sel - dot)

    indices = row * stride_dm + idx * stride_dn
    if not dlogits_is_none:
        add_vals += tl.load(dlogits_ptr + indices, mask=k_mask)
    tl.store(dlogits_full_ptr + indices, add_vals, mask=k_mask)


@triton.jit(repr=_topk_over_softmax_bwd_repr)
def _topk_over_softmax_bwd_kernel(
    logits_ptr,
    dlogits_ptr,
    dscore_ptr,
    idx_ptr,
    score_ptr,
    stride_lm: tl.constexpr,
    stride_le: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_scm: tl.constexpr,
    stride_scn: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    norm_topk_probs: tl.constexpr,
):
    row = tl.program_id(axis=0)

    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    logits = tl.load(
        logits_ptr + row * stride_lm + e_offs * stride_le,
        mask=e_mask,
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    exp_vals = tl.exp(logits - row_max)
    row_sum = tl.sum(exp_vals, axis=0)
    p = exp_vals / row_sum

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0
    ).to(tl.int32)
    g_sel = tl.load(
        dscore_ptr + row * stride_sm + k_offs * stride_sn, mask=k_mask, other=0
    ).to(tl.float32)

    sel_logits = tl.load(
        logits_ptr + row * stride_lm + idx * stride_le, mask=k_mask, other=-float("inf")
    ).to(tl.float32)
    p_sel = tl.exp(sel_logits - row_max) / row_sum

    if norm_topk_probs:
        scores = tl.load(
            score_ptr + row * stride_scm + k_offs * stride_scn, mask=k_mask, other=0
        ).to(tl.float32)
        dot_s = tl.sum(g_sel * scores, axis=0)
        S = tl.sum(p_sel, axis=0)
        dp_sel = (g_sel - dot_s) / S
    else:
        dp_sel = g_sel

    dot = tl.sum(dp_sel * p_sel, axis=0)

    dp = tl.zeros([BLOCK_E], dtype=tl.float32)
    for k_iter in tl.static_range(K):
        cur_dp = tl.sum(tl.where(k_offs == k_iter, dp_sel, 0.0))
        cur_idx = tl.sum(tl.where(k_offs == k_iter, idx, 0))
        dp = tl.where(e_offs == cur_idx, cur_dp, dp)

    dlogits = p * (dp - dot)
    tl.store(dlogits_ptr + row * stride_dm + e_offs * stride_dn, dlogits, mask=e_mask)
