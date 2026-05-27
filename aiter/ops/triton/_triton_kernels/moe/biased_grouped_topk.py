# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


_biased_grouped_topk_repr = make_kernel_repr(
    "_biased_grouped_topk_kernel",
    [
        "BLOCK_M",
        "NUM_EXPERTS",
        "NUM_GROUPS",
        "TOPK_GROUP",
        "TOPK",
        "NEED_RENORM",
        "HAS_BIAS",
    ],
)


@triton.jit(repr=_biased_grouped_topk_repr)
def _biased_grouped_topk_kernel(
    logits_ptr,              # (M, NUM_EXPERTS)  any float dtype
    bias_ptr,                # (NUM_EXPERTS,)    same dtype as logits (or None when HAS_BIAS==0)
    topk_weights_ptr,        # (M, TOPK)         fp32 OUT
    topk_ids_ptr,            # (M, TOPK)         int32 OUT
    M,
    stride_logits_m,
    stride_logits_n,
    stride_topk_weights_m,
    stride_topk_weights_n,
    stride_topk_ids_m,
    stride_topk_ids_n,
    routed_scaling_factor,
    BLOCK_M: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    NEED_RENORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """DeepSeek-style biased grouped top-k.

    For each token:
      scores              = sigmoid(logits)                                   # raw, used for output weights
      scores_for_choice   = scores + bias                                     # used for expert selection
      group_score[g]      = top2(scores_for_choice[g*EPG : (g+1)*EPG]).sum()  # per-group score
      selected_groups     = top_TOPK_GROUP(group_score)
      masked_scores       = scores_for_choice  (only experts in selected_groups, else -inf)
      topk_ids            = top_TOPK(masked_scores)                           # expert indices
      topk_weights        = scores[topk_ids]                                  # raw sigmoid (not biased)
      if NEED_RENORM: topk_weights /= sum(topk_weights, axis=-1, keepdim=True)
      topk_weights       *= routed_scaling_factor
    """
    pid_m = tl.program_id(axis=0)
    EXPERTS_PER_GROUP: tl.constexpr = NUM_EXPERTS // NUM_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NUM_EXPERTS)
    mask_m = offs_m < M

    # Load logits and compute sigmoid (raw scores).
    logits_ptrs = (
        logits_ptr
        + offs_m[:, None].to(tl.int64) * stride_logits_m
        + offs_n[None, :].to(tl.int64) * stride_logits_n
    )
    logits = tl.load(
        logits_ptrs,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sigmoid(logits)  # (BLOCK_M, NUM_EXPERTS)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n).to(tl.float32)
        scores_for_choice = scores + bias[None, :]
    else:
        scores_for_choice = scores

    # ------------------------------------------------------------------
    # Per-group scores: sum of top-2 within each group of EXPERTS_PER_GROUP.
    # ------------------------------------------------------------------
    # Reshape to (BLOCK_M, NUM_GROUPS, EXPERTS_PER_GROUP) so that tl.max can
    # operate along the inner axis. We need top-2 sum; do it as:
    #   max1 = max(scores_in_group)
    #   max2 = max(scores_in_group with max1 masked to -inf)
    #   group_score = max1 + max2
    sfc_3d = tl.reshape(
        scores_for_choice, (BLOCK_M, NUM_GROUPS, EXPERTS_PER_GROUP)
    )
    max1 = tl.max(sfc_3d, axis=2)                                # (BLOCK_M, NUM_GROUPS)
    # Mask the position of max1 within each group, then take max again.
    NEG_INF: tl.constexpr = float("-inf")
    is_max1 = sfc_3d == max1[:, :, None]
    sfc_minus_max = tl.where(is_max1, NEG_INF, sfc_3d)
    max2 = tl.max(sfc_minus_max, axis=2)                         # (BLOCK_M, NUM_GROUPS)
    group_scores = max1 + max2                                   # (BLOCK_M, NUM_GROUPS)

    # ------------------------------------------------------------------
    # Pick TOPK_GROUP groups per token (no need for sorted).
    # Iterate TOPK_GROUP times, each time taking argmax and masking it out.
    # NUM_GROUPS is small (8 for DSR1) so this loop is cheap.
    # ------------------------------------------------------------------
    group_mask = tl.zeros((BLOCK_M, NUM_GROUPS), dtype=tl.int32)
    gs_work = group_scores
    for i in tl.static_range(TOPK_GROUP):
        idx = tl.argmax(gs_work, axis=1, tie_break_left=True)    # (BLOCK_M,)
        sel = tl.arange(0, NUM_GROUPS)[None, :] == idx[:, None]  # (BLOCK_M, NUM_GROUPS)
        group_mask = tl.where(sel, 1, group_mask)
        gs_work = tl.where(sel, NEG_INF, gs_work)

    # Expand group mask to expert mask.
    expert_mask_3d = tl.broadcast_to(
        group_mask[:, :, None], (BLOCK_M, NUM_GROUPS, EXPERTS_PER_GROUP)
    )
    expert_mask = tl.reshape(expert_mask_3d, (BLOCK_M, NUM_EXPERTS))
    masked_scores = tl.where(expert_mask == 1, scores_for_choice, NEG_INF)

    # ------------------------------------------------------------------
    # Pick TOPK experts from the masked scores.
    # ------------------------------------------------------------------
    topk_ids_acc = tl.zeros((BLOCK_M, TOPK), dtype=tl.int32)
    topk_w_acc = tl.zeros((BLOCK_M, TOPK), dtype=tl.float32)
    ms_work = masked_scores
    for i in tl.static_range(TOPK):
        idx = tl.argmax(ms_work, axis=1, tie_break_left=True)    # (BLOCK_M,)
        # Gather raw sigmoid score at idx (NOT the biased one).
        sel = tl.arange(0, NUM_EXPERTS)[None, :] == idx[:, None]
        raw_w = tl.sum(tl.where(sel, scores, 0.0), axis=1)       # (BLOCK_M,)
        # Splat into column i of accumulators using a one-hot over TOPK.
        col_sel = tl.arange(0, TOPK)[None, :] == i               # (1, TOPK)
        topk_ids_acc = tl.where(col_sel, idx[:, None], topk_ids_acc)
        topk_w_acc = tl.where(col_sel, raw_w[:, None], topk_w_acc)
        ms_work = tl.where(sel, NEG_INF, ms_work)

    # ------------------------------------------------------------------
    # Renormalize + scale.
    # ------------------------------------------------------------------
    if NEED_RENORM:
        denom = tl.sum(topk_w_acc, axis=1)                       # (BLOCK_M,)
        # Avoid div-by-zero for rows that fall outside [0, M).
        denom = tl.where(denom > 0.0, denom, 1.0)
        topk_w_acc = topk_w_acc / denom[:, None]

    topk_w_acc = topk_w_acc * routed_scaling_factor

    # ------------------------------------------------------------------
    # Store.
    # ------------------------------------------------------------------
    offs_k = tl.arange(0, TOPK)
    weights_ptrs = (
        topk_weights_ptr
        + offs_m[:, None].to(tl.int64) * stride_topk_weights_m
        + offs_k[None, :].to(tl.int64) * stride_topk_weights_n
    )
    ids_ptrs = (
        topk_ids_ptr
        + offs_m[:, None].to(tl.int64) * stride_topk_ids_m
        + offs_k[None, :].to(tl.int64) * stride_topk_ids_n
    )
    tl.store(weights_ptrs, topk_w_acc, mask=mask_m[:, None])
    tl.store(ids_ptrs, topk_ids_acc, mask=mask_m[:, None])
