# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.moe.biased_grouped_topk import (
    _biased_grouped_topk_kernel,
)

_LOGGER = AiterTritonLogger()


def biased_grouped_topk_triton(
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
    block_m: int = 16,
    num_warps: int = 4,
) -> None:
    """In-place Triton implementation of DeepSeek-style biased grouped top-k routing.

    This mirrors ``aiter.biased_grouped_topk`` (the HIP/ASM dispatcher in
    ``aiter/ops/topk.py``) and the torch reference ``biased_grouped_topk_torch``:

      scores              = sigmoid(gating_output)
      scores_for_choice   = scores + correction_bias
      group_score[g]      = top2(scores_for_choice in group g).sum()
      pick top ``topk_group`` groups, mask out experts outside them
      topk_ids            = top ``topk`` experts by scores_for_choice (masked)
      topk_weights        = scores[topk_ids]     # raw sigmoid, NOT biased
      optional renormalize, then multiply by routed_scaling_factor.

    Args:
        gating_output: (M, num_experts) routing logits. Any float dtype.
        correction_bias: (num_experts,) additive bias used only for expert
            selection. Pass ``None`` to disable (plain grouped_topk).
        topk_weights: (M, topk) fp32 output buffer.
        topk_ids: (M, topk) int32 output buffer.
        num_expert_group: number of expert groups (must divide num_experts).
        topk_group: number of groups to pick per token.
        need_renorm: if True, normalize selected weights to sum to 1.
        routed_scaling_factor: final multiplier applied to weights.
        block_m: rows per program. Default 16.
        num_warps: warps per program. Default 4.

    Returns:
        None. Writes into ``topk_weights`` and ``topk_ids``.
    """
    _LOGGER.info(
        f"BIASED_GROUPED_TOPK: gating={tuple(gating_output.shape)} "
        f"num_groups={num_expert_group} topk_group={topk_group} "
        f"topk={topk_weights.shape[-1]} renorm={need_renorm}"
    )

    assert gating_output.is_cuda
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == torch.int32
    assert topk_weights.shape == topk_ids.shape
    assert topk_weights.shape[0] == gating_output.shape[0]

    M, num_experts = gating_output.shape
    topk = topk_ids.shape[-1]
    assert num_experts % num_expert_group == 0
    assert topk_group <= num_expert_group
    assert topk <= (num_experts // num_expert_group) * topk_group, (
        "top-k must fit within the experts selectable from chosen groups"
    )

    has_bias = correction_bias is not None
    if has_bias:
        assert correction_bias.shape == (num_experts,)
        bias_ptr = correction_bias
    else:
        # Pass a 1-element placeholder; HAS_BIAS=0 guards all reads.
        bias_ptr = gating_output

    grid = (triton.cdiv(M, block_m),)
    _biased_grouped_topk_kernel[grid](
        gating_output,
        bias_ptr,
        topk_weights,
        topk_ids,
        M,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        routed_scaling_factor,
        BLOCK_M=block_m,
        NUM_EXPERTS=num_experts,
        NUM_GROUPS=num_expert_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        NEED_RENORM=need_renorm,
        HAS_BIAS=has_bias,
        num_warps=num_warps,
        num_stages=1,
    )
