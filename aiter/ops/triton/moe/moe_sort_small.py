# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_sort_small import (
    _moe_sort_small_kernel,
)

_EXPERTS_PER_CTA = 16
_TOKENS_PER_CTA = 8
_BUF_BLOCK = 4096


def moe_sort_small(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> None:
    """Single-launch equivalent of ``moe_sorting_opus_fwd`` for small batches.

    Writes the same sorted_ids / sorted_weights / sorted_expert_ids /
    num_valid_ids as the Opus sort and zeroes all of ``moe_buf``, which must be
    contiguous. Every CTA rebuilds the expert histogram, so cost grows with
    ``topk_ids.numel()``; callers gate it to decode-sized batches.
    """
    m, topk = topk_ids.shape
    num_tok_cta = triton.cdiv(m, _TOKENS_PER_CTA)
    num_buf_cta = triton.cdiv(moe_buf.numel(), _BUF_BLOCK)
    grid = (num_tok_cta + num_experts // _EXPERTS_PER_CTA + num_buf_cta,)
    _moe_sort_small_kernel[grid](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        moe_buf.numel(),
        m,
        num_tok_cta,
        TOPK=topk,
        E=num_experts,
        BM=block_size,
        TB=_TOKENS_PER_CTA,
        E_PER=_EXPERTS_PER_CTA,
        HCHUNK=512 if m * topk <= 2048 else 1024,
        BUF_BLOCK=_BUF_BLOCK,
        num_warps=8,
    )
