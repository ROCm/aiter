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


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


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
    extra_zero_buf: torch.Tensor | None = None,
) -> None:
    """Single-launch equivalent of ``moe_sorting_opus_fwd`` for small batches.

    Writes the same sorted_ids / sorted_weights / sorted_expert_ids /
    num_valid_ids as the Opus sort and zeroes all of ``moe_buf`` and, if given,
    ``extra_zero_buf``; both must be contiguous. Every CTA rebuilds the expert
    histogram, so cost grows with ``topk_ids.numel()``; callers gate it to
    decode-sized batches.
    """
    m, topk = topk_ids.shape
    # Expert CTAs each own _EXPERTS_PER_CTA experts and the first one writes
    # num_valid_ids, so the experts must tile into at least one expert CTA.
    if num_experts < _EXPERTS_PER_CTA or not _is_pow2(num_experts):
        raise ValueError(
            f"moe_sort_small needs a power-of-two num_experts >= "
            f"{_EXPERTS_PER_CTA}, got {num_experts}"
        )
    if not _is_pow2(block_size):
        raise ValueError(
            f"moe_sort_small needs power-of-two block_size, got {block_size}"
        )
    if not _is_pow2(_TOKENS_PER_CTA * topk):
        raise ValueError(f"moe_sort_small needs power-of-two topk, got {topk}")
    if m == 0:
        raise ValueError("moe_sort_small needs at least one token")
    if not moe_buf.is_contiguous() or (
        extra_zero_buf is not None and not extra_zero_buf.is_contiguous()
    ):
        raise ValueError("moe_sort_small needs contiguous zero buffers")

    if extra_zero_buf is None:
        extra_zero_buf = moe_buf[:0]
    num_tok_cta = triton.cdiv(m, _TOKENS_PER_CTA)
    num_buf_cta = triton.cdiv(moe_buf.numel(), _BUF_BLOCK)
    num_extra_cta = triton.cdiv(extra_zero_buf.numel(), _BUF_BLOCK)
    grid = (
        num_tok_cta + num_experts // _EXPERTS_PER_CTA + num_buf_cta + num_extra_cta,
    )
    _moe_sort_small_kernel[grid](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        moe_buf.numel(),
        num_buf_cta,
        extra_zero_buf,
        extra_zero_buf.numel(),
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
