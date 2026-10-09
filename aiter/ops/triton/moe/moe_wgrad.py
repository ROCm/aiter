# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE weight-gradient wrapper (group_sizes interface)."""

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_wgrad import _moe_wgrad_kernel
from aiter.ops.triton.utils.device_info import get_num_xcds

__all__ = ["moe_wgrad"]


def moe_wgrad(
    grad: torch.Tensor,
    input: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_experts: int,
    top_k: int,
    weight_shape: tuple,
    block_size_m: int = 64,
) -> torch.Tensor:
    """Compute MoE weight gradients using sorted token infrastructure.

    Uses ``@triton.autotune`` to search optimal ``(BLOCK_SIZE_N, BLOCK_SIZE_K)``
    on first launch, keyed on ``(N, K)`` dimensions.

    Args:
        grad: Gradient tensor ``[num_tokens, N]``.  When ``top_k > 1`` this
            tensor must already incorporate per-route gating weights; the kernel
            accumulates it as-is without applying gating internally.
        input: Input activation tensor ``[num_tokens, K]``.
        sorted_token_ids: From ``moe_align_block_size``.  Padding slots must
            contain a value >= ``num_valid_tokens`` (i.e. >= ``num_tokens *
            top_k``) so the token mask treats them as invalid; the kernel relies
            on this sentinel rather than reading ``token_nums`` per block.
        expert_ids: From ``moe_align_block_size``.
        num_tokens_post_padded: From ``moe_align_block_size``.
        num_experts: Number of experts.
        top_k: Number of experts per token.
        weight_shape: Shape of weight tensor ``(E, N, K)``.
        block_size_m: Must match the ``block_size`` used in
            ``moe_align_block_size``.

    Returns:
        ``dW`` tensor with shape ``weight_shape``.
    """
    _E, N, K = weight_shape

    dW = torch.zeros(weight_shape, dtype=grad.dtype, device=grad.device)

    num_sorted = sorted_token_ids.shape[0]
    if num_sorted == 0:
        return dW

    grid = lambda META: (
        num_sorted // block_size_m,
        triton.cdiv(N, META["BLOCK_SIZE_N"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    _moe_wgrad_kernel[grid](
        grad,
        input,
        dW,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        grad.shape[0] * top_k,
        grad.stride(0),
        grad.stride(1),
        input.stride(0),
        input.stride(1),
        dW.stride(0),
        dW.stride(1),
        dW.stride(2),
        num_sorted,
        top_k=top_k,
        BLOCK_SIZE_M=block_size_m,
        NUM_XCDS=get_num_xcds(),
    )
    return dW
