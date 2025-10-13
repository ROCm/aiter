# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.moe_align_block_size import (
    _moe_align_block_size_stage1_kernel,
    _moe_align_block_size_stage2_kernel,
    _moe_align_block_size_stage3_kernel,
    _moe_align_block_size_stage4_kernel,
)

_LOGGER = AiterTritonLogger()


def ceil_div(a, b):
    return (a + b - 1) // b


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,  # [num_tkns, num_experts]
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    _LOGGER.info(
        f"MOE_ALIGN_BLOCK_SIZE_TRITON:  topk_ids={tuple(topk_ids.shape)} num_experts={num_experts}  sorted_token_ids={tuple(sorted_token_ids.shape)} "
        + "block_size={block_size} expert_ids={tuple(expert_ids.shape)} num_tokens_post_pad={tuple(num_tokens_post_pad.shape)}"
    )
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    _moe_align_block_size_stage1_kernel[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )

    _moe_align_block_size_stage2_kernel[grid](
        tokens_cnts,
        num_experts,
    )

    _moe_align_block_size_stage3_kernel[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )

    _moe_align_block_size_stage4_kernel[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )
