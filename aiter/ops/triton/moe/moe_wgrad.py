# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils.device_info import get_num_xcds


def _get_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=2,
        ),
    ]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N", "K"],
)
@triton.jit
def _moe_wgrad_kernel(
    grad_ptr,
    input_ptr,
    dw_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    stride_gm,
    stride_gn,
    stride_im,
    stride_ik,
    stride_dwe,
    stride_dwn,
    stride_dwk,
    num_sorted_tokens,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Weight gradient kernel for MoE.

    Computes dW[e] = sum_{tokens t assigned to expert e} grad[t].T @ input[t]

    BLOCK_SIZE_M must match the block_size used in moe_align_block_size to
    ensure expert_ids[pid_m] correctly describes all tokens in the block.
    """
    pid_m = tl.program_id(0)
    pid_nk = tl.program_id(1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    total_nk = num_pid_n * num_pid_k
    if pid_nk >= total_nk:
        return

    pid_nk_remapped = remap_xcd(pid_nk, total_nk, NUM_XCDS)
    pid_n = pid_nk_remapped // num_pid_k
    pid_k = pid_nk_remapped % num_pid_k

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n_mask = offs_n < N
    k_mask = offs_k < K

    grad_ptrs = (
        grad_ptr
        + (offs_token // top_k)[:, None] * stride_gm
        + offs_n[None, :] * stride_gn
    )
    grad_block = tl.load(
        grad_ptrs,
        mask=token_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    input_ptrs = (
        input_ptr
        + (offs_token // top_k)[:, None] * stride_im
        + offs_k[None, :] * stride_ik
    )
    input_block = tl.load(
        input_ptrs,
        mask=token_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    # grad.T @ input = [BLOCK_N, BLOCK_M] @ [BLOCK_M, BLOCK_K] = [BLOCK_N, BLOCK_K]
    acc = tl.dot(tl.trans(grad_block), input_block)

    dw_ptrs = (
        dw_ptr
        + off_expert * stride_dwe
        + offs_n[:, None].to(tl.int64) * stride_dwn
        + offs_k[None, :].to(tl.int64) * stride_dwk
    )
    dw_mask = n_mask[:, None] & k_mask[None, :]
    tl.atomic_add(dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=dw_mask, sem="relaxed")


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

    Uses @triton.autotune to search optimal (BLOCK_SIZE_N, BLOCK_SIZE_K)
    on first launch, keyed on (N, K) dimensions.

    Args:
        grad: Gradient tensor [num_tokens, N].
        input: Input activation tensor [num_tokens, K].
        sorted_token_ids: From moe_align_block_size.
        expert_ids: From moe_align_block_size.
        num_tokens_post_padded: From moe_align_block_size.
        num_experts: Number of experts.
        top_k: Number of experts per token.
        weight_shape: Shape of weight tensor (E, N, K).
        block_size_m: Must match the block_size used in moe_align_block_size.

    Returns:
        dW tensor with shape weight_shape.
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
