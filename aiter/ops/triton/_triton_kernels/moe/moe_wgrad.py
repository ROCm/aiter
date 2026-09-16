# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for MoE weight-gradient computation.

Computes dW[e] = sum_{tokens t assigned to expert e} grad[t].T @ input[t]
using sorted token infrastructure from moe_align_block_size.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd


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
    # dW is accumulated via tl.atomic_add; reset it to zero before each
    # benchmark trial so timing runs don't corrupt the final result.
    reset_to_zero=["dw_ptr"],
)
# NOTE: repr= is intentionally omitted here. Combining @triton.autotune with
# @triton.jit(repr=...) corrupts the kernel execution on current Triton versions.
# The autotune key ["N", "K"] already encodes the compiled variant identity.
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
