# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional

from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8
from aiter.ops.triton.utils.types import torch_to_triton_dtype

# Source:
# MoE Kernel adapted from VLLM

_PADDING_SIZE = 0

_MOE_A_QUANT_FUNC = dynamic_per_tensor_quant_fp8_i8

_USE_MOE_PERSISTENT_KERNEL = False


def moe_set_use_persistent_kernel(value: bool):
    global _USE_MOE_PERSISTENT_KERNEL
    _USE_MOE_PERSISTENT_KERNEL = value


def moe_set_padding_size(size: int):
    """
    Override padding size
    """
    global _PADDING_SIZE
    _PADDING_SIZE = size


def moe_set_quant_func(func):
    """
    Override 'A' matrix ie activations quantization function.
    Default function does dynamic quantization.
    """
    global _MOE_A_QUANT_FUNC
    _MOE_A_QUANT_FUNC = func


@triton.heuristics(
    {
        "GRID_MN": lambda args: triton.cdiv(args["EM"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"])
    }
)
@triton.jit
def e2e_moe_kernel(
    A,
    W1,
    W2,
    Out,
    A_scale,
    W1_scale,
    W2_scale,
    stride_am,
    stride_ak,
    stride_w1e,
    stride_w1n,
    stride_w1k,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_cm,
    stride_w1se,
    stride_w1sn,
    stride_w2se,
    stride_w2sk,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,  # original block_size_k
    BLOCK_SIZE_K2: tl.constexpr,  # outputs (EM, BLOCK_SIZE_K2)
    GROUP_SIZE_M: tl.constexpr,
    GRID_MN: tl.constexpr,
    atomic_num_stages: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - a: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - w1: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - w2: The stacked MOE weight tensor with shape (E, K, N // 2), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - c: The output cache tensor with shape (M, topk, K), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: a tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: a tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in a.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_w1e > 0)
    tl.assume(stride_w1n > 0)
    tl.assume(stride_w1k > 0)
    tl.assume(stride_w2e > 0)
    tl.assume(stride_w2n > 0)
    tl.assume(stride_w2k > 0)
    tl.assume(stride_cm > 0)
    if use_int8_w8a16:
        tl.assume(stride_w1se > 0)
        tl.assume(stride_w1sn > 0)
        tl.assume(stride_w2se > 0)
        tl.assume(stride_w2sk > 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    NUM_XCDS: tl.constexpr = 8

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K2)

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    # (i % 2): [0, 1, 0, 1, ...] (alternating)
    # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
    # So offs_w1n now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
    offs_w1n = (offs_half + (i % 2) * (N // 2)) % N

    mask_w1n = (pid_n * BLOCK_SIZE_N + i) < N

    a_ptrs = A + (
        offs_token[:, None] // top_k * stride_am + offs_k1[None, :] * stride_ak
    )
    w1_ptrs = (
        W1
        + off_experts * stride_w1e
        + (offs_k1[:, None] * stride_w1k + offs_w1n[None, :] * stride_w1n)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if use_int8_w8a16:
        w1_scale_ptrs = (
            W1_scale + off_experts * stride_w1se + offs_w1n[None, :] * stride_w1sn
        )
        w1_scale = tl.load(w1_scale_ptrs)

    if use_fp8_w8a8:
        a_scale = tl.load(A_scale)
        w1_scale = tl.load(W1_scale + off_experts)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K1)):
        # Masking ensures we don't load from invalid tokens or indices
        if EVEN_K:
            a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)
            w1 = tl.load(w1_ptrs, mask=mask_w1n[None, :], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=(token_mask[:, None] & (offs_k1[None, :] < K - k * BLOCK_SIZE_K1)),
                other=0.0,
            )
            w1 = tl.load(
                w1_ptrs,
                mask=(offs_k1[:, None] < K - k * BLOCK_SIZE_K1) & mask_w1n[None, :],
                other=0.0,
            )
        # w1 = tl.zeros((BLOCK_SIZE_K1, BLOCK_SIZE_N), dtype=dtype)

        if use_int8_w8a16:
            accumulator = tl.dot(a, w1.to(a.type), acc=accumulator)
        elif use_fp8_w8a8:
            accumulator += tl.dot(a, w1)
        else:
            accumulator = tl.dot(a, w1, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K1 * stride_ak
        w1_ptrs += BLOCK_SIZE_K1 * stride_w1k

    if use_int8_w8a16:
        accumulator = accumulator * w1_scale
    elif use_fp8_w8a8:
        accumulator = accumulator * a_scale * w1_scale

    silu_acc, mul_acc = accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    silu_acc = silu_acc / (1.0 + tl.exp2(-(silu_acc * 1.44269504089)))
    acc = (silu_acc * mul_acc).to(dtype)

    # TODO scale acc
    acc_scale = 1.0
    # TODO scale acc
    # -------------------------------

    offs_w2n = tl.arange(0, BLOCK_SIZE_N // 2) + pid_n * (BLOCK_SIZE_N // 2)

    w2_ptrs = (
        W2
        + off_experts * stride_w2e
        + (offs_k2[None, :] * stride_w2k + offs_w2n[:, None] * stride_w2n)
    )
    out_ptrs = Out + stride_cm * offs_token[:, None] + offs_k2[None, :]

    # if use_int8_w8a16:
    #     w2_scale_ptrs = W2_scale + off_experts * stride_w2se + offs_w2n[None, :]
    #     w2_scale = tl.load(w2_scale_ptrs)
    if use_fp8_w8a8:
        # acc_quantized, _, acc_scale = quantize_tensor_triton(acc, dtype=fp8_type)
        w2_scale = tl.load(W2_scale + off_experts)

    # minus if pid_m is even otherwise positive
    k_sign = (pid_m % 2) * 2 - 1
    num_k = tl.cdiv(K, BLOCK_SIZE_K2)
    for _k in tl.range(0, num_k, num_stages=atomic_num_stages):
        k = (num_k + (_k * k_sign)) % num_k
        k = ((k + pid_n * 4)) % num_k
        # k = _k

        if use_int8_w8a16:
            w2_scale_ptrs = (
                W2_scale
                + off_experts * stride_w2se
                + (offs_k2 + k * BLOCK_SIZE_K2)[None, :] * stride_w2sk
            )
            w2_scale = tl.load(w2_scale_ptrs)

        if EVEN_K:
            w2 = tl.load(
                w2_ptrs + k * BLOCK_SIZE_K2 * stride_w2k,
                mask=(offs_w2n[:, None] < N // 2),
                other=0.0,
            )
        else:
            w2 = tl.load(
                w2_ptrs + k * BLOCK_SIZE_K2 * stride_w2k,
                mask=(
                    (offs_w2n[:, None] < N // 2)
                    & ((offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K)
                ),
                other=0.0,
            )
        # w2 = tl.zeros((BLOCK_SIZE_HALF, BLOCK_SIZE_K2), dtype=dtype)

        if use_int8_w8a16:
            out = tl.dot(acc, w2.to(dtype))
        else:
            out = tl.dot(acc, w2)

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            out = out * moe_weight[:, None]

        if use_int8_w8a16:
            out = out * w2_scale
        elif use_fp8_w8a8:
            out = out * acc_scale * w2_scale

        # # atomic add
        if EVEN_K:
            c_mask = token_mask[:, None]
        else:
            c_mask = token_mask[:, None] & ((offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K)

        # TODO check scope
        tl.atomic_add(
            out_ptrs + k * BLOCK_SIZE_K2,
            out.to(dtype),
            mask=c_mask,
            sem="relaxed",
            scope="cta",
        )
        # tl.store(out_ptrs + k * BLOCK_SIZE_K2, out, mask=c_mask)


@triton.jit
def e2e_moe_persistent_kernel(
    A,
    W1,
    W2,
    intermediate_ptr,
    Out,
    A_scale,
    W1_scale,
    W2_scale,
    stride_am,
    stride_ak,
    stride_w1e,
    stride_w1n,
    stride_w1k,
    stride_w2e,
    stride_w2n,
    stride_w2k,
    stride_cm,
    stride_w1se,
    stride_w1sn,
    stride_w2se,
    stride_w2sk,
    stride_im,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,  # original block_size_k
    BLOCK_SIZE_K2: tl.constexpr,  # outputs (EM, BLOCK_SIZE_K2)
    NUM_SMS: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N1)
    num_pid_k: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K2)
    m_tile_per_sm = num_pid_m // NUM_SMS

    if start_m < num_pid_m % NUM_SMS:
        m_tile_per_sm += 1

    N_HALF: tl.constexpr = N // 2
    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N1 // 2

    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K2)
    offs_n1 = tl.arange(0, BLOCK_SIZE_N1)
    offs_n1_half = tl.arange(0, BLOCK_SIZE_HALF)
    offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    i = offs_n1.to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2

    dtype = Out.dtype.element_ty

    pid_m = start_m

    for _ in range(0, m_tile_per_sm):
        # pid_m = pid_m_start + m_off
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

        # Here we assume that valid tokens are in the range [0, M).
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m)
        # tl.device_print("pid_m", pid_m)
        # TODO mem fault when when pid_n != 0
        for pid_n in range(0, num_pid_n):
            offs_half = (pid_n * BLOCK_SIZE_HALF + i_floor) % N_HALF
            # (i % 2): [0, 1, 0, 1, ...] (alternating)
            # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
            # So offs_w1n now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
            offs_w1n = (offs_half + (i % 2) * (N_HALF)) % N

            mask_w1n = (pid_n * BLOCK_SIZE_N1 + i) < N

            a_ptrs = A + (
                offs_token[:, None] // top_k * stride_am + offs_k1[None, :] * stride_ak
            )
            w1_ptrs = (
                W1
                + off_experts * stride_w1e
                + (offs_k1[:, None] * stride_w1k + offs_w1n[None, :] * stride_w1n)
            )

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N1), dtype=tl.float32)

            if use_int8_w8a16:
                w1_scale_ptrs = (
                    W1_scale
                    + off_experts * stride_w1se
                    + offs_w1n[None, :] * stride_w1sn
                )
                w1_scale = tl.load(w1_scale_ptrs)
            if use_fp8_w8a8:
                a_scale = tl.load(A_scale)
                w1_scale = tl.load(W1_scale + off_experts)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K1)):
                # Masking ensures we don't load from invalid tokens or indices
                if EVEN_K:
                    a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)
                    # TODO memory fault N dim, might be k as well
                    w1 = tl.load(w1_ptrs, mask=mask_w1n[None, :], other=0.0)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=(
                            token_mask[:, None]
                            & (offs_k1[None, :] < K - k * BLOCK_SIZE_K1)
                        ),
                        other=0.0,
                    )
                    w1 = tl.load(
                        w1_ptrs,
                        mask=(offs_k1[:, None] < K - k * BLOCK_SIZE_K1)
                        & mask_w1n[None, :],
                        other=0.0,
                    )

                if use_int8_w8a16:
                    accumulator = tl.dot(a, w1.to(a.type), acc=accumulator)
                elif use_fp8_w8a8:
                    accumulator += tl.dot(a, w1)
                else:
                    accumulator = tl.dot(a, w1, acc=accumulator)
                a_ptrs += BLOCK_SIZE_K1 * stride_ak
                w1_ptrs += BLOCK_SIZE_K1 * stride_w1k

            if use_int8_w8a16:
                accumulator = accumulator * w1_scale
            elif use_fp8_w8a8:
                accumulator = accumulator * a_scale * w1_scale

            silu_acc, mul_acc = accumulator.reshape(
                BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2
            ).split()
            silu_acc = silu_acc / (1.0 + tl.exp2(-(silu_acc * 1.44269504089)))
            acc = (silu_acc * mul_acc).to(dtype)

            offs_in = pid_n * BLOCK_SIZE_HALF + offs_n1_half
            i_mask = token_mask[:, None] & (offs_in[None, :] < N_HALF)
            i_ptrs = (
                intermediate_ptr + stride_im * offs_token[:, None] + offs_in[None, :]
            )
            # TODO dtye??
            tl.atomic_add(i_ptrs, acc, mask=i_mask, sem="release")
            # TODO quantization

        for pid_k in range(0, num_pid_k):
            offs_w2k = (pid_k * BLOCK_SIZE_K2 + offs_k2) % K
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

            intermediate_ptrs = intermediate_ptr + (
                offs_token[:, None] * stride_im + offs_n2[None, :]
            )
            w2_ptrs = (
                W2
                + off_experts * stride_w2e
                + (offs_n2[:, None] * stride_w2n + offs_w2k[None, :] * stride_w2k)
            )

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K2), dtype=tl.float32)

            mask_w2k = (pid_k * BLOCK_SIZE_K2 + offs_k2) < K

            if use_int8_w8a16:
                w2_scale_ptrs = (
                    W2_scale
                    + off_experts * stride_w2se
                    + offs_k2[None, :] * stride_w2sk
                )
                w2_scale = tl.load(w2_scale_ptrs)

            if use_fp8_w8a8:
                # TODO calculate the intermediate scale and scale intermediate
                # a_scale = tl.load(A_scale)
                i_scale = 1
                w2_scale = tl.load(W2_scale + off_experts)

            for n in range(0, tl.cdiv(N_HALF, BLOCK_SIZE_N2)):
                # Masking ensures we don't load from invalid tokens or indices

                if EVEN_N:
                    intermediate = tl.load(
                        intermediate_ptrs, mask=(token_mask[:, None]), other=0.0
                    )
                    w2 = tl.load(w2_ptrs)
                else:
                    intermediate = tl.load(
                        intermediate_ptrs,
                        mask=(
                            token_mask[:, None]
                            & (offs_n2[None, :] < N_HALF - n * BLOCK_SIZE_N2)
                        ),
                        other=0.0,
                    )
                    w2 = tl.load(
                        w2_ptrs,
                        mask=(offs_n2[:, None] < N_HALF - n * BLOCK_SIZE_N2)
                        & mask_w2k[None, :],
                        other=0.0,
                    )

                if use_int8_w8a16:
                    accumulator = tl.dot(
                        intermediate.to(dtype), w2.to(dtype), acc=accumulator
                    )
                elif use_fp8_w8a8:
                    accumulator += tl.dot(intermediate, w2)
                else:
                    accumulator = tl.dot(intermediate.to(dtype), w2, acc=accumulator)
                intermediate_ptrs += BLOCK_SIZE_N2
                w2_ptrs += BLOCK_SIZE_N2 * stride_w2n

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(
                    topk_weights_ptr + offs_token, mask=token_mask, other=0
                )
                accumulator = accumulator * moe_weight[:, None]

            if use_int8_w8a16:
                accumulator = accumulator * w2_scale
            elif use_fp8_w8a8:
                accumulator = accumulator * i_scale * w2_scale

            offs_ck = pid_k * BLOCK_SIZE_K2 + offs_k2
            c_mask = token_mask[:, None] & (offs_ck[None, :] < K)
            out_ptrs = Out + stride_cm * offs_token[:, None] + offs_ck[None, :]
            tl.store(out_ptrs, accumulator.to(dtype), mask=c_mask)
        pid_m += NUM_SMS


def e2e_moe(
    A: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    Intermediate: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    W1_scale: Optional[torch.Tensor],
    W2_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    topk_ids,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    #TODO: Add doc
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    # if use_fp8_w8a8:
    #     assert W1_scale is not None
    #     assert W2_scale is not None
    #     if block_shape is None:
    #         output = torch.zeros(A.shape, device=A.device, dtype=torch.float8_e4m3fnuz)
    #         A_scale = torch.zeros(1, device=A.device, dtype=torch.float32)
    #         A, A_scale = _MOE_A_QUANT_FUNC(output, A, A_scale)
    #     else:
    #         #TODO: Add support for per token group quantization
    #         assert len(block_shape) == 2
    #         block_n, block_k = block_shape[0], block_shape[1]
    #         #A, A_scale = per_token_group_quant_fp8(A, block_k)
    #         assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    #         assert triton.cdiv(W1.shape[-2], block_n) == B_scale.shape[-2]
    #         assert triton.cdiv(W1.shape[-1], block_k) == B_scale.shape[-1]
    # elif use_int8_w8a16 or use_int4_w4a16:
    #     assert W1_scale is not None
    #     assert W2_scale is not None
    #     assert block_shape is None or block_shape[0] == 0
    # else:
    #     assert A_scale is None
    #     assert W1_scale is None
    #     assert W2_scale is None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    N = W1.shape[1]
    K = A.shape[1] - _PADDING_SIZE
    EVEN_K = K % config["BLOCK_SIZE_K1"] == 0

    if EM > 1024:
        atomic_num_stages = 2
    else:
        atomic_num_stages = 1

    stride_cm = C.stride(1)
    if _USE_MOE_PERSISTENT_KERNEL:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
        # TODO add N_split support to get more parallelism
        grid = lambda META: (  # noqa: E731
            min(NUM_SMS, triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])),
        )
        stride_im = Intermediate.stride(0)
        EVEN_N = (N // 2) % config["BLOCK_SIZE_N2"] == 0

        e2e_moe_persistent_kernel[grid](
            A,
            W1,
            W2,
            Intermediate,
            C,
            A_scale,
            W1_scale,
            W2_scale,
            A.stride(0),
            A.stride(1),
            W1.stride(0),
            W1.stride(1),
            W1.stride(2),
            W2.stride(0),
            W2.stride(2),
            W2.stride(1),
            stride_cm,
            W1_scale.stride(0) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
            W1_scale.stride(1) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
            W2_scale.stride(0) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
            W1_scale.stride(1) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
            stride_im,
            top_k,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_ids.numel(),
            EM,
            N,
            K,
            EVEN_K,
            EVEN_N,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            NUM_SMS=NUM_SMS,
            **config,
        )

        return C
    else:
        grid = lambda META: (  # noqa: E731
            triton.cdiv(EM, META["BLOCK_SIZE_M"])
            * triton.cdiv(W1.shape[1], META["BLOCK_SIZE_N"]),
        )
        dtype = C.dtype
        Out = C.to(torch.float32) if dtype == torch.bfloat16 else C

        e2e_moe_kernel[grid](
            A,
            W1,
            W2,
            Out,
            A_scale,
            W1_scale,
            W2_scale,
            A.stride(0),
            A.stride(1),
            W1.stride(0),
            W1.stride(1),
            W1.stride(2),
            W2.stride(0),
            W2.stride(2),
            W2.stride(1),
            stride_cm,
            W1_scale.stride(0) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
            W1_scale.stride(1) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
            W2_scale.stride(0) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
            W1_scale.stride(1) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
            top_k,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_ids.numel(),
            EM,
            N,
            K,
            EVEN_K,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            atomic_num_stages=atomic_num_stages,
            dtype=torch_to_triton_dtype[dtype],
            **config,
        )

    return Out.to(dtype)
