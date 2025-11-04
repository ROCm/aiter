# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from .activation import _silu_exp2

from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd


@triton.jit
def group_broadcast(
    x,
    xM: tl.constexpr,
    xN: tl.constexpr,
    group_size: tl.constexpr,
    broadcast_dim: tl.constexpr,
):
    """
    Broadcasts the input tensor `x` along the specified dimension `broadcast_dim`
    in groups of size `group_size`.

    Parameters:
    - x: Input tensor to be broadcasted.
    - group_size: Size of each group for broadcasting.
    - broadcast_dim: Dimension along which to perform the broadcasting.

    Returns:
    - A tensor with the same shape as `x`, but with values broadcasted
      in groups along the specified dimension.
    """
    if broadcast_dim == 0:
        if xM > 1:
            x = x.reshape(xM, 1, xN)
            x = tl.broadcast_to(x, (xM, group_size, xN))
            x = x.reshape(xM * group_size, xN)
        # else: singleton dimension, no need to broadcast
    else:
        if xN > 1:
            x = x.reshape(xM, xN, 1)
            x = tl.broadcast_to(x, (xM, xN, group_size))
            x = x.reshape(xM, xN * group_size)

    return x


# Source:
# MoE Kernel adapted from VLLM
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
    Intermediate,
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
    stride_om,
    stride_ok,
    stride_im,
    stride_asm,
    stride_ask,
    stride_w1se,
    stride_w1sn,
    stride_w1sk,
    stride_w2se,
    stride_w2sk,
    stride_w2sn,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_K2: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_block_scale: tl.constexpr, 
    use_fp8_w8a8: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GRID_MN: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    SKINNY: tl.constexpr,
    dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    return_intermediate: tl.constexpr = False,
    PER_TOKEN_QUANT_A: tl.constexpr = False,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (M, K).
    - W1: The first layer weight tensor with shape (E, N, K).
    - W2: The second layer weight tensor with shape (E, K, N // 2).
        N // 2 represents the size of the intermediate token after the gated activation.
    - Out: The output tensor with shape (M, topk, K).
    - sorted_token_ids: a tensor with shape (max_num_tokens_padded). Contains ids for the top k repeated tokens + delimiter tokens + padding tokens.
        Delimiter tokens are for aligning to BLOCK_SIZE_M loading, and padding tokens to pad to static size
        max_num_tokens_padded=topk * M + E * (BLOCK_SIZE_M – 1). Static size needed for cuda graph.
        It assumes the worst case where each expert will occupy multiple full rows
        and exactly one row that only has one non-padding value.
    - expert_ids: a tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in a. length is ceiling division of max_num_tokens_padded and BLOCK_SIZE_M.

    Sizes:
    - M: number of tokens
    - E: number of experts
    - K: hidden size
    - N: moe intermediate size
    - topk: number of experts the token is routed to
    """
    # TODO:
    # add gelu support
    # add int8, int4, fp4 support
    # consider non pow2 N (e.g. 768)
    # extensive tuning

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_w1e > 0)
    tl.assume(stride_w1n > 0)
    tl.assume(stride_w1k > 0)
    tl.assume(stride_w2e > 0)
    tl.assume(stride_w2n > 0)
    tl.assume(stride_w2k > 0)
    tl.assume(stride_om > 0)
    if use_fp8_w8a8:
        tl.assume(stride_w1se > 0)
        tl.assume(stride_w1sn > 0)
        tl.assume(stride_w1sk > 0)
        tl.assume(stride_w2se > 0)
        tl.assume(stride_w2sk > 0)
        tl.assume(stride_w2sn > 0)
        tl.assume(stride_asm > 0)
        tl.assume(stride_ask > 0)

    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K2)

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2

    # how many scaling factors along block dimensions. Needed for broadcasting
    if use_fp8_w8a8 and use_block_scale:
        num_scales_along_n: tl.constexpr = (BLOCK_SIZE_HALF + group_n - 1) // group_n
        num_scales_along_k2: tl.constexpr = (BLOCK_SIZE_K2 + group_k - 1) // group_k
        num_scales_along_k1: tl.constexpr = (BLOCK_SIZE_K1 + group_k - 1) // group_k
        tl.static_assert(num_scales_along_k1 == 1, "BLOCK_SIZE_K1 must be <= group_k")
        if PER_TOKEN_QUANT_A:
            tl.static_assert(
                group_k == K,
                "per-token quantization requires group k to be K",
            )

    offs_i0 = tl.arange(0, BLOCK_SIZE_HALF).to(tl.int64)
    offs_i1 = (tl.arange(0, BLOCK_SIZE_HALF) + N // 2).to(tl.int64)
    # offset for silu_acc
    i0 = (pid_n * BLOCK_SIZE_HALF + offs_i0) % N
    # offset for mul_acc
    i1 = (pid_n * BLOCK_SIZE_HALF + offs_i1) % N

    a_ptrs = A + (
        offs_token[:, None] // top_k * stride_am + offs_k1[None, :] * stride_ak
    )
    w1_ptrs_i0 = (
        W1
        + off_experts * stride_w1e
        + (offs_k1[:, None] * stride_w1k + i0[None, :] * stride_w1n)
    )

    w1_ptrs_i1 = (
        W1
        + off_experts * stride_w1e
        + (offs_k1[:, None] * stride_w1k + i1[None, :] * stride_w1n)
    )

    silu_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_HALF), dtype=tl.float32)
    mul_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_HALF), dtype=tl.float32)

    if use_fp8_w8a8 and use_block_scale:
        a_scale_ptrs = A_scale + (offs_token[:, None] // top_k * stride_asm)
        i0s = pid_n * (BLOCK_SIZE_HALF // group_n) + tl.arange(0, num_scales_along_n)
        i1s = i0s + ((N // 2) // group_n)
        w1_i0_scale_ptrs = (
            W1_scale + off_experts * stride_w1se + i0s[None, :] * stride_w1sn
        )
        w1_i1_scale_ptrs = (
            W1_scale + off_experts * stride_w1se + i1s[None, :] * stride_w1sn
        )

    if use_fp8_w8a8 and not use_block_scale:
        w1_scale = tl.load(W1_scale + off_experts)
        if PER_TOKEN_QUANT_A:
            a_scale = tl.load(A_scale + (offs_token[:, None] // top_k * stride_asm))
        else:
            a_scale = tl.load(A_scale)

    num_k1 = tl.cdiv(K, BLOCK_SIZE_K1)
    for k1 in tl.range(0, num_k1):
        a_ptrs_k = a_ptrs + k1 * BLOCK_SIZE_K1 * stride_ak
        w1_ptrs_i0_k = w1_ptrs_i0 + k1 * BLOCK_SIZE_K1 * stride_w1k
        w1_ptrs_i1_k = w1_ptrs_i1 + k1 * BLOCK_SIZE_K1 * stride_w1k
        if use_fp8_w8a8 and use_block_scale:
            w1_i0_scale_ptrs_k = w1_i0_scale_ptrs + k1 * BLOCK_SIZE_K1 // group_k * stride_w1sk
            w1_i1_scale_ptrs_k = w1_i1_scale_ptrs + k1 * BLOCK_SIZE_K1 // group_k * stride_w1sk
            a_scale_ptrs_k = a_scale_ptrs + k1 * BLOCK_SIZE_K1 // group_k * stride_ask

        # pipeline silu acc and mul acc so they can use the same LDS for weight loading

        # silu acc
        if EVEN_K:
            a = tl.load(a_ptrs_k, mask=(token_mask[:, None]), other=0.0)
            w1 = tl.load(w1_ptrs_i0_k)
        else:
            a = tl.load(
                a_ptrs_k,
                mask=(
                    token_mask[:, None] & (offs_k1[None, :] < K - k1 * BLOCK_SIZE_K1)
                ),
                other=0.0,
            )
            w1 = tl.load(
                w1_ptrs_i0_k,
                mask=(offs_k1[:, None] < K - k1 * BLOCK_SIZE_K1),
                other=0.0,
            )

        if use_fp8_w8a8 and use_block_scale:
            w1_scale = tl.load(w1_i0_scale_ptrs_k)
            w1_scale = group_broadcast(w1_scale, 1, num_scales_along_n, group_n, 1)
            a_scale = tl.load(
                a_scale_ptrs_k, mask=token_mask[:, None], other=0.0
            )

            silu_acc += tl.dot(a, w1, out_dtype=tl.float32) * a_scale * w1_scale
        else:
            silu_acc = tl.dot(a, w1, acc=silu_acc)

        # mul acc
        if EVEN_K:
            w1 = tl.load(w1_ptrs_i1_k)
        else:
            w1 = tl.load(
                w1_ptrs_i1,
                mask=(offs_k1[:, None] < K - k1 * BLOCK_SIZE_K1),
                other=0.0,
            )
        if use_fp8_w8a8 and use_block_scale:
            w1_scale = tl.load(w1_i1_scale_ptrs_k)
            w1_scale = group_broadcast(w1_scale, 1, num_scales_along_n, group_n, 1)
            mul_acc += tl.dot(a, w1, out_dtype=tl.float32) * a_scale * w1_scale
        else:
            mul_acc = tl.dot(a, w1, acc=mul_acc)

        

    if use_fp8_w8a8 and not use_block_scale:
        silu_acc = silu_acc * a_scale * w1_scale
        mul_acc = mul_acc * a_scale * w1_scale
    
    # gated activation
    silu_acc = _silu_exp2(silu_acc)
    acc = silu_acc * mul_acc

    if return_intermediate:
        offs_in = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
        i_ptrs = Intermediate + stride_im * offs_token[:, None] + offs_in[None, :]
        i_mask = token_mask[:, None] & (offs_in[None, :] < N // 2)
        tl.store(i_ptrs, acc.to(out_dtype), mask=i_mask)

    acc = tl.where(
        (pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)[None, :]) < N // 2,
        acc,
        0.0,
    )

    if use_fp8_w8a8:
        acc = acc.to(tl.bfloat16)
    else:
        acc = acc.to(dtype)

    offs_w2n = (tl.arange(0, BLOCK_SIZE_HALF) + pid_n * (BLOCK_SIZE_HALF)) % (N // 2)

    w2_ptrs = (
        W2
        + off_experts * stride_w2e
        + (offs_k2[None, :] * stride_w2k + offs_w2n[:, None] * stride_w2n)
    )

    if use_fp8_w8a8 and use_block_scale:
        # offs_w2_sn = offs_w2n // group_n
        # instead load only the unique scaling factors and broadcast.
        offs_w2_sn = (pid_n * BLOCK_SIZE_HALF) // group_n + tl.arange(
            0, num_scales_along_n
        )
        # ... + tl.arange(0, num_scales_along_n) * group_n // group_n = ... + tl.arange(0, num_scales_along_n)
        w2_scale_ptrs = (
            W2_scale + off_experts * stride_w2se + offs_w2_sn[:, None] * stride_w2sn
        )
        w2_scale_ptrs += (tl.arange(0, num_scales_along_k2)[None, :]) * stride_w2sk
        # w2_scale_ptrs: (num_scales_along_n, num_scales_along_k2)

    out_ptrs = Out + stride_om * offs_token[:, None] + offs_k2[None, :] * stride_ok

    if use_fp8_w8a8 and not use_block_scale:
        w2_scale = tl.load(W2_scale + off_experts)

    num_k2 = tl.cdiv(K, BLOCK_SIZE_K2)
    # TODO: add num_stages > 1 support for use_fp8_w8a8. Now produces a compile error.
    for k2 in tl.range(0, num_k2, num_stages=1):
        w2 = tl.load(
            w2_ptrs + k2 * BLOCK_SIZE_K2 * stride_w2k,
        )
        if use_fp8_w8a8:
            if use_block_scale:
                w2_scale = tl.load(
                    w2_scale_ptrs + (k2 * BLOCK_SIZE_K2) // group_k * stride_w2sk,
                )  # (num_scales_along_n, num_scales_along_k2)
                # only need to do descale before dot if inner dimension (n) spans multiple scale values
                if num_scales_along_n > 1:
                    w2_scale = group_broadcast(
                        w2_scale, num_scales_along_n, num_scales_along_k2, group_n, 0
                    )
                    w2_scale = group_broadcast(
                        w2_scale,
                        num_scales_along_n * group_n,
                        num_scales_along_k2,
                        group_k,
                        1,
                    )
                    w2 = w2.to(tl.float32) * w2_scale.to(tl.float32)
                else:
                    w2_scale = group_broadcast(w2_scale, 1, num_scales_along_k2, group_k, 1)

            w2 = w2.to(tl.bfloat16)
            out = tl.dot(acc, w2)
        else:
            out = tl.dot(acc, w2)

        if use_fp8_w8a8 and ((not use_block_scale) or num_scales_along_n == 1):
            out = out * w2_scale

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            out = out * moe_weight[:, None]

        if EVEN_K2:
            out_mask = token_mask[:, None]
        else:
            out_mask = token_mask[:, None] & (
                offs_k2[None, :] < (K - k2 * BLOCK_SIZE_K2)
            )
        
        if SKINNY:
            # Skinny means that there is only one pid along N (i.e. BLOCK_SIZE_N >= N).
            # Thus we don't need atomics, as there is only workgroup updating a output location.
            out = out.to(out_dtype)
            tl.store(out_ptrs + k2 * BLOCK_SIZE_K2 * stride_ok, out, mask=out_mask)
        else:
            out = out.to(tl.float32)  # atomics need to be done in fp32
            tl.atomic_add(
                out_ptrs + k2 * BLOCK_SIZE_K2 * stride_ok,
                out,
                mask=out_mask
            )

        
