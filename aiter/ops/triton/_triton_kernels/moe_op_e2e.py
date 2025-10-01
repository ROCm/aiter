# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from .activation import _silu_exp2

from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd


@triton.jit
def group_broadcast(x, xM: tl.constexpr, xN: tl.constexpr, group_size: tl.constexpr, broadcast_dim: tl.constexpr):
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
        assert xM > 0, "broadcast_dim must be specified"
        if xM <= 1:
            return x # singleton dimension, no need to broadcast
        x = x.reshape(xM, 1, xN)
        x = tl.broadcast_to(x, (xM, group_size, xN))
        x = x.reshape(xM * group_size, xN)
    else:
        assert xN > 0, "broadcast_dim must be specified"
        if xN <= 1:
            return x # singleton dimension, no need to broadcast
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
    MUL_ROUTED_WEIGHT: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,  # original block_size_k
    BLOCK_SIZE_K2: tl.constexpr,  # outputs (EM, BLOCK_SIZE_K2)
    GROUP_SIZE_M: tl.constexpr,
    GRID_MN: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    SKINNY: tl.constexpr,
    dtype: tl.constexpr,
    out_dtype: tl.constexpr,
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

    off_experts = tl.load(expert_ids_ptr + pid_m)
    # TODO: add off_experts=-1 return condition

    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K2)

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2 # TODO: quarantee that BLOCK_SIZE_HALF*2=BLOCK_SIZE_N

    if use_fp8_w8a8:
        num_scales_along_n: tl.constexpr = (BLOCK_SIZE_HALF + group_n - 1) // group_n
        num_scales_along_k2: tl.constexpr = (BLOCK_SIZE_K2 + group_k - 1) // group_k

    offs_i0 = tl.arange(0, BLOCK_SIZE_HALF)
    offs_i1 = tl.arange(0, BLOCK_SIZE_HALF) + N // 2

    # offset for silu_acc
    i0 = pid_n * BLOCK_SIZE_HALF + offs_i0
    # offset for mul_acc
    i1 = pid_n * BLOCK_SIZE_HALF + offs_i1

    
    # TODO: add EVEN_N and pid_n is not last pid_n so no need for masking conditions
    # same mask applicable to both silu and mul acc
    mask_w1n = i0 < (N // 2)

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

    if use_fp8_w8a8:
        a_scale_ptrs = A_scale + (offs_token[:, None] // top_k * stride_asm) 
        i0s = pid_n * (BLOCK_SIZE_HALF // group_n) + tl.arange(0, num_scales_along_n)
        i1s = i0s + ((N // 2) // group_n)
        w1_i0_scale_ptrs = (
            W1_scale + off_experts * stride_w1se + i0s[None, :] * stride_w1sn 
        )
        w1_i1_scale_ptrs = (
            W1_scale + off_experts * stride_w1se + i1s[None, :] * stride_w1sn 
        )

    num_k1 = tl.cdiv(K, BLOCK_SIZE_K1)
    for k in tl.range(0, num_k1): 
        # Masking ensures we don't load from invalid tokens or indices
        if EVEN_K:
            a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)
            w1_i0 = tl.load(w1_ptrs_i0, mask=mask_w1n[None, :], other=0.0)
            w1_i1 = tl.load(w1_ptrs_i1, mask=mask_w1n[None, :], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=(token_mask[:, None] & (offs_k1[None, :] < K - k * BLOCK_SIZE_K1)),
                other=0.0,
            )
            w1_i0 = tl.load(
                w1_ptrs_i0,
                mask=(offs_k1[:, None] < K - k * BLOCK_SIZE_K1) & mask_w1n[None, :],
                other=0.0,
            )
            w1_i1 = tl.load(
                w1_ptrs_i1,
                mask=(offs_k1[:, None] < K - k * BLOCK_SIZE_K1) & mask_w1n[None, :],
                other=0.0,
            )
        
        if use_fp8_w8a8:
            start_k = k * BLOCK_SIZE_K1 // group_k

            w1_i0_scale = tl.load(w1_i0_scale_ptrs + start_k * stride_w1sk)
            w1_i1_scale = tl.load(w1_i1_scale_ptrs + start_k * stride_w1sk)

            if num_scales_along_n > 1: # singleton dimension get automatic broadcast
                w1_i0_scale = group_broadcast(w1_i0_scale, 1, num_scales_along_n, group_n, 1)
                w1_i1_scale = group_broadcast(w1_i1_scale, 1, num_scales_along_n, group_n, 1)

            a_scale = tl.load(a_scale_ptrs + start_k * stride_ask, mask=token_mask[:, None], other=0.0)
            
            silu_acc += tl.dot(a, w1_i0, out_dtype=tl.float32) * a_scale * w1_i0_scale
            mul_acc += tl.dot(a, w1_i1, out_dtype=tl.float32) * a_scale * w1_i1_scale
        else:
            mul_acc = tl.dot(a, w1_i1, acc=mul_acc) 
            silu_acc = tl.dot(a, w1_i0, acc=silu_acc) 
     

        a_ptrs += BLOCK_SIZE_K1 * stride_ak
        w1_ptrs_i0 += BLOCK_SIZE_K1 * stride_w1k
        w1_ptrs_i1 += BLOCK_SIZE_K1 * stride_w1k
    
    silu_acc = _silu_exp2(silu_acc)
    if use_fp8_w8a8:
        acc = (silu_acc * mul_acc).to(tl.bfloat16)
    else:
        acc = (silu_acc * mul_acc).to(dtype)

    offs_w2n = tl.arange(0, BLOCK_SIZE_HALF) + pid_n * (BLOCK_SIZE_HALF)

    w2_ptrs = (
        W2
        + off_experts * stride_w2e
        + (offs_k2[None, :] * stride_w2k + offs_w2n[:, None] * stride_w2n)
    )

    if use_fp8_w8a8:
        # offs_w2_sn = offs_w2n // group_n
        # instead load only the unique scaling factors and broadcast. 
        offs_w2_sn = pid_n * (BLOCK_SIZE_HALF) // group_n + tl.arange(0, num_scales_along_n) 
        # ... + tl.arange(0, num_scales_along_n) * group_n // group_n = ... + tl.arange(0, num_scales_along_n)
        w2_scale_ptrs = (
            W2_scale + off_experts * stride_w2se + offs_w2_sn[:, None] * stride_w2sn
        )
        w2_scale_ptrs += (tl.arange(0, num_scales_along_k2)[None, :]) * stride_w2sk
        # w2_scale_ptrs: (num_scales_along_n, num_scales_along_k2)

    out_ptrs = Out + stride_cm * offs_token[:, None] + offs_k2[None, :]

    num_k = tl.cdiv(K, BLOCK_SIZE_K2)
    for k in tl.range(0, num_k):

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

        if use_fp8_w8a8:
            w2_scale = tl.load(w2_scale_ptrs + k * BLOCK_SIZE_K2 // group_k * stride_w2sk) # (num_scales_along_n, num_scales_along_k2)
            w2_scale = group_broadcast(w2_scale, num_scales_along_n, num_scales_along_k2, group_n, 0)            
            if num_scales_along_k2 > 1: # singleton dimension get automatic broadcast
                w2_scale = group_broadcast(w2_scale, num_scales_along_n * group_n, num_scales_along_k2, group_k, 1)

            w2 = (w2.to(tl.float32) * w2_scale.to(tl.float32)).to(tl.bfloat16)
            out = tl.dot(acc, w2)
        else:
            out = tl.dot(acc, w2)

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            out = out * moe_weight[:, None]

        out = out.to(out_dtype)

        if EVEN_K:
            c_mask = token_mask[:, None]
        else:
            c_mask = token_mask[:, None] & ((offs_k2 + k * BLOCK_SIZE_K2)[None, :] < K)

        if SKINNY:
            # Skinny means that we can fit the whole intermediate representation of a token in register memory (i.e. N <= BLOCK_SIZE_N). 
            # Thus we don't need atomics, as there is only workgroup updating the output tile.
            tl.store(out_ptrs + k * BLOCK_SIZE_K2, out, mask=c_mask)
        else:
            tl.atomic_add(
                out_ptrs + k * BLOCK_SIZE_K2,
                out.to(dtype),
                mask=c_mask,
                sem="relaxed",
                scope="cta",
            )


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
    NUM_XCDS: tl.constexpr,
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
