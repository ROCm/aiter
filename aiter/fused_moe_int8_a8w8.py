'''
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

import aiter as moe_kernels

logger = moe_kernels.getLogger()
VLLM_MOE_PADDING = bool(int(os.getenv("VLLM_MOE_PADDING", "1")))
FUSED_MOE_PERSISTENT = bool(int(os.getenv("FUSED_MOE_PERSISTENT", "0")))
ENABLE_MOE_LDS_BYPASS = bool(int(os.getenv("ENABLE_MOE_LDS_BYPASS", "1")))
print(f'{FUSED_MOE_PERSISTENT=}, {ENABLE_MOE_LDS_BYPASS=}, {VLLM_MOE_PADDING=}')
VLLM_FUSED_MOE_CHUNK_SIZE = 65536
#padding_size = 128 if VLLM_MOE_PADDING else 0
padding_size = 0


@triton.jit
def fused_moe_int8_a8w8_fuse_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    a_scale_ptr = a_scale_ptr + offs_token // top_k
    b_scale_ptr = b_scale_ptr + off_experts * stride_bse + offs_cn
    _ALPHA0 = tl.zeros([1], dtype=a_scale_ptr.dtype.element_ty)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    a_scale = tl.load(a_scale_ptr, mask=token_mask, other=_ALPHA0).to(tl.float32)
    b_scale = tl.load(b_scale_ptr, mask=offs_cn < N, other=_ALPHA0).to(tl.float32)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    _A0 = tl.zeros([1, 1], dtype=a_ptrs.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=b_ptrs.dtype.element_ty)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=_A0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=_B0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator * a_scale[:,None] * b_scale[None,:]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
@triton.jit
def fused_moe_int8_a8w8_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    a_scale_ptr = a_scale_ptr + offs_token // top_k
    b_scale_ptr = b_scale_ptr + off_experts * stride_bse + offs_cn
    _ALPHA0 = tl.zeros([1], dtype=a_scale_ptr.dtype.element_ty)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    a_scale = tl.load(a_scale_ptr, mask=token_mask, other=_ALPHA0).to(tl.float32)
    b_scale = tl.load(b_scale_ptr, mask=offs_cn < N, other=_ALPHA0).to(tl.float32)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    _A0 = tl.zeros([1, 1], dtype=a_ptrs.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=b_ptrs.dtype.element_ty)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=_A0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=_B0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator * a_scale[:,None] * b_scale[None,:]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def fused_moe_persistent_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    This is the persistent version of the fused_moe kernel.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Simply compute how many iterations each persistent block needs to do
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_tiles = num_pid_m * num_pid_n
    tile_id = start_pid

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_token = tl.zeros((BLOCK_SIZE_M,), dtype=tl.int32)
    # token_mask = tl.zeros((BLOCK_SIZE_M,), dtype=tl.int1)

    # Load tile-invariant runtime constant
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Compute how many tiles are outside the padding region
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    pid_m = 0
    tile_id2 = start_pid - NUM_SMS
    num_valid_tiles = -1
    while pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
        num_valid_tiles += 1
        tile_id2 += NUM_SMS
        group_id = tile_id2 // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id2 % num_pid_in_group) % group_size_m)

    for _ in range(0, num_valid_tiles):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
        # Compute the mask
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens
        # Compute the A pointer
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
            offs_k[None, :] * stride_ak)
        # Compute the B pointer
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        off_experts = tl.load(expert_ids_ptr + pid_m)
        b_ptrs = (b_ptr + off_experts * stride_be +
            (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

        if use_fp8:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if EVEN_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] &
                        (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0
                )
            # We accumulate along the K dimension.
            if use_fp8:
                accumulator = tl.dot(a, b, acc=accumulator)
            else:
                accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token,
                                mask=token_mask,
                                other=0)
            accumulator = accumulator * moe_weight[:, None]

        if use_fp8:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
        else:
            accumulator = accumulator.to(compute_type)
        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (c_ptr + stride_cm * offs_token[:, None] +
                stride_cn * offs_cn[None, :])
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

        # advance tile_id
        tile_id += NUM_SMS

@triton.jit
def _abs_max(val1, val2):
    val1_abs = tl.abs(val1)
    val2_abs = tl.abs(val2)
    if val1_abs >= val2_abs:
        return val1_abs
    else:
        return val2_abs


_dynamic_quant_configs = [
    triton.Config(
        {},
        num_warps=warps,
    )
    for warps in [2, 4, 8, 16]
]


@triton.jit
def _triton_dynamic_quantize_kernel(
    output_ptr,
    input_ptr,
    scale_ptr,
    stride_outputm,
    stride_outputn,
    stride_inputm,
    stride_inputn,
    n_elements,
    N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, N)
    mask = offsets < n_elements
    input_ptrs = input_ptr + pid * stride_inputm + offsets
    input_vals = tl.load(input_ptrs, mask=mask, other=1e-6)
    abs_max_f = tl.reduce(input_vals, 0, _abs_max)
    dynamic_per_token_scale = 127.0 / abs_max_f
    precison_mask = tl.where(input_vals > 0, 0.5, -0.5)
    output_vals = (input_vals * dynamic_per_token_scale + precison_mask).to(tl.int8)
    output_ptrs = output_ptr + pid * stride_outputm + offsets
    tl.store(output_ptrs, output_vals, mask=mask)
    tl.store(scale_ptr + pid, abs_max_f / 127.0)

def triton_dynamic_quantize(out, input, scale):
    assert input.is_contiguous(), "input must be contiguous"
    num_tokens = input.size(0)
    hidden_size = input.size(1)
    # tl.reduce requires the number of elements
    # must be power-of-two
    hidden_size_padded = triton.next_power_of_2(int(hidden_size))
    kwargs = [
        out,
        input,
        scale,
        out.stride(0),
        out.stride(1),
        input.stride(0),
        input.stride(1),
        input.size(1),
    ]
    grid = (num_tokens, 1, 1)
    const_kwargs = {"N": hidden_size_padded}
    method_name = "dynamic_quant_" + str(hidden_size_padded)
    if 0:
        dynamic_quant = triton.autotune(configs=_dynamic_quant_configs, key=['N'])(_triton_dynamic_quantize_kernel)
    else:
        dynamic_quant = _triton_dynamic_quantize_kernel
    dynamic_quant[grid](*kwargs,**const_kwargs)

def moe_align_block_size(
        topk_ids: torch.Tensor, block_size: int,
        num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    #print("BLOCK_SIZE_M in aligh", block_size)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.zeros((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    token_num = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    moe_kernels.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, token_num, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad

# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
#    if scale is not None:
#        # static-per-tensor quantization.
#        assert symmetric == (
#            azp is
#            None), "azp must only be provided for asymmetric quantization."
#        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
#        return output, scale, None

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    moe_kernels.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales

def invoke_fused_moe_int8_a8w8_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            A_scale: torch.Tensor,
                            B_scale: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any], compute_type: tl.dtype,
                            use_fp8_w8a8: bool, use_int8_w8a16: bool) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    assert B_scale is not None
    B_scale = B_scale.view(-1, B.shape[1])
  #  print("llm Bscale shape:",A_scale)

    if not FUSED_MOE_PERSISTENT:
        grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
            "BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]), )

        fused_moe_int8_a8w8_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            #B.shape[2] - padding_size,
            B.shape[2] ,
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(1),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        #    enable_moe_lds_bypass=ENABLE_MOE_LDS_BYPASS
        )
    else:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
        grid = lambda META: (min(
                NUM_SMS,
                triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) *
                    triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"])
                ), )

        fused_moe_persistent_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            B.shape[2] - padding_size,
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            NUM_SMS=NUM_SMS,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8=use_fp8_w8a8,
            **config,
        #    enable_moe_lds_bypass=ENABLE_MOE_LDS_BYPASS
        )


def get_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
    # device_name = current_platform.get_device_name().replace(" ", "_")
    device_name = 'AMD_Instinct_OAM'  # TODO: need to update
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int,
                    dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.",
                        config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.info("---> MOE tuned file not found at %s",config_file_path)
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
) -> Dict[str, int]:
    config = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 128, # reqd. for MOE shuffle
        'BLOCK_SIZE_K': 128, # reqd. for MOE shuffle
        'GROUP_SIZE_M': 8
    }
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 128, # reqd. for MOE shuffle
            'BLOCK_SIZE_K': 128, # reqd. for MOE shuffle
            'GROUP_SIZE_M': 1
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    override_config: Optional[Dict[str, Any]] = None,
    is_marlin: bool = False,
):
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        configs = get_moe_configs(E, N, dtype)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                        is_marlin)
    return config


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    moe_kernels.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
        renormalize
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek-V2 model
def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores,
                                        k=topk,
                                        dim=-1,
                                        sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_config_dtype_str(dtype: torch.dtype,
                         use_int8_w8a16: Optional[bool] = False,
                         use_fp8_w8a8: Optional[bool] = False):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def fused_experts_int8_a8w8(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  w1_scale: torch.Tensor,
                  w2_scale: torch.Tensor,
                  a1_scale: torch.Tensor,
                  a2_scale: torch.Tensor,
                  inplace: bool = False,
                  override_config: Optional[Dict[str, Any]] = None,
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a16: bool = False):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2] - padding_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        dtype=hidden_states.dtype)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padding_size),
        topk_ids.shape[1],
        config_dtype,
        override_config=override_config,
    )

    config = get_config_func(M)
    hidden_states_quant = torch.empty(
        hidden_states.shape, dtype=torch.int8, device=hidden_states.device
    )
    hidden_states_scales = torch.empty(
        hidden_states.shape[0], dtype=torch.half, device=hidden_states.device
    )
    triton_dynamic_quantize(hidden_states_quant, hidden_states, hidden_states_scales)

    intermediate_cache1 = torch.zeros((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=torch.half)
    intermediate_cache2 = torch.zeros((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=torch.half)
    intermediate_cache3 = torch.zeros((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=torch.half)

    compute_type = (tl.bfloat16
                    if hidden_states.dtype == torch.bfloat16 else tl.float16)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    # print("init config:", config)
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states_quant[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break
    #    print("tokens_in_chunk,", tokens_in_chunk, "CHUNK_SIZE",CHUNK_SIZE, "num_tokens",num_tokens)
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)
        #    print("inside branch:")


        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], E))
    #    print("llm : hidden_state:", curr_hidden_states)
    #    print("llm: w1:", w1)
    #    print("llm w1+scale", w1_scale)
        invoke_fused_moe_int8_a8w8_kernel(curr_hidden_states,
                                w1,
                                intermediate_cache1,
                                #a1_scale,
                                hidden_states_scales,
                                w1_scale,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                topk_ids.shape[1],
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a16=use_int8_w8a16)
    #    print("llm : intermediate_cache1", intermediate_cache1)
        moe_kernels.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        intermediate_cache2_quant = torch.empty(
            intermediate_cache2.shape, dtype=torch.int8, device=hidden_states.device
        )
        intermediate_cache2_scales = torch.empty(
            intermediate_cache2.shape[0], dtype=torch.half, device=hidden_states.device
        )
        triton_dynamic_quantize(
            intermediate_cache2_quant, intermediate_cache2, intermediate_cache2_scales
        )
#        print("llm: cache2:", intermediate_cache2_quant)
#        print("llm: cache2_sclae:", intermediate_cache2_scales)

        invoke_fused_moe_int8_a8w8_kernel(intermediate_cache2_quant,
                                w2,
                                intermediate_cache3,
                                intermediate_cache2_scales,
                                w2_scale,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                True,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a16=use_int8_w8a16)
        #print("dtype:",intermediate_cache3.dtype)
#        moe_kernels.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
#                            out_hidden_states[begin_chunk_idx:end_chunk_idx])
#
    #return out_hidden_states
   # return intermediate_cache3
   # print("before sum : intermediate_cache3",intermediate_cache3)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)


def fused_moe_int8_a8w8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
#        print("fused topk")
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize = False)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
#    print("moe topk_weights",topk_weights,"topk_ids",topk_ids)
    return fused_experts_int8_a8w8(hidden_states,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale,
                         inplace=inplace,
                         override_config=override_config,
                         use_fp8_w8a8=use_fp8_w8a8,
                         use_int8_w8a16=use_int8_w8a16)
