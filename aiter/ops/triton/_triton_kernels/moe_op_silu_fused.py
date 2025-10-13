# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from .activation import _silu_exp2
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton.moe_common import _write_zeros_to_output


# Source:
# MoE Kernel adapted from VLLM


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_silu_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
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
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
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
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # silu ptrs
    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    # (i % 2): [0, 1, 0, 1,...] (alternating)
    # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
    # So offs_bn now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
    offs_bn = (offs_half + (i % 2) * (N // 2)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs)

        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    silu_acc, mul_acc = (
        accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    )
    silu_acc = _silu_exp2(silu_acc)
    accumulator = (silu_acc * mul_acc).to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_persistent_silu_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
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
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
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
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
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
    start_pid = tl.program_id(axis=0)
    # Load tile-invariant runtime constant
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_tiles = num_pid_m * num_pid_n
    # Compute how many tiles are outside the padding region
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)
    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        # Compute the mask
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        # silu ptrs
        BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
        i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
        i_floor = i // 2
        offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
        # (i % 2): [0, 1, 0, 1,...] (alternating)
        # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
        # So offs_bn now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
        offs_bn = (offs_half + (i % 2) * (N // 2)) % N

        # Compute the A pointer
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

        if use_int4_w4a16:
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] // 2) * stride_bk
                + offs_bn[None, :] * stride_bn
            )
            b_shifter = (offs_k[:, None] % 2) * 4
        elif use_int8_w8a16:
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

        if not has_zp and use_int4_w4a16:
            b_zp_num = 8
        if not has_zp and use_int8_w8a16:
            b_zp_num = 128
        elif has_zp and use_int4_w4a16:
            b_zp_shifter = (offs_bn[None, :] % 2) * 4

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.

            if not block_k_diviable:
                k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
                k_other = 0.0
            else:
                k_mask = None
                k_other = None

            if EVEN_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(b_ptrs)

            if use_int4_w4a16:
                b = (b >> b_shifter) & 0xF

            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + offs_bn[None, :] * stride_bsn
                + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
            )
            b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
            b_scale = b_scale.to(tl.float32)

            if has_zp and use_int4_w4a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + (offs_bn[None, :] // 2) * stride_bzn
                    + offs_k_true * stride_bzk
                )
                b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                b_zp = (b_zp >> b_zp_shifter) & 0xF
                b_zp = b_zp.to(tl.float32)
            elif has_zp and use_int8_w8a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + offs_bn[None, :] * stride_bzn
                    + offs_k_true * stride_bzk
                )
                b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                b_zp = b_zp.to(tl.float32)

            # We accumulate along the K dimension.
            if has_zp:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
            accumulator = tl.dot(a, b, acc=accumulator)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            if use_int4_w4a16:
                b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            else:
                b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)

        silu_acc, mul_acc = (
            accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
        )
        silu_acc = _silu_exp2(silu_acc)
        accumulator = (silu_acc * mul_acc).to(compute_type)

        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
        tl.store(c_ptrs, accumulator, mask=c_mask)

        tile_id += NUM_SMS


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_silu_kernel(
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
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
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
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # silu ptrs
    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    # (i % 2): [0, 1, 0, 1,...] (alternating)
    # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
    # So offs_bn now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
    offs_bn = (offs_half + (i % 2) * (N // 2)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
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
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    # silu_and_mul
    silu_acc, mul_acc = (
        accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    )
    silu_acc = _silu_exp2(silu_acc)
    accumulator = (silu_acc * mul_acc).to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_persistent_silu_kernel(
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
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
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
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
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

    # Load tile-invariant runtime constant
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_tiles = num_pid_m * num_pid_n

    # Compute how many tiles are outside the padding region
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)

    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        # Compute the mask
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        # silu ptrs
        BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
        i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
        i_floor = i // 2
        offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
        # (i % 2): [0, 1, 0, 1,...] (alternating)
        # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
        # So offs_bn now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
        offs_bn = (offs_half + (i % 2) * (N // 2)) % N

        # Compute the A pointer
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        # Compute the B pointer
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        if use_int8_w8a16:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (
                    b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
                )
            else:
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
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )
            # We accumulate along the K dimension.
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator = tl.dot(a, b, acc=accumulator)
            else:
                accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            accumulator = accumulator * moe_weight[:, None]

        if use_int8_w8a16:
            accumulator = (accumulator * b_scale).to(compute_type)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                accumulator = accumulator.to(compute_type)
            else:
                accumulator = (accumulator * a_scale * b_scale).to(compute_type)
        else:
            accumulator = accumulator.to(compute_type)

        # silu_and_mul
        silu_acc, mul_acc = (
            accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
        )
        silu_acc = _silu_exp2(silu_acc)
        accumulator = (silu_acc * mul_acc).to(compute_type)

        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
        tl.store(c_ptrs, accumulator, mask=c_mask)

        # advance tile_id
        tile_id += NUM_SMS
