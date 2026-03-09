# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-Scatter communication primitive using Iris.

This module provides a reduce-scatter operation along the M dimension using
GPU-initiated communication via the Iris library.
"""

import torch
from torch import Tensor
import triton
import triton.language as tl
import logging

import iris

# If we got here, iris is available
IRIS_AVAILABLE = True

logger = logging.getLogger("aiter")


@triton.jit
def _reduce_scatter_impl(
    pid,
    input_ptr,
    output_ptr,
    M,
    M_shard,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Shared reduce-scatter implementation using pull-based approach with iris.load.

    Each rank computes its own output shard by:
    - Loading the relevant portion from all ranks (including itself)
    - Accumulating the sum locally
    - Storing the result

    For example, rank 0 computes output[0:M_shard, :] by:
    - Loading input[0:M_shard, :] from rank 0 (local)
    - Loading input[0:M_shard, :] from rank 1 (remote via iris.load)
    - ...
    - Loading input[0:M_shard, :] from rank 7 (remote via iris.load)
    - Summing all loaded data

    Args:
        pid: Program ID (from tl.program_id(0) or passed from parent kernel)
    """
    num_pid_m = tl.cdiv(M_shard, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # Persistent loop over tiles
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Swizzle pattern
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Local indices in this rank's output shard (M_shard × N)
        rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Add compiler hints
        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        # Masks
        mask_m_local = rm_local < M_shard
        mask_n = rn < N
        mask = mask_m_local[:, None] & mask_n[None, :]

        # Calculate which rows to read from each source rank's input
        # This rank (cur_rank) needs rows [cur_rank*M_shard : (cur_rank+1)*M_shard]
        # from ALL source ranks
        rm_global = cur_rank * M_shard + rm_local
        mask_m_global = rm_global < M
        load_mask = mask_m_global[:, None] & mask_n[None, :]

        # Accumulator for the sum across all ranks
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Pointers to the data we need from all ranks
        src_ptrs = input_ptr + rm_global[:, None] * stride_im + rn[None, :] * stride_in

        # Load from all source ranks and accumulate
        for src_rank in tl.static_range(world_size):
            data = iris.load(src_ptrs, cur_rank, src_rank, heap_bases, mask=load_mask)
            accumulator += data.to(tl.float32)

        # Store the result to output shard
        output_ptrs = (
            output_ptr + rm_local[:, None] * stride_om + rn[None, :] * stride_on
        )
        tl.store(output_ptrs, accumulator.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _reduce_scatter_kernel(
    input_ptr,  # Local input tensor in IRIS memory: *[M, N]
    output_ptr,  # Output shard in IRIS memory: *[M_shard, N]
    M,
    M_shard,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Reduce-scatter kernel entry point.

    This is a wrapper around _reduce_scatter_impl that gets the program ID.
    """
    pid = tl.program_id(0)
    _reduce_scatter_impl(
        pid,
        input_ptr,
        output_ptr,
        M,
        M_shard,
        N,
        stride_im,
        stride_in,
        stride_om,
        stride_on,
        cur_rank,
        world_size,
        heap_bases,
        BLOCK_M,
        BLOCK_N,
        GROUP_SIZE_M,
        NUM_SMS,
    )


def reduce_scatter(
    input_tensor: Tensor,
    ctx: "IrisCommContext" = None,
    block_m: int = 16,
    block_n: int = 64,
    group_size_m: int = 8,
    num_sms: int = 256,
) -> Tensor:
    """
    Perform reduce-scatter along the M (row) dimension.

    This operation:
    1. Sums the input_tensor across all ranks (M×N on each rank)
    2. Splits the result along the M dimension
    3. Each rank receives (M/world_size)×N

    Args:
        input_tensor (Tensor): Input tensor of shape [M, N] in Iris shared memory
        ctx (IrisCommContext): Iris communication context. Optional if global context exists.
        block_m (int): Block size for M dimension. Default: 16
        block_n (int): Block size for N dimension. Default: 64
        group_size_m (int): Group size for swizzling. Default: 8
        num_sms (int): Number of SMs to use (persistent kernel). Default: 256

    Returns:
        Tensor: Output shard of shape [M_shard, N] where M_shard = M // world_size

    Example:
        >>> with IrisCommContext() as ctx:
        >>>     input_tensor = ctx.iris_ctx.zeros((8192, 7168), dtype=torch.float32)
        >>>     # ... initialize input_tensor ...
        >>>     output_shard = reduce_scatter(input_tensor, ctx)
        >>>     print(output_shard.shape)  # [1024, 7168] for world_size=8
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError(
            "Iris library is not available. Cannot perform reduce-scatter."
        )

    if not ctx.is_initialized:
        raise RuntimeError(
            "Iris context not initialized. Use IrisCommContext as context manager."
        )

    # Get distributed parameters from context
    cur_rank = ctx.cur_rank
    world_size = ctx.num_ranks
    heap_bases = ctx.get_heap_bases()
    iris_ctx = ctx.iris_ctx

    # Input shape
    M, N = input_tensor.shape
    M_shard = M // world_size

    if M % world_size != 0:
        raise ValueError(f"M ({M}) must be divisible by world_size ({world_size})")

    logger.info(
        f"Rank {cur_rank}/{world_size}: Reduce-scatter M={M}, N={N} -> M_shard={M_shard}"
    )

    # Allocate output buffer in IRIS shared memory
    output_shard = iris_ctx.zeros((M_shard, N), dtype=input_tensor.dtype)

    # Launch kernel
    grid = (num_sms,)
    _reduce_scatter_kernel[grid](
        input_tensor,
        output_shard,
        M,
        M_shard,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_shard.stride(0),
        output_shard.stride(1),
        cur_rank,
        world_size,
        heap_bases,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=num_sms,
        num_warps=16,
        num_stages=4,
        waves_per_eu=4,
    )

    # Synchronize
    iris_ctx.barrier()

    logger.info(
        f"Rank {cur_rank}: Reduce-scatter complete, output_shard shape: {output_shard.shape}"
    )

    return output_shard
