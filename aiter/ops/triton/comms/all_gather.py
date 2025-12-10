# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
All-Gather communication primitive using Iris.

This module provides an all-gather operation along the M dimension using
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
def _all_gather_impl(
    pid,
    shard_ptr,
    out_ptr,
    M,
    M_shard,
    N,
    stride_sm,
    stride_sn,
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
    Shared all-gather implementation using push-based approach with iris.put. 1D persistent-style PID mapping

    Each rank sends its (M_shard)×N to all other ranks at the appropriate offset.

    Args:
        pid: Program ID,  1D persistent-style PID mapping
        from tl.program_id(0) or passed from parent kernel
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

        # Local indices
        rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        mask_m_local = rm_local < M_shard
        mask_n = rn < N

        # Load local shard
        shard_ptrs = shard_ptr + rm_local[:, None] * stride_sm + rn[None, :] * stride_sn
        shard_data = tl.load(
            shard_ptrs, mask=mask_m_local[:, None] & mask_n[None, :], other=0.0
        )

        # Send to all ranks at the appropriate M offset
        for dst in range(world_size):
            # Calculate global M indices
            rm_global = cur_rank * M_shard + rm_local
            mask_m_global = rm_global < M
            final_mask = mask_m_global[:, None] & mask_n[None, :]

            out_ptrs = (
                out_ptr + rm_global[:, None] * stride_om + rn[None, :] * stride_on
            )

            if dst == cur_rank:
                # Local store
                tl.store(out_ptrs, shard_data, mask=final_mask)
            else:
                # Remote store using iris.put
                # from_ptr: local source, to_ptr: remote destination
                iris.put(
                    shard_ptr + rm_local[:, None] * stride_sm + rn[None, :] * stride_sn,
                    out_ptrs,
                    cur_rank,
                    dst,
                    heap_bases,
                    mask=final_mask,
                )


@triton.jit
def _all_gather_kernel(
    shard_ptr,  # *[M_shard, N]
    out_ptr,  # *[M, N]
    M,
    M_shard,
    N,
    stride_sm,
    stride_sn,
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
    All-gather kernel entry point.

    This is a wrapper around _all_gather_impl that gets the program ID.
    """
    pid = tl.program_id(0)
    _all_gather_impl(
        pid,
        shard_ptr,
        out_ptr,
        M,
        M_shard,
        N,
        stride_sm,
        stride_sn,
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


def all_gather(
    input_shard: Tensor,
    ctx: "IrisCommContext" = None,
    block_m: int = 64,
    block_n: int = 64,
    group_size_m: int = 8,
    num_sms: int = 256,
) -> Tensor:
    """
    Perform all-gather along the M (row) dimension.

    This operation:
    1. Each rank has a shard of shape [M_shard, N]
    2. All ranks send their shards to all other ranks
    3. Each rank receives a full tensor of shape [M, N] where M = M_shard * world_size

    Args:
        input_shard (Tensor): Input shard of shape [M_shard, N] in Iris shared memory
        ctx (IrisCommContext): Iris communication context. Optional if global context exists.
        block_m (int): Block size for M dimension. Default: 64
        block_n (int): Block size for N dimension. Default: 64
        group_size_m (int): Group size for swizzling. Default: 8
        num_sms (int): Number of SMs to use (persistent kernel). Default: 256

    Returns:
        Tensor: Full tensor of shape [M, N] where M = M_shard * world_size

    Example:
        >>> with IrisCommContext() as ctx:
        >>>     input_shard = ctx.iris_ctx.zeros((1024, 7168), dtype=torch.float32)
        >>>     # ... initialize input_shard ...
        >>>     full_tensor = all_gather(input_shard, ctx)
        >>>     print(full_tensor.shape)  # [8192, 7168] for world_size=8
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError("Iris library is not available. Cannot perform all-gather.")

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
    M_shard, N = input_shard.shape
    M = M_shard * world_size

    logger.info(
        f"Rank {cur_rank}/{world_size}: All-gather M_shard={M_shard}, N={N} -> M={M}"
    )

    # Allocate output buffer in IRIS shared memory
    full_output = iris_ctx.zeros((M, N), dtype=input_shard.dtype)

    # Launch kernel
    grid = (num_sms,)
    _all_gather_kernel[grid](
        input_shard,
        full_output,
        M,
        M_shard,
        N,
        input_shard.stride(0),
        input_shard.stride(1),
        full_output.stride(0),
        full_output.stride(1),
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
    torch.cuda.synchronize()
    iris_ctx.barrier()

    logger.info(
        f"Rank {cur_rank}: All-gather complete, output shape: {full_output.shape}"
    )

    return full_output
