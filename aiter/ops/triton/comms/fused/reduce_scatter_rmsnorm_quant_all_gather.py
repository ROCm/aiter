# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Reduce-Scatter + RMSNorm + Quantization + All-Gather

This module implements a fully fused Triton kernel that performs:
1. Reduce-scatter: Pull-based via iris.load
2. RMSNorm: Root mean square normalization on local shard
3. Quantization: Optional per-token quantization (FP8)
4. All-gather: Push-based via iris.put

Based on Iris example: examples/22_rs_rmsnorm_fp8quant_ag/reduce_scatter_rmsnorm_quant_fused.py
"""

import torch
from torch import Tensor
import triton
import triton.language as tl
import logging
from typing import Optional

try:
    import iris

    IRIS_AVAILABLE = True
except ImportError:
    IRIS_AVAILABLE = False
    logging.warning(
        "Iris library not available. Fused communication kernels will not work."
    )

logger = logging.getLogger("aiter")


@triton.jit
def _reduce_scatter_stage(
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
    """Reduce-scatter stage using pull-based iris.load"""
    num_pid_m = tl.cdiv(M_shard, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        mask_m_local = rm_local < M_shard
        mask_n = rn < N
        mask = mask_m_local[:, None] & mask_n[None, :]

        rm_global = cur_rank * M_shard + rm_local
        mask_m_global = rm_global < M
        load_mask = mask_m_global[:, None] & mask_n[None, :]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        src_ptrs = input_ptr + rm_global[:, None] * stride_im + rn[None, :] * stride_in

        for src_rank in tl.static_range(world_size):
            data = iris.load(src_ptrs, cur_rank, src_rank, heap_bases, mask=load_mask)
            accumulator += data.to(tl.float32)

        output_ptrs = (
            output_ptr + rm_local[:, None] * stride_om + rn[None, :] * stride_on
        )
        tl.store(output_ptrs, accumulator.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _rmsnorm_stage(
    pid,
    input_ptr,
    output_ptr,
    g_ptr,
    rsigma_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
):
    """RMSNorm stage with AITER optimizations"""
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        for row_idx in tl.range(pid, n_rows, NUM_WORKERS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs, cache_modifier=".cg").to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            sum_squares += tl.sum(x * x, axis=0)

            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs, cache_modifier=".cg").to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)
    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(pid, n_rows, NUM_WORKERS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)
            rms_norm = row * norm_factor * g
            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _quantize_fp8_stage(
    pid,
    qx_ptr,
    scale_out_ptr,
    x_in_ptr,
    n_rows: int,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
):
    """FP8 per-token quantization stage"""
    offsets = tl.arange(0, NUM_COL_POW2)
    mask_cols = offsets < cols
    for row_idx in tl.range(pid, n_rows, NUM_WORKERS, num_stages=1):
        row_offsets = row_idx * x_in_stride_r + offsets
        row = tl.load(x_in_ptr + row_offsets, mask=mask_cols, cache_modifier=".cg")

        row_max = tl.max(tl.abs(row), axis=-1)
        scale = tl.maximum(row_max.to(tl.float32) / DTYPE_MAX, 1e-8)
        scale_recip = 1.0 / scale

        clamped = row * scale_recip
        fp8_max = tl.full((), DTYPE_MAX, tl.float32)
        clamped = tl.maximum(tl.minimum(clamped, fp8_max), -fp8_max)
        quantized = clamped.to(qx_ptr.dtype.element_ty)

        tl.store(scale_out_ptr + row_idx, scale)
        tl.store(qx_ptr + row_offsets, quantized, mask=mask_cols, cache_modifier=".cs")


@triton.jit
def _all_gather_stage(
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
    """All-gather stage using push-based iris.put"""
    num_pid_m = tl.cdiv(M_shard, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        mask_m_local = rm_local < M_shard
        mask_n = rn < N
        mask = mask_m_local[:, None] & mask_n[None, :]

        shard_data = tl.load(
            shard_ptr + rm_local[:, None] * stride_sm + rn[None, :] * stride_sn,
            mask=mask,
            other=0.0,
        )

        for dst in range(world_size):
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
                iris.put(
                    shard_ptr + rm_local[:, None] * stride_sm + rn[None, :] * stride_sn,
                    out_ptrs,
                    cur_rank,
                    dst,
                    heap_bases,
                    mask=final_mask,
                )


@triton.jit
def fused_pipeline_kernel(
    input_ptr,
    shard_ptr,
    norm_ptr,
    fp8_ptr,
    gather_ptr,
    scale_ptr,
    gamma_ptr,
    rsigma_ptr,
    heap_bases,
    M,
    M_shard,
    N,
    stride_im,
    stride_in,
    stride_sm,
    stride_sn,
    stride_nm,
    stride_nn,
    stride_fp8_m,
    stride_fp8_n,
    stride_full_m,
    stride_full_n,
    eps,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    quant_mode: tl.constexpr,
    do_allgather: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    RMS_BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
    FP8_NUM_COL_POW2: tl.constexpr,
    FP8_DTYPE_MAX: tl.constexpr,
):
    """
    Fused pipeline kernel combining all stages.

    Stages:
    1. Reduce-scatter (pull-based)
    2. RMSNorm
    3. FP8 quantization (optional)
    4. All-gather (push-based, optional)
    """
    pid = tl.program_id(axis=0)

    # Stage 1: Reduce-scatter
    _reduce_scatter_stage(
        pid,
        input_ptr,
        shard_ptr,
        M,
        M_shard,
        N,
        stride_im,
        stride_in,
        stride_sm,
        stride_sn,
        cur_rank,
        world_size,
        heap_bases,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
    )

    # Stage 2: RMSNorm
    _rmsnorm_stage(
        pid,
        shard_ptr,
        norm_ptr,
        gamma_ptr,
        rsigma_ptr,
        stride_sm,
        stride_nm,
        M_shard,
        N,
        eps,
        BLOCK_SIZE=RMS_BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        NUM_WORKERS=NUM_WORKERS,
    )

    # Stage 3 & 4: Quantization + All-gather (if requested)
    if quant_mode == 1:  # FP8
        _quantize_fp8_stage(
            pid,
            fp8_ptr,
            scale_ptr,
            norm_ptr,
            M_shard,
            N,
            stride_nm,
            NUM_COL_POW2=FP8_NUM_COL_POW2,
            DTYPE_MAX=FP8_DTYPE_MAX,
            NUM_WORKERS=NUM_WORKERS,
        )

        if do_allgather:
            _all_gather_stage(
                pid,
                fp8_ptr,
                gather_ptr,
                M,
                M_shard,
                N,
                stride_fp8_m,
                stride_fp8_n,
                stride_full_m,
                stride_full_n,
                cur_rank,
                world_size,
                heap_bases,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                GROUP_SIZE_M=GROUP_SIZE_M,
                NUM_SMS=NUM_SMS,
            )
    else:  # No quantization
        if do_allgather:
            _all_gather_stage(
                pid,
                norm_ptr,
                gather_ptr,
                M,
                M_shard,
                N,
                stride_nm,
                stride_nn,
                stride_full_m,
                stride_full_n,
                cur_rank,
                world_size,
                heap_bases,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                GROUP_SIZE_M=GROUP_SIZE_M,
                NUM_SMS=NUM_SMS,
            )


def reduce_scatter_rmsnorm_quant_all_gather(
    input_tensor: Tensor,
    gamma: Tensor,
    epsilon: float = 1e-6,
    ctx=None,
    heap_size: int = 1 << 30,
    quant_mode: str = "none",
    do_allgather: bool = True,
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Fused reduce-scatter + RMSNorm + quantization + all-gather operation.

    Args:
        input_tensor (Tensor): Input tensor of shape [M, N] in Iris shared memory
        gamma (Tensor): RMSNorm weight of shape [N]
        epsilon (float): RMSNorm epsilon value. Default: 1e-6
        ctx: Optional IrisCommContext. If None, a global context will be used.
        heap_size (int): Heap size for Iris context if ctx is None
        quant_mode (str): Quantization mode - "none", "fp8_per_token". Default: "none"
        do_allgather (bool): Whether to perform all-gather stage. Default: True

    Returns:
        tuple: (normalized_shard, quantized_output, full_gathered_output)
            - normalized_shard: RMSNorm output [M_shard, N]
            - quantized_output: FP8 quantized output [M_shard, N] (if quant_mode="fp8_per_token")
            - full_gathered_output: All-gathered result [M, N] (if do_allgather=True)

    Example:
        >>> with IrisCommContext() as ctx:
        >>>     input_tensor = ctx.iris_ctx.shmem.ones((8192, 7168), dtype=torch.float32)
        >>>     gamma = torch.ones(7168, device="cuda")
        >>>     norm, quant, gathered = reduce_scatter_rmsnorm_quant_all_gather(
        >>>         input_tensor, gamma, ctx=ctx, quant_mode="fp8_per_token"
        >>>     )
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError(
            "Iris library is not available. Cannot perform fused operation."
        )

    if ctx is None:
        from ..iris import _get_or_create_iris_context

        ctx = _get_or_create_iris_context(heap_size=heap_size)

    if not ctx._initialized:
        raise RuntimeError(
            "Iris context not initialized. Use IrisCommContext as context manager."
        )

    # Get distributed parameters from context
    cur_rank = ctx.cur_rank
    world_size = ctx.num_ranks
    heap_bases = ctx.get_heap_bases()
    shmem = ctx.iris_ctx.shmem

    # Input shape
    M, N = input_tensor.shape
    M_shard = M // world_size

    if M % world_size != 0:
        raise ValueError(f"M ({M}) must be divisible by world_size ({world_size})")

    logger.info(
        f"Rank {cur_rank}/{world_size}: Fused pipeline M={M}, N={N} -> M_shard={M_shard}"
    )

    # Allocate buffers
    device = input_tensor.device
    dtype = input_tensor.dtype

    rs_buffer = shmem.zeros((M_shard, N), dtype=dtype)
    norm_buffer = torch.empty((M_shard, N), dtype=torch.float32, device=device)
    rsigma = torch.empty(M_shard, dtype=torch.float32, device=device)

    # Quantization buffers
    if quant_mode == "fp8_per_token":
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.int8)
        fp8_out = shmem.empty((M_shard, N), dtype=fp8_dtype)
        scales = torch.empty(M_shard, dtype=torch.float32, device=device)
        fp8_dtype_max = (
            448.0
            if fp8_out.dtype == getattr(torch, "float8_e4m3fn", None)
            else float(torch.iinfo(torch.int8).max)
        )
    else:
        fp8_out = norm_buffer  # Reuse norm buffer
        scales = torch.zeros(M_shard, dtype=torch.float32, device=device)
        fp8_dtype_max = 1.0

    # All-gather buffer
    if do_allgather:
        gather_dtype = fp8_out.dtype if quant_mode == "fp8_per_token" else dtype
        gather_out = shmem.zeros((M, N), dtype=gather_dtype)
    else:
        gather_out = norm_buffer  # Reuse norm buffer

    # Kernel parameters
    BLOCK_M = 16
    BLOCK_N = 64
    GROUP_SIZE_M = 8
    device_props = torch.cuda.get_device_properties(device)
    NUM_SMS = device_props.multi_processor_count

    rms_block = 1024 if N >= 1024 else triton.next_power_of_2(N)
    rms_block = min(rms_block, 1024)
    use_blocked = N > rms_block
    fp8_num_cols_pow2 = triton.next_power_of_2(N)

    grid = (NUM_SMS,)
    quant_mode_flag = 0 if quant_mode == "none" else 1

    # Launch fused kernel
    fused_pipeline_kernel[grid](
        input_tensor,
        rs_buffer,
        norm_buffer,
        fp8_out,
        gather_out,
        scales,
        gamma,
        rsigma,
        heap_bases,
        M,
        M_shard,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        rs_buffer.stride(0),
        rs_buffer.stride(1),
        norm_buffer.stride(0),
        norm_buffer.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        gather_out.stride(0),
        gather_out.stride(1),
        epsilon,
        cur_rank=cur_rank,
        world_size=world_size,
        quant_mode=quant_mode_flag,
        do_allgather=do_allgather,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        RMS_BLOCK_SIZE=rms_block,
        USE_BLOCKED=use_blocked,
        NUM_WORKERS=NUM_SMS,
        FP8_NUM_COL_POW2=fp8_num_cols_pow2,
        FP8_DTYPE_MAX=fp8_dtype_max,
        num_warps=16,
        num_stages=4,
        waves_per_eu=4,
    )

    # Synchronize
    torch.cuda.synchronize()
    shmem.barrier()

    logger.info(f"Rank {cur_rank}: Fused pipeline complete")

    # Return results based on configuration
    norm_result = norm_buffer if quant_mode == "none" else None
    quant_result = fp8_out if quant_mode == "fp8_per_token" else None
    gather_result = gather_out if do_allgather else None

    return norm_result, quant_result, gather_result
