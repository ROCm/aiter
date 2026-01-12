# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..iris import IrisCommContext

import iris

# If we got here, iris is available
IRIS_AVAILABLE = True

# Import shared implementations
from ..reduce_scatter import _reduce_scatter_impl
from ..all_gather import _all_gather_impl
from ..._triton_kernels.rmsnorm import _rms_norm_kernel

logger = logging.getLogger("aiter")


# Note: Shared implementations imported to avoid code duplication:
# - _reduce_scatter_impl from reduce_scatter.py
# - _all_gather_impl from all_gather.py
# - _rms_norm_kernel from _triton_kernels/rmsnorm.py (tl.program_id(0) works in nested calls)


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


# Note: _all_gather_impl is now imported from all_gather.py
# This avoids code duplication between standalone and fused kernels


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

    # Stage 1: Reduce-scatter (using shared implementation)
    _reduce_scatter_impl(
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

    # Stage 2: RMSNorm (using shared kernel - tl.program_id(0) works in nested calls)
    _rms_norm_kernel(
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
        NUM_PRGMS=NUM_WORKERS,
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
            _all_gather_impl(
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
            _all_gather_impl(
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
    ctx: "IrisCommContext" = None,
    quant_mode: str = "none",
    do_allgather: bool = True,
    # Pre-allocated buffers (optional, for reuse across iterations)
    rs_buffer: Optional[Tensor] = None,
    norm_buffer: Optional[Tensor] = None,
    fp8_out: Optional[Tensor] = None,
    gather_out: Optional[Tensor] = None,
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Fused reduce-scatter + RMSNorm + quantization + all-gather operation.

    Args:
        input_tensor (Tensor): Input tensor of shape [M, N] in Iris shared memory
        gamma (Tensor): RMSNorm weight of shape [N]
        epsilon (float): RMSNorm epsilon value. Default: 1e-6
        ctx (IrisCommContext): Iris communication context (required)
        quant_mode (str): Quantization mode - "none", "fp8_per_token". Default: "none"
        do_allgather (bool): Whether to perform all-gather stage. Default: True
        rs_buffer (Tensor, optional): Pre-allocated reduce-scatter buffer [M_shard, N]
        norm_buffer (Tensor, optional): Pre-allocated normalization buffer [M_shard, N]
        fp8_out (Tensor, optional): Pre-allocated FP8 output buffer [M_shard, N]
        gather_out (Tensor, optional): Pre-allocated all-gather output buffer [M, N]

    Returns:
        tuple: (normalized_shard, quantized_output, full_gathered_output)
            - normalized_shard: RMSNorm output [M_shard, N] in **float32** (if quant_mode="none")
              Note: Computed in float32 for numerical accuracy. Convert to input dtype if needed.
            - quantized_output: FP8 quantized output [M_shard, N] (if quant_mode="fp8_per_token")
            - full_gathered_output: All-gathered result [M, N] (if do_allgather=True)

    Example:
        >>> with IrisCommContext() as ctx:
        >>>     # Input must be in Iris shared memory for reduce-scatter
        >>>     input_tensor = ctx.iris_ctx.ones((8192, 7168), dtype=torch.float32)
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
        raise ValueError(
            "ctx is required. Use IrisCommContext as a context manager:\n"
            "  with IrisCommContext() as ctx:\n"
            "      input_tensor = ctx.iris_ctx.ones((M, N), dtype=...)\n"
            "      result = reduce_scatter_rmsnorm_quant_all_gather(input_tensor, gamma, ctx=ctx)"
        )

    if not ctx.is_initialized:
        raise RuntimeError(
            "Iris context not initialized. Use IrisCommContext as context manager."
        )

    # Get distributed parameters from context
    cur_rank = ctx.cur_rank
    world_size = ctx.num_ranks
    heap_bases = ctx.get_heap_bases()

    # Input shape
    M, N = input_tensor.shape
    M_shard = M // world_size

    if M % world_size != 0:
        raise ValueError(f"M ({M}) must be divisible by world_size ({world_size})")

    logger.info(
        f"Rank {cur_rank}/{world_size}: Fused pipeline M={M}, N={N} -> M_shard={M_shard}"
    )

    # Allocate or reuse buffers
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Reduce-scatter buffer (allocate if not provided)
    if rs_buffer is None:
        rs_buffer = ctx.iris_ctx.zeros((M_shard, N), dtype=dtype)

    # Normalization buffer (allocate if not provided)
    if norm_buffer is None:
        norm_buffer = torch.empty((M_shard, N), dtype=torch.float32, device=device)

    rsigma = torch.empty(M_shard, dtype=torch.float32, device=device)

    # Quantization buffers
    if quant_mode == "fp8_per_token":
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.int8)
        if fp8_out is None:
            fp8_out = ctx.iris_ctx.empty((M_shard, N), dtype=fp8_dtype)
        scales = torch.empty(M_shard, dtype=torch.float32, device=device)
        # Determine max value based on dtype (use consistent fallback)
        fp8_dtype_max = (
            448.0
            if hasattr(torch, "float8_e4m3fn") and fp8_out.dtype == torch.float8_e4m3fn
            else float(torch.iinfo(torch.int8).max)
        )
    else:
        if fp8_out is None:
            fp8_out = norm_buffer  # Reuse norm buffer
        scales = torch.zeros(M_shard, dtype=torch.float32, device=device)
        fp8_dtype_max = 1.0

    # All-gather buffer (allocate if not provided)
    if do_allgather:
        gather_dtype = fp8_out.dtype if quant_mode == "fp8_per_token" else dtype
        if gather_out is None:
            gather_out = ctx.iris_ctx.zeros((M, N), dtype=gather_dtype)
    else:
        if gather_out is None:
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
    ctx.iris_ctx.barrier()

    logger.info(f"Rank {cur_rank}: Fused pipeline complete")

    # Return results based on configuration
    # Note: norm_buffer is in float32 for numerical accuracy during RMSNorm
    # Users should convert to input dtype if needed: norm_buffer.to(input_dtype)
    norm_result = norm_buffer if quant_mode == "none" else None
    quant_result = fp8_out if quant_mode == "fp8_per_token" else None
    gather_result = gather_out if do_allgather else None

    return norm_result, quant_result, gather_result
