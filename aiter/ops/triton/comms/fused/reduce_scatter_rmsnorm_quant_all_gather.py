# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Reduce-Scatter + RMSNorm + Quantization + All-Gather

This module implements a fused kernel that combines:
1. Reduce-scatter: Reduce data across ranks and scatter results
2. RMSNorm: Root mean square normalization
3. Quantization: Per-token quantization (optional)
4. All-gather: Gather results from all ranks

This fusion enables fine-grained overlap between communication and computation,
reducing memory bandwidth and improving overall performance.
"""

import torch
from torch import Tensor
import triton
import triton.language as tl
import logging

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
def fused_reduce_scatter_rmsnorm_quant_all_gather_kernel(
    input_ptr,
    output_ptr,
    residual_in_ptr,
    weight_ptr,
    bias_ptr,
    xscale_ptr,
    yscale_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    epsilon: tl.constexpr,
    world_size: tl.constexpr,
    cur_rank: tl.constexpr,
    heap_bases: tl.tensor,
    use_quant: tl.constexpr,
):
    """
    Fused kernel: Reduce-scatter -> RMSNorm -> Quant -> All-gather

    This kernel performs:
    1. Reduce-scatter: Each rank reduces its chunk and scatters to others
    2. RMSNorm: Normalize the reduced data
    3. Quantization: Quantize the normalized data (if enabled)
    4. All-gather: Gather quantized data from all ranks

    Args:
        input_ptr: Input tensor pointer
        output_ptr: Output tensor pointer
        residual_in_ptr: Residual input for RMSNorm
        weight_ptr: RMSNorm weight
        bias_ptr: RMSNorm bias
        xscale_ptr: Quantization scale for input
        yscale_ptr: Quantization scale for output
        N: Total number of elements
        BLOCK_SIZE: Block size for processing
        epsilon: RMSNorm epsilon
        world_size: Number of ranks
        cur_rank: Current rank
        heap_bases: Iris heap bases tensor
        use_quant: Whether to use quantization
    """
    pid = tl.program_id(0)

    # Calculate chunk size per rank
    chunk_size = N // world_size
    chunk_start = cur_rank * chunk_size
    chunk_end = chunk_start + chunk_size

    # Process elements in blocks
    num_programs = tl.num_programs(0)
    for i in range(pid * BLOCK_SIZE, chunk_size, BLOCK_SIZE * num_programs):
        block_start = chunk_start + i
        block_end = tl.minimum(block_start + BLOCK_SIZE, chunk_end)

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = (offsets >= chunk_start) & (offsets < chunk_end)

        # Step 1: Reduce-scatter
        # Load local data
        local_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

        # Reduce from all ranks using atomic operations
        for rank in range(world_size):
            if rank != cur_rank:
                # Load data from remote rank
                remote_data = iris.load(
                    input_ptr + offsets,
                    cur_rank,
                    rank,
                    heap_bases,
                    mask=mask,
                )
                local_data = local_data + remote_data

        # Step 2: RMSNorm
        # Load residual
        residual = tl.load(residual_in_ptr + offsets, mask=mask, other=0.0)
        combined = local_data + residual

        # Compute RMS
        rms = tl.sqrt(tl.sum(combined * combined) / chunk_size + epsilon)

        # Normalize
        normalized = combined / rms

        # Apply weight and bias
        weight = tl.load(weight_ptr + (offsets - chunk_start), mask=mask, other=0.0)
        bias = tl.load(bias_ptr + (offsets - chunk_start), mask=mask, other=0.0)
        normed = normalized * weight + bias

        # Step 3: Quantization (if enabled)
        if use_quant:
            xscale = tl.load(xscale_ptr + (offsets - chunk_start), mask=mask, other=1.0)
            # Quantize (simplified - actual implementation would use proper quantization)
            quantized = (normed / xscale).to(tl.int8)
            yscale = tl.load(yscale_ptr + (offsets - chunk_start), mask=mask, other=1.0)
            output_data = quantized.to(tl.float32) * yscale
        else:
            output_data = normed

        # Step 4: All-gather
        # Store to local output
        tl.store(output_ptr + offsets, output_data, mask=mask)

        # Store to remote ranks using Iris store
        for rank in range(world_size):
            if rank != cur_rank:
                iris.store(
                    output_ptr + offsets,
                    output_data,
                    cur_rank,
                    rank,
                    heap_bases,
                    mask=mask,
                )


def reduce_scatter_rmsnorm_quant_all_gather(
    input_tensor: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    bias: Tensor,
    xscale: Tensor = None,
    epsilon: float = 1e-5,
    ctx=None,
    heap_size: int = 1 << 30,
    use_quant: bool = False,
) -> Tensor:
    """
    Fused reduce-scatter + RMSNorm + quantization + all-gather operation.

    This function combines multiple operations into a single Triton kernel
    for improved performance through fine-grained communication-computation overlap.

    Args:
        input_tensor: Input tensor to reduce-scatter
        residual_in: Residual input for RMSNorm
        weight: RMSNorm weight tensor
        bias: RMSNorm bias tensor
        xscale: Quantization scale (required if use_quant=True)
        epsilon: RMSNorm epsilon value
        ctx: Optional IrisCommContext. If None, a global context will be used.
        heap_size: Heap size for Iris context if ctx is None
        use_quant: Whether to apply quantization

    Returns:
        Tensor: Result of all-gather after reduce-scatter, RMSNorm, and quantization
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError("Iris library is not available.")

    # Get rank and world size from context
    if ctx is None:
        from ..iris import _get_or_create_iris_context

        ctx = _get_or_create_iris_context(heap_size=heap_size)

    rank = ctx.cur_rank
    world_size = ctx.num_ranks

    if world_size == 1:
        # Single rank: just do RMSNorm + quant locally
        # TODO: Implement local-only path
        raise NotImplementedError("Single rank path not yet implemented")

    # Allocate output tensor
    output = torch.empty_like(input_tensor)

    # Get heap bases
    heap_bases = ctx.get_heap_bases()

    # Launch kernel
    N = input_tensor.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(N // world_size, meta["BLOCK_SIZE"]),)

    # Prepare quantization scales if needed
    if use_quant and xscale is None:
        raise ValueError("xscale is required when use_quant=True")

    yscale = torch.ones_like(xscale) if use_quant else None

    fused_reduce_scatter_rmsnorm_quant_all_gather_kernel[grid](
        input_tensor,
        output,
        residual_in,
        weight,
        bias,
        xscale if use_quant else torch.empty(0, device=input_tensor.device),
        yscale if use_quant else torch.empty(0, device=input_tensor.device),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        epsilon=epsilon,
        world_size=world_size,
        cur_rank=rank,
        heap_bases=heap_bases,
        use_quant=use_quant,
    )

    return output


# TODO: Add more fused kernels:
# - reduce_scatter_gemm_all_gather
# - all_reduce_rmsnorm_quant (Triton version)
# - reduce_scatter_attention_all_gather
# etc.
