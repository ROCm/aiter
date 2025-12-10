# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Minimal Iris context wrapper for AITER communication operations.

This module provides a thin wrapper around the iris library context to
support reduce-scatter and all-gather operations. All core iris functions
(load, store, put, atomic_*, etc.) are provided by the iris library itself.
"""

import logging
import math
from typing import Optional

import torch
import iris

# If we got here, iris is available
IRIS_AVAILABLE = True

logger = logging.getLogger("aiter")


def calculate_heap_size(
    M: int,
    N: int,
    dtype: "torch.dtype",
    world_size: Optional[int] = None,
    quant_mode: str = "none",
    all_gather: bool = True,
    overhead_factor: float = 1.2,
) -> int:
    """
    Calculate required Iris heap size for communication operations.

    This function estimates the total memory needed in the Iris symmetric heap
    for reduce-scatter, RMSNorm, quantization, and all-gather operations.

    Args:
        M (int): Number of rows in input tensor
        N (int): Number of columns in input tensor
        dtype (torch.dtype): Data type of input tensor (e.g., torch.float32, torch.float16)
        world_size (int, optional): Number of GPUs. If None, uses torch.distributed.get_world_size()
        quant_mode (str): Quantization mode - "none", "fp8_per_token", or "fp4_per_token"
        all_gather (bool): Whether all-gather is performed after quantization
        overhead_factor (float): Safety margin multiplier (default: 1.2 for 20% overhead)

    Returns:
        int: Required heap size in bytes

    Example:
        >>> # For 2 GPUs, 4096x4096 float32 tensor with FP8 quant and all-gather
        >>> heap_size = calculate_heap_size(4096, 4096, torch.float32,
        ...                                  world_size=2, quant_mode="fp8_per_token")
        >>> with IrisCommContext(heap_size=heap_size) as ctx:
        ...     # Your operations here
        ...     pass
    """
    if world_size is None:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed not initialized and world_size not provided. "
                "Either initialize torch.distributed or pass world_size explicitly."
            )
        world_size = torch.distributed.get_world_size()

    # Calculate element size in bytes (fast path for common types)
    if dtype in (torch.float32, torch.int32):
        elem_bytes = 4
    elif dtype in (torch.float16, torch.bfloat16, torch.int16):
        elem_bytes = 2
    elif dtype in (torch.float64, torch.int64):
        elem_bytes = 8
    elif dtype == torch.int8:
        elem_bytes = 1
    else:
        # Fallback for uncommon types (e.g., float8, complex types, future dtypes)
        elem_bytes = torch.empty(0, dtype=dtype).element_size()

    M_shard = M // world_size

    # Memory for input tensor (M x N)
    mem_input = M * N * elem_bytes

    # Memory for reduce-scatter output (M_shard x N)
    mem_rs = M_shard * N * elem_bytes

    # Memory for quantization output
    mem_quant = 0
    if quant_mode == "fp8_per_token":
        mem_quant = M_shard * N  # 1 byte per element
    elif quant_mode == "fp4_per_token":
        if N % 32 != 0:
            raise ValueError("FP4 quantization requires N divisible by 32")
        mem_quant = M_shard * (N // 2)  # 0.5 bytes per element

    # Memory for all-gather output
    mem_gather = 0
    if all_gather:
        if quant_mode == "fp8_per_token":
            mem_gather = M * N  # 1 byte per element
        elif quant_mode == "fp4_per_token":
            mem_gather = M * (N // 2)  # 0.5 bytes per element
        else:
            mem_gather = M * N * elem_bytes  # Full precision

    # Total with overhead
    total_bytes = mem_input + mem_rs + mem_quant + mem_gather
    total_with_overhead = int(math.ceil(total_bytes * overhead_factor))

    logger.debug(
        f"Heap size calculation: M={M}, N={N}, dtype={dtype}, world_size={world_size}, "
        f"quant_mode={quant_mode}, all_gather={all_gather}\n"
        f"  Input: {mem_input:,} bytes\n"
        f"  RS buffer: {mem_rs:,} bytes\n"
        f"  Quant buffer: {mem_quant:,} bytes\n"
        f"  Gather buffer: {mem_gather:,} bytes\n"
        f"  Total (with {overhead_factor}x overhead): {total_with_overhead:,} bytes "
        f"({total_with_overhead / (1024**3):.2f} GB)"
    )

    return total_with_overhead


class IrisCommContext:
    """
    Minimal context wrapper for Iris-based communication operations.

    This is a thin wrapper around iris.iris() that provides convenient access
    to the iris context for use in reduce-scatter and all-gather operations.

    Example:
        >>> # Manual heap size
        >>> with IrisCommContext(heap_size=2**30) as ctx:
        >>>     shard = ctx.iris_ctx.zeros((1024, 1024), dtype=torch.float32)
        >>>     full = all_gather(shard, ctx)

        >>> # Automatic heap size calculation
        >>> heap_size = calculate_heap_size(4096, 4096, torch.float32,
        ...                                  world_size=2, quant_mode="fp8_per_token")
        >>> with IrisCommContext(heap_size=heap_size) as ctx:
        >>>     # Guaranteed to have enough memory for your operations
        >>>     input_tensor = ctx.iris_ctx.empty((4096, 4096), dtype=torch.float32)
    """

    def __init__(self, heap_size: int = 1 << 30):
        """
        Initialize Iris communication context.

        Args:
            heap_size (int): Size of the symmetric heap in bytes. Default: 1GB
                Use calculate_heap_size() to automatically determine the required size
                based on your tensor dimensions and operations.

        Example:
            >>> # Option 1: Fixed size
            >>> ctx = IrisCommContext(heap_size=2**30)  # 1GB

            >>> # Option 2: Auto-calculated size
            >>> M, N = 4096, 4096
            >>> heap_size = calculate_heap_size(M, N, torch.float32,
            ...                                  world_size=2, quant_mode="fp8_per_token")
            >>> ctx = IrisCommContext(heap_size=heap_size)
        """
        if not IRIS_AVAILABLE:
            raise RuntimeError("Iris library is not available. Please install iris.")

        self.heap_size = heap_size
        self.iris_ctx = None
        self._initialized = False

    def __enter__(self):
        """Initialize Iris context when entering context manager."""
        if not self._initialized:
            self.iris_ctx = iris.iris(heap_size=self.heap_size)
            self._initialized = True
            self.cur_rank = self.iris_ctx.cur_rank
            self.num_ranks = self.iris_ctx.num_ranks

            logger.info(
                f"Iris context initialized: rank {self.cur_rank}/{self.num_ranks}, heap_size={self.heap_size}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up when exiting context manager.

        Used by Python's with statement (automatically called).

        Example:
            with IrisCommContext(...) as ctx:
                pass
            # __exit__() is called here automatically
        """
        # Iris context cleanup is handled automatically
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if the Iris context has been initialized."""
        return self._initialized

    def get_heap_bases(self):
        """Get the heap bases tensor for use in Triton kernels."""
        if not self.is_initialized:
            raise RuntimeError("Iris context not initialized. Use as context manager.")
        return self.iris_ctx.heap_bases
