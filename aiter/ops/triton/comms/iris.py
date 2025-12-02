# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Minimal Iris context wrapper for AITER communication operations.

This module provides a thin wrapper around the iris library context to
support reduce-scatter and all-gather operations. All core iris functions
(load, store, put, atomic_*, etc.) are provided by the iris library itself.
"""

import logging

try:
    import iris

    IRIS_AVAILABLE = True
except ImportError:
    IRIS_AVAILABLE = False
    logging.warning(
        "Iris library not available. Iris-based communication will not work."
    )

logger = logging.getLogger("aiter")


class IrisCommContext:
    """
    Minimal context wrapper for Iris-based communication operations.

    This is a thin wrapper around iris.iris() that provides convenient access
    to the iris context for use in reduce-scatter and all-gather operations.

    Example:
        >>> with IrisCommContext(heap_size=2**30) as ctx:
        >>>     shard = ctx.iris_ctx.shmem.zeros((1024, 1024), dtype=torch.float32)
        >>>     full = all_gather(shard, ctx)
    """

    def __init__(self, heap_size=1 << 30):
        """
        Initialize Iris communication context.

        Args:
            heap_size (int): Size of the symmetric heap in bytes. Default: 1GB
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
        """Clean up when exiting context manager."""
        # Iris context cleanup is handled automatically
        pass

    def get_heap_bases(self):
        """Get the heap bases tensor for use in Triton kernels."""
        if not self._initialized:
            raise RuntimeError("Iris context not initialized. Use as context manager.")
        return self.iris_ctx.heap_bases
