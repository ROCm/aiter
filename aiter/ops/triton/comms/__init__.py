# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based communication primitives for AITER.

This submodule contains communication operations implemented using Triton,
including Iris-based GPU-initiated communication.

Iris-based primitives are only available when the iris package is installed.
"""

import logging
from contextlib import contextmanager

_logger = logging.getLogger(__name__)

try:
    from .iris import IrisCommContext, calculate_heap_size
    from .reduce_scatter import reduce_scatter
    from .all_gather import all_gather
    from .fused import reduce_scatter_rmsnorm_quant_all_gather

    IRIS_COMM_AVAILABLE = True
except Exception as e:
    _logger.warning("Iris communication primitives not available: %s", e)
    IRIS_COMM_AVAILABLE = False

_graph_capturing = False


@contextmanager
def graph_capture():
    """Context manager to mark CUDA graph capture as active.

    While active, iris device barriers are skipped to avoid flag drift
    during PIECEWISE graph capture warmup and recording phases.
    """
    global _graph_capturing
    _graph_capturing = True
    try:
        yield
    finally:
        _graph_capturing = False


def is_graph_capturing() -> bool:
    """Return True if CUDA graph capture is in progress."""
    return _graph_capturing


__all__ = [
    "IrisCommContext",
    "calculate_heap_size",
    "reduce_scatter",
    "all_gather",
    "reduce_scatter_rmsnorm_quant_all_gather",
    "IRIS_COMM_AVAILABLE",
    "graph_capture",
    "is_graph_capturing",
]
