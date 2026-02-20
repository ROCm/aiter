# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based communication primitives for AITER.

This submodule contains communication operations implemented using Triton,
including Iris-based GPU-initiated communication.

Iris-based primitives are only available when the iris package is installed.
"""

import logging

_logger = logging.getLogger(__name__)

try:
    from .iris import IrisCommContext, calculate_heap_size
    from .reduce_scatter import reduce_scatter
    from .all_gather import all_gather
    from .fused import reduce_scatter_rmsnorm_quant_all_gather

    IRIS_COMM_AVAILABLE = True
except ImportError as e:
    _logger.warning("Iris communication primitives not available: %s", e)
    IRIS_COMM_AVAILABLE = False

__all__ = [
    "IrisCommContext",
    "calculate_heap_size",
    "reduce_scatter",
    "all_gather",
    "reduce_scatter_rmsnorm_quant_all_gather",
    "IRIS_COMM_AVAILABLE",
]
