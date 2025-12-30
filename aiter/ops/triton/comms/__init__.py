# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based communication primitives for AITER.

This submodule contains communication operations implemented using Triton,
including Iris-based GPU-initiated communication.

If Iris is not available, importing this module will raise ImportError.
"""

# Import all Iris-based communication primitives
# If Iris is not installed, this import will fail and the entire
# aiter.ops.triton.comms module will be unavailable
from .iris import IrisCommContext, calculate_heap_size
from .reduce_scatter import reduce_scatter
from .all_gather import all_gather
from .fused import reduce_scatter_rmsnorm_quant_all_gather

__all__ = [
    "IrisCommContext",
    "calculate_heap_size",
    "reduce_scatter",
    "all_gather",
    "reduce_scatter_rmsnorm_quant_all_gather",
    "IRIS_COMM_AVAILABLE",
]

# If we got here, Iris is available
IRIS_COMM_AVAILABLE = True
