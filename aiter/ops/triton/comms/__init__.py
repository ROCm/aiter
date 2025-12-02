# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based communication primitives for AITER.

This submodule contains communication operations implemented using Triton,
including Iris-based GPU-initiated communication (optional).
"""

import logging

logger = logging.getLogger("aiter")

# Try to import iris-based communication primitives
# These are optional and only available if iris is installed
__all__ = []

try:
    from .iris import IrisCommContext
    from .reduce_scatter import reduce_scatter
    from .all_gather import all_gather
    from .fused import reduce_scatter_rmsnorm_quant_all_gather

    __all__.extend(
        [
            "IrisCommContext",
            "reduce_scatter",
            "all_gather",
            "reduce_scatter_rmsnorm_quant_all_gather",
        ]
    )

    IRIS_COMM_AVAILABLE = True
    logger.debug("Iris communication primitives loaded successfully")

except ImportError as e:
    IRIS_COMM_AVAILABLE = False
    logger.debug(f"Iris communication primitives not available: {e}")
    # Define stub variables so code can check for availability
    IrisCommContext = None
    reduce_scatter = None
    all_gather = None
    reduce_scatter_rmsnorm_quant_all_gather = None

__all__.append("IRIS_COMM_AVAILABLE")
