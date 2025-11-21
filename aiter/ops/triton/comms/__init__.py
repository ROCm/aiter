# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based communication primitives for AITER.

This submodule contains communication operations implemented using Triton,
including Iris-based GPU-initiated communication.
"""

from .iris import (
    IrisCommContext,
    all_reduce_iris,
    all_reduce_iris_atomic,
)

# Import communication primitives
from .reduce_scatter import (
    reduce_scatter_iris,
)

from .all_gather import (
    all_gather_iris,
)

# Import fused kernels
from .fused import (
    reduce_scatter_rmsnorm_quant_all_gather,
)

__all__ = [
    "IrisCommContext",
    "all_reduce_iris",
    "all_reduce_iris_atomic",
    "reduce_scatter_iris",
    "all_gather_iris",
    # Fused kernels
    "reduce_scatter_rmsnorm_quant_all_gather",
]
