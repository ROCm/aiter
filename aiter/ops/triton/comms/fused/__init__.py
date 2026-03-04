# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Communication + Computation Kernels

This submodule contains Triton kernels that fuse communication operations
with computation operations for improved performance.

Examples:
- reduce_scatter + rmsnorm + quant + all_gather
- all_reduce + rmsnorm + quant
- reduce_scatter + gemm + all_gather
"""

from .reduce_scatter_rmsnorm_quant_all_gather import (
    reduce_scatter_rmsnorm_quant_all_gather,
)

__all__ = [
    "reduce_scatter_rmsnorm_quant_all_gather",
]
