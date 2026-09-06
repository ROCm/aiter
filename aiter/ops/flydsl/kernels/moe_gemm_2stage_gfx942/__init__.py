# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE two-stage kernels: stage1 gate-up, stage2 down-projection, and reduce.

The public API mirrors FlyDSL's ``kernels.moe.moe_gemm_2stage`` package:
configuration-specific compile functions return cached FlyDSL launchers.
"""

from .gemm import compile_gemm
from .gemm1 import compile_moe_gemm1
from .gemm2 import compile_moe_gemm2
from .moe_reduce import (
    compile_moe_reduction,
    invert_sorted_ids,
    sorted_sum,
)
from .quant import flydsl_absmax, flydsl_quant_per_tensor

__all__ = [
    "compile_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "invert_sorted_ids",
    "sorted_sum",
]
