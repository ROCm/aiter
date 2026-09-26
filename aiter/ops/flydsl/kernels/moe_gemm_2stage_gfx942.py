# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports; implementations live in moe_gemm_2stage."""

from .moe_gemm_2stage import (
    compile_gemm,
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_reduction,
    flydsl_absmax,
    flydsl_quant_per_tensor,
    invert_sorted_ids,
    precompile_moe_quant_kernels,
    precompile_moe_reduction_kernels,
    sorted_sum,
)
from .moe_gemm_2stage.common import (
    _SIMPLIFIED_BF16_RTA,
    _SIMPLIFIED_BF16_RTE,
    _TORCH_TO_FX,
    _f32_to_bf16,
    _f32_to_bf16_rta,
    _f32_to_bf16_rte,
)
from .moe_gemm_2stage.common import (
    torch_tensor_to_pointer as _ptr,
)

__all__ = [
    "_SIMPLIFIED_BF16_RTA",
    "_SIMPLIFIED_BF16_RTE",
    "_TORCH_TO_FX",
    "_f32_to_bf16",
    "_f32_to_bf16_rta",
    "_f32_to_bf16_rte",
    "_ptr",
    "compile_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "invert_sorted_ids",
    "precompile_moe_quant_kernels",
    "precompile_moe_reduction_kernels",
    "sorted_sum",
]
