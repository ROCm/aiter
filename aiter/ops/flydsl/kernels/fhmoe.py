# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Heterogeneous MoE adapters for the shared stage1 compiler."""

import functools

from .mixed_moe_gemm_2stage import compile_mixed_moe_gemm1


@functools.cache
def compile_mixed_fhmoe_gemm1(
    *,
    shared_expert_id: int,
    v2_output_layout: bool = False,
    xcd_swizzle: int = 0,
    **kwargs,
):
    """Compile a stage1 kernel with an FP8 shared expert."""
    return compile_mixed_moe_gemm1(
        v2_output_layout=v2_output_layout,
        xcd_swizzle=xcd_swizzle,
        _shared_expert_id=shared_expert_id,
        **kwargs,
    )
