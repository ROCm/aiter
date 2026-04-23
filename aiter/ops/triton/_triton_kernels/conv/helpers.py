# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    x = tl.minimum(tl.maximum(x, -10.0), 10.0)
    e2x = tl.exp(2 * x)
    return (e2x - 1) / (e2x + 1)


# ========================================================================
# AUTOTUNE CONFIGS — centralized to avoid duplication and cross-imports
# ========================================================================

# -- 1x1 kernel --
AUTOTUNE_1x1_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
]

# -- General conv kernel --
AUTOTUNE_GENERAL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
]

# -- 3x3 NHWC kernel --
AUTOTUNE_3x3_NHWC_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_C": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_C": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_C": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_C": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_C": 32, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
]

# -- 3x3 channel-blocked kernel --
AUTOTUNE_3x3_CBLOCKED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_C": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_C": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_C": 128, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_C": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_C": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
]

# -- Winograd F(4,3) GEMM --
AUTOTUNE_WINO_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
]

# -- Winograd F(4,3) input transform --
AUTOTUNE_WINO4_INPUT_CONFIGS = [
    triton.Config({"BLOCK_C": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_C": 32}, num_warps=4, num_stages=1),
]

# -- Winograd F(4,3) output transform --
AUTOTUNE_WINO4_OUTPUT_CONFIGS = [
    triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=1),
]

# -- Winograd F(4,3) fused GEMM+output --
AUTOTUNE_FUSED_F4X3_CONFIGS = [
    triton.Config(
        {"BLOCK_T": 16, "BLOCK_K": 64, "BLOCK_C": 64}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_T": 16, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=8, num_stages=1
    ),
    triton.Config(
        {"BLOCK_T": 32, "BLOCK_K": 64, "BLOCK_C": 64}, num_warps=4, num_stages=1
    ),
]
