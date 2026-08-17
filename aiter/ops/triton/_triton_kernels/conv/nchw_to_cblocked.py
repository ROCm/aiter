# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils.conv_config_utils import get_conv_config

from .helpers import CONV_AUTOTUNE_ENABLED


def _get_config(shape_key=None, M=None, variants=()):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config(
        "CONV-PREPACK", shape_key=shape_key, M=M, variants=variants
    )


@triton.jit
def _nchw_to_cblocked_kernel(
    X,
    Y,
    C,
    HW,
    C_PAD: tl.constexpr,
    CB: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Transpose NCHW's [C, HW] plane into [C/CB, HW, CB]."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    x_ptrs = X + pid_n * C * HW + offs_c[:, None] * HW + offs_m[None, :]
    tile = tl.load(
        x_ptrs,
        mask=(offs_c[:, None] < C) & (offs_m[None, :] < HW),
        other=0.0,
    )

    c_block = offs_c // CB
    c_local = offs_c % CB
    y_ptrs = (
        Y
        + pid_n * C_PAD * HW
        + c_block[None, :] * HW * CB
        + offs_m[:, None] * CB
        + c_local[None, :]
    )
    tl.store(
        y_ptrs,
        tl.trans(tile),
        mask=(offs_m[:, None] < HW) & (offs_c[None, :] < C_PAD),
    )


AUTOTUNE_NCHW_TO_CBLOCKED_CONFIGS = [
    triton.Config({"BLOCK_C": 32, "BLOCK_M": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_C": 32, "BLOCK_M": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_C": 32, "BLOCK_M": 128}, num_warps=8, num_stages=1),
    triton.Config(
        {"BLOCK_C": 32, "BLOCK_M": 128, "waves_per_eu": 2},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config({"BLOCK_C": 64, "BLOCK_M": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_C": 64, "BLOCK_M": 64}, num_warps=8, num_stages=1),
]


if CONV_AUTOTUNE_ENABLED:
    _nchw_to_cblocked_kernel = triton.autotune(
        configs=AUTOTUNE_NCHW_TO_CBLOCKED_CONFIGS,
        key=["C", "HW"],
        cache_results=True,
    )(_nchw_to_cblocked_kernel)
