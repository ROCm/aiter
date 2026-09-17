# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.conv_config_utils import get_conv_config, has_conv_config


def _get_config(shape_key=None, M=None):
    # Untuned architectures reuse the safe fallback tile for the [C, D*H*W]
    # transpose.
    name = "CONV3D-PREPACK" if has_conv_config("CONV3D-PREPACK") else "CONV-PREPACK"
    return get_conv_config(name, shape_key=shape_key, M=M)


_ncdhw_to_cblocked_kernel_repr = make_kernel_repr(
    "_ncdhw_to_cblocked_kernel",
    ["BLOCK_C", "BLOCK_M", "CB"],
)


@triton.jit(repr=_ncdhw_to_cblocked_kernel_repr)
def _ncdhw_to_cblocked_kernel(
    X,
    Y,
    C,
    DHW,
    C_PAD: tl.constexpr,
    CB: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Transpose NCDHW's ``[C, D*H*W]`` plane to NCDHWc storage."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    x_ptrs = X + pid_n * C * DHW + offs_c[:, None] * DHW + offs_m[None, :]
    tile = tl.load(
        x_ptrs,
        mask=(offs_c[:, None] < C) & (offs_m[None, :] < DHW),
        other=0.0,
    )

    c_block = offs_c // CB
    c_local = offs_c % CB
    y_ptrs = (
        Y
        + pid_n * C_PAD * DHW
        + c_block[None, :] * DHW * CB
        + offs_m[:, None] * CB
        + c_local[None, :]
    )
    tl.store(
        y_ptrs,
        tl.trans(tile),
        mask=(offs_m[:, None] < DHW) & (offs_c[None, :] < C_PAD),
    )
