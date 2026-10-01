# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.activation import _sigmoid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

_fused_sigmoid_mul_repr = make_kernel_repr(
    "_fused_sigmoid_mul_kernel",
    ["BLOCK_SIZE_N", "NEED_MASK", "num_warps"],
)

_fused_sigmoid_mul_2d_repr = make_kernel_repr(
    "_fused_sigmoid_mul_2d_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "NEED_MASK", "num_warps"],
)


def _get_config(key: str = "any") -> dict:
    config_dir = resolve_config_dir("fusions", "FUSED_SIGMOID_MUL")
    return dict(load_config_json(f"{config_dir}/DEFAULT.json", required=True)[key])


@triton.jit(repr=_fused_sigmoid_mul_repr)
def _fused_sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """
    out[i] = x[i] * sigmoid(gate[i])
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = None
    if NEED_MASK:
        mask = offs < N

    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, x * _sigmoid(gate), mask=mask)


@triton.jit(repr=_fused_sigmoid_mul_2d_repr, do_not_specialize=["M"])
def _fused_sigmoid_mul_2d_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    M,
    N,
    stride_x_m,
    stride_gate_m,
    stride_out_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """
    out[m, n] = x[m, n] * sigmoid(gate[m, n]) for 2-D views
    """
    rows = (tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = None
    if NEED_MASK:
        mask = (rows[:, None] < M) & (cols[None, :] < N)

    gate = tl.load(
        gate_ptr + rows[:, None] * stride_gate_m + cols[None, :], mask=mask
    ).to(tl.float32)
    x = tl.load(x_ptr + rows[:, None] * stride_x_m + cols[None, :], mask=mask).to(
        tl.float32
    )
    tl.store(
        out_ptr + rows[:, None] * stride_out_m + cols[None, :],
        x * _sigmoid(gate),
        mask=mask,
    )
