# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Replaces the eager two-kernel x * gate.sigmoid() (one sigmoid pass writing
a temporary, one multiply pass reading it back) with a single pass
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_sigmoid_mul import (
    _fused_sigmoid_mul_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["fused_sigmoid_mul"]


def _is_row_strided_2d(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride(1) == 1 and t.stride(0) >= t.shape[1]


def fused_sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out = x * sigmoid(gate)`` in one pass; writes into x when out is None.

    x, gate and out are contiguous, or 2-D views with a dense last dimension.
    """
    _LOGGER.info("FUSED_SIGMOID_MUL: x=%s dtype=%s", tuple(x.shape), x.dtype)

    assert x.is_cuda, "x must be a CUDA tensor"
    assert gate.device == x.device, "x and gate must be on the same device"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"unsupported dtype: {x.dtype}"
    assert x.shape == gate.shape, f"shape mismatch: {x.shape} vs {gate.shape}"
    assert x.dtype == gate.dtype, f"dtype mismatch: {x.dtype} vs {gate.dtype}"

    if out is None:
        out = x
    else:
        assert out.shape == x.shape, f"out shape mismatch: {out.shape} vs {x.shape}"
        assert out.dtype == x.dtype, f"out dtype mismatch: {out.dtype} vs {x.dtype}"
        assert out.device == x.device, "out must be on the same device as x"

    if x.is_contiguous() and gate.is_contiguous() and out.is_contiguous():
        # Any contiguous shape runs as one row of x.numel() elements.
        x, gate, out_2d = (t.view(1, -1) for t in (x, gate, out))
        config = _get_config()
    else:
        assert all(
            _is_row_strided_2d(t) for t in (x, gate, out)
        ), "x, gate and out must be contiguous, or 2-D with a dense last dimension"
        out_2d = out
        config = _get_config("strided")

    M, N = x.shape
    if M == 0 or N == 0:
        return out

    BLOCK_SIZE_M = config.pop("BLOCK_SIZE_M")
    BLOCK_SIZE_N = config.pop("BLOCK_SIZE_N")
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    _fused_sigmoid_mul_kernel[grid](
        x,
        gate,
        out_2d,
        M,
        N,
        x.stride(0),
        gate.stride(0),
        out_2d.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NEED_MASK=(M % BLOCK_SIZE_M != 0) or (N % BLOCK_SIZE_N != 0),
        **config,
    )
    return out
