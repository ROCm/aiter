# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Replaces the eager two-kernel x * gate.sigmoid() (one sigmoid pass writing
a temporary, one multiply pass reading it back) with a single pass
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_sigmoid_mul import (
    _fused_sigmoid_mul_2d_kernel,
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
    """
    Fused elementwise ``out = x * sigmoid(gate)``.

    Args:
        x: any shape, contiguous; or a 2-D row-strided view (see below)
        gate: same shape and dtype as x
        out: optional destination

    Returns:
        out if given, else x (in place).

    Constraints:
        x and gate must be the same shape and dtype. When x, gate and out are
        all contiguous a flat 1-D kernel runs. Otherwise each must be 2-D with
        a dense last dimension
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

    if not (x.is_contiguous() and gate.is_contiguous() and out.is_contiguous()):
        assert all(
            _is_row_strided_2d(t) for t in (x, gate, out)
        ), "x, gate and out must be contiguous, or 2-D with a dense last dimension"
        return _fused_sigmoid_mul_2d(x, gate, out)

    N = x.numel()
    if N == 0:
        return out

    config = _get_config()
    BLOCK_SIZE_N = config.pop("BLOCK_SIZE_N")

    _fused_sigmoid_mul_kernel[(triton.cdiv(N, BLOCK_SIZE_N),)](
        x,
        gate,
        out,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NEED_MASK=N % BLOCK_SIZE_N != 0,
        **config,
    )
    return out


def _fused_sigmoid_mul_2d(
    x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    M, N = x.shape
    if M == 0 or N == 0:
        return out

    config = _get_config("strided")
    BLOCK_SIZE_M = config.pop("BLOCK_SIZE_M")
    BLOCK_SIZE_N = config.pop("BLOCK_SIZE_N")

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    _fused_sigmoid_mul_2d_kernel[grid](
        x,
        gate,
        out,
        M,
        N,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NEED_MASK=(M % BLOCK_SIZE_M != 0) or (N % BLOCK_SIZE_N != 0),
        **config,
    )
    return out
