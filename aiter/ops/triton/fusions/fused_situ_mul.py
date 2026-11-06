# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
SiTU-and-mul in one pass
"""

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_situ_mul import (
    _fused_situ_mul_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["fused_situ_mul"]

_LOG2_E = 1.4426950408889634


@functools.lru_cache(maxsize=None)
def _num_cus(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _is_row_strided_2d(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride(1) == 1 and t.stride(0) >= t.shape[1]


def fused_situ_mul(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """``beta * tanh(g / beta) * sigmoid(g) * up'`` for ``x = [g | up]`` on the last dim.

    ``up' = linear_beta * tanh(up / linear_beta)`` when linear_beta is given, else ``up``.
    x is contiguous, or 2-D with a dense last dimension. Returns a new tensor with x's
    leading shape and half its last dimension.
    """
    _LOGGER.info("FUSED_SITU_MUL: x=%s dtype=%s", tuple(x.shape), x.dtype)

    assert x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"unsupported dtype: {x.dtype}"
    assert x.shape[-1] % 2 == 0, f"last dim must be even (2 * d): {x.shape}"
    assert beta > 0, f"beta must be positive: {beta}"
    clip_up = linear_beta is not None and linear_beta > 0

    d = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    if x.is_contiguous():
        x_2d = x.view(-1, 2 * d)
    else:
        assert _is_row_strided_2d(
            x
        ), "x must be contiguous, or 2-D with a dense last dimension"
        x_2d = x
    out_2d = out.view(-1, d)

    M, N = out_2d.shape
    if M == 0 or N == 0:
        return out

    config = _get_config(M, N)
    BLOCK_SIZE_M = config.pop("BLOCK_SIZE_M")
    BLOCK_SIZE_N = min(config.pop("BLOCK_SIZE_N"), triton.next_power_of_2(N))
    programs_per_cu = config.pop("PROGRAMS_PER_CU")
    tiles_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_tiles = tiles_n * triton.cdiv(M, BLOCK_SIZE_M)
    if programs_per_cu > 0:
        grid = (min(num_tiles, programs_per_cu * _num_cus(x.device)),)
    else:  # 0 = one program per tile (no persistent loop)
        grid = (num_tiles,)
    _fused_situ_mul_kernel[grid](
        x_2d,
        out_2d,
        M,
        N,
        x_2d.stride(0),
        out_2d.stride(0),
        -2.0 * _LOG2_E / beta,
        -2.0 * _LOG2_E / linear_beta if clip_up else 0.0,
        float(beta),
        float(linear_beta) if clip_up else 1.0,
        num_tiles,
        tiles_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NEED_MASK=(M % BLOCK_SIZE_M != 0) or (N % BLOCK_SIZE_N != 0),
        CLIP_UP=clip_up,
        **config,
    )
    return out
