# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fast_transpose import (
    _transpose_2d_kernel,
    _transpose_packed_fp4_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = ["fast_transpose_2d", "transpose_packed_fp4"]

_LOGGER = AiterTritonLogger()

# Both entry points use the same fixed 32x32 logical transpose tile.  For the
# packed path, the second compile-time extent is half as large because each
# input byte contains two logical columns.  The shape-dependent ``min`` below
# only keeps ``tl.arange`` legal for small dimensions; this is not a separate
# architecture-specific tuning table.
_TRANSPOSE_TILE_M = 32
_TRANSPOSE_TILE_N = 32


def fast_transpose_2d(x: torch.Tensor) -> torch.Tensor:
    """Transpose a contiguous 2D tensor using a Triton tiled kernel.

    Returns a contiguous (N, M) tensor from a (M, N) input.
    Works with any dtype including FP8 (e4m3, e5m2, fnuz variants).
    Replaces the ``tensor.t().contiguous()`` pattern which dispatches a
    full ``aten::copy_`` kernel.
    """
    _LOGGER.info(f"FAST_TRANSPOSE_2D: x={tuple(x.shape)}")
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape

    out = torch.empty((N, M), dtype=x.dtype, device=x.device)

    BLOCK_M = _TRANSPOSE_TILE_M
    BLOCK_N = _TRANSPOSE_TILE_N
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # num_warps=1: 32×32=1024-element tile is small; benchmarks on MI308X show
    # nw=1/2 tie for best, nw≥4 regresses (nw=16 is 2.5× slower than nw=1).
    _transpose_2d_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        waves_per_eu=2,
        num_stages=2,
    )
    return out


def transpose_packed_fp4(data_fp4: torch.Tensor) -> torch.Tensor:
    """Transpose a row-major matrix packed as two FP4 values per byte.

    Args:
        data_fp4: CUDA tensor with shape ``(M, N // 2)`` and dtype ``uint8``
            or ``float4_e2m1fn_x2``. Each byte stores the even logical column
            in its low nibble and the odd logical column in its high nibble.
            ``M`` must be even so the transposed logical rows can be packed in
            pairs.

    Returns:
        A contiguous tensor with shape ``(N, M // 2)`` and the input dtype,
        containing the bit-exact logical transpose in the same low/high-nibble
        convention.

    This operation reads and writes row-major packed data. Use AITER's existing
    weight shuffle separately when a GEMM-specific physical layout is needed.
    """
    _LOGGER.info(f"TRANSPOSE_PACKED_FP4: data_fp4={tuple(data_fp4.shape)}")
    if data_fp4.dim() != 2:
        raise ValueError(f"data_fp4 must be 2-D, got {data_fp4.dim()}-D")
    packed_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if data_fp4.dtype != torch.uint8 and data_fp4.dtype != packed_dtype:
        raise TypeError(
            "data_fp4 must have dtype torch.uint8 or "
            f"torch.float4_e2m1fn_x2, got {data_fp4.dtype}"
        )
    if not data_fp4.is_cuda:
        raise ValueError("data_fp4 must be a CUDA tensor")

    M, N_PACKED = data_fp4.shape
    if M == 0 or N_PACKED == 0:
        raise ValueError(
            f"data_fp4 dimensions must be non-zero, got {tuple(data_fp4.shape)}"
        )
    if M % 2 != 0:
        raise ValueError(f"data_fp4 rows must be even, got M={M}")

    input_bytes = data_fp4.view(torch.uint8)
    output = torch.empty(
        (N_PACKED * 2, M // 2), dtype=torch.uint8, device=data_fp4.device
    )
    BLOCK_M = min(_TRANSPOSE_TILE_M, triton.next_power_of_2(M))
    BLOCK_N_PACKED = min(_TRANSPOSE_TILE_N // 2, triton.next_power_of_2(N_PACKED))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_PACKED, BLOCK_N_PACKED))
    _transpose_packed_fp4_kernel[grid](
        input_bytes,
        output,
        M,
        N_PACKED,
        input_bytes.stride(0),
        input_bytes.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N_PACKED=BLOCK_N_PACKED,
    )
    return output.view(data_fp4.dtype)
