# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP8 GEMM: Y = X @ W^T with microscaling (E8M0 block scales).

MXFP8 uses E8M0 (uint8) exponent-only scales — one per block of 32 (or 64)
elements along the K dimension.  This wrapper converts E8M0 scales to FP32
power-of-two values, then delegates to :func:`gemm_a8w8_blockscale` which
already handles the block-scale accumulation in its Triton kernel.

Scale conversion:  ``scale_fp32 = 2^(e8m0_val - 127)``
Implemented as a bit-shift:  ``(e8m0.to(int32) << 23).view(float32)``
"""

import torch

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def e8m0_to_fp32(scales_e8m0: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 uint8 scales to FP32 power-of-2 scale factors.

    E8M0 stores the biased exponent (bias=127) as a uint8.
    The FP32 representation is obtained by placing the byte
    in the exponent field of IEEE-754 float32.

    Args:
        scales_e8m0: Tensor of dtype uint8 with E8M0 values.

    Returns:
        Tensor of dtype float32 where each element is ``2^(e8m0 - 127)``.
    """
    flat = scales_e8m0.reshape(-1).to(torch.int32)
    fp32_bits = flat << 23
    fp32_vals = fp32_bits.view(torch.float32)
    return fp32_vals.reshape(scales_e8m0.shape)


def gemm_mxfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    quant_block_size: int = 32,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """MXFP8 GEMM: Y = (X * x_scale) @ (W * w_scale)^T.

    Uses the existing blockscale GEMM kernel after converting E8M0
    scales to FP32.

    Args:
        x: FP8 input matrix ``(M, K)``.
        w: FP8 weight matrix ``(N, K)``, internally transposed.
        x_scale: E8M0 uint8 scales for x with shape ``(M, K // quant_block_size)``.
        w_scale: E8M0 uint8 scales for w with shape ``(N, K // quant_block_size)``.
        quant_block_size: MXFP8 block size (typically 32).
        dtype: Output dtype.
        y: Optional pre-allocated output ``(M, N)``.

    Returns:
        Output tensor ``(M, N)`` in higher precision.
    """
    _LOGGER.info(
        f"GEMM_MXFP8: x={tuple(x.shape)} w={tuple(w.shape)} "
        f"x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)} "
        f"block_size={quant_block_size}"
    )

    assert x.shape[1] == w.shape[1], "Incompatible K dimensions"
    M, K = x.shape

    expected_scale_k = K // quant_block_size
    assert x_scale.shape == (
        M,
        expected_scale_k,
    ), f"x_scale shape {x_scale.shape} != expected ({M}, {expected_scale_k})"
    assert (
        w_scale.shape[1] == expected_scale_k
    ), f"w_scale K-dim {w_scale.shape[1]} != expected {expected_scale_k}"

    x_scale_fp32 = e8m0_to_fp32(x_scale)
    w_scale_fp32 = e8m0_to_fp32(w_scale)

    return gemm_a8w8_blockscale(
        x,
        w,
        x_scale_fp32,
        w_scale_fp32,
        dtype=dtype,
        y=y,
    )
