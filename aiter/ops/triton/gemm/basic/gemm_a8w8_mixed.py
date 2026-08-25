# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Mixed-dtype FP8 GEMM: supports operands with different FP8 types
# (e.g. E5M2 grad x E4M3 weight for hybrid backward pass).
# Uses the same Triton kernel as gemm_a8w8 — tl.dot accumulates in FP32
# regardless of input element types, so mixed E4M3/E5M2 is valid.


import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import (
    _gemm_a8w8_kernel,
    _get_config,
)
from aiter.ops.triton.utils.device_info import get_num_xcds


def gemm_a8w8_mixed(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    w_transposed: bool = False,
):
    """Mixed-dtype FP8 GEMM: Y = (X @ W) * (x_scale * w_scale).

    Unlike ``gemm_a8w8``, this function explicitly allows ``x`` and ``w``
    to have **different** FP8 dtypes (e.g. ``float8_e5m2fnuz`` and
    ``float8_e4m3fnuz``).  The underlying Triton kernel accumulates in
    FP32 via ``tl.dot(..., input_precision="ieee")``, so the operand
    element type only affects the load precision — the computation is
    always FP32.

    On MI300X (gfx942), MFMA FP8 instructions support per-operand format
    selection, so mixed E4M3/E5M2 is handled natively by the hardware.

    Args:
        x: Activation / LHS tensor ``(M, K)`` in any FP8 dtype.
        w: Weight tensor. If ``w_transposed=False``: ``(N, K)`` — standard
           PyTorch convention. If ``w_transposed=True``: ``(K, N)`` — already
           transposed (avoids a contiguous copy for pre-transposed weights).
        x_scale: Per-row or per-tensor scale for ``x``.
        w_scale: Per-row or per-tensor scale for ``w``.
        bias: Optional bias ``(N,)``.
        dtype: Output dtype (default ``bfloat16``).
        y: Pre-allocated output ``(M, N)``.
        config: Kernel tuning parameters.
        w_transposed: If True, ``w`` is already ``(K, N)`` layout.

    Returns:
        Output tensor ``(M, N)`` in ``dtype``.
    """
    assert (
        x.dtype.is_floating_point and w.dtype.is_floating_point
    ), f"Expected FP8 inputs, got {x.dtype} and {w.dtype}"

    M, K = x.shape
    if w_transposed:
        assert w.shape[0] == K, "Incompatible dimensions"
        N = w.shape[1]
    else:
        assert w.shape[1] == K, "Incompatible dimensions"
        N = w.shape[0]
        w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config, _ = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        **config,
    )

    return y
