# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Python wrappers for FP8 block-wise quantization kernels."""

import math

import torch

from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
    quant_fp8_blockwise_for_weight_kernel,
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
    requant_fp8_row_to_col_kernel,
)

__all__ = [
    "quant_fp8_blockwise",
    "quant_fp8_blockwise_for_act_grad",
    "quant_fp8_blockwise_for_weight",
    "quant_fp8_blockwise_segment_m",
    "requant_fp8_row_to_col",
]

_FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max
_BLOCK_SIZE = 128


def _check_block_fp8(block_size: int, fp8_max: float) -> None:
    # BLOCK_SIZE feeds tl.arange(0, BLOCK_SIZE), which needs a power of two.
    assert (
        block_size > 0 and (block_size & (block_size - 1)) == 0
    ), f"block_size must be a positive power of two, got {block_size}"
    # Output is float8_e4m3fnuz; a bound above its max would clamp against a
    # range the payload cannot represent.
    assert (
        0 < fp8_max <= _FP8_MAX
    ), f"fp8_max must be in (0, {_FP8_MAX}] for float8_e4m3fnuz, got {fp8_max}"


def quant_fp8_blockwise(
    x: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    fp8_max: float = _FP8_MAX,
    axis: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization of a 2-D tensor.

    Args:
        x:          Contiguous input tensor ``[M, N]``, any float dtype.
        block_size: Quantization block size (positive power of two).
        fp8_max:    FP8 dynamic range maximum (default: e4m3fnuz max); must be
                    in ``(0, e4m3fnuz max]``.
        axis:       Scale axis — ``1`` = row-wise (one scale per row-block),
                    ``0`` = col-wise (one scale per col-block).

    Returns:
        ``(x_fp8, scales)`` where ``x_fp8`` is ``[M, N]`` float8_e4m3fnuz and
        ``scales`` are the per-block *inverse* quantisation scales.
    """
    # The kernel flat-indexes x as contiguous; a strided tensor would read/write
    # the wrong layout silently.
    assert (
        x.ndim == 2 and x.is_contiguous()
    ), f"expected 2-D contiguous input, got shape {x.shape} strides {x.stride()}"
    assert axis in (0, 1), f"axis must be 0 or 1, got {axis}"
    _check_block_fp8(block_size, fp8_max)
    M, N = x.shape
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    if axis == 1:
        scales = torch.empty(
            M, math.ceil(N / block_size), dtype=torch.float32, device=x.device
        )
    else:
        scales = torch.empty(
            math.ceil(M / block_size), N, dtype=torch.float32, device=x.device
        )

    grid = (math.ceil(M / block_size), math.ceil(N / block_size))
    quant_fp8_blockwise_kernel[grid](
        x,
        x_fp8,
        scales,
        x_fp8,  # col outputs unused when DUAL=False (constexpr-pruned)
        scales,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        AXIS=axis,
        DUAL=False,
    )
    return x_fp8, scales


def quant_fp8_blockwise_segment_m(
    x: torch.Tensor,
    batch_size: int,
    seg_indptr: torch.Tensor,
    scales_seg_indptr: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    fp8_max: float = _FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Col-wise (BLOCK×1) FP8 quantization over variable-length row segments.

    Each segment (grouped-GEMM / MoE group) is quantized independently: its
    rows are split into ``block_size`` row-blocks and one scale is produced per
    (row-block, column). Segments need not align to ``block_size``; the kernel
    skips empty tiles.

    Args:
        x:                 ``[M, N]`` contiguous input (``M`` = sum of segment
                           lengths).
        batch_size:        Number of segments.
        seg_indptr:        ``[batch_size + 1]`` int tensor of cumulative row
                           offsets per segment.
        scales_seg_indptr: ``[batch_size + 1]`` int tensor of cumulative
                           row-block counts per segment.
        block_size:        Quantization block size (positive power of two).
        fp8_max:           FP8 bound in ``(0, e4m3fnuz max]``.

    Returns:
        ``(x_fp8, scales)`` where ``x_fp8`` is ``[M, N]`` float8_e4m3fnuz and
        ``scales`` is ``[ceil(M / block_size) + batch_size, N]`` (an upper bound
        on the total row-block count; unused trailing rows are left untouched).
    """
    assert (
        x.ndim == 2 and x.is_contiguous()
    ), f"expected 2-D contiguous input, got shape {x.shape} strides {x.stride()}"
    _check_block_fp8(block_size, fp8_max)
    M, N = x.shape
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    scales = torch.empty(
        math.ceil(M / block_size) + batch_size,
        N,
        dtype=torch.float32,
        device=x.device,
    )
    grid = (math.ceil(M / block_size) + batch_size, math.ceil(N / block_size))
    quant_fp8_blockwise_segment_m_kernel[grid](
        x,
        x_fp8,
        scales,
        N,
        batch_size,
        seg_indptr,
        scales_seg_indptr,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
    )
    return x_fp8, scales


def quant_fp8_blockwise_for_weight(
    w: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    fp8_max: float = _FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization for a batched weight tensor.

    Args:
        w:          Contiguous weight tensor ``[B, M, N]`` or ``[M, N]``.
        block_size: Quantization block size (positive power of two).
        fp8_max:    FP8 bound in ``(0, e4m3fnuz max]``.

    Returns:
        ``(w_fp8, scales)`` where each block has its own scale.
    """
    if w.ndim == 2:
        w = w.unsqueeze(0)
    # The kernel addresses w as dense [B, M, N] (bid*M*N + row*N + col); a
    # transposed/sliced weight would be quantized with the wrong layout.
    assert w.ndim == 3 and w.is_contiguous(), (
        f"expected contiguous 2-D or 3-D weight, got shape {w.shape} "
        f"strides {w.stride()}"
    )
    _check_block_fp8(block_size, fp8_max)
    B, M, N = w.shape
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fnuz)
    scales = torch.empty(
        B,
        math.ceil(M / block_size),
        math.ceil(N / block_size),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (B, math.ceil(M / block_size), math.ceil(N / block_size))
    quant_fp8_blockwise_for_weight_kernel[grid](
        w,
        w_fp8,
        scales,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
    )
    return w_fp8, scales


def quant_fp8_blockwise_for_act_grad(
    x: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    fp8_max: float = _FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual row+col FP8 quantization for activation gradients.

    Produces both a row-wise (1×BLOCK) and col-wise (BLOCK×1) FP8 copy of
    the input in a single pass — needed by blockwise WGrad backward.

    Args:
        x:          Contiguous input tensor ``[M, N]``.
        block_size: Quantization block size (positive power of two).
        fp8_max:    FP8 bound in ``(0, e4m3fnuz max]``.

    Returns:
        ``(x_fp8_row, scales_row, x_fp8_col, scales_col)``
    """
    assert (
        x.ndim == 2 and x.is_contiguous()
    ), f"expected 2-D contiguous input, got shape {x.shape} strides {x.stride()}"
    _check_block_fp8(block_size, fp8_max)
    M, N = x.shape
    x_fp8_row = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    x_fp8_col = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    scales_row = torch.empty(
        M, math.ceil(N / block_size), dtype=torch.float32, device=x.device
    )
    scales_col = torch.empty(
        math.ceil(M / block_size), N, dtype=torch.float32, device=x.device
    )

    # Same kernel as quant_fp8_blockwise, with DUAL=True to emit the col copy
    # (axis=0) alongside the row copy (axis=1) from a single tile load.
    grid = (math.ceil(M / block_size), math.ceil(N / block_size))
    quant_fp8_blockwise_kernel[grid](
        x,
        x_fp8_row,
        scales_row,
        x_fp8_col,
        scales_col,
        M,
        N,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
        AXIS=1,
        DUAL=True,
    )
    return x_fp8_row, scales_row, x_fp8_col, scales_col


def requant_fp8_row_to_col(
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    block_size: int = _BLOCK_SIZE,
    fp8_max: float = _FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize FP8 row-wise (1×BLOCK) to col-wise (BLOCK×1) in one pass.

    Dequantises with the saved row scales then re-quantises along the column
    axis without a BF16 roundtrip.  Used in blockwise WGrad backward.

    Args:
        x_fp8:    ``[M, K]`` contiguous FP8 input with row-wise block
                  quantisation.
        x_scales: ``[M, ceil(K / block_size)]`` dequant scales for ``x_fp8``.
        block_size: Quantization block size (positive power of two).
        fp8_max:    FP8 bound in ``(0, e4m3fnuz max]``.

    Returns:
        ``(y_fp8, y_scales)`` col-wise quantised, shapes ``[M, K]`` and
        ``[ceil(M / block_size), K]``.
    """
    assert x_fp8.ndim == 2 and x_fp8.is_contiguous(), (
        f"expected 2-D contiguous input, got shape {x_fp8.shape} "
        f"strides {x_fp8.stride()}"
    )
    _check_block_fp8(block_size, fp8_max)
    M, K = x_fp8.shape
    # x_scales is read with flat contiguous indexing at [M, ceil(K/block)].
    expected_scale_shape = (M, math.ceil(K / block_size))
    assert x_scales.is_contiguous() and tuple(x_scales.shape) == expected_scale_shape, (
        f"x_scales shape {tuple(x_scales.shape)} != {expected_scale_shape} "
        "or not contiguous"
    )
    assert x_scales.device == x_fp8.device, "x_fp8 and x_scales must share a device"
    y_fp8 = torch.empty_like(x_fp8)
    y_scales = torch.empty(
        math.ceil(M / block_size), K, dtype=torch.float32, device=x_fp8.device
    )

    grid = (math.ceil(M / block_size), math.ceil(K / block_size))
    requant_fp8_row_to_col_kernel[grid](
        x_fp8,
        x_scales,
        y_fp8,
        y_scales,
        M,
        K,
        BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max,
    )
    return y_fp8, y_scales
