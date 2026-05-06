# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Literal, Optional

import torch
import triton

from aiter import dtypes as aiter_dtypes
from aiter.ops.triton._triton_kernels.fusions.fused_clamp_act_mul_quant import (
    _fused_clamp_silu_mul_fp8_group_quant_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_clamp_act_mul_fp8_group_quant(
    inp: torch.Tensor,
    out_fp8: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    swiglu_limit: float = 0,
    activation: Literal["silu", "gelu", "gelu_tanh"] = "silu",
    weights: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype | None = None,
    transpose_scale: bool = False,
) -> None:
    """
    Fused clamp (SwiGLU-style) + SiLU(gate) * up + optional weights + FP8 group quant.

    Args:
        inp: ``[M, D]`` with ``D = 2 * N``, contiguous; first ``N`` columns are gate,
            second ``N`` are up (same as ``chunk(2, dim=-1)`` on gate-up GEMM output).
        out_fp8: pre-allocated ``[M, N]`` FP8 tensor (typically ``aiter_dtypes.fp8``).
        scale: pre-allocated ``[M, (N + 127) // 128]`` float32 block scales.
        swiglu_limit: if ``> 0``, apply reference clamps; if ``<= 0``, skip clamping.
        weights: optional ``[M, 1]`` (broadcast) or ``[M, N]`` row weights, multiplied
            into ``silu(gate) * up`` (same as reference ``weights * x``).
        dtype_quant: ignored; scales and FP8 layout follow ``out_fp8.dtype``.

    Constraints:
        ``N`` must be a power of two, ``N >= 128``, and ``N % 128 == 0`` so each row
        uses one ``_fp8_quant_op`` tile (``BLOCK_SIZE_M=1``, ``BLOCK_SIZE_N=N``).
    """
    if dtype_quant is not None and out_fp8 is not None and dtype_quant != out_fp8.dtype:
        _LOGGER.info(
            "fused_clamp_act_mul_quant: dtype_quant=%s ignored; using out_fp8.dtype=%s",
            dtype_quant,
            out_fp8.dtype,
        )

    assert inp.dim() == 2
    M, D = inp.shape
    assert D % 2 == 0
    n_half = D // 2
    if out_fp8 is None:
        out_fp8 = torch.empty((M, n_half), dtype=dtype_quant, device=inp.device)
    else:
        assert out_fp8.shape == (M, n_half)
    num_blocks = (n_half + 127) // 128
    if scale is None:
        if transpose_scale:
            scale = torch.empty((num_blocks, M), dtype=torch.float32, device=inp.device)
        else:
            scale = torch.empty((M, num_blocks), dtype=torch.float32, device=inp.device)
    else:
        if transpose_scale:
            assert scale.shape == (num_blocks, M)
        else:
            assert scale.shape == (M, num_blocks)

    assert n_half >= 128
    assert (n_half & (n_half - 1)) == 0, "N=D//2 must be a power of 2 for this kernel"
    assert n_half % 128 == 0

    BLOCK_SIZE_N = triton.next_power_of_2(n_half)
    assert BLOCK_SIZE_N == n_half

    HAVE_WEIGHTS = weights is not None
    if HAVE_WEIGHTS:
        assert weights.is_cuda and weights.is_contiguous()
        assert weights.shape[0] == M
        if weights.shape[1] == 1:
            WEIGHT_BROADCAST = True
        else:
            assert weights.shape[1] == n_half
            WEIGHT_BROADCAST = False
    else:
        WEIGHT_BROADCAST = False

    DTYPE_MAX = (
        torch.finfo(out_fp8.dtype).max
        if torch.is_floating_point(out_fp8)
        else float(torch.iinfo(out_fp8.dtype).max)
    )

    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    else:
        num_warps = 8

    HAVE_SWIGLU_CLAMP = swiglu_limit > 0

    if transpose_scale:
        scale_row_stride = scale.stride(1)
        scale_col_stride = scale.stride(0)
        num_bs_cols = scale.shape[0]
    else:
        scale_row_stride = scale.stride(0)
        scale_col_stride = scale.stride(1)
        num_bs_cols = scale.shape[1]

    _fused_clamp_silu_mul_fp8_group_quant_kernel[(M,)](
        inp,
        out_fp8,
        scale,
        weights if HAVE_WEIGHTS else inp,
        M,
        n_half,
        inp.stride(0),
        inp.stride(1),
        out_fp8.stride(0),
        out_fp8.stride(1),
        scale_row_stride,
        scale_col_stride,
        weights.stride(0) if HAVE_WEIGHTS else 0,
        weights.stride(1) if HAVE_WEIGHTS else 0,
        swiglu_limit,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=128,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_WEIGHTS=HAVE_WEIGHTS,
        WEIGHT_BROADCAST=WEIGHT_BROADCAST,
        HAVE_SWIGLU_CLAMP=HAVE_SWIGLU_CLAMP,
        ACTIVATION=activation,
        num_warps=num_warps,
    )
    if transpose_scale:
        scale = scale.view(M, num_bs_cols)

    return out_fp8, scale
