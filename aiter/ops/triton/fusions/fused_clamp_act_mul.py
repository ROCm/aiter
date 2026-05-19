# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Literal, Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_clamp_act_mul import (
    _fused_clamp_silu_mul_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_clamp_act_mul(
    inp: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    swiglu_limit: float = 0,
    activation: Literal["silu", "gelu", "gelu_tanh"] = "silu",
    weights: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype | None = None,
    transpose_scale: bool = False,
    quant_block_size: int = 128,
    scale_dtype_fmt: Literal["fp32", "ue8m0"] = "fp32",
    shuffle_scale: bool = False,
):
    """
    Fused clamp (SwiGLU-style) + act(gate) * up + optional weights, with optional FP8 group quant.

    Args:
        inp: ``[M, D]`` with ``D = 2 * N``, contiguous; first ``N`` columns are gate,
            second ``N`` are up (same as ``chunk(2, dim=-1)`` on gate-up GEMM output).
        out: pre-allocated ``[M, N]`` output tensor. If ``None``, allocated internally
            with dtype = ``dtype_quant`` when quantizing, else ``inp.dtype``.
        scale: pre-allocated ``[M, (N + 127) // 128]`` float32 block scales. Only used
            and returned when ``dtype_quant`` is not ``None``.
        swiglu_limit: if ``> 0``, apply reference clamps; if ``<= 0``, skip clamping.
        weights: optional ``[M, 1]`` (broadcast) or ``[M, N]`` row weights, multiplied
            into ``silu(gate) * up`` (same as reference ``weights * x``).
        dtype_quant: if ``None``, no quantization; output is written in ``inp.dtype``
            (or the dtype of a pre-allocated ``out``) and ``scale`` is unused. Otherwise
            the result is FP8-group-quantized with ``dtype_quant`` and per-128 scales.

    Constraints:
        ``N`` must be a power of two, ``N >= 128``, and ``N % 128 == 0`` so each row
        uses one ``_fp8_quant_op`` tile (``BLOCK_SIZE_M=1``, ``BLOCK_SIZE_N=N``).
    """
    assert inp.dim() == 2
    M, D = inp.shape
    assert D % 2 == 0
    n_half = D // 2

    HAS_QUANT = dtype_quant is not None

    # Step 5 ue8m0 mode: per-1×32 group quant, uint8 scale.
    assert scale_dtype_fmt in ("fp32", "ue8m0")
    if scale_dtype_fmt == "ue8m0":
        assert HAS_QUANT, "scale_dtype_fmt='ue8m0' requires dtype_quant"
        assert (
            quant_block_size == 32
        ), f"ue8m0 requires quant_block_size=32 got {quant_block_size}"
        assert dtype_quant in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ), f"ue8m0 requires fp8 e4m3, got {dtype_quant}"
        if shuffle_scale and transpose_scale:
            raise ValueError("shuffle_scale incompatible with transpose_scale")
        _scale_storage_dtype = torch.uint8
    else:
        if shuffle_scale:
            raise ValueError("shuffle_scale only valid with scale_dtype_fmt='ue8m0'")
        _scale_storage_dtype = torch.float32

    if HAS_QUANT:
        if out is None:
            out = torch.empty((M, n_half), dtype=dtype_quant, device=inp.device)
        else:
            assert out.shape == (M, n_half)
            if out.dtype != dtype_quant:
                _LOGGER.info(
                    "fused_clamp_act_mul: dtype_quant=%s ignored; using out.dtype=%s",
                    dtype_quant,
                    out.dtype,
                )
        num_blocks = (n_half + quant_block_size - 1) // quant_block_size
        if scale is None:
            if transpose_scale:
                scale = torch.empty(
                    (num_blocks, M), dtype=_scale_storage_dtype, device=inp.device
                )
            else:
                scale = torch.empty(
                    (M, num_blocks), dtype=_scale_storage_dtype, device=inp.device
                )
        else:
            if transpose_scale:
                assert scale.shape == (num_blocks, M)
            else:
                assert scale.shape == (M, num_blocks)
    else:
        if out is None:
            out = torch.empty((M, n_half), dtype=inp.dtype, device=inp.device)
        else:
            assert out.shape == (M, n_half)

    assert n_half >= 128
    assert n_half % 128 == 0

    BLOCK_SIZE_N = triton.next_power_of_2(n_half)

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

    if HAS_QUANT:
        DTYPE_MAX = (
            torch.finfo(out.dtype).max
            if torch.is_floating_point(out)
            else float(torch.iinfo(out.dtype).max)
        )
    else:
        DTYPE_MAX = 0.0

    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    else:
        num_warps = 8

    HAVE_SWIGLU_CLAMP = swiglu_limit > 0

    if HAS_QUANT:
        if transpose_scale:
            scale_row_stride = scale.stride(1)
            scale_col_stride = scale.stride(0)
            num_bs_cols = scale.shape[0]
        else:
            scale_row_stride = scale.stride(0)
            scale_col_stride = scale.stride(1)
            num_bs_cols = scale.shape[1]
        scale_arg = scale
    else:
        scale_row_stride = 0
        scale_col_stride = 0
        scale_arg = inp  # placeholder, unused when HAS_QUANT is False

    _fused_clamp_silu_mul_kernel[(M,)](
        inp,
        out,
        scale_arg,
        weights if HAVE_WEIGHTS else inp,
        M,
        n_half,
        inp.stride(0),
        inp.stride(1),
        out.stride(0),
        out.stride(1),
        scale_row_stride,
        scale_col_stride,
        weights.stride(0) if HAVE_WEIGHTS else 0,
        weights.stride(1) if HAVE_WEIGHTS else 0,
        swiglu_limit,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=quant_block_size,
        SCALE_FMT=scale_dtype_fmt,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_WEIGHTS=HAVE_WEIGHTS,
        WEIGHT_BROADCAST=WEIGHT_BROADCAST,
        HAVE_SWIGLU_CLAMP=HAVE_SWIGLU_CLAMP,
        HAS_QUANT=HAS_QUANT,
        ACTIVATION=activation,
        num_warps=num_warps,
    )

    if HAS_QUANT:
        if transpose_scale:
            scale = scale.view(M, num_bs_cols)
        if shuffle_scale:
            from aiter.utility import fp4_utils

            scale = fp4_utils.e8m0_shuffle(scale)
        return out, scale
    return out
