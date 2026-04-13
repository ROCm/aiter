# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from functools import cache

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.rmsnorm_input_quant_fp8 import (
    rms_norm_input_quant_fp8_kernel,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


def get_fp8_min_max_bounds(fp8_dtype: torch.dtype) -> tuple[float, float]:
    """Match vLLM ``quant_utils.get_fp8_min_max`` for ``fp8_dtype`` (incl. ROCm fnuz ±224)."""
    if fp8_dtype == torch.float8_e4m3fnuz:
        return -224.0, 224.0
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max)


@cache
def _num_compute_units(device_id: int = 0) -> int:
    """Match vLLM ``vllm.utils.platform_utils.num_compute_units`` (``current_platform.num_compute_units``)."""
    return torch.cuda.get_device_properties(device_id).multi_processor_count


def calc_rows_per_block(M: int, device: torch.device) -> int:
    """Same heuristic as vLLM ``input_quant_fp8.calc_rows_per_block``."""
    if device.type != "cuda":
        raise ValueError(
            "rmsnorm_input_quant_fp8 targets AMD ROCm (HIP); expected a CUDA/HIP device."
        )
    device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    sm_count = max(int(_num_compute_units(device_id)), 1)
    rows_per_block = triton.next_power_of_2(triton.cdiv(M, 2 * sm_count))
    return min(int(rows_per_block), 4)


def rmsnorm_input_quant_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor,
    eps: float,
    *,
    norm_before_gate: bool = True,
    use_ue8m0: bool = False,
    activation: str = "silu",
    out_dtype: torch.dtype | None = None,
    fp8_min: float | None = None,
    fp8_max: float | None = None,
    fp8_min_scaling_factor: float | None = None,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm (with optional bias), optional multiplicative gate from ``z``,
    and FP8 quantization (same contract as vLLM ``_rmsnorm_quantize_group_native`` for
    ``group_size == N``).

    ``x`` and ``z`` must be 2D contiguous with identical shape ``(M, N)``.
    Returns ``(x_quant_fp8, scales)`` where ``scales`` is ``(M,)`` float32 if
    ``group_size`` is ``None`` (one scale per row), or ``(M, N // group_size)`` float32
    when ``group_size`` divides ``N`` (one scale per row per column group).

    ``fp8_min`` / ``fp8_max`` / ``fp8_min_scaling_factor`` default from ``out_dtype`` (or
    ``get_fp8_e4m3_dtype()``) using the same rules as vLLM ``get_fp8_min_max`` and
    ``1.0 / (_FP8_MAX * 512)``. Pass them explicitly when you want to pin values (e.g. from
    vLLM's ``get_fp8_min_max()`` at model init).

    Raises:
        ValueError: if ``group_size`` is not ``None`` and ``group_size > N``,
            ``group_size <= 0``, or ``N`` is not divisible by ``group_size``.
    """
    assert x.is_contiguous() and z.is_contiguous()
    assert x.shape == z.shape, "x and z must have the same shape"
    fp8_dtype = out_dtype if out_dtype is not None else get_fp8_e4m3_dtype()
    if (fp8_min is None) ^ (fp8_max is None):
        raise ValueError("fp8_min and fp8_max must be passed together or both omitted.")
    if fp8_min is None:
        fp8_min, fp8_max = get_fp8_min_max_bounds(fp8_dtype)
    if fp8_min_scaling_factor is None:
        fp8_min_scaling_factor = 1.0 / (fp8_max * 512.0)

    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, N = x.shape
    if group_size is not None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if group_size > N:
            raise ValueError(
                f"group_size ({group_size}) must be less than or equal to hidden size "
                f"N ({N}); per-column FP8 groups cannot exceed the row width."
            )
        if N % group_size != 0:
            raise ValueError(
                f"hidden size N ({N}) must be divisible by group_size ({group_size})."
            )

    effective_gs = N if group_size is None else int(group_size)
    num_groups = N // effective_gs

    MAX_FUSED_SIZE = 65536 // x.element_size()
    if N > MAX_FUSED_SIZE:
        raise RuntimeError("This RMSNorm quant kernel does not support N >= 64KB.")

    rms_tile = min(512, triton.next_power_of_2(N))
    block_g = triton.next_power_of_2(effective_gs)
    rows_per_block = calc_rows_per_block(M, x.device)
    num_warps = min(max(block_g // 256, 1), 8)

    x_quant = torch.empty(M, N, dtype=fp8_dtype, device=x.device)
    if group_size is None:
        scales = torch.empty(M, dtype=torch.float32, device=x.device)
        stride_s_row = int(scales.stride(0))
        stride_s_g = 0
    else:
        scales = torch.empty(M, num_groups, dtype=torch.float32, device=x.device)
        stride_s_row, stride_s_g = (int(scales.stride(0)), int(scales.stride(1)))

    grid = (triton.cdiv(M, rows_per_block),)
    rms_norm_input_quant_fp8_kernel[grid](
        x,
        weight,
        bias,
        z,
        x_quant,
        scales,
        x.stride(0),
        z.stride(0),
        x_quant.stride(0),
        stride_s_row,
        stride_s_g,
        M,
        N,
        eps,
        RMS_TILE=rms_tile,
        ROWS_PER_BLOCK=rows_per_block,
        GROUP_SIZE=effective_gs,
        NUM_GROUPS=num_groups,
        BLOCK_G=block_g,
        NORM_BEFORE_GATE=norm_before_gate,
        FP8_MIN=fp8_min,
        FP8_MAX=fp8_max,
        USE_UE8M0=use_ue8m0,
        FP8_MIN_SCALING_FACTOR=fp8_min_scaling_factor,
        num_warps=num_warps,
        ACTIVATION=activation,
    )
    return x_quant, scales
