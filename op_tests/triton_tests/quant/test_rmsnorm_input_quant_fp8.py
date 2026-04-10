# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``rmsnorm_input_quant_fp8`` (kernel lives under ``_triton_kernels/quant``)."""

import pytest
import torch

from aiter.ops.triton.quant.rmsnorm_input_quant_fp8 import (
    get_fp8_min_max_bounds,
    rmsnorm_input_quant_fp8,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

cuda_ok = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device required"
)


def ref_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor,
    eps: float,
    norm_before_gate: bool,
    activation: str,
    fmin: float,
    fmax: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x32 = x.float()
    z32 = z.float()
    n = x.shape[-1]
    var = x32.pow(2).mean(-1, keepdim=True)
    x_hat = x32 * torch.rsqrt(var + eps)
    y = x_hat * weight.float()
    if bias is not None:
        y = y + bias.float()
    if norm_before_gate:
        if activation in ("silu", "swish"):
            y = y * (z32 * torch.sigmoid(z32))
        elif activation == "sigmoid":
            y = y * torch.sigmoid(z32)
    fp8_dtype = get_fp8_e4m3_dtype()
    scales = y.abs().amax(dim=-1).clamp_min(1e-12) / fmax
    y_scaled = y / scales.unsqueeze(-1)
    q = y_scaled.clamp(fmin, fmax).to(fp8_dtype)
    return q, scales


@cuda_ok
def test_rmsnorm_input_quant_fp8_matches_ref():
    device = "cuda"
    torch.manual_seed(0)
    M, N = 32, 64
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    z = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, device=device, dtype=torch.bfloat16)
    bias = torch.randn(N, device=device, dtype=torch.bfloat16)

    fp8_dtype = get_fp8_e4m3_dtype()
    fmin, fmax = get_fp8_min_max_bounds(fp8_dtype)
    scale_floor = 1.0 / (fmax * 512.0)

    y_q, scales_t = rmsnorm_input_quant_fp8(
        x,
        w,
        bias,
        z,
        1e-5,
        norm_before_gate=True,
        use_ue8m0=False,
        activation="silu",
        fp8_min=fmin,
        fp8_max=fmax,
        fp8_min_scaling_factor=scale_floor,
    )
    y_ref, scales_ref = ref_rmsnorm_quant(
        x, w, bias, z, 1e-5, True, "silu", fmin, fmax
    )

    torch.testing.assert_close(scales_t, scales_ref, rtol=1e-3, atol=1e-3)
    dq = y_q.float() * scales_t.unsqueeze(-1)
    dq_ref = y_ref.float() * scales_ref.unsqueeze(-1)
    torch.testing.assert_close(dq, dq_ref, rtol=0.15, atol=0.15)

    y_default, scales_default = rmsnorm_input_quant_fp8(
        x,
        w,
        bias,
        z,
        1e-5,
        norm_before_gate=True,
        use_ue8m0=False,
        activation="silu",
    )
    torch.testing.assert_close(scales_t, scales_default, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y_q.float(), y_default.float(), rtol=0.0, atol=0.0)
