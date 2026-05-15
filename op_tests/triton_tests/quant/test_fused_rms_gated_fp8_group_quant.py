# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``fused_rms_gated_fp8_group_quant`` (kernel in ``_triton_kernels/quant/fused_fp8_quant``)."""

import pytest
import torch

from aiter.ops.triton.quant.fused_fp8_quant import (
    fused_rms_gated_fp8_group_quant,
    get_fp8_min_max_bounds,
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
    group_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x32 = x.float()
    z32 = z.float()
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
    gs = x.shape[1] if group_size is None else group_size
    ng = x.shape[1] // gs
    yg = y.view(y.shape[0], ng, gs)
    scales = yg.abs().amax(dim=-1).clamp_min(1e-12) / fmax
    y_scaled = yg / scales.unsqueeze(-1)
    q = y_scaled.clamp(fmin, fmax).to(fp8_dtype).view_as(y)
    if group_size is None:
        scales = scales.squeeze(-1)
    return q, scales


def _scale_broadcast(
    scales: torch.Tensor, N: int, group_size: int | None
) -> torch.Tensor:
    if group_size is None:
        return scales.unsqueeze(-1).expand(-1, N)
    return scales.repeat_interleave(group_size, dim=1)


@cuda_ok
def test_fused_rms_gated_fp8_group_quant_matches_ref():
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

    y_q, scales_t = fused_rms_gated_fp8_group_quant(
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
        x, w, bias, z, 1e-5, True, "silu", fmin, fmax, None
    )

    torch.testing.assert_close(scales_t, scales_ref, rtol=1e-3, atol=1e-3)
    sb = _scale_broadcast(scales_ref, N, None)
    dq = y_q.float() * sb
    dq_ref = y_ref.float() * sb
    torch.testing.assert_close(dq, dq_ref, rtol=0.15, atol=0.15)

    y_default, scales_default = fused_rms_gated_fp8_group_quant(
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


_MS = [1, 3, 4, 512, 1024, 2048, 4096]
_NS = [128, 256]
_GROUP_SIZES = {
    128: [1, 2, 4, 8, 16, 32, 64, 128],
    256: [1, 2, 4, 8, 16, 32, 64, 128, 256],
}


def _sweep_cases():
    out = []
    for N in _NS:
        for M in _MS:
            for g in _GROUP_SIZES[N]:
                out.append(pytest.param(M, N, g, id=f"M{M}-N{N}-g{g}"))
    return out


@cuda_ok
@pytest.mark.parametrize(("M", "N", "group_size"), _sweep_cases())
def test_fused_rms_gated_fp8_group_quant_sweep(M: int, N: int, group_size: int):
    device = "cuda"
    torch.manual_seed(1)
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    z = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, device=device, dtype=torch.bfloat16)
    bias = torch.randn(N, device=device, dtype=torch.bfloat16)
    fmin, fmax = get_fp8_min_max_bounds(get_fp8_e4m3_dtype())
    scale_floor = 1.0 / (fmax * 512.0)

    y_q, scales_t = fused_rms_gated_fp8_group_quant(
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
        group_size=group_size,
    )
    y_ref, scales_ref = ref_rmsnorm_quant(
        x, w, bias, z, 1e-5, True, "silu", fmin, fmax, group_size
    )

    assert scales_t.shape == scales_ref.shape
    torch.testing.assert_close(scales_t, scales_ref, rtol=1e-3, atol=1e-3)
    sb = _scale_broadcast(scales_ref, N, group_size)
    dq = y_q.float() * sb
    dq_ref = y_ref.float() * sb
    torch.testing.assert_close(dq, dq_ref, rtol=0.15, atol=0.15)


@cuda_ok
def test_fused_rms_gated_fp8_group_quant_group_size_errors():
    device = "cuda"
    x = torch.randn(2, 128, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    w = torch.randn(128, device=device, dtype=torch.bfloat16)
    b = torch.randn(128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="less than or equal to hidden size"):
        fused_rms_gated_fp8_group_quant(x, w, b, z, 1e-5, group_size=256)
    with pytest.raises(ValueError, match="divisible by group_size"):
        fused_rms_gated_fp8_group_quant(x, w, b, z, 1e-5, group_size=48)
    with pytest.raises(ValueError, match="positive"):
        fused_rms_gated_fp8_group_quant(x, w, b, z, 1e-5, group_size=0)
