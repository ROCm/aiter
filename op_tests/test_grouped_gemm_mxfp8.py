# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and guard tests for ``grouped_gemm_mxfp8_hip_fwd`` (gfx950).

Reference: dequantize the **same** MXFP8 e4m3 payload + e8m0 1×32 block scales the
kernel consumes, then bf16 matmul in fp32 — same bar style as decode small-M MXFP8 PRs.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

from aiter.ops.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant
from aiter.utility.fp4_utils import e8m0_to_f32


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    arch = str(getattr(p, "gcn_arch_name", None) or getattr(p, "gcnArchName", "") or "")
    return "gfx950" in arch.lower()


requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(),
    reason="grouped_gemm_mxfp8 HIP tile requires gfx950 (device path under __gfx950__)",
)


def _mxfp8_dequant_blocked(
    x_fp8: torch.Tensor,
    scale_fp8: torch.Tensor,
    *,
    block: int = 32,
) -> torch.Tensor:
    """OCP MXFP8: per-block scale (e8m0) × fp8 e4m3 block values → fp32."""
    assert x_fp8.shape[-1] % block == 0
    *lead, k = x_fp8.shape
    nb = k // block
    assert scale_fp8.shape == tuple(lead + [nb]), (scale_fp8.shape, tuple(lead + [nb]))
    s_f32 = e8m0_to_f32(scale_fp8.view(torch.uint8)).view(*lead, nb, 1)
    x_f = x_fp8.reshape(*lead, nb, block).float()
    return (x_f * s_f32).reshape(*lead, k)


def _grouped_ref_g1(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """G=1 NT GEMM: C[m,n] = sum_k dq(a)[m,k] * dq(b)[0,n,k]."""
    a_dq = _mxfp8_dequant_blocked(a, a_scale)
    b_dq = _mxfp8_dequant_blocked(b[0], b_scale[0])
    return a_dq @ b_dq.transpose(0, 1)


@requires_gfx950
def test_grouped_gemm_mxfp8_matches_mx_dequant_ref():
    device = "cuda"
    m_total, n, k, g = 512, 256, 384, 1
    tiles_m = (m_total + 255) // 256
    tiles_n = (n + 255) // 256
    total_tiles = tiles_m * tiles_n

    torch.manual_seed(1)
    a_bf16 = (
        torch.randn(m_total, k, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    a, a_s_u8 = dynamic_mxfp8_quant(a_bf16)
    a_scale = a_s_u8.view(torch.float8_e8m0fnu)

    b_bf16 = (
        torch.randn(g * n, k, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    b_flat, b_s_u8 = dynamic_mxfp8_quant(b_bf16)
    b = b_flat.view(g, n, k).contiguous()
    b_scale = b_s_u8.view(g, n, k // 32).contiguous().view(torch.float8_e8m0fnu)

    group_offs = torch.tensor([0, m_total], device=device, dtype=torch.int64)
    tile_offs = torch.tensor([0, total_tiles], device=device, dtype=torch.int32)
    block_to_expert = torch.zeros(total_tiles, device=device, dtype=torch.int32)

    out = grouped_gemm_mxfp8_hip_fwd(
        a,
        b,
        a_scale,
        b_scale,
        group_offs,
        block_to_expert,
        tile_offs,
        torch.bfloat16,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()

    ref = _grouped_ref_g1(a, b, a_scale, b_scale)
    got = out.float().reshape(-1)
    r = ref.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(r.unsqueeze(0), got.unsqueeze(0)).item()
    assert cos >= 0.999, f"cosine_similarity={cos}"
    den = r.abs().clamp(min=1e-4)
    max_rel = ((got - r).abs() / den).max().item()
    assert max_rel < 0.05, f"max_rel_err={max_rel}"


@requires_gfx950
def test_grouped_gemm_mxfp8_k_below_min_raises():
    device = "cuda"
    m_total, n, k_bad, g = 256, 128, 320, 1  # 320 % 32 == 0 but < 384
    a = torch.zeros(m_total, k_bad, device=device, dtype=torch.float8_e4m3fn)
    b = torch.zeros(g, n, k_bad, device=device, dtype=torch.float8_e4m3fn)
    sc = (k_bad + 31) // 32
    z = torch.zeros(m_total, sc, device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    zb = torch.zeros(g, n, sc, device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    go = torch.tensor([0, m_total], device=device, dtype=torch.int64)
    to = torch.tensor([0, 1], device=device, dtype=torch.int32)
    b2e = torch.zeros(1, device=device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="K must be >= 384"):
        grouped_gemm_mxfp8_hip_fwd(a, b, z, zb, go, b2e, to, torch.bfloat16)


@requires_gfx950
def test_grouped_gemm_mxfp8_k_not_multiple_of_32_raises():
    device = "cuda"
    m_total, n, k_bad, g = 256, 128, 370, 1  # 370 % 32 != 0
    a = torch.zeros(m_total, k_bad, device=device, dtype=torch.float8_e4m3fn)
    b = torch.zeros(g, n, k_bad, device=device, dtype=torch.float8_e4m3fn)
    sc = (k_bad + 31) // 32
    z = torch.zeros(m_total, sc, device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    zb = torch.zeros(g, n, sc, device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    go = torch.tensor([0, m_total], device=device, dtype=torch.int64)
    to = torch.tensor([0, 1], device=device, dtype=torch.int32)
    b2e = torch.zeros(1, device=device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="K must be multiple"):
        grouped_gemm_mxfp8_hip_fwd(a, b, z, zb, go, b2e, to, torch.bfloat16)
