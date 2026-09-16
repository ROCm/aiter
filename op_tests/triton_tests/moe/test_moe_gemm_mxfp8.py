# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for moe_gemm_mxfp8.

Reference: dequantize inputs and run standard torch.matmul per expert,
then compare against kernel output within FP8-precision tolerance.
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_gemm_mxfp8 import moe_gemm_mxfp8

# E8M0 biased byte → FP32 scale: 2^(b - 127)
_E8M0_BIAS = 127


def _e8m0_to_fp32(b: torch.Tensor) -> torch.Tensor:
    return (b.to(torch.int32) << 23).view(torch.float32)


def _make_mxfp8_inputs(E, N, K, qbs, dtype=torch.float8_e4m3fnuz, device="cuda"):
    torch.manual_seed(0)
    total_tokens = E * 64  # 64 tokens per expert
    group_sizes = torch.full((E,), 64, dtype=torch.int32, device=device)

    lhs = torch.randint(-3, 4, (total_tokens, K), dtype=torch.int8, device=device).to(
        dtype
    )
    rhs = torch.randint(-3, 4, (E, N, K), dtype=torch.int8, device=device).to(dtype)
    # E8M0 scales: biased bytes in [120, 134] → 2^(-7..7)
    x_scale = torch.randint(
        120, 135, (total_tokens, K // qbs), dtype=torch.uint8, device=device
    )
    w_scale = torch.randint(
        120, 135, (E, N, K // qbs), dtype=torch.uint8, device=device
    )
    return lhs, rhs, x_scale, w_scale, group_sizes


def _mxfp8_reference(lhs, rhs, x_scale, w_scale, group_sizes, qbs, out_dtype):
    """Per-expert torch reference for moe_gemm_mxfp8."""
    E, N, K = rhs.shape
    total = lhs.shape[0]
    out = torch.zeros(total, N, dtype=out_dtype, device=lhs.device)
    offset = 0
    for e in range(E):
        m = int(group_sizes[e].item())
        if m == 0:
            continue
        a = lhs[offset : offset + m].float()
        b = rhs[e].float()  # [N, K]
        # dequant: scale per (token, k-block) and (n, k-block)
        xs = _e8m0_to_fp32(x_scale[offset : offset + m])  # [m, K//qbs]
        ws = _e8m0_to_fp32(w_scale[e])  # [N, K//qbs]
        # expand to full K
        xs_full = xs.repeat_interleave(qbs, dim=1)[:, :K]
        ws_full = ws.repeat_interleave(qbs, dim=1)[:, :K]
        a_dq = a * xs_full
        b_dq = b * ws_full  # [N, K]
        out[offset : offset + m] = (a_dq @ b_dq.T).to(out_dtype)
        offset += m
    return out


@pytest.mark.parametrize("E, N, K", [(4, 256, 128), (8, 128, 256)])
@pytest.mark.parametrize("qbs", [32])
def test_moe_gemm_mxfp8_correctness(E, N, K, qbs):
    lhs, rhs, x_scale, w_scale, group_sizes = _make_mxfp8_inputs(E, N, K, qbs)
    out = moe_gemm_mxfp8(lhs, rhs, x_scale, w_scale, group_sizes, quant_block_size=qbs)
    ref = _mxfp8_reference(lhs, rhs, x_scale, w_scale, group_sizes, qbs, torch.bfloat16)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    scale = ref.abs().max().clamp_min(1e-4)
    err = (out.float() - ref.float()).abs().max()
    assert err <= 0.2 * scale, f"max err {err:.4f} > 0.2 * {scale:.4f}"


@pytest.mark.parametrize("E, N, K", [(4, 256, 128), (8, 128, 256)])
def test_moe_gemm_mxfp8_with_bias(E, N, K):
    qbs = 32
    lhs, rhs, x_scale, w_scale, group_sizes = _make_mxfp8_inputs(E, N, K, qbs)
    bias = torch.randn(E, N, dtype=torch.bfloat16, device="cuda") * 0.1
    out = moe_gemm_mxfp8(
        lhs, rhs, x_scale, w_scale, group_sizes, quant_block_size=qbs, bias=bias
    )
    out_no_bias = moe_gemm_mxfp8(
        lhs, rhs, x_scale, w_scale, group_sizes, quant_block_size=qbs
    )
    assert out.shape == (lhs.shape[0], N)
    # bias should change the output
    assert not torch.allclose(out, out_no_bias)


def test_moe_gemm_mxfp8_empty_tokens():
    """Zero total_tokens returns empty output without error."""
    E, N, K, qbs = 4, 128, 64, 32
    lhs = torch.empty(0, K, dtype=torch.float8_e4m3fnuz, device="cuda")
    rhs = torch.randint(-3, 4, (E, N, K), dtype=torch.int8, device="cuda").to(
        torch.float8_e4m3fnuz
    )
    x_scale = torch.empty(0, K // qbs, dtype=torch.uint8, device="cuda")
    w_scale = torch.randint(
        127, 128, (E, N, K // qbs), dtype=torch.uint8, device="cuda"
    )
    group_sizes = torch.zeros(E, dtype=torch.int32, device="cuda")
    out = moe_gemm_mxfp8(lhs, rhs, x_scale, w_scale, group_sizes, quant_block_size=qbs)
    assert out.shape == (0, N)
