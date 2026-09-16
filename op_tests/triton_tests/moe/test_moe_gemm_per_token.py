# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for moe_gemm_per_token.

Reference: dequantize inputs and run standard torch.matmul per expert,
then compare against kernel output within FP8-precision tolerance.
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_gemm_per_token import moe_gemm_per_token


def _make_per_token_inputs(E, N, K, device="cuda"):
    torch.manual_seed(1)
    total = E * 64
    group_sizes = torch.full((E,), 64, dtype=torch.int32, device=device)
    lhs = torch.randint(-3, 4, (total, K), dtype=torch.int8, device=device).to(
        torch.float8_e4m3fnuz
    )
    rhs = torch.randint(-3, 4, (E, N, K), dtype=torch.int8, device=device).to(
        torch.float8_e4m3fnuz
    )
    x_scale = torch.rand(total, dtype=torch.float32, device=device) * 0.1 + 0.9
    w_scale = torch.rand(E, dtype=torch.float32, device=device) * 0.1 + 0.9
    return lhs, rhs, x_scale, w_scale, group_sizes


def _per_token_reference(lhs, rhs, x_scale, w_scale, group_sizes, out_dtype):
    E, N, _K = rhs.shape
    total = lhs.shape[0]
    out = torch.zeros(total, N, dtype=out_dtype, device=lhs.device)
    offset = 0
    for e in range(E):
        m = int(group_sizes[e].item())
        if m == 0:
            continue
        a = lhs[offset : offset + m].float()
        b = rhs[e].float()
        xs = x_scale[offset : offset + m]
        ws = float(w_scale[e].item())
        out[offset : offset + m] = ((a @ b.T) * xs[:, None] * ws).to(out_dtype)
        offset += m
    return out


@pytest.mark.parametrize("E, N, K", [(4, 256, 128), (8, 128, 256)])
def test_moe_gemm_per_token_correctness(E, N, K):
    lhs, rhs, x_scale, w_scale, group_sizes = _make_per_token_inputs(E, N, K)
    out = moe_gemm_per_token(lhs, rhs, x_scale, w_scale, group_sizes)
    ref = _per_token_reference(lhs, rhs, x_scale, w_scale, group_sizes, torch.bfloat16)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    scale = ref.abs().max().clamp_min(1e-4)
    err = (out.float() - ref.float()).abs().max()
    assert err <= 0.2 * scale, f"max err {err:.4f} > 0.2 * {scale:.4f}"


def test_moe_gemm_per_token_empty():
    E, N, K = 4, 128, 128
    lhs = torch.empty(0, K, dtype=torch.float8_e4m3fnuz, device="cuda")
    rhs = torch.randint(-3, 4, (E, N, K), dtype=torch.int8, device="cuda").to(
        torch.float8_e4m3fnuz
    )
    x_scale = torch.empty(0, dtype=torch.float32, device="cuda")
    w_scale = torch.ones(E, dtype=torch.float32, device="cuda")
    group_sizes = torch.zeros(E, dtype=torch.int32, device="cuda")
    out = moe_gemm_per_token(lhs, rhs, x_scale, w_scale, group_sizes)
    assert out.shape == (0, N)
