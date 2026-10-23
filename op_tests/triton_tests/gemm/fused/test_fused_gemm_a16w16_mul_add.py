# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_mul_add import (
    fused_gemm_a16w16_mul_add,
)


def generate_fused_gemm_a16w16_mul_add_inputs(
    B, S, N, K, dtype, a_kind, b_kind, bias, x_3d=True, x_strided=False, seed=0
):
    """Make x, w, a, b and bias. a_kind and b_kind are "scalar", "channel"
    (N,), "gate" (B, 1, N) or "full" (B, S, N), or (M, N) when x is 2D."""
    torch.manual_seed(seed)
    if x_strided:
        # x is a slice of a longer sequence, so rows of a batch are not contiguous.
        x = torch.randn((B, S + 13, K), dtype=dtype, device="cuda")[:, :S]
    else:
        x = torch.randn((B, S, K), dtype=dtype, device="cuda")
    if not x_3d:
        x = x.reshape(B * S, K)
    w = torch.randn((N, K), dtype=dtype, device="cuda") / K**0.5
    bias_tensor = torch.randn((N,), dtype=dtype, device="cuda") if bias else None

    def operand(kind):
        if kind == "scalar":
            return float(torch.randn(1, device="cuda", dtype=torch.float32).item())
        if kind == "channel":
            return torch.randn((N,), dtype=dtype, device="cuda")
        if kind == "gate":
            return torch.randn((B, 1, N), dtype=dtype, device="cuda")
        shape = (B, S, N) if x_3d else (B * S, N)
        return torch.randn(shape, dtype=dtype, device="cuda")

    return x, w, operand(a_kind), operand(b_kind), bias_tensor


def run_torch(x, w, a, b, bias, dtype):
    y = torch.nn.functional.linear(
        x.float(), w.float(), None if bias is None else bias.float()
    )
    a = a.float() if isinstance(a, torch.Tensor) else a
    b = b.float() if isinstance(b, torch.Tensor) else b
    return (a * y + b).to(dtype)


def get_2d_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(1024, 1024, 1024)]
    x_vals += [(4096, 1152, 1152)]
    return x_vals


@pytest.mark.parametrize("M, N, K", get_2d_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("a_kind", ["scalar", "channel", "full"])
@pytest.mark.parametrize("b_kind", ["scalar", "full"])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_gemm_a16w16_mul_add_2d(M, N, K, dtype, a_kind, b_kind, bias):
    x, w, a, b, bias_tensor = generate_fused_gemm_a16w16_mul_add_inputs(
        1, M, N, K, dtype, a_kind, b_kind, bias, x_3d=False
    )
    torch_out = run_torch(x, w, a, b, bias_tensor, dtype)
    triton_out = fused_gemm_a16w16_mul_add(x, w, a, b, bias=bias_tensor, dtype=dtype)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", [(128, 256, 256), (1024, 1152, 1152)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("step", [2, 3])
def test_fused_gemm_a16w16_mul_add_strided_bias(M, N, K, dtype, step):
    """The bias can be an (N,) view with a stride, for example one column of a
    packed bias table."""
    x, w, a, b, _ = generate_fused_gemm_a16w16_mul_add_inputs(
        1, M, N, K, dtype, "scalar", "scalar", bias=False, x_3d=False
    )
    bias = torch.randn((N, step), dtype=dtype, device="cuda")[:, step - 1]
    assert not bias.is_contiguous()
    torch_out = run_torch(x, w, a, b, bias, dtype)
    triton_out = fused_gemm_a16w16_mul_add(x, w, a, b, bias=bias, dtype=dtype)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


def get_3d_vals():
    x_vals = [(2, 77, 640, 320)]  # S is not a multiple of the tile
    x_vals += [(2, 333, 2432, 2432)]  # SD3.5-Large context stream
    x_vals += [(2, 1024, 1152, 1152)]  # PixArt-XL at 512px
    x_vals += [(1, 4096, 1536, 1536)]  # Wan2.1-1.3B
    return x_vals


@pytest.mark.parametrize("B, S, N, K", get_3d_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("a_kind", ["gate", "full"])
@pytest.mark.parametrize("x_strided", [False, True])
@pytest.mark.parametrize("output", [False, True])
def test_fused_gemm_a16w16_mul_add_gated_residual(
    B, S, N, K, dtype, a_kind, x_strided, output
):
    x, w, a, b, bias_tensor = generate_fused_gemm_a16w16_mul_add_inputs(
        B, S, N, K, dtype, a_kind, "full", bias=True, x_strided=x_strided
    )
    torch_out = run_torch(x, w, a, b, bias_tensor, dtype)
    y = torch.empty((B, S, N), dtype=dtype, device="cuda") if output else None
    triton_out = fused_gemm_a16w16_mul_add(
        x, w, a, b, bias=bias_tensor, dtype=dtype, y=y
    )
    if output:
        assert triton_out is y
    assert triton_out.shape == (B, S, N)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)
