# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest tests for mHC (manifold-constrained Hyper Connection) fused kernel.

Tests correctness of the Triton implementation against PyTorch reference
for various input shapes and configurations.

Notation (from mHC paper):
    - n: Number of streams (expansion factor, typically 4)
    - C: Hidden dimension per stream
    - nC: Total flattened input dimension (K in kernel)
    - M: Batch size
    - x_l ∈ ℝ^(M×nC): Flattened n-stream residual
    - φ ∈ ℝ^(nC×n): Projection matrix
    - H ∈ ℝ^(M×n): Output mapping coefficients
"""

import torch
import pytest
from aiter.ops.triton.fusions.mhc import mhc


# =============================================================================
# PyTorch Reference Implementation
# =============================================================================


def mhc_torch(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: float,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of mHC projection mapping.

    H = α · (x · φ / ||x||_rms) + b

    Args:
        x: Input x_l with shape (M, nC) - flattened n-stream residual
        phi: Projection φ with shape (nC, n)
        alpha: Scaling factor α
        bias: Bias b with shape (n,)
        eps: Epsilon for RMSNorm numerical stability

    Returns:
        H with shape (M, n) - mapping coefficients
    """
    x_f32 = x.to(torch.float32)
    mean_sq = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms

    phi_f32 = phi.to(torch.float32)
    result = x_norm @ phi_f32

    bias_f32 = bias.to(torch.float32)
    out = alpha * result + bias_f32

    return out.to(x.dtype)


# =============================================================================
# Test Input Generation
# =============================================================================


def generate_mhc_inputs(
    M: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Generate test inputs for mHC mapping.

    Args:
        M: Batch size
        n: Number of streams (expansion factor)
        C: Hidden dimension per stream

    Returns:
        Tuple of (x, phi, alpha, bias) where:
        - x: (M, nC) flattened n-stream residual
        - phi: (nC, n) projection matrix
        - alpha: scaling factor
        - bias: (n,) bias vector
    """
    nC = n * C  # Total flattened dimension

    # flattened n-stream residual
    x = torch.randn(M, nC, dtype=dtype, device=device)

    # projection matrix
    phi = torch.randn(nC, n, dtype=dtype, device=device) * 0.1

    # scaling factor
    alpha = 0.5 + torch.rand(1).item()

    # bias
    bias = torch.randn(n, dtype=torch.float32, device=device) * 0.1

    return x, phi, alpha, bias


# =============================================================================
# Test Configurations
# =============================================================================


def get_test_shapes():
    """
    Generate test shape configurations.

    Returns list of (M, n, C) tuples where:
        M: batch size
        n: number of streams
        C: hidden dimension per stream
    """
    shapes = []

    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for n in [1, 2, 4, 8]:
            for C in [512, 1024, 2048, 4096]:
                shapes.append((M, n, C))
    # Edge cases
    shapes += [
        (1, 4, 256),      # Minimal batch
        (1, 16, 4096),    # Single sample, large C
        (2048, 4, 512),   # Large batch, small C
        (128, 4, 7168),   # Non-power-of-2 C
        (64, 8, 2112),    # Non-power-of-2 C
    ]

    return shapes


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("M, n, C", get_test_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_correctness(M, n, C, dtype):
    """Test that Triton implementation matches PyTorch reference."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha, bias = generate_mhc_inputs(M, n, C, dtype)

    out_torch = mhc_torch(x, phi, alpha, bias)
    out_triton = mhc(x, phi, alpha, bias)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("M, n, C", get_test_shapes())
def test_mhc_preallocated_output(M, n, C):
    """Test with pre-allocated output tensor."""
    torch.cuda.empty_cache()

    x, phi, alpha, bias = generate_mhc_inputs(M, n, C)
    out = torch.empty(M, n, dtype=x.dtype, device=x.device)

    out_torch = mhc_torch(x, phi, alpha, bias)
    result = mhc(x, phi, alpha, bias, out=out)

    assert result is out

    torch.testing.assert_close(
        out.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
@pytest.mark.parametrize("M, n, C", get_test_shapes())
def test_mhc_different_epsilon(eps, M, n, C):
    """Test with different epsilon values for RMSNorm."""
    torch.cuda.empty_cache()

    x, phi, alpha, bias = generate_mhc_inputs(M, n, C)

    out_torch = mhc_torch(x, phi, alpha, bias, eps=eps)
    out_triton = mhc(x, phi, alpha, bias, eps=eps)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_mhc_different_alpha(alpha):
    """Test with different scaling factors α."""
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    x, phi, _, bias = generate_mhc_inputs(M, n, C)

    out_torch = mhc_torch(x, phi, alpha, bias)
    out_triton = mhc(x, phi, alpha, bias)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


def test_mhc_zero_input():
    """Test with zero input (edge case for RMSNorm where ||x||_rms → ε)."""
    torch.cuda.empty_cache()

    M, n, C = 16, 4, 512
    nC = n * C

    x = torch.zeros(M, nC, dtype=torch.bfloat16, device="cuda")
    phi = torch.randn(nC, n, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha = 1.0
    bias = torch.randn(n, dtype=torch.float32, device="cuda") * 0.1

    out_torch = mhc_torch(x, phi, alpha, bias)
    out_triton = mhc(x, phi, alpha, bias)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


def test_mhc_large_values():
    """Test numerical stability with large input values."""
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    nC = n * C

    x = torch.randn(M, nC, dtype=torch.bfloat16, device="cuda") * 100
    phi = torch.randn(nC, n, dtype=torch.bfloat16, device="cuda") * 0.01
    alpha = 1.0
    bias = torch.randn(n, dtype=torch.float32, device="cuda")

    out_torch = mhc_torch(x, phi, alpha, bias)
    out_triton = mhc(x, phi, alpha, bias)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=0.1,
        rtol=0.05,
    )
