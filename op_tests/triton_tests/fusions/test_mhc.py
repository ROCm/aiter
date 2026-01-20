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
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of mHC projection mapping (Eq 14-18).

    Implements:
    - Eq 14: H̃ = x̃φ
    - Eq 15: r = ||x̃||₂ / √(nC)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre)
    - Eq 18: H^post = 2σ(H^post)

    Args:
        x: Input x_l with shape (M, nC) - flattened n-stream residual
        phi: Projection φ with shape (nC, n² + 2n)
        alpha_pre: Scaling factor for pre-stream
        alpha_post: Scaling factor for post-stream
        alpha_res: Scaling factor for residual stream
        bias: Bias b with shape (n² + 2n,)
        n: Number of streams
        eps: Epsilon for RMSNorm numerical stability

    Returns:
        H with shape (M, n² + 2n) - contains [H^pre, H^post, H^res]
    """
    x_f32 = x.to(torch.float32)
    nC = x.shape[1]
    
    # Eq 15: r = ||x̃||₂ / √(nC)
    mean_sq = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms

    # Eq 14: H̃ = x̃φ
    phi_f32 = phi.to(torch.float32)
    H_tilde = x_norm @ phi_f32

    # Split into three streams
    n_squared = n * n
    H_tilde_pre = H_tilde[:, :n_squared]  # n² coefficients
    H_tilde_post = H_tilde[:, n_squared:n_squared + n]  # n coefficients
    H_tilde_res = H_tilde[:, n_squared + n:]  # n coefficients

    # Split bias
    bias_f32 = bias.to(torch.float32)
    bias_pre = bias_f32[:n_squared]
    bias_post = bias_f32[n_squared:n_squared + n]
    bias_res = bias_f32[n_squared + n:]

    # Eq 16: Apply scaling and bias (note: already normalized above)
    H_pre = alpha_pre * H_tilde_pre + bias_pre
    H_post = alpha_post * H_tilde_post + bias_post
    H_res = alpha_res * H_tilde_res + bias_res
    
    # Eq 17: H^pre = σ(H^pre)
    H_pre = torch.sigmoid(H_pre)
    
    # Eq 18: H^post = 2σ(H^post)
    H_post = 2.0 * torch.sigmoid(H_post)
    
    # H^res: identity (no Sinkhorn-Knopp for now)
    # H_res stays as is
    
    # Concatenate streams
    out = torch.cat([H_pre, H_post, H_res], dim=1)

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
        Tuple of (x, phi, alpha_pre, alpha_post, alpha_res, bias, n) where:
        - x: (M, nC) flattened n-stream residual
        - phi: (nC, n² + 2n) projection matrix
        - alpha_pre: scaling factor for pre-stream
        - alpha_post: scaling factor for post-stream
        - alpha_res: scaling factor for residual stream
        - bias: (n² + 2n,) bias vector
        - n: number of streams
    """
    nC = n * C  # Total flattened dimension
    N_total = n * n + 2 * n  # n² + 2n

    # flattened n-stream residual
    x = torch.randn(M, nC, dtype=dtype, device=device)

    # projection matrix (Eq 10)
    phi = torch.randn(nC, N_total, dtype=dtype, device=device) * 0.1

    # scaling factors (Eq 12)
    alpha_pre = 0.5 + torch.rand(1).item()
    alpha_post = 0.5 + torch.rand(1).item()
    alpha_res = 0.5 + torch.rand(1).item()

    # bias (Eq 13)
    bias = torch.randn(N_total, dtype=torch.float32, device=device) * 0.1

    return x, phi, alpha_pre, alpha_post, alpha_res, bias, n


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
    """Test that Triton implementation matches PyTorch reference (Eq 14-18)."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C, dtype)

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

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

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C)
    N_total = n * n + 2 * n
    out = torch.empty(M, N_total, dtype=x.dtype, device=x.device)

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    result = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, out=out)

    assert result is out

    torch.testing.assert_close(
        out.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
@pytest.mark.parametrize("M, n, C", [(32, 4, 1024)])
def test_mhc_different_epsilon(eps, M, n, C):
    """Test with different epsilon values for RMSNorm."""
    torch.cuda.empty_cache()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C)

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, eps=eps)
    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, eps=eps)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("alpha_scale", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_mhc_different_alpha(alpha_scale):
    """Test with different scaling factors α."""
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    x, phi, _, _, _, bias, n_streams = generate_mhc_inputs(M, n, C)
    
    # Use same alpha for all streams, scaled by alpha_scale
    alpha_pre = alpha_scale
    alpha_post = alpha_scale
    alpha_res = alpha_scale

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

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
    N_total = n * n + 2 * n

    x = torch.zeros(M, nC, dtype=torch.bfloat16, device="cuda")
    phi = torch.randn(nC, N_total, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha_pre = alpha_post = alpha_res = 1.0
    bias = torch.randn(N_total, dtype=torch.float32, device="cuda") * 0.1

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)

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
    N_total = n * n + 2 * n

    x = torch.randn(M, nC, dtype=torch.bfloat16, device="cuda") * 100
    phi = torch.randn(nC, N_total, dtype=torch.bfloat16, device="cuda") * 0.01
    alpha_pre = alpha_post = alpha_res = 1.0
    bias = torch.randn(N_total, dtype=torch.float32, device="cuda")

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=0.1,
        rtol=0.05,
    )


@pytest.mark.parametrize("M, n, C", [(32, 4, 1024), (64, 4, 2048), (128, 8, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_small_shapes(M, n, C, dtype):
    """Test mHC with a subset of representative shapes for quick validation."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C, dtype)

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


def test_mhc_output_range():
    """Test output ranges for each stream."""
    torch.cuda.empty_cache()

    M, n, C = 64, 4, 1024
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C)

    out_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    # Split output into streams
    n_squared = n * n
    out_pre = out_triton[:, :n_squared]
    out_post = out_triton[:, n_squared:n_squared + n]
    out_res = out_triton[:, n_squared + n:]

    # Pre-stream: sigmoid output should be in [0, 1]
    assert torch.all(out_pre >= 0.0), "Pre-stream has values < 0"
    assert torch.all(out_pre <= 1.0), "Pre-stream has values > 1"

    # Post-stream: 2*sigmoid output should be in [0, 2]
    assert torch.all(out_post >= 0.0), "Post-stream has values < 0"
    assert torch.all(out_post <= 2.0), "Post-stream has values > 2"
    
    # Res-stream: no constraints (identity activation)
    # Just verify it exists
    assert out_res.shape == (M, n), f"Res-stream shape mismatch"

