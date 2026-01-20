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
from aiter.ops.triton.fusions.mhc import mhc, sinkhorn_knopp


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
    PyTorch reference implementation of mHC projection mapping with sigmoid.

    H = sigmoid(α · (x · φ / ||x||_rms) + b)

    Args:
        x: Input x_l with shape (M, nC) - flattened n-stream residual
        phi: Projection φ with shape (nC, n)
        alpha: Scaling factor α
        bias: Bias b with shape (n,)
        eps: Epsilon for RMSNorm numerical stability

    Returns:
        H with shape (M, n) - mapping coefficients after sigmoid activation
    """
    x_f32 = x.to(torch.float32)
    mean_sq = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms

    phi_f32 = phi.to(torch.float32)
    result = x_norm @ phi_f32

    bias_f32 = bias.to(torch.float32)
    linear_out = alpha * result + bias_f32
    
    # Apply sigmoid
    out = torch.sigmoid(linear_out)

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


@pytest.mark.parametrize("M, n, C", [(32, 4, 1024), (64, 4, 2048), (128, 8, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_small_shapes(M, n, C, dtype):
    """Test mHC with a subset of representative shapes for quick validation."""
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


def test_mhc_output_range():
    """Test that sigmoid output is in valid range [0, 1]."""
    torch.cuda.empty_cache()

    M, n, C = 64, 4, 1024
    x, phi, alpha, bias = generate_mhc_inputs(M, n, C)

    out_triton = mhc(x, phi, alpha, bias)

    # Sigmoid output should be in [0, 1]
    assert torch.all(out_triton >= 0.0), "Sigmoid output has values < 0"
    assert torch.all(out_triton <= 1.0), "Sigmoid output has values > 1"


# =============================================================================
# Sinkhorn-Knopp Tests
# =============================================================================


def sinkhorn_knopp_exp_domain_torch(
    logits: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Sinkhorn-Knopp in exponential domain.

    Args:
        logits: Input raw logits with shape (M, N, N)
        num_iters: Number of Sinkhorn iterations
        eps: Small epsilon for numerical stability in division

    Returns:
        Doubly stochastic matrices with shape (M, N, N)
    """
    M, N, _ = logits.shape

    A = logits.to(torch.float32)

    # Ensure positivity via exp (subtract max for numerical stability)
    A_max = A.amax(dim=(-2, -1), keepdim=True)  # Max per matrix
    P = torch.exp(A - A_max)

    # Alternatingly iterate on row-column normalization
    for _ in range(num_iters):
        # Row normalization: make each row sum to 1
        row_sums = P.sum(dim=-1, keepdim=True)  # (M, N, 1)
        P = P / (row_sums + eps)

        # Column normalization: make each column sum to 1
        col_sums = P.sum(dim=-2, keepdim=True)  # (M, 1, N)
        P = P / (col_sums + eps)

    return P.to(logits.dtype)


def sinkhorn_knopp_log_domain_torch(
    logits: torch.Tensor,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Sinkhorn-Knopp in log domain.

    Algorithm:
        1. log_A = A (input logits, already in log domain)
        2. Initialize: log_u = 0, log_v = 0
        3. For each iteration:
           - log_u = -logsumexp_j(log_A + log_v)  # Row normalization
           - log_v = -logsumexp_i(log_A + log_u)  # Column normalization
        4. Output: P = exp(log_A + log_u + log_v)

    Args:
        logits: Input raw logits with shape (M, N, N)
        num_iters: Number of Sinkhorn iterations

    Returns:
        Doubly stochastic matrices with shape (M, N, N)
    """
    M, N, _ = logits.shape

    log_A = logits.to(torch.float32)

    # Initialize log scaling factors (log(1) = 0, so no initial scaling)
    log_u = torch.zeros(M, N, device=logits.device, dtype=torch.float32)
    log_v = torch.zeros(M, N, device=logits.device, dtype=torch.float32)

    for _ in range(num_iters):
        # Row normalization in log domain:
        # log_u[i] = -logsumexp_j(log_A[i,j] + log_v[j])
        scaled = log_A + log_v.unsqueeze(1)  # (M, N, N)
        log_row_sums = torch.logsumexp(scaled, dim=2)  # (M, N)
        log_u = -log_row_sums

        # Column normalization in log domain:
        # log_v[j] = -logsumexp_i(log_A[i,j] + log_u[i])
        scaled = log_A + log_u.unsqueeze(2)  # (M, N, N)
        log_col_sums = torch.logsumexp(scaled, dim=1)  # (M, N)
        log_v = -log_col_sums

    # Compute final matrix: P = exp(log_A + log_u + log_v)
    log_P = log_A + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    P = torch.exp(log_P)

    return P.to(logits.dtype)


def is_doubly_stochastic(P: torch.Tensor, tol: float = 1e-3) -> bool:
    """
    Check if a batch of matrices is doubly stochastic.

    Args:
        P: Tensor of shape (M, N, N)
        tol: Tolerance for sum checks

    Returns:
        True if all matrices are doubly stochastic within tolerance
    """
    # Check non-negative
    if not torch.all(P >= -tol):
        return False

    # Check row sums ≈ 1
    row_sums = P.sum(dim=-1)  # (M, N)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tol):
        return False

    # Check column sums ≈ 1
    col_sums = P.sum(dim=-2)  # (M, N)
    if not torch.allclose(col_sums, torch.ones_like(col_sums), atol=tol):
        return False

    return True


# Test shape configurations for Sinkhorn-Knopp
def get_sk_test_shapes():
    """
    Generate test shape configurations for Sinkhorn-Knopp.

    Returns list of (M, N) tuples where:
        M: batch size (number of matrices)
        N: matrix size (must be power of 2, max 64)
    """
    shapes = []

    # Various batch sizes with typical matrix sizes
    for M in [1, 4, 16, 64, 256]:
        for N in [2, 4, 8, 16, 32]:
            shapes.append((M, N))

    return shapes


@pytest.mark.parametrize("M, N", get_sk_test_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sk_correctness(M, N, dtype):
    """Test that Triton Sinkhorn-Knopp matches PyTorch reference."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    logits = torch.randn(M, N, N, dtype=dtype, device="cuda")

    out_torch_exp = sinkhorn_knopp_exp_domain_torch(logits, num_iters=20)
    out_torch_log = sinkhorn_knopp_log_domain_torch(logits, num_iters=20)
    out_triton = sinkhorn_knopp(logits, num_iters=20)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch_log.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch_exp.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

@pytest.mark.parametrize("M, N", get_sk_test_shapes())
def test_sk_doubly_stochastic(M, N):
    """Test that output matrices are doubly stochastic."""
    torch.cuda.empty_cache()

    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    out = sinkhorn_knopp(logits, num_iters=20)

    # Convert to float32 for accurate sum checking
    out_f32 = out.to(torch.float32)

    assert is_doubly_stochastic(out_f32, tol=1e-2), (
        f"Output is not doubly stochastic. "
        f"Row sums: {out_f32.sum(dim=-1)}, Col sums: {out_f32.sum(dim=-2)}"
    )


@pytest.mark.parametrize("num_iters", [5, 10, 20, 50])
def test_sk_different_iters(num_iters):
    """Test with different numbers of Sinkhorn iterations."""
    torch.cuda.empty_cache()

    M, N = 16, 4
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    out_torch_exp = sinkhorn_knopp_exp_domain_torch(logits, num_iters=num_iters)
    out_torch_log = sinkhorn_knopp_log_domain_torch(logits, num_iters=num_iters)
    out_triton = sinkhorn_knopp(logits, num_iters=num_iters)

    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch_log.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        out_triton.to(torch.float32),
        out_torch_exp.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("M", [1, 4, 16, 64, 256, 1024])
def test_sk_batch_sizes(M):
    """Test with various batch sizes."""
    torch.cuda.empty_cache()

    N = 4  # Typical mHC stream count
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    out = sinkhorn_knopp(logits, num_iters=10)

    assert out.shape == (M, N, N)
    assert is_doubly_stochastic(out.to(torch.float32), tol=1e-2)


@pytest.mark.parametrize("N", [3, 4, 8, 16])
def test_sk_matrix_sizes(N):
    """Test with various matrix sizes (must be power of 2)."""
    torch.cuda.empty_cache()

    M = 16
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    out = sinkhorn_knopp(logits, num_iters=10)

    assert out.shape == (M, N, N)
    assert is_doubly_stochastic(out.to(torch.float32), tol=1e-2)


def test_sk_numerical_stability_large_values():
    """Test numerical stability with large input values."""
    torch.cuda.empty_cache()

    M, N = 16, 4
    # Large positive and negative values
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda") * 10

    out = sinkhorn_knopp(logits, num_iters=50)

    # Primary stability check: no NaN or Inf
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    # Values should be valid probabilities
    assert torch.all(out >= 0.0), "Output has negative values"
    assert torch.all(out <= 1.0), "Output has values > 1"
    # Large values produce peaked matrices; use relaxed tolerance due to bf16 precision
    assert is_doubly_stochastic(out.to(torch.float32), tol=5e-2)


def test_sk_numerical_stability_small_values():
    """Test numerical stability with small input values."""
    torch.cuda.empty_cache()

    M, N = 16, 4
    # Small values (close to zero)
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda") * 0.01

    out = sinkhorn_knopp(logits, num_iters=20)

    # Primary check: log domain should handle this gracefully - no numerical issues
    assert not torch.isnan(out).any(), "Triton output contains NaN"
    assert not torch.isinf(out).any(), "Triton output contains Inf"
    # All values should be valid probabilities
    assert torch.all(out >= 0.0), "Output has negative values"
    assert torch.all(out <= 1.0), "Output has values > 1"
    # Use relaxed tolerance (10%) due to bf16 precision with extremely peaked matrices
    assert is_doubly_stochastic(out.to(torch.float32), tol=0.1), (
        "Triton output is not doubly stochastic for large inputs"
    )

def test_sk_log_domain_stability_large_values():
    """
    Demonstrate that log domain (Triton) handles large values without overflow/underflow.

    With very large input values (* 20), the matrices become extremely peaked
    (close to permutation matrices with values like 1e-16). The key advantage
    of log domain is numerical stability - no NaN/Inf even with extreme inputs.

    Note: bf16 precision limits mean row/col sums may not be exactly 1 for
    extremely peaked matrices, so we use a relaxed tolerance.
    """
    torch.cuda.empty_cache()

    M, N = 16, 4

    # Large values that stress exponential domain
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda") * 20

    out_triton = sinkhorn_knopp(logits, num_iters=50)

    # Primary check: log domain should handle this gracefully - no numerical issues
    assert not torch.isnan(out_triton).any(), "Triton output contains NaN"
    assert not torch.isinf(out_triton).any(), "Triton output contains Inf"
    # All values should be valid probabilities
    assert torch.all(out_triton >= 0.0), "Output has negative values"
    assert torch.all(out_triton <= 1.0), "Output has values > 1"
    # Use relaxed tolerance (10%) due to bf16 precision with extremely peaked matrices
    assert is_doubly_stochastic(out_triton.to(torch.float32), tol=0.1), (
        "Triton output is not doubly stochastic for large inputs"
    )


def test_sk_preallocated_output():
    """Test with pre-allocated output tensor."""
    torch.cuda.empty_cache()

    M, N = 16, 4
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, N, dtype=logits.dtype, device=logits.device)

    result = sinkhorn_knopp(logits, num_iters=10, out=out)

    assert result is out
    assert is_doubly_stochastic(out.to(torch.float32), tol=1e-2)


def test_sk_identity_initialization():
    """Test that identity-like input produces identity-like output."""
    torch.cuda.empty_cache()

    M, N = 8, 4
    # Create logits that favor diagonal (identity-like)
    logits = torch.zeros(M, N, N, dtype=torch.bfloat16, device="cuda")
    logits[:, range(N), range(N)] = 10.0  # Large diagonal values

    out = sinkhorn_knopp(logits, num_iters=20)

    # Output should be close to identity
    identity = torch.eye(N, device="cuda").unsqueeze(0).expand(M, -1, -1)
    torch.testing.assert_close(
        out.to(torch.float32),
        identity,
        atol=0.1,
        rtol=0.1,
    )


def test_sk_uniform_input():
    """Test that uniform input produces uniform output (1/N everywhere)."""
    torch.cuda.empty_cache()

    M, N = 8, 4
    # Uniform logits (all same value)
    logits = torch.ones(M, N, N, dtype=torch.bfloat16, device="cuda")

    out = sinkhorn_knopp(logits, num_iters=20)

    # Output should be uniform: 1/N everywhere
    expected = torch.full((M, N, N), 1.0 / N, device="cuda")
    torch.testing.assert_close(
        out.to(torch.float32),
        expected,
        atol=1e-2,
        rtol=1e-2,
    )


def test_sk_convergence():
    """Test that more iterations lead to better convergence."""
    torch.cuda.empty_cache()

    M, N = 16, 4
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    # Compute with different iteration counts
    out_5 = sinkhorn_knopp(logits, num_iters=5).to(torch.float32)
    out_20 = sinkhorn_knopp(logits, num_iters=20).to(torch.float32)

    # Measure how close row/col sums are to 1
    def sum_error(P):
        row_err = (P.sum(dim=-1) - 1).abs().max()
        col_err = (P.sum(dim=-2) - 1).abs().max()
        return max(row_err.item(), col_err.item())

    err_5 = sum_error(out_5)
    err_20 = sum_error(out_20)

    # More iterations should give better convergence
    assert err_20 <= err_5, f"More iterations should improve convergence: {err_20} > {err_5}"


def test_sk_output_range():
    """Test that all output values are in valid range [0, 1]."""
    torch.cuda.empty_cache()

    M, N = 64, 4
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    out = sinkhorn_knopp(logits, num_iters=10)

    assert torch.all(out >= 0.0), "Output has negative values"
    assert torch.all(out <= 1.0), "Output has values > 1"
