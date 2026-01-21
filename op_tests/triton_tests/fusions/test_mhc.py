# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest tests for mHC (manifold-constrained Hyper Connection) fused kernel.

Tests correctness of the Triton implementation (equations 14-18) against
PyTorch reference for various input shapes and configurations.

Notation (from mHC paper arXiv:2512.24880v2):
    - M: Batch/sequence dimension
    - n: Stream parameter controlling manifold dimension
    - C: Hidden dimension per stream
    - nC: Total flattened input dimension (K in kernel, K = n × C)
    - N: Total output dimension (n² + 2n)
    - x_l ∈ ℝ^(M×nC): Flattened n-stream residual (input)
    - φ ∈ ℝ^(nC×N): Projection matrix for transformation to 3 streams
    - H ∈ ℝ^(M×N): Output containing [H^pre, H^post, H^res]
      - H^pre: [0:n] manifold projection with sigmoid activation (n elements, H^{pre} ∈ ℝ^{1×n})
      - H^post: [n:2n] post-processing with 2*sigmoid activation (n elements, H^{post} ∈ ℝ^{1×n})
      - H^res: [2n:2n+n²] residual connection (identity activation) (n² elements, H^{res} ∈ ℝ^{n×n})
"""

import torch
import pytest
from aiter.ops.triton.fusions.mhc import mhc, sinkhorn_knopp
from op_tests.triton_tests.utils.mhc_ref import (
    mhc_torch,
    sinkhorn_knopp_exp_domain_torch,
    sinkhorn_knopp_log_domain_torch,
    is_doubly_stochastic,
    generate_mhc_inputs,
    get_test_shapes,
    get_sk_test_shapes,
)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("M, n, C", get_test_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_correctness(M, n, C, dtype):
    """
    Test that Triton kernel matches PyTorch reference for equations 14-18.
    
    Validates correctness across various shapes and data types.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C, dtype)

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    H_pre_triton, H_post_triton, H_res_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    torch.testing.assert_close(
        H_pre_triton.to(torch.float32),
        H_pre_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        H_post_triton.to(torch.float32),
        H_post_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    # Relaxed tolerance for H_res due to Sinkhorn-Knopp iterative algorithm
    # which amplifies small numerical differences, especially with bfloat16
    torch.testing.assert_close(
        H_res_triton.to(torch.float32),
        H_res_torch.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize("M, n, C", get_test_shapes())
def test_mhc_preallocated_output(M, n, C):
    """
    Test mHC with pre-allocated output tensors.
    
    Verifies that the kernel correctly writes to user-provided output buffers.
    """
    torch.cuda.empty_cache()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C)
    n_squared = n * n
    out_pre = torch.empty(M, n, dtype=x.dtype, device=x.device)
    out_post = torch.empty(M, n, dtype=x.dtype, device=x.device)
    out_res = torch.empty(M, n_squared, dtype=x.dtype, device=x.device)

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    result_pre, result_post, result_res = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, 
                                                out_pre=out_pre, out_post=out_post, out_res=out_res)

    assert result_pre is out_pre
    assert result_post is out_post
    assert result_res is out_res

    torch.testing.assert_close(
        out_pre.to(torch.float32),
        H_pre_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        out_post.to(torch.float32),
        H_post_torch.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    # Relaxed tolerance for H_res due to Sinkhorn-Knopp iterative algorithm
    torch.testing.assert_close(
        out_res.to(torch.float32),
        H_res_torch.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
@pytest.mark.parametrize("M, n, C", [(32, 4, 1024)])
def test_mhc_different_epsilon(eps, M, n, C):
    """
    Test mHC with different epsilon values for RMSNorm (Eq 15).
    
    Validates numerical stability parameter handling.
    """
    torch.cuda.empty_cache()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C)

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, eps=eps)
    H_pre_triton, H_post_triton, H_res_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, eps=eps)

    for torch_out, triton_out in [(H_pre_torch, H_pre_triton), (H_post_torch, H_post_triton), (H_res_torch, H_res_triton)]:
        # Use relaxed tolerance for H_res due to Sinkhorn-Knopp
        is_res = torch_out is H_res_torch
        atol = 5e-2 if is_res else 1e-2
        rtol = 5e-2 if is_res else 1e-2
        torch.testing.assert_close(
            triton_out.to(torch.float32),
            torch_out.to(torch.float32),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("alpha_scale", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_mhc_different_alpha(alpha_scale):
    """
    Test mHC with different scaling factors α (Eq 16).
    
    Validates stream-specific scaling behavior across range of α values.
    """
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    x, phi, _, _, _, bias, n_streams = generate_mhc_inputs(M, n, C)
    
    # Use same alpha for all streams, scaled by alpha_scale
    alpha_pre = alpha_scale
    alpha_post = alpha_scale
    alpha_res = alpha_scale

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    H_pre_triton, H_post_triton, H_res_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    for torch_out, triton_out in [(H_pre_torch, H_pre_triton), (H_post_torch, H_post_triton), (H_res_torch, H_res_triton)]:
        # Use relaxed tolerance for H_res due to Sinkhorn-Knopp
        is_res = torch_out is H_res_torch
        atol = 5e-2 if is_res else 1e-2
        rtol = 5e-2 if is_res else 1e-2
        torch.testing.assert_close(
            triton_out.to(torch.float32),
            torch_out.to(torch.float32),
            atol=atol,
            rtol=rtol,
        )


def test_mhc_zero_input():
    """
    Test mHC with zero input (edge case for RMSNorm).
    
    When x = 0, RMS norm → ε, testing numerical stability of Eq 15.
    """
    torch.cuda.empty_cache()

    M, n, C = 16, 4, 512
    nC = n * C
    N_total = n * n + 2 * n

    x = torch.zeros(M, nC, dtype=torch.bfloat16, device="cuda")
    phi = torch.randn(nC, N_total, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha_pre = alpha_post = alpha_res = 1.0
    bias = torch.randn(N_total, dtype=torch.float32, device="cuda") * 0.1

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
    H_pre_triton, H_post_triton, H_res_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)

    for torch_out, triton_out in [(H_pre_torch, H_pre_triton), (H_post_torch, H_post_triton), (H_res_torch, H_res_triton)]:
        # Use relaxed tolerance for H_res due to Sinkhorn-Knopp
        is_res = torch_out is H_res_torch
        atol = 5e-2 if is_res else 1e-2
        rtol = 5e-2 if is_res else 1e-2
        torch.testing.assert_close(
            triton_out.to(torch.float32),
            torch_out.to(torch.float32),
            atol=atol,
            rtol=rtol,
        )


def test_mhc_large_values():
    """
    Test numerical stability with large input values.
    
    Validates that float32 accumulation prevents overflow/underflow.
    """
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    nC = n * C
    N_total = n * n + 2 * n

    x = torch.randn(M, nC, dtype=torch.bfloat16, device="cuda") * 100
    phi = torch.randn(nC, N_total, dtype=torch.bfloat16, device="cuda") * 0.01
    alpha_pre = alpha_post = alpha_res = 1.0
    bias = torch.randn(N_total, dtype=torch.float32, device="cuda")

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
    H_pre_triton, H_post_triton, H_res_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)

    for torch_out, triton_out in [(H_pre_torch, H_pre_triton), (H_post_torch, H_post_triton), (H_res_torch, H_res_triton)]:
        # Use even more relaxed tolerance for large values + Sinkhorn-Knopp
        is_res = torch_out is H_res_torch
        atol = 0.2 if is_res else 0.1
        rtol = 0.1 if is_res else 0.05
        torch.testing.assert_close(
            triton_out.to(torch.float32),
            torch_out.to(torch.float32),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("M, n, C", [(32, 4, 1024), (64, 4, 2048), (128, 8, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_small_shapes(M, n, C, dtype):
    """
    Quick smoke test with representative shapes.
    
    Subset of test_mhc_correctness for faster validation during development.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C, dtype)

    H_pre_torch, H_post_torch, H_res_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    H_pre_triton, H_post_triton, H_res_triton = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    for torch_out, triton_out in [(H_pre_torch, H_pre_triton), (H_post_torch, H_post_triton), (H_res_torch, H_res_triton)]:
        # Use relaxed tolerance for H_res due to Sinkhorn-Knopp
        is_res = torch_out is H_res_torch
        atol = 5e-2 if is_res else 1e-2
        rtol = 5e-2 if is_res else 1e-2
        torch.testing.assert_close(
            triton_out.to(torch.float32),
            torch_out.to(torch.float32),
            atol=atol,
            rtol=rtol,
        )


def test_mhc_output_range():
    """
    Validate output value ranges for each stream.
    
    Verifies activation functions produce expected bounds:
    - Pre-stream (Eq 17): σ(·) ∈ [0, 1]
    - Post-stream (Eq 18): 2σ(·) ∈ [0, 2]
    - Res-stream: No constraints (identity activation)
    """
    torch.cuda.empty_cache()

    M, n, C = 64, 4, 1024
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(M, n, C)

    H_pre, H_post, H_res = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    # Pre-stream (Eq 17): sigmoid output should be in [0, 1]
    assert torch.all(H_pre >= 0.0), "Pre-stream has values < 0"
    assert torch.all(H_pre <= 1.0), "Pre-stream has values > 1"

    # Post-stream (Eq 18): 2*sigmoid output should be in [0, 2]
    assert torch.all(H_post >= 0.0), "Post-stream has values < 0"
    assert torch.all(H_post <= 2.0), "Post-stream has values > 2"
    
    # Res-stream: no constraints (identity activation)
    # Just verify it exists
    n_squared = n * n
    assert H_res.shape == (M, n_squared), f"Res-stream shape mismatch"


# =============================================================================
# Sinkhorn-Knopp Tests
# =============================================================================


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


@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64])
def test_sk_matrix_sizes(N):
    """Test with various matrix sizes (must be power of 2)."""
    torch.cuda.empty_cache()

    M = 16
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda")

    out = sinkhorn_knopp(logits, num_iters=10)

    assert out.shape == (M, N, N)
    # N=2 requires slightly higher tolerance due to bfloat16 precision with less averaging
    tol = 2e-2 if N == 2 else 1e-2
    assert is_doubly_stochastic(out.to(torch.float32), tol=tol)


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
    assert is_doubly_stochastic(out.to(torch.float32), tol=0.2), (
        "Triton output is not doubly stochastic for large inputs"
    )


@pytest.mark.parametrize("value_scale", [10.0, 20.0])
def test_sk_log_domain_stability_large_values(value_scale):
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
    logits = torch.randn(M, N, N, dtype=torch.bfloat16, device="cuda") * value_scale

    out_triton = sinkhorn_knopp(logits, num_iters=50)

    # Primary check: log domain should handle this gracefully - no numerical issues
    assert not torch.isnan(out_triton).any(), "Triton output contains NaN"
    assert not torch.isinf(out_triton).any(), "Triton output contains Inf"
    # All values should be valid probabilities
    assert torch.all(out_triton >= 0.0), "Output has negative values"
    assert torch.all(out_triton <= 1.0), "Output has values > 1"
    assert is_doubly_stochastic(out_triton.to(torch.float32), tol=0.2), (
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
