# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch reference implementations for mHC (manifold-constrained Hyper Connection).

This module provides reference implementations for validating Triton kernels:
- mhc_torch: Reference for mHC projection mapping (Eq 14-19, Sinkhorn mode)
- mhc_lite_torch: Reference for mHC-lite (exact doubly stochastic via permutations)
- sinkhorn_knopp_exp_domain_torch: Sinkhorn-Knopp in exponential domain
- sinkhorn_knopp_log_domain_torch: Sinkhorn-Knopp in log domain
- is_doubly_stochastic: Helper to validate doubly stochastic matrices

Also provides test input generation utilities:
- generate_mhc_inputs: Generate test inputs for mHC mapping (supports both sinkhorn and lite modes via hres_mode parameter)
- get_test_shapes: Test shape configurations for mHC
- get_sk_test_shapes: Test shape configurations for Sinkhorn-Knopp

Notation (from mHC paper arXiv:2512.24880v2):
    - M: Batch/sequence dimension
    - n: Stream parameter controlling manifold dimension
    - C: Hidden dimension per stream
    - nC: Total flattened input dimension (K in kernel, K = n × C)
    - N: Total output dimension (n² + 2n for sinkhorn, n! + 2n for lite)
"""

from itertools import permutations as itertools_permutations
from math import factorial
import torch

__all__ = [
    "mhc_torch",
    "mhc_lite_torch",
    "sinkhorn_knopp_exp_domain_torch",
    "sinkhorn_knopp_log_domain_torch",
    "is_doubly_stochastic",
    "generate_mhc_inputs",
    "get_test_shapes",
    "get_sk_test_shapes",
]

# =============================================================================
# PyTorch Reference Implementations
# =============================================================================


def mhc_torch(
    x: torch.Tensor,
    phi_pre: torch.Tensor,
    phi_post: torch.Tensor,
    phi_res: torch.Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    return_with_sinkhorn: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of mHC projection mapping (Eq 14-19).

    This serves as ground truth for validating the Triton kernel implementation.

    Implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b (scaling)
    - Eq 17: H^pre = σ(H^pre) (sigmoid activation for pre-stream)
    - Eq 18: H^post = 2σ(H^post) (scaled sigmoid activation for post-stream)
    - Eq 19: H^res = Sinkhorn(H^res) (project residual stream onto doubly stochastic manifold)

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) manifold projection with sigmoid
        - H_post: (M, n) post-processing with 2*sigmoid
        - H_res: (M, n²) doubly stochastic residual connection
    """
    x_f32 = x.to(torch.float32)

    # Eq 15: r = ||x̃||₂ / √(nC)
    mean_sq = torch.mean(x_f32**2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms

    # Eq 14: H̃ = x̃φ - compute each stream separately
    phi_pre_f32 = phi_pre.to(torch.float32)
    phi_post_f32 = phi_post.to(torch.float32)
    phi_res_f32 = phi_res.to(torch.float32)

    H_tilde_pre = x_norm @ phi_pre_f32  # (M, n)
    H_tilde_post = x_norm @ phi_post_f32  # (M, n)
    H_tilde_res = x_norm @ phi_res_f32  # (M, n²)

    # Split bias
    bias_f32 = bias.to(torch.float32)
    bias_pre = bias_f32[:n]
    bias_post = bias_f32[n : 2 * n]
    bias_res = bias_f32[2 * n :]

    # Eq 16: Apply stream-specific scaling and bias
    # Note: normalization already applied in x_norm above
    H_pre = alpha_pre * H_tilde_pre + bias_pre
    H_post = alpha_post * H_tilde_post + bias_post
    H_res = alpha_res * H_tilde_res + bias_res

    # Eq 17: Apply sigmoid activation to pre-stream
    # H^pre = σ(H^pre)
    H_pre = torch.sigmoid(H_pre)

    # Eq 18: Apply scaled sigmoid activation to post-stream
    # H^post = 2σ(H^post)
    H_post = 2.0 * torch.sigmoid(H_post)

    # Eq 19: Apply Sinkhorn-Knopp to H^res for doubly stochastic constraint
    # Reshape H_res from (M, n²) to (M, n, n) for Sinkhorn algorithm
    # Note: Use 20 iterations to match the default in the Triton mhc() implementation
    if return_with_sinkhorn:
        M = H_res.shape[0]
        H_res_3d = H_res.view(M, n, n)
        H_res_ds = sinkhorn_knopp_log_domain_torch(H_res_3d, num_iters=20)
        H_res = H_res_ds.view(M, -1)  # Reshape back to (M, n²)
    else:
        H_res = H_res.to(torch.float32)

    return H_pre.to(x.dtype), H_post.to(x.dtype), H_res.to(x.dtype)


def mhc_lite_torch(
    x: torch.Tensor,
    phi_pre: torch.Tensor,
    phi_post: torch.Tensor,
    phi_res: torch.Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of mHC-lite (exact doubly stochastic).

    This serves as ground truth for validating the Triton kernel implementation
    with hres_mode="lite".

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) manifold projection with sigmoid
        - H_post: (M, n) post-processing with 2*sigmoid
        - H_res: (M, n!) exact doubly stochastic residual connection
    """
    x_f32 = x.to(torch.float32)
    M = x.shape[0]
    n_factorial = factorial(n)
    n_squared = n * n

    # Eq 15: r = ||x̃||₂ / √(nC)
    mean_sq = torch.mean(x_f32**2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms

    # Eq 14: H̃ = x̃φ - compute each stream separately
    phi_pre_f32 = phi_pre.to(torch.float32)
    phi_post_f32 = phi_post.to(torch.float32)
    phi_res_f32 = phi_res.to(torch.float32)

    H_tilde_pre = x_norm @ phi_pre_f32  # (M, n)
    H_tilde_post = x_norm @ phi_post_f32  # (M, n)
    H_tilde_res = x_norm @ phi_res_f32  # (M, n!)

    # Split bias - for lite mode, bias_res has n! elements
    bias_f32 = bias.to(torch.float32)
    bias_pre = bias_f32[:n]
    bias_post = bias_f32[n : 2 * n]
    bias_res = bias_f32[2 * n : 2 * n + n_factorial]

    # Eq 16: Apply stream-specific scaling and bias
    H_pre = alpha_pre * H_tilde_pre + bias_pre
    H_post = alpha_post * H_tilde_post + bias_post
    H_res_weights = alpha_res * H_tilde_res + bias_res  # (M, n!)

    # Eq 17: Apply sigmoid activation to pre-stream
    H_pre = torch.sigmoid(H_pre)

    # Eq 18: Apply scaled sigmoid activation to post-stream
    H_post = 2.0 * torch.sigmoid(H_post)

    # mHC-lite: softmax + permutation combination
    # Apply softmax to get convex coefficients
    alpha_coeffs = torch.softmax(H_res_weights, dim=-1)  # (M, n!)

    # Generate all n! permutation matrices
    all_perms = list(itertools_permutations(range(n)))
    P = torch.zeros(n_factorial, n, n, device=x.device, dtype=torch.float32)
    for k, perm in enumerate(all_perms):
        for i, j in enumerate(perm):
            P[k, i, j] = 1.0

    # Weighted sum: H_res[m] = Σ_k α[m,k] * P[k]
    # Using einsum: (M, n!) x (n!, n, n) -> (M, n, n)
    H_res = torch.einsum("mk,kij->mij", alpha_coeffs, P)  # (M, n, n)
    H_res = H_res.view(M, n_squared)  # Flatten to (M, n²)

    return H_pre.to(x.dtype), H_post.to(x.dtype), H_res.to(x.dtype)


def sinkhorn_knopp_exp_domain_torch(
    logits: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Sinkhorn-Knopp in exponential domain.

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


# =============================================================================
# Test Input Generation
# =============================================================================


def generate_mhc_inputs(
    M: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    mode: str = "mhc",
):
    """
    Generate test inputs for mHC mapping.

    Args:
        mode: Benchmark mode - "mhc", "mhc_lite", or "sinkhorn_knopp_only". Default: "mhc"

    Returns:
        Tuple of (x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n) where:
        - x: (M, nC) flattened n-stream residual input
        - phi_pre: (nC, n) pre-stream projection matrix
        - phi_post: (nC, n) post-stream projection matrix
        - phi_res: (nC, n²) for mhc/sinkhorn_knopp_only modes, (nC, n!) for mhc_lite mode
        - alpha_pre: α^pre scaling factor for pre-stream (Eq 12)
        - alpha_post: α^post scaling factor for post-stream (Eq 12)
        - alpha_res: α^res scaling factor for residual stream (Eq 12)
        - bias: (n² + 2n,) for mhc/sinkhorn_knopp_only modes, (n! + 2n,) for mhc_lite mode
        - n: stream parameter (returned for convenience)
    """
    if mode not in ("mhc", "mhc_lite", "sinkhorn_knopp_only"):
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'mhc', 'mhc_lite', or 'sinkhorn_knopp_only'"
        )

    nC = n * C  # Total flattened dimension

    # Determine phi_res and bias dimensions based on mode
    if mode == "mhc_lite":
        n_res = factorial(n)  # n! columns for lite mode
    else:
        n_res = n * n  # n² columns for sinkhorn mode

    N_total = n_res + 2 * n

    # flattened n-stream residual
    x = torch.randn(M, nC, dtype=dtype, device=device)

    # Separate projection matrices for each stream
    phi_pre = torch.randn(nC, n, dtype=dtype, device=device) * 0.1
    phi_post = torch.randn(nC, n, dtype=dtype, device=device) * 0.1
    phi_res = torch.randn(nC, n_res, dtype=dtype, device=device) * 0.1

    # scaling factors (Eq 12)
    alpha_pre = 0.5 + torch.rand(1).item()
    alpha_post = 0.5 + torch.rand(1).item()
    alpha_res = 0.5 + torch.rand(1).item()

    # bias (Eq 13)
    bias = torch.randn(N_total, dtype=torch.float32, device=device) * 0.1

    return x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n


# =============================================================================
# Test Configurations
# =============================================================================


def get_test_shapes():
    """
    Generate test shape configurations.

    Returns list of (M, n, C) tuples where:
        M: batch/sequence dimension
        n: stream parameter (manifold dimension controller)
        C: hidden dimension per stream
    """
    shapes = []

    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for n in [1, 2, 4]:
            for C in [512, 1024, 2048, 4096]:
                shapes.append((M, n, C))
    # Edge cases
    shapes += [
        (1, 4, 256),  # Minimal batch
        (1, 16, 4096),  # Single sample, large C
        (2048, 4, 512),  # Large batch, small C
        (128, 4, 7168),  # Non-power-of-2 C
        (64, 8, 2112),  # Non-power-of-2 C
    ]

    return shapes


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
