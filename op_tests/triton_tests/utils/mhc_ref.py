# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch reference implementations for mHC (manifold-constrained Hyper Connection).

This module provides reference implementations for validating Triton kernels:
- mhc_torch: Reference for mHC projection mapping (Eq 14-19, Sinkhorn mode)
- sinkhorn_knopp_exp_domain_torch: Sinkhorn-Knopp in exponential domain
- sinkhorn_knopp_log_domain_torch: Sinkhorn-Knopp in log domain
- is_doubly_stochastic: Helper to validate doubly stochastic matrices

Also provides test input generation utilities:
- generate_mhc_inputs: Generate test inputs for mHC mapping
- get_test_shapes: Test shape configurations for mHC

Notation (from mHC paper arXiv:2512.24880v2):
    - M: Batch/sequence dimension
    - n: Stream parameter controlling manifold dimension
    - C: Hidden dimension per stream
    - nC: Total flattened input dimension (K in kernel, K = n × C)
    - N: Total output dimension (n² + 2n)
"""

import torch

__all__ = [
    "mhc_torch",
    "sinkhorn_knopp_exp_domain_torch",
    "sinkhorn_knopp_log_domain_torch",
    "is_doubly_stochastic",
    "generate_mhc_inputs",
    "get_test_shapes",
    "mhc_post_torch",
    "generate_mhc_post_inputs",
]

# =============================================================================
# PyTorch Reference Implementations
# =============================================================================


def mhc_torch(
    x: torch.Tensor,
    phi: torch.Tensor,  # (K, n + n + n_res)
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    hc_pre_eps: float = 0.0,
    sinkhorn_iters: int = 20,
    return_with_sinkhorn: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single PyTorch reference for mHC (Eq 14-19) plus the apply-pre fusion step,
    exposing every meaningful intermediate so each test picks what it needs.

    The Triton ``mhc()`` wrapper (and HIP ``aiter.mhc_pre``) only return three
    of the four outputs (``hpost``, ``hres``, ``layer_input``); ``H^pre`` is
    consumed in-kernel. This ref additionally returns the raw pre-sigmoid
    ``hpre`` so tests that want to assert on it (or re-derive ``pre_mix``
    via ``σ(hpre) + hc_pre_eps``) can do so

    Implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b (scaling)
    - Eq 17: H^pre = σ(H^pre) + hc_pre_eps; folded into
      ``layer_input[m, c] = Σᵢ pre_mix[m, i] · x[m, i*C + c]`` (apply-pre).
    - Eq 18: H^post = 2σ(H^post) (scaled sigmoid activation for post-stream)
    - Eq 19: H^res = Sinkhorn(H^res) (project residual stream onto doubly stochastic manifold)

    Args:
        x: (M, K) input tensor where K = n × C (flattened n-stream residual)
        phi: (K, N) unified projection matrix where N = n + n + n²
            Layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n²-1]
        alpha_pre: α^pre scaling factor for pre-stream (Eq 12)
        alpha_post: α^post scaling factor for post-stream (Eq 12)
        alpha_res: α^res scaling factor for residual stream (Eq 12)
        bias: (N,) bias vector where N = n + n + n²
        n: stream parameter (manifold dimension controller)
        eps: epsilon for RMS normalization stability
        hc_pre_eps: additive epsilon on σ(H^pre) before the apply-pre fold.
            Note: ``hpre`` is returned **before** sigmoid and **before** this
            additive eps; the eps only affects ``layer_input``.
        sinkhorn_iters: number of Sinkhorn-Knopp iterations for H^res (Eq 19)
        return_with_sinkhorn: if True, apply Sinkhorn-Knopp to H_res; else
            return raw H^res logits as ``hres`` (matches ``mhc(..., sinkhorn_iters=0)``
            which skips Eq 19).

    Returns:
        Tuple ``(hpost, hres, hpre, layer_input)`` all in fp32:
        - ``hpost``: (M, n, 1) — ``2σ(H^post)`` (Eq 18, unsqueezed to match ``mhc()``).
        - ``hres``: (M, n, n) — Sinkhorn-projected H^res (or raw logits when
          ``return_with_sinkhorn=False``).
        - ``hpre``: (M, n) — **raw H^pre logits**, post-Eq-16, pre-Eq-17. Apply
          ``σ`` and ``+hc_pre_eps`` to recover ``pre_mix`` if needed.
        - ``layer_input``: (M, C) — apply-pre fused output, Σᵢ pre_mix[i] · x_i.
    """
    # Extract individual phi components from unified tensor
    phi_pre = phi[:, :n]
    phi_post = phi[:, n : 2 * n]
    phi_res = phi[:, 2 * n :]
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
    H_pre = alpha_pre * H_tilde_pre + bias_pre
    H_post = alpha_post * H_tilde_post + bias_post
    H_res = alpha_res * H_tilde_res + bias_res

    # Eq 17 + apply-pre fold: pre_mix = σ(H_pre) + hc_pre_eps, then
    # layer_input[m, c] = Σᵢ pre_mix[m, i] * x[m, i*C + c].
    pre_mix = torch.sigmoid(H_pre) + hc_pre_eps  # (M, n)
    M, K = x.shape
    C = K // n
    x_3d = x_f32.view(M, n, C)  # (M, n, C)
    layer_input = torch.einsum("mn,mnc->mc", pre_mix, x_3d)  # (M, C)

    # Eq 18: hpost = 2σ(H^post), reshaped to (M, n, 1) to match mhc()'s output.
    hpost = (2.0 * torch.sigmoid(H_post)).unsqueeze(-1)  # (M, n, 1) fp32

    # Eq 19: Apply Sinkhorn-Knopp to H^res for doubly stochastic constraint.
    H_res_3d = H_res.view(M, n, n)
    if return_with_sinkhorn:
        hres = sinkhorn_knopp_log_domain_torch(H_res_3d, num_iters=sinkhorn_iters)
    else:
        hres = H_res_3d  # already fp32

    return hpost, hres, H_pre, layer_input


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
):
    """
    Generate test inputs for mHC mapping.

    Returns:
        Tuple of (x, phi, alpha_pre, alpha_post, alpha_res, bias, n) where:
        - x: (M, nC) flattened n-stream residual input
        - phi: (nC, n + n + n²) unified projection matrix [pre | post | res]
        - alpha_pre: α^pre scaling factor for pre-stream (Eq 12)
        - alpha_post: α^post scaling factor for post-stream (Eq 12)
        - alpha_res: α^res scaling factor for residual stream (Eq 12)
        - bias: (n² + 2n,) bias vector
        - n: stream parameter (returned for convenience)
    """

    nC = n * C  # Total flattened dimension

    n_res = n * n  # n² columns

    N_total = n_res + 2 * n

    # flattened n-stream residual
    x = torch.randn(M, nC, dtype=dtype, device=device)

    # Unified projection matrix (nC, n + n + n_res)
    # Layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
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


def mhc_post_torch(
    x: torch.Tensor,  # (m, hidden_size)
    residual: torch.Tensor,  # (m, hc_mult, hidden_size)
    post_layer_mix: torch.Tensor,  # (m, hc_mult) or (m, hc_mult, 1)
    comb_res_mix: torch.Tensor,  # (m, hc_mult, hc_mult)  [dim1=src, dim2=dst]
) -> torch.Tensor:
    """
    PyTorch reference for mhc_post implementing the 3-step strategy:
    1. Multiply post-stream with x
    2. Reduce res-stream (weighted sum over source heads)
    3. Combine and return
    """
    post_layer_mix = (
        post_layer_mix.squeeze(-1) if post_layer_mix.ndim == 3 else post_layer_mix
    )
    x_f32 = x.float()
    res_f32 = residual.float()

    # Step 1: post_mix * x broadcast to each dst head: (m, hc_mult, hidden_size)
    out = post_layer_mix[:, :, None] * x_f32[:, None, :]

    # Step 2: + sum over src heads: einsum[m, src_h, k] * [m, src_h, dst_hc] -> [m, dst_hc, k]
    # comb_res_mix[m, src, dst] so contract dim-1 (src)
    out = out + torch.einsum("mhk,mhH->mHk", res_f32, comb_res_mix.float())

    # Step 3: Return in original dtype
    return out.to(x.dtype)


def generate_mhc_post_inputs(
    m: int,
    hc_mult: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Generate test inputs for mhc_post.

    Returns:
        Tuple of (x, residual, post_layer_mix, comb_res_mix) where:
        - x: (m, hidden_size) in dtype
        - residual: (m, hc_mult, hidden_size) in dtype
        - post_layer_mix: (m, hc_mult) fp32
        - comb_res_mix: (m, hc_mult, hc_mult) fp32
    """
    x = torch.randn(m, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(m, hc_mult, hidden_size, dtype=dtype, device=device)
    post_layer_mix = torch.randn(m, hc_mult, dtype=torch.float32, device=device) * 0.1
    comb_res_mix = (
        torch.randn(m, hc_mult, hc_mult, dtype=torch.float32, device=device) * 0.1
    )

    return x, residual, post_layer_mix, comb_res_mix
