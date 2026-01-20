# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
High-level Python wrapper for mHC (manifold-constrained Hyper Connection).

Implements equations 14-18 from the mHC paper in a single optimized kernel call.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import _mhc_fused_kernel


def mhc(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute mHC projection mapping with all three streams (equations 14-18).

    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre)
    - Eq 18: H^post = 2σ(H^post)
    - H^res: identity (ready for Sinkhorn-Knopp)

    All operations are fused in a single Triton kernel for optimal performance.

    Args:
        x: Input tensor with shape (M, nC) - flattened n-stream residual
        phi: Projection matrix with shape (nC, n² + 2n)
        alpha_pre: Scaling factor for pre-stream (n² elements)
        alpha_post: Scaling factor for post-stream (n elements)
        alpha_res: Scaling factor for residual stream (n elements)
        bias: Bias vector with shape (n² + 2n,)
        n: Number of streams
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        out: Optional pre-allocated output tensor with shape (M, n² + 2n)

    Returns:
        Output tensor H with shape (M, n² + 2n) containing [H^pre, H^post, H^res]

    Shape requirements:
        - x: (M, nC) where nC = n * C (flattened streams)
        - phi: (nC, n² + 2n)
        - bias: (n² + 2n,)
        - output: (M, n² + 2n)

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C  # 4096
        >>> N_total = n * n + 2 * n  # 24
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi = torch.randn(nC, N_total, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(N_total, dtype=torch.float32, device='cuda')
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> H = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
        >>> H.shape  # (32, 24)
    """
    # Validate inputs
    M, K = x.shape
    K_phi, N = phi.shape
    N_total_expected = n * n + 2 * n

    assert K == K_phi, f"Dimension mismatch: x has K={K}, phi has K={K_phi}"
    assert N == N_total_expected, f"Dimension mismatch: phi has N={N}, expected {N_total_expected}"
    assert bias.shape[0] == N, f"Bias shape mismatch: expected ({N},), got {bias.shape}"
    assert x.device == phi.device == bias.device, "All tensors must be on the same device"
    assert x.device.type == "cuda", "mHC kernel requires CUDA device"

    # Allocate output if not provided
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    else:
        assert out.shape == (M, N), f"Output shape mismatch: expected ({M}, {N}), got {out.shape}"
        assert out.dtype == x.dtype, f"Output dtype mismatch: expected {x.dtype}, got {out.dtype}"
        assert out.device == x.device, f"Output device mismatch"

    # Determine block sizes for optimal performance
    # For matrix multiplication: balance between occupancy and memory usage
    BLOCK_M = 64 if M >= 64 else 32
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = min(128, triton.next_power_of_2(K))

    # Launch kernel with 2D grid over (M, N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _mhc_fused_kernel[grid](
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        out,
        M=M,
        K=K,
        N=N,
        n=n,
        eps=eps,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_phik=phi.stride(0),
        stride_phin=phi.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out
