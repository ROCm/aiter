# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
High-level Python wrapper for mHC (manifold-constrained Hyper Connection).

Implements equations 14-18 from the mHC paper in a single optimized kernel call.
"""

from typing import Optional
import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_fused_kernel,
    _sinkhorn_knopp_log_domain_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def mhc(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute mHC projection mapping with all three streams (equations 14-18).

    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - H^res: identity (no activation, ready for Eq 19: Sinkhorn-Knopp)

    All operations are fused in a single Triton kernel for optimal performance.

    Args:
        x: Input tensor with shape (M, nC) where M is batch/sequence length and
           nC is the input feature dimension (n × C in paper notation)
        phi: Projection matrix with shape (nC, n² + 2n) for transforming input
             to three output streams
        alpha_pre: Scaling factor α^pre for pre-stream (first n elements)
        alpha_post: Scaling factor α^post for post-stream (next n elements)
        alpha_res: Scaling factor α^res for residual stream (last n² elements)
        bias: Bias vector b with shape (n² + 2n,) applied after scaling
        n: Stream parameter - hyperparameter controlling manifold dimension.
           Determines output size: n (pre) + n (post) + n² (res) = n² + 2n
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        out (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, n² + 2n)

    Returns:
        Output tensor H with shape (M, n² + 2n) containing three concatenated streams:
        - H^pre: [0:n] - manifold projection with sigmoid activation (n elements, H^{pre} ∈ ℝ^{1×n})
        - H^post: [n:2n] - post-processing with scaled sigmoid activation (n elements, H^{post} ∈ ℝ^{1×n})
        - H^res: [2n:2n+n²] - residual connection (identity, for later Sinkhorn-Knopp) (n² elements, H^{res} ∈ ℝ^{n×n})

    Shape requirements:
        - x: (M, nC) where nC = n * C (flattened streams)
        - phi: (nC, n² + 2n)
        - bias: (n² + 2n,)
        - output: (M, n² + 2n)

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C  # 4096 input features
        >>> N_total = n * n + 2 * n  # 24 output features (16 + 4 + 4)
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi = torch.randn(nC, N_total, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(N_total, dtype=torch.float32, device='cuda')
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> H = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
        >>> H.shape  # (32, 24)
        >>> # H contains: [H^pre (4 elements), H^post (4 elements), H^res (16 elements)]
    """
    _LOGGER.info(
        f"MHC: x={tuple(x.shape)} phi={tuple(phi.shape)} alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res}"
    )
    
    # Input shape extraction
    M, K = x.shape  # M: batch/sequence, K: nC (input features)
    K_phi, N = phi.shape  # K_phi: should match K, N: n² + 2n (output features)
    N_total_expected = n * n + 2 * n

    # Validate tensor shapes
    assert K == K_phi, f"Dimension mismatch: x has K={K}, phi has K={K_phi}"
    assert N == N_total_expected, f"Dimension mismatch: phi has N={N}, expected {N_total_expected} (n²+2n with n={n})"
    assert bias.shape[0] == N, f"Bias shape mismatch: expected ({N},), got {bias.shape}"
    
    # Validate devices
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
    # BLOCK_M: Row tile size - balance between occupancy and register pressure
    # BLOCK_N: Column tile size - should align with output dimension
    # BLOCK_K: Reduction tile size - affects memory reuse in matmul
    BLOCK_M = 64 if M >= 64 else 32
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = min(128, triton.next_power_of_2(K))

    # Launch 2D grid: one thread block per (BLOCK_M x BLOCK_N) output tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Invoke the fused Triton kernel for equations 14-18
    _mhc_fused_kernel[grid](
        x,           # Input tensor (M, nC)
        phi,         # Projection matrix (nC, n²+2n)
        alpha_pre,   # Scaling factor for pre-stream
        alpha_post,  # Scaling factor for post-stream
        alpha_res,   # Scaling factor for residual stream
        bias,        # Bias vector (n²+2n,)
        out,         # Output tensor (M, n²+2n)
        # Shape parameters
        M=M,         # Number of rows (batch/sequence dimension)
        K=K,         # Input features (nC)
        N=N,         # Output features (n²+2n)
        n=n,         # Stream parameter
        eps=eps,     # Numerical stability epsilon for RMSNorm
        # Tensor strides for memory access
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_phik=phi.stride(0),
        stride_phin=phi.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        # Block sizes for tiling
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


def sinkhorn_knopp(
    logits: torch.Tensor,
    num_iters: int = 10,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Projects batched raw logits onto doubly stochastic matrices using log-domain Sinkhorn-Knopp.

    A doubly stochastic matrix has:
        - All rows sum to 1
        - All columns sum to 1
        - All entries are non-negative

    This is used in mHC to constrain the mixing matrix W to the Birkhoff polytope,
    ensuring stable training by preserving identity mapping properties.

    Args:
        logits (torch.Tensor): Input raw logits with shape (M, N, N), where:
            - M is the batch size (e.g., number of layers or heads)
            - N is the matrix size (e.g., n_streams, typically 4)
            N must be a power of 2 and <= 64.
        num_iters (int): Number of Sinkhorn-Knopp iterations. Default: 10.
            More iterations = better convergence to doubly stochastic.
            Typically 10-20 iterations suffice.
        out (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N, N).
            If None, a new tensor is allocated.

    Returns:
        torch.Tensor: Doubly stochastic matrices with shape (M, N, N).
            Each matrix in the batch has rows and columns summing to 1.

    Example:
        >>> logits = torch.randn(16, 4, 4, device='cuda')  # 16 matrices, 4x4 each
        >>> P = sinkhorn_knopp(logits, num_iters=10)
        >>> print(P.sum(dim=-1))  # Row sums ≈ 1
        >>> print(P.sum(dim=-2))  # Col sums ≈ 1
    """
    _LOGGER.info(
        f"Sinkhorn-Knopp: logits={tuple(logits.shape)} num_iters={num_iters}"
    )

    # Validate inputs
    assert logits.dim() == 3, f"logits must be 3D (M, N, N), got {logits.dim()}D"

    M, N, N2 = logits.shape
    assert N == N2, f"Last two dimensions must be equal, got ({N}, {N2})"
    # Cap N at 64 to avoid overflow in log domain
    assert N <= 64, f"Matrix size N={N} exceeds maximum of 64"

    # Check N is power of 2
    N_pow2 = triton.next_power_of_2(N)
    assert N == N_pow2, f"Matrix size N={N} must be a power of 2"

    assert num_iters > 0, f"num_iters must be positive, got {num_iters}"

    # Ensure contiguous
    logits = logits.contiguous()

    # Allocate output if not provided
    if out is None:
        out = torch.empty((M, N, N), dtype=logits.dtype, device=logits.device)
    else:
        assert out.shape == (M, N, N), f"out.shape {out.shape} must be ({M}, {N}, {N})"
        out = out.contiguous()

    # Grid: one program per batch element
    grid = (M,)

    _sinkhorn_knopp_log_domain_kernel[grid](
        logits,
        out,
        M,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        N=N,
        NUM_ITERS=num_iters,
    )

    return out
