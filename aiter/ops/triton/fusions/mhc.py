# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Python wrapper for mHC (manifold-constrained Hyper Connection) fused kernel.

Provides the mhc function using a fused Triton kernel
with the deferred RMSNorm division optimization and sigmoid activation.
"""

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.fusions.mhc import (
    _mhc_fused_rmsnorm_matmul_sigmoid_kernel,
    _sinkhorn_knopp_log_domain_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def mhc(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: float,
    bias: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes mHC mapping with fused RMSNorm + MatMul + Scale + Bias + Sigmoid.

    Implements: H = sigmoid(α · (x · φ / ||x||_rms) + b)

    Uses deferred RMSNorm division optimization: computes (x @ φ) / rms instead of
    (x / rms) @ φ, reducing divisions from K to N elements per row.

    Args:
        x (torch.Tensor): Input tensor with shape (M, K), where K = n_streams * hidden_dim.
            Typically BF16 or FP16.
        phi (torch.Tensor): Projection weight matrix with shape (K, N), where N = n_streams.
            Typically BF16 or FP16.
        alpha (float): Scaling factor applied after normalization.
        bias (torch.Tensor): Bias tensor with shape (N,). Typically FP32.
        eps (float): Epsilon for numerical stability in RMSNorm. Default: 1e-6.
        out (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
            If None, a new tensor is allocated.

    Returns:
        torch.Tensor: Output tensor with shape (M, N) after sigmoid activation.
    """
    _LOGGER.info(
        f"MHC: x={tuple(x.shape)} phi={tuple(phi.shape)} alpha={alpha}"
    )

    # Validate inputs
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    assert phi.dim() == 2, f"phi must be 2D, got {phi.dim()}D"
    assert bias.dim() == 1, f"bias must be 1D, got {bias.dim()}D"

    M, K = x.shape
    K_phi, N = phi.shape

    assert K == K_phi, f"x.shape[1] ({K}) must match phi.shape[0] ({K_phi})"
    assert bias.shape[0] == N, f"bias.shape[0] ({bias.shape[0]}) must match N ({N})"

    # Ensure contiguous tensors
    x = x.contiguous()
    phi = phi.contiguous()
    bias = bias.contiguous()

    # Allocate output if not provided
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    else:
        assert out.shape == (M, N), f"out.shape {out.shape} must be ({M}, {N})"
        out = out.contiguous()

    # Block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = triton.next_power_of_2(min(N, 64))
    BLOCK_SIZE_K = min(128, triton.next_power_of_2(K))

    # 2D grid: (num_m_blocks, num_n_blocks)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    _mhc_fused_rmsnorm_matmul_sigmoid_kernel[grid](
        x,
        phi,
        out,
        bias,
        alpha,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        phi.stride(0),
        phi.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
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
