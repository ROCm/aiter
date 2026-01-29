# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
High-level Python wrapper for mHC (manifold-constrained Hyper Connection).

Provides two main functions:
- fused_mhc(): Low-level interface for equations 14-18 (fused kernel only)
- mhc(): Complete pipeline implementing equations 14-19 (kernel + Sinkhorn-Knopp)
"""

from typing import Optional
import torch
import triton
import itertools
import math

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_fused_reduce_kernel,
    _sinkhorn_knopp_log_domain_kernel,
    _sinkhorn_knopp_lite,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config

_LOGGER = AiterTritonLogger()


def fused_mhc(
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
    out_pre: Optional[torch.Tensor] = None,
    out_post: Optional[torch.Tensor] = None,
    out_res: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Triton kernel inferface for mHC projection mapping (equations 14-18).


    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream

    All operations are fused in an optimized Triton kernel for maximum performance.

    Args:
        x: Input tensor with shape (M, nC) where M is batch/sequence length and
           nC is the input feature dimension (n × C in paper notation)
        phi_pre: Pre-stream projection matrix with shape (nC, n)
        phi_post: Post-stream projection matrix with shape (nC, n)
        phi_res: Residual stream projection matrix with shape (nC, n²)
        alpha_pre: Scaling factor α^pre for pre-stream (first n elements)
        alpha_post: Scaling factor α^post for post-stream (next n elements)
        alpha_res: Scaling factor α^res for residual stream (last n² elements)
        bias: Bias vector b with shape (n² + 2n,) applied after scaling
        n: Stream parameter - hyperparameter controlling manifold dimension.
           Determines output sizes: n (pre) + n (post) + n² (res) = n² + 2n
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        out_pre (Optional[torch.Tensor]): Pre-allocated output for H^pre with shape (M, n)
        out_post (Optional[torch.Tensor]): Pre-allocated output for H^post with shape (M, n)
        out_res (Optional[torch.Tensor]): Pre-allocated output for H^res with shape (M, n²)
        config (Optional[dict]): Kernel tuning parameters. If None, loaded from JSON config files.

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation (H^{pre} ∈ ℝ^{M×n})
        - H_post: (M, n) - post-processing with scaled sigmoid (H^{post} ∈ ℝ^{M×n})
        - H_res: (M, n²) - raw residual stream (identity activation, NOT doubly stochastic)

    Shape requirements:
        - x: (M, nC) where nC = n * C (flattened streams)
        - phi_pre: (nC, n)
        - phi_post: (nC, n)
        - phi_res: (nC, n²)
        - bias: (n² + 2n,)
        - outputs: H_pre (M, n), H_post (M, n), H_res (M, n²)

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C  # 4096 input features
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi_pre = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_post = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_res = torch.randn(nC, n*n, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda')
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> H_pre, H_post, H_res = fused_mhc(x, phi_pre, phi_post, phi_res, 
        ...                                   alpha_pre, alpha_post, alpha_res, bias, n)
        >>> H_pre.shape, H_post.shape, H_res.shape  # (32, 4), (32, 4), (32, 16)
    """
    # Input shape extraction
    M, K = x.shape  # M: batch/sequence, K: nC (input features)
    C = K // n  # Derive C from K and n
    K_pre, n_pre = phi_pre.shape
    K_post, n_post = phi_post.shape
    K_res, n_squared = phi_res.shape
    N_total_expected = n * n + 2 * n

    # Get config from JSON files if not provided
    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C)
    else:
        config = dict(config)  # Copy to avoid mutation
    
    num_ksplit = config.get("NUM_KSPLIT", 1)

    # Pop block sizes from config, or compute defaults
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    # BLOCK_N: Column tile size - should align with output dimension
    BLOCK_N = n_squared
    # Ensure BLOCK_K doesn't exceed K dimension
    BLOCK_K = config.pop("BLOCK_K", 64)
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))
        
    _LOGGER.info(
        f"FUSED_MHC: x={tuple(x.shape)} phi_pre={tuple(phi_pre.shape)} phi_post={tuple(phi_post.shape)} phi_res={tuple(phi_res.shape)} alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res} num_ksplit={num_ksplit}"
    )

    # Validate tensor shapes
    assert K == K_pre == K_post == K_res, (
        f"Dimension mismatch: x has K={K}, but phi_pre={K_pre}, phi_post={K_post}, phi_res={K_res}"
    )
    assert n_pre == n, f"phi_pre shape mismatch: expected (K, {n}), got ({K_pre}, {n_pre})"
    assert n_post == n, f"phi_post shape mismatch: expected (K, {n}), got ({K_post}, {n_post})"
    assert n_squared == n * n, f"phi_res shape mismatch: expected (K, {n*n}), got ({K_res}, {n_squared})"
    assert bias.shape[0] == N_total_expected, f"Bias shape mismatch: expected ({N_total_expected},), got {bias.shape}"
    assert num_ksplit >= 1, f"num_ksplit must be >= 1, got {num_ksplit}"
    
    # Validate devices
    assert x.device == phi_pre.device == phi_post.device == phi_res.device == bias.device, (
        "All tensors must be on the same device"
    )
    assert x.device.type == "cuda", "mHC kernel requires CUDA device"

    # Calculate total output dimension
    N = n * n + 2 * n
    
    # Allocate outputs if not provided
    if out_pre is None:
        out_pre = torch.empty(M, n, dtype=x.dtype, device=x.device)
    else:
        assert out_pre.shape == (M, n), f"out_pre shape mismatch: expected ({M}, {n}), got {out_pre.shape}"
        assert out_pre.dtype == x.dtype and out_pre.device == x.device
    
    if out_post is None:
        out_post = torch.empty(M, n, dtype=x.dtype, device=x.device)
    else:
        assert out_post.shape == (M, n), f"out_post shape mismatch: expected ({M}, {n}), got {out_post.shape}"
        assert out_post.dtype == x.dtype and out_post.device == x.device
    
    if out_res is None:
        out_res = torch.empty(M, n_squared, dtype=x.dtype, device=x.device)
    else:
        assert out_res.shape == (M, n_squared), f"out_res shape mismatch: expected ({M}, {n_squared}), got {out_res.shape}"
        assert out_res.dtype == x.dtype and out_res.device == x.device

    # Stream-aware grid: Each program processes exactly one stream
    n_blocks_pre = triton.cdiv(n, BLOCK_N)
    n_blocks_post = triton.cdiv(n, BLOCK_N)
    n_blocks_res = triton.cdiv(n * n, BLOCK_N)
    total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res

    if num_ksplit > 1:
        # Split-K path: use split and reduce kernels
        splitk_block_size = triton.cdiv(K, num_ksplit)
        actual_ksplit = triton.cdiv(K, splitk_block_size)
        max_ksplit = triton.next_power_of_2(num_ksplit)
        
        # Allocate intermediate buffers (float32 for precision)
        acc_pre_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
        acc_post_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
        acc_res_partial = torch.empty((num_ksplit, M, n_squared), dtype=torch.float32, device=x.device)
        acc_sq_partial = torch.empty((num_ksplit, M), dtype=torch.float32, device=x.device)
        
        # Launch split kernel with 3D grid: (M_blocks, N_blocks_total, NUM_KSPLIT)
        grid_split = (triton.cdiv(M, BLOCK_M), total_n_blocks, num_ksplit)
        _mhc_fused_split_kernel[grid_split](
            x,
            phi_pre,
            phi_post,
            phi_res,
            acc_pre_partial,
            acc_post_partial,
            acc_res_partial,
            acc_sq_partial,
            # Dimensions
            M=M,
            K=K,
            N=N,
            n=n,
            # Input strides
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_phi_pre_k=phi_pre.stride(0),
            stride_phi_pre_n=phi_pre.stride(1),
            stride_phi_post_k=phi_post.stride(0),
            stride_phi_post_n=phi_post.stride(1),
            stride_phi_res_k=phi_res.stride(0),
            stride_phi_res_n=phi_res.stride(1),
            # Intermediate buffer strides
            stride_acc_pre_k=acc_pre_partial.stride(0),
            stride_acc_pre_m=acc_pre_partial.stride(1),
            stride_acc_pre_n=acc_pre_partial.stride(2),
            stride_acc_post_k=acc_post_partial.stride(0),
            stride_acc_post_m=acc_post_partial.stride(1),
            stride_acc_post_n=acc_post_partial.stride(2),
            stride_acc_res_k=acc_res_partial.stride(0),
            stride_acc_res_m=acc_res_partial.stride(1),
            stride_acc_res_n=acc_res_partial.stride(2),
            stride_acc_sq_k=acc_sq_partial.stride(0),
            stride_acc_sq_m=acc_sq_partial.stride(1),
            # Block sizes
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLITK_BLOCK_SIZE=splitk_block_size,
            **config,
        )
        
        # Launch reduce kernel with 2D grid: (M_blocks, N_blocks_total)
        grid_reduce = (triton.cdiv(M, BLOCK_M), total_n_blocks)
        _mhc_fused_reduce_kernel[grid_reduce](
            acc_pre_partial,
            acc_post_partial,
            acc_res_partial,
            acc_sq_partial,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            out_pre,
            out_post,
            out_res,
            # Dimensions
            M=M,
            K=K,
            N=N,
            n=n,
            eps=eps,
            # Intermediate buffer strides
            stride_acc_pre_k=acc_pre_partial.stride(0),
            stride_acc_pre_m=acc_pre_partial.stride(1),
            stride_acc_pre_n=acc_pre_partial.stride(2),
            stride_acc_post_k=acc_post_partial.stride(0),
            stride_acc_post_m=acc_post_partial.stride(1),
            stride_acc_post_n=acc_post_partial.stride(2),
            stride_acc_res_k=acc_res_partial.stride(0),
            stride_acc_res_m=acc_res_partial.stride(1),
            stride_acc_res_n=acc_res_partial.stride(2),
            stride_acc_sq_k=acc_sq_partial.stride(0),
            stride_acc_sq_m=acc_sq_partial.stride(1),
            # Output strides
            stride_pre_m=out_pre.stride(0),
            stride_pre_n=out_pre.stride(1),
            stride_post_m=out_post.stride(0),
            stride_post_n=out_post.stride(1),
            stride_res_m=out_res.stride(0),
            stride_res_n=out_res.stride(1),
            # Block sizes
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ACTUAL_KSPLIT=actual_ksplit,
            **config,
        )
    else:
        # Launch 2D grid: (row blocks, stream-aware column blocks)
        grid = (triton.cdiv(M, BLOCK_M), total_n_blocks)
        # Invoke the fused Triton kernel for equations 14-18
        _mhc_fused_kernel[grid](
            x,                     # Input tensor (M, nC)
            phi_pre,               # Pre-stream projection matrix (nC, n)
            phi_post,              # Post-stream projection matrix (nC, n)
            phi_res,               # Residual stream projection matrix (nC, n²)
            alpha_pre,             # Scaling factor for pre-stream
            alpha_post,            # Scaling factor for post-stream
            alpha_res,             # Scaling factor for residual stream
            bias,                  # Bias vector (n²+2n,)
            out_pre,               # Output tensor for pre-stream (M, n)
            out_post,              # Output tensor for post-stream (M, n)
            out_res,               # Output tensor for res-stream (M, n²)
            # Shape parameters
            M=M,                   # Number of rows (batch/sequence dimension)
            K=K,                   # Input features (nC)
            N=N,                   # Output features (n²+2n)
            n=n,                   # Stream parameter
            eps=eps,               # Numerical stability epsilon for RMSNorm
            # Tensor strides for memory access
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_phi_pre_k=phi_pre.stride(0),
            stride_phi_pre_n=phi_pre.stride(1),
            stride_phi_post_k=phi_post.stride(0),
            stride_phi_post_n=phi_post.stride(1),
            stride_phi_res_k=phi_res.stride(0),
            stride_phi_res_n=phi_res.stride(1),
            stride_pre_m=out_pre.stride(0),
            stride_pre_n=out_pre.stride(1),
            stride_post_m=out_post.stride(0),
            stride_post_n=out_post.stride(1),
            stride_res_m=out_res.stride(0),
            stride_res_n=out_res.stride(1),
            # Block sizes for tiling
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            **config,
        )

    return out_pre, out_post, out_res


def mhc(
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
    sinkhorn_iters: int = 20,
    out_pre: Optional[torch.Tensor] = None,
    out_post: Optional[torch.Tensor] = None,
    out_res: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mHC projection mapping with all three streams (equations 14-19 from the paper).

    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - Eq 19: H^res = Sinkhorn(H^res) - project residual stream onto doubly stochastic
             manifold (identity activation followed by iterative row/column normalization)

    All operations are fused in optimized Triton kernels for maximum performance.

    Args:
        x: Input tensor with shape (M, nC) where M is batch/sequence length and
           nC is the input feature dimension (n × C in paper notation)
        phi_pre: Pre-stream projection matrix with shape (nC, n)
        phi_post: Post-stream projection matrix with shape (nC, n)
        phi_res: Residual stream projection matrix with shape (nC, n²)
        alpha_pre: Scaling factor α^pre for pre-stream (first n elements)
        alpha_post: Scaling factor α^post for post-stream (next n elements)
        alpha_res: Scaling factor α^res for residual stream (last n² elements)
        bias: Bias vector b with shape (n² + 2n,) applied after scaling
        n: Stream parameter - hyperparameter controlling manifold dimension.
           Determines output sizes: n (pre) + n (post) + n² (res) = n² + 2n
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations for Eq 19 (default: 20)
        out_pre (Optional[torch.Tensor]): Pre-allocated output for H^pre with shape (M, n)
        out_post (Optional[torch.Tensor]): Pre-allocated output for H^post with shape (M, n)
        out_res (Optional[torch.Tensor]): Pre-allocated output for H^res with shape (M, n²)

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation (H^{pre} ∈ ℝ^{M×n})
        - H_post: (M, n) - post-processing with scaled sigmoid (H^{post} ∈ ℝ^{M×n})
        - H_res: (M, n²) - doubly stochastic residual connection (H^{res} ∈ ℝ^{M×n²})

    Shape requirements:
        - x: (M, nC) where nC = n * C (flattened streams)
        - phi_pre: (nC, n)
        - phi_post: (nC, n)
        - phi_res: (nC, n²)
        - bias: (n² + 2n,)
        - outputs: H_pre (M, n), H_post (M, n), H_res (M, n²)

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C  # 4096 input features
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi_pre = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_post = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_res = torch.randn(nC, n*n, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda')
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> 
        >>> # Full mHC with Sinkhorn-Knopp (Eq 14-19)
        >>> H_pre, H_post, H_res = mhc(x, phi_pre, phi_post, phi_res, 
        ...                            alpha_pre, alpha_post, alpha_res, bias, n)
        >>> H_pre.shape, H_post.shape, H_res.shape  # (32, 4), (32, 4), (32, 16)
    """
    _LOGGER.info(
        f"MHC: calling fused_mhc() then sinkhorn_knopp() with {sinkhorn_iters} iterations"
    )
    
    # Call fused_mhc function (Eq 14-18)
    out_pre, out_post, out_res = fused_mhc(
        x, phi_pre, phi_post, phi_res,
        alpha_pre, alpha_post, alpha_res,
        bias, n, eps,
        out_pre=out_pre, out_post=out_post, out_res=out_res
    )
    
    # Apply Sinkhorn-Knopp (Equation 19) to make H_res doubly stochastic
    # Reshape H_res from (M, n²) to (M, n, n) for Sinkhorn kernel
    M = out_res.shape[0]
    H_res_3d = out_res.view(M, n, n)
    
    # # Iterative Sinkhorn-Knopp - modifies H_res_3d in-place
    # sinkhorn_knopp(H_res_3d, num_iters=sinkhorn_iters, out=H_res_3d)
    
    # Zero-iteration Sinkhorn-Knopp (mHC-Lite) - modifies H_res_3d in-place
    sinkhorn_knopp_lite(H_res_3d, out=H_res_3d)
    
    return out_pre, out_post, out_res


def sinkhorn_knopp(
    logits: torch.Tensor,
    num_iters: int = 20,
    out: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
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
        config (Optional[dict]): Kernel tuning parameters. If None, loaded from JSON config files.

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

    # Check N is power of 2 because Triton arange requires even number of sizes
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

    # Get config from JSON files if not provided
    if config is None:
        config, _ = get_mhc_config("MHC_SINKHORN", M)
    else:
        config = dict(config)  # Copy to avoid mutation
    
    # Filter out BLOCK_SIZE - only used by sinkhorn_knopp_lite, not the log domain kernel
    kernel_config = {k: v for k, v in config.items() if k != "BLOCK_SIZE"}

    # Grid: one program per batch element, need large batch size for optimal performance
    grid = (triton.cdiv(M, config["BLOCK_M"]),)

    _sinkhorn_knopp_log_domain_kernel[grid](
        logits,
        out,
        M,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        N=N,
        NUM_ITERS=num_iters,
        **kernel_config,
    )

    return out


# Global cache for permutation matrices to avoid recomputation
_PERM_MATS_CACHE = {}


def _generate_permutation_matrices(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate all n! permutation matrices of size n×n.
    
    A permutation matrix is a square binary matrix with exactly one 1 in each row and column.
    All permutation matrices are doubly stochastic (rows and columns sum to 1).
    
    Args:
        n: Matrix size (typically 4 for mHC with 4 streams)
        device: Target device (e.g., 'cuda', 'cpu')
        dtype: Data type for matrices
    
    Returns:
        torch.Tensor: Shape (n!, n, n) containing all permutation matrices
    
    Example:
        >>> perms = _generate_permutation_matrices(3, 'cpu')
        >>> perms.shape
        torch.Size([6, 3, 3])  # 3! = 6
        >>> perms[0]  # Identity permutation
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
    """
    cache_key = (n, device, dtype)
    if cache_key in _PERM_MATS_CACHE:
        return _PERM_MATS_CACHE[cache_key]
    
    # Generate all permutations of indices [0, 1, ..., n-1]
    perms = list(itertools.permutations(range(n)))
    n_factorial = len(perms)
    
    # Convert permutations to permutation matrices directly on target device
    # For each permutation p, create matrix where M[i, p[i]] = 1
    indices = torch.tensor(perms, dtype=torch.long, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    perm_mats = eye[indices]  # Shape: (n!, n, n)
    
    # Cache for future use
    _PERM_MATS_CACHE[cache_key] = perm_mats
    
    return perm_mats


def sinkhorn_knopp_lite(
    logits: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Zero-iteration Sinkhorn-Knopp using permutation matrix basis (mHC-Lite).
    
    Instead of iteratively normalizing matrices to be doubly stochastic, this function
    directly parameterizes them as convex combinations of permutation matrices, based
    on the Birkhoff-von Neumann theorem.
    
    Mathematical foundation:
        Any doubly stochastic matrix M can be written as:
        M = Σ_{k=1}^{n!} λ_k * P_k
        where P_k are permutation matrices and λ_k ≥ 0, Σλ_k = 1
    
    Algorithm:
        1. Generate all n! permutation matrices (cached)
        2. Apply softmax to input logits to get weights λ
        3. Compute weighted sum: M = Σ_k λ_k * P_k
        4. Result is automatically doubly stochastic (no iterations!)
    
    Advantages over iterative Sinkhorn-Knopp:
        ✓ Zero iterations required (one-shot computation)
        ✓ Exact doubly stochastic property
        ✓ Faster inference (no iteration loop)
        ✓ Better numerical stability
    
    Limitations:
        ✗ Requires storing n! permutation matrices in memory
        ✗ Only practical for small n (typically n=4, where 4!=24)
        ✗ For n=5: 5!=120, n=6: 6!=720 (memory intensive)
    
    Args:
        logits (torch.Tensor): Input raw logits with shape (M, N, N), where:
            - M is the batch size
            - N is the matrix size (typically n=4)
        out (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N, N).
            If None, a new tensor is allocated. When provided, enables in-place operation.
        config (Optional[dict]): Kernel tuning parameters. If None, loaded from JSON config files.
    
    Returns:
        torch.Tensor: Doubly stochastic matrices with shape (M, N, N).
            Each matrix in the batch has rows and columns summing to 1.
            If `out` is provided, returns the same tensor (in-place operation).
    
    Example:
        >>> # Zero-iteration doubly stochastic projection
        >>> logits = torch.randn(16, 4, 4, device='cuda')
        >>> P = sinkhorn_knopp_lite(logits, out=logits)
        >>> print(P.sum(dim=-1))  # Row sums = 1
        >>> print(P.sum(dim=-2))  # Col sums = 1
    
    Reference:
        "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations"
        https://arxiv.org/abs/2601.05732
    """
    import math
    
    # Input validation
    assert logits.dim() == 3, f"logits must be 3D (M, N, N), got {logits.dim()}D"
    M, n, n2 = logits.shape
    assert n == n2, f"Last two dimensions must be equal, got ({n}, {n2})"
    assert n <= 6, f"Matrix size n={n} too large (max n=6 for n!=720)"
    
    _LOGGER.info(
        f"Sinkhorn-Knopp-Lite: logits=({M}, {n}, {n}) (zero iterations)"
    )
    
    n_factorial = math.factorial(n)
    
    # Generate permutation matrices (cached)
    perm_mats = _generate_permutation_matrices(n, logits.device, logits.dtype)
    assert perm_mats.shape == (n_factorial, n, n), (
        f"Expected perm_mats shape ({n_factorial}, {n}, {n}), "
        f"got {perm_mats.shape}"
    )
    
    # Ensure contiguous
    logits = logits.contiguous()
    perm_mats = perm_mats.contiguous()
    
    # Allocate output if not provided
    if out is None:
        out = torch.empty((M, n, n), dtype=logits.dtype, device=logits.device)
    else:
        assert out.shape == (M, n, n), f"out.shape {out.shape} must be ({M}, {n}, {n})"
        out = out.contiguous()
    
    # Get config from JSON files if not provided
    if config is None:
        config, _ = get_mhc_config("MHC_SINKHORN", M)
    else:
        config = dict(config)  # Copy to avoid mutation
    
    # Grid: one program per BLOCK_M batch elements
    grid = (triton.cdiv(M, config["BLOCK_M"]),)
    
    # Compute next power of 2 for N_FACTORIAL (required by Triton's tl.arange)
    n_factorial_pow2 = triton.next_power_of_2(n_factorial)
    
    _sinkhorn_knopp_lite[grid](
        logits,
        perm_mats,
        out,
        M,
        N=n,
        N_FACTORIAL=n_factorial,
        N_FACTORIAL_POW2=n_factorial_pow2,
        stride_logits_m=logits.stride(0),
        stride_logits_perm=logits.stride(1),
        stride_perm_idx=perm_mats.stride(0),
        stride_perm_row=perm_mats.stride(1),
        stride_perm_col=perm_mats.stride(2),
        stride_out_batch=out.stride(0),
        stride_out_row=out.stride(1),
        stride_out_col=out.stride(2),
        **config,
    )
    
    return out
