# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
High-level Python wrapper for mHC (manifold-constrained Hyper Connection).

Provides two main functions:
- fused_mhc(): Low-level interface for equations 14-18 (fused kernel only)
- mhc(): Complete pipeline implementing equations 14-19 (kernel + Sinkhorn-Knopp)

Supports two modes for H_res computation:
- "sinkhorn": Standard mHC with Sinkhorn-Knopp iterations (approximate doubly stochastic)
- "lite": mHC-lite with convex combination of permutations (exact doubly stochastic)
"""

from itertools import permutations as itertools_permutations
from math import factorial
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
    _mhc_lite_fused_split_kernel,
    _mhc_lite_fused_reduce_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config

_LOGGER = AiterTritonLogger()

# Cache for permutation matrices to avoid regenerating them
_PERM_MATRICES_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def get_permutation_matrices(n: int, device: torch.device) -> torch.Tensor:
    """
    Generate all n! permutation matrices for mHC-lite mode.
    
    Args:
        n: Stream parameter
        device: Target CUDA device
        
    Returns:
        P: (n!, n, n) tensor of permutation matrices, where P[k] is the k-th
           permutation matrix with exactly one 1.0 per row and column.
           
    Example:
        >>> P = get_permutation_matrices(4, torch.device('cuda'))
        >>> P.shape  # (24, 4, 4) - 24 = 4!
        >>> P[0]  # Identity matrix [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    """
    key = (n, device)
    if key not in _PERM_MATRICES_CACHE:
        all_perms = list(itertools_permutations(range(n)))
        K = len(all_perms)  # K = n!
        P = torch.zeros(K, n, n, device=device, dtype=torch.float32)
        for k, perm in enumerate(all_perms):
            for i, j in enumerate(perm):
                P[k, i, j] = 1.0
        _PERM_MATRICES_CACHE[key] = P
    return _PERM_MATRICES_CACHE[key]


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
    hres_mode: str = "sinkhorn",
    out_pre: Optional[torch.Tensor] = None,
    out_post: Optional[torch.Tensor] = None,
    out_res: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Triton kernel interface for mHC projection mapping (equations 14-18).

    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - H^res activation depends on hres_mode:
        - "sinkhorn": identity (raw logits for Sinkhorn-Knopp post-processing)
        - "lite": softmax + permutation combination (exact doubly stochastic)

    All operations are fused in an optimized Triton kernel for maximum performance.

    Args:
        x: Input tensor with shape (M, nC) where M is batch/sequence length and
           nC is the input feature dimension (n × C in paper notation)
        phi_pre: Pre-stream projection matrix with shape (nC, n)
        phi_post: Post-stream projection matrix with shape (nC, n)
        phi_res: Residual stream projection matrix with shape:
            - (nC, n²) for sinkhorn mode
            - (nC, n!) for lite mode (e.g., (nC, 24) for n=4)
        alpha_pre: Scaling factor α^pre for pre-stream
        alpha_post: Scaling factor α^post for post-stream
        alpha_res: Scaling factor α^res for residual stream
        bias: Bias vector b with shape:
            - (n² + 2n,) for sinkhorn mode
            - (n! + 2n,) for lite mode (e.g., (32,) for n=4)
        n: Stream parameter - hyperparameter controlling manifold dimension.
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        hres_mode: Mode for H_res computation:
            - "sinkhorn": Raw logits with identity activation (default)
            - "lite": Softmax + permutation combination (exact doubly stochastic)
        out_pre (Optional[torch.Tensor]): Pre-allocated output for H^pre with shape (M, n)
        out_post (Optional[torch.Tensor]): Pre-allocated output for H^post with shape (M, n)
        out_res (Optional[torch.Tensor]): Pre-allocated output for H^res with shape (M, n²)
        config (Optional[dict]): Kernel tuning parameters. If None, loaded from JSON config files.

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation
        - H_post: (M, n) - post-processing with scaled sigmoid
        - H_res: (M, n²) - residual stream:
            - sinkhorn mode: raw logits (NOT doubly stochastic)
            - lite mode: exact doubly stochastic matrix

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C  # 4096 input features
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi_pre = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_post = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> # Sinkhorn mode (default)
        >>> phi_res = torch.randn(nC, n*n, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda')
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> H_pre, H_post, H_res = fused_mhc(x, phi_pre, phi_post, phi_res, 
        ...                                   alpha_pre, alpha_post, alpha_res, bias, n)
        >>> 
        >>> # Lite mode (exact doubly stochastic)
        >>> from math import factorial
        >>> phi_res_lite = torch.randn(nC, factorial(n), dtype=torch.bfloat16, device='cuda')
        >>> bias_lite = torch.randn(factorial(n) + 2*n, dtype=torch.float32, device='cuda')
        >>> H_pre, H_post, H_res = fused_mhc(x, phi_pre, phi_post, phi_res_lite, 
        ...                                   alpha_pre, alpha_post, alpha_res, bias_lite, n, hres_mode="lite")
    """
    # Validate hres_mode
    assert hres_mode in ("sinkhorn", "lite"), (
        f"hres_mode must be 'sinkhorn' or 'lite', got {hres_mode}"
    )
    hres_lite_mode = hres_mode == "lite"
    
    # Input shape extraction
    M, K = x.shape  # M: batch/sequence, K: nC (input features)
    C = K // n  # Derive C from K and n
    K_pre, n_pre = phi_pre.shape
    K_post, n_post = phi_post.shape
    K_res, phi_res_cols = phi_res.shape
    
    # Calculate expected dimensions based on mode
    n_squared = n * n
    n_factorial = factorial(n) if hres_lite_mode else n_squared
    n_res_expected = n_factorial if hres_lite_mode else n_squared
    N_total_expected = n_res_expected + 2 * n  # n (pre) + n (post) + n_res

    # Get config from JSON files if not provided
    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C)
    else:
        config = dict(config)  # Copy to avoid mutation
    
    num_ksplit = config.get("NUM_KSPLIT", 1)

    # Pop block sizes from config, or compute defaults
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    # BLOCK_N: Column tile size (must be power of 2 for Triton arange)
    # fit both input weights and output matrix
    BLOCK_N = max(n_factorial, n_squared) if hres_lite_mode else n_squared
    BLOCK_N = triton.next_power_of_2(BLOCK_N)
    # Ensure BLOCK_K doesn't exceed K dimension
    BLOCK_K = config.pop("BLOCK_K", 64)
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))
        
    _LOGGER.info(
        f"FUSED_MHC: x={tuple(x.shape)} phi_pre={tuple(phi_pre.shape)} phi_post={tuple(phi_post.shape)} "
        f"phi_res={tuple(phi_res.shape)} alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res} "
        f"hres_mode={hres_mode} num_ksplit={num_ksplit}"
    )

    # Validate tensor shapes
    assert K == K_pre == K_post == K_res, (
        f"Dimension mismatch: x has K={K}, but phi_pre={K_pre}, phi_post={K_post}, phi_res={K_res}"
    )
    assert n_pre == n, f"phi_pre shape mismatch: expected (K, {n}), got ({K_pre}, {n_pre})"
    assert n_post == n, f"phi_post shape mismatch: expected (K, {n}), got ({K_post}, {n_post})"
    
    # Validate phi_res shape based on mode
    if hres_lite_mode:
        assert phi_res_cols == n_factorial, (
            f"In lite mode, phi_res must have {n_factorial} columns (n!={n_factorial}), got {phi_res_cols}"
        )
    else:
        assert phi_res_cols == n_squared, (
            f"In sinkhorn mode, phi_res must have {n_squared} columns (n²), got {phi_res_cols}"
        )
    
    assert bias.shape[0] == N_total_expected, (
        f"Bias shape mismatch: expected ({N_total_expected},), got {bias.shape}"
    )
    assert num_ksplit >= 1, f"num_ksplit must be >= 1, got {num_ksplit}"
    
    # Validate devices
    assert x.device == phi_pre.device == phi_post.device == phi_res.device == bias.device, (
        "All tensors must be on the same device"
    )
    assert x.device.type == "cuda", "mHC kernel requires CUDA device"

    # Calculate total output dimension (for bias indexing)
    N = N_total_expected
    
    # Get permutation matrices for lite mode
    perm_matrices = get_permutation_matrices(n, x.device) if hres_lite_mode else None
    
    # Allocate outputs if not provided
    # H_res is always (M, n²) regardless of mode (lite outputs n² after perm combination)
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
    # For lite mode, res stream needs to fit in one block for softmax
    n_blocks_res = 1 if hres_lite_mode else triton.cdiv(n_squared, BLOCK_N)
    total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res

    if num_ksplit > 1:
        # Split-K path: use split and reduce kernels
        splitk_block_size = triton.cdiv(K, num_ksplit)
        actual_ksplit = triton.cdiv(K, splitk_block_size)
        # max_ksplit = triton.next_power_of_2(num_ksplit)
        
        # Allocate intermediate buffers (float32 for precision)
        # acc_res size depends on mode: n² for sinkhorn, n! for lite
        acc_res_cols = n_factorial if hres_lite_mode else n_squared
        acc_pre_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
        acc_post_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
        acc_res_partial = torch.empty((num_ksplit, M, acc_res_cols), dtype=torch.float32, device=x.device)
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
            n_squared=n_squared,
            n_factorial=n_factorial,
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
            HRES_LITE_MODE=hres_lite_mode,
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
            perm_matrices if hres_lite_mode else torch.empty(0, device=x.device),  # perm_ptr
            # Dimensions
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            n_factorial=n_factorial,
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
            stride_perm_k=perm_matrices.stride(0) if hres_lite_mode else 0,
            stride_perm_ij=1 if hres_lite_mode else 0,  # contiguous flattened
            # Block sizes
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ACTUAL_KSPLIT=actual_ksplit,
            HRES_LITE_MODE=hres_lite_mode,
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
            phi_res,               # Residual stream projection matrix (nC, n² or n!)
            alpha_pre,             # Scaling factor for pre-stream
            alpha_post,            # Scaling factor for post-stream
            alpha_res,             # Scaling factor for residual stream
            bias,                  # Bias vector
            out_pre,               # Output tensor for pre-stream (M, n)
            out_post,              # Output tensor for post-stream (M, n)
            out_res,               # Output tensor for res-stream (M, n²)
            perm_matrices if hres_lite_mode else torch.empty(0, device=x.device),  # perm_ptr
            # Shape parameters
            M=M,                   # Number of rows (batch/sequence dimension)
            K=K,                   # Input features (nC)
            N=N,                   # Output features
            n=n,                   # Stream parameter
            n_squared=n_squared,   # n*n (precomputed for constexpr usage)
            n_factorial=n_factorial,  # n! for lite mode
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
            stride_perm_k=perm_matrices.stride(0) if hres_lite_mode else 0,
            stride_perm_ij=1 if hres_lite_mode else 0,  # contiguous flattened
            # Block sizes for tiling
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            HRES_LITE_MODE=hres_lite_mode,
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
    hres_mode: str = "sinkhorn",
    sinkhorn_iters: int = 20,
    out_pre: Optional[torch.Tensor] = None,
    out_post: Optional[torch.Tensor] = None,
    out_res: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mHC projection mapping (iterative Sinkhorn-Knopp).

    mHC (manifold-constrained Hyper Connection) implements a novel neural architecture
    component that projects inputs through three specialized streams with manifold
    constraints. This is the standard implementation using iterative Sinkhorn-Knopp
    refinement. This function implements:
    
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - H^res doubly stochastic via hres_mode:
        - "sinkhorn": Eq 19 Sinkhorn-Knopp iterations (approximate)
        - "lite": mHC-lite convex combination of permutations (exact)

    All operations are fused in optimized Triton kernels for maximum performance.

    Args:
        x: Input tensor with shape (M, nC) where M is batch/sequence length and
           nC is the input feature dimension (n × C in paper notation)
        phi_pre: Pre-stream projection matrix with shape (nC, n)
        phi_post: Post-stream projection matrix with shape (nC, n)
        phi_res: Residual stream projection matrix with shape:
            - (nC, n²) for sinkhorn mode
            - (nC, n!) for lite mode
        alpha_pre: Scaling factor α^pre for pre-stream
        alpha_post: Scaling factor α^post for post-stream
        alpha_res: Scaling factor α^res for residual stream
        bias: Bias vector b with shape:
            - (n² + 2n,) for sinkhorn mode
            - (n! + 2n,) for lite mode
        n: Stream parameter - hyperparameter controlling manifold dimension.
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        hres_mode: Mode for H_res computation:
            - "sinkhorn": Use Sinkhorn-Knopp iterations
            - "lite": Use mHC-lite convex combination
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (only for sinkhorn mode)
        out_pre (Optional[torch.Tensor]): Pre-allocated output for H^pre with shape (M, n)
        out_post (Optional[torch.Tensor]): Pre-allocated output for H^post with shape (M, n)
        out_res (Optional[torch.Tensor]): Pre-allocated output for H^res with shape (M, n²).
                 Note: Will be iteratively refined by Sinkhorn-Knopp to be doubly stochastic.

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation
        - H_post: (M, n) - post-processing with scaled sigmoid
        - H_res: (M, n²) - doubly stochastic residual connection

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi_pre = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_post = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> # Sinkhorn mode (default)
        >>> phi_res = torch.randn(nC, n*n, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda')
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> H_pre, H_post, H_res = mhc(x, phi_pre, phi_post, phi_res, 
        ...                             alpha_pre, alpha_post, alpha_res, bias, n, hres_mode="sinkhorn")
        >>> 
        >>> # Lite mode (exact doubly stochastic)
        >>> from math import factorial
        >>> phi_res_lite = torch.randn(nC, factorial(n), dtype=torch.bfloat16, device='cuda')
        >>> bias_lite = torch.randn(factorial(n) + 2*n, dtype=torch.float32, device='cuda')
        >>> H_pre, H_post, H_res = mhc(x, phi_pre, phi_post, phi_res_lite, 
        ...                            alpha_pre, alpha_post, alpha_res, bias_lite, n, hres_mode="lite")
    """
    _LOGGER.info(
        f"MHC: calling fused_mhc() with hres_mode='lite'" if hres_mode == "lite" \
        else f"MHC: calling fused_mhc() then sinkhorn_knopp() with {sinkhorn_iters} iterations"
    )
    res = fused_mhc(
        x, phi_pre, phi_post, phi_res,
        alpha_pre, alpha_post, alpha_res,
        bias, n, eps, hres_mode=hres_mode,
        out_pre=out_pre, out_post=out_post, out_res=out_res,
        config=config
    )

    if hres_mode == "lite":
        return res
    else:
        # Call fused_mhc function (Eq 14-18)
        out_pre, out_post, out_res = res
        
        # Apply Sinkhorn-Knopp (Equation 19) to make H_res doubly stochastic
        # Reshape H_res from (M, n²) to (M, n, n) for Sinkhorn kernel
        M = out_res.shape[0]
        out_res_3d = out_res.view(M, n, n)
        sinkhorn_knopp(out_res_3d, num_iters=sinkhorn_iters, out=out_res_3d)
        
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

    # Pop BLOCK_M for grid calculation (handle legacy BLOCK_SIZE name)
    BLOCK_M = config.pop("BLOCK_M", config.pop("BLOCK_SIZE", 8))
    
    # Grid: one program per batch element, need large batch size for optimal performance
    grid = (triton.cdiv(M, BLOCK_M),)

    _sinkhorn_knopp_log_domain_kernel[grid](
        logits,
        out,
        M,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        N=N,
        NUM_ITERS=num_iters,
        BLOCK_M=BLOCK_M,
        **config,
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
    
    # Convert permutations to permutation matrices directly on target device
    # For each permutation p, create matrix where M[i, p[i]] = 1
    indices = torch.tensor(perms, dtype=torch.long, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    perm_mats = eye[indices]  # Shape: (n!, n, n)
    
    # Cache for future use
    _PERM_MATS_CACHE[cache_key] = perm_mats
    
    return perm_mats


def mhc_lite(
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
    Compute mHC-Lite projection mapping (non-iterative Sinkhorn-Knopp).

    mHC-Lite (manifold-constrained Hyper Connection Lite) is a computationally efficient 
    variant of mHC that eliminates the need for 20+ Sinkhorn-Knopp iterations by directly 
    computing doubly stochastic matrices through a convex combination of permutation matrices. 
    This is the optimized implementation with single-pass doubly stochastic projection. 
    This function implements:
    
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - Eq 19 (optimized, eq 4 & 5 of "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations"): 
             H^res = Σ_k a_k · P_k where a = softmax(α^res·H̃^res + b^res)
             Instead of iterative Sinkhorn-Knopp, compute H^res as a weighted sum
             of all n! permutation matrices P_k, with weights a_k from softmax. 
             Here P is the sequence of n × n permutation matrices.
             This produces doubly stochastic matrices by construction since any
             convex combination of permutation matrices is doubly stochastic. 

    CHARACTERISTICS of mHC-Lite:
    - Uses phi_res with shape (nC, n!)
    - Produces n! logits instead of n² raw elements
    - Applies softmax to obtain convex combination weights
    - Fuses doubly stochastic projection into forward pass (no iteration needed)
    - Bias vector has shape (n! + 2n,) instead of (n² + 2n,)
    - Single-pass operation: all equations 14-19 executed in one kernel launch

    All operations are fused in optimized Triton kernels with split-reduce for large K.
    Unlike mhc, there's no separate Sinkhorn-Knopp stage. The optimized 
    Eq 19 (softmax + permutation matrix combination) is integrated into the reduce kernel.

    Args:
        x: Input tensor with shape (M, nC) where M is batch/sequence length and
           nC is the input feature dimension (n × C in paper notation)
        phi_pre: Pre-stream projection matrix with shape (nC, n)
        phi_post: Post-stream projection matrix with shape (nC, n)
        phi_res: Residual stream projection matrix with shape (nC, n!) where n! is
                 the number of permutation matrices (factorial of n).
        alpha_pre: Scaling factor α^pre for pre-stream (first n elements)
        alpha_post: Scaling factor α^post for post-stream (next n elements)
        alpha_res: Scaling factor α^res for residual stream (last n! elements)
        bias: Bias vector b with shape (n! + 2n,) applied after scaling.
              Contains biases for all three streams concatenated.
        n: Stream parameter - hyperparameter controlling manifold dimension.
           Determines output sizes: n (pre) + n (post) + n² (res) = n² + 2n
           Also determines number of permutation matrices: n!
           Typical values: 4 (24 perms), 5 (120 perms)
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        out_pre (Optional[torch.Tensor]): Pre-allocated output for H^pre with shape (M, n)
        out_post (Optional[torch.Tensor]): Pre-allocated output for H^post with shape (M, n)
        out_res (Optional[torch.Tensor]): Pre-allocated output for H^res with shape (M, n²).
                Note: Despite phi_res having n! columns, the output H^res has shape (M, n²)
                because it's the weighted sum of n²-element permutation matrices.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N, BLOCK_K, NUM_KSPLIT).
                                If None, automatically loaded from configuration files.

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation (H^{pre} ∈ ℝ^{M×n})
        - H_post: (M, n) - post-processing with scaled sigmoid (H^{post} ∈ ℝ^{M×n})
        - H_res: (M, n²) - doubly stochastic residual connection (H^{res} ∈ ℝ^{M×n²})
                 **GUARANTEED DOUBLY STOCHASTIC** by construction (no iterations needed)

    Shape requirements:
        - x: (M, nC) where nC = n * C (flattened streams)
        - phi_pre: (nC, n)
        - phi_post: (nC, n)
        - phi_res: (nC, n!)
        - bias: (n! + 2n,)
        - outputs: H_pre (M, n), H_post (M, n), H_res (M, n²)

    Example:
        >>> M, n, C = 32, 4, 1024
        >>> nC = n * C  # 4096 input features
        >>> n_factorial = math.factorial(n)  # 24 for n=4
        >>> 
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi_pre = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_post = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_res = torch.randn(nC, n_factorial, dtype=torch.bfloat16, device='cuda')  # Note: n! not n²
        >>> bias = torch.randn(n_factorial + 2*n, dtype=torch.float32, device='cuda')  # Note: n! + 2n
        >>> alpha_pre, alpha_post, alpha_res = 1.0, 1.5, 0.8
        >>> 
        >>> # mHC-Lite: Single forward pass, no Sinkhorn iterations needed!
        >>> H_pre, H_post, H_res = mhc_lite(x, phi_pre, phi_post, phi_res, 
        ...                                  alpha_pre, alpha_post, alpha_res, bias, n)
        >>> H_pre.shape, H_post.shape, H_res.shape  # (32, 4), (32, 4), (32, 16)
        >>> 
        >>> # Verify H_res is doubly stochastic (rows and columns sum to 1)
        >>> H_res_matrix = H_res.view(M, n, n)
        >>> row_sums = H_res_matrix.sum(dim=-1)  # Should be all 1s
        >>> col_sums = H_res_matrix.sum(dim=-2)  # Should be all 1s

    Reference:
        Paper: "mHC-Lite: You Don't Need 20 Sinkhorn-Knopp Iterations"
        arXiv: https://arxiv.org/abs/2601.05732
        
        The key insight: Any convex combination of permutation matrices is doubly
        stochastic. By projecting to n! logits, applying softmax (which gives convex
        weights), and computing H^res = Σ_k softmax(logits)_k · P_k, we obtain a
        doubly stochastic matrix without any iterative refinement.
    """
    # Input validation
    M, K = x.shape
    import math
    n_factorial = math.factorial(n)
    n_squared = n * n
    
    # Validate phi_res has n! columns (one per permutation matrix). Instead of n² used in mHC
    assert phi_res.shape == (K, n_factorial), (
        f"phi_res shape mismatch: expected ({K}, {n_factorial}), got {phi_res.shape}"
    )
    # Validate bias has n! + 2n elements (n! for res + n for pre + n for post)
    assert bias.shape[0] == n_factorial + 2 * n, (
        f"bias shape mismatch: expected ({n_factorial + 2 * n},), got {bias.shape}"
    )
    
    _LOGGER.info(
        f"TRUE mHC-Lite: x=({M}, {K}), phi_res=({K}, {n_factorial}), n={n}"
    )
    
    # Allocate outputs if not provided
    if out_pre is None:
        out_pre = torch.empty(M, n, dtype=x.dtype, device=x.device)
    if out_post is None:
        out_post = torch.empty(M, n, dtype=x.dtype, device=x.device)
    if out_res is None:
        out_res = torch.empty(M, n_squared, dtype=x.dtype, device=x.device)
    
    # Generate permutation matrices (cached)
    perm_mats = _generate_permutation_matrices(n, x.device, x.dtype)
    
    # Find next power of 2 >= n_factorial for vectorized softmax
    n_factorial_pow2 = 1 << (n_factorial - 1).bit_length()
    
    # Load configuration
    C = K // n
    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C)
    else:
        base_config, _ = get_mhc_config("MHC_FUSED", M, C)
        config = {**base_config, **config}
    
    # Pop block sizes and num_ksplit from config to avoid conflicts
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    BLOCK_N = config.pop("BLOCK_N", 16)
    BLOCK_K = config.pop("BLOCK_K", 64)
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))
    num_ksplit = config.pop("NUM_KSPLIT", 2)  # Default to 2 for split-reduce parallelization
    
    # Total bias size for kernel
    N = n_factorial + 2 * n
    
    # Calculate total output blocks: pre + post + res
    n_blocks_pre = triton.cdiv(n, BLOCK_N)
    n_blocks_post = triton.cdiv(n, BLOCK_N)
    n_blocks_res = triton.cdiv(n_squared, BLOCK_N)
    total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res
    
    # Require split-reduce optimization (NUM_KSPLIT > 1)
    assert num_ksplit > 1, "NUM_KSPLIT must be > 1 for split-reduce optimization"
    
    # Split-K path: use split and reduce kernels
    splitk_block_size = triton.cdiv(K, num_ksplit)
    actual_ksplit = triton.cdiv(K, splitk_block_size)
        
    # Allocate intermediate buffers (float32 for precision)
    acc_pre_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
    acc_post_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
    acc_res_partial = torch.empty((num_ksplit, M, n_factorial), dtype=torch.float32, device=x.device)  # n! not n²
    acc_sq_partial = torch.empty((num_ksplit, M), dtype=torch.float32, device=x.device)
    
    # Launch split kernel with 3D grid: (M_blocks, N_blocks_total, NUM_KSPLIT)
    grid_split = (triton.cdiv(M, BLOCK_M), total_n_blocks, num_ksplit)
    _mhc_lite_fused_split_kernel[grid_split](
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
        N_FACTORIAL=n_factorial,
        # Input strides
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_phi_pre_k=phi_pre.stride(0),
        stride_phi_pre_n=phi_pre.stride(1),
        stride_phi_post_k=phi_post.stride(0),
        stride_phi_post_n=phi_post.stride(1),
        stride_phi_res_k=phi_res.stride(0),
        stride_phi_res_n=phi_res.stride(1),  # n! dimension
        # Intermediate buffer strides
        stride_acc_pre_k=acc_pre_partial.stride(0),
        stride_acc_pre_m=acc_pre_partial.stride(1),
        stride_acc_pre_n=acc_pre_partial.stride(2),
        stride_acc_post_k=acc_post_partial.stride(0),
        stride_acc_post_m=acc_post_partial.stride(1),
        stride_acc_post_n=acc_post_partial.stride(2),
        stride_acc_res_k=acc_res_partial.stride(0),
        stride_acc_res_m=acc_res_partial.stride(1),
        stride_acc_res_n=acc_res_partial.stride(2),  # n! dimension
        stride_acc_sq_k=acc_sq_partial.stride(0),
        stride_acc_sq_m=acc_sq_partial.stride(1),
        # Block sizes
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_KSPLIT=num_ksplit,
        SPLITK_BLOCK_SIZE=splitk_block_size,
        **config,
    )
    
    # Launch reduce kernel with 2D grid: (M_blocks, N_blocks_total)
    grid_reduce = (triton.cdiv(M, BLOCK_M), total_n_blocks)
    _mhc_lite_fused_reduce_kernel[grid_reduce](
        acc_pre_partial,
        acc_post_partial,
        acc_res_partial,
        acc_sq_partial,
        perm_mats,
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
        N_FACTORIAL=n_factorial,
        N_FACTORIAL_POW2=n_factorial_pow2,
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
        stride_acc_res_n=acc_res_partial.stride(2),  # n! dimension
        stride_acc_sq_k=acc_sq_partial.stride(0),
        stride_acc_sq_m=acc_sq_partial.stride(1),
        # Permutation matrix strides
        stride_perm_idx=perm_mats.stride(0),
        stride_perm_row=perm_mats.stride(1),
        stride_perm_col=perm_mats.stride(2),
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
        NUM_KSPLIT=num_ksplit,
        ACTUAL_KSPLIT=actual_ksplit,
        **config,
    )
    
    return out_pre, out_post, out_res
