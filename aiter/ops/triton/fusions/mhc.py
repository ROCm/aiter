# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
High-level Python wrapper for mHC (manifold-constrained Hyper Connection).

Provides two main functions:
- fused_mhc(): Low-level interface for equations 14-18 (fused kernel only)
- mhc(): Complete pipeline implementing equations 14-19 (kernel + Sinkhorn-Knopp)

Supports two modes for H_res computation via hres_mode parameter:
- "sinkhorn": Standard mHC with Sinkhorn-Knopp iterations (approximate doubly stochastic)
- "lite": mHC-lite with convex combination of permutations (exact doubly stochastic)
"""

from itertools import permutations as itertools_permutations
from math import factorial
from typing import Optional
import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_fused_reduce_kernel,
    _mhc_fused_lite_kernel,
    _mhc_fused_split_lite_kernel,
    _mhc_fused_reduce_lite_kernel,
    _sinkhorn_knopp_log_domain_kernel,
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
        P: (n!, n*n) tensor of flattened permutation matrices, where P[k] is the k-th
           permutation matrix flattened to 1D with exactly one 1.0 per row and column.
           
    Example:
        >>> P = get_permutation_matrices(4, torch.device('cuda'))
        >>> P.shape  # (24, 16) - 24 = 4!, 16 = 4*4
    """
    key = (n, device)
    if key not in _PERM_MATRICES_CACHE:
        all_perms = list(itertools_permutations(range(n)))
        K = len(all_perms)  # K = n!
        # Create flattened permutation matrices: (n!, n*n)
        P = torch.zeros(K, n * n, device=device, dtype=torch.float32)
        for k, perm in enumerate(all_perms):
            for i, j in enumerate(perm):
                P[k, i * n + j] = 1.0
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
        >>> nC = n * C
        >>> x = torch.randn(M, nC, dtype=torch.bfloat16, device='cuda')
        >>> phi_pre = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> phi_post = torch.randn(nC, n, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> # Sinkhorn mode (default)
        >>> phi_res = torch.randn(nC, n*n, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda')
        >>> H_pre, H_post, H_res = fused_mhc(x, phi_pre, phi_post, phi_res, 
        ...                                   1.0, 1.5, 0.8, bias, n)
        >>> 
        >>> # Lite mode (exact doubly stochastic)
        >>> from math import factorial
        >>> phi_res_lite = torch.randn(nC, factorial(n), dtype=torch.bfloat16, device='cuda')
        >>> bias_lite = torch.randn(factorial(n) + 2*n, dtype=torch.float32, device='cuda')
        >>> H_pre, H_post, H_res = fused_mhc(x, phi_pre, phi_post, phi_res_lite,
        ...                                   1.0, 1.5, 0.8, bias_lite, n, hres_mode="lite")
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
        config_name = "MHC_FUSED_LITE" if hres_lite_mode else "MHC_FUSED"
        config, _ = get_mhc_config(config_name, M, C)
    else:
        config = dict(config)  # Copy to avoid mutation
    
    num_ksplit = config.get("NUM_KSPLIT", 1)

    # Pop block sizes from config, or compute defaults
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    # BLOCK_N: For lite mode, must fit n_factorial for softmax; for sinkhorn, n_squared
    if hres_lite_mode:
        BLOCK_N = triton.next_power_of_2(max(n_factorial, n_squared))
    else:
        BLOCK_N = n_squared
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

    if hres_lite_mode:
        # mHC-lite path: use lite kernels
        # Pre/post streams use standard block calculation
        # Res stream must be 1 block to fit n_factorial for softmax computation
        n_blocks_pre = triton.cdiv(n, BLOCK_N)
        n_blocks_post = triton.cdiv(n, BLOCK_N)
        n_blocks_res = 1  # Required: res stream needs full n_factorial for softmax
        total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res
        
        if num_ksplit > 1:
            # Split-K path for lite mode
            splitk_block_size = triton.cdiv(K, num_ksplit)
            actual_ksplit = triton.cdiv(K, splitk_block_size)
            
            # Intermediate buffers: res uses n_factorial columns
            acc_pre_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
            acc_post_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
            acc_res_partial = torch.empty((num_ksplit, M, n_factorial), dtype=torch.float32, device=x.device)
            acc_sq_partial = torch.empty((num_ksplit, M), dtype=torch.float32, device=x.device)
            
            grid_split = (triton.cdiv(M, BLOCK_M), total_n_blocks, num_ksplit)
            _mhc_fused_split_lite_kernel[grid_split](
                x, phi_pre, phi_post, phi_res,
                acc_pre_partial, acc_post_partial, acc_res_partial, acc_sq_partial,
                M=M, K=K, N=N, n=n, n_squared=n_squared, n_factorial=n_factorial,
                stride_xm=x.stride(0), stride_xk=x.stride(1),
                stride_phi_pre_k=phi_pre.stride(0), stride_phi_pre_n=phi_pre.stride(1),
                stride_phi_post_k=phi_post.stride(0), stride_phi_post_n=phi_post.stride(1),
                stride_phi_res_k=phi_res.stride(0), stride_phi_res_n=phi_res.stride(1),
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
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                SPLITK_BLOCK_SIZE=splitk_block_size,
                **config,
            )
            
            grid_reduce = (triton.cdiv(M, BLOCK_M), total_n_blocks)
            _mhc_fused_reduce_lite_kernel[grid_reduce](
                acc_pre_partial, acc_post_partial, acc_res_partial, acc_sq_partial,
                alpha_pre, alpha_post, alpha_res, bias,
                out_pre, out_post, out_res, perm_matrices,
                M=M, K=K, N=N, n=n, n_squared=n_squared, n_factorial=n_factorial, eps=eps,
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
                stride_pre_m=out_pre.stride(0), stride_pre_n=out_pre.stride(1),
                stride_post_m=out_post.stride(0), stride_post_n=out_post.stride(1),
                stride_res_m=out_res.stride(0), stride_res_n=out_res.stride(1),
                stride_perm_k=perm_matrices.stride(0), stride_perm_ij=perm_matrices.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ACTUAL_KSPLIT=actual_ksplit,
                **config,
            )
        else:
            # Single fused kernel for lite mode
            grid = (triton.cdiv(M, BLOCK_M), total_n_blocks)
            _mhc_fused_lite_kernel[grid](
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res, bias,
                out_pre, out_post, out_res, perm_matrices,
                M=M, K=K, N=N, n=n, n_squared=n_squared, n_factorial=n_factorial, eps=eps,
                stride_xm=x.stride(0), stride_xk=x.stride(1),
                stride_phi_pre_k=phi_pre.stride(0), stride_phi_pre_n=phi_pre.stride(1),
                stride_phi_post_k=phi_post.stride(0), stride_phi_post_n=phi_post.stride(1),
                stride_phi_res_k=phi_res.stride(0), stride_phi_res_n=phi_res.stride(1),
                stride_pre_m=out_pre.stride(0), stride_pre_n=out_pre.stride(1),
                stride_post_m=out_post.stride(0), stride_post_n=out_post.stride(1),
                stride_res_m=out_res.stride(0), stride_res_n=out_res.stride(1),
                stride_perm_k=perm_matrices.stride(0), stride_perm_ij=perm_matrices.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                **config,
            )
    else:
        # Standard mHC path: use original kernels
        n_blocks_pre = triton.cdiv(n, BLOCK_N)
        n_blocks_post = triton.cdiv(n, BLOCK_N)
        n_blocks_res = triton.cdiv(n_squared, BLOCK_N)
        total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res

        if num_ksplit > 1:
            # Split-K path: use split and reduce kernels
            splitk_block_size = triton.cdiv(K, num_ksplit)
            actual_ksplit = triton.cdiv(K, splitk_block_size)
            
            # Allocate intermediate buffers (float32 for precision)
            acc_pre_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
            acc_post_partial = torch.empty((num_ksplit, M, n), dtype=torch.float32, device=x.device)
            acc_res_partial = torch.empty((num_ksplit, M, n_squared), dtype=torch.float32, device=x.device)
            acc_sq_partial = torch.empty((num_ksplit, M), dtype=torch.float32, device=x.device)
            
            # Launch split kernel with 3D grid: (M_blocks, N_blocks_total, NUM_KSPLIT)
            grid_split = (triton.cdiv(M, BLOCK_M), total_n_blocks, num_ksplit)
            _mhc_fused_split_kernel[grid_split](
                x, phi_pre, phi_post, phi_res,
                acc_pre_partial, acc_post_partial, acc_res_partial, acc_sq_partial,
                M=M, K=K, N=N, n=n,
                stride_xm=x.stride(0), stride_xk=x.stride(1),
                stride_phi_pre_k=phi_pre.stride(0), stride_phi_pre_n=phi_pre.stride(1),
                stride_phi_post_k=phi_post.stride(0), stride_phi_post_n=phi_post.stride(1),
                stride_phi_res_k=phi_res.stride(0), stride_phi_res_n=phi_res.stride(1),
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
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                SPLITK_BLOCK_SIZE=splitk_block_size,
                **config,
            )
            
            # Launch reduce kernel with 2D grid: (M_blocks, N_blocks_total)
            grid_reduce = (triton.cdiv(M, BLOCK_M), total_n_blocks)
            _mhc_fused_reduce_kernel[grid_reduce](
                acc_pre_partial, acc_post_partial, acc_res_partial, acc_sq_partial,
                alpha_pre, alpha_post, alpha_res, bias,
                out_pre, out_post, out_res,
                M=M, K=K, N=N, n=n, eps=eps,
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
                stride_pre_m=out_pre.stride(0), stride_pre_n=out_pre.stride(1),
                stride_post_m=out_post.stride(0), stride_post_n=out_post.stride(1),
                stride_res_m=out_res.stride(0), stride_res_n=out_res.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ACTUAL_KSPLIT=actual_ksplit,
                **config,
            )
        else:
            # Launch 2D grid: (row blocks, stream-aware column blocks)
            grid = (triton.cdiv(M, BLOCK_M), total_n_blocks)
            _mhc_fused_kernel[grid](
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res, bias,
                out_pre, out_post, out_res,
                M=M, K=K, N=N, n=n, eps=eps,
                stride_xm=x.stride(0), stride_xk=x.stride(1),
                stride_phi_pre_k=phi_pre.stride(0), stride_phi_pre_n=phi_pre.stride(1),
                stride_phi_post_k=phi_post.stride(0), stride_phi_post_n=phi_post.stride(1),
                stride_phi_res_k=phi_res.stride(0), stride_phi_res_n=phi_res.stride(1),
                stride_pre_m=out_pre.stride(0), stride_pre_n=out_pre.stride(1),
                stride_post_m=out_post.stride(0), stride_post_n=out_post.stride(1),
                stride_res_m=out_res.stride(0), stride_res_n=out_res.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mHC projection mapping with all three streams (equations 14-19 from the paper).

    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation for pre-stream
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - H^res computation depends on hres_mode:
        - "sinkhorn": Eq 19 via iterative Sinkhorn-Knopp (approximate doubly stochastic)
        - "lite": Birkhoff-von Neumann reparameterization (exact doubly stochastic)

    All operations are fused in optimized Triton kernels for maximum performance.

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
            - (n! + 2n,) for lite mode
        n: Stream parameter - hyperparameter controlling manifold dimension.
        eps: Epsilon for RMSNorm numerical stability (default: 1e-6)
        hres_mode: Mode for H_res computation:
            - "sinkhorn": Iterative Sinkhorn-Knopp (default)
            - "lite": Exact doubly stochastic via permutation combination
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (only used when hres_mode="sinkhorn")
        out_pre (Optional[torch.Tensor]): Pre-allocated output for H^pre with shape (M, n)
        out_post (Optional[torch.Tensor]): Pre-allocated output for H^post with shape (M, n)
        out_res (Optional[torch.Tensor]): Pre-allocated output for H^res with shape (M, n²)

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
        >>> # Standard mHC with Sinkhorn-Knopp
        >>> phi_res = torch.randn(nC, n*n, dtype=torch.bfloat16, device='cuda')
        >>> bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda')
        >>> H_pre, H_post, H_res = mhc(x, phi_pre, phi_post, phi_res,
        ...                            1.0, 1.5, 0.8, bias, n)
        >>> 
        >>> # mHC-lite (exact doubly stochastic, no Sinkhorn iterations)
        >>> from math import factorial
        >>> phi_res_lite = torch.randn(nC, factorial(n), dtype=torch.bfloat16, device='cuda')
        >>> bias_lite = torch.randn(factorial(n) + 2*n, dtype=torch.float32, device='cuda')
        >>> H_pre, H_post, H_res = mhc(x, phi_pre, phi_post, phi_res_lite,
        ...                            1.0, 1.5, 0.8, bias_lite, n, hres_mode="lite")
    """
    hres_lite_mode = hres_mode == "lite"
    
    if hres_lite_mode:
        _LOGGER.info(
            f"MHC (lite): calling fused_mhc() with hres_mode='lite' (exact doubly stochastic)"
        )
    else:
        _LOGGER.info(
            f"MHC: calling fused_mhc() then sinkhorn_knopp() with {sinkhorn_iters} iterations"
        )
    
    # Call fused_mhc function (Eq 14-18, or Eq 14-18 + lite activation)
    out_pre, out_post, out_res = fused_mhc(
        x, phi_pre, phi_post, phi_res,
        alpha_pre, alpha_post, alpha_res,
        bias, n, eps, hres_mode=hres_mode,
        out_pre=out_pre, out_post=out_post, out_res=out_res
    )
    
    # For lite mode, H_res is already exactly doubly stochastic - skip Sinkhorn-Knopp
    if not hres_lite_mode:
        # Apply Sinkhorn-Knopp (Equation 19) to make H_res doubly stochastic
        # Reshape H_res from (M, n²) to (M, n, n) for Sinkhorn kernel
        M = out_res.shape[0]
        H_res_3d = out_res.view(M, n, n)
        
        # Iterative Sinkhorn-Knopp - modifies H_res_3d in-place
        sinkhorn_knopp(H_res_3d, num_iters=sinkhorn_iters, out=H_res_3d)
    
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
    
    # Filter out BLOCK_SIZE - not used by the log domain kernel
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
