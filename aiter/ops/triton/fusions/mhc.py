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

from aiter.ops.triton._triton_kernels.fusions.mhc import (
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_fused_reduce_kernel,
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

    Returns:
        P: (n!, n, n) tensor of permutation matrices, where P[k] is the k-th
           permutation matrix with exactly one 1.0 per row and column.
    """
    key = (n, device)
    if key not in _PERM_MATRICES_CACHE:
        all_perms = list(itertools_permutations(range(n)))
        K = len(all_perms)
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

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation
        - H_post: (M, n) - post-processing with scaled sigmoid
        - H_res: (M, n²) - residual stream:
            - sinkhorn mode: raw logits (NOT doubly stochastic)
            - lite mode: exact doubly stochastic matrix
    """

    assert hres_mode in (
        "sinkhorn",
        "lite",
    ), f"hres_mode must be 'sinkhorn' or 'lite', got {hres_mode}"
    hres_lite_mode = hres_mode == "lite"

    M, K = x.shape
    C = K // n  # Derive C from K and n
    K_pre, n_pre = phi_pre.shape
    K_post, n_post = phi_post.shape
    K_res, phi_res_cols = phi_res.shape

    n_squared = n * n
    n_factorial = factorial(n)

    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C, mode=hres_mode)
    config = dict(config)  # Always copy to avoid mutating LRU cache

    num_ksplit = config.get("NUM_KSPLIT", 1)
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    if hres_lite_mode:
        min_block_n = max(n_factorial, n_squared)
        config_block_n = config.pop("BLOCK_N", min_block_n)
        BLOCK_N = triton.next_power_of_2(max(config_block_n, min_block_n))

        n_res_expected = n_factorial
        assert (
            phi_res_cols == n_res_expected
        ), f"In lite mode, phi_res must have {n_res_expected} columns (n!={n_res_expected}), got {phi_res_cols}"

        perm_matrices = get_permutation_matrices(n, x.device)
        n_blocks_res = 1
        stride_perm_k = perm_matrices.stride(0)
        stride_perm_ij = 1
    else:
        BLOCK_N = triton.next_power_of_2(config.pop("BLOCK_N", n_squared))

        n_res_expected = n_squared
        assert (
            phi_res_cols == n_squared
        ), f"In sinkhorn mode, phi_res must have {n_squared} columns (n²), got {phi_res_cols}"

        perm_matrices = torch.empty(0, device=x.device)
        n_blocks_res = triton.cdiv(n_squared, BLOCK_N)
        stride_perm_k = 0
        stride_perm_ij = 0

    N_total_expected = n_res_expected + 2 * n  # n (pre) + n (post) + n_res

    BLOCK_K = config.pop("BLOCK_K", 64)
    # Ensure BLOCK_K doesn't exceed K dimension
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))

    _LOGGER.info(
        f"FUSED_MHC: x={tuple(x.shape)} phi_pre={tuple(phi_pre.shape)} phi_post={tuple(phi_post.shape)} "
        f"phi_res={tuple(phi_res.shape)} alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res} "
        f"hres_mode={hres_mode} num_ksplit={num_ksplit}"
    )

    assert (
        K == K_pre == K_post == K_res
    ), f"Dimension mismatch: x has K={K}, but phi_pre={K_pre}, phi_post={K_post}, phi_res={K_res}"
    assert (
        n_pre == n
    ), f"phi_pre shape mismatch: expected (K, {n}), got ({K_pre}, {n_pre})"
    assert (
        n_post == n
    ), f"phi_post shape mismatch: expected (K, {n}), got ({K_post}, {n_post})"

    assert (
        bias.shape[0] == N_total_expected
    ), f"Bias shape mismatch: expected ({N_total_expected},), got {bias.shape}"
    assert num_ksplit >= 1, f"num_ksplit must be >= 1, got {num_ksplit}"

    assert (
        x.device == phi_pre.device == phi_post.device == phi_res.device == bias.device
    ), "All tensors must be on the same device"
    assert x.device.type == "cuda", "mHC kernel requires CUDA device"

    N = N_total_expected

    # Allocate outputs if not provided
    # H_res is always (M, n²) regardless of mode (lite outputs n² after perm combination)
    if out_pre is None:
        out_pre = torch.empty(M, n, dtype=x.dtype, device=x.device)
    else:
        assert out_pre.shape == (
            M,
            n,
        ), f"out_pre shape mismatch: expected ({M}, {n}), got {out_pre.shape}"
        assert out_pre.dtype == x.dtype and out_pre.device == x.device

    if out_post is None:
        out_post = torch.empty(M, n, dtype=x.dtype, device=x.device)
    else:
        assert out_post.shape == (
            M,
            n,
        ), f"out_post shape mismatch: expected ({M}, {n}), got {out_post.shape}"
        assert out_post.dtype == x.dtype and out_post.device == x.device

    if out_res is None:
        out_res = torch.empty(M, n_squared, dtype=x.dtype, device=x.device)
    else:
        assert out_res.shape == (
            M,
            n_squared,
        ), f"out_res shape mismatch: expected ({M}, {n_squared}), got {out_res.shape}"
        assert out_res.dtype == x.dtype and out_res.device == x.device

    # Stream-aware grid: Each program processes exactly one stream
    n_blocks_pre = triton.cdiv(n, BLOCK_N)
    n_blocks_post = triton.cdiv(n, BLOCK_N)
    total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res

    if num_ksplit > 1:
        # Split-K path: use split and reduce kernels
        splitk_block_size = triton.cdiv(K, num_ksplit)
        actual_ksplit = triton.cdiv(K, splitk_block_size)

        # Allocate intermediate buffers (float32 for precision)
        # acc_res size depends on mode: n² for sinkhorn, n! for lite
        acc_res_cols = n_res_expected
        acc_pre_partial = torch.empty(
            (num_ksplit, M, n), dtype=torch.float32, device=x.device
        )
        acc_post_partial = torch.empty(
            (num_ksplit, M, n), dtype=torch.float32, device=x.device
        )
        acc_res_partial = torch.empty(
            (num_ksplit, M, acc_res_cols), dtype=torch.float32, device=x.device
        )
        acc_sq_partial = torch.empty(
            (num_ksplit, M), dtype=torch.float32, device=x.device
        )

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
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            n_factorial=n_res_expected,
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_phi_pre_k=phi_pre.stride(0),
            stride_phi_pre_n=phi_pre.stride(1),
            stride_phi_post_k=phi_post.stride(0),
            stride_phi_post_n=phi_post.stride(1),
            stride_phi_res_k=phi_res.stride(0),
            stride_phi_res_n=phi_res.stride(1),
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
            perm_matrices,
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            n_factorial=n_res_expected,
            eps=eps,
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
            stride_pre_m=out_pre.stride(0),
            stride_pre_n=out_pre.stride(1),
            stride_post_m=out_post.stride(0),
            stride_post_n=out_post.stride(1),
            stride_res_m=out_res.stride(0),
            stride_res_n=out_res.stride(1),
            stride_perm_k=stride_perm_k,
            stride_perm_ij=stride_perm_ij,
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
            x,
            phi_pre,
            phi_post,
            phi_res,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            out_pre,
            out_post,
            out_res,
            perm_matrices,
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            n_factorial=n_res_expected,
            eps=eps,
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
            stride_perm_k=stride_perm_k,
            stride_perm_ij=stride_perm_ij,
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

    All operations are fused in optimized Triton kernels for maximum performance.

    Returns:
        Tuple of three tensors (H_pre, H_post, H_res):
        - H_pre: (M, n) - manifold projection with sigmoid activation
        - H_post: (M, n) - post-processing with scaled sigmoid
        - H_res: (M, n²) - doubly stochastic residual connection
    """
    _LOGGER.info(
        f"MHC: calling fused_mhc() with hres_mode='lite'"
        if hres_mode == "lite"
        else f"MHC: calling fused_mhc() then sinkhorn_knopp() with {sinkhorn_iters} iterations"
    )
    res = fused_mhc(
        x,
        phi_pre,
        phi_post,
        phi_res,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps,
        hres_mode=hres_mode,
        out_pre=out_pre,
        out_post=out_post,
        out_res=out_res,
        config=config,
    )

    if hres_mode == "lite":
        return res
    else:
        out_pre, out_post, out_res = res

        M = out_res.shape[0]
        C = x.shape[1] // n
        out_res_3d = out_res.view(M, n, n)
        sinkhorn_knopp(out_res_3d, C=C, num_iters=sinkhorn_iters, out=out_res_3d)

        return out_pre, out_post, out_res


def sinkhorn_knopp(
    logits: torch.Tensor,
    C: int,
    num_iters: int = 20,
    out: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Projects batched raw logits onto doubly stochastic matrices using log-domain Sinkhorn-Knopp.

    Returns:
        torch.Tensor: Doubly stochastic matrices with shape (M, N, N).
            Each matrix in the batch has rows and columns summing to 1.
    """
    _LOGGER.info(f"Sinkhorn-Knopp: logits={tuple(logits.shape)} num_iters={num_iters}")

    assert logits.dim() == 3, f"logits must be 3D (M, N, N), got {logits.dim()}D"

    M, N, N2 = logits.shape
    assert N == N2, f"Last two dimensions must be equal, got ({N}, {N2})"
    # Cap N at 64 to avoid overflow in log domain
    assert N <= 64, f"Matrix size N={N} exceeds maximum of 64"

    # Check N is power of 2 because Triton arange requires even number of sizes
    N_pow2 = triton.next_power_of_2(N)
    assert N == N_pow2, f"Matrix size N={N} must be a power of 2"

    assert num_iters > 0, f"num_iters must be positive, got {num_iters}"

    logits = logits.contiguous()

    # Allocate output if not provided
    if out is None:
        out = torch.empty((M, N, N), dtype=logits.dtype, device=logits.device)
    else:
        assert out.shape == (M, N, N), f"out.shape {out.shape} must be ({M}, {N}, {N})"
        out = out.contiguous()

    if config is None:
        config, _ = get_mhc_config("MHC_SINKHORN", M, C)
    config = dict(config)  # Always copy to avoid mutating LRU cache

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
