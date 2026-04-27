# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
High-level Python wrapper for mHC (manifold-constrained Hyper Connection).

Provides two main functions:
- fused_mhc(): Low-level interface for equations 14-18 plus the layer_input
  apply step (raw H_res, Sinkhorn-Knopp NOT applied).
- mhc(): Complete pipeline implementing equations 14-19 (kernel + Sinkhorn-Knopp).
"""

from typing import Optional, Tuple
import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_fused_reduce_kernel,
    _sinkhorn_knopp_log_domain_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config

_LOGGER = AiterTritonLogger()


def fused_mhc(
    x: torch.Tensor,
    phi: torch.Tensor,  # Unified phi: (K, n + n + n_squared)
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    hc_pre_eps: float = 0.0,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Triton kernel interface for mHC projection mapping (equations 14-18)
    plus the layer_input apply step.

    This function implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
    - Eq 17: H^pre = σ(H^pre) - sigmoid activation
    - Eq 18: H^post = 2σ(H^post) - scaled sigmoid activation for post-stream
    - layer_input[m, c] = Σᵢ (σ(H^pre[m, i]) + hc_pre_eps) · x[m, i, c]

    Args:
        x: (M, K) input tensor where K = n*C
        phi: (K, N) unified projection matrix where N = n + n + n²
             Layout: [pre | post | res] where res is n²
        alpha_pre: Scaling factor for pre-stream
        alpha_post: Scaling factor for post-stream
        alpha_res: Scaling factor for residual stream
        bias: (N,) bias vector (fp32)
        n: Stream parameter (manifold dimension)
        eps: Epsilon for RMS-norm numerical stability
        hc_pre_eps: Additive epsilon on σ(H^pre) before the apply step
            (default 0.0 for Eq-17 parity).
        config: Optional kernel configuration dict

    Returns:
        Tuple `(h_post, h_res, layer_input)`:
        - h_post:         (M, n, 1)  in x.dtype  - 2σ(H^post)
        - h_res:          (M, n, n)  in x.dtype  - raw H^res before Sinkhorn
        - layer_input:      (M, C)     in x.dtype  - Σᵢ pre_mix_i · x_i
    """
    M, K = x.shape
    C = K // n  # Derive C from K and n
    K_phi, total_phi_cols = phi.shape

    n_squared = n * n

    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
    config = dict(config)  # Always copy to avoid mutating LRU cache

    num_ksplit = config.get("NUM_KSPLIT", 1)
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    BLOCK_N = triton.next_power_of_2(config.pop("BLOCK_N", n_squared))

    n_blocks_res = triton.cdiv(n_squared, BLOCK_N)

    N_total = n_squared + 2 * n

    BLOCK_K = config.pop("BLOCK_K", 64)
    # Ensure BLOCK_K doesn't exceed K dimension
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))

    # TODO: have BLOCK_C in json config file and load it here later
    BLOCK_C = config.pop("BLOCK_C", BLOCK_K)
    BLOCK_C = min(BLOCK_C, triton.next_power_of_2(C))

    _LOGGER.info(
        f"FUSED_MHC: x={tuple(x.shape)} phi={tuple(phi.shape)} "
        f"alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res} "
        f"hc_pre_eps={hc_pre_eps} num_ksplit={num_ksplit} "
        f"BLOCK_N={BLOCK_N} BLOCK_C={BLOCK_C}"
    )

    assert K == K_phi, f"Dimension mismatch: x has K={K}, but phi has K={K_phi}"
    assert (
        total_phi_cols == N_total
    ), f"phi shape mismatch: expected (K, {N_total}), got ({K_phi}, {total_phi_cols})"

    assert (
        bias.shape[0] == N_total
    ), f"Bias shape mismatch: expected ({N_total},), got {bias.shape}"
    assert num_ksplit >= 1, f"num_ksplit must be >= 1, got {num_ksplit}"

    assert (
        x.device == phi.device == bias.device
    ), "All tensors must be on the same device"
    assert x.device.type == "cuda", "mHC kernel requires CUDA device"
    # Pre-stream programs assume one program owns all H^pre cols for its row block.
    assert (
        BLOCK_N >= n
    ), f"BLOCK_N ({BLOCK_N}) must be >= n ({n}) for the apply-pre fusion"

    N = N_total

    # Output `out` shrinks to (M, n + n_squared) with layout [post | res] —
    # H^pre is consumed inside the kernel, so its slice is no longer materialized.
    N_out_post_res = n + n_squared
    out = torch.empty(M, N_out_post_res, dtype=x.dtype, device=x.device)
    layer_input = torch.empty(M, C, dtype=x.dtype, device=x.device)

    # Stream-aware grid: Each program processes exactly one stream
    n_blocks_pre = triton.cdiv(n, BLOCK_N)
    n_blocks_post = triton.cdiv(n, BLOCK_N)
    total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res

    if num_ksplit > 1:
        # Split-K path: use split and reduce kernels.
        splitk_block_size = triton.cdiv(K, num_ksplit)
        actual_ksplit = triton.cdiv(K, splitk_block_size)

        acc_partial = torch.empty(
            (num_ksplit, M, N_total), dtype=torch.float32, device=x.device
        )
        acc_sq_partial = torch.empty(
            (num_ksplit, M), dtype=torch.float32, device=x.device
        )

        grid_split = (triton.cdiv(M, BLOCK_M), total_n_blocks, num_ksplit)
        _mhc_fused_split_kernel[grid_split](
            x,
            phi,
            acc_partial,
            acc_sq_partial,
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_phi_k=phi.stride(0),
            stride_phi_n=phi.stride(1),
            stride_acc_k=acc_partial.stride(0),
            stride_acc_m=acc_partial.stride(1),
            stride_acc_n=acc_partial.stride(2),
            stride_acc_sq_k=acc_sq_partial.stride(0),
            stride_acc_sq_m=acc_sq_partial.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLITK_BLOCK_SIZE=splitk_block_size,
            **config,
        )

        grid_reduce = (triton.cdiv(M, BLOCK_M), total_n_blocks)
        _mhc_fused_reduce_kernel[grid_reduce](
            acc_partial,
            acc_sq_partial,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            out,
            x,
            layer_input,
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            C=C,
            eps=eps,
            hc_pre_eps=hc_pre_eps,
            stride_acc_k=acc_partial.stride(0),
            stride_acc_m=acc_partial.stride(1),
            stride_acc_n=acc_partial.stride(2),
            stride_acc_sq_k=acc_sq_partial.stride(0),
            stride_acc_sq_m=acc_sq_partial.stride(1),
            stride_out_m=out.stride(0),
            stride_out_n=out.stride(1),
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_li_m=layer_input.stride(0),
            stride_li_c=layer_input.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_C=BLOCK_C,
            ACTUAL_KSPLIT=actual_ksplit,
            **config,
        )
    else:
        grid = (triton.cdiv(M, BLOCK_M), total_n_blocks)
        _mhc_fused_kernel[grid](
            x,
            phi,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            out,
            layer_input,
            M=M,
            K=K,
            N=N,
            n=n,
            n_squared=n_squared,
            C=C,
            eps=eps,
            hc_pre_eps=hc_pre_eps,
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_phi_k=phi.stride(0),
            stride_phi_n=phi.stride(1),
            stride_out_m=out.stride(0),
            stride_out_n=out.stride(1),
            stride_li_m=layer_input.stride(0),
            stride_li_c=layer_input.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_C=BLOCK_C,
            **config,
        )

    # `out` layout is [post | res]: out[:, :n] is H^post, out[:, n:] is H^res
    h_post = out[:, :n].unsqueeze(-1)  # (M, n, 1)
    h_res = out[:, n:].view(M, n, n)  # (M, n, n)
    return h_post, h_res, layer_input


def mhc(
    x: torch.Tensor,
    phi: torch.Tensor,  # Unified phi: (K, n + n + n_squared)
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    hc_pre_eps: float = 0.0,
    sinkhorn_iters: int = 20,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mHC projection mapping (equations 14-19).

    Runs `fused_mhc()` then applies log-domain Sinkhorn-Knopp in-place to the
    `h_res` view to project H^res onto the doubly-stochastic manifold (Eq 19).

    Args:
        x: (M, K) input tensor where K = n*C
        phi: (K, N) unified projection matrix where N = n + n + n²
             Layout: [pre | post | res] where res is n²
        alpha_pre: Scaling factor for pre-stream
        alpha_post: Scaling factor for post-stream
        alpha_res: Scaling factor for residual stream
        bias: (N,) bias vector (fp32)
        n: Stream parameter (manifold dimension)
        eps: Epsilon for RMS-norm numerical stability
        hc_pre_eps: Additive epsilon on σ(H^pre) before the apply step
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        config: Optional kernel configuration dict

    Returns:
        Tuple `(h_post, h_res, layer_input)`:
        - h_post:    (M, n, 1)  in x.dtype  - 2σ(H^post)
        - h_res:    (M, n, n)  in x.dtype  - doubly-stochastic Sinkhorn output
        - layer_input: (M, C)     in x.dtype  - Σᵢ σ(H^pre[m, i]) + hc_pre_eps · x_i
    """
    _LOGGER.info(
        f"MHC: calling fused_mhc() then sinkhorn_knopp() with {sinkhorn_iters} iterations"
    )
    h_post, h_res, layer_input = fused_mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps=eps,
        hc_pre_eps=hc_pre_eps,
        config=config,
    )

    C = x.shape[1] // n
    # In-place on the comb_mix view (no copy-back path needed)
    sinkhorn_knopp(h_res, C=C, num_iters=sinkhorn_iters, out=h_res)

    return h_post, h_res, layer_input


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

    # Allocate output if not provided
    if out is None:
        # Kernel uses logits strides; allocate matching layout.
        out = torch.empty_strided(
            size=logits.shape,
            stride=logits.stride(),
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        assert out.shape == (M, N, N), f"out.shape {out.shape} must be ({M}, {N}, {N})"
        assert out.dtype == logits.dtype and out.device == logits.device
        assert out.stride() == logits.stride(), (
            f"out.stride {out.stride()} must match logits.stride {logits.stride()} "
            "for sinkhorn kernel"
        )

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
