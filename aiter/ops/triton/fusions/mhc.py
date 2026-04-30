# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_fused_reduce_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config

_LOGGER = AiterTritonLogger()


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
    hc_post_mult_value: float = 2.0,
    sinkhorn_iters: int = 20,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single entry point ``mhc()`` runs equations 14-19 plus the layer_input apply
    step in a single Triton launch (or split-K + reduce launch pair). The
    log-domain Sinkhorn-Knopp projection of H^res (Eq 19) is fused into the
    res-stream branch of the kernel.

        This function implements:
        - Eq 14: H̃ = x̃φ (matrix multiplication)
        - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
        - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b
        - Eq 17: H^pre = σ(H^pre) + hc_pre_eps  (folded into layer_input below)
        - Eq 18: H^post = hc_post_mult_value · σ(H^post)
        - Eq 19: H^res = SinkhornKnopp(H^res)  (log-domain, in-kernel; skipped when
          sinkhorn_iters == 0, in which case raw H^res logits are returned)
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
            hc_post_mult_value: Multiplier on σ(H^post) (default 2.0 for Eq-18 parity).
            sinkhorn_iters: Number of in-kernel log-domain Sinkhorn-Knopp iterations
                on H^res. ``0`` returns raw H^res logits (skips Eq 19).
            config: Optional kernel configuration dict

        Returns:
            Tuple `(h_post, h_res, layer_input)`:
            - h_post:      (M, n, 1)  in x.dtype  - hc_post_mult_value · σ(H^post)
            - h_res:       (M, n, n)  in x.dtype  - doubly-stochastic Sinkhorn output
                                                     (or raw logits when sinkhorn_iters==0)
            - layer_input: (M, C)     in x.dtype  - Σᵢ pre_mix_i · x_i
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
        f"MHC: x={tuple(x.shape)} phi={tuple(phi.shape)} "
        f"alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res} "
        f"hc_pre_eps={hc_pre_eps} hc_post_mult_value={hc_post_mult_value} "
        f"sinkhorn_iters={sinkhorn_iters} num_ksplit={num_ksplit} "
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
    assert sinkhorn_iters >= 0, f"sinkhorn_iters must be >= 0, got {sinkhorn_iters}"

    assert (
        x.device == phi.device == bias.device
    ), "All tensors must be on the same device"
    assert x.device.type == "cuda", "mHC kernel requires CUDA device"
    # Pre-stream programs assume one program owns all H^pre cols for its row block.
    assert (
        BLOCK_N >= n
    ), f"BLOCK_N ({BLOCK_N}) must be >= n ({n}) for the apply-pre fusion"
    # In-kernel SK requires a single program to own the full (n, n) res tile and
    # uses tl.reshape((BLOCK_M, BLOCK_N) -> (BLOCK_M, n, n)), which needs the
    # tile to be exactly n_squared columns wide.
    if sinkhorn_iters > 0:
        assert BLOCK_N == n_squared, (
            f"sinkhorn_iters>0 requires BLOCK_N ({BLOCK_N}) == n_squared "
            f"({n_squared}); for non-power-of-2 n, run with sinkhorn_iters=0."
        )

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
            hc_post_mult_value=hc_post_mult_value,
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
            NUM_SINKHORN_ITERS=sinkhorn_iters,
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
            hc_post_mult_value=hc_post_mult_value,
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
            NUM_SINKHORN_ITERS=sinkhorn_iters,
            **config,
        )

    # `out` layout is [post | res]: out[:, :n] is H^post, out[:, n:] is H^res
    h_post = out[:, :n].unsqueeze(-1)  # (M, n, 1)
    h_res = out[:, n:].view(M, n, n)  # (M, n, n)
    return h_post, h_res, layer_input


def mhc_post(
    out: Optional[torch.Tensor],
    x: torch.Tensor,  # (m, hidden_size)  bf16/fp16
    residual: torch.Tensor,  # (m, hc_mult, hidden_size)  bf16/fp16
    post_layer_mix: torch.Tensor,  # (m, hc_mult) or (m, hc_mult, 1)  fp32
    comb_res_mix: torch.Tensor,  # (m, hc_mult, hc_mult)  fp32 [dim1=src, dim2=dst]
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    mhc_post: Apply post-stream and res-stream mixing to produce updated multi-stream residual.

    Implements: out[m, hc, k] = post_mix[m, hc] * x[m, k] + sum_h( comb_mix[m, h, hc] * residual[m, h, k] )

    Args:
        out: Optional pre-allocated output tensor (m, hc_mult, hidden_size)
        x: Transformer layer output (m, hidden_size)
        residual: Current multi-stream residual (m, hc_mult, hidden_size)
        post_layer_mix: Post-stream coefficients (m, hc_mult) or (m, hc_mult, 1)
        comb_res_mix: Res-stream mixing coefficients (m, hc_mult, hc_mult) [dim1=src, dim2=dst]
        config: Optional kernel configuration dict

    Returns:
        Updated multi-stream residual (m, hc_mult, hidden_size)
    """
    from aiter.ops.triton._triton_kernels.fusions import _mhc_post_kernel
    from aiter.ops.triton.utils.mhc_config_utils import get_mhc_post_config

    # Step 1: Extract dimensions
    m, hc_mult, hidden_size = residual.shape
    assert hc_mult == 4, f"hc_mult must be 4, got {hc_mult}"
    assert x.shape == (
        m,
        hidden_size,
    ), f"x shape mismatch: expected ({m}, {hidden_size}), got {x.shape}"

    # Step 2: Handle post_layer_mix shape - squeeze if (m, hc_mult, 1)
    if post_layer_mix.ndim == 3:
        post_layer_mix = post_layer_mix.squeeze(-1)
    assert post_layer_mix.shape == (
        m,
        hc_mult,
    ), f"post_layer_mix shape mismatch: expected ({m}, {hc_mult}), got {post_layer_mix.shape}"
    assert comb_res_mix.shape == (
        m,
        hc_mult,
        hc_mult,
    ), f"comb_res_mix shape mismatch: expected ({m}, {hc_mult}, {hc_mult}), got {comb_res_mix.shape}"

    # Step 3: Resolve BLOCK_K (matches HIP dispatch logic)
    if config is None:
        config = get_mhc_post_config(hidden_size)
    config = dict(config)  # Copy to avoid mutating cache
    BLOCK_K = config.pop("BLOCK_K", 512)

    _LOGGER.info(
        f"MHC_POST: x={tuple(x.shape)} residual={tuple(residual.shape)} "
        f"post_layer_mix={tuple(post_layer_mix.shape)} "
        f"comb_res_mix={tuple(comb_res_mix.shape)} BLOCK_K={BLOCK_K}"
    )

    # Step 4: Allocate output if needed
    if out is None:
        out = torch.empty(m, hc_mult, hidden_size, dtype=x.dtype, device=x.device)

    # Step 5: Launch kernel with grid (m, hc_mult)
    grid = (m, hc_mult)
    _mhc_post_kernel[grid](
        out,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        M=m,
        hidden_size=hidden_size,
        stride_x_m=x.stride(0),
        stride_res_m=residual.stride(0),
        stride_res_hc=residual.stride(1),
        stride_out_m=out.stride(0),
        stride_out_hc=out.stride(1),
        stride_post_m=post_layer_mix.stride(0),
        stride_comb_m=comb_res_mix.stride(0),
        stride_comb_src=comb_res_mix.stride(1),
        HC=hc_mult,
        BLOCK_K=BLOCK_K,
    )

    # Step 6: Return output
    return out
