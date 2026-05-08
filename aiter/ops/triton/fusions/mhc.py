# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_post_kernel,
    _mhc_reduce_apply_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mhc_config_utils import (
    get_mhc_config,
    get_mhc_post_config,
)

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
    """Fused mHC layer in one Triton launch (or a split-K + reduce-apply
    launch pair when ``NUM_KSPLIT > 1``).

    Pipeline (per M-tile, ``alphas = (alpha_pre, alpha_post, alpha_res)``):

        r = ||x||_2 / sqrt(n*C)                              # RMS-norm
        H = ((x @ phi) * alphas) / r + bias                  # 3-stream projection

                       [ H_pre ]            [ H_post ]            [ H_res ]
                          (n)                   (n)                 (n*n)
                           |                    |                     |
                           v                    v                     v
                  sigmoid + hc_pre_eps   hc_post * sigmoid        log-domain
                                                                  Sinkhorn-Knopp
                                                                  (sk_iters=0: raw)
                           |                    |                     |
                           v                    |                     |
                  layer_input[m, c]             |                     |
                    = sum_i pre_mix_i           |                     |
                        * x[m, i, c]            |                     |
                           |                    |                     |
                           v                    v                     v
                      layer_input             h_post                h_res
                        (M, C)              (M, n, 1)             (M, n, n)

    All outputs in ``x.dtype``. ``x`` and ``phi`` are bf16 / fp16; ``bias`` is fp32.

    Execution paths (chosen by ``NUM_KSPLIT``):
      - ``== 1``: ``_mhc_fused_kernel`` runs the full pipeline inline.
      - ``>  1``: ``_mhc_fused_split_kernel`` writes per-K projection partials,
                  then ``_mhc_reduce_apply_kernel`` does reduce + activations
                  + apply.

    Args:
        x:                  (M, n*C) bf16 / fp16 input.
        phi:                (K, N=n+n+n*n) bf16 / fp16; cols ``[pre|post|res]``.
        alpha_pre/post/res: per-stream scaling factors.
        bias:               (N,) fp32.
        n:                  stream / manifold dimension.
        eps:                RMS-norm epsilon (default 1e-6).
        hc_pre_eps:         added to ``sigmoid(H_pre)`` (default 0.0).
        hc_post_mult_value: multiplier on ``sigmoid(H_post)`` (default 2.0).
        sinkhorn_iters:     log-domain SK iterations on H_res; ``0`` skips SK
                            (default 20).
        config:             optional config dict; loaded from per-arch tuned
                            configs when ``None``.

    Returns ``(h_post, h_res, layer_input)``:
        h_post      : (M, n, 1)  hc_post_mult_value * sigmoid(H_post)
        h_res       : (M, n, n)  Sinkhorn output (raw logits if sk_iters == 0)
        layer_input : (M, C)     sum_i (sigmoid(H_pre[m,i]) + hc_pre_eps) * x[m,i,c]

    Reference: arXiv:2512.24880.
    """
    M, K = x.shape
    C = K // n  # Derive C from K and n
    K_phi, total_phi_cols = phi.shape

    n_squared = n * n
    N_POW2 = triton.next_power_of_2(n)
    N_POW2_RES = triton.next_power_of_2(n_squared)
    N_TOTAL_POW2 = triton.next_power_of_2(n_squared + 2 * n)

    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
    config = dict(config)  # Always copy to avoid mutating LRU cache

    num_ksplit = config.pop("NUM_KSPLIT", 1)
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    BLOCK_N = triton.next_power_of_2(config.pop("BLOCK_N", n_squared))

    n_blocks_res = triton.cdiv(n_squared, BLOCK_N)

    N_total = n_squared + 2 * n

    BLOCK_K = config.pop("BLOCK_K", 64)
    # Ensure BLOCK_K doesn't exceed K dimension
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))
    BLOCK_C = config.pop("BLOCK_C", min(64, triton.next_power_of_2(C)))

    # Pin h_post to pid_c == 0 and h_res (with the sinkhorn loop) to pid_c == 1
    # in `_mhc_reduce_apply_kernel`. When only one C-tile per M-tile exists,
    # fall back to pid_c == 0 doing both. Resolved at compile time via constexpr.
    NUM_C_BLOCKS = triton.cdiv(C, BLOCK_C)
    RES_PID_C = 0 if NUM_C_BLOCKS == 1 else 1

    _LOGGER.info(
        f"MHC: x={tuple(x.shape)} phi={tuple(phi.shape)} "
        f"alpha_pre={alpha_pre} alpha_post={alpha_post} alpha_res={alpha_res} "
        f"hc_pre_eps={hc_pre_eps} hc_post_mult_value={hc_post_mult_value} "
        f"sinkhorn_iters={sinkhorn_iters} num_ksplit={num_ksplit} "
        f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} BLOCK_C={BLOCK_C} "
        f"N_TOTAL_POW2={N_TOTAL_POW2} RES_PID_C={RES_PID_C}"
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
    # reshapes its res sub-tile to (BLOCK_M, n, n); both kernels need
    # n_squared to be a power of 2 (i.e. N_POW2_RES == n_squared) for the reshape
    # to compile, and the non-split-K kernel additionally needs BLOCK_N == n_squared.
    if sinkhorn_iters > 0:
        assert BLOCK_N == n_squared, (
            f"sinkhorn_iters>0 requires BLOCK_N ({BLOCK_N}) == n_squared "
            f"({n_squared}); for non-power-of-2 n, run with sinkhorn_iters=0."
        )
        assert N_POW2_RES == n_squared, (
            f"sinkhorn_iters>0 requires n_squared ({n_squared}) to be a power of 2; "
            f"got N_POW2_RES={N_POW2_RES}. For non-power-of-2 n, run with "
            f"sinkhorn_iters=0."
        )

    N = N_total

    N_out_post_res = n + n_squared
    out = torch.empty(M, N_out_post_res, dtype=x.dtype, device=x.device)
    layer_input = torch.empty(M, C, dtype=x.dtype, device=x.device)

    # Stream-aware grid for the non-split-K fused kernel: one program per stream per M-tile
    n_blocks_pre = triton.cdiv(n, BLOCK_N)
    n_blocks_post = triton.cdiv(n, BLOCK_N)
    total_n_blocks = n_blocks_pre + n_blocks_post + n_blocks_res

    if num_ksplit > 1:
        # Split-K path: split GEMM kernel, then a single (M, C)-parallel reduce-apply
        # kernel that fuses RMS reduce, all 3 stream activations, and the apply step.
        splitk_block_size = triton.cdiv(K, num_ksplit)
        actual_ksplit = triton.cdiv(K, splitk_block_size)

        acc_partial = torch.empty(
            (num_ksplit, M, N_total), dtype=torch.float32, device=x.device
        )
        acc_sq_partial = torch.empty(
            (num_ksplit, M), dtype=torch.float32, device=x.device
        )

        grid_split = (triton.cdiv(M, BLOCK_M), num_ksplit)
        _mhc_fused_split_kernel[grid_split](
            x,
            phi,
            acc_partial,
            acc_sq_partial,
            M=M,
            K=K,
            N=N_total,
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
            N_TOTAL_POW2=N_TOTAL_POW2,
            BLOCK_K=BLOCK_K,
            SPLITK_BLOCK_SIZE=splitk_block_size,
            **config,
        )

        grid_reduce_apply = (triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_C))
        _mhc_reduce_apply_kernel[grid_reduce_apply](
            acc_partial,
            acc_sq_partial,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            x,
            out,
            layer_input,
            M=M,
            K=K,
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
            stride_xm=x.stride(0),
            stride_xk=x.stride(1),
            stride_out_m=out.stride(0),
            stride_out_n=out.stride(1),
            stride_li_m=layer_input.stride(0),
            stride_li_c=layer_input.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_C=BLOCK_C,
            N_POW2=N_POW2,
            N_POW2_RES=N_POW2_RES,
            ACTUAL_KSPLIT=actual_ksplit,
            NUM_SINKHORN_ITERS=sinkhorn_iters,
            RES_PID_C=RES_PID_C,
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
            N_POW2=N_POW2,
            NUM_SINKHORN_ITERS=sinkhorn_iters,
            **config,
        )

    # `out` layout is [post + res]: out[:, :n] is H^post, out[:, n:] is H^res
    h_post = out[:, :n].unsqueeze(-1)  # (M, n, 1)
    h_res = out[:, n:].view(M, n, n)  # (M, n, n)
    return h_post, h_res, layer_input


def mhc_post(
    out: Optional[torch.Tensor],
    layer_input: torch.Tensor,  # (M, C)  bf16 / fp16
    residual: torch.Tensor,  # (M, n, C)  bf16 / fp16
    post_mix: torch.Tensor,  # (M, n) or (M, n, 1)  fp32
    comb_mix: torch.Tensor,  # (M, n, n)  fp32 [src, dst]
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Fused mHC post step in one Triton launch.

    Computes the updated multi-stream residual by mixing the transformer
    output ``layer_input`` with the previous-layer residual streams:

        out[m, j, c] = post_mix[m, j] * layer_input[m, c]
                     + sum_h comb_mix[m, h, j] * residual[m, h, c]

    Pipeline (per (M-tile, C-tile), with ``n`` iterated inside):

                  layer_input            residual              post_mix       comb_mix
                    (M, C)              (M, n, C)              (M, n)        (M, n, n)
                       \\                  |                   /              /
                        \\                 |                  /              /
                         v                 v                 v              v
                              elementwise mix per dst stream j in [0, n)
                                  acc_j = post_mix[:, j] * layer_input
                                        + sum_h comb_mix[:, h, j] * residual[:, h, :]
                                                   |
                                                   v
                                              out[:, j, :]
                                                (M, n, C)

    All tensors live on the same CUDA device. ``layer_input`` and ``residual``
    are bf16 / fp16; ``post_mix`` and ``comb_mix`` are fp32. ``post_mix`` may be
    passed as ``(M, n, 1)`` (the layout produced by ``mhc()``) and is squeezed
    internally.

    Args:
        out:         optional pre-allocated output of shape ``(M, n, C)`` and
                     dtype ``layer_input.dtype``. Allocated if ``None``.
        layer_input: (M, C) bf16 / fp16 - mhc()'s ``layer_input`` output, i.e.
                     the transformer block input fed back into the residual.
        residual:    (M, n, C) bf16 / fp16 - the previous-layer multi-stream
                     residual ``x_l``.
        post_mix:    (M, n) or (M, n, 1) fp32 - mhc()'s ``h_post``.
        comb_mix:    (M, n, n) fp32 - mhc()'s ``h_res``; ``[m, h, j]`` is the
                     coefficient on residual stream ``h`` for output stream ``j``.
        config:      optional config dict ``{BLOCK_M, BLOCK_C, num_warps,
                     num_stages, waves_per_eu}``. Loaded from per-arch tuned
                     configs when ``None``.

    Returns the updated ``(M, n, C)`` multi-stream residual ``x_{l+1}``.

    Reference: arXiv:2512.24880.
    """
    M, n, C = residual.shape
    assert layer_input.shape == (
        M,
        C,
    ), f"layer_input shape mismatch: expected ({M}, {C}), got {layer_input.shape}"

    if post_mix.ndim == 3:
        post_mix = post_mix.squeeze(-1)
    assert post_mix.shape == (
        M,
        n,
    ), f"post_mix shape mismatch: expected ({M}, {n}), got {post_mix.shape}"
    assert comb_mix.shape == (
        M,
        n,
        n,
    ), f"comb_mix shape mismatch: expected ({M}, {n}, {n}), got {comb_mix.shape}"
    assert (
        layer_input.device == residual.device == post_mix.device == comb_mix.device
    ), "All tensors must be on the same device"
    assert layer_input.device.type == "cuda", "mHC post kernel requires CUDA device"

    if config is None:
        config = get_mhc_post_config(M, C)
    config = dict(config)  # Always copy to avoid mutating LRU cache

    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    BLOCK_C = config.pop("BLOCK_C", min(256, triton.next_power_of_2(C)))
    BLOCK_C = min(BLOCK_C, triton.next_power_of_2(C))

    _LOGGER.info(
        f"MHC_POST: layer_input={tuple(layer_input.shape)} "
        f"residual={tuple(residual.shape)} post_mix={tuple(post_mix.shape)} "
        f"comb_mix={tuple(comb_mix.shape)} "
        f"BLOCK_M={BLOCK_M} BLOCK_C={BLOCK_C}"
    )

    if out is None:
        out = torch.empty(M, n, C, dtype=layer_input.dtype, device=layer_input.device)
    else:
        assert out.shape == (
            M,
            n,
            C,
        ), f"out shape mismatch: expected ({M}, {n}, {C}), got {out.shape}"

    grid = (triton.cdiv(M, BLOCK_M),)
    _mhc_post_kernel[grid](
        out,
        layer_input,
        residual,
        post_mix,
        comb_mix,
        M=M,
        C=C,
        stride_x_m=layer_input.stride(0),
        stride_x_c=layer_input.stride(1),
        stride_res_m=residual.stride(0),
        stride_res_n=residual.stride(1),
        stride_res_c=residual.stride(2),
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        stride_out_c=out.stride(2),
        stride_post_m=post_mix.stride(0),
        stride_post_n=post_mix.stride(1),
        stride_comb_m=comb_mix.stride(0),
        stride_comb_src=comb_mix.stride(1),
        stride_comb_dst=comb_mix.stride(2),
        n=n,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        **config,
    )

    return out
