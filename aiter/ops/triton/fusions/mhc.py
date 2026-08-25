# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton

from aiter.ops.triton._triton_kernels.fusions import (
    _mhc_asymmetric_sinkhorn_kernel,
    _mhc_fused_kernel,
    _mhc_fused_split_kernel,
    _mhc_head_kernel,
    _mhc_post_kernel,
    _mhc_post_pre_reduce_apply_kernel,
    _mhc_post_pre_split_kernel,
    _mhc_reduce_apply_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.mhc_config_utils import (
    get_mhc_config,
    get_mhc_post_config,
)

DEVICE_ARCH = arch_info.get_arch()

_LOGGER = AiterTritonLogger()

# DSV4 forward paths below are Triton. Backward deliberately recomputes the
# formulas with PyTorch until dedicated Triton backward kernels are available.
MHC_DSV4_BACKWARD_FALLBACK = "pytorch_recompute"


def _validate_dot_config(config: dict, *, name: str) -> None:
    """Reject explicit Triton launch tiles that cannot form legal dot operands."""
    for key in ("BLOCK_M", "BLOCK_K"):
        if key in config:
            value = config[key]
            if value < 16 or value & (value - 1):
                raise ValueError(f"{name} {key} must be a power of two >= 16")
    if "BLOCK_N" in config:
        value = config["BLOCK_N"]
        if value < 16 or value & (value - 1):
            raise ValueError(f"{name} BLOCK_N must be a power of two >= 16")
    if "BLOCK_C" in config:
        value = config["BLOCK_C"]
        if value < 1 or value & (value - 1):
            raise ValueError(f"{name} BLOCK_C must be a positive power of two")
    if config.get("NUM_KSPLIT", 1) < 1:
        raise ValueError(f"{name} NUM_KSPLIT must be >= 1")


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
    config: dict | None = None,
    post_res_dtype: torch.dtype | None = None,
    alphas: torch.Tensor | None = None,
    phi_transposed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        phi:                (K, N=n+n+n*n), or (N, K) when
                            ``phi_transposed=True``. FP32 checkpoint storage is
                            cast tile-wise to ``x.dtype`` inside Triton.
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
        alphas:             optional (3,) FP32 device tensor replacing the
                            three scalar alpha arguments without host sync.
        phi_transposed:     interpret phi as checkpoint layout (N, K).

    Returns ``(h_post, h_res, layer_input)``:
        h_post      : (M, n, 1)  hc_post_mult_value * sigmoid(H_post)
        h_res       : (M, n, n)  Sinkhorn output (raw logits if sk_iters == 0)
        layer_input : (M, C)     sum_i (sigmoid(H_pre[m,i]) + hc_pre_eps) * x[m,i,c]

    ``post_res_dtype`` optionally overrides the storage dtype for ``h_post``
    and ``h_res``; the legacy default remains ``x.dtype``.

    Reference: arXiv:2512.24880.
    """
    M, K = x.shape
    C = K // n  # Derive C from K and n
    if phi_transposed:
        total_phi_cols, K_phi = phi.shape
        stride_phi_k, stride_phi_n = phi.stride(1), phi.stride(0)
    else:
        K_phi, total_phi_cols = phi.shape
        stride_phi_k, stride_phi_n = phi.stride(0), phi.stride(1)

    n_squared = n * n
    N_POW2 = triton.next_power_of_2(n)
    N_POW2_RES = triton.next_power_of_2(n_squared)
    N_TOTAL_POW2 = max(16, triton.next_power_of_2(n_squared + 2 * n))

    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
    else:
        _validate_dot_config(config, name="mhc")
    config = dict(config)  # Always copy to avoid mutating LRU cache

    num_ksplit = config.pop("NUM_KSPLIT", 1)
    BLOCK_M = max(16, config.pop("BLOCK_M", 64 if M >= 64 else 32))
    BLOCK_N = triton.next_power_of_2(config.pop("BLOCK_N", n_squared))

    n_blocks_res = triton.cdiv(n_squared, BLOCK_N)

    N_total = n_squared + 2 * n

    BLOCK_K = config.pop("BLOCK_K", 64)
    # Ensure BLOCK_K doesn't exceed K dimension
    BLOCK_K = max(16, min(BLOCK_K, triton.next_power_of_2(K)))
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
    if alphas is not None:
        assert alphas.shape == (3,), f"alphas must have shape (3,), got {alphas.shape}"
        assert alphas.dtype == torch.float32, "alphas must be float32"
        assert alphas.device == x.device, "alphas must be on the input device"
    alpha_ptr = alphas if alphas is not None else bias
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
    out = torch.empty(
        M,
        N_out_post_res,
        dtype=post_res_dtype or x.dtype,
        device=x.device,
    )
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
            stride_phi_k=stride_phi_k,
            stride_phi_n=stride_phi_n,
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
            alpha_ptr,
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
            ALPHAS_ARE_POINTER=alphas is not None,
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
            alpha_ptr,
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
            stride_phi_k=stride_phi_k,
            stride_phi_n=stride_phi_n,
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
            ALPHAS_ARE_POINTER=alphas is not None,
            **config,
        )

    # `out` layout is [post + res]: out[:, :n] is H^post, out[:, n:] is H^res
    h_post = out[:, :n].unsqueeze(-1)  # (M, n, 1)
    h_res = out[:, n:].view(M, n, n)  # (M, n, n)
    return h_post, h_res, layer_input


def mhc_post(
    out: torch.Tensor | None,
    layer_input: torch.Tensor,  # (M, C)  bf16 / fp16
    residual: torch.Tensor,  # (M, n, C)  bf16 / fp16
    post_mix: torch.Tensor,  # (M, n) or (M, n, 1)  fp32
    comb_mix: torch.Tensor,  # (M, n, n)  fp32 [src, dst]
    config: dict | None = None,
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


def mhc_post_pre(
    layer_input: torch.Tensor,  # (M, C)        bf16 / fp16
    residual_in: torch.Tensor,  # (M, n, C)     bf16 / fp16
    post_mix: torch.Tensor,  # (M, n) or (M, n, 1) fp32
    comb_mix: torch.Tensor,  # (M, n, n)     fp32
    phi: torch.Tensor,  # (n*C, N=n+n+n*n) bf16 / fp16
    alphas: torch.Tensor,  # (3,) fp32 — [alpha_pre, alpha_post, alpha_res]
    bias: torch.Tensor,  # (N,) fp32
    n: int,
    eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_iters: int = 20,
    asymmetric_exp_domain: bool = False,
    hc_sinkhorn_eps: float = 1e-6,
    residual_out: torch.Tensor | None = None,
    h_post: torch.Tensor | None = None,
    h_res: torch.Tensor | None = None,
    layer_input_out: torch.Tensor | None = None,
    acc_partial: torch.Tensor | None = None,
    acc_sq_partial: torch.Tensor | None = None,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mhc_post + (next) mhc_pre across two Triton launches.

    Launch 1 (``_mhc_post_pre_split_kernel``): per (M-tile, C-tile), one CTA
    does the mhc_post elementwise mix AND contributes one split-K K-slice of
    the next mhc_pre GEMM + sqrsum, sharing the post-output residual tile in
    registers between the two operations.

    Launch 2 (``_mhc_reduce_apply_kernel``, reused as-is): finalizes RMS-norm
    + bias + alpha + sigmoid (pre/post) + log-domain Sinkhorn (res) + the
    apply-pre fold ``layer_input_out[m, c] = sum_i pre_mix[m, i] *
    residual_out[m, i, c]``.

    The C-tile axis of the split kernel IS the pre's split-K axis (one CTA
    per K-split chunk of size ``n * BLOCK_C`` in the flattened pre input).

    Pipeline:
        residual_out[m, j, c] = post_mix[m, j] * layer_input[m, c]
                              + sum_h comb_mix[m, h, j] * residual_in[m, h, c]
        h_post                = hc_post_mult_value * sigmoid(...)        # of next pre
        h_res                 = Sinkhorn(...)                            # of next pre
        layer_input_out[m, c] = sum_i (sigmoid(H^pre[m, i]) + hc_pre_eps)
                                    * residual_out[m, i, c]

    Args:
        layer_input:        (M, C) bf16 / fp16, attn/ffn output (the post step's ``x``)
        residual_in:        (M, n, C) bf16 / fp16, prev-layer multi-stream residual
        post_mix:           (M, n) or (M, n, 1) fp32, from preceding mhc_pre
        comb_mix:           (M, n, n) fp32, from preceding mhc_pre; [src, dst]
        phi:                (n*C, N) bf16 / fp16, next pre's projection matrix
        alphas:             (3,) fp32 — ``[alpha_pre, alpha_post, alpha_res]``.
                            Loaded by the kernel via tl.load; pass the
                            ``hc_scale`` parameter tensor unchanged.
        bias:               (N,) fp32, next pre's bias
        n:                  stream / manifold dimension
        eps:                RMS-norm epsilon (default 1e-6)
        hc_pre_eps:         added to sigmoid(H^pre) (default 0.0)
        hc_post_mult_value: multiplier on sigmoid(H^post) (default 2.0)
        sinkhorn_iters:     SK iterations on next pre's H^res (default 20).
        asymmetric_exp_domain: when True, runs HIP-compatible exp-domain
                            Sinkhorn (``mhc_kernels.cu:493-507``): first iter
                            ``softmax(row) + eps`` then ``div(col + eps)``,
                            followed by ``sinkhorn_iters - 1`` symmetric
                            ``div(row+eps), div(col+eps)`` iters. Default
                            False (canonical log-domain Sinkhorn-Knopp).
        hc_sinkhorn_eps:    eps used during asymmetric-exp-domain Sinkhorn;
                            ignored when ``asymmetric_exp_domain`` is False
                            (default 1e-6).
        residual_out:       optional pre-allocated (M, n, C) output of dtype
                            ``layer_input.dtype``. Allocated if None.
        h_post:             optional pre-allocated post-stream output. Accepts
                            either (M, n) or (M, n, 1). Any float dtype — the
                            kernel implicit-casts on store. Allocated as
                            (M, n, 1) in ``layer_input.dtype`` if None.
        h_res:              optional pre-allocated res-stream output of shape
                            (M, n, n). Any float dtype. Allocated as
                            (M, n, n) in ``layer_input.dtype`` if None.
        layer_input_out:    optional pre-allocated (M, C) output of dtype
                            ``layer_input.dtype``. Allocated if None.
        config:             optional config dict; loaded from
                            get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
                            when None. NUM_KSPLIT is ignored — the split kernel's
                            grid uses ``cdiv(C, BLOCK_C)`` K-splits.

    Returns ``(h_post, h_res, layer_input_out, residual_out)``:
        h_post:          (M, n, 1)  hc_post_mult_value * sigmoid(H^post)
        h_res:           (M, n, n)  Sinkhorn output (raw logits if sk_iters == 0)
        layer_input_out: (M, C)     sum_i (sigmoid(H^pre[m, i]) + hc_pre_eps)
                                          * residual_out[m, i, c]
        residual_out:    (M, n, C)  new mHC residual stream (must be kept;
                                    consumed by the next layer's hc_post)

    Reference: arXiv:2512.24880.
    """
    M, C = layer_input.shape
    M_res, n_res, C_res = residual_in.shape
    assert (M, n, C) == (
        M_res,
        n_res,
        C_res,
    ), f"shape mismatch: layer_input=({M},{C}), residual_in=({M_res},{n_res},{C_res}), n={n}"

    K_phi, N_phi = phi.shape
    n_squared = n * n
    N_total = 2 * n + n_squared
    K = n * C
    assert K_phi == K, f"phi K mismatch: expected {K}, got {K_phi}"
    assert N_phi == N_total, f"phi N mismatch: expected {N_total}, got {N_phi}"

    if post_mix.ndim == 3:
        post_mix = post_mix.squeeze(-1)
    assert post_mix.shape == (
        M,
        n,
    ), f"post_mix shape mismatch: expected ({M}, {n}), got {tuple(post_mix.shape)}"
    assert comb_mix.shape == (
        M,
        n,
        n,
    ), f"comb_mix shape mismatch: expected ({M}, {n}, {n}), got {tuple(comb_mix.shape)}"
    assert (
        bias.shape[0] == N_total
    ), f"bias shape mismatch: expected ({N_total},), got {tuple(bias.shape)}"
    assert (
        layer_input.device
        == residual_in.device
        == phi.device
        == bias.device
        == post_mix.device
        == comb_mix.device
    ), "All tensors must be on the same device"
    assert layer_input.device.type == "cuda", "mHC kernel requires CUDA device"
    assert sinkhorn_iters >= 0, f"sinkhorn_iters must be >= 0, got {sinkhorn_iters}"

    device = layer_input.device
    dtype = layer_input.dtype

    N_POW2 = triton.next_power_of_2(n)
    N_POW2_RES = triton.next_power_of_2(n_squared)
    N_TOTAL_POW2 = triton.next_power_of_2(N_total)

    if sinkhorn_iters > 0:
        assert N_POW2_RES == n_squared, (
            f"sinkhorn_iters>0 requires n_squared ({n_squared}) to be a power of 2; "
            f"got N_POW2_RES={N_POW2_RES}. For non-power-of-2 n, run with "
            f"sinkhorn_iters=0."
        )

    # Reuse MHC_FUSED tuning. Two block-sizes serve distinct roles here:
    #   - BLOCK_K  : per-CTA K-axis width in the post_pre split kernel; one
    #                CTA covers BLOCK_K K-elements of the next-pre GEMM,
    #                equivalent to BLOCK_K/n contiguous C-elements per stream.
    #                NUM_KSPLIT = cdiv(K, BLOCK_K).
    #   - BLOCK_C  : C-axis tile for _mhc_reduce_apply_kernel only (apply-pre
    #                output parallelism). Independent of BLOCK_K.
    if config is None:
        config, _ = get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
    config = dict(config)
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    if DEVICE_ARCH in ("gfx942",):
        BLOCK_M = 16
    BLOCK_K = config.pop("BLOCK_K", min(512, triton.next_power_of_2(K)))
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))
    BLOCK_C = config.pop("BLOCK_C", min(128, triton.next_power_of_2(C)))
    BLOCK_C = min(BLOCK_C, triton.next_power_of_2(C))
    # Knobs irrelevant to mhc_post_pre.
    config.pop("BLOCK_N", None)
    config.pop("NUM_KSPLIT", None)

    assert BLOCK_K % n == 0, (
        f"BLOCK_K ({BLOCK_K}) must be divisible by n ({n}) — each CTA owns "
        f"n contiguous BLOCK_K/n-wide C-slices of the next-pre GEMM"
    )
    # Per-CTA C-slice width (constexpr inside the split kernel).
    BLOCK_C_SPLIT = BLOCK_K // n
    NUM_KSPLIT = triton.cdiv(C, BLOCK_C_SPLIT)  # == cdiv(K, BLOCK_K)
    KSPLIT_POW2 = triton.next_power_of_2(NUM_KSPLIT)

    # `_mhc_reduce_apply_kernel` uses a (num_M_tiles, NUM_C_BLOCKS + 2) grid.
    NUM_C_BLOCKS = triton.cdiv(C, BLOCK_C)

    _LOGGER.info(
        f"MHC_POST_PRE: layer_input={tuple(layer_input.shape)} "
        f"residual_in={tuple(residual_in.shape)} phi={tuple(phi.shape)} "
        f"alphas={tuple(alphas.shape)}@{alphas.dtype} "
        f"hc_pre_eps={hc_pre_eps} hc_post_mult_value={hc_post_mult_value} "
        f"sinkhorn_iters={sinkhorn_iters} "
        f"BLOCK_M={BLOCK_M} BLOCK_K={BLOCK_K} BLOCK_C_SPLIT={BLOCK_C_SPLIT} "
        f"BLOCK_C={BLOCK_C} NUM_KSPLIT={NUM_KSPLIT} NUM_C_BLOCKS={NUM_C_BLOCKS} "
        f"N_TOTAL_POW2={N_TOTAL_POW2}"
    )
    assert alphas.shape == (
        3,
    ), f"alphas shape mismatch: expected (3,), got {tuple(alphas.shape)}"

    if residual_out is None:
        residual_out = torch.empty(M, n, C, dtype=dtype, device=device)
    else:
        assert residual_out.shape == (
            M,
            n,
            C,
        ), f"residual_out shape mismatch: expected ({M}, {n}, {C}), got {tuple(residual_out.shape)}"
        assert (
            residual_out.dtype == dtype
        ), f"residual_out dtype mismatch: expected {dtype}, got {residual_out.dtype}"

    # Split-K reduce-apply scratch. Write-before-read within this call (the
    # split kernel writes every entry the reduce kernel reads, masked the same
    # way), so callers may pass persistent buffers — keeping `.data_ptr()`
    # stable lets CUDAGraph capture/replay work without surprise re-allocation.
    if acc_partial is None:
        acc_partial = torch.empty(
            (NUM_KSPLIT, M, N_total), dtype=torch.float32, device=device
        )
    else:
        assert acc_partial.shape == (NUM_KSPLIT, M, N_total), (
            f"acc_partial shape mismatch: expected ({NUM_KSPLIT}, {M}, {N_total}), "
            f"got {tuple(acc_partial.shape)}"
        )
        assert (
            acc_partial.dtype == torch.float32
        ), f"acc_partial dtype mismatch: expected float32, got {acc_partial.dtype}"
    if acc_sq_partial is None:
        acc_sq_partial = torch.empty(
            (NUM_KSPLIT, M), dtype=torch.float32, device=device
        )
    else:
        assert acc_sq_partial.shape == (NUM_KSPLIT, M), (
            f"acc_sq_partial shape mismatch: expected ({NUM_KSPLIT}, {M}), "
            f"got {tuple(acc_sq_partial.shape)}"
        )
        assert acc_sq_partial.dtype == torch.float32, (
            f"acc_sq_partial dtype mismatch: expected float32, "
            f"got {acc_sq_partial.dtype}"
        )

    # --- Launch 1: fused post + partial pre GEMM/sqrsum, one CTA per (M-tile, C-tile).
    grid_split = (triton.cdiv(M, BLOCK_M), NUM_KSPLIT)
    _mhc_post_pre_split_kernel[grid_split](
        layer_input,
        residual_in,
        post_mix,
        comb_mix,
        residual_out,
        phi,
        acc_partial,
        acc_sq_partial,
        M=M,
        N=N_total,
        n=n,
        C=C,
        stride_x_m=layer_input.stride(0),
        stride_x_c=layer_input.stride(1),
        stride_resin_m=residual_in.stride(0),
        stride_resin_n=residual_in.stride(1),
        stride_resin_c=residual_in.stride(2),
        stride_post_m=post_mix.stride(0),
        stride_post_n=post_mix.stride(1),
        stride_comb_m=comb_mix.stride(0),
        stride_comb_src=comb_mix.stride(1),
        stride_comb_dst=comb_mix.stride(2),
        stride_resout_m=residual_out.stride(0),
        stride_resout_n=residual_out.stride(1),
        stride_resout_c=residual_out.stride(2),
        stride_phi_k=phi.stride(0),
        stride_phi_n=phi.stride(1),
        stride_acc_k=acc_partial.stride(0),
        stride_acc_m=acc_partial.stride(1),
        stride_acc_n=acc_partial.stride(2),
        stride_acc_sq_k=acc_sq_partial.stride(0),
        stride_acc_sq_m=acc_sq_partial.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C_SPLIT,
        N_TOTAL_POW2=N_TOTAL_POW2,
        **config,
    )

    # --- Launch 2: reduce-apply kernel writes h_post and h_res directly.
    # Allocate any caller-omitted outputs in `layer_input.dtype`. Callers can
    # pre-allocate them in any float dtype (e.g. fp32) to skip a downstream
    # cast — the kernel implicit-casts at the store.
    if h_post is None:
        h_post = torch.empty(M, n, 1, dtype=dtype, device=device)
    else:
        if h_post.ndim == 2:
            assert h_post.shape == (
                M,
                n,
            ), f"h_post 2D shape mismatch: expected ({M}, {n}), got {tuple(h_post.shape)}"
        else:
            assert h_post.shape == (
                M,
                n,
                1,
            ), f"h_post 3D shape mismatch: expected ({M}, {n}, 1), got {tuple(h_post.shape)}"
    if h_res is None:
        h_res = torch.empty(M, n, n, dtype=dtype, device=device)
    else:
        assert h_res.shape == (
            M,
            n,
            n,
        ), f"h_res shape mismatch: expected ({M}, {n}, {n}), got {tuple(h_res.shape)}"
    if layer_input_out is None:
        layer_input_out = torch.empty(M, C, dtype=dtype, device=device)
    else:
        assert layer_input_out.shape == (M, C), (
            f"layer_input_out shape mismatch: expected ({M}, {C}), "
            f"got {tuple(layer_input_out.shape)}"
        )

    # 2D (M, n) view for the kernel — h_post.ndim is 2 or 3, both OK after view.
    h_post_2d = h_post.view(M, n)
    # 2D (M, n*n) view for the kernel — h_res is contiguous (M, n, n).
    h_res_2d = h_res.view(M, n_squared)

    # residual_out (M, n, C) is contiguous after `torch.empty`; flatten gives
    # the (M, n*C) layout the reduce-apply kernel expects as ``x``.
    x_flat = residual_out.view(M, n * C)

    # Grid: NUM_C_BLOCKS apply-pre CTAs + 2 dedicated CTAs for post & res.
    BLOCK_M_POST_RES = 1
    grid_reduce_apply = (
        triton.cdiv(M, BLOCK_M) * NUM_C_BLOCKS + triton.cdiv(M, BLOCK_M_POST_RES) * 2,
    )
    _mhc_post_pre_reduce_apply_kernel[grid_reduce_apply](
        acc_partial,
        acc_sq_partial,
        alphas,
        bias,
        x_flat,
        h_post_2d,
        h_res_2d,
        layer_input_out,
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
        stride_xm=x_flat.stride(0),
        stride_xk=x_flat.stride(1),
        stride_hp_m=h_post_2d.stride(0),
        stride_hp_n=h_post_2d.stride(1),
        stride_hr_m=h_res_2d.stride(0),
        stride_hr_n=h_res_2d.stride(1),
        stride_li_m=layer_input_out.stride(0),
        stride_li_c=layer_input_out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        N_POW2=N_POW2,
        N_POW2_RES=N_POW2_RES,
        ACTUAL_KSPLIT=NUM_KSPLIT,
        KSPLIT_POW2=KSPLIT_POW2,
        BLOCK_M_POST_RES=BLOCK_M_POST_RES,
        NUM_SINKHORN_ITERS=sinkhorn_iters,
        ASYMMETRIC_EXP_DOMAIN=asymmetric_exp_domain,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        **config,
    )

    # Reshape h_post to (M, n, 1) on return for compatibility with downstream
    # callers that expect the trailing singleton (e.g. HIP mhc_post).
    if h_post.ndim == 2:
        h_post = h_post.unsqueeze(-1)
    return h_post, h_res, layer_input_out, residual_out


def _mhc_pre_dsv4_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable correctness fallback used only by DSV4 backward."""
    M, n, C = residual.shape
    x = residual.float().reshape(M, n * C)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + eps)
    phi = fn.T.to(residual.dtype)
    logits = (x / rms) @ phi.float()
    stream_scale = torch.cat(
        (
            scale[0].expand(n),
            scale[1].expand(n),
            scale[2].expand(n * n),
        )
    )
    logits = logits * stream_scale + base.float()
    pre = torch.sigmoid(logits[:, :n]) + pre_eps
    post = (post_multiplier * torch.sigmoid(logits[:, n : 2 * n])).unsqueeze(-1)
    comb = logits[:, 2 * n :].reshape(M, n, n)
    if sinkhorn_iters > 0:
        row_max = comb.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb - row_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + sinkhorn_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = torch.einsum("mn,mnc->mc", pre, residual.float())
    return post, comb, layer_input.to(residual.dtype)


def _mhc_head_dsv4_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    eps: float,
    pre_eps: float,
) -> torch.Tensor:
    """Differentiable correctness fallback used only by DSV4 backward."""
    M, n, C = residual.shape
    x = residual.float().reshape(M, n * C)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + eps)
    phi = fn.T.to(residual.dtype)
    logits = ((x / rms) @ phi.float()) * scale.reshape(1) + base.float()
    mix = torch.sigmoid(logits) + pre_eps
    out = torch.einsum("mn,mnc->mc", mix, residual.float())
    return out.to(residual.dtype)


def _validate_dsv4_parameters(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    head: bool,
) -> tuple[int, int, int]:
    if residual.ndim != 3:
        raise ValueError(
            f"residual must have shape [M,n,C], got {tuple(residual.shape)}"
        )
    M, n, C = residual.shape
    rows = n if head else 2 * n + n * n
    if fn.shape != (rows, n * C):
        raise ValueError(f"fn must have shape ({rows}, {n * C}), got {tuple(fn.shape)}")
    scale_size = 1 if head else 3
    if scale.shape != (scale_size,):
        raise ValueError(
            f"scale must have shape ({scale_size},), got {tuple(scale.shape)}"
        )
    if base.shape != (rows,):
        raise ValueError(f"base must have shape ({rows},), got {tuple(base.shape)}")
    if residual.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("residual must be float16 or bfloat16")
    if (
        fn.dtype != torch.float32
        or scale.dtype != torch.float32
        or base.dtype != torch.float32
    ):
        raise TypeError("fn, scale, and base must be float32")
    if not (residual.device == fn.device == scale.device == base.device):
        raise ValueError("residual, fn, scale, and base must be on the same device")
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous for CUDA-graph-safe execution")
    if not fn.is_contiguous():
        raise ValueError("fn must be contiguous for CUDA-graph-safe execution")
    if residual.device.type != "cuda":
        raise ValueError("DSV4 MHC Triton forward requires a CUDA device")
    if not head and n * n != triton.next_power_of_2(n * n):
        raise ValueError("DSV4 Sinkhorn requires n*n to be a power of two")
    return M, n, C


def _mhc_pre_dsv4_forward(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    sinkhorn_iters: int,
    config: dict | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, n, C = _validate_dsv4_parameters(residual, fn, scale, base, head=False)
    x = residual.view(M, n * C)
    post, raw_comb, layer_input = mhc(
        x,
        fn,
        0.0,
        0.0,
        0.0,
        base,
        n,
        eps=eps,
        hc_pre_eps=pre_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_iters=0,
        config=config,
        post_res_dtype=torch.float32,
        alphas=scale,
        phi_transposed=True,
    )
    raw_comb_flat = raw_comb.view(M, n * n)
    comb = torch.empty((M, n, n), dtype=raw_comb.dtype, device=raw_comb.device)
    comb_flat = comb.view(M, n * n)
    _mhc_asymmetric_sinkhorn_kernel[(M,)](
        raw_comb_flat,
        comb_flat,
        M=M,
        stride_logits_m=raw_comb_flat.stride(0),
        stride_logits_n=raw_comb_flat.stride(1),
        stride_out_m=comb_flat.stride(0),
        stride_out_n=comb_flat.stride(1),
        n=n,
        N_POW2_RES=triton.next_power_of_2(n * n),
        NUM_SINKHORN_ITERS=sinkhorn_iters,
        eps=sinkhorn_eps,
        num_warps=1,
    )
    return post, comb, layer_input


class _MHCPreDSV4(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        residual,
        fn,
        scale,
        base,
        eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        sinkhorn_iters,
        config,
    ):
        ctx.save_for_backward(residual, fn, scale, base)
        ctx.params = (eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters)
        return _mhc_pre_dsv4_forward(
            residual,
            fn,
            scale,
            base,
            eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_iters,
            config,
        )

    @staticmethod
    def backward(ctx, grad_post, grad_comb, grad_layer):
        residual, fn, scale, base = ctx.saved_tensors
        inputs = tuple(
            t.detach().requires_grad_(True) for t in (residual, fn, scale, base)
        )
        with torch.enable_grad():
            outputs = _mhc_pre_dsv4_torch(*inputs, *ctx.params)
        grad_outputs = tuple(
            torch.zeros_like(output) if grad is None else grad
            for output, grad in zip(outputs, (grad_post, grad_comb, grad_layer))
        )
        grads = torch.autograd.grad(outputs, inputs, grad_outputs)
        return (*grads, None, None, None, None, None, None)


def mhc_pre_dsv4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run DSV4 MHC pre-processing with a pure-Triton production forward.

    ``residual`` is ``[M,n,C]`` in fp16/bf16; ``fn`` is
    ``[2n+n²,n*C]`` FP32; ``scale`` is ``[3]`` FP32; and ``base`` is
    ``[2n+n²]`` FP32. Returns ``(post_mix, comb_mix, layer_input)`` with
    shapes ``[M,n,1]``, ``[M,n,n]``, and ``[M,C]``. The mixes are FP32.
    Backward is the documented ``pytorch_recompute`` correctness fallback.
    """
    if sinkhorn_repeat < 0:
        raise ValueError("sinkhorn_repeat must be non-negative")
    return _MHCPreDSV4.apply(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        int(sinkhorn_repeat),
        config,
    )


class _MHCPostDSV4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_input, residual, post_mix, comb_mix, config):
        ctx.save_for_backward(layer_input, residual, post_mix, comb_mix)
        return mhc_post(None, layer_input, residual, post_mix, comb_mix, config)

    @staticmethod
    def backward(ctx, grad_out):
        layer_input, residual, post_mix, comb_mix = ctx.saved_tensors
        inputs = tuple(
            t.detach().requires_grad_(True)
            for t in (layer_input, residual, post_mix, comb_mix)
        )
        with torch.enable_grad():
            x, res, post, comb = inputs
            post_2d = post.squeeze(-1) if post.ndim == 3 else post
            out = post_2d[:, :, None] * x.float()[:, None, :]
            out = out + torch.einsum("mhc,mhj->mjc", res.float(), comb.float())
            out = out.to(x.dtype)
        grads = torch.autograd.grad(out, inputs, grad_out)
        return (*grads, None)


def mhc_post_dsv4(
    layer_input: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    config: dict | None = None,
) -> torch.Tensor:
    """Apply DSV4 post mixing with Triton forward and recomputation backward.

    Shapes are ``layer_input=[M,C]``, ``residual=[M,n,C]``,
    ``post_mix=[M,n,1]`` (or ``[M,n]``), and ``comb_mix=[M,n,n]``.
    """
    return _MHCPostDSV4.apply(layer_input, residual, post_mix, comb_mix, config)


def _mhc_head_dsv4_forward(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    eps: float,
    pre_eps: float,
    config: dict | None,
) -> torch.Tensor:
    M, n, C = _validate_dsv4_parameters(residual, fn, scale, base, head=True)
    x = residual.view(M, n * C)
    config = {} if config is None else dict(config)
    _validate_dot_config(config, name="mhc_head_dsv4")
    BLOCK_M = max(16, config.pop("BLOCK_M", 16))
    BLOCK_K = config.pop("BLOCK_K", min(64, triton.next_power_of_2(n * C)))
    BLOCK_K = max(16, min(BLOCK_K, triton.next_power_of_2(n * C)))
    BLOCK_C = config.pop("BLOCK_C", min(128, triton.next_power_of_2(C)))
    out = torch.empty(M, C, dtype=residual.dtype, device=residual.device)
    _mhc_head_kernel[(triton.cdiv(M, BLOCK_M),)](
        x,
        fn,
        scale,
        base,
        out,
        M=M,
        K=n * C,
        n=n,
        C=C,
        eps=eps,
        pre_eps=pre_eps,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_fn_n=fn.stride(0),
        stride_fn_k=fn.stride(1),
        stride_om=out.stride(0),
        stride_oc=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        N_TILE=max(16, triton.next_power_of_2(n)),
        **config,
    )
    return out


class _MHCHeadDSV4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, fn, scale, base, eps, pre_eps, config):
        ctx.save_for_backward(residual, fn, scale, base)
        ctx.params = (eps, pre_eps)
        return _mhc_head_dsv4_forward(residual, fn, scale, base, eps, pre_eps, config)

    @staticmethod
    def backward(ctx, grad_out):
        inputs = tuple(t.detach().requires_grad_(True) for t in ctx.saved_tensors)
        with torch.enable_grad():
            out = _mhc_head_dsv4_torch(*inputs, *ctx.params)
        grads = torch.autograd.grad(out, inputs, grad_out)
        return (*grads, None, None, None)


def mhc_head_dsv4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    config: dict | None = None,
) -> torch.Tensor:
    """Contract DSV4 residual heads using a fused pure-Triton forward.

    ``residual`` is ``[M,n,C]`` fp16/bf16, ``fn`` is ``[n,n*C]`` FP32,
    ``scale`` is ``[1]`` FP32, and ``base`` is ``[n]`` FP32. Returns
    ``[M,C]`` in ``residual.dtype``. Backward uses PyTorch recomputation.
    """
    return _MHCHeadDSV4.apply(residual, fn, scale, base, rms_eps, hc_pre_eps, config)
