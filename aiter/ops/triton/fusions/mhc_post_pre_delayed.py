# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused delayed ("shifted") mHC seam used by DeepSeek-V4.1: post-mix, gate
projection, pre/post/comb gates, stream collapse with the carried pre gate and its
RMSNorm in two Triton launches. See the kernel module for the math and structure."""

import torch
import triton

from aiter.ops.mhc import get_mhc_fused_post_pre_config
from aiter.ops.triton._triton_kernels.fusions.mhc_post_pre_delayed import (
    _mhc_post_pre_delayed_main_kernel,
    _mhc_post_pre_delayed_reduce_kernel,
)


def mhc_post_pre_delayed(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused delayed mHC seam.

    Args:
        residual: (T, 4, H) bf16 residual streams entering the seam.
        fn: (24, 4*H) fp32 gate projection; hc_scale (3,), hc_base (24,) fp32.
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat:
            the model's hc constants.
        pre_mix: (T, 4) fp32 pre gate carried from the previous seam; None means the
            identity gate (the collapse is stream 0), as at the draft model's entry seam.
        sublayer_out / post_layer_mix / comb_res_mix: the previous sub-layer's output
            (T, H) bf16 and its post (T, 4[, 1]) / comb (T, 4, 4) fp32 gates; when
            given the post-mix is applied here and the new residual returned,
            otherwise ``residual`` is projected as is and returned.
        norm_weight: (H,) RMSNorm weight applied to the collapse; norm_eps its epsilon.
        residual_out: optional (T, 4, H) bf16 buffer for the new residual.

    Returns:
        (residual_out, post_mix (T, 4, 1), comb_mix (T, 4, 4), layer_input (T, H) bf16,
         next_pre_mix (T, 4)); the gates are fp32.
    """
    assert (
        residual.dim() == 3
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
    )
    T, n, H = residual.shape
    assert n == 4, f"the fused seam kernel is specialised for hc_mult=4, got {n}"
    assert fn.shape == (24, 4 * H) and fn.dtype == torch.float32 and fn.is_contiguous()
    assert hc_scale.shape == (3,) and hc_base.shape == (24,)
    assert norm_weight is not None and norm_weight.shape == (H,)
    device = residual.device
    if pre_mix is None:
        pre_mix = torch.zeros(T, 4, dtype=torch.float32, device=device)
        pre_mix[:, 0] = 1.0
    pre_2d = pre_mix.reshape(T, 4).contiguous()
    norm_weight = norm_weight.contiguous()

    has_post = sublayer_out is not None
    if has_post:
        assert post_layer_mix is not None and comb_res_mix is not None
        assert sublayer_out.shape == (T, H) and sublayer_out.dtype == torch.bfloat16
        sublayer_out = sublayer_out.contiguous()
        post_2d = post_layer_mix.reshape(T, 4).contiguous()
        comb_3d = comb_res_mix.reshape(T, 4, 4).contiguous()
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        assert (
            residual_out.shape == residual.shape
            and residual_out.dtype == torch.bfloat16
        )
    else:
        assert residual_out is None
        residual_out = residual
        sublayer_out = post_2d = comb_3d = residual  # unused

    post_out = torch.empty(T, 4, 1, dtype=torch.float32, device=device)
    comb_out = torch.empty(T, 4, 4, dtype=torch.float32, device=device)
    next_pre = torch.empty(T, 4, dtype=torch.float32, device=device)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=device)
    if T == 0:
        return residual_out, post_out, comb_out, layer_input, next_pre

    split_k = get_mhc_fused_post_pre_config(T, H)[0]
    block_m, tile_k = 16, 32
    partial = torch.empty(T, split_k, 32, dtype=torch.float32, device=device)

    _mhc_post_pre_delayed_main_kernel[(triton.cdiv(T, block_m), split_k)](
        residual,
        sublayer_out,
        post_2d,
        comb_3d,
        pre_2d,
        fn,
        residual_out,
        layer_input,
        partial,
        T,
        H=H,
        BLOCK_M=block_m,
        TILE_K=tile_k,
        NUM_KSPLIT=split_k,
        HAS_POST=has_post,
        num_warps=2,
        num_stages=3,
    )

    _mhc_post_pre_delayed_reduce_kernel[(T,)](
        partial,
        hc_scale,
        hc_base,
        next_pre,
        post_out,
        comb_out,
        layer_input,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        norm_eps,
        H=H,
        NUM_KSPLIT=split_k,
        NUM_SINKHORN_ITERS=int(sinkhorn_repeat),
        BLOCK_C=1024,
        num_warps=1,
    )
    return residual_out, post_out, comb_out, layer_input, next_pre
