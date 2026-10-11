# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gates of a delayed mHC seam (pre, post, comb) from split-K partials of the pre
projection. See the kernel module for the math."""

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.mhc_delayed_gates import (
    _mhc_delayed_gates_kernel,
)


def mhc_delayed_gates(
    gemm_out: torch.Tensor,
    sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    hc_hidden_size: int,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delayed mHC gates from split-K partials.

    Args:
        gemm_out: (S, T, >= 24) fp32 split-K partials of flatten(R') @ fn^T, last dim
            contiguous (``mhc_pre_gemm_sqrsum``'s output, or the gfx942 seam kernel's).
        sqrsum: (S, T) fp32 partial sums of R'^2, any strides (for the seam kernel's
            layout pass ``gemm_out[:, :, 24]``).
        hc_scale (3,), hc_base (24,): fp32 gate scale and bias.
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat:
            the model's hc constants, as for ``mhc_pre``.
        hc_hidden_size: K = hc_mult * hidden_size, the length the RMS is taken over.
        out: optional preallocated (post_mix (T, 4, 1), comb_mix (T, 4, 4),
            pre_mix (T, 4)), fp32 and contiguous.

    Returns:
        (post_mix (T, 4, 1), comb_mix (T, 4, 4), pre_mix (T, 4)), fp32.
    """
    S, T = gemm_out.shape[0], gemm_out.shape[1]
    assert gemm_out.dtype == torch.float32 and sqrsum.dtype == torch.float32
    assert gemm_out.shape[2] >= 24 and gemm_out.stride(2) == 1
    assert sqrsum.shape == (S, T)
    assert hc_scale.shape == (3,) and hc_base.shape == (24,)
    assert hc_scale.dtype == hc_base.dtype == torch.float32
    assert hc_scale.is_contiguous() and hc_base.is_contiguous()
    device = gemm_out.device
    if out is None:
        out = (
            torch.empty(T, 4, 1, device=device, dtype=torch.float32),
            torch.empty(T, 4, 4, device=device, dtype=torch.float32),
            torch.empty(T, 4, device=device, dtype=torch.float32),
        )
    post_mix, comb_mix, pre_mix = out
    assert all(t.is_contiguous() and t.dtype == torch.float32 for t in out)
    if T == 0:
        return post_mix, comb_mix, pre_mix
    # One wave per program: the split-lane reduction stays inside the wave (DPP, no
    # barriers; with two waves the 25 cross-wave reductions cost ~11 us).
    split_lanes = 1 if S <= 4 else min(64, triton.next_power_of_2(S))
    block_t = 64 // split_lanes
    _mhc_delayed_gates_kernel[(triton.cdiv(T, block_t),)](
        gemm_out,
        sqrsum,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        pre_mix,
        T,
        S,
        gemm_out.stride(0),
        gemm_out.stride(1),
        sqrsum.stride(0),
        sqrsum.stride(1),
        1.0 / hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        REPEAT=int(sinkhorn_repeat),
        BLOCK_T=block_t,
        SPLIT_LANES=split_lanes,
        num_warps=1,
    )
    return post_mix, comb_mix, pre_mix
