# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused delayed mHC seam (``mhc_fused_post_pre_delayed_rmsnorm``)."""

import pytest
import torch

from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
    mhc_fused_post_pre_delayed_rmsnorm,
)

# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat
ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)


def ref_mhc_fused_post_pre_delayed_rmsnorm(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    sinkhorn_repeat,
    pre_mix,
    sublayer_out=None,
    post_layer_mix=None,
    comb_res_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    """fp32 torch reference (vLLM's mhc_post_torch + mhc_pre_delayed_torch + RMSNorm)."""
    T, n, _H = residual.shape
    if sublayer_out is not None:
        mixed = torch.einsum(
            "tij,tih->tjh", comb_res_mix.float().view(T, n, n), residual.float()
        )
        R = (
            mixed
            + post_layer_mix.float().view(T, n, 1) * sublayer_out.float().unsqueeze(1)
        ).to(residual.dtype)
    else:
        R = residual
    x = R.flatten(1).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_pre_eps
    post = (
        torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
        * hc_post_mult
    )
    comb = mixes[:, 2 * n :].view(-1, n, n) * hc_scale[2] + hc_base[2 * n :].view(
        1, n, n
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    li = (pre_mix.float().view(T, n, 1) * R.float()).sum(dim=1).to(residual.dtype)
    lf = li.float()
    li = (
        lf
        * torch.rsqrt(lf.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).to(residual.dtype)
    return R, post.unsqueeze(-1), comb, li, pre


def make_inputs(T, H, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    residual = (torch.randn(T, 4, H, device=device, generator=g) * 1.5).to(
        torch.bfloat16
    )
    y = torch.randn(T, H, device=device, generator=g).to(torch.bfloat16)
    fn = torch.randn(24, 4 * H, device=device, generator=g) * 0.01
    hc_scale = torch.tensor([0.7, 1.3, 0.9], device=device)
    hc_base = torch.randn(24, device=device, generator=g) * 0.5
    post = torch.rand(T, 4, 1, device=device, generator=g) * 2
    comb = torch.softmax(torch.randn(T, 4, 4, device=device, generator=g), -1)
    pre = torch.sigmoid(torch.randn(T, 4, device=device, generator=g)) + 1e-6
    w = (1 + 0.2 * torch.randn(H, device=device, generator=g)).to(torch.bfloat16)
    return residual, y, fn, hc_scale, hc_base, post, comb, pre, w


# post: a regular seam; no_post: an Engram seam, whose post block runs separately;
# identity_pre: pre_mix=None, as at the draft model's entry seam
@pytest.mark.parametrize("mode", ["post", "no_post", "identity_pre"])
@pytest.mark.parametrize("T", [1, 100, 384, 16384])
def test_mhc_fused_post_pre_delayed_rmsnorm(T, mode):
    residual, y, fn, hc_scale, hc_base, post, comb, pre, w = make_inputs(T, 5120)
    kw = {"norm_weight": w}
    if mode != "no_post":
        kw.update(sublayer_out=y, post_layer_mix=post, comb_res_mix=comb)
    ref_pre = pre
    if mode == "identity_pre":
        ref_pre = torch.zeros_like(pre)
        ref_pre[:, 0] = 1
        pre = None
    R, post_o, comb_o, li, pre_o = mhc_fused_post_pre_delayed_rmsnorm(
        residual, fn, hc_scale, hc_base, *ARGS, pre_mix=pre, **kw
    )
    ref = ref_mhc_fused_post_pre_delayed_rmsnorm(
        residual, fn, hc_scale, hc_base, *ARGS, pre_mix=ref_pre, **kw
    )
    torch.testing.assert_close(R, ref[0])
    # a one-ulp rounding difference in the new residual carries into the collapse,
    # so near-zero layer_input values need an absolute tolerance
    torch.testing.assert_close(li, ref[3], atol=2e-2, rtol=1e-2)
    for out, expected in ((post_o, ref[1]), (comb_o, ref[2]), (pre_o, ref[4])):
        torch.testing.assert_close(out, expected, atol=5e-4, rtol=1e-3)


def test_mhc_fused_post_pre_delayed_rmsnorm_empty():
    residual, y, fn, hc_scale, hc_base, post, comb, pre, w = make_inputs(0, 5120)
    R, _, _, li, pre_o = mhc_fused_post_pre_delayed_rmsnorm(
        residual,
        fn,
        hc_scale,
        hc_base,
        *ARGS,
        pre_mix=pre,
        sublayer_out=y,
        post_layer_mix=post,
        comb_res_mix=comb,
        norm_weight=w,
    )
    assert R.shape == (0, 4, 5120) and li.shape == (0, 5120) and pre_o.shape == (0, 4)
