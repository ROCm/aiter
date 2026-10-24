# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression for precision lost while merging GDN inverse blocks."""

import math

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("hg,h", [(8, 24), (4, 12)])
@pytest.mark.parametrize("use_exp2", [False, True])
def test_gdn_prepare_correlated_keys(hg, h, use_exp2):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Requires ROCm")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if not arch.startswith(("gfx94", "gfx95")):
        pytest.skip("Requires a CDNA GDN prepare device")
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        gdn_prepare_fwd_flydsl,
    )

    torch.manual_seed(2026)
    # Nearly parallel keys, weak decay, and beta close to one expose residuals
    # lost when the off-diagonal block products are rounded straight to BF16.
    base = torch.randn(1, 1, hg, 128, device="cuda")
    k = F.normalize(
        base + 0.01 * torch.randn(1, 64, hg, 128, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(1, 64, h, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.full((1, 64, h), -0.0001, device="cuda")
    beta = torch.full_like(g, 0.999)
    w, u, gc = gdn_prepare_fwd_flydsl(k, v, g, beta, use_exp2=use_exp2)

    keys = k.float().repeat_interleave(h // hg, dim=2).transpose(1, 2)
    values = v.float().transpose(1, 2)
    gates = g.transpose(1, 2).cumsum(-1)
    betas = beta.transpose(1, 2)
    lower = torch.tril(
        (keys @ keys.transpose(-1, -2))
        * betas[..., None]
        * (gates[..., :, None] - gates[..., None, :]).exp(),
        diagonal=-1,
    )
    identity = torch.eye(64, device="cuda").expand(1, h, 64, 64)
    inverse = torch.linalg.solve_triangular(identity + lower, identity, upper=False)
    # Match the final BF16 WY GEMM operand conversions after the FP32 solve.
    inverse = inverse.bfloat16().float()
    w_ref = inverse @ (keys * (betas * gates.exp())[..., None]).bfloat16().float()
    u_ref = inverse @ (values * betas[..., None]).bfloat16().float()
    gc_ref = gates * math.log2(math.e) if use_exp2 else gates
    torch.testing.assert_close(gc, gc_ref, atol=1e-6, rtol=1e-5)
    for name, actual, expected in (("w", w, w_ref), ("u", u, u_ref)):
        relative_rmse = (
            (actual.float() - expected).square().mean() / expected.square().mean()
        ).sqrt()
        assert (
            relative_rmse < 0.003
        ), f"{name} relative RMS error: {relative_rmse.item()}"
