# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression coverage for gfx1250 no-quant RMSNorm dispatch.

Run with: pytest -q op_tests/test_rmsnorm_gfx1250_dispatch.py
Both wave32 optimized cases and unmodified width/dual-input fallbacks are checked.
"""
import pytest
import torch

from aiter.ops.rmsnorm import add_rmsnorm, rmsnorm
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("requires gfx1250")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("add", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize(
    "m,n",
    [
        (1, 128),
        (7, 128),
        (17, 6144),
        (255, 6144),
        (256, 6144),
        (257, 6144),
        (3, 256),
        (2, 8192),
    ],
)
def test_no_quant(dtype, gemma, add, padded, m, n):
    _gpu()
    torch.manual_seed(2026)

    def tensor(fill=None):
        storage = torch.empty(
            (m, n + (16 if padded else 0)), device="cuda", dtype=dtype
        )
        if fill is None:
            storage.normal_()
        else:
            storage.fill_(fill)
        return storage, storage[:, :n]

    _, x = tensor()
    _, residual = tensor()
    weight = torch.randn(n, device="cuda", dtype=dtype)
    xf = x.float() + residual.float() if add else x.float()
    ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    ref = ref * (weight.float() + (1 if gemma else 0))
    for path in ("atom", "generic"):
        backing, out = tensor(7)
        res_backing, res_out = tensor(7)
        if path == "atom":
            fused_qk_rmsnorm_group_quant(
                q=x,
                q_weight=weight,
                q_epsilon=1e-6,
                q_out_unquantized=out,
                q_out_scale=None,
                q_residual=residual if add else None,
                q_res_out=res_out if add else None,
                gemma_norm=gemma,
            )
        elif add:
            add_rmsnorm(out, x, residual, res_out, weight, 1e-6, gemma)
        else:
            rmsnorm(out, x, weight, 1e-6, gemma)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, rtol=0.01, atol=0.01)
        if add:
            torch.testing.assert_close(res_out, xf.to(dtype), rtol=0, atol=0)
        if padded:
            assert torch.all(backing[:, n:] == 7)
            assert torch.all(res_backing[:, n:] == 7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("m", [7, 1025])
def test_dual_input_fallback(dtype, gemma, m):
    _gpu()
    torch.manual_seed(2027)
    q = torch.randn(m, 128, device="cuda", dtype=dtype)
    k = torch.randn(m, 256, device="cuda", dtype=dtype)
    r = torch.randn_like(q)
    qw = torch.randn(128, device="cuda", dtype=dtype)
    kw = torch.randn(256, device="cuda", dtype=dtype)
    qo, ko, ro = torch.empty_like(q), torch.empty_like(k), torch.empty_like(q)
    fused_qk_rmsnorm_group_quant(
        q=q,
        q_weight=qw,
        q_epsilon=1e-6,
        q_out_unquantized=qo,
        q_residual=r,
        q_res_out=ro,
        k=k,
        k_weight=kw,
        k_epsilon=1e-5,
        k_out=ko,
        gemma_norm=gemma,
    )
    for value, weight, output, eps in (
        (q.float() + r.float(), qw, qo, 1e-6),
        (k.float(), kw, ko, 1e-5),
    ):
        ref = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        ref = ref * (weight.float() + (1 if gemma else 0))
        torch.testing.assert_close(output.float(), ref, rtol=0.01, atol=0.01)
    torch.testing.assert_close(ro, (q.float() + r.float()).to(dtype), rtol=0, atol=0)
