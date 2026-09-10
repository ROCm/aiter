# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate the mutation contract of the allocating mHC entry points."""

import pytest
import torch

import aiter

OPS = ("mhc_pre", "mhc_fused_post_pre", "mhc_fused_post_pre_large_m")


@pytest.mark.parametrize("name", OPS)
def test_mhc_inputs_are_readonly(name):
    op = getattr(torch.ops.aiter, name).default
    assert all(
        arg.alias_info is None or not arg.alias_info.is_write
        for arg in op._schema.arguments
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("with_norm", [False, True])
def test_mhc_compile_contract(name, with_norm):
    # Exercise both fused decode and split post/pre prefill implementations.
    m = 1024 if name.endswith("large_m") else 65
    h = 1280
    device = "cuda"
    residual = torch.randn(m, 4, h, device=device, dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * h, device=device, dtype=torch.float32) * 0.01
    scale = torch.full((3,), 0.1, device=device)
    base = torch.zeros(24, device=device)
    kwargs = {"hc_post_mult_value": 2.0}
    if with_norm:
        kwargs["norm_weight"] = torch.ones(h, device=device, dtype=torch.bfloat16)
    if name == "mhc_pre":
        args = (residual, fn, scale, base)
    else:
        post, comb, x = aiter.mhc_pre(residual, fn, scale, base, **kwargs)
        args = (x, residual, post.squeeze(-1), comb, fn, scale, base)
    op = getattr(torch.ops.aiter, name).default
    # Checks actual writes/aliases against the schema and verifies compiled
    # execution against eager, including dynamic-shape functionalization.
    torch.library.opcheck(
        op,
        args,
        kwargs,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
