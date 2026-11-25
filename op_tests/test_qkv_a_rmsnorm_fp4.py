# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""qkv_a_rmsnorm_fp4: the widths and arch `qkv_a_rmsnorm_fp4_supported` reports,
and on a supported GPU the launch's bytes against rmsnorm followed by the 32x4
MXFP4 quant."""

import pytest
import torch

import aiter.ops.rmsnorm as rn
from aiter import dtypes
from aiter.ops.quant import per_1x32_mx_quant_hip


@pytest.mark.parametrize(
    "arch, q_dim, kv_dim, served",
    [
        ("gfx1250", 1536, 512, True),
        ("gfx1250-strict", 1536, 512, True),
        ("gfx1250", 1152, 504, True),
        ("gfx1250", 1408, 0, True),
        ("gfx1250", 1024, 512, False),
        ("gfx1250", 1025, 512, False),
        ("gfx1250", 1600, 512, False),
        ("gfx1250", 1664, 512, False),
        ("gfx1250", 1536, 520, False),  # a multiple of 8 above 512
        ("gfx1250", 1536, 500, False),  # not a multiple of 8
        ("gfx950", 1536, 512, False),
        ("gfx942", 1536, 512, False),
    ],
)
def test_supported_widths(monkeypatch, arch, q_dim, kv_dim, served):
    monkeypatch.setattr(rn, "get_gfx_runtime", lambda: arch)
    assert rn.qkv_a_rmsnorm_fp4_supported(q_dim, kv_dim) is served


@pytest.mark.skipif(
    not torch.cuda.is_available() or not rn.qkv_a_rmsnorm_fp4_supported(1536, 512),
    reason="qkv_a_rmsnorm_fp4 runs on the gfx1250 family",
)
@pytest.mark.parametrize("m", [1, 31, 32, 33, 257])
@pytest.mark.parametrize("q_dim, kv_dim", [(1536, 512), (1152, 504)])
def test_bytes_match_rmsnorm_and_quant(m, q_dim, kv_dim):
    torch.manual_seed(m)
    dev = "cuda"
    q_in = torch.randn(m, q_dim, dtype=torch.bfloat16, device=dev)
    kv_in = torch.randn(m, kv_dim, dtype=torch.bfloat16, device=dev)
    q_w = 1 + 0.1 * torch.randn(q_dim, dtype=torch.bfloat16, device=dev)
    kv_w = 1 + 0.1 * torch.randn(kv_dim, dtype=torch.bfloat16, device=dev)
    eps = 1e-6
    q_out = torch.empty(m, q_dim // 2, dtype=torch.uint8, device=dev)
    q_scale = torch.empty(
        (m + 31) // 32 * 32, q_dim // 32, dtype=torch.uint8, device=dev
    )
    kv_out = torch.empty_like(kv_in)
    rn.qkv_a_rmsnorm_fp4(q_out, q_scale, kv_out, q_in, kv_in, q_w, kv_w, eps, eps)

    q_ref = torch.empty_like(q_in)
    kv_ref = torch.empty_like(kv_in)
    rn.rmsnorm(q_ref, q_in, q_w, eps)
    rn.rmsnorm(kv_ref, kv_in, kv_w, eps)
    x_ref, s_ref = per_1x32_mx_quant_hip(
        q_ref, quant_dtype=dtypes.fp4x2, scale_layout_m32k4=True
    )
    assert torch.equal(kv_out, kv_ref)
    assert torch.equal(q_out, x_ref.view(torch.uint8))
    assert torch.equal(q_scale, s_ref.view(torch.uint8).reshape(q_scale.shape))
