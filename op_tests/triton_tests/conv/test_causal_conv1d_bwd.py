# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test skeleton for causal conv1d backward (Kimi-K3, gfx950).

Placeholder. The backward kernel is not implemented yet, so every test here is
skipped. Requirements: https://github.com/ZhangDanyang-AMD/Lumen/issues/35

Kimi-K3 KDA short conv: depthwise weight ``[dim, 1, 4]`` with ``dim = 12288``
for each of q, k and v. The TP8 shard has ``dim = 1536`` per tensor.
"""

import pytest
import torch
import torch.nn.functional as F

KIMI_K3_WIDTH = 4
KIMI_K3_DIM = 12288
KIMI_K3_DIM_TP8 = 1536


def causal_conv1d_bwd_ref(x, weight, bias, dout, activation="silu"):
    """fp32 autograd reference for one sequence.

    x: (dim, seqlen), weight: (dim, width), bias: (dim,) or None,
    dout: (dim, seqlen). Returns (dx, dweight, dbias).
    """
    dim, seqlen = x.shape
    width = weight.shape[1]
    x32 = x.float().detach().requires_grad_(True)
    w32 = weight.float().detach().requires_grad_(True)
    b32 = None if bias is None else bias.float().detach().requires_grad_(True)
    out = F.conv1d(
        x32.unsqueeze(0), w32.unsqueeze(1), b32, padding=width - 1, groups=dim
    )[..., :seqlen]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    out.backward(dout.float().unsqueeze(0))
    return x32.grad, w32.grad, None if b32 is None else b32.grad


@pytest.mark.skip(reason="causal conv1d backward is not implemented yet")
@pytest.mark.parametrize("dim", [KIMI_K3_DIM, KIMI_K3_DIM_TP8])
@pytest.mark.parametrize("seqlen", [1, 8, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_causal_conv1d_bwd_kimi_k3(dim, seqlen, dtype):
    x = torch.randn((dim, seqlen), dtype=dtype, device="cuda")
    weight = torch.randn((dim, KIMI_K3_WIDTH), dtype=dtype, device="cuda")
    dout = torch.randn((dim, seqlen), dtype=dtype, device="cuda")
    dx_ref, dw_ref, _ = causal_conv1d_bwd_ref(x, weight, None, dout)
    assert dx_ref.shape == x.shape and dw_ref.shape == weight.shape
