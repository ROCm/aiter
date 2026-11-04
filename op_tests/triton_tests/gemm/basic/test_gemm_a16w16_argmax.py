# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
    gemm_a16w16,
    gemm_a16w16_argmax_keys,
    gemm_a16w16_argmax_units,
)
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon kernels"
)

N, K = 62080, 2048  # a Qwen3.5 LM head shard at TP4


def _weights(case):
    gen = torch.Generator(device="cuda").manual_seed(7)
    w = torch.randn((N, K), generator=gen, dtype=torch.float32, device="cuda")
    w = (w * 0.02).to(torch.bfloat16)
    if case == "ties":  # equal maxima in other tiles and at both ends
        w[[3, 64, 65, 200, 31000, N - 81, N - 1]] = w[40000] * 10
    return w


def _inputs(m, case):
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn((m, K), generator=gen, dtype=torch.float32, device="cuda")
    if case == "zeros":  # every logit 0
        x = torch.zeros_like(x)
    if case == "coarse":  # many equal logits
        x = (x * 2).round() / 2
    return x.to(torch.bfloat16)


def _unpack(units, world, m):
    u = units.view(world, m, 8).float().to(torch.int64)
    words = [
        u[..., 4 * i] | u[..., 4 * i + 1] << 8 | u[..., 4 * i + 2] << 16 for i in (0, 1)
    ]
    words = [w | u[..., 4 * i + 3] << 24 for i, w in enumerate(words)]
    val = words[0].to(torch.int32).view(torch.float32)
    return val, words[1].to(torch.int32)


@pytest.mark.parametrize("m", [1, 5, 16, 33, 64, 80, 300])
@pytest.mark.parametrize("case", ["random", "ties", "zeros", "coarse"])
def test_gemm_a16w16_argmax_units(m, case):
    """The pack image is torch.argmax of the bf16 logits gemm_a16w16 stores (the
    first maximum), with that maximum, in this rank's slot and 0 in the others."""
    w, x = _weights(case), _inputs(m, case)
    logits = gemm_a16w16(x, w)
    ref_val = logits.float().max(dim=1).values
    ref_idx = torch.argmax(logits, dim=1)
    world, rank, start = 4, 2, 2 * N
    units = gemm_a16w16_argmax_units(x, w, start, rank, world)
    val, idx = _unpack(units, world, m)
    assert torch.equal(idx[rank], (ref_idx + start).to(torch.int32))
    assert torch.equal(val[rank], ref_val)  # -0.0 == +0.0
    others = [r for r in range(world) if r != rank]
    assert not units.view(world, m, 8)[others].any()
    keys = gemm_a16w16_argmax_keys(x, w)
    assert keys.dtype == torch.int64 and keys.shape[0] == m
