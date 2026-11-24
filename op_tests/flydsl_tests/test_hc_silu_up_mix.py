# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 fused scaled SiLU + HC up projection + gated stream mean (Qwen3.8)."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hc_mix_gfx1250 import (
    HC_SILU_UP_MIX_MAX_M,
    hc_interleave_up_weight,
    hc_silu_up_mix,
)

H, HC, R = 2560, 4, 320

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)


def _reference(d, w_up, normed):
    """The unfused chain's rounding: SiLU input/output, up output and mean in BF16."""
    x = (d[:, :R].float() / HC).bfloat16().float()
    gate = (x * torch.sigmoid(x)).bfloat16()
    up = (gate.float() @ w_up.float().T).bfloat16().float()
    streams = torch.sigmoid(up).view(-1, HC, H) * normed.float().view(-1, HC, H)
    return (streams.sum(1) / HC).bfloat16()


def _inputs(m, gen):
    d = (torch.randn(m, R + HC, generator=gen) * 4).bfloat16().cuda()
    w_up = (torch.randn(HC * H, R, generator=gen) * 0.05).bfloat16().cuda()
    normed = torch.randn(m, HC * H, generator=gen).bfloat16().cuda()
    return d, w_up, normed


def _check(out, ref):
    # The up GEMM accumulates in a different order than the FP32 reference, so a
    # few BF16 roundings of `up` land one ulp apart.
    mismatch = (out != ref).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2e} of the outputs differ"
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("m", [1, 3, 4, 8, 15, 16, 17, 24, 40, HC_SILU_UP_MIX_MAX_M])
def test_hc_silu_up_mix(m):
    gen = torch.Generator().manual_seed(m)
    d, w_up, normed = _inputs(m, gen)
    out = hc_silu_up_mix(d, hc_interleave_up_weight(w_up, HC), normed, HC)
    _check(out, _reference(d, w_up, normed))


def test_hc_silu_up_mix_graph_replay():
    """One captured launch stays correct as the inputs under it change."""
    m = 24
    gen = torch.Generator().manual_seed(0)
    d, w_up, normed = _inputs(m, gen)
    w_il = hc_interleave_up_weight(w_up, HC)
    hc_silu_up_mix(d, w_il, normed, HC)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = hc_silu_up_mix(d, w_il, normed, HC)
    for _ in range(3):
        nd, _, nn = _inputs(m, gen)
        d.copy_(nd)
        normed.copy_(nn)
        graph.replay()
        torch.cuda.synchronize()
        _check(out, _reference(d, w_up, normed))


def test_hc_silu_up_mix_rejects_large_m():
    gen = torch.Generator().manual_seed(1)
    d, w_up, normed = _inputs(HC_SILU_UP_MIX_MAX_M + 1, gen)
    with pytest.raises(ValueError):
        hc_silu_up_mix(d, hc_interleave_up_weight(w_up, HC), normed, HC)
