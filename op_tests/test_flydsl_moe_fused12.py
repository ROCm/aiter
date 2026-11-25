# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused grouped MoE stage 1 + stage 2 (AITER_MOE_FUSED12=1, gfx1250): one launch instead of two.

Correctness is bitwise against the two-launch path (flag off) for the whole grouped MoE output (route/quant, MoE1,
MoE2, gather-reduce), eagerly and under CUDA graph replay, at decode token counts with several routings and the
served tile configs; the per-m-tile handoff counters must be back to zero after every launch and no bounded spin may
have timed out (error slot).
"""

import os

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250", reason="gfx1250 only"
)

D, INTER, E, TOPK = 2048, 512, 256, 8
ROWS = [1, 3, 4, 7, 8, 12, 16, 20, 24, 32, 48, 64]


def _weights(seed=100):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u8 = torch.uint8
    w1 = torch.randint(0, 256, (E, 2 * INTER, D // 2), dtype=u8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (E, D, INTER // 2), dtype=u8, device="cuda", generator=g)
    s1 = torch.randint(118, 126, (E, 2 * INTER, D // 32), dtype=u8, device="cuda", generator=g)
    s2 = torch.randint(118, 126, (E, D, INTER // 32), dtype=u8, device="cuda", generator=g)
    return w1.view(dtypes.fp4x2), w2.view(dtypes.fp4x2), s1, s2


def _logits(m, kind, g):
    if kind == "uniform":
        return torch.randn(m, E, device="cuda", generator=g) * 2
    # rows in groups of 4 (one sequence's MTP verify rows) share a base logit
    base = torch.randn(-(-m // 4), E, device="cuda", generator=g).repeat_interleave(4, 0)[:m]
    return (base + 0.5 * torch.randn(m, E, device="cuda", generator=g)) * 2


def _bufs(m):
    return (
        torch.empty(m, TOPK, dtype=torch.float32, device="cuda"),
        torch.empty(m, TOPK, dtype=torch.int32, device="cuda"),
        torch.tensor([m], dtype=torch.int32, device="cuda"),
    )


def _moe(hidden, logits, weights, out, bufs):
    import aiter.ops.flydsl.grouped_moe_gfx1250 as gm

    w1, w2, s1, s2 = weights
    tw, ti, nvr = bufs
    gm.defer_topk_softmax(tw, ti, logits, True, nvr, num_fused_shared_experts=0)
    return fused_moe(
        hidden, w1, w2, tw, ti, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        w1_scale=s1, w2_scale=s2, num_local_tokens=nvr, gate_mode="interleave", output=out,
    )


@pytest.fixture(scope="module")
def weights():
    return _weights()


def _check_counters():
    from aiter.ops.flydsl.moe_fused12 import fused12_counters

    cnt = fused12_counters(torch.device("cuda"))
    torch.cuda.synchronize()
    assert int(cnt[-1]) == 0, "a stage-2 wait timed out"
    assert int((cnt[: 1 << 15] != 0).sum()) == 0, "handoff counters not reset"


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("kind", ["seq4", "uniform"])
def test_fused12_bitwise(m, kind, weights, monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(7 + m)
    logits = _logits(m, kind, g).to(torch.bfloat16)
    hidden = torch.randn(m, D, device="cuda", generator=g).to(torch.bfloat16)
    out = torch.empty(m, D, dtype=torch.bfloat16, device="cuda")
    bufs = _bufs(m)
    monkeypatch.setenv("AITER_MOE_FUSED12", "0")
    ref = _moe(hidden, logits, weights, out, bufs).clone()
    monkeypatch.setenv("AITER_MOE_FUSED12", "1")
    for _ in range(3):
        got = _moe(hidden, logits, weights, out, bufs)
        assert torch.equal(got, ref)
    _check_counters()
    # CUDA graph replay of the fused path
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _moe(hidden, logits, weights, out, bufs)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _moe(hidden, logits, weights, out, bufs)
    for _ in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
    _check_counters()


if __name__ == "__main__":
    os.environ.setdefault("AITER_MOE_FUSED12", "1")
    pytest.main([__file__, "-q"])
