# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD finalize fused with the grouped MoE's route + psum on gfx1250."""

import pytest
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.herd_topk import herd_finalize_candidates, herd_route_psum
from op_tests.test_flydsl_herd import _candidates, _logits_diff, _moe_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="the fused HERD route runs on the gfx1250 grouped MoE only",
)


@pytest.mark.parametrize("padding", [0, 5])
@pytest.mark.parametrize("tile_m", [16, 64])
@pytest.mark.parametrize("tokens", [16, 32, 64, 128])
@pytest.mark.parametrize("experts,topk", [(896, 16), (128, 4)])
def test_matches_finalize_and_route_psum(tokens, experts, topk, tile_m, padding):
    ids, values = _candidates(tokens, experts, topk)
    valid = tokens - padding
    num_valid = (
        torch.tensor([valid], dtype=torch.int32, device="cuda") if padding else None
    )
    ref_w, ref_ids = herd_finalize_candidates(values, ids, experts, 2.5, num_valid)
    w, kept, masked_m, rows, psum = herd_route_psum(
        values, ids, experts, tile_m, 2.5, num_valid
    )
    torch.testing.assert_close(kept, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(w, ref_w, rtol=0, atol=0)

    counts = torch.bincount(ref_ids[:valid].flatten().long(), minlength=experts)
    aligned = (counts + tile_m - 1) // tile_m * tile_m
    starts = torch.cumsum(aligned, 0) - aligned
    torch.testing.assert_close(masked_m, counts.int(), rtol=0, atol=0)
    torch.testing.assert_close(psum, (starts + counts).int(), rtol=0, atol=0)
    # Slots within an expert follow atomic order: check the layout, not rows.
    expert, row = kept[:valid].long(), rows[:valid].long()
    offset = row - starts[expert]
    assert bool(((offset >= 0) & (offset < counts[expert])).all())
    assert row.unique().numel() == row.numel()
    assert bool((rows[valid:] == -1).all())


def _herd_moe(tokens, experts=896, topk=16, dim=512):
    from aiter.fused_moe import fused_moe

    generator = torch.Generator(device="cuda").manual_seed(2713)
    w1, w2, kwargs = _moe_weights(experts, dim, generator)
    hidden = (
        torch.randn(
            tokens, dim, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.01
    )
    ids, values = _candidates(tokens, experts, topk)

    def run(rows=tokens, num_local_tokens=None):
        return fused_moe(
            hidden[:rows],
            w1,
            w2,
            values[:rows],
            ids[:rows],
            **kwargs,
            num_local_tokens=num_local_tokens,
            herd_topk=topk,
            herd_routed_scaling_factor=2.5,
        )

    def reference(rows=tokens):
        """The K-route MoE over the routes the HERD drop keeps."""
        weights, kept = herd_finalize_candidates(
            values[:rows], ids[:rows], experts, 2.5
        )
        return fused_moe(hidden[:rows], w1, w2, weights, kept, **kwargs)

    return run, reference


def _kernels(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return " ".join(e.name for e in prof.events() if e.device_type == DeviceType.CUDA)


@pytest.mark.parametrize("tokens", [16, 64, 128])
@pytest.mark.parametrize("experts,topk", [(896, 16), (128, 4)])
def test_fused_moe_routes_in_one_kernel(tokens, experts, topk):
    run, reference = _herd_moe(tokens, experts, topk)
    expected = reference()
    actual, kernels = run(), _kernels(run)
    assert _logits_diff(actual, expected) < 1e-6
    assert "moe_herd_route_psum" in kernels
    for gone in ("herd_finalize", "moe_route", "psum_remap"):
        assert gone not in kernels, f"{gone} still launched"


def test_fused_moe_padding_rows():
    run, reference = _herd_moe(64)
    expected = reference(rows=50)
    actual = run(num_local_tokens=torch.tensor([50], dtype=torch.int32, device="cuda"))
    assert torch.isfinite(actual).all()
    assert _logits_diff(actual[:50], expected) < 1e-6


def test_fused_moe_cuda_graph():
    run, _ = _herd_moe(64)
    expected = run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    graph.replay()
    torch.cuda.synchronize()
    assert _logits_diff(result, expected) < 1e-6


@pytest.mark.parametrize("experts,topk", [(896, 16), (128, 4)])
def test_rejects_rows_past_the_block(experts, topk):
    ids, values = _candidates(1025, experts, topk)
    with pytest.raises(ValueError, match="at most 1024 rows"):
        herd_route_psum(values, ids, experts, 32)
