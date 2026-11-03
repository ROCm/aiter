# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 TP prefill routing without the standalone counter fill."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
fm = importlib.import_module("aiter.fused_moe")

EXPERTS = 129
TOPK = 5
TILE_M = 128


def require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 prefill route fusion")


def make_ids(tokens, seed=393):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + tokens)
    routed = torch.randint(
        EXPERTS - 1,
        (tokens, TOPK - 1),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    shared = torch.full(
        (tokens, 1),
        EXPERTS - 1,
        dtype=torch.int32,
        device="cuda",
    )
    return torch.cat((routed, shared), dim=1).contiguous()


def route_reference(ids):
    flat = ids.cpu().long().flatten()
    counts = torch.bincount(flat, minlength=EXPERTS)
    aligned = (counts + TILE_M - 1) // TILE_M * TILE_M
    starts = aligned.cumsum(0) - aligned
    seen = torch.zeros(EXPERTS, dtype=torch.int64)
    rows = torch.empty_like(flat)
    for route, expert in enumerate(flat.tolist()):
        rows[route] = starts[expert] + seen[expert]
        seen[expert] += 1
    return counts.int(), rows.view_as(ids).int(), (starts + counts).int()


def call_route(ids):
    max_m = ((ids.numel() + TILE_M - 1) // TILE_M) * TILE_M
    return gm.fused_route_fill_psum_remap(
        ids,
        EXPERTS,
        max_m,
        TILE_M,
    )


@pytest.mark.parametrize("tokens", [2517, 4536])
def test_prefill_route_exact_deterministic_and_two_kernels(tokens):
    require_gfx1250()
    ids = make_ids(tokens)
    expected = route_reference(ids)

    eager = []
    for _ in range(5):
        eager.append(tuple(t.clone() for t in call_route(ids)))
    torch.cuda.synchronize()
    for actual in eager:
        for got, want in zip(actual, expected):
            assert torch.equal(got.cpu(), want)
    for actual in eager[1:]:
        for got, first in zip(actual, eager[0]):
            assert torch.equal(got, first)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        call_route(ids)
        torch.cuda.synchronize()
    names = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert names == [
        "moe_route_lds_nofill",
        "moe_block_counts_psum_remap",
    ]

    changed = [make_ids(tokens, seed) for seed in (911, 1771, 3931)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_out = call_route(ids)
        for next_ids in changed:
            ids.copy_(next_ids)
            graph.replay()
            stream.synchronize()
            for got, want in zip(graph_out, route_reference(ids)):
                assert torch.equal(got.cpu(), want)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("tokens", [2517, 4536])
def test_public_moe_prefill_matches_split_and_graph(tokens, monkeypatch):
    require_gfx1250()
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_f7_moe_reference", path)
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    try:
        spec.loader.exec_module(ref)
        captured = []
        public = ref.fused_moe

        def capture(*args, **kwargs):
            kwargs["quant_dtype_a"] = fm.dtypes.fp8
            captured.append((args, kwargs))
            return public(*args, **kwargs)

        ref.fused_moe = capture
        with monkeypatch.context() as split:
            split.setattr(gm, "_FUSED_ROUTE_PSUM_MAX_NUMEL", 0)
            split.setattr(gm, "_FUSED_ROUTE_BLOCK_COUNTS_MAX_NUMEL", 0)
            baseline, reference, _, _ = ref._run_grouped_via_fused_moe(
                experts=EXPERTS,
                tokens=tokens,
                topk=TOPK,
                model_dim=256,
                inter_dim=256,
                data_format="a8w4",
                use_bias=True,
                seed=393000 + tokens,
            )
        assert ref._logits_diff(baseline, reference) < ref.LOGITS_DIFF_TOL
        args, kwargs = captured[-1]
        eager = [fm.fused_moe(*args, **kwargs) for _ in range(3)]
        assert all(torch.equal(out, baseline) for out in eager)
        assert all(torch.equal(out, eager[0]) for out in eager[1:])

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = fm.fused_moe(*args, **kwargs)
            for _ in range(3):
                graph.replay()
                stream.synchronize()
                assert torch.equal(output, baseline)
        torch.cuda.current_stream().wait_stream(stream)
    finally:
        torch.set_default_device(default)
