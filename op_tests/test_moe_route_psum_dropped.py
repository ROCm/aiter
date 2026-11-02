# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Direct integer contracts and public MoE dropped-route graph regression."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

W = Path(__file__).resolve().parents[1]
fm = importlib.import_module("aiter.fused_moe")
gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
kernel = importlib.import_module("aiter.ops.flydsl.kernels.moe_contiguous_psum")
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg


@pytest.fixture
def fixed():
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("fused grouped route is exercised on gfx1250")
    return kernel.build_moe_route_psum_fused_module()


def buffers(numel, E):
    sentinel = 1397
    tensors = [
        torch.full((n + 2,), sentinel, device="cuda", dtype=torch.int32)
        for n in (numel, E, E, E)
    ]
    return tensors, [x[1:-1] for x in tensors]


def launch(fn, ids, outs, E, max_m, tile):
    _run_compiled(
        fn,
        ptr_arg(ids),
        *(ptr_arg(x) for x in outs),
        ids.numel(),
        E,
        max_m,
        tile,
        torch.cuda.current_stream()
    )


def reference(ids, outs, E, tile, backing):
    cpu = ids.cpu().long()
    rows, counts, starts, psum = [x.cpu().long() for x in outs]
    valid = (cpu >= 0) & (cpu < E)
    hist = torch.bincount(cpu[valid], minlength=E)
    padded = (hist + tile - 1) // tile * tile
    refstarts = padded.cumsum(0) - padded
    assert torch.equal(counts, hist)
    assert torch.equal(starts, refstarts)
    assert torch.equal(psum, refstarts + hist)
    assert (rows[~valid] == DROPPED_ROUTE_ROW).all()
    for e in range(E):
        chosen = rows[valid & (cpu == e)].sort().values
        assert torch.equal(
            chosen, torch.arange(int(starts[e]), int(starts[e] + counts[e]))
        )
    for x in backing:
        assert int(x[0]) == 1397 and int(x[-1]) == 1397


@pytest.mark.parametrize("experts", [1, 17, 129, 512])
@pytest.mark.parametrize("numel", [0, 1, 35, 511, 512, 513, 2048, 4097])
@pytest.mark.parametrize("pattern", ["valid", "mixed", "all_drop"])
def test_route_reference_and_replay(experts, numel, pattern, fixed):
    ids = torch.arange(numel, device="cuda", dtype=torch.int32) % experts
    if pattern == "mixed":
        ids[::3] = -1
        ids[1::7] = experts
    if pattern == "all_drop":
        ids.fill_(-1)
    backing, out = buffers(numel, experts)
    max_m = max(1, numel)
    launch(fixed, ids, out, experts, max_m, 16)
    reference(ids, out, experts, 16, backing)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch(fixed, ids, out, experts, max_m, 16)
    torch.cuda.current_stream().wait_stream(stream)
    for change in ("empty", "live"):
        (
            ids.fill_(-1)
            if change == "empty"
            else ids.copy_(
                torch.arange(numel, device="cuda", dtype=torch.int32) % experts
            )
        )
        graph.replay()
        reference(ids, out, experts, 16, backing)


@pytest.mark.parametrize("fmt", ["a4w4", "a8w4"])
@pytest.mark.parametrize("valid", [0, 7, 16])
def test_public_moe_padding_matches_split(fmt, valid, monkeypatch, fixed):
    spec = importlib.util.spec_from_file_location(
        "_route_safety_moe_reference",
        W / "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py",
    )
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    try:
        spec.loader.exec_module(ref)
        captured = []
        public = ref.fused_moe

        def capture(*a, **kw):
            kw["quant_dtype_a"] = fm.dtypes.fp8 if fmt == "a8w4" else fm.dtypes.fp4x2
            captured.append((a, kw))
            return public(*a, **kw)

        ref.fused_moe = capture
        with monkeypatch.context() as m:
            m.setattr(gm, "_FUSED_ROUTE_PSUM_MAX_NUMEL", 0)
            expected, independent, _, _ = ref._run_grouped_via_fused_moe(
                experts=8,
                tokens=16,
                topk=2,
                model_dim=256,
                inter_dim=256,
                data_format=fmt,
                activation=fm.ActivationType.Swiglu,
                use_bias=True,
                seed=77,
                num_valid_tokens=max(valid, 1),
            )
        a, kw = captured[-1]
        if valid == 0:
            # The generic reference cannot reshape a zero-row input. All routes
            # have zero weight: the mathematical output is exactly zero.
            a[4].fill_(-1)
            a[3].zero_()
            kw["num_local_tokens"].zero_()
            independent = torch.zeros_like(expected)
            with monkeypatch.context() as split:
                split.setattr(gm, "_FUSED_ROUTE_PSUM_MAX_NUMEL", 0)
                expected = fm.fused_moe(*a, **kw)
            assert torch.equal(expected, independent)
        assert a[4].numel() <= gm._FUSED_ROUTE_PSUM_MAX_NUMEL
        route_calls = []
        route_fn = gm.fused_route_psum_remap

        def counted_route(*args, **kwargs):
            route_calls.append(True)
            return route_fn(*args, **kwargs)

        monkeypatch.setattr(gm, "fused_route_psum_remap", counted_route)
        got = fm.fused_moe(*a, **kw)
        assert route_calls, "Public call must exercise the fused route"
        assert torch.isfinite(got).all() and (got[valid:] == 0).all()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        if valid:
            assert (
                ref._logits_diff(got[:valid], independent[:valid]) < ref.LOGITS_DIFF_TOL
            )
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            out = fm.fused_moe(*a, **kw)
        torch.cuda.current_stream().wait_stream(s)
        for _ in range(2):
            a[0].neg_()
            graph.replay()
            torch.testing.assert_close(out, fm.fused_moe(*a, **kw), rtol=0, atol=0)
    finally:
        torch.set_default_device(default)


def test_valid_high_bit_masked_offsets(fixed):
    # The temporary masked row can exceed signed i32 even when the final
    # contiguous output row is small. Preserve the original unsigned math.
    experts, max_m, tile = 4, 1 << 30, 16
    ids = torch.tensor([0, 1, 2, 3, -1, 3, 2], device="cuda", dtype=torch.int32)
    backing, outs = buffers(ids.numel(), experts)
    launch(fixed, ids, outs, experts, max_m, tile)
    reference(ids, outs, experts, tile, backing)
