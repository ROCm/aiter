# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused TP routing scan: integer invariants and public MoE graph regression."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
fm = importlib.import_module("aiter.fused_moe")
kernel = importlib.import_module("aiter.ops.flydsl.kernels.moe_contiguous_psum")
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg


@pytest.fixture
def launcher():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 fused TP scan optimization")
    return kernel.build_moe_route_psum_fused_module()


def check(ids, outputs, backing, experts, tile):
    rows, counts, starts, ends = [t.cpu().long() for t in outputs]
    cpu_ids = ids.cpu().long()
    hist = torch.bincount(cpu_ids, minlength=experts)
    padded = (hist + tile - 1) // tile * tile
    expected_starts = padded.cumsum(0) - padded
    assert torch.equal(counts, hist)
    assert torch.equal(starts, expected_starts)
    assert torch.equal(ends, expected_starts + hist)
    for expert in range(experts):
        slots = rows[cpu_ids == expert].sort().values
        assert torch.equal(slots, torch.arange(int(starts[expert]), int(ends[expert])))
    for b in backing:
        assert (b[:2] == 713).all() and (b[-2:] == 713).all()


@pytest.mark.parametrize("experts", [1, 17, 129, 512])
@pytest.mark.parametrize("numel", [0, 1, 35, 511, 512, 513, 2048, 4097])
@pytest.mark.parametrize("tile", [16, 128])
def test_fused_scan_integer_reference_and_replay(experts, numel, tile, launcher):
    # Dropped IDs are a separate pre-existing contract issue (#7222). This
    # release test does not run undefined negative IDs on unpatched main.
    ids = torch.arange(numel, device="cuda", dtype=torch.int32) % experts
    backing = [
        torch.full((n + 4,), 713, device="cuda", dtype=torch.int32)
        for n in (numel, experts, experts, experts)
    ]
    outputs = [b[2:-2] for b in backing]

    def call():
        _run_compiled(
            launcher,
            ptr_arg(ids),
            *(ptr_arg(t) for t in outputs),
            numel,
            experts,
            max(1, numel),
            tile,
            torch.cuda.current_stream()
        )

    call()
    check(ids, outputs, backing, experts, tile)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        for hot in (True, False):
            if hot:
                ids.fill_(experts - 1)
            else:
                ids.copy_(
                    torch.arange(numel, device="cuda", dtype=torch.int32) % experts
                )
            graph.replay()
            check(ids, outputs, backing, experts, tile)
    torch.cuda.current_stream().wait_stream(stream)


def test_masked_offsets_use_unsigned_high_bit(launcher):
    experts, numel, max_m = 512, 257, 8388608
    assert 2**31 <= (experts - 1) * max_m
    assert (experts - 1) * max_m + numel < 2**32
    ids = torch.full((numel,), experts - 1, device="cuda", dtype=torch.int32)
    backing = [
        torch.full((n + 4,), 713, device="cuda", dtype=torch.int32)
        for n in (numel, experts, experts, experts)
    ]
    outputs = [b[2:-2] for b in backing]
    _run_compiled(
        launcher,
        ptr_arg(ids),
        *(ptr_arg(t) for t in outputs),
        numel,
        experts,
        max_m,
        16,
        torch.cuda.current_stream()
    )
    check(ids, outputs, backing, experts, 16)


@pytest.mark.parametrize("fmt", ["a4w4", "a8w4"])
@pytest.mark.parametrize("tokens", [1, 16, 256])
def test_public_moe_matches_split_and_graph(fmt, tokens, launcher, monkeypatch):
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_fused_scan_reference", path)
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    try:
        spec.loader.exec_module(ref)
        captured = []
        public = ref.fused_moe

        def capture(*args, **kwargs):
            kwargs["quant_dtype_a"] = (
                fm.dtypes.fp8 if fmt == "a8w4" else fm.dtypes.fp4x2
            )
            captured.append((args, kwargs))
            return public(*args, **kwargs)

        ref.fused_moe = capture
        with monkeypatch.context() as split:
            split.setattr(gm, "_FUSED_ROUTE_PSUM_MAX_NUMEL", 0)
            baseline, reference, _, _ = ref._run_grouped_via_fused_moe(
                experts=8,
                tokens=tokens,
                topk=2,
                model_dim=256,
                inter_dim=256,
                data_format=fmt,
                activation=fm.ActivationType.Swiglu,
                use_bias=True,
                seed=513,
            )
        assert ref._logits_diff(baseline, reference) < ref.LOGITS_DIFF_TOL
        args, kwargs = captured[-1]
        monkeypatch.setattr(gm, "_get_compiled_route_psum_fused", lambda: launcher)
        out = fm.fused_moe(*args, **kwargs)
        assert torch.equal(out, baseline)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = fm.fused_moe(*args, **kwargs)
            for _ in range(3):
                args[0].neg_()
                graph.replay()
                with monkeypatch.context() as split:
                    split.setattr(gm, "_FUSED_ROUTE_PSUM_MAX_NUMEL", 0)
                    expected = fm.fused_moe(*args, **kwargs)
                assert torch.equal(output, expected)
        torch.cuda.current_stream().wait_stream(stream)
    finally:
        torch.set_default_device(default)
