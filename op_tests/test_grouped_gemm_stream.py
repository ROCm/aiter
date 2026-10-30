# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Default stream handle selection must preserve explicit streams and ordering."""

import ctypes
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

gm = importlib.import_module("aiter.ops.flydsl.grouped_gemm_mxfp4")


def _handle(value):
    from flydsl.expr import Stream

    stream = value if isinstance(value, Stream) else Stream(value)
    storage = ctypes.c_void_p()
    _, fill = stream.__c_abi_spec__()[0]
    fill(stream, storage)
    return storage.value or 0


def test_raw_handle_is_queried_each_time(monkeypatch):
    from flydsl.expr import Stream

    state = {"device": 0, "handles": [101, 202]}
    seen = []

    def raw(device):
        seen.append(device)
        return state["handles"][device]

    monkeypatch.setattr(gm, "_get_current_raw_stream", raw)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["device"])
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *a, **kw: pytest.fail("constructed a torch Stream on the fast path"),
    )
    first = gm._current_flydsl_stream()
    assert isinstance(first, Stream) and _handle(first) == 101
    state["handles"][0] = 303
    assert _handle(gm._current_flydsl_stream()) == 303
    state["device"] = 1
    assert _handle(gm._current_flydsl_stream()) == 202
    assert seen == [0, 0, 1]
    assert _handle(first) == 101  # No mutable Stream object reused across calls.


def test_missing_private_api_uses_public_fallback(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(gm, "_get_current_raw_stream", None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: sentinel)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("fallback must delegate device selection to torch"),
    )
    assert gm._current_flydsl_stream() is sentinel


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("grouped GEMM requires gfx1250")


def test_device_and_thread_local_handles():
    _gpu()
    # Handle selection itself is device independent. Do not reuse a compiled
    # FlyDSL executable on another device; that is a separate runtime concern.
    for device in range(min(torch.cuda.device_count(), 2)):
        with torch.cuda.device(device):
            streams = [torch.cuda.default_stream(), torch.cuda.Stream()]
            for stream in streams:
                with torch.cuda.stream(stream):
                    assert _handle(gm._current_flydsl_stream()) == stream.cuda_stream

    device = torch.cuda.current_device()
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]

    def worker(index):
        with torch.cuda.device(device), torch.cuda.stream(streams[index]):
            return all(
                _handle(gm._current_flydsl_stream()) == streams[index].cuda_stream
                for _ in range(100)
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(worker, range(2)))


@pytest.mark.parametrize("data_format", ["a4w4", "a8w4"])
@pytest.mark.parametrize("tokens", [1, 16, 257])
def test_real_moe_stream_order_and_graph(data_format, tokens, monkeypatch):
    _gpu()
    fm = importlib.import_module("aiter.fused_moe")
    path = Path(__file__).parent / "flydsl_tests" / "test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_stream_grouped_reference", path)
    reference = importlib.util.module_from_spec(spec)
    default_device = torch.get_default_device()
    try:
        spec.loader.exec_module(reference)
        inputs = []
        original_public = reference.fused_moe

        def capture(*args, **kwargs):
            kwargs["quant_dtype_a"] = (
                fm.dtypes.fp8 if data_format == "a8w4" else fm.dtypes.fp4x2
            )
            inputs.append((args, kwargs))
            return original_public(*args, **kwargs)

        reference.fused_moe = capture
        actual, expected, _, _ = reference._run_grouped_via_fused_moe(
            experts=8,
            tokens=tokens,
            topk=2,
            model_dim=256,
            inter_dim=256,
            data_format=data_format,
            activation=fm.ActivationType.Swiglu,
            use_bias=True,
            seed=42,
            bench=False,
        )
        assert reference._logits_diff(actual, expected) < reference.LOGITS_DIFF_TOL
        args, kwargs = inputs[-1]
        new = gm.flydsl_grouped_gemm_a8w4_masked

        def old(*a, **kw):
            if kw.get("stream") is None:
                kw["stream"] = torch.cuda.current_stream()
            return new(*a, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(gm, "flydsl_grouped_gemm_a8w4_masked", old)
            before = fm.fused_moe(*args, **kwargs)
        assert torch.equal(before, actual)

        launches = []

        def record(*a, **kw):
            launches.append((a, kw))
            return new(*a, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(gm, "flydsl_grouped_gemm_a8w4_masked", record)
            fm.fused_moe(*args, **kwargs)
        a, kw = launches[0]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Explicit stream must be forwarded untouched, without querying the
            # default-stream helper. Compare against the original launch.
            old(*a, **{**kw, "stream": stream})
            explicit_expected = a[0].clone()
            with monkeypatch.context() as patch:
                patch.setattr(
                    gm,
                    "_current_flydsl_stream",
                    lambda: pytest.fail("explicit stream replaced"),
                )
                new(*a, **{**kw, "stream": stream})
            assert torch.equal(a[0], explicit_expected)
            fm.fused_moe(*args, **kwargs)  # Warm before capture.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                replayed = fm.fused_moe(*args, **kwargs)
            for _ in range(2):
                args[0].neg_()
                graph.replay()
                with monkeypatch.context() as patch:
                    patch.setattr(gm, "flydsl_grouped_gemm_a8w4_masked", old)
                    eager = fm.fused_moe(*args, **kwargs)
                assert torch.equal(replayed, eager)
        torch.cuda.current_stream().wait_stream(stream)
    finally:
        torch.set_default_device(default_device)
