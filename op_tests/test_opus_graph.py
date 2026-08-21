# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph, stream-ownership, and lifetime regressions for OPUS workspaces."""

from __future__ import annotations

import gc
import importlib
import weakref

import pytest
import torch

from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
from op_tests.opus_a16w16_test_utils import (
    _launch_a16w16_with_torch_workspace,
)


_GRAPH_CASES = {
    "gfx950": dict(kid=200, M=64, N=64, K=512, split_k=2),
    "gfx942": dict(kid=10200, M=128, N=128, K=512, split_k=2),
    "gfx1250": dict(kid=20000, M=16, N=32, K=512, split_k=2),
}


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


def _require_graph_case(arch: str):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    return dict(_GRAPH_CASES[arch])


def _load_raw_binding_without_workspace_launch(gemm) -> None:
    """Load the JIT module before capture without prewarming any workspace."""
    device = torch.device("cuda", torch.cuda.current_device())
    XQ = torch.empty((1, 1, 2), device=device, dtype=torch.bfloat16)
    WQ = torch.empty((1, 1, 2), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 1, 1), device=device, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="unknown kid -999"):
        gemm._launch_a16w16_backend(
            XQ, WQ, Y, None, None, -999, 0
        )


def _golden(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A.float() @ B.float().transpose(-1, -2)


def _make_gfx950_a8_case(seed: int):
    from aiter import dtypes

    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (256, 256)
    XQ = torch.randint(
        -2, 3, shape, generator=generator, device="cuda", dtype=torch.int32
    ).to(dtypes.fp8)
    WQ = torch.randint(
        -3, 4, shape, generator=generator, device="cuda", dtype=torch.int32
    ).to(dtypes.fp8)
    x_scale = torch.ones((256, 2), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((2, 2), device="cuda", dtype=torch.float32)
    return XQ, WQ, x_scale, w_scale


def _launch_gfx950_a8_family(family, opus, XQ, WQ, Y, x_scale, w_scale):
    if family == "noscale":
        return opus.opus_gemm(XQ, WQ, Y, kid=2)
    return opus.opus_gemm(
        XQ,
        WQ,
        Y,
        kid=1,
        x_scale=x_scale,
        w_scale=w_scale,
    )


@pytest.mark.parametrize("family", ["noscale", "blockscale"])
def test_gfx950_a8_logical_2d_gemm_adapter(family):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware; a skip is not a pass")
    opus = importlib.import_module("aiter.ops.opus")
    XQ, WQ, x_scale, w_scale = _make_gfx950_a8_case(0x2DA8)
    Y = torch.empty((256, 256), device="cuda", dtype=torch.float32)
    if family == "noscale":
        returned = opus.opus_gemm(XQ, WQ, Y, kid=2)
    else:
        returned = opus.opus_gemm(
            XQ,
            WQ,
            Y,
            kid=1,
            x_scale=x_scale,
            w_scale=w_scale,
        )
    torch.cuda.synchronize()
    assert returned is Y
    torch.testing.assert_close(Y, _golden(XQ, WQ), rtol=0, atol=0)


@pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
def test_graph_capture_replay_allocates_in_capture_without_prewarm(monkeypatch, arch):
    spec = _require_graph_case(arch)
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    opus = importlib.import_module("aiter.ops.opus")
    _load_raw_binding_without_workspace_launch(gemm)
    assert not hasattr(gemm, "opus_gemm_workspace_init")
    real_raw = gemm._launch_a16w16_backend
    allocation_ptrs = []

    def record_raw(XQ, WQ, Y, bias, workspace, kid, split_k):
        allocation_ptrs.append(workspace.data_ptr())
        return real_raw(XQ, WQ, Y, bias, workspace, kid, split_k)

    monkeypatch.setattr(
        gemm, "_launch_a16w16_backend", record_raw
    )
    A = torch.randn((1, spec["M"], spec["K"]), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, spec["N"], spec["K"]), device="cuda", dtype=torch.bfloat16)
    output = torch.empty(
        (1, spec["M"], spec["N"]), device="cuda", dtype=torch.bfloat16
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = opus.opus_bmm(
            A,
            B,
            output,
            kid=spec["kid"],
            split_k=spec["split_k"],
        )
    assert returned is output

    # The target shape was never launched eagerly. Its single workspace was
    # allocated by torch.empty while capture was active; replay is pure graph.
    assert len(allocation_ptrs) == 1
    for seed in (7, 11, 19):
        generator = torch.Generator(device=A.device).manual_seed(seed)
        A.copy_(
            torch.randn(A.shape, device=A.device, dtype=A.dtype, generator=generator)
        )
        B.copy_(
            torch.randn(B.shape, device=B.device, dtype=B.dtype, generator=generator)
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(), _golden(A, B), rtol=0.03, atol=0.5
        )
        assert len(allocation_ptrs) == 1


@pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
def test_two_streams_hold_distinct_call_scoped_workspaces(monkeypatch, arch):
    spec = _require_graph_case(arch)
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    opus = importlib.import_module("aiter.ops.opus")
    _load_raw_binding_without_workspace_launch(gemm)
    real_raw = gemm._launch_a16w16_backend
    held_workspaces = []

    def record_raw(XQ, WQ, Y, bias, workspace, kid, split_k):
        held_workspaces.append(workspace)
        return real_raw(XQ, WQ, Y, bias, workspace, kid, split_k)

    monkeypatch.setattr(
        gemm, "_launch_a16w16_backend", record_raw
    )
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    inputs = [
        (
            torch.randn(
                (1, spec["M"], spec["K"]), device="cuda", dtype=torch.bfloat16
            ),
            torch.randn(
                (1, spec["N"], spec["K"]), device="cuda", dtype=torch.bfloat16
            ),
        )
        for _ in streams
    ]
    outputs = [
        torch.empty(
            (1, spec["M"], spec["N"]), device="cuda", dtype=torch.bfloat16
        )
        for _ in streams
    ]
    producer = torch.cuda.current_stream()
    for stream, (A, B), output in zip(streams, inputs, outputs, strict=True):
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            returned = opus.opus_bmm(
                A,
                B,
                output,
                kid=spec["kid"],
                split_k=spec["split_k"],
            )
            assert returned is output
    for stream in streams:
        producer.wait_stream(stream)
    torch.cuda.synchronize()

    assert len(held_workspaces) == 2
    assert held_workspaces[0] is not held_workspaces[1]
    assert held_workspaces[0].data_ptr() != held_workspaces[1].data_ptr()
    for output, (A, B) in zip(outputs, inputs, strict=True):
        torch.testing.assert_close(
            output.float(), _golden(A, B), rtol=0.03, atol=0.5
        )


@pytest.mark.parametrize("family", ["noscale", "blockscale"])
def test_gfx950_a8_graph_capture_replay(family):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware; a skip is not a pass")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    _load_raw_binding_without_workspace_launch(gemm)
    opus = importlib.import_module("aiter.ops.opus")
    XQ, WQ, x_scale, w_scale = _make_gfx950_a8_case(0x950A8)
    Y = torch.empty((256, 256), device="cuda", dtype=torch.float32)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = _launch_gfx950_a8_family(
            family, opus, XQ, WQ, Y, x_scale, w_scale
        )
    assert returned is Y

    for seed in (7, 11, 19):
        new_XQ, new_WQ, _, _ = _make_gfx950_a8_case(seed)
        XQ.copy_(new_XQ)
        WQ.copy_(new_WQ)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(Y, _golden(XQ, WQ), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["noscale", "blockscale"])
def test_gfx950_a8_two_streams(family):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware; a skip is not a pass")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    _load_raw_binding_without_workspace_launch(gemm)
    opus = importlib.import_module("aiter.ops.opus")
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    cases = [_make_gfx950_a8_case(seed) for seed in (0x95051, 0x95052)]
    outputs = [
        torch.empty((256, 256), device="cuda", dtype=torch.float32)
        for _ in streams
    ]
    producer = torch.cuda.current_stream()

    for stream, case, Y in zip(streams, cases, outputs, strict=True):
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            returned = _launch_gfx950_a8_family(
                family, opus, *case[:2], Y, *case[2:]
            )
            assert returned is Y
    for stream in streams:
        producer.wait_stream(stream)
    torch.cuda.synchronize()

    for Y, (XQ, WQ, _x_scale, _w_scale) in zip(
        outputs, cases, strict=True
    ):
        torch.testing.assert_close(Y, _golden(XQ, WQ), rtol=0, atol=0)


def test_many_shapes_leave_no_python_workspace_tensor_cache():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    dead_refs = []

    for M, N in ((33, 17), (64, 64), (65, 97), (129, 33)):
        plan = _get_cached_a16w16_launch_plan(
            "gfx950",
            M,
            N,
            512,
            1,
            256,
            False,
            torch.bfloat16,
            torch.bfloat16,
            200,
            2,
        )
        XQ = torch.empty((1, M, 512), dtype=torch.bfloat16)
        WQ = torch.empty((1, N, 512), dtype=torch.bfloat16)
        Y = torch.empty((1, M, N), dtype=torch.bfloat16)

        def fake_raw(_XQ, _WQ, _Y, _bias, workspace, _kid, _split_k):
            dead_refs.append(weakref.ref(workspace))

        _launch_a16w16_with_torch_workspace(
            fake_raw, XQ, WQ, Y, None, plan
        )

    del XQ, WQ, Y
    gc.collect()
    assert dead_refs and all(reference() is None for reference in dead_refs)

    cached = [
        name for name, value in vars(gemm).items() if isinstance(value, torch.Tensor)
    ]
    assert cached == [], f"{gemm.__name__} caches Tensor globals: {cached}"


def test_explicit_scalar_plan_cache_does_not_retain_tensors(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    plan_module = importlib.import_module("aiter.ops.opus.launch_plan")
    plan_module._get_cached_a16w16_launch_plan.cache_clear()
    monkeypatch.setattr(
        gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256)
    )
    real_build = plan_module._build_a16w16_launch_plan
    builder_calls = 0
    dead_refs = []

    def counted_build(**kwargs):
        nonlocal builder_calls
        builder_calls += 1
        return real_build(**kwargs)

    monkeypatch.setattr(plan_module, "_build_a16w16_launch_plan", counted_build)

    def fake_raw(XQ, WQ, Y, _bias, workspace, _kid, _split_k):
        dead_refs.extend(weakref.ref(tensor) for tensor in (XQ, WQ, Y, workspace))

    monkeypatch.setattr(
        gemm, "_launch_a16w16_backend", fake_raw
    )
    for _ in range(2):
        XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
        WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
        Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
        workspace = torch.empty((2, 1, 64, 64), dtype=torch.float32)
        gemm._execute_a16w16(
            XQ, WQ, Y, kid=200, split_k=2, workspace=workspace
        )

    del XQ, WQ, Y, workspace
    gc.collect()
    assert builder_calls == 1
    assert plan_module._get_cached_a16w16_launch_plan.cache_info().maxsize == 256
    assert dead_refs and all(reference() is None for reference in dead_refs)
    plan_module._get_cached_a16w16_launch_plan.cache_clear()


def test_workspace_module_keeps_only_private_exact_kid_entry():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    opus = importlib.import_module("aiter.ops.opus")
    assert gemm.__all__ == []
    assert not hasattr(gemm, "_init_a16w16_workspace")
    assert not hasattr(gemm, "_launch_a16w16_with_torch_workspace")
    assert not hasattr(gemm, "opus_gemm_workspace_init")
    assert not hasattr(gemm, "gemm_a16w16_opus")
    assert opus.__all__ == ["opus_gemm", "opus_bmm"]
