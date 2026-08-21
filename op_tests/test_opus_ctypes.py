# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Focused regressions for the private A16W16 C ABI backend."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan


_ROOT = Path(__file__).resolve().parents[1]


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


def _gfx950_case(output_dtype=torch.bfloat16):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    plan = _get_cached_a16w16_launch_plan(
        "gfx950",
        64,
        64,
        512,
        1,
        torch.cuda.get_device_properties(device).multi_processor_count,
        False,
        torch.bfloat16,
        output_dtype,
        200,
        2,
    )
    XQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    WQ = torch.randn_like(XQ)
    Y = torch.empty((1, 64, 64), device=device, dtype=output_dtype)
    assert plan.workspace_spec is not None
    workspace = torch.empty(
        plan.workspace_spec.shape,
        device=device,
        dtype=plan.workspace_spec.dtype,
    )
    return gemm, plan, XQ, WQ, Y, workspace


def _launch(case, workspace=None):
    gemm, plan, XQ, WQ, Y, allocated = case
    gemm._launch_a16w16_backend(
        XQ,
        WQ,
        Y,
        None,
        allocated if workspace is None else workspace,
        plan.resolved_kid,
        plan.abi_split_k,
    )


def _golden(XQ, WQ):
    return torch.bmm(XQ.float(), WQ.float().transpose(1, 2))


def test_cabi_is_one_private_checked_backend_with_error_bridge():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    header = (_ROOT / "csrc/opus_gemm/include/opus_gemm.h").read_text()
    implementation = (_ROOT / "csrc/opus_gemm/opus_gemm.cu").read_text()
    python_source = (_ROOT / "aiter/ops/opus/gemm_op_a16w16.py").read_text()

    assert tuple(inspect.signature(gemm._launch_a16w16_backend).parameters) == (
        "XQ",
        "WQ",
        "Y",
        "bias",
        "workspace",
        "kid",
        "split_k",
    )
    assert gemm.__all__ == []
    assert "AITER_C_ITFS int opus_gemm_a16w16_launch_cabi(" in header
    assert "AITER_CTYPES_ERROR_DEF" in implementation
    assert "AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(" in implementation
    assert "opus_gemm_a16w16_launch_cabi," in implementation
    assert "const OpusCabiDeviceStreamGuard device_stream_guard" in implementation
    assert "ctypes.CDLL(module_path)" in python_source
    assert "def _invoke_opus_a16w16_cabi(" in python_source

    cabi_body = implementation.split(
        "AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(\n"
        "    opus_gemm_a16w16_launch_cabi,",
        1,
    )[1].split("static void opus_check_a8_family_tensors", 1)[0]
    assert "hipMalloc" not in cabi_body
    assert "hipFree" not in cabi_body


def test_backend_fake_is_torch_compile_visible():
    from torch._subclasses.fake_tensor import FakeTensorMode

    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    mode = FakeTensorMode()
    with mode:
        XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
        WQ = torch.empty_like(XQ)
        Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
        workspace = torch.empty((2, 1, 64, 64), dtype=torch.float32)

        def call(x, w, y, ws):
            gemm._launch_a16w16_backend(x, w, y, None, ws, 200, 2)
            return y

        result = torch.compile(call, backend="eager", fullgraph=True)(
            XQ, WQ, Y, workspace
        )
        assert result.fake_mode is mode


def test_first_launch_primes_pybind_on_the_input_device(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    launch = inspect.unwrap(gemm._launch_a16w16_backend)
    target = torch.device("cuda", 1)
    events = []

    class DeviceGuard:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            events.append(("enter", self.device))

        def __exit__(self, *_args):
            events.append(("exit", self.device))

    monkeypatch.setattr(torch.cuda, "device", DeviceGuard)
    monkeypatch.setattr(
        gemm, "_opus_gemm_a16w16_launch_raw", lambda *args: events.append("launch")
    )
    monkeypatch.setattr(
        gemm, "_load_opus_a16w16_cabi", lambda: events.append("load")
    )
    monkeypatch.setattr(gemm, "_opus_a16w16_cabi_primed", False)
    launch(
        SimpleNamespace(device=target), object(), object(), None, object(), 200, 2
    )

    assert events == [("enter", target), "launch", "load", ("exit", target)]
    assert gemm._opus_a16w16_cabi_primed is True


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_gfx950_cabi_matches_pybind_and_torch(output_dtype):
    case = _gfx950_case(output_dtype)
    gemm, plan, XQ, WQ, Y_cabi, workspace = case
    Y_pybind = torch.empty_like(Y_cabi)
    gemm._opus_gemm_a16w16_launch_raw(
        XQ, WQ, Y_pybind, None, workspace, plan.resolved_kid, plan.abi_split_k
    )
    _launch(case)
    torch.cuda.synchronize(Y_cabi.device)

    rtol, atol = ((0.03, 0.5) if output_dtype == torch.bfloat16 else (1e-3, 0.05))
    expected = _golden(XQ, WQ)
    torch.testing.assert_close(Y_cabi.float(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(Y_cabi, Y_pybind, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["dtype", "capacity"])
def test_gfx950_cabi_transports_workspace_errors(failure):
    case = _gfx950_case()
    _launch(case)
    allocated = case[-1]
    workspace = (
        torch.empty(
            allocated.numel(), device=allocated.device, dtype=torch.bfloat16
        )
        if failure == "dtype"
        else torch.empty(
            allocated.numel() - 1,
            device=allocated.device,
            dtype=allocated.dtype,
        )
    )

    with pytest.raises(RuntimeError, match="opus_gemm_a16w16_launch_cabi failed"):
        _launch(case, workspace)


def test_gfx950_cabi_graph_capture_and_replay():
    case = _gfx950_case()
    _launch(case)
    torch.cuda.synchronize(case[4].device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(case)

    gemm, plan, XQ, WQ, Y, workspace = case
    del gemm, plan, workspace
    for seed in (17, 29):
        generator = torch.Generator(device=Y.device).manual_seed(seed)
        XQ.copy_(torch.randn(XQ.shape, device=Y.device, dtype=XQ.dtype, generator=generator))
        WQ.copy_(torch.randn(WQ.shape, device=Y.device, dtype=WQ.dtype, generator=generator))
        graph.replay()
        torch.cuda.synchronize(Y.device)
        torch.testing.assert_close(Y.float(), _golden(XQ, WQ), rtol=0.03, atol=0.5)
