# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""C ABI/ctypes tests for the private OPUS A16W16 exact-kid backend.

The ctypes raw remains private, while the unified public exact-kid entry uses
it after scalar validation and workspace planning. The original pybind raw
remains available as an A/B endpoint.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


def _gfx950_case(*, output_dtype: torch.dtype = torch.bfloat16):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires idle gfx950 hardware; a skip is not a pass")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    config = gemm._resolve_exact_a16w16_config(
        arch="gfx950",
        M=64,
        N=64,
        K=512,
        batch=1,
        cu_num=torch.cuda.get_device_properties(device).multi_processor_count,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=output_dtype,
        kid=200,
        split_k=2,
    )
    XQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    WQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), device=device, dtype=output_dtype)
    workspace = gemm._init_a16w16_workspace(config, XQ, Y)
    assert workspace is not None
    return gemm, config, XQ, WQ, Y, workspace


def _ctypes_launch(gemm, config, XQ, WQ, Y, workspace) -> None:
    gemm._opus_gemm_a16w16_launch_ctypes_raw(
        XQ,
        WQ,
        Y,
        None,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )


def _golden(XQ: torch.Tensor, WQ: torch.Tensor) -> torch.Tensor:
    return torch.bmm(XQ.float(), WQ.float().transpose(1, 2))


def test_ctypes_phase1_surface_is_private_production_backend():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    expected_raw = ("XQ", "WQ", "Y", "bias", "workspace", "kid", "split_k")
    assert tuple(
        inspect.signature(gemm._opus_gemm_a16w16_launch_ctypes_raw).parameters
    ) == expected_raw
    assert callable(gemm._opus_gemm_a16w16_launch_raw)
    assert "_opus_gemm_a16w16_launch_ctypes_raw" not in gemm.__all__
    assert not hasattr(gemm, "_experimental_opus_gemm_a16w16_launch_ctypes")

    python_source = (
        _ROOT / "aiter/ops/opus/gemm_op_a16w16.py"
    ).read_text(encoding="utf-8")
    assert "ctypes.CDLL(module_path)" in python_source
    assert "def _load_opus_a16w16_cabi(" in python_source
    assert "def _invoke_opus_a16w16_cabi(" in python_source
    assert 'ffi_type="ctypes"' not in python_source
    assert "ctypes_force_torch_exclude" not in python_source
    assert "_opus_gemm_a16w16_launch_ctypes_raw" in inspect.getsource(
        gemm._launch_a16w16
    )
    assert gemm.__all__ == []


def test_ctypes_cabi_reuses_checked_launcher_and_tls_error_bridge():
    header = (_ROOT / "csrc/opus_gemm/include/opus_gemm.h").read_text(
        encoding="utf-8"
    )
    implementation = (_ROOT / "csrc/opus_gemm/opus_gemm.cu").read_text(
        encoding="utf-8"
    )
    core = (_ROOT / "aiter/jit/core.py").read_text(encoding="utf-8")

    assert "AITER_C_ITFS int opus_gemm_a16w16_launch_cabi(" in header
    assert "AITER_CTYPES_ERROR_DEF" in implementation
    assert "AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(" in implementation
    assert "opus_gemm_a16w16_launch_cabi," in implementation
    assert (
        "const OpusCabiDeviceStreamGuard device_stream_guard(XQ->device_id, stream);"
        in implementation
    )
    assert "opus_gemm_a16w16_launch(" in implementation
    assert "ctypes_force_torch_exclude" not in core
    assert "force_torch_exclude" not in core

    cabi_body = implementation.split(
        "AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(\n"
        "    opus_gemm_a16w16_launch_cabi,",
        1,
    )[1].split("static void opus_check_a8_family_tensors", 1)[0]
    for forbidden in (
        "hipMalloc",
        "hipFree",
        "PreparedWorkspace",
        "prevalidated",
    ):
        assert forbidden not in cabi_body


def test_ctypes_raw_fake_registration_is_torch_compile_visible():
    from torch._subclasses.fake_tensor import FakeTensorMode

    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    mode = FakeTensorMode()
    with mode:
        XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
        WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
        Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
        workspace = torch.empty((2, 1, 64, 64), dtype=torch.float32)

        def raw_call(XQ, WQ, Y, workspace):
            gemm._opus_gemm_a16w16_launch_ctypes_raw(
                XQ, WQ, Y, None, workspace, 200, 2
            )
            return Y

        assert raw_call(XQ, WQ, Y, workspace) is Y
        # This test exercises Dynamo visibility of the registered fake
        # implementation. Inductor creates its own FakeTensorMode on PyTorch
        # 2.9, which conflicts with the explicit mode owning these inputs.
        compiled = torch.compile(raw_call, backend="eager", fullgraph=True)
        result = compiled(XQ, WQ, Y, workspace)
        assert result.fake_mode is mode


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_gfx950_ctypes_matches_pybind_and_golden(output_dtype):
    gemm, config, XQ, WQ, Y_ctypes, workspace = _gfx950_case(
        output_dtype=output_dtype
    )
    Y_pybind = torch.empty_like(Y_ctypes)
    gemm._opus_gemm_a16w16_launch_raw(
        XQ,
        WQ,
        Y_pybind,
        None,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )
    _ctypes_launch(gemm, config, XQ, WQ, Y_ctypes, workspace)
    torch.cuda.synchronize(Y_ctypes.device)

    expected = _golden(XQ, WQ)
    rtol, atol = ((0.03, 0.5) if output_dtype == torch.bfloat16 else (1e-3, 0.05))
    torch.testing.assert_close(Y_pybind.float(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(Y_ctypes.float(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(Y_ctypes, Y_pybind, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "requires a workspace tensor"),
        ("dtype", "workspace dtype must be"),
        ("short", "workspace capacity"),
        ("noncontiguous", "workspace must be contiguous"),
        ("alignment", "workspace address must be aligned"),
    ],
)
def test_gfx950_ctypes_errors_cross_cabi_safely(failure, message):
    gemm, config, XQ, WQ, Y, allocated = _gfx950_case()
    # The local adapter intentionally lets the existing pybind wrapper own the
    # first lazy JIT build, then uses the C ABI. Prime with a valid launch so
    # this test exercises exception transport across the C ABI itself even
    # when selected in isolation.
    _ctypes_launch(gemm, config, XQ, WQ, Y, allocated)
    if failure == "missing":
        workspace = None
    elif failure == "dtype":
        workspace = torch.empty(
            allocated.numel(), device=Y.device, dtype=torch.bfloat16
        )
    elif failure == "short":
        workspace = torch.empty(
            allocated.numel() - 1, device=Y.device, dtype=allocated.dtype
        )
    elif failure == "noncontiguous":
        workspace = torch.empty(
            (allocated.numel(), 2), device=Y.device, dtype=allocated.dtype
        )[:, 0]
        assert not workspace.is_contiguous()
    else:
        workspace = torch.empty(
            allocated.numel() + 1, device=Y.device, dtype=allocated.dtype
        )[1:]
        assert workspace.data_ptr() % 16 != 0

    with pytest.raises(RuntimeError) as error:
        _ctypes_launch(gemm, config, XQ, WQ, Y, workspace)
    text = str(error.value)
    assert "opus_gemm_a16w16_launch_cabi failed:" in text
    assert message in text


def test_gfx950_ctypes_uses_live_nondefault_stream():
    gemm, config, XQ, WQ, Y, workspace = _gfx950_case()
    stream = torch.cuda.Stream(device=Y.device)
    producer = torch.cuda.current_stream(Y.device)
    stream.wait_stream(producer)
    with torch.cuda.stream(stream):
        _ctypes_launch(gemm, config, XQ, WQ, Y, workspace)
    producer.wait_stream(stream)
    torch.cuda.synchronize(Y.device)
    torch.testing.assert_close(Y.float(), _golden(XQ, WQ), rtol=0.03, atol=0.5)


def test_gfx950_ctypes_graph_capture_and_replay():
    gemm, config, XQ, WQ, Y, workspace = _gfx950_case()
    # Build/load and validate once before capture; capture itself still invokes
    # the exact same raw C ABI with the graph's live stream.
    _ctypes_launch(gemm, config, XQ, WQ, Y, workspace)
    torch.cuda.synchronize(Y.device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ctypes_launch(gemm, config, XQ, WQ, Y, workspace)

    for seed in (17, 29):
        generator = torch.Generator(device=Y.device).manual_seed(seed)
        XQ.copy_(
            torch.randn(XQ.shape, device=Y.device, dtype=XQ.dtype, generator=generator)
        )
        WQ.copy_(
            torch.randn(WQ.shape, device=Y.device, dtype=WQ.dtype, generator=generator)
        )
        graph.replay()
        torch.cuda.synchronize(Y.device)
        torch.testing.assert_close(
            Y.float(), _golden(XQ, WQ), rtol=0.03, atol=0.5
        )


def test_gfx950_ctypes_two_streams_keep_workspace_ownership_external():
    cases = [_gfx950_case(), _gfx950_case()]
    streams = [torch.cuda.Stream(device=case[4].device) for case in cases]
    producer = torch.cuda.current_stream(cases[0][4].device)

    for stream, case in zip(streams, cases, strict=True):
        gemm, config, XQ, WQ, Y, workspace = case
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            _ctypes_launch(gemm, config, XQ, WQ, Y, workspace)

    for stream in streams:
        producer.wait_stream(stream)
    torch.cuda.synchronize(cases[0][4].device)

    workspaces = [case[5] for case in cases]
    assert workspaces[0] is not workspaces[1]
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()
    for _gemm, _config, XQ, WQ, Y, _workspace in cases:
        torch.testing.assert_close(
            Y.float(), _golden(XQ, WQ), rtol=0.03, atol=0.5
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two gfx950 devices for the C ABI device guard",
)
def test_gfx950_ctypes_switches_and_restores_current_device():
    initial = torch.cuda.current_device()
    target = (initial + 1) % torch.cuda.device_count()
    target_arch = str(
        getattr(torch.cuda.get_device_properties(target), "gcnArchName", "")
    ).split(":", 1)[0].lower()
    if _runtime_arch() != "gfx950" or target_arch != "gfx950":
        pytest.skip("requires two gfx950 devices; a skip is not a pass")

    device = torch.device("cuda", target)
    XQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    WQ = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), device=device, dtype=torch.bfloat16)
    workspace = torch.empty(
        (2, 1, 64, 64), device=device, dtype=torch.float32
    )

    opus = importlib.import_module("aiter.ops.opus")
    opus.opus_gemm(
        XQ, WQ, Y, kid=200, split_k=2, workspace=workspace
    )
    assert torch.cuda.current_device() == initial
    torch.cuda.synchronize(device)
    torch.testing.assert_close(Y.float(), _golden(XQ, WQ), rtol=0.03, atol=0.5)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two devices for the C ABI mixed-device rejection",
)
def test_gfx950_ctypes_rejects_mixed_input_devices_before_launch():
    gemm, config, XQ, valid_WQ, Y, workspace = _gfx950_case()
    _ctypes_launch(gemm, config, XQ, valid_WQ, Y, workspace)
    other_index = (XQ.device.index + 1) % torch.cuda.device_count()
    other_arch = str(
        getattr(torch.cuda.get_device_properties(other_index), "gcnArchName", "")
    ).split(":", 1)[0].lower()
    if other_arch != "gfx950":
        pytest.skip("requires a second gfx950 device; a skip is not a pass")
    WQ = torch.randn(
        (1, 64, 512),
        device=torch.device("cuda", other_index),
        dtype=torch.bfloat16,
    )

    with pytest.raises(RuntimeError, match="XQ/WQ/Y device ids must match"):
        _ctypes_launch(gemm, config, XQ, WQ, Y, workspace)
