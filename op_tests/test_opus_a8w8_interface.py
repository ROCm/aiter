# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks for the boundary between general A8W8 APIs and OPUS."""

from __future__ import annotations

import inspect

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as general_a8w8
from aiter.ops.opus import opus_gemm
from aiter.ops.opus import gemm_op_a8w8 as opus_a8w8


def test_general_a8w8_restores_required_scales_and_blockscale_dtypes():
    fake_parameters = inspect.signature(general_a8w8.gemm_a8w8_fake).parameters
    assert fake_parameters["x_scale"].default is inspect.Parameter.empty
    assert fake_parameters["w_scale"].default is inspect.Parameter.empty

    schema = str(torch.ops.aiter.gemm_a8w8.default._schema)
    assert "x_scale=None" not in schema
    assert "w_scale=None" not in schema

    XQ = torch.empty((1, 1))
    WQ = torch.empty((1, 1))
    with pytest.raises(RuntimeError, match="x_scale"):
        general_a8w8.gemm_a8w8(XQ, WQ)

    scale = torch.empty((1, 1))
    with pytest.raises(
        AssertionError,
        match="Output dtype=torch.float32 is currently not supported",
    ):
        general_a8w8.gemm_a8w8_blockscale(
            XQ,
            WQ,
            scale,
            scale,
            dtype=torch.float32,
        )

    assert not hasattr(general_a8w8, "_OPUS_A8W8_NOSCALE_KID")
    assert not hasattr(general_a8w8, "_OPUS_A8W8_BLOCKSCALE_KID")


def test_general_scaled_a8w8_keeps_legacy_backend_route(monkeypatch):
    calls = []
    result = torch.empty((2, 3), dtype=torch.bfloat16)

    def fake_ck(XQ, WQ, x_scale, w_scale, bias, dtype, splitK):
        calls.append((XQ, WQ, x_scale, w_scale, bias, dtype, splitK))
        return result

    monkeypatch.setattr(general_a8w8, "_ck_a8w8_supported", lambda: True)
    monkeypatch.setattr(general_a8w8, "gemm_a8w8_CK", fake_ck)

    XQ = torch.empty((2, 4), dtype=torch.int8)
    WQ = torch.empty((3, 4), dtype=torch.int8)
    x_scale = torch.empty((2, 1), dtype=torch.float32)
    w_scale = torch.empty((1, 3), dtype=torch.float32)
    actual = general_a8w8.gemm_a8w8(
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        splitK=3,
    )

    assert actual.data_ptr() == result.data_ptr()
    assert len(calls) == 1
    call = calls[0]
    assert call[0] is XQ
    assert call[1] is WQ
    assert call[2] is x_scale
    assert call[3] is w_scale
    assert call[4:] == (None, torch.bfloat16, 3)


@pytest.mark.parametrize(
    ("kid", "raw_name", "layout", "with_scale", "output_dtype"),
    [
        pytest.param(
            2,
            "_opus_gemm_a8w8_launch_raw",
            "plain",
            False,
            torch.float32,
            id="noscale",
        ),
        pytest.param(
            1,
            "_opus_gemm_a8w8_blockscale_launch_raw",
            "plain",
            True,
            torch.float32,
            id="blockscale",
        ),
        pytest.param(
            11000,
            "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
            "bpreshuffle",
            True,
            torch.bfloat16,
            id="blockscale-bpreshuffle",
        ),
    ],
)
def test_a8w8_opus_families_use_explicit_exact_kids(
    monkeypatch,
    kid,
    raw_name,
    layout,
    with_scale,
    output_dtype,
):
    calls = []
    raw_names = (
        "_opus_gemm_a8w8_launch_raw",
        "_opus_gemm_a8w8_blockscale_launch_raw",
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
    )

    for candidate in raw_names:

        def fake(*args, _name=candidate, **kwargs):
            calls.append((_name, args, kwargs))

        monkeypatch.setattr(opus_a8w8, candidate, fake)

    XQ = torch.empty((128, 256), dtype=dtypes.fp8)
    WQ = torch.empty((128, 256), dtype=dtypes.fp8)
    Y = torch.empty((128, 128), dtype=output_dtype)
    x_scale = torch.empty((128, 2), dtype=torch.float32)
    w_scale = torch.empty((1, 2), dtype=torch.float32)
    kwargs = {"kid": kid, "layout": layout}
    if with_scale:
        kwargs.update(x_scale=x_scale, w_scale=w_scale)

    assert opus_gemm(XQ, WQ, Y, **kwargs) is Y
    assert len(calls) == 1

    called_name, raw_args, raw_kwargs = calls[0]
    assert called_name == raw_name
    assert raw_kwargs == {}
    assert raw_args[-1] == kid

    raw_xq, raw_wq = raw_args[:2]
    assert raw_xq.shape == (1, *XQ.shape)
    assert raw_xq.data_ptr() == XQ.data_ptr()
    assert raw_wq.shape == (1, *WQ.shape)
    assert raw_wq.data_ptr() == WQ.data_ptr()

    if layout == "bpreshuffle":
        raw_x_scale, raw_w_scale, raw_y = raw_args[2:5]
    elif with_scale:
        raw_y, raw_x_scale, raw_w_scale = raw_args[2:5]
    else:
        assert len(raw_args) == 4
        raw_y = raw_args[2]

    assert raw_y.shape == (1, *Y.shape)
    assert raw_y.data_ptr() == Y.data_ptr()
    if with_scale:
        assert len(raw_args) == 6
        assert raw_x_scale is x_scale
        assert raw_w_scale is w_scale


def test_bpreshuffle_uses_opus_for_tuned_row(monkeypatch):
    opus_calls = []
    ck_calls = []
    config = {"libtype": "opus", "kernelId": 11000}

    def fake_opus_gemm(XQ, WQ, Y, **kwargs):
        opus_calls.append(kwargs)
        return Y

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, kernelName=""):
        ck_calls.append(kernelName)
        return Y

    import aiter.ops.opus as opus

    monkeypatch.setattr(general_a8w8, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(general_a8w8, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(general_a8w8, "get_CKGEMM_config", lambda *_args: config)
    monkeypatch.setattr(
        general_a8w8,
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        fake_ck,
    )
    monkeypatch.setattr(opus, "opus_gemm", fake_opus_gemm)

    XQ = torch.empty((2, 128), dtype=dtypes.fp8)
    WQ = torch.empty((128, 128), dtype=dtypes.fp8)
    x_scale = torch.empty((2, 1), dtype=torch.float32)
    w_scale = torch.empty((1, 1), dtype=torch.float32)

    bf16_result = general_a8w8.gemm_a8w8_blockscale_bpreshuffle(
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
    )
    assert bf16_result.dtype == torch.bfloat16
    assert len(opus_calls) == 1
    assert opus_calls[0]["kid"] == 11000
    assert opus_calls[0]["layout"] == "bpreshuffle"
    assert opus_calls[0]["x_scale"] is x_scale
    assert opus_calls[0]["w_scale"] is w_scale
    assert ck_calls == []
