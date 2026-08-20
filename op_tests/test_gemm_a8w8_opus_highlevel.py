# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib

import pytest
import torch


def _fp8_dtype() -> torch.dtype:
    from aiter import dtypes

    return dtypes.fp8


def _plain_problem():
    fp8 = _fp8_dtype()
    XQ = torch.empty((4, 256), dtype=fp8)
    WQ = torch.empty((128, 256), dtype=fp8)
    x_scale = torch.empty((4, 2), dtype=torch.float32)
    w_scale = torch.empty((1, 2), dtype=torch.float32)
    return XQ, WQ, x_scale, w_scale


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0].lower()


def test_gemm_a8w8_without_scales_routes_to_opus_noscale(monkeypatch):
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    opus = importlib.import_module("aiter.ops.opus")
    XQ, WQ, _, _ = _plain_problem()
    calls = []

    def fake_opus_gemm(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(opus, "opus_gemm", fake_opus_gemm)

    Y = gemm.gemm_a8w8(XQ, WQ, dtype=torch.float32)

    assert Y.shape == (4, 128)
    assert Y.dtype == torch.float32
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == (XQ, WQ)
    assert args[2].data_ptr() == Y.data_ptr()
    assert kwargs == {"kid": 2, "layout": "plain"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "requires dtype=torch.float32"),
        ({"bias": torch.empty(128), "dtype": torch.float32}, "does not support bias"),
        ({"dtype": torch.float32, "splitK": 1}, "does not support splitK"),
    ],
)
def test_gemm_a8w8_noscale_rejects_unsupported_options(kwargs, message):
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    XQ, WQ, _, _ = _plain_problem()

    with pytest.raises(ValueError, match=message):
        gemm.gemm_a8w8(XQ, WQ, **kwargs)


def test_gemm_a8w8_requires_scale_pair():
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    XQ, WQ, x_scale, _ = _plain_problem()

    with pytest.raises(ValueError, match="requires x_scale and w_scale together"):
        gemm.gemm_a8w8(XQ, WQ, x_scale=x_scale, dtype=torch.float32)


def test_scaled_gemm_a8w8_keeps_legacy_ck_route(monkeypatch):
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    XQ, WQ, x_scale, w_scale = _plain_problem()
    expected = torch.empty((4, 128), dtype=torch.bfloat16)
    calls = []

    def fake_ck(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(gemm, "_ck_a8w8_supported", lambda: True)
    monkeypatch.setattr(gemm, "gemm_a8w8_CK", fake_ck)

    Y = gemm.gemm_a8w8(XQ, WQ, x_scale, w_scale)

    assert Y.data_ptr() == expected.data_ptr()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == (XQ, WQ, x_scale, w_scale)
    assert kwargs == {}


def test_fp32_plain_blockscale_routes_to_opus(monkeypatch):
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    opus = importlib.import_module("aiter.ops.opus")
    XQ, WQ, x_scale, w_scale = _plain_problem()
    calls = []

    def fake_opus_gemm(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(opus, "opus_gemm", fake_opus_gemm)

    Y = gemm.gemm_a8w8_blockscale(
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype=torch.float32,
    )

    assert Y.shape == (4, 128)
    assert Y.dtype == torch.float32
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == (XQ, WQ)
    assert args[2].data_ptr() == Y.data_ptr()
    assert kwargs == {
        "kid": 1,
        "layout": "plain",
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def test_bf16_plain_blockscale_keeps_legacy_ck_route(monkeypatch):
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    XQ, WQ, x_scale, w_scale = _plain_problem()
    calls = []

    def fake_ck(*args, **kwargs):
        calls.append((args, kwargs))
        return args[4]

    monkeypatch.setattr(gemm, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(gemm, "get_CKGEMM_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gemm, "gemm_a8w8_blockscale_ck", fake_ck)

    Y = gemm.gemm_a8w8_blockscale(XQ, WQ, x_scale, w_scale)

    assert Y.dtype == torch.bfloat16
    assert len(calls) == 1
    assert calls[0][0][:4] == (XQ, WQ, x_scale, w_scale)


def test_fp32_blockscale_rejects_preshuffled_weight():
    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    XQ, WQ, x_scale, w_scale = _plain_problem()

    with pytest.raises(ValueError, match="OPUS plain-W layout"):
        gemm.gemm_a8w8_blockscale(
            XQ,
            WQ,
            x_scale,
            w_scale,
            dtype=torch.float32,
            isBpreshuffled=True,
        )


@pytest.mark.parametrize("family", ["noscale", "blockscale"])
def test_gfx950_highlevel_plain_opus_correctness(family):
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")

    gemm = importlib.import_module("aiter.ops.gemm_op_a8w8")
    fp8 = _fp8_dtype()
    generator = torch.Generator(device="cuda").manual_seed(0xA8F00D)
    XQ = torch.randint(
        -2,
        3,
        (256, 256),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    ).to(fp8)
    WQ = torch.randint(
        -3,
        4,
        (256, 256),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    ).to(fp8)
    golden = XQ.float() @ WQ.float().T

    if family == "noscale":
        Y = gemm.gemm_a8w8(XQ, WQ, dtype=torch.float32)
    else:
        x_scale = torch.ones((256, 2), device="cuda", dtype=torch.float32)
        w_scale = torch.ones((2, 2), device="cuda", dtype=torch.float32)
        Y = gemm.gemm_a8w8_blockscale(
            XQ,
            WQ,
            x_scale,
            w_scale,
            dtype=torch.float32,
        )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y, golden, rtol=0, atol=0)
