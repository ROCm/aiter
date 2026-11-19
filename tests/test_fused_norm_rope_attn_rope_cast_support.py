# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Metadata capability tests; the existing op_test exercises real kernels."""

from types import SimpleNamespace

import pytest
import torch

from conftest import requires_built_aiter

requires_built_aiter()

from aiter import (  # noqa: E402 - requires_built_aiter must precede this import
    is_fused_norm_rope_attn_rope_cast_supported as supports,
)


@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("backend", [None, "opus"])
def test_offline_supported_shapes(monkeypatch, heads, backend):
    def no_device(*args, **kwargs):
        raise AssertionError("An explicit architecture must not query a device")

    monkeypatch.setattr(torch.cuda, "is_available", no_device)
    monkeypatch.setattr(torch.cuda, "get_device_properties", no_device)
    assert supports(heads, 512, backend=backend, arch="gfx950")


@pytest.mark.parametrize(
    "kwargs",
    [{"num_heads": h} for h in (0, 8, 15, 17, 48, 256)]
    + [{"head_dim": d} for d in (0, 256, 511, 513)]
    + [{"rope_dim": r} for r in (0, 32, 63, 65, 128)]
    + [
        {name: dtype}
        for name in ("q_dtype", "cos_dtype", "sin_dtype")
        for dtype in (torch.float16, torch.float32, torch.float8_e4m3fn)
    ]
    + [{"backend": "triton"}],
)
def test_unsupported_metadata(kwargs):
    assert not supports(arch="gfx950", **kwargs)


@pytest.mark.parametrize("arch", ["gfx942", "gfx1250", "gfx1250-strict", "unknown"])
def test_other_architectures(arch):
    assert not supports(arch=arch)


def test_no_gpu_never_queries_properties(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unavailable(*args):
        raise AssertionError("No GPU")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unavailable)
    assert not supports()


def test_runtime_uses_selected_device_not_build_arch(monkeypatch):
    monkeypatch.setenv("GPU_ARCHS", "gfx942;gfx1250")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    devices = []

    def properties(device):
        devices.append(device)
        arch = "gfx950:sramecc+:xnack-" if device == 1 else "gfx942:sramecc+"
        return SimpleNamespace(gcnArchName=arch)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert supports(device=1)
    assert not supports(device=0)
    assert devices == [1, 0]


def test_device_errors_are_not_hidden(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def broken(device):
        raise RuntimeError("device query failed")

    monkeypatch.setattr(torch.cuda, "get_device_properties", broken)
    with pytest.raises(RuntimeError, match="device query failed"):
        supports()
