# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Checks which backend the LayerNorm entry points in aiter.ops.norm call.

The CK bindings and the Triton kernels are replaced by recorders, so module_norm
is not built, no kernel is launched, and the inputs are CPU tensors.
"""

import pytest
import torch

import aiter.ops.triton.normalization.norm as triton_norm
from aiter.ops import norm

B, S, N = 2, 3, 128
EPS = 1e-5
MARK = 3.0

ENTRY_POINTS = [
    "layer_norm",
    "layernorm2d_fwd",
    "layernorm2d_fwd_with_add",
    "layernorm2d_fwd_with_smoothquant",
    "layernorm2d_fwd_with_add_smoothquant",
]

TRITON_NAME = {
    "layer_norm": "layer_norm",
    "layernorm2d_fwd": "layer_norm",
    "layernorm2d_fwd_with_add": "layernorm2d_fwd_with_add",
    "layernorm2d_fwd_with_smoothquant": "layernorm2d_fwd_with_smoothquant",
    "layernorm2d_fwd_with_add_smoothquant": "layernorm2d_fwd_with_add_smoothquant",
}


class Recorder:
    def __init__(self):
        self.calls = []

    def make(self, name, returns=False):
        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if returns:
                return torch.full_like(args[0], MARK)
            args[0].fill_(MARK)

        return fn


def _forbid(name):
    def fn(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")

    return fn


def _make_inputs():
    torch.manual_seed(0)
    dt = torch.bfloat16
    return {
        "x": torch.randn(B, S, N, dtype=dt),
        "x_bias": torch.randn(N, dtype=dt),
        "weight": torch.randn(N, dtype=dt),
        "bias": torch.randn(N, dtype=dt),
        "residual": torch.randn(B, S, N, dtype=dt),
        "xscale": torch.ones(N, dtype=torch.float32),
        "yscale": torch.empty(B * S, 1, dtype=torch.float32),
    }


def _call(entry, t, x_bias):
    """Calls an entry point; returns (result, out) where out is the caller's output tensor or None."""
    x, w, b = t["x"], t["weight"], t["bias"]
    if entry in ("layer_norm", "layernorm2d_fwd"):
        return getattr(norm, entry)(x, w, b, EPS, x_bias), None
    if entry == "layernorm2d_fwd_with_add":
        out = torch.zeros_like(x)
        norm.layernorm2d_fwd_with_add(
            out, x, t["residual"], torch.zeros_like(x), w, b, EPS, x_bias
        )
        return None, out
    out = torch.zeros(B, S, N, dtype=torch.int8)
    if entry == "layernorm2d_fwd_with_smoothquant":
        norm.layernorm2d_fwd_with_smoothquant(
            out, x, t["xscale"], t["yscale"], w, b, EPS, x_bias
        )
    else:
        norm.layernorm2d_fwd_with_add_smoothquant(
            out,
            x,
            t["residual"],
            torch.zeros_like(x),
            t["xscale"],
            t["yscale"],
            w,
            b,
            EPS,
            x_bias,
        )
    return None, out


@pytest.fixture
def triton_recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(triton_norm, "layer_norm", rec.make("layer_norm", True))
    for name in set(TRITON_NAME.values()) - {"layer_norm"}:
        monkeypatch.setattr(triton_norm, name, rec.make(name))
    return rec


@pytest.mark.parametrize("use_x_bias", [False, True])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_ck_free_calls_triton(monkeypatch, triton_recorder, entry, use_x_bias):
    monkeypatch.setattr(norm, "ENABLE_CK", False)
    for name in ENTRY_POINTS:
        monkeypatch.setattr(norm, f"{name}_ck", _forbid(f"{name}_ck"))
    t = _make_inputs()
    x_bias = t["x_bias"] if use_x_bias else None

    ret, out = _call(entry, t, x_bias)

    assert [c[0] for c in triton_recorder.calls] == [TRITON_NAME[entry]]
    _, args, kwargs = triton_recorder.calls[0]
    # The Triton kernels take 2-D tensors and have no x_bias argument.
    rows = args[0] if entry in ("layer_norm", "layernorm2d_fwd") else args[1]
    want = t["x"] + x_bias if use_x_bias else t["x"]
    torch.testing.assert_close(rows, want.reshape(-1, N), rtol=0, atol=0)
    assert all(a.dim() <= 2 for a in args if isinstance(a, torch.Tensor))
    assert "x_bias" not in kwargs
    if out is None:
        assert ret.shape == t["x"].shape
        assert bool((ret == MARK).all())
    else:
        assert bool((out == MARK).all())


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_ck_calls_ck(monkeypatch, entry):
    monkeypatch.setattr(norm, "ENABLE_CK", True)
    for name in set(TRITON_NAME.values()):
        monkeypatch.setattr(triton_norm, name, _forbid(f"triton {name}"))
    rec = Recorder()
    for name in ENTRY_POINTS:
        monkeypatch.setattr(
            norm,
            f"{name}_ck",
            rec.make(name, returns=name in ("layer_norm", "layernorm2d_fwd")),
        )
    t = _make_inputs()

    _call(entry, t, t["x_bias"])

    assert [c[0] for c in rec.calls] == [entry]
    _, args, _ = rec.calls[0]
    assert args[-1] is t["x_bias"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
