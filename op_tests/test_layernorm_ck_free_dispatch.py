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

PUBLIC_SCHEMAS = {
    "layer_norm": (
        "aiter::layer_norm(Tensor(a0!) input, Tensor(a1!)? weight=None, "
        "Tensor(a2!)? bias=None, float epsilon=1.0000000000000001e-05, "
        "Tensor(a4!)? x_bias=None) -> Tensor"
    ),
    "layernorm2d_fwd": (
        "aiter::layernorm2d_fwd(Tensor(a0!) input, Tensor(a1!) weight, "
        "Tensor(a2!) bias, float epsilon=1.0000000000000001e-05, "
        "Tensor(a4!)? x_bias=None) -> Tensor"
    ),
    "layernorm2d_fwd_with_add": (
        "aiter::layernorm2d_fwd_with_add(Tensor(a0!) out, Tensor(a1!) input, "
        "Tensor(a2!) residual_in, Tensor(a3!) residual_out, Tensor(a4!) weight, "
        "Tensor(a5!) bias, float epsilon, Tensor(a7!)? x_bias=None) -> ()"
    ),
    "layernorm2d_fwd_with_smoothquant": (
        "aiter::layernorm2d_fwd_with_smoothquant(Tensor(a0!) out, "
        "Tensor(a1!) input, Tensor(a2!) xscale, Tensor(a3!) yscale, "
        "Tensor(a4!) weight, Tensor(a5!) bias, float epsilon, "
        "Tensor(a7!)? x_bias=None) -> ()"
    ),
    "layernorm2d_fwd_with_add_smoothquant": (
        "aiter::layernorm2d_fwd_with_add_smoothquant(Tensor(a0!) out, "
        "Tensor(a1!) input, Tensor(a2!) residual_in, Tensor(a3!) residual_out, "
        "Tensor(a4!) xscale, Tensor(a5!) yscale, Tensor(a6!) weight, "
        "Tensor(a7!) bias, float epsilon, Tensor(a9!)? x_bias=None) -> ()"
    ),
}

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


def _call(entry, t, x_bias, api=norm):
    """Calls an entry point; returns (result, out) where out is the caller's output tensor or None."""
    x, w, b = t["x"], t["weight"], t["bias"]
    if entry in ("layer_norm", "layernorm2d_fwd"):
        return getattr(api, entry)(x, w, b, EPS, x_bias), None
    if entry == "layernorm2d_fwd_with_add":
        out = torch.zeros_like(x)
        api.layernorm2d_fwd_with_add(
            out, x, t["residual"], torch.zeros_like(x), w, b, EPS, x_bias
        )
        return None, out
    out = torch.zeros(B, S, N, dtype=torch.int8)
    if entry == "layernorm2d_fwd_with_smoothquant":
        api.layernorm2d_fwd_with_smoothquant(
            out, x, t["xscale"], t["yscale"], w, b, EPS, x_bias
        )
    else:
        api.layernorm2d_fwd_with_add_smoothquant(
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


def test_public_custom_op_names_and_schemas():
    for name, schema in PUBLIC_SCHEMAS.items():
        assert str(getattr(torch.ops.aiter, name).default._schema) == schema
        assert hasattr(torch.ops.aiter, f"{name}_ck")


@pytest.mark.parametrize("enable_ck", [False, True])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_public_custom_ops_dispatch(monkeypatch, triton_recorder, entry, enable_ck):
    monkeypatch.setattr(norm, "ENABLE_CK", enable_ck)
    rec = Recorder()
    for name in ENTRY_POINTS:
        monkeypatch.setattr(
            norm,
            f"{name}_ck",
            rec.make(name, returns=name in ("layer_norm", "layernorm2d_fwd")),
        )
    if enable_ck:
        for name in set(TRITON_NAME.values()):
            monkeypatch.setattr(triton_norm, name, _forbid(f"triton {name}"))

    ret, out = _call(entry, _make_inputs(), None, api=torch.ops.aiter)

    if enable_ck:
        assert [c[0] for c in rec.calls] == [entry]
        assert triton_recorder.calls == []
    else:
        assert rec.calls == []
        assert [c[0] for c in triton_recorder.calls] == [TRITON_NAME[entry]]
    result = ret if out is None else out
    assert bool((result == MARK).all())


@pytest.mark.parametrize("enable_ck", [False, True])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_public_custom_ops_compile_fullgraph(monkeypatch, entry, enable_ck):
    monkeypatch.setattr(norm, "ENABLE_CK", enable_ck)

    def fake(*args, **kwargs):
        if entry in ("layer_norm", "layernorm2d_fwd"):
            return torch.full_like(args[0], MARK)
        args[0].fill_(MARK)
        if entry in (
            "layernorm2d_fwd_with_add",
            "layernorm2d_fwd_with_smoothquant",
        ):
            args[3].fill_(MARK + 1)
        elif entry == "layernorm2d_fwd_with_add_smoothquant":
            args[3].fill_(MARK + 1)
            args[5].fill_(MARK + 2)

    monkeypatch.setattr(norm, f"{entry}_ck", fake)
    monkeypatch.setattr(triton_norm, TRITON_NAME[entry], fake)

    def call(x, residual, weight, bias, xscale, yscale, out, residual_out):
        op = getattr(torch.ops.aiter, entry)
        if entry in ("layer_norm", "layernorm2d_fwd"):
            return op(x, weight, bias, EPS, None)
        if entry == "layernorm2d_fwd_with_add":
            op(out, x, residual, residual_out, weight, bias, EPS, None)
            return out, residual_out
        if entry == "layernorm2d_fwd_with_smoothquant":
            op(out, x, xscale, yscale, weight, bias, EPS, None)
            return out, yscale
        op(
            out,
            x,
            residual,
            residual_out,
            xscale,
            yscale,
            weight,
            bias,
            EPS,
            None,
        )
        return out, residual_out, yscale

    t = _make_inputs()
    out_dtype = torch.bfloat16 if entry == "layernorm2d_fwd_with_add" else torch.int8
    args = (
        t["x"],
        t["residual"],
        t["weight"],
        t["bias"],
        t["xscale"],
        t["yscale"],
        torch.zeros(B, S, N, dtype=out_dtype),
        torch.zeros_like(t["x"]),
    )
    torch._dynamo.reset()
    result = torch.compile(call, backend="eager", fullgraph=True)(*args)

    results = result if isinstance(result, tuple) else (result,)
    for i, tensor in enumerate(results):
        assert bool((tensor == MARK + i).all())


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


@pytest.mark.parametrize("missing", ["weight", "bias"])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_ck_free_rejects_missing_affine(monkeypatch, triton_recorder, entry, missing):
    monkeypatch.setattr(norm, "ENABLE_CK", False)
    t = _make_inputs()
    t[missing] = None

    with pytest.raises(TypeError):
        _call(entry, t, None)
    assert triton_recorder.calls == []


@pytest.mark.parametrize(
    "entry", ["layernorm2d_fwd_with_add", "layernorm2d_fwd_with_add_smoothquant"]
)
def test_ck_free_add_rows_share_stride(monkeypatch, entry):
    # The Triton fused-add kernels use the input row stride for both residuals.
    monkeypatch.setattr(norm, "ENABLE_CK", False)
    calls = []

    def fake(*args, **kwargs):
        calls.append(args)
        args[3].fill_(MARK)

    monkeypatch.setattr(triton_norm, entry, fake)
    t = _make_inputs()
    x = torch.randn(B * S, 2 * N, dtype=torch.bfloat16)[:, :N]
    residual_in = t["residual"].reshape(-1, N)
    backing = torch.zeros(B * S, 2 * N, dtype=torch.bfloat16)
    residual_out = backing[:, N:]
    if entry == "layernorm2d_fwd_with_add":
        out = torch.zeros(B * S, N, dtype=torch.bfloat16)
        norm.layernorm2d_fwd_with_add(
            out, x, residual_in, residual_out, t["weight"], t["bias"], EPS
        )
    else:
        out = torch.zeros(B * S, N, dtype=torch.int8)
        norm.layernorm2d_fwd_with_add_smoothquant(
            out,
            x,
            residual_in,
            residual_out,
            t["xscale"],
            t["yscale"],
            t["weight"],
            t["bias"],
            EPS,
        )

    (args,) = calls
    rows, res_in, res_out = args[1], args[2], args[3]
    assert rows.stride() == res_in.stride() == res_out.stride() == (N, 1)
    torch.testing.assert_close(rows, x, rtol=0, atol=0)
    torch.testing.assert_close(res_in, residual_in, rtol=0, atol=0)
    assert bool((residual_out == MARK).all())
    assert bool((backing[:, :N] == 0).all())


@pytest.mark.parametrize("entry", ENTRY_POINTS[2:])
def test_ck_free_permuted_outputs(monkeypatch, entry):
    # Permuted 3-D out / residual_out have no (-1, N) view.
    monkeypatch.setattr(norm, "ENABLE_CK", False)
    has_residual = "add" in entry
    calls = []

    def fake(*args, **kwargs):
        calls.append(args)
        rows = torch.arange(args[0].shape[0]).unsqueeze(1)
        args[0].copy_(rows.expand_as(args[0]))
        if has_residual:
            args[3].copy_((rows + 10).expand_as(args[3]))

    monkeypatch.setattr(triton_norm, entry, fake)
    t = _make_inputs()
    out_dtype = torch.bfloat16 if entry == "layernorm2d_fwd_with_add" else torch.int8
    out = torch.zeros(S, B, N, dtype=out_dtype).permute(1, 0, 2)
    residual_out = torch.zeros(S, B, N, dtype=torch.bfloat16).permute(1, 0, 2)
    w, b = t["weight"], t["bias"]
    if entry == "layernorm2d_fwd_with_add":
        norm.layernorm2d_fwd_with_add(
            out, t["x"], t["residual"], residual_out, w, b, EPS
        )
    elif entry == "layernorm2d_fwd_with_smoothquant":
        norm.layernorm2d_fwd_with_smoothquant(
            out, t["x"], t["xscale"], t["yscale"], w, b, EPS
        )
    else:
        norm.layernorm2d_fwd_with_add_smoothquant(
            out,
            t["x"],
            t["residual"],
            residual_out,
            t["xscale"],
            t["yscale"],
            w,
            b,
            EPS,
        )

    (args,) = calls
    assert args[0].dim() == 2 and args[0].stride(-1) == 1
    want = torch.arange(B * S).view(B, S, 1).expand(B, S, N)
    torch.testing.assert_close(out, want.to(out_dtype), rtol=0, atol=0)
    if has_residual:
        assert args[3].is_contiguous()
        torch.testing.assert_close(
            residual_out, (want + 10).to(torch.bfloat16), rtol=0, atol=0
        )


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
