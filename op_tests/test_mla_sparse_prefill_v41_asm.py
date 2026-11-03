# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness checks for gfx1250 H64 fused V4.1 sparse MLA.

Set ``AITER_TEST_V41_ASM_COMPILE=1`` to include the torch.compile check.
"""

import importlib
import math
import os

import pytest
import torch

from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41
from aiter.test_common import checkAllclose
from op_tests.test_mla_sparse_prefill import (
    _get_gpu_arch,
    _make_inputs_v41,
    _ref_mla_sparse_prefill_fp8,
)

_SCALE = 1.0 / math.sqrt(512)


@pytest.fixture(scope="module", autouse=True)
def _requires_gfx1250():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    if _get_gpu_arch() != "gfx1250":
        pytest.skip("the fused V4.1 A8W4 CO requires gfx1250")


def _case(n=2, csa_rows=65, swa_rows=128, *, no_sink=False):
    data = _make_inputs_v41(
        n,
        64,
        csa_rows,
        swa_rows,
        topk=0,
        prefix_fp4=True,
        seed=0,
    )
    inputs = data["kernel"]
    for name in ("unified_kv", "unified_kv_scale"):
        value = inputs[name]
        inputs[name] = value.view(torch.uint8).contiguous().view(value.dtype)
    if no_sink:
        inputs["attn_sink"] = None
        data["ref"]["attn_sink"] = torch.full(
            (64,), -math.inf, dtype=torch.float32, device="cuda"
        )
    reference = _ref_mla_sparse_prefill_fp8(**data["ref"], softmax_scale=_SCALE)
    return inputs, reference


def _invoke(inputs, splits=1, *, out=None, backend="asm", softmax_scale=_SCALE):
    return mla_sparse_prefill_v41(
        **inputs,
        softmax_scale=softmax_scale,
        out=out,
        num_kv_splits=splits,
        backend=backend,
    )


def _check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    error = checkAllclose(actual, expected, rtol=0.01, atol=0.01, tol_err_ratio=0.01)
    assert error <= 0.01, f"mismatch fraction {error} exceeds 0.01"


@pytest.mark.parametrize("splits", range(1, 17))
@pytest.mark.parametrize("csa_rows,swa_rows", [(1025, 128), (1, 1)])
def test_explicit_splits(splits, csa_rows, swa_rows):
    # 1025 CSA rows exercise every split; 1+1 tiles exercise empty splits.
    inputs, expected = _case(csa_rows=csa_rows, swa_rows=swa_rows)
    _check(_invoke(inputs, splits), expected)


@pytest.mark.parametrize("splits", [1, 5, 16])
@pytest.mark.parametrize(
    "n,csa_rows,swa_rows",
    [
        (2, 0, 0),
        (2, 0, 1),
        (2, 1, 0),
        (2, 63, 0),
        (2, 64, 0),
        (2, 65, 0),
        (2, 129, 0),
        (65, 0, 65),
        (65, 63, 65),
        (65, 64, 65),
        (65, 65, 65),
    ],
)
def test_pool_and_tile_boundaries(splits, n, csa_rows, swa_rows):
    # N=65 covers SWA's 64/65 boundary with causal CSR.
    inputs, expected = _case(n, csa_rows, swa_rows)
    _check(_invoke(inputs, splits), expected)


@pytest.mark.parametrize("splits", [None, 1, 3, 16])
@pytest.mark.parametrize("provided_out", [False, True])
def test_no_sink_and_output(splits, provided_out):
    inputs, expected = _case(n=3, no_sink=True)
    out = torch.empty_like(expected) if provided_out else None
    actual = _invoke(inputs, splits, out=out)
    if out is not None:
        assert actual.data_ptr() == out.data_ptr()
    _check(actual, expected)


@pytest.mark.parametrize("splits", [1, 4])
def test_public_default_reaches_native_entry(monkeypatch, splits):
    native_module = importlib.import_module("aiter.ops.mla_sparse_prefill_v41_asm")
    native_entry = native_module.asm_mla_sparse_prefill_v41_fwd
    calls = []

    def record_and_launch(*args):
        calls.append(args)
        return native_entry(*args)

    monkeypatch.setattr(
        native_module, "asm_mla_sparse_prefill_v41_fwd", record_and_launch
    )
    inputs, expected = _case(no_sink=True)
    actual = _invoke(inputs, splits, backend=None)
    _check(actual, expected)
    assert len(calls) == 1
    args = calls[0]
    assert len(args) == 16
    assert args[10] is not None and args[10].shape == (64,)
    assert bool(torch.isneginf(args[10]).all())
    assert args[11].data_ptr() == actual.data_ptr()
    assert args[12] == _SCALE and args[13] == splits
    if splits == 1:
        assert args[14] is None and args[15] is None
    else:
        assert args[14].shape == (2, splits, 65, 512)
        assert args[15].shape == (2, splits, 64)
        assert args[14].dtype == args[15].dtype == torch.float32


@pytest.mark.parametrize("n,csa_rows", [(8, 4096), (16, 1024), (32, 1024), (96, 1024)])
def test_auto_split_matches_forced(monkeypatch, n, csa_rows):
    native_module = importlib.import_module("aiter.ops.mla_sparse_prefill_v41_asm")
    native_entry = native_module.asm_mla_sparse_prefill_v41_fwd
    selected = []

    def record_and_launch(*args):
        selected.append(args[13])
        return native_entry(*args)

    monkeypatch.setattr(
        native_module, "asm_mla_sparse_prefill_v41_fwd", record_and_launch
    )
    inputs, expected = _case(n=n, csa_rows=csa_rows)
    nnz = inputs["kv_indices_prefix"].numel() + inputs["kv_indices_extend"].numel()
    split = native_module._auto_v41_asm_splits(n, nnz)
    actual = _invoke(inputs, None)
    _check(actual, expected)
    forced = _invoke(inputs, split)
    assert selected == [split, split]
    assert torch.equal(actual, forced)


def test_zero_queries_does_not_launch(monkeypatch):
    native_module = importlib.import_module("aiter.ops.mla_sparse_prefill_v41_asm")

    def unexpected_launch(*_args):
        pytest.fail("N=0 must return before the native entry")

    monkeypatch.setattr(
        native_module, "asm_mla_sparse_prefill_v41_fwd", unexpected_launch
    )
    inputs, expected = _case(0, 0, 0, no_sink=True)
    actual = _invoke(inputs, 16)
    assert actual.shape == expected.shape == (0, 64, 512)
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("splits", [1, 4])
def test_raw_byte_csa(splits):
    inputs, expected = _case()
    inputs["unified_kv"] = inputs["unified_kv"].view(torch.uint8)
    inputs["unified_kv_scale"] = inputs["unified_kv_scale"].view(torch.uint8)
    _check(_invoke(inputs, splits), expected)


@pytest.mark.parametrize("splits", [0, -1, 17, 32])
def test_reject_invalid_splits(splits):
    inputs, _ = _case()
    with pytest.raises(ValueError, match="num_kv_splits"):
        _invoke(inputs, splits)


def test_reject_fractional_split():
    inputs, _ = _case()
    # Torch's public SymInt schema may reject this before the Python guard.
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _invoke(inputs, 1.5)


@pytest.mark.parametrize("scale", [0.08, 0.0, -1.0, math.nan, math.inf])
def test_reject_unsupported_softmax_scale(scale):
    inputs, _ = _case(n=1, csa_rows=1, swa_rows=1)
    with pytest.raises(ValueError, match="softmax_scale"):
        _invoke(inputs, softmax_scale=scale)


def test_accept_fp32_rounded_default_scale():
    inputs, expected = _case(n=1, csa_rows=1, swa_rows=1)
    rounded_scale = torch.tensor(_SCALE, dtype=torch.float32).item()
    _check(_invoke(inputs, softmax_scale=rounded_scale), expected)


@pytest.mark.parametrize(
    "bad_argument",
    [
        "packed_csa",
        "heads",
        "dimension",
        "q_dtype",
        "q_scale_dtype",
        "csr_dtype",
        "csr_shape",
        "swa_scale_rows",
        "sink_dtype",
        "cpu_scale",
        "out_dtype",
        "out_stride",
    ],
)
def test_reject_invalid_layout(bad_argument):
    inputs, expected = _case()
    out = None
    if bad_argument == "packed_csa":
        packed = _make_inputs_v41(2, 64, 65, 128, topk=0, prefix_fp4=True)["kernel"]
        inputs["unified_kv"] = packed["unified_kv"]
        inputs["unified_kv_scale"] = packed["unified_kv_scale"]
    elif bad_argument == "heads":
        inputs["q"] = inputs["q"][:, :32].contiguous()
    elif bad_argument == "dimension":
        inputs["q"] = inputs["q"][..., :256].contiguous()
    elif bad_argument == "q_dtype":
        inputs["q"] = inputs["q"].to(torch.bfloat16)
    elif bad_argument == "q_scale_dtype":
        inputs["q_scale"] = inputs["q_scale"].to(torch.int32)
    elif bad_argument == "csr_dtype":
        inputs["kv_indices_prefix"] = inputs["kv_indices_prefix"].to(torch.int64)
    elif bad_argument == "csr_shape":
        inputs["kv_indptr_extend"] = inputs["kv_indptr_extend"][:-1]
    elif bad_argument == "swa_scale_rows":
        inputs["kv_scale"] = inputs["kv_scale"][:-1]
    elif bad_argument == "sink_dtype":
        inputs["attn_sink"] = inputs["attn_sink"].to(torch.bfloat16)
    elif bad_argument == "cpu_scale":
        inputs["q_scale"] = inputs["q_scale"].cpu()
    elif bad_argument == "out_dtype":
        out = torch.empty_like(expected, dtype=torch.float32)
    elif bad_argument == "out_stride":
        out = torch.empty((2, 512, 64), device="cuda", dtype=torch.bfloat16).transpose(
            1, 2
        )
    with pytest.raises(ValueError):
        _invoke(inputs, out=out)


@pytest.mark.parametrize("splits", [1, 4])
def test_current_stream(splits):
    inputs, expected = _case()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = _invoke(inputs, splits)
        # The consumer is queued on the same nondefault stream as the launch.
        consumed = actual.clone()
    stream.synchronize()
    _check(consumed, expected)


@pytest.mark.parametrize("splits", [1, 4])
def test_graph_replay(splits):
    inputs, expected = _case(no_sink=True)
    out = torch.empty_like(expected)
    # Finish JIT/module loading and initialize the nondefault stream before capture.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        _invoke(inputs, splits, out=out)
    warmup.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup):
        captured = _invoke(inputs, splits, out=out)
    for _ in range(3):
        out.fill_(math.nan)
        graph.replay()
        _check(captured, expected)


@pytest.mark.skipif(
    os.environ.get("AITER_TEST_V41_ASM_COMPILE") != "1",
    reason="set AITER_TEST_V41_ASM_COMPILE=1 to test torch.compile fullgraph",
)
def test_torch_compile_fullgraph():
    inputs, expected = _case()
    # Warm the native entry before testing Torch's graph/fake-tensor dispatch.
    _check(_invoke(inputs, 4), expected)

    def forward(q):
        call_inputs = dict(inputs)
        call_inputs["q"] = q
        return _invoke(call_inputs, 4)

    compiled = torch.compile(forward, fullgraph=True)
    _check(compiled(inputs["q"]), expected)
