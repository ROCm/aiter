# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Existing H64 ASM suite, updated for the 144B BF16-Q / FP8-O Q/O contract.

AITER_TEST_V41_ASM_COMPILE=1 enables fullgraph checks for both allocating and
preallocated outputs. OPUS attention-only routing is tested separately below.
"""

import importlib
import inspect
import math
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from op_tests.test_mla_sparse_prefill import (
    _check_v41_qo_output,
    _get_gpu_arch,
    _make_inputs_v41,
    _prepare_v41_qo_inputs,
    _qo_dequantize,
    _qo_quantize,
    _ref_mla_sparse_prefill_fp8,
    _ref_mla_sparse_prefill_qo,
)

native = importlib.import_module("aiter.ops.mla_sparse_prefill_v41_asm")
_SCALE = 512**-0.5
_ORDER = [
    "q_nope",
    "cos",
    "sin",
    "positions",
    "unified_kv",
    "unified_kv_scale",
    "kv",
    "kv_scale",
    "kv_indptr_prefix",
    "kv_indices_prefix",
    "kv_indptr_extend",
    "kv_indices_extend",
    "attn_sink",
]


@pytest.fixture(scope="module", autouse=True)
def _requires_gfx1250():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires ROCm")
    if _get_gpu_arch() != "gfx1250":
        pytest.skip("Q/O ASM requires gfx1250")


def _case(n=2, csa_rows=65, swa_rows=128, *, no_sink=False, window=None):
    data = _make_inputs_v41(
        n, 64, csa_rows, swa_rows, topk=0, window=window, prefix_fp4=True, seed=0
    )
    inputs = _prepare_v41_qo_inputs(data)
    if no_sink:
        inputs["attn_sink"] = None
    reference = _ref_mla_sparse_prefill_qo(inputs, data["ref"], _SCALE)
    return inputs, (reference, inputs, data["ref"])


def _invoke(
    inputs,
    splits=1,
    *,
    out=None,
    out_scale=None,
    partial_r=None,
    partial_lse=None,
    softmax_scale=_SCALE,
):
    return native._asm_sparse_prefill_v41(
        **inputs,
        softmax_scale=softmax_scale,
        num_kv_splits=splits,
        out=out,
        out_scale=out_scale,
        partial_r=partial_r,
        partial_lse=partial_lse,
    )


def _check(actual, expected):
    ref, inputs, _ = expected
    return _check_v41_qo_output(actual, ref, inputs)


def _equal(a, b):
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("splits", range(1, 17))
@pytest.mark.parametrize("csa_rows,swa_rows", [(1025, 128), (1, 1)])
def test_explicit_splits(splits, csa_rows, swa_rows):
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
    inputs, expected = _case(n, csa_rows, swa_rows)
    _check(_invoke(inputs, splits), expected)


@pytest.mark.parametrize("splits", [1, 4, 16])
@pytest.mark.parametrize(
    "n,csa_rows,window",
    [(129, 65, 0), (129, 0, 128), (129, 65, 128), (257, 65, 128), (129, 65, 1000)],
)
def test_sliding_window(splits, n, csa_rows, window):
    inputs, expected = _case(n, csa_rows, n, window=window)
    ip, ix = inputs["kv_indptr_extend"].tolist(), inputs["kv_indices_extend"].tolist()
    for i in [0, 127, 128, n - 1]:
        assert ix[ip[i] : ip[i + 1]] == list(range(max(0, i + 1 - window), i + 1))
    _check(_invoke(inputs, splits), expected)


@pytest.mark.parametrize("splits", [None, 1, 3, 16])
@pytest.mark.parametrize("provided", [False, True])
def test_no_sink_and_preallocated_outputs(splits, provided):
    inputs, expected = _case(n=3, no_sink=True)
    out = (
        torch.empty((3, 64, 512), device="cuda", dtype=torch.float8_e4m3fn)
        if provided
        else None
    )
    sc = (
        torch.empty((3, 64, 16), device="cuda", dtype=torch.uint8) if provided else None
    )
    actual = _invoke(inputs, splits, out=out, out_scale=sc)
    if provided:
        assert actual[0] is out and actual[1] is sc
    _check(actual, expected)


@pytest.mark.parametrize("splits", [1, 4])
def test_wrapper_uses_exact_kernel_order(monkeypatch, splits):
    actual_entry = native.asm_mla_sparse_prefill_v41_fwd
    calls = []

    def record(*args):
        calls.append(args)
        return actual_entry(*args)

    monkeypatch.setattr(native, "asm_mla_sparse_prefill_v41_fwd", record)
    inputs, expected = _case(no_sink=True)
    out, scale = _invoke(inputs, splits)
    _check((out, scale), expected)
    assert len(calls) == 1 and len(calls[0]) == 19
    args = calls[0]
    for i, name in enumerate(_ORDER[:-1]):
        assert args[i].data_ptr() == inputs[name].data_ptr()
    assert torch.isneginf(args[12]).all() and args[12].shape == (64,)
    assert args[13] == _SCALE and args[14] == splits
    assert (
        args[15].data_ptr() == out.data_ptr()
        and args[16].data_ptr() == scale.data_ptr()
    )
    if splits == 1:
        assert args[17] is None and args[18] is None
    else:
        assert args[17].shape == (2, splits, 65, 512) and args[18].shape == (
            2,
            splits,
            64,
        )


@pytest.mark.parametrize(
    "n,topk,splits",
    [(8, 512, 1), (32, 512, 1), (32, 1024, 4), (64, 1024, 2), (96, 1024, 1)],
)
def test_auto_split_matches_forced(n, topk, splits):
    data = _make_inputs_v41(
        n, 64, topk, 128, topk=topk, window=128, prefix_fp4=True, seed=0
    )
    x = _prepare_v41_qo_inputs(data)
    assert (
        native._auto_v41_asm_splits(
            n,
            x["kv_indices_prefix"].numel() + x["kv_indices_extend"].numel(),
            prefix_nnz=x["kv_indices_prefix"].numel(),
            prefix_rows=x["unified_kv"].shape[0],
        )
        == splits
    )
    a, b = _invoke(x, None), _invoke(x, splits)
    for x, y in zip(a, b):
        _equal(x, y)


@pytest.mark.parametrize(
    "n,pool,topk,splits",
    [
        (8, 65536, 512, 4),
        (16, 65536, 512, 4),
        (8, 65536, 1024, 8),
        (16, 65536, 1024, 8),
        (32, 65536, 512, 1),
        (32, 1048576, 512, 4),
        (48, 65536, 1024, 4),
        (64, 65536, 1024, 4),
    ],
)
def test_tuned_auto_split_matches_forced(n, pool, topk, splits):
    data = _make_inputs_v41(
        n, 64, pool, 128, topk=topk, window=128, prefix_fp4=True, seed=0
    )
    x = _prepare_v41_qo_inputs(data)
    ref = _ref_mla_sparse_prefill_qo(x, data["ref"], _SCALE)
    a, b = _invoke(x, None), _invoke(x, splits)
    _check_v41_qo_output(a, ref, x)
    for actual, forced in zip(a, b):
        _equal(actual, forced)


@pytest.mark.parametrize("splits", [1, 3, 16])
def test_positions_and_input_immutability(splits):
    x, expected = _case(3)
    x["positions"] = torch.tensor([1030, 0, 517], device="cuda", dtype=torch.int64)
    ref = _ref_mla_sparse_prefill_qo(x, expected[2], _SCALE)
    before = {
        k: v.view(torch.uint8).clone()
        for k, v in x.items()
        if isinstance(v, torch.Tensor)
    }
    a = _invoke(x, splits)
    _check_v41_qo_output(a, ref, x)
    b = _invoke(x, splits)
    for k, v in before.items():
        _equal(x[k], v)
    for v, w in zip(a, b):
        _equal(v, w)


@pytest.mark.parametrize("splits", [1, 7, 16])
@pytest.mark.parametrize("fill", [0xA5, 0x5A, 0xFF])
def test_guards_workspace_and_repeated_output(splits, fill):
    x, expected = _case(2, 129, 129)

    def guarded(shape, dtype):
        count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        raw = torch.full((count + 1024,), 0x35, device="cuda", dtype=torch.uint8)
        raw[512:-512].fill_(fill)
        return raw[512:-512].view(dtype).view(shape), raw

    out, g0 = guarded((2, 64, 512), torch.float8_e4m3fn)
    sc, g1 = guarded((2, 64, 16), torch.uint8)
    r = lse = None
    guards = [g0, g1]
    if splits > 1:
        r, g2 = guarded((2, splits, 65, 512), torch.float32)
        lse, g3 = guarded((2, splits, 64), torch.float32)
        guards += [g2, g3]
    a = _invoke(x, splits, out=out, out_scale=sc, partial_r=r, partial_lse=lse)
    _check(a, expected)
    for guard in guards:
        assert bool((guard[:512] == 0x35).all() and (guard[-512:] == 0x35).all())
    if r is not None:
        assert bool((r[:, :, 64].contiguous().view(torch.uint8) == fill).all())


@pytest.mark.parametrize("value", [0.0, 1e-15, 1e-10, 448 * 2.0**-20, 449 * 2.0**-20])
@pytest.mark.parametrize("splits", [1, 16])
def test_zero_tiny_and_scale_boundaries(value, splits):
    x, expected = _case(1, 0, 1, no_sink=True, window=1)
    x["q_nope"].zero_()
    x["cos"].fill_(1)
    x["sin"].zero_()
    real = torch.full((1, 512), value, device="cuda", dtype=torch.float32)
    kv, sc = _qo_quantize(real)
    x["kv"] = kv
    x["kv_scale"] = sc
    expected[2]["kv_fp32"] = _qo_dequantize(kv, sc)
    ref = _ref_mla_sparse_prefill_qo(x, expected[2], _SCALE)
    out, scale = _invoke(x, splits)
    _check_v41_qo_output((out, scale), ref, x)
    rq, rs = _qo_quantize(ref)
    _equal(out, rq)
    _equal(scale, rs)


def test_zero_queries_does_not_launch(monkeypatch):
    x, _expected = _case(0, 0, 0)

    def forbidden(*args):
        pytest.fail("zero query launched")

    monkeypatch.setattr(native, "asm_mla_sparse_prefill_v41_fwd", forbidden)
    out, scale = _invoke(x)
    assert out.shape == (0, 64, 512) and scale.shape == (0, 64, 16)


@pytest.mark.parametrize("splits", [0, -1, 17, 1.5, True])
def test_reject_invalid_splits(splits):
    x, _ = _case()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _invoke(x, splits)


@pytest.mark.parametrize("scale", [0.0, 0.04, float("nan"), float("inf")])
def test_reject_unsupported_scale(scale):
    x, _ = _case()
    with pytest.raises((ValueError, RuntimeError)):
        _invoke(x, softmax_scale=scale)


@pytest.mark.parametrize(
    "bad",
    [
        "q_fp8",
        "position_i32",
        "position_shape",
        "cos_fp32",
        "sin_rows",
        "q_stride",
        "packed_csa",
        "csr_i64",
        "out_bf16",
        "scale_fp8",
        "workspace_pair",
        "workspace_shape",
    ],
)
def test_reject_invalid_interface(bad):
    x, _ = _case()
    kw = {}
    if bad == "q_fp8":
        x["q_nope"] = x["q_nope"].to(torch.float8_e4m3fn)
    elif bad == "position_i32":
        x["positions"] = x["positions"].int()
    elif bad == "position_shape":
        x["positions"] = x["positions"][:, None]
    elif bad == "cos_fp32":
        x["cos"] = x["cos"].float()
    elif bad == "sin_rows":
        x["sin"] = x["sin"][:-1]
    elif bad == "q_stride":
        x["q_nope"] = torch.empty((2, 64, 1024), device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif bad == "packed_csa":
        d = _make_inputs_v41(2, 64, 65, 128, topk=0, prefix_fp4=True)["kernel"]
        x["unified_kv"] = d["unified_kv"]
        x["unified_kv_scale"] = d["unified_kv_scale"]
    elif bad == "csr_i64":
        x["kv_indptr_prefix"] = x["kv_indptr_prefix"].long()
    elif bad == "out_bf16":
        kw["out"] = torch.empty_like(x["q_nope"])
    elif bad == "scale_fp8":
        kw["out_scale"] = torch.empty(
            (2, 64, 16), device="cuda", dtype=torch.float8_e8m0fnu
        )
    elif bad == "workspace_pair":
        kw["partial_r"] = torch.empty((2, 4, 65, 512), device="cuda")
    elif bad == "workspace_shape":
        kw["partial_r"] = torch.empty((2, 4, 64, 512), device="cuda")
        kw["partial_lse"] = torch.empty((2, 4, 64), device="cuda")
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _invoke(x, 4, **kw)


@pytest.mark.parametrize("kind", ["out_q", "scale_q", "out_scale", "workspace_q"])
def test_reject_overlapping_buffers(kind):
    x, _ = _case()
    kw = {}
    raw = x["q_nope"].view(torch.uint8).flatten()
    if kind == "out_q":
        kw["out"] = raw[: 2 * 64 * 512].view(torch.float8_e4m3fn).view(2, 64, 512)
    elif kind == "scale_q":
        kw["out_scale"] = raw[: 2 * 64 * 16].view(2, 64, 16)
    elif kind == "out_scale":
        out = torch.empty((2, 64, 512), device="cuda", dtype=torch.float8_e4m3fn)
        kw["out"] = out
        kw["out_scale"] = out.view(torch.uint8).flatten()[: 2 * 64 * 16].view(2, 64, 16)
    else:
        raw = torch.empty(2 * 4 * 65 * 512 * 4, device="cuda", dtype=torch.uint8)
        x["q_nope"] = raw[: 2 * 64 * 512 * 2].view(torch.bfloat16).view(2, 64, 512)
        kw["partial_r"] = raw.view(torch.float32).view(2, 4, 65, 512)
        kw["partial_lse"] = torch.empty((2, 4, 64), device="cuda")
    with pytest.raises((ValueError, RuntimeError), match="overlap"):
        _invoke(x, 4, **kw)


@pytest.mark.parametrize("splits", [1, 4, 16])
def test_current_stream_and_graph(splits):
    x, expected = _case(3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out, scale = _invoke(x, splits)
        r = torch.empty((3, splits, 65, 512), device="cuda") if splits > 1 else None
        lse = torch.empty((3, splits, 64), device="cuda") if splits > 1 else None
        _invoke(x, splits, out=out, out_scale=scale, partial_r=r, partial_lse=lse)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _invoke(x, splits, out=out, out_scale=scale, partial_r=r, partial_lse=lse)
        seen = out.view(torch.uint8).clone()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _check((out, scale), expected)
    _equal(out, seen)


@pytest.mark.skipif(
    os.environ.get("AITER_TEST_V41_ASM_COMPILE") != "1", reason="opt-in torch.compile"
)
@pytest.mark.parametrize("provided", [False, True])
def test_torch_compile_fullgraph(provided):
    x, expected = _case()
    out, scale = _invoke(x, 4)

    def forward(q):
        inputs = dict(x, q_nope=q)
        return _invoke(
            inputs,
            4,
            out=out if provided else None,
            out_scale=scale if provided else None,
        )

    compiled = torch.compile(forward, fullgraph=True)
    _check(compiled(x["q_nope"]), expected)


@pytest.mark.parametrize("n,h,split", [(2, 64, 1), (7, 128, 1), (8, 64, 4)])
def test_opus_a8w8_numerical_regression(n, h, split):
    # Real OPUS A8W8 execution, not only a mocked backend check.
    d = _make_inputs_v41(n, h, 65, 128, topk=0, prefix_fp4=False, seed=0)
    ref = _ref_mla_sparse_prefill_fp8(**d["ref"], softmax_scale=_SCALE)
    from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41

    for backend in ["opus", None]:
        out = mla_sparse_prefill_v41(
            **d["kernel"], softmax_scale=_SCALE, num_kv_splits=split, backend=backend
        )
        assert isinstance(out, torch.Tensor) and out.dtype == torch.bfloat16
        assert (
            float(
                (~torch.isclose(out.float(), ref.float(), rtol=0.01, atol=0.01))
                .float()
                .mean()
            )
            <= 0.01
        )


@pytest.mark.parametrize(
    "bad", ["positions_i32", "out_bf16", "scale_e8m0", "partial_shape", "out_q_alias"]
)
def test_native_entry_checks_before_launch(bad):
    x, _expected = _case()
    q_before = x["q_nope"].clone()
    out, scale = _invoke(x, 4)
    args = [x[name] for name in _ORDER] + [
        _SCALE,
        4,
        out,
        scale,
        torch.empty((2, 4, 65, 512), device="cuda"),
        torch.empty((2, 4, 64), device="cuda"),
    ]
    if bad == "positions_i32":
        args[3] = x["positions"].int()
    elif bad == "out_bf16":
        args[15] = torch.empty_like(x["q_nope"])
    elif bad == "scale_e8m0":
        args[16] = scale.view(torch.float8_e8m0fnu)
    elif bad == "partial_shape":
        args[17] = torch.empty((2, 4, 64, 512), device="cuda")
    else:
        args[15] = (
            x["q_nope"]
            .view(torch.uint8)
            .flatten()[: 2 * 64 * 512]
            .view(torch.float8_e4m3fn)
            .view(2, 64, 512)
        )
    with pytest.raises(RuntimeError):
        native.asm_mla_sparse_prefill_v41_fwd(*args)
    _equal(q_before, x["q_nope"])


# Metadata-only capability and dispatcher checks share this existing suite.

public = importlib.import_module("aiter.ops.mla_sparse_prefill")


query = public.get_mla_sparse_prefill_v41_backend


execute = inspect.unwrap(public.mla_sparse_prefill_v41)


SCALE = 1 / math.sqrt(512)


@pytest.fixture
def runtime(monkeypatch):
    state = {"gfx": "gfx1250", "asm": True, "available": True}
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: state["available"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            gcnArchName=state.get(f"gfx{index}", state["gfx"])
        ),
    )
    monkeypatch.setattr(public, "get_gfx_runtime", lambda: state["gfx"].split(":")[0])
    monkeypatch.setattr(public, "is_gfx1250_asm_supported", lambda: state["asm"])
    return state


class MetadataTensor:
    """Only tensor metadata is available; touching values is a test failure."""

    def __init__(
        self, shape, dtype, *, device="cuda:0", row_strided=False, aligned=True
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.row_strided = row_strided
        self.aligned = aligned

    def dim(self):
        return len(self.shape)

    def size(self, dim):
        return self.shape[dim]

    def stride(self, dim=None):
        if len(self.shape) == 3:
            _, h, d = self.shape
            strides = (h * d, d, 1)
        else:
            strides = (self.shape[-1] + (32 if self.row_strided else 0), 1)
        return strides if dim is None else strides[dim]

    def data_ptr(self):
        return 256 if self.aligned else 257

    def element_size(self):
        return 1

    def is_contiguous(self):
        return not self.row_strided

    def item(self):
        pytest.fail("capability/dispatch read tensor contents")


def arguments(
    n=8, h=64, d=512, prefix_dtype=torch.uint8, layout="contiguous", **dtypes
):
    fp4 = prefix_dtype in (torch.uint8, public.dtypes.fp4x2)
    q = MetadataTensor((n, h, d), dtypes.get("q_dtype", public.dtypes.fp8))
    qs = MetadataTensor((n, h, 16), dtypes.get("q_scale_dtype", torch.uint8))
    prefix = MetadataTensor(
        (128, 256 if fp4 else 512), prefix_dtype, row_strided=layout == "row_strided"
    )
    ps = MetadataTensor(
        (128, 32 if fp4 else 16), dtypes.get("unified_kv_scale_dtype", torch.uint8)
    )
    kv = MetadataTensor((128, 512), dtypes.get("kv_dtype", public.dtypes.fp8))
    ks = MetadataTensor((128, 16), dtypes.get("kv_scale_dtype", torch.uint8))
    return (q, qs, prefix, ps, object(), object(), kv, ks, object(), object(), None)


CASES = [
    ("gfx1250", 8, 64, 512, torch.uint8, SCALE, "contiguous", True, "asm"),
    ("gfx1250", 65535, 64, 512, torch.uint8, SCALE, "contiguous", True, "asm"),
    ("gfx1250", 65537, 64, 512, torch.uint8, SCALE, "contiguous", True, None),
    ("gfx1250", 0, 64, 512, torch.uint8, SCALE, "contiguous", True, "asm"),
    ("gfx1250", 8, 32, 512, torch.uint8, SCALE, "contiguous", True, None),
    ("gfx1250", 8, 128, 512, torch.uint8, SCALE, "contiguous", True, None),
    ("gfx1250", 8, 64, 256, torch.uint8, SCALE, "contiguous", True, None),
    ("gfx1250", 8, 64, 512, torch.uint8, 0.04, "contiguous", True, None),
    ("gfx1250", 8, 64, 512, torch.uint8, SCALE, "row_strided", True, None),
    ("gfx1250", 8, 64, 512, torch.uint8, SCALE, "contiguous", False, None),
    ("gfx950", 8, 16, 512, torch.uint8, SCALE, "contiguous", True, "opus"),
    ("gfx950", 65537, 128, 512, torch.uint8, 0.04, "row_strided", True, "opus"),
    ("gfx950", 8, 64, 256, torch.uint8, SCALE, "contiguous", True, None),
    ("gfx1250", 8, 128, 512, public.dtypes.fp8, 0.04, "contiguous", False, "opus"),
    ("gfx1250", 8, 32, 512, public.dtypes.fp8, 0.0, "row_strided", True, "opus"),
    ("gfx950", 8, 64, 512, public.dtypes.fp8, SCALE, "contiguous", True, "opus"),
    ("gfx942", 8, 64, 512, torch.uint8, SCALE, "contiguous", True, None),
    ("gfx1250-strict", 8, 64, 512, torch.uint8, SCALE, "contiguous", True, "asm"),
    ("gfx1250-strict", 8, 64, 512, torch.uint8, SCALE, "contiguous", False, None),
    ("gfx1250-strict", 8, 64, 512, public.dtypes.fp8, SCALE, "contiguous", True, None),
]


class TestV41Capability:
    @pytest.fixture(autouse=True)
    def _requires_gfx1250(self):
        """Capability tests use mocked runtime metadata on any architecture."""

    def test_query_is_publicly_exported(
        self,
    ):
        import aiter

        assert "get_mla_sparse_prefill_v41_backend" in public.__all__
        if not aiter.AITER_AOT_IMPORT:
            assert aiter.get_mla_sparse_prefill_v41_backend is query
        assert "_asm_sparse_prefill_v41" in public.__all__

    @pytest.mark.parametrize("gfx,n,h,d,prefix,scale,layout,asm,expected", CASES)
    def test_query_matches_automatic_dispatch(
        self, runtime, monkeypatch, gfx, n, h, d, prefix, scale, layout, asm, expected
    ):
        runtime.update(gfx=gfx, asm=asm)
        q_dtype = (
            torch.bfloat16
            if gfx.startswith("gfx1250") and prefix == torch.uint8
            else public.dtypes.fp8
        )
        assert (
            query(
                "cuda:0",
                (n, h, d),
                unified_kv_dtype=prefix,
                softmax_scale=scale,
                layout=layout,
                q_dtype=q_dtype,
            )
            == expected
        )
        calls = []
        for backend in ("asm", "opus"):
            monkeypatch.setattr(
                public,
                f"_{backend}_sparse_prefill_v41",
                lambda *args, _backend=backend, **kwargs: calls.append(_backend)
                or _backend,
            )
        monkeypatch.setattr(
            public._plugin,
            "_dispatch",
            lambda op, canonicalize, native, args, kwargs: native(*args, **kwargs),
        )
        args = arguments(n, h, d, prefix, layout)
        if expected in (None, "asm"):
            with pytest.raises(RuntimeError, match="no native backend"):
                execute(*args, scale)
            assert not calls
        else:
            assert execute(*args, scale) == expected
            assert calls == [expected]

    @pytest.mark.parametrize(
        "field",
        [
            "q_dtype",
            "q_scale_dtype",
            "unified_kv_scale_dtype",
            "kv_dtype",
            "kv_scale_dtype",
        ],
    )
    def test_unsupported_quantized_dtype(self, runtime, monkeypatch, field):
        kwargs = {field: torch.float16}
        assert (
            query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8, **kwargs)
            is None
        )
        monkeypatch.setattr(
            public._plugin,
            "_dispatch",
            lambda op, canonicalize, native, args, kw: native(*args, **kw),
        )
        with pytest.raises(RuntimeError, match="no native backend"):
            execute(*arguments(**kwargs), SCALE)

    def test_query_is_metadata_only(self, runtime, monkeypatch):
        def forbidden(*args, **kwargs):
            pytest.fail("query allocated, compiled, launched or read tensor contents")

        for name in ("empty", "zeros", "full", "tensor"):
            monkeypatch.setattr(torch, name, forbidden)
        monkeypatch.setattr(torch.Tensor, "item", forbidden)
        monkeypatch.setattr(public, "_asm_sparse_prefill_v41", forbidden)
        monkeypatch.setattr(public, "_opus_sparse_prefill_v41", forbidden)
        monkeypatch.setattr(public, "compile_ops", forbidden)
        assert (
            query(
                "cuda:0",
                (65535, 64, 512),
                unified_kv_dtype=torch.uint8,
                q_dtype=torch.bfloat16,
            )
            == "asm"
        )
        runtime["gfx"] = "gfx950"
        assert query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8) == "opus"

    def test_fp32_scale_equivalence(self, runtime):
        fp32_scale = struct.unpack("<f", struct.pack("<f", SCALE))[0]
        assert (
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                softmax_scale=fp32_scale,
                q_dtype=torch.bfloat16,
            )
            == "asm"
        )
        assert (
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                softmax_scale=fp32_scale + 1e-5,
            )
            is None
        )

    def test_runtime_availability_and_target_device(self, runtime, monkeypatch):
        assert query("cpu", (8, 64, 512), unified_kv_dtype=torch.uint8) is None
        runtime["available"] = False
        assert query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8) is None
        runtime["available"] = True
        monkeypatch.setattr(torch.version, "hip", None)
        assert query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8) is None
        monkeypatch.setattr(torch.version, "hip", "7.0")
        runtime["gfx1"] = "gfx950"
        assert query("cuda:1", (8, 64, 512), unified_kv_dtype=torch.uint8) is None
        with pytest.raises(ValueError, match="out of range"):
            query("cuda:2", (8, 64, 512), unified_kv_dtype=torch.uint8)

    @pytest.mark.parametrize("shape", [(8, 64), (-1, 64, 512), (8, 0, 512)])
    def test_malformed_shape_raises(self, runtime, shape):
        with pytest.raises(ValueError):
            query("cuda:0", shape, unified_kv_dtype=torch.uint8)

    @pytest.mark.parametrize("scale", [float("nan"), float("inf"), 1e100])
    def test_malformed_scale_raises(self, runtime, scale):
        with pytest.raises(ValueError, match="finite FP32"):
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                softmax_scale=scale,
            )

    def test_malformed_metadata_types(self, runtime):
        with pytest.raises(TypeError, match="integer"):
            query("cuda:0", (True, 64, 512), unified_kv_dtype=torch.uint8)
        with pytest.raises(TypeError, match="torch.dtype"):
            query("cuda:0", (8, 64, 512), unified_kv_dtype="fp4")
        with pytest.raises(TypeError, match="host number"):
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                softmax_scale=object(),
            )
        assert (
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                layout="interleaved_raw",
            )
            is None
        )

    @pytest.mark.parametrize("backend", ["opus"])
    def test_explicit_backend_errors_propagate(self, runtime, monkeypatch, backend):
        error = ValueError("execution validation or launch failed")

        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(public, f"_{backend}_sparse_prefill_v41", fail)
        monkeypatch.setattr(
            public._plugin,
            "_dispatch",
            lambda op, canonicalize, native, args, kw: native(*args, **kw),
        )
        with pytest.raises(ValueError) as caught:
            execute(*arguments(n=65537), SCALE, backend=backend)
        assert caught.value is error

    @pytest.mark.parametrize("backend", [None, "asm", "opus"])
    def test_provider_path_preserves_v3_contract(self, runtime, monkeypatch, backend):
        def unexpected(*args, **kwargs):
            pytest.fail("provider path should not probe or execute a native backend")

        monkeypatch.setattr(public, "get_mla_sparse_prefill_v41_backend", unexpected)
        monkeypatch.setattr(public, "_asm_sparse_prefill_v41", unexpected)
        monkeypatch.setattr(public, "_opus_sparse_prefill_v41", unexpected)
        # The public spelling still takes `out`; v3 is what a provider is offered,
        # and it carries no output buffer. A call that names one does not reach
        # here at all -- it bypasses routing -- so the only shape this path sees
        # is the one below, with the `out` slot gone.
        args = (*arguments(), SCALE, None, None)
        contract_args = (*args[:-2], args[-1])

        def provider(op, canonicalize, native, actual_args, kwargs):
            assert op.name == "mla_sparse_prefill_v41" and op.version == 3
            assert actual_args == contract_args and kwargs == {}
            bound, kw = canonicalize(*actual_args, **kwargs)
            assert len(bound) == 13 and "backend" not in kw
            return "provider"

        monkeypatch.setattr(public._plugin, "_dispatch", provider)
        assert execute(*args, backend=backend) == "provider"

    def test_old_asm_pin_reports_new_contract(self, runtime, monkeypatch):
        monkeypatch.setattr(
            public._plugin,
            "_dispatch",
            lambda op, canonicalize, native, args, kw: native(*args, **kw),
        )
        with pytest.raises(RuntimeError, match="raw BF16 q_nope"):
            execute(*arguments(), SCALE, backend="asm")

    def test_asm_query_rejects_old_q_and_n65536(self, runtime):
        assert query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8) is None
        assert (
            query(
                "cuda:0",
                (65536, 64, 512),
                unified_kv_dtype=torch.uint8,
                q_dtype=torch.bfloat16,
            )
            is None
        )
        assert (
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                q_dtype=torch.bfloat16,
            )
            == "asm"
        )

    def test_asm_query_checks_kv_dtypes_without_q_scale(self, runtime):
        # The generic query keeps q_scale_dtype for OPUS only.
        assert (
            query(
                "cuda:0",
                (8, 64, 512),
                unified_kv_dtype=torch.uint8,
                q_dtype=torch.bfloat16,
                q_scale_dtype=torch.float16,
            )
            == "asm"
        )
        for field in ["unified_kv_scale_dtype", "kv_dtype", "kv_scale_dtype"]:
            assert (
                query(
                    "cuda:0",
                    (8, 64, 512),
                    unified_kv_dtype=torch.uint8,
                    q_dtype=torch.bfloat16,
                    **{field: torch.float16},
                )
                is None
            )

    @pytest.mark.parametrize(
        "n",
        [
            8,
            16,
            32,
            48,
            64,
            96,
            192,
            384,
            768,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            16384,
        ],
    )
    @pytest.mark.parametrize("pool", [65536, 1048576])
    @pytest.mark.parametrize("topk", [512, 1024])
    def test_auto_policy_matches_measured_winners(self, n, pool, topk):
        if n in (8, 16):
            expected = 4 if topk == 512 else 8
        elif topk == 512:
            expected = 4 if n == 32 and pool == 1048576 else 1
        else:
            expected = 4 if n in (32, 48, 64) else 1
        for swa_nnz in (0, n * 128, n * 4096):
            assert (
                native._auto_v41_asm_splits(
                    n, n * topk + swa_nnz, prefix_nnz=n * topk, prefix_rows=pool
                )
                == expected
            )

    @pytest.mark.parametrize("n", [1, 2, 4, 24, 33, 63])
    @pytest.mark.parametrize("topk", [512, 1024])
    def test_auto_policy_keeps_unmeasured_batches(self, n, topk):
        previous = 1 if topk == 512 else 4 if n <= 32 else 2
        assert (
            native._auto_v41_asm_splits(
                n, n * (topk + 128), prefix_nnz=n * topk, prefix_rows=65536
            )
            == previous
        )

    @pytest.mark.parametrize("pool", [None, 512, 32768, 131072, 2097152])
    @pytest.mark.parametrize(
        "n,topk,expected", [(8, 512, 1), (16, 1024, 4), (32, 512, 1), (64, 1024, 2)]
    )
    def test_auto_policy_keeps_unmeasured_pools(self, pool, n, topk, expected):
        assert (
            native._auto_v41_asm_splits(
                n, n * (topk + 128), prefix_nnz=n * topk, prefix_rows=pool
            )
            == expected
        )

    def test_auto_policy_other_prefixes_keep_fallback(self, monkeypatch):
        calls = []
        monkeypatch.setattr(native, "get_cu_num", lambda: 256)
        monkeypatch.setattr(
            public, "_asm_v41_kv_splits", lambda *args: calls.append(args) or 3
        )
        assert (
            native._auto_v41_asm_splits(
                16, 16000, prefix_nnz=16000 - 1, prefix_rows=65536
            )
            == 3
        )
        assert calls == [(16, 16000, 256)]
