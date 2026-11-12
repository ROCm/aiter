# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Metadata-only capability and native-dispatch agreement; no GPU launches."""

import importlib
import inspect
import math
import struct
from types import SimpleNamespace

import pytest
import torch

public = importlib.import_module("aiter.ops.mla_sparse_prefill")
query = public.get_mla_sparse_prefill_v41_backend
execute = inspect.unwrap(public.mla_sparse_prefill_v41)
SCALE = 1 / math.sqrt(512)


def test_query_is_publicly_exported():
    import aiter

    assert "get_mla_sparse_prefill_v41_backend" in public.__all__
    assert aiter.get_mla_sparse_prefill_v41_backend is query


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
    ("gfx1250", 65536, 64, 512, torch.uint8, SCALE, "contiguous", True, "asm"),
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
    ("gfx1250-strict", 8, 64, 512, torch.uint8, SCALE, "contiguous", True, None),
]


@pytest.mark.parametrize("gfx,n,h,d,prefix,scale,layout,asm,expected", CASES)
def test_query_matches_automatic_dispatch(
    runtime, monkeypatch, gfx, n, h, d, prefix, scale, layout, asm, expected
):
    runtime.update(gfx=gfx, asm=asm)
    assert (
        query(
            "cuda:0",
            (n, h, d),
            unified_kv_dtype=prefix,
            softmax_scale=scale,
            layout=layout,
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
    if expected is None:
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
def test_unsupported_quantized_dtype(runtime, monkeypatch, field):
    kwargs = {field: torch.float16}
    assert query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8, **kwargs) is None
    monkeypatch.setattr(
        public._plugin,
        "_dispatch",
        lambda op, canonicalize, native, args, kw: native(*args, **kw),
    )
    with pytest.raises(RuntimeError, match="no native backend"):
        execute(*arguments(**kwargs), SCALE)


def test_query_is_metadata_only(runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("query allocated, compiled, launched or read tensor contents")

    for name in ("empty", "zeros", "full", "tensor"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(public, "_asm_sparse_prefill_v41", forbidden)
    monkeypatch.setattr(public, "_opus_sparse_prefill_v41", forbidden)
    monkeypatch.setattr(public, "compile_ops", forbidden)
    assert query("cuda:0", (65536, 64, 512), unified_kv_dtype=torch.uint8) == "asm"
    runtime["gfx"] = "gfx950"
    assert query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8) == "opus"


def test_fp32_scale_equivalence(runtime):
    fp32_scale = struct.unpack("<f", struct.pack("<f", SCALE))[0]
    assert (
        query(
            "cuda:0",
            (8, 64, 512),
            unified_kv_dtype=torch.uint8,
            softmax_scale=fp32_scale,
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


def test_runtime_availability_and_target_device(runtime, monkeypatch):
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
def test_malformed_shape_raises(runtime, shape):
    with pytest.raises(ValueError):
        query("cuda:0", shape, unified_kv_dtype=torch.uint8)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 1e100])
def test_malformed_scale_raises(runtime, scale):
    with pytest.raises(ValueError, match="finite FP32"):
        query("cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8, softmax_scale=scale)


def test_malformed_metadata_types(runtime):
    with pytest.raises(TypeError, match="integer"):
        query("cuda:0", (True, 64, 512), unified_kv_dtype=torch.uint8)
    with pytest.raises(TypeError, match="torch.dtype"):
        query("cuda:0", (8, 64, 512), unified_kv_dtype="fp4")
    with pytest.raises(TypeError, match="host number"):
        query(
            "cuda:0", (8, 64, 512), unified_kv_dtype=torch.uint8, softmax_scale=object()
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


@pytest.mark.parametrize("backend", ["asm", "opus"])
def test_explicit_backend_errors_propagate(runtime, monkeypatch, backend):
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
def test_provider_path_preserves_v2_contract(runtime, monkeypatch, backend):
    def unexpected(*args, **kwargs):
        pytest.fail("provider path should not probe or execute a native backend")

    monkeypatch.setattr(public, "get_mla_sparse_prefill_v41_backend", unexpected)
    monkeypatch.setattr(public, "_asm_sparse_prefill_v41", unexpected)
    monkeypatch.setattr(public, "_opus_sparse_prefill_v41", unexpected)
    args = (*arguments(), SCALE, None, None)

    def provider(op, canonicalize, native, actual_args, kwargs):
        assert op.name == "mla_sparse_prefill_v41" and op.version == 2
        assert actual_args == args and kwargs == {}
        bound, kw = canonicalize(*actual_args, **kwargs)
        assert len(bound) == 14 and "backend" not in kw
        return "provider"

    monkeypatch.setattr(public._plugin, "_dispatch", provider)
    assert execute(*args, backend=backend) == "provider"
