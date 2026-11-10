# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Backend selection of ``mla_sparse_prefill`` on gfx950.

Lives in ``tests/`` because CI runs ``python3 -m pytest -q tests`` on pull requests
to main, while ``op_tests/test_mla_sparse_prefill.py`` runs there only as its
default sweep, which never calls these tests. The inputs and the arch query are
the op test's.
"""

import math
import sys

import pytest
import torch

from aiter.ops.mla_sparse_prefill import mla_sparse_prefill
from op_tests.test_mla_sparse_prefill import (
    _get_gpu_arch,
    _make_inputs,
    pa_prefill_sparse,
)


@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
def test_default_backend_gfx950(has_sink):
    """backend=None runs triton for a sink-free H=32 call with one source, else opus."""
    if _get_gpu_arch() != "gfx950" or pa_prefill_sparse is None:
        pytest.skip("the triton backend needs gfx950 and Triton >= 3.6")
    inputs = _make_inputs(
        n=256,
        h=32,
        d=512,
        total_pages=2048,
        total_tokens=0,
        dtype=torch.bfloat16,
        topk=512,
        seed=42,
    )
    if not has_sink:
        inputs["attn_sink"] = None
    scale = 1.0 / math.sqrt(512)
    expected = mla_sparse_prefill(
        **inputs, softmax_scale=scale, backend="opus" if has_sink else "triton"
    )
    assert torch.equal(mla_sparse_prefill(**inputs, softmax_scale=scale), expected)


def test_triton_backend_raises_where_the_single_pass_kernel_cannot_serve():
    """backend="triton" raises for non-unit inner strides and for fewer than
    MIN_TOKENS query rows, where pa_prefill_sparse would run mla_gluon."""
    if _get_gpu_arch() != "gfx950" or pa_prefill_sparse is None:
        pytest.skip("the triton backend needs gfx950 and Triton >= 3.6")
    kw = {
        "h": 32,
        "d": 512,
        "total_pages": 2048,
        "total_tokens": 0,
        "dtype": torch.bfloat16,
        "topk": 512,
        "seed": 42,
    }
    scale = 1.0 / math.sqrt(512)
    few = _make_inputs(n=8, **kw)
    with pytest.raises(RuntimeError, match="query rows"):
        mla_sparse_prefill(**few, softmax_scale=scale, backend="triton")
    strided = _make_inputs(n=256, **kw)
    wide = torch.randn(256, 32, 1024, dtype=torch.bfloat16, device=strided["q"].device)
    strided["q"] = wide[..., ::2]
    with pytest.raises(RuntimeError, match="unit inner strides"):
        mla_sparse_prefill(**strided, softmax_scale=scale, backend="triton")


def test_default_backend_keeps_opus_without_the_triton_kernels(monkeypatch):
    """Where pa_prefill_sparse cannot be imported (Triton < 3.6), backend=None
    stays on opus instead of raising."""
    if _get_gpu_arch() != "gfx950":
        pytest.skip("the triton backend is gfx950's")
    inputs = _make_inputs(
        n=256,
        h=32,
        d=512,
        total_pages=2048,
        total_tokens=0,
        dtype=torch.bfloat16,
        topk=512,
        seed=42,
    )
    inputs["attn_sink"] = None
    scale = 1.0 / math.sqrt(512)
    expected = mla_sparse_prefill(**inputs, softmax_scale=scale, backend="opus")
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(
        sys.modules, "aiter.ops.triton.attention.pa_prefill_sparse", None
    )
    assert torch.equal(mla_sparse_prefill(**inputs, softmax_scale=scale), expected)
