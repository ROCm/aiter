# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The triton backend of ``mla_sparse_prefill`` on gfx950: its output, and when
``backend=None`` takes it.

Lives in ``tests/`` because CI runs ``python3 -m pytest -q tests`` on pull requests
to main, while ``op_tests/test_mla_sparse_prefill.py`` runs there only as its
default sweep, which never calls these tests. The inputs, the reference and the
arch query are the op test's.
"""

import math
import sys

import pytest
import torch

import aiter.ops.mla_sparse_prefill as msp
from aiter.ops.mla_sparse_prefill import mla_sparse_prefill
from op_tests.test_mla_sparse_prefill import (
    _get_gpu_arch,
    _make_inputs,
    pa_prefill_sparse,
    run_mla_sparse_prefill,
)

_SCALE = 1.0 / math.sqrt(512)


def _require_triton_backend():
    if _get_gpu_arch() != "gfx950" or pa_prefill_sparse is None:
        pytest.skip("the triton backend needs gfx950 and Triton >= 3.6")


def _triton_inputs(h=32):
    """A one-source call the triton backend serves at H = h, with a sink."""
    return _make_inputs(
        n=256,
        h=h,
        d=512,
        total_pages=2048,
        total_tokens=0,
        dtype=torch.bfloat16,
        topk=512,
        seed=42,
    )


@pytest.mark.parametrize(
    "n,h,total_pages,topk",
    [(8, 32, 2048, 512), (64, 32, 0, 128), (256, 32, 2048, 512), (256, 64, 2048, 512)],
    ids=["8-rows", "empty-pool", "h32", "h64"],
)
def test_triton_backend(n, h, total_pages, topk):
    """backend="triton" against the op test's reference."""
    _require_triton_backend()
    row = run_mla_sparse_prefill(
        n=n,
        h=h,
        d=512,
        total_pages=total_pages,
        total_tokens=0,
        prec="bf16",
        topk=topk,
        backends=("triton",),
        bench=False,
    )
    assert "triton err" in row, row


@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
def test_default_backend_gfx950(has_sink, h):
    """backend=None runs triton for a sink-free H=32 or 64 call with one source,
    else opus."""
    _require_triton_backend()
    inputs = _triton_inputs(h)
    if not has_sink:
        inputs["attn_sink"] = None
    expected = mla_sparse_prefill(
        **inputs, softmax_scale=_SCALE, backend="opus" if has_sink else "triton"
    )
    assert torch.equal(mla_sparse_prefill(**inputs, softmax_scale=_SCALE), expected)


def test_triton_backend_rejects():
    """backend="triton" raises for a call the single-pass kernel cannot serve:
    q with inner stride 2, or a unified_kv that is not 2-D."""
    _require_triton_backend()
    inputs = _triton_inputs()
    q = inputs["q"]
    strided_q = torch.empty(
        *q.shape[:2], 2 * q.shape[2], dtype=q.dtype, device=q.device
    )[..., ::2]
    strided_q.copy_(q)
    calls = {
        "unit inner strides": dict(inputs, q=strided_q),
        r"\[total_pages, 512\] unified_kv": dict(
            inputs, unified_kv=inputs["unified_kv"].unsqueeze(1)
        ),
    }
    for match, call in calls.items():
        with pytest.raises(RuntimeError, match=match):
            mla_sparse_prefill(**call, softmax_scale=_SCALE, backend="triton")


def test_default_backend_without_gluon(monkeypatch):
    """backend=None runs opus where pa_prefill_sparse cannot be imported."""
    if _get_gpu_arch() != "gfx950":
        pytest.skip("the triton backend serves gfx950 only")
    inputs = _triton_inputs()
    inputs["attn_sink"] = None
    expected = mla_sparse_prefill(**inputs, softmax_scale=_SCALE, backend="opus")
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(
        sys.modules, "aiter.ops.triton.attention.pa_prefill_sparse", None
    )
    msp._pa_prefill_sparse_module.cache_clear()
    try:
        assert torch.equal(mla_sparse_prefill(**inputs, softmax_scale=_SCALE), expected)
    finally:
        msp._pa_prefill_sparse_module.cache_clear()
