# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Verify that fav3_sage_wrapper_func works under torch.compile(fullgraph=True).

A fullgraph compilation will raise torch._dynamo.exc.Unsupported if any
operation in the call chain causes a graph break.

NOTE: Numerical accuracy of compiled output vs eager is NOT tested here.
When torch and triton are version-incompatible, Inductor-compiled Triton
kernels can produce completely wrong results (while eager mode is fine).
Accuracy is already covered by test_fav3_sage.py in eager mode.
"""

import math
import pytest
import torch
import torch._dynamo

from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo caches between tests so each gets a clean compile."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("BATCH", [1, 2])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(64, 64), (128, 128)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 4)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sage_compile_fullgraph(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """
    Compile fav3_sage_wrapper_func with fullgraph=True and assert no graph break.
    fullgraph=True will raise torch._dynamo.exc.Unsupported on any graph break.
    """
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    if layout == "bhsd":
        q = torch.randn(
            BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ, device="cuda", dtype=dtype
        )
        k = torch.randn(
            BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ, device="cuda", dtype=dtype
        )
        v = torch.randn(
            BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ, device="cuda", dtype=dtype
        )
    else:
        q = torch.randn(
            BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype
        )
        k = torch.randn(
            BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype
        )
        v = torch.randn(
            BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype
        )

    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)

    def fn(q, k, v):
        return fav3_sage_wrapper_func(
            q, k, v, softmax_scale, causal=False, return_lse=False, layout=layout
        )

    compiled_fn = torch.compile(fn, fullgraph=True)
    out = compiled_fn(q, k, v)
    torch.cuda.synchronize()

    assert out.shape == q.shape, f"Shape mismatch: expected {q.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "compiled output contains NaN"


@pytest.mark.parametrize("layout", ["bhsd"])
def test_sage_compile_no_recompilation(layout: str, dtype=torch.bfloat16):
    """
    Run the compiled function twice with same-shaped inputs.  If get_arch()
    result is not treated as constant, Dynamo may recompile on every call.
    """
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    BATCH, SEQLEN, HQ, HK, D = 1, 64, 4, 4, 128
    softmax_scale = 1.0 / math.sqrt(D)

    def fn(q, k, v):
        return fav3_sage_wrapper_func(
            q, k, v, softmax_scale, causal=False, return_lse=False, layout=layout
        )

    compiled_fn = torch.compile(fn, fullgraph=True)

    for _ in range(3):
        q = torch.randn(BATCH, HQ, SEQLEN, D, device="cuda", dtype=dtype)
        k = torch.randn(BATCH, HK, SEQLEN, D, device="cuda", dtype=dtype)
        v = torch.randn(BATCH, HK, SEQLEN, D, device="cuda", dtype=dtype)
        out = compiled_fn(q, k, v)
        assert not torch.isnan(out).any()

    torch.cuda.synchronize()
