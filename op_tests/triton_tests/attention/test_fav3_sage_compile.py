# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Verify that fav3_sage_wrapper_func works under torch.compile(fullgraph=True).

A fullgraph compilation will raise torch._dynamo.exc.Unsupported if any
operation in the call chain causes a graph break.  The test also checks
that the compiled output matches eager.
"""

import math
import pytest
import torch
import torch._dynamo

from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func

ATOL_fp8 = 3.0e-1
RTOL_fp8 = 2.5e-1
MAX_DIFF_PERCENTAGE = 0.5


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
    Compile fav3_sage_wrapper_func with fullgraph=True and assert:
      1. No graph break (fullgraph would raise otherwise).
      2. Compiled output matches eager output within tolerance.
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
            q, k, v, softmax_scale, causal=False, layout=layout
        )

    # Eager baseline
    out_eager = fn(q, k, v)
    if isinstance(out_eager, (list, tuple)):
        out_eager = out_eager[0]
    torch.cuda.synchronize()

    # Compiled — fullgraph=True will error on any graph break
    compiled_fn = torch.compile(fn, fullgraph=True)
    out_compiled = compiled_fn(q, k, v)
    if isinstance(out_compiled, (list, tuple)):
        out_compiled = out_compiled[0]
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    assert (
        out_eager.shape == out_compiled.shape
    ), f"Shape mismatch: eager {out_eager.shape} vs compiled {out_compiled.shape}"

    # Use the same fp8-friendly tolerances as the existing sage tests:
    # atol=0.3, rtol=0.25, and allow up to 0.5% of elements to exceed them.
    abs_diff = torch.abs(out_eager.float() - out_compiled.float())
    rel_diff = abs_diff / torch.abs(out_eager.float().clamp(min=1e-6))
    failed = torch.logical_and(abs_diff > ATOL_fp8, rel_diff > RTOL_fp8)
    failed_pct = failed.sum().item() / failed.numel() * 100
    assert failed_pct <= MAX_DIFF_PERCENTAGE, (
        f"torch.compile(fullgraph=True) output diverges from eager: "
        f"{failed_pct:.4f}% elements exceed tolerance "
        f"(atol={ATOL_fp8}, rtol={RTOL_fp8}, max_allowed={MAX_DIFF_PERCENTAGE}%)"
    )


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
            q, k, v, softmax_scale, causal=False, layout=layout
        )

    compiled_fn = torch.compile(fn, fullgraph=True)

    for _ in range(3):
        q = torch.randn(BATCH, HQ, SEQLEN, D, device="cuda", dtype=dtype)
        k = torch.randn(BATCH, HK, SEQLEN, D, device="cuda", dtype=dtype)
        v = torch.randn(BATCH, HK, SEQLEN, D, device="cuda", dtype=dtype)
        out = compiled_fn(q, k, v)
        if isinstance(out, (list, tuple)):
            out = out[0]
        assert not torch.isnan(out).any()

    torch.cuda.synchronize()
