# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness tests for remap_workgroup_spatial (AITER_TRITON_MHA_SWIZZLE="spatial").

Verifies that swizzle="spatial" produces bit-identical output to swizzle="default"
across GQA and MHA configurations, causal and non-causal, single and batched.

Configuration:
    Environment variable : AITER_TRITON_MHA_SWIZZLE  (values: "default", "spatial")
    Programmatic API     : mha_set_swizzle("default" | "spatial")
"""

import math

import pytest
import torch

from aiter.ops.triton.attention.mha import flash_attn_func, mha_set_swizzle


@pytest.fixture(autouse=True)
def restore_swizzle():
    """Save swizzle state before each test and restore it on teardown."""
    from aiter.ops.triton.attention import mha as mha_mod

    saved = mha_mod._MHA_SWIZZLE
    yield
    mha_set_swizzle(saved)


def _run(swizzle: str, B: int, Hq: int, Hk: int, S: int, D: int, causal: bool):
    mha_set_swizzle(swizzle)
    torch.manual_seed(42)
    q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v, softmax_scale=1.0 / math.sqrt(D), causal=causal)
    torch.cuda.synchronize()
    return (out[0] if isinstance(out, tuple) else out).float()


# ---------------------------------------------------------------------------
# GQA configs — spatial_gqa path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "Hq,Hk",
    [
        (128, 8),  # aligned HK == NUM_XCDS
        (64, 8),
        (32, 8),
    ],
)
def test_spatial_gqa_aligned(Hq, Hk, causal):
    """HK == NUM_XCDS (8): one KV head per XCD."""
    out0 = _run("default", 1, Hq, Hk, 8192, 128, causal)
    out1 = _run("spatial", 1, Hq, Hk, 8192, 128, causal)
    assert torch.equal(out0, out1), f"HQ={Hq} HK={Hk} causal={causal}: outputs differ"


@pytest.mark.parametrize(
    "Hq,Hk",
    [
        (128, 16),  # HK > NUM_XCDS
        (128, 32),
    ],
)
def test_spatial_gqa_hk_gt_nxcd(Hq, Hk):
    """HK > NUM_XCDS: each XCD owns multiple KV heads."""
    out0 = _run("default", 1, Hq, Hk, 8192, 128, causal=True)
    out1 = _run("spatial", 1, Hq, Hk, 8192, 128, causal=True)
    assert torch.equal(out0, out1), f"HQ={Hq} HK={Hk}: outputs differ"


@pytest.mark.parametrize(
    "Hq,Hk",
    [
        (128, 4),  # HK < NUM_XCDS
        (128, 2),
    ],
)
def test_spatial_gqa_hk_lt_nxcd(Hq, Hk):
    """HK < NUM_XCDS: multiple XCDs share each KV head."""
    out0 = _run("default", 1, Hq, Hk, 8192, 128, causal=True)
    out1 = _run("spatial", 1, Hq, Hk, 8192, 128, causal=True)
    assert torch.equal(out0, out1), f"HQ={Hq} HK={Hk}: outputs differ"


@pytest.mark.parametrize("causal", [True, False])
def test_spatial_gqa_batched(causal):
    """GQA aligned, batch > 1."""
    out0 = _run("default", 4, 128, 8, 8192, 128, causal)
    out1 = _run("spatial", 4, 128, 8, 8192, 128, causal)
    assert torch.equal(out0, out1), f"causal={causal}: batched GQA outputs differ"


# ---------------------------------------------------------------------------
# MHA configs — spatial_mha path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("H", [16, 32, 64, 128])
def test_spatial_mha(H, causal):
    """MHA: HQ == HK, various head counts."""
    out0 = _run("default", 1, H, H, 8192, 128, causal)
    out1 = _run("spatial", 1, H, H, 8192, 128, causal)
    assert torch.equal(out0, out1), f"H={H} causal={causal}: outputs differ"


def test_spatial_mha_nonpow2_heads():
    """MHA with non-power-of-2 head count (e.g. Wan2.2-14B DiT, H=40)."""
    out0 = _run("default", 1, 40, 40, 8192, 128, causal=True)
    out1 = _run("spatial", 1, 40, 40, 8192, 128, causal=True)
    assert torch.equal(out0, out1), "H=40 (Wan): outputs differ"


def test_spatial_mha_batched():
    """MHA, batch > 1."""
    out0 = _run("default", 2, 128, 128, 8192, 128, causal=True)
    out1 = _run("spatial", 2, 128, 128, 8192, 128, causal=True)
    assert torch.equal(out0, out1), "batched MHA outputs differ"
