# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for fmha_fwd_native_gfx942 via the aiter::mha_fwd() dispatch.

These tests call aiter.mha_fwd() directly — NOT aiter.flash_attn_func() — so
they exercise the fmha_fwd_native_gfx942 C++ function added to mha_fwd.cu,
which is compiled into module_mha_fwd under FAV_NATIVE_ON=1.

Coverage:
  - Positive path: D64 BF16 causal/no-mask, MHA/GQA, square/rectangular,
    batch>1 — verifying the split-K path produces correct output.
  - Fallthrough path: D128 on gfx942 — fmha_fwd_native_gfx942 must return
    -1 (unsupported) so the call falls through to FAV2/CK without raising.
"""

from __future__ import annotations

import math

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx


def _is_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx() == "gfx942"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx942(),
    reason="fmha_fwd_native_gfx942 targets gfx942 only; skip on other arches",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cmp(a: torch.Tensor, b: torch.Tensor, *, rtol: float = 1e-2, atol: float = 1e-2,
         msg: str = "") -> None:
    """fp32 CPU compare — avoids bf16 element-wise hangs after ASM launches."""
    torch.testing.assert_close(
        a.detach().float().cpu(), b.detach().float().cpu(), rtol=rtol, atol=atol, msg=msg
    )


def _mha_fwd(q, k, v, causal: bool):
    """Call aiter.mha_fwd() with no-extras settings, return (out, lse)."""
    out, lse, _, _ = aiter.mha_fwd(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=False,
    )
    return out, lse


def _sdpa_ref(q, k, v, causal: bool):
    """Float32 SDPA reference — expand KV heads for GQA."""
    hq = q.size(2)
    hk = k.size(2)
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    # SDPA expects BHSD; tensors are BSHD.
    # The native kernel aligns the causal diagonal bottom-right (mask_shift =
    # seqlen_k - seqlen_q), so a single decode query attends the full KV cache.
    # torch's is_causal=True aligns top-left (tril diagonal=0), which for sq<sk
    # would mask the query down to key 0. Build the bottom-right mask explicitly.
    attn_mask = None
    if causal:
        sq, sk = q.size(1), k.size(1)
        attn_mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device).tril(
            diagonal=sk - sq
        )
    out = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        attn_mask=attn_mask,
        scale=q.size(-1) ** -0.5,
    )
    return out.transpose(1, 2).bfloat16()


# ---------------------------------------------------------------------------
# Positive-path tests: D64 BF16 → fmha_fwd_native_gfx942 must handle these
# ---------------------------------------------------------------------------

# (batch, seqlen_q, seqlen_k, nhead_q, nhead_k)
_CONFIGS = [
    # MHA square
    (1, 512,  512,  8, 1),
    (1, 1024, 1024, 8, 1),
    # MHA decode (sq << sk)
    (1, 1,    2048, 8, 1),
    (1, 1,    4096, 64, 1),
    # GQA
    (1, 256,  512,  8, 2),
    (1, 1,    2048, 8, 2),
    # batch > 1
    (2, 128,  512,  8, 1),
    (2, 1,    2048, 8, 1),
    # rectangular (sq < sk, non-power-of-two)
    (1, 130,  2048, 8, 1),
    (1, 128,  2300, 8, 1),
    # multi-head decode — many heads drive the split heuristic
    (1, 1,    8192, 64, 8),
]

_CAUSALS = [True, False]


@pytest.mark.parametrize("causal", _CAUSALS)
@pytest.mark.parametrize("b,sq,sk,hq,hk", _CONFIGS)
def test_mha_fwd_native_d64(b, sq, sk, hq, hk, causal):
    if causal and sq > sk:
        pytest.skip("causal + sq>sk produces fully-masked rows — not supported by native path")

    d = 64
    device = torch.device("cuda")
    torch.manual_seed(42)
    q = torch.randn(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)

    out, _ = _mha_fwd(q, k, v, causal)
    ref = _sdpa_ref(q, k, v, causal)

    _cmp(out, ref, msg=f"b={b} sq={sq} sk={sk} hq={hq} hk={hk} causal={causal}")


# ---------------------------------------------------------------------------
# Fallthrough test: D128 must not raise — native path returns -1, FAV2/CK takes over
# ---------------------------------------------------------------------------

_D128_CONFIGS = [
    (1, 512, 512,  8, 1),
    (1, 128, 2048, 8, 1),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("b,sq,sk,hq,hk", _D128_CONFIGS)
def test_mha_fwd_d128_fallthrough(b, sq, sk, hq, hk, causal):
    """D128 must fall through native guard to FAV2/CK without raising."""
    d = 128
    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)

    # Should not raise; if mha_fwd returns -1 all the way it will raise
    # "fused attn configs not supported", so a clean return confirms fallthrough.
    out, _ = _mha_fwd(q, k, v, causal)
    ref = _sdpa_ref(q, k, v, causal)

    _cmp(out, ref, msg=f"D128 fallthrough b={b} sq={sq} sk={sk} causal={causal}")
