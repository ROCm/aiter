# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and API tests for fmha_fwd_native_gfx942 / mha_fwd_calculate_num_splits.

These tests call aiter.mha_fwd() directly — NOT aiter.flash_attn_func() — so
they exercise the native gfx942 split-K path compiled into module_mha_fwd under
FAV_NATIVE_ON=1.

Coverage:
  - mha_fwd_calculate_num_splits eligibility / heuristic:
      * Returns > 0 for eligible D64 BF16 configs where split-K is beneficial.
      * Returns  0 for eligible configs where split-K is not beneficial.
      * Returns -1 for ineligible configs (D128, wrong arch, etc.).
  - Positive path with caller-allocated workspace:
      D64 BF16 causal/no-mask, MHA/GQA, square/rectangular, batch>1.
  - Fallthrough path: D128 on gfx942 → native guard returns -1 → FAV2/CK.
"""

from __future__ import annotations

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

# kHeadDim constant from runner/params.hpp — must match the C++ definition.
_NATIVE_HEAD_DIM: int = 64


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


def _mha_fwd_workspace_size(num_splits: int, q: torch.Tensor) -> int:
    """Workspace buffer size in bytes for the native split-K kernel.

    Mirrors aiter::mha_fwd_workspace_size() declared in csrc/include/mha_fwd.h:
      num_splits * batch * nhead_q * seqlen_q * (kHeadDim + 1) * sizeof(float)
    Returns 0 for num_splits <= 0.
    """
    if num_splits <= 0:
        return 0
    B, Sq, Hq, _ = q.shape
    return num_splits * B * Hq * Sq * (_NATIVE_HEAD_DIM + 1) * 4  # *4 for sizeof(float)


def _mha_fwd_calculate_num_splits(q: torch.Tensor, k: torch.Tensor, causal: bool) -> int:
    """Mirror of aiter::mha_fwd_calculate_num_splits() in csrc/cpp_itfs/mha_fwd.cu.

    Applies the same eligibility guards then delegates to the Python heuristic
    (_native_splitkv_heuristic in aiter/ops/mha.py) which mirrors the C++ version.

    Returns:
        -1  native split-K is not eligible for this config
         0  eligible but heuristic judges split-K not beneficial
        >0  optimal split count
    """
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.mha import _native_splitkv_heuristic

    # Eligibility guards (mirrors native_splitkv_check_eligigibility in mha_fwd.cu).
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return -1
    if q.shape[-1] != _NATIVE_HEAD_DIM or k.shape[-1] != _NATIVE_HEAD_DIM:
        return -1
    B, Sq, Hq, _ = q.shape
    _, Sk, Hk, _ = k.shape
    if Hq % Hk != 0:
        return -1
    if causal and Sq > Sk:
        return -1

    # All guards passed — run the heuristic (0 means not beneficial).
    return _native_splitkv_heuristic(B, Hq, Sq, Sk, get_cu_num())


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


def _mha_fwd_with_workspace(q, k, v, causal: bool, num_splits: int):
    """Call aiter.mha_fwd() with an explicit caller-allocated workspace.

    Demonstrates the intended usage pattern for the new API:
      1. Call _mha_fwd_calculate_num_splits() to check eligibility.
      2. Allocate workspace using _mha_fwd_workspace_size().
      3. Pass num_splits + workspace pointer to mha_fwd().
    """
    ws_bytes = _mha_fwd_workspace_size(num_splits, q)
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=q.device)
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
        num_splits=num_splits,
        splitkv_workspace=workspace,
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
# mha_fwd_calculate_num_splits() API tests
# ---------------------------------------------------------------------------

# Configs where split-K should be eligible (D64 BF16 batch mode, gfx942).
_ELIGIBLE_CONFIGS = [
    # decode-like: few Q tiles → heuristic strongly prefers splits
    (1, 1,    2048, 8,  1),
    (1, 1,    4096, 64, 1),
    (1, 1,    8192, 64, 8),
]

# Configs that are ineligible by guard (wrong dtype / hdim / varlen / etc.).
_INELIGIBLE_CONFIGS_D128 = [
    (1, 512, 512, 8, 1),
    (1, 128, 2048, 8, 1),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("b,sq,sk,hq,hk", _ELIGIBLE_CONFIGS)
def test_get_num_splits_eligible(b, sq, sk, hq, hk, causal):
    """_mha_fwd_calculate_num_splits() must return >= 0 for D64 BF16 eligible configs.

    The exact value depends on the hardware occupancy; we only assert >= 0.
    A return of 0 means 'feasible but not beneficial' (acceptable).
    """
    d = _NATIVE_HEAD_DIM
    device = torch.device("cuda")
    q = torch.empty(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.empty(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    result = _mha_fwd_calculate_num_splits(q, k, causal)
    assert result >= 0, (
        f"Expected >= 0 for eligible D64 BF16 config, got {result} "
        f"(b={b} sq={sq} sk={sk} hq={hq} hk={hk} causal={causal})"
    )


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("b,sq,sk,hq,hk", _ELIGIBLE_CONFIGS)
def test_get_num_splits_decode_positive(b, sq, sk, hq, hk, causal):
    """For decode-like shapes (sq=1, large sk, many heads), expect G > 0."""
    if causal and sq > sk:
        pytest.skip("causal + sq>sk is guard-rejected")
    d = _NATIVE_HEAD_DIM
    device = torch.device("cuda")
    q = torch.empty(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.empty(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    result = _mha_fwd_calculate_num_splits(q, k, causal)
    assert result > 0, (
        f"Expected G > 0 for decode-like shape, got {result} "
        f"(b={b} sq={sq} sk={sk} hq={hq} hk={hk} causal={causal})"
    )


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("b,sq,sk,hq,hk", _INELIGIBLE_CONFIGS_D128)
def test_get_num_splits_d128_returns_minus1(b, sq, sk, hq, hk, causal):
    """_mha_fwd_calculate_num_splits() must return -1 for D128 (guard: hdim != 64)."""
    d = 128
    device = torch.device("cuda")
    q = torch.empty(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.empty(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    result = _mha_fwd_calculate_num_splits(q, k, causal)
    assert result == -1, (
        f"Expected -1 for D128 (ineligible), got {result} "
        f"(b={b} sq={sq} sk={sk} hq={hq} hk={hk} causal={causal})"
    )


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

    d = _NATIVE_HEAD_DIM
    device = torch.device("cuda")
    torch.manual_seed(42)
    q = torch.randn(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)

    out, _ = _mha_fwd(q, k, v, causal)
    ref = _sdpa_ref(q, k, v, causal)

    _cmp(out, ref, msg=f"b={b} sq={sq} sk={sk} hq={hq} hk={hk} causal={causal}")


@pytest.mark.parametrize("causal", _CAUSALS)
@pytest.mark.parametrize("b,sq,sk,hq,hk", _CONFIGS)
def test_mha_fwd_native_d64_explicit_workspace(b, sq, sk, hq, hk, causal):
    """Same correctness check but uses the new caller-allocated workspace API.

    Pattern:
      1. _mha_fwd_calculate_num_splits() → G
      2. Skip if G <= 0 (not beneficial or ineligible for this shape).
      3. Allocate workspace via _mha_fwd_workspace_size(), call mha_fwd() with
         num_splits=G + workspace.
      4. Compare against SDPA reference.
    """
    if causal and sq > sk:
        pytest.skip("causal + sq>sk produces fully-masked rows — not supported by native path")

    d = _NATIVE_HEAD_DIM
    device = torch.device("cuda")
    torch.manual_seed(42)
    q = torch.randn(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)

    G = _mha_fwd_calculate_num_splits(q, k, causal)
    if G <= 0:
        pytest.skip(f"_mha_fwd_calculate_num_splits returned {G} — native split-K not active")

    out, _ = _mha_fwd_with_workspace(q, k, v, causal, G)
    ref = _sdpa_ref(q, k, v, causal)

    _cmp(out, ref, msg=f"explicit-workspace b={b} sq={sq} sk={sk} hq={hq} hk={hk} causal={causal} G={G}")


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
    """D128: _mha_fwd_calculate_num_splits() returns -1 and mha_fwd() falls through to FAV2/CK."""
    d = 128
    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(b, sq, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device=device)

    # Verify the local helper correctly reports ineligibility.
    G = _mha_fwd_calculate_num_splits(q, k, causal)
    assert G == -1, f"Expected -1 for D128, got {G}"

    # Verify mha_fwd() still works (falls through to FAV2/CK) without raising.
    out, _ = _mha_fwd(q, k, v, causal)
    ref = _sdpa_ref(q, k, v, causal)

    _cmp(out, ref, msg=f"D128 fallthrough b={b} sq={sq} sk={sk} causal={causal}")
