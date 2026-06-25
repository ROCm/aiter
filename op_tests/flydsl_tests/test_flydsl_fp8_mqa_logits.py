# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness harness for the FlyDSL gfx942 FP8 MQA logits kernel.

Compares ``flydsl_fp8_mqa_logits`` against the torch reference
``ref_fp8_mqa_logits`` (the same DeepGEMM-derived reference and ``calc_diff``
metric used by the Triton test) over the DeepSeek dims and window shapes.

While the FlyDSL kernel is a scaffolding placeholder it raises
``NotImplementedError``; this harness marks those as ``xfail`` so the suite
stays green and flips to real pass/fail automatically once the kernel lands.

Run with:
    python3 -m pytest op_tests/flydsl_tests/test_flydsl_fp8_mqa_logits.py -q
"""
import pytest
import torch

from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits

# Reuse the verified reference, fp8 cast, masking helpers and CP data generator
# from the Triton test so both kernels are graded against an identical bar.
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    calc_diff,
    per_custom_dims_cast_to_fp8,
    ref_fp8_mqa_logits,
    generate_cp_test_data,
    e4m3_type,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_flydsl_available()),
    reason="requires a GPU and an installed, gfx-supported flydsl package",
)


def _flydsl_fp8_mqa_logits():
    # Imported lazily so the module still collects when flydsl is absent.
    from aiter.ops.flydsl import flydsl_fp8_mqa_logits

    return flydsl_fp8_mqa_logits


@pytest.mark.parametrize(
    "s_q, s_k",
    [
        (1, 1),
        (1, 16),
        (1, 113),
        (17, 76),
        (61, 113),
        (61, 1024),
        (128, 1024),
        (1024, 1024),
        (1024, 1560),
    ],
)
@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("disable_cp", [True, False])
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    disable_cp: bool,
    clean_logits: bool,
) -> None:
    flydsl_fp8_mqa_logits = _flydsl_fp8_mqa_logits()

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)

    if disable_cp or s_k % s_q != 0 or s_q % 2 != 0:
        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)
    else:
        ks, ke = generate_cp_test_data(s_q, s_k)

    q_fp8 = q.to(e4m3_type)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref_logits, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )

    try:
        logits = flydsl_fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits
        )
    except NotImplementedError as exc:
        pytest.xfail(f"FlyDSL kernel not implemented yet: {exc}")

    def _rehydrate(out):
        # When clean_logits is off, the kernel only writes the in-window region;
        # rebuild the full [s_q, s_k] grid with -inf outside (matches Triton test).
        if clean_logits:
            return out
        assert out.size() == (s_q, s_k)
        full = torch.full((s_q, s_k), float("-inf"), device="cuda")
        for i in range(s_q):
            full[i, ks[i] : ke[i]] = out[i, : ke[i] - ks[i]]
        return full

    logits = _rehydrate(logits)

    # 1) Verify against the torch reference (ground-truth bar, calc_diff < 1e-3).
    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask), "FlyDSL vs torch-ref mask"
    if not ref_neginf_mask.all():
        diff = calc_diff(
            logits.masked_fill(neginf_mask, 0),
            ref_logits.masked_fill(ref_neginf_mask, 0),
        )
        assert diff < 1e-3, f"FlyDSL vs torch-ref {diff=}"

    # 2) Cross-check against the Triton kernel (ticket asks for both). Same fp8
    #    inputs, same clean_logits, identical post-processing.
    tri_logits = _rehydrate(
        triton_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)
    )
    tri_neginf_mask = tri_logits == float("-inf")
    assert torch.equal(neginf_mask, tri_neginf_mask), "FlyDSL vs Triton mask"
    if not tri_neginf_mask.all():
        diff_tri = calc_diff(
            logits.masked_fill(neginf_mask, 0),
            tri_logits.masked_fill(tri_neginf_mask, 0),
        )
        assert diff_tri < 1e-3, f"FlyDSL vs Triton {diff_tri=}"


# ---------------------------------------------------------------------------
# i32 byte-offset overflow regression test
#
# For row*stride*sizeof(f32) > 2^31-1 (i.e. row*stride > 536870912), the
# output element offset computation overflows signed i32 in buffer_store,
# causing silent writes to the wrong address.  The minimum shape that
# triggers this has s_q=16385, s_kv=32768 (stride=32768): the element at
# [16384, 32767] has byte offset 16384*32768*4 = 2147483648 = 2^31, which
# overflows i32.  The output tensor is ~2.1 GB; the test is skipped if
# insufficient GPU memory is available.
# ---------------------------------------------------------------------------

_OVERFLOW_MIN_FREE_GB = 3.0  # headroom above the 2.15 GB tensor itself


def _has_enough_vram(gb: float) -> bool:
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free >= gb * 1e9


@pytest.mark.skipif(
    not _has_enough_vram(_OVERFLOW_MIN_FREE_GB),
    reason=f"requires >{_OVERFLOW_MIN_FREE_GB} GB free GPU VRAM",
)
@pytest.mark.parametrize("num_heads,head_dim", [(64, 128)])
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits_i32_overflow(num_heads: int, head_dim: int) -> None:
    """Regression: i32 byte-offset overflow for row*stride*4 > 2^31-1.

    Uses the minimum shape that overflows (s_q=16385, s_kv=32768) and checks
    that the at-risk element [s_q-1, s_kv-1] is written correctly.
    """
    flydsl_fp8_mqa_logits = _flydsl_fp8_mqa_logits()

    # s_kv=32768 → stride=32768 (256-aligned); row=16384 → row*stride*4=2^31 (overflow).
    s_q, s_kv = 16385, 32768
    torch.manual_seed(42)

    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv_ref = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)

    # Causal-style window: row i covers KV [0, i+1).  The last row (16384)
    # covers the full [0, 32768) range, so col=32767 must be written.
    # But that row's KV window is [0, 16385), clamped by the causal rule:
    # use ke[i] = min(i + 1 + (s_kv - s_q), s_kv) = i + (s_kv - s_q + 1).
    ks = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    ke = torch.clamp(
        torch.arange(s_q, dtype=torch.int32, device="cuda") + (s_kv - s_q + 1),
        max=s_kv,
    )
    # Verify the last row reaches col s_kv-1.
    assert ke[-1].item() == s_kv, f"last ke={ke[-1].item()}, expected {s_kv}"

    q_fp8 = q.to(e4m3_type)

    # Use Triton as reference — it handles large shapes without materializing
    # the full [H, s_q, s_kv] intermediate that ref_fp8_mqa_logits would need.
    tri_logits = triton_logits(q_fp8, kv_fp8, scales, weights, ks, ke,
                               clean_logits=True)

    try:
        logits = flydsl_fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke,
                                        clean_logits=True)
    except NotImplementedError as exc:
        pytest.xfail(f"FlyDSL kernel not implemented yet: {exc}")

    # Primary check: the at-risk element (last row, last col) must not be -inf
    # and must match Triton.  Before the fix this position was silently written
    # to the wrong address and remained -inf (the clean_logits pre-fill value).
    at_risk_tri = tri_logits[-1, -1].item()
    at_risk_got = logits[-1, -1].item()
    assert at_risk_got != float("-inf"), (
        "i32 overflow regression: logits[-1, -1] is -inf (element not written)"
    )
    assert abs(at_risk_got - at_risk_tri) / (abs(at_risk_tri) + 1e-6) < 1e-2, (
        f"i32 overflow regression: logits[-1,-1]={at_risk_got:.4f} "
        f"triton={at_risk_tri:.4f}"
    )

    # Broader check: full output matches Triton within tolerance.
    # Mask -inf positions (outside each row's window) before calc_diff to avoid
    # NaN from inf arithmetic inside the similarity formula.
    neginf_mask = tri_logits == float("-inf")
    diff = calc_diff(
        logits.masked_fill(neginf_mask, 0),
        tri_logits.masked_fill(neginf_mask, 0),
    )
    assert diff < 1e-3, f"i32 overflow shape: calc_diff vs Triton={diff:.2e}"
