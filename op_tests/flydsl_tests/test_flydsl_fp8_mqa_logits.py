# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness harness for the FlyDSL gfx942 FP8 MQA logits kernel.

Compares ``flydsl_fp8_mqa_logits`` against the torch reference
``ref_fp8_mqa_logits`` (the same DeepGEMM-derived reference and ``calc_diff``
metric used by the Triton test) over the DeepSeek dims and window shapes.

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
# Cover both the native fp8 (fnuz on gfx942) and the e4m3 FN path, which on
# gfx942 exercises the in-kernel FN->FNUZ byte conversion + kv_scales x2.
@pytest.mark.parametrize(
    "q_fp8_dtype", [e4m3_type, torch.float8_e4m3fn], ids=["qfnuz", "qfn"]
)
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    disable_cp: bool,
    clean_logits: bool,
    q_fp8_dtype: torch.dtype,
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

    q_fp8 = q.to(q_fp8_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref_logits, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )

    logits = flydsl_fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

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


@pytest.mark.parametrize("s_q, s_k", [(64, 512), (128, 1024), (61, 1024)])
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize(
    "q_fp8_dtype", [e4m3_type, torch.float8_e4m3fn], ids=["qfnuz", "qfn"]
)
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits_windowed_starts(
    s_q: int, s_k: int, clean_logits: bool, q_fp8_dtype: torch.dtype
) -> None:
    """Per-row windows with NONZERO, BKV-misaligned starts.

    Exercises the writer's ``col >= start`` bound and the BKV-down-aligned tile
    loop (the indexer's general ``[cu_starts, cu_ends)`` contract); the main test
    only covers ``start == 0``.
    """
    flydsl_fp8_mqa_logits = _flydsl_fp8_mqa_logits()
    num_heads, head_dim = 64, 128

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)

    # Nonzero, per-row-varying, intentionally BKV(=128)-misaligned windows.
    rows = torch.arange(s_q, device="cuda")
    ks = ((rows * 53 + 100) % max(1, s_k // 2)).to(torch.int32)
    ke = torch.minimum(ks + max(1, s_k // 3), torch.full_like(ks, s_k)).to(torch.int32)

    q_fp8 = q.to(q_fp8_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    ref_logits, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    logits = flydsl_fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

    if not clean_logits:
        # clean_logits=False writes in-window logits at their real column
        # positions and leaves the rest untouched; mask the out-of-window region.
        full = torch.full((s_q, s_k), float("-inf"), device="cuda")
        for i in range(s_q):
            full[i, ks[i] : ke[i]] = logits[i, ks[i] : ke[i]]
        logits = full

    ref_mask = ref_logits == float("-inf")
    assert torch.equal(logits == float("-inf"), ref_mask), "windowed -inf mask"
    if not ref_mask.all():
        diff = calc_diff(
            logits.masked_fill(ref_mask, 0), ref_logits.masked_fill(ref_mask, 0)
        )
        assert diff < 1e-3, f"windowed FlyDSL vs torch-ref {diff=}"


@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 7, 16])
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize(
    "q_fp8_dtype", [e4m3_type, torch.float8_e4m3fn], ids=["qfnuz", "qfn"]
)
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits_column_split(
    num_splits: int, clean_logits: bool, q_fp8_dtype: torch.dtype, monkeypatch
) -> None:
    """Force the grid.y KV-column split (incl. non-divisor counts).

    Each block owns a BKV-aligned slice of the row window; the slices must tile
    ``[cu_starts, cu_ends)`` with no gaps and no double-writes. Pinning
    ``_auto_num_splits`` lets us exercise that independent of the shape heuristic
    (the production tests above only hit the ``num_splits == 1`` collapse).
    """
    import aiter.ops.flydsl.kernels.fp8_mqa_logits as kmod

    monkeypatch.setattr(kmod, "_auto_num_splits", lambda *a, **k: num_splits)
    flydsl_fp8_mqa_logits = _flydsl_fp8_mqa_logits()

    s_q, s_k, num_heads, head_dim = 70, 2048, 64, 128
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)

    # Nonzero, per-row-varying, BKV(=128)-misaligned windows.
    rows = torch.arange(s_q, device="cuda")
    ks = ((rows * 53 + 100) % (s_k // 2)).to(torch.int32)
    ke = torch.minimum(ks + s_k // 3, torch.full_like(ks, s_k)).to(torch.int32)

    q_fp8 = q.to(q_fp8_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    ref_logits, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    logits = flydsl_fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

    if not clean_logits:
        full = torch.full((s_q, s_k), float("-inf"), device="cuda")
        for i in range(s_q):
            full[i, ks[i] : ke[i]] = logits[i, ks[i] : ke[i]]
        logits = full

    ref_mask = ref_logits == float("-inf")
    assert torch.equal(logits == float("-inf"), ref_mask), "split -inf mask"
    diff = calc_diff(
        logits.masked_fill(ref_mask, 0), ref_logits.masked_fill(ref_mask, 0)
    )
    assert diff < 1e-3, f"column-split FlyDSL vs torch-ref {diff=}"


def _kv_in_dtype(kv_fp8_fnuz, kv_dtype):
    """Re-express fnuz-quantized KV in ``kv_dtype``.

    For FN, store the fnuz numeric values in the FN grid; the kernel reads FN
    bytes as fnuz (= value/2) and compensates with ``kv_scales *= 2``, so the
    dequantized KV still matches the reference.
    """
    if kv_dtype == e4m3_type:
        return kv_fp8_fnuz
    return kv_fp8_fnuz.to(torch.float32).to(kv_dtype)


@pytest.mark.parametrize("s_q, s_k", [(128, 1024), (61, 1024)])
@pytest.mark.parametrize("clean_logits", [True, False])
# q in {fnuz, fn} x kv = fn covers the convert_kv_fn path with scale_mul 2 (q
# fnuz) and 4 (q fn) -- the all-fnuz path is already covered by the main test.
@pytest.mark.parametrize(
    "q_fp8_dtype", [e4m3_type, torch.float8_e4m3fn], ids=["qfnuz", "qfn"]
)
@pytest.mark.parametrize("kv_fp8_dtype", [torch.float8_e4m3fn], ids=["kvfn"])
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits_fn_kv(
    s_q, s_k, clean_logits, q_fp8_dtype, kv_fp8_dtype
) -> None:
    """FP8 FN KV operand: exercises ``convert_kv_fn`` + the ``kv_scales`` x2/x4
    compensation that the main test (always-fnuz KV) never hits."""
    flydsl_fp8_mqa_logits = _flydsl_fp8_mqa_logits()
    num_heads, head_dim = 64, 128

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)

    q_fp8 = q.to(q_fp8_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv_fp8 = _kv_in_dtype(kv_fp8, kv_fp8_dtype)
    assert kv_fp8.dtype == kv_fp8_dtype

    ref_logits, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    logits = flydsl_fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

    if not clean_logits:
        full = torch.full((s_q, s_k), float("-inf"), device="cuda")
        for i in range(s_q):
            full[i, ks[i] : ke[i]] = logits[i, : ke[i] - ks[i]]
        logits = full

    ref_mask = ref_logits == float("-inf")
    assert torch.equal(logits == float("-inf"), ref_mask), "fn-kv mask"
    if not ref_mask.all():
        diff = calc_diff(
            logits.masked_fill(ref_mask, 0), ref_logits.masked_fill(ref_mask, 0)
        )
        assert diff < 1e-3, f"fn-kv FlyDSL vs torch-ref {diff=}"


def test_auto_num_splits_unit(monkeypatch) -> None:
    """Direct unit test of the split heuristic (pure host arithmetic).

    The split tests above monkeypatch ``_auto_num_splits`` wholesale and the
    other shapes all have ``seq_len_kv < 4096``, so the heuristic's own
    ceil/clamp logic is otherwise never executed. Pin the CU count for
    determinism (304 -> target 1216 blocks).
    """
    import aiter.ops.flydsl.kernels.fp8_mqa_logits as kmod

    monkeypatch.setattr(kmod, "_device_cu_count", lambda idx: 304)
    f = kmod._auto_num_splits

    assert f(1024, 2048, 2, 0) == 1     # window too short (< 4096)
    assert f(64, 4095, 2, 0) == 1       # just under the length threshold
    assert f(0, 8192, 2, 0) == 1        # grid_x == 0 guard (no ZeroDivisionError)
    assert f(4096, 8192, 2, 0) == 1     # row grid already saturates (2048 >= 1216)
    assert f(64, 4096, 2, 0) == 4       # tiny grid; capped by max_splits=4
    assert f(2, 4096, 2, 0) == 4        # grid_x=1; still capped by max_splits=4
    assert f(512, 65536, 2, 0) == 5     # ceil(1216/256)=5 < max_splits=64
    # Never returns < 1 for any plausible shape.
    for slp in (0, 1, 64, 512, 4096, 200000):
        for slk in (1, 2048, 4096, 131072, 200000):
            assert f(slp, slk, 2, 0) >= 1


@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_flydsl_fp8_mqa_logits_adaptive_split_organic(clean_logits, monkeypatch) -> None:
    """End-to-end small-M / large-N shape that trips ``_auto_num_splits > 1``
    organically (heuristic NOT monkeypatched), so the real adaptive grid.y path
    runs and is graded against the reference."""
    import aiter.ops.flydsl.kernels.fp8_mqa_logits as kmod

    seen = {}
    real = kmod._auto_num_splits

    def spy(*a, **k):
        seen["n"] = real(*a, **k)
        return seen["n"]

    monkeypatch.setattr(kmod, "_auto_num_splits", spy)
    flydsl_fp8_mqa_logits = _flydsl_fp8_mqa_logits()

    s_q, s_k, num_heads, head_dim = 64, 8192, 64, 128
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    ref_logits, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    logits = flydsl_fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

    assert seen.get("n", 1) >= 2, f"expected adaptive split > 1, got {seen.get('n')}"

    if not clean_logits:
        full = torch.full((s_q, s_k), float("-inf"), device="cuda")
        for i in range(s_q):
            full[i, ks[i] : ke[i]] = logits[i, : ke[i] - ks[i]]
        logits = full

    ref_mask = ref_logits == float("-inf")
    assert torch.equal(logits == float("-inf"), ref_mask), "organic-split mask"
    diff = calc_diff(
        logits.masked_fill(ref_mask, 0), ref_logits.masked_fill(ref_mask, 0)
    )
    assert diff < 1e-3, f"organic-split FlyDSL vs torch-ref {diff=}"
