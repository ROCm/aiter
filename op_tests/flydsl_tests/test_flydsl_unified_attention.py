# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the FlyDSL fp8 unified-attention adapter, called directly.

Checks the adapter against `ref_paged_attn` (the same fp32 reference the Triton
unified-attention suite uses), so a failure here is the FlyDSL path's, not a
disagreement between two fp8 kernels.

Routing through the public `unified_attention` entry point is covered separately
in op_tests/test_unified_attention_flydsl.py.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402
from op_tests.triton_tests.attention.test_unified_attention import (  # noqa: E402
    ref_paged_attn,
)

try:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    _ARCH = get_gfx_runtime()
except Exception:  # noqa: BLE001
    _ARCH = None

pytestmark = [
    pytest.mark.skipif(_ARCH != "gfx950", reason=f"gfx950 only, got {_ARCH}"),
    pytest.mark.skipif(not is_flydsl_available(), reason="flydsl not available"),
]

FP8 = torch.float8_e4m3fn
PAGE = 64
HEAD_DIM = 128
DEV = "cuda"

# Project numeric gate for this kernel is `err < 1e-1, cos > 0.99, bad_rows == 0`
# against an fp32 reference, where bad_rows counts rows whose own worst error
# exceeds the threshold (an aggregate cosine stays high even when whole
# sequences are wrong, so the per-row count is what catches that).
#
# The absolute 1e-1 is calibrated for softmax_scale = 1/sqrt(d). fp8 rounding
# error scales with output magnitude, and a larger scale sharpens the softmax
# and grows the outputs, so the same kernel measures 0.06 abs error at 1x and
# 0.19 at 4x while its cosine *improves* (0.99983 -> 0.99993). Thresholding on
# error relative to the reference's own magnitude keeps one number meaningful
# across the scale sweep.
#
# Calibrated 2026-08-07 by sweeping every regime this file covers (GQA 1:1 and
# 16:1, 16k prefill, 64-way decode, mixed batch, ragged, and a 4x softmax
# scale): worst observed relative error 0.0478, on the 64-way decode; cosine
# never fell below 0.99965. 8e-2 leaves ~1.7x headroom over that worst case,
# which a 5e-2 bound did not -- it landed exactly on the boundary and failed
# intermittently. Anything near 8e-2 is a real regression, not noise.
MAX_REL_ERR = 8e-2
MIN_COS = 0.99


def _q8(x):
    """Quantize to fp8 with a per-tensor scale, returning (fp8, descale)."""
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(FP8), s.reshape(1).float().to(DEV)


def _build(query_lens, kv_lens, num_heads, num_kv_heads, seed=0):
    """Build a unified-attention call: packed Q, a scattered page pool, and a
    block table whose pages are deliberately non-contiguous."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    total_q = sum(query_lens)
    q_bf = torch.randn(total_q, num_heads, HEAD_DIM, generator=g, device=DEV,
                       dtype=torch.bfloat16)

    pages_per = [(ln + PAGE - 1) // PAGE for ln in kv_lens]
    stride = max(max(pages_per), 1)
    n_pages = max(sum(pages_per), 1)
    k_bf = torch.randn(n_pages, PAGE, num_kv_heads, HEAD_DIM, generator=g,
                       device=DEV, dtype=torch.bfloat16)
    # randn_like ignores `g` and draws from the global RNG, which would make
    # _build non-deterministic and silently invalidate any A/B that rebuilds.
    v_bf = torch.randn(n_pages, PAGE, num_kv_heads, HEAD_DIM, generator=g,
                       device=DEV, dtype=torch.bfloat16)

    # Scatter pages so the block table cannot be satisfied by contiguous reads.
    perm = torch.randperm(n_pages, generator=torch.Generator().manual_seed(seed + 1))
    bt = torch.zeros(len(kv_lens), stride, device=DEV, dtype=torch.int32)
    c = 0
    for i, np_ in enumerate(pages_per):
        for t in range(np_):
            bt[i, t] = int(perm[c])
            c += 1

    q, q_ds = _q8(q_bf)
    k, k_ds = _q8(k_bf)
    v, v_ds = _q8(v_bf)

    cu_q = torch.zeros(len(query_lens) + 1, device=DEV, dtype=torch.int32)
    cu_q[1:] = torch.tensor(query_lens, device=DEV, dtype=torch.int32).cumsum(0)
    seqused_k = torch.tensor(kv_lens, device=DEV, dtype=torch.int32)
    out = torch.empty(total_q, num_heads, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    return q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds


def _run(query_lens, kv_lens, num_heads, num_kv_heads, softmax_scale=None, seed=0):
    """Call the adapter and the reference; return (got, want) or None if declined."""
    from aiter.ops.flydsl.unified_attention_kernels import flydsl_unified_attention

    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, num_heads, num_kv_heads, seed
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

    got = flydsl_unified_attention(
        q, k, v, out, cu_q, max(query_lens), seqused_k, max(kv_lens),
        softmax_scale, True, (-1, -1), bt, 0, q_ds, k_ds, v_ds,
        num_kv_heads=num_kv_heads, block_size=PAGE,
        num_queries_per_kv=num_heads // num_kv_heads, num_seqs=len(query_lens),
    )
    if got is None:
        return None

    want = ref_paged_attn(
        query=q, key_cache=k, value_cache=v, query_lens=query_lens,
        kv_lens=kv_lens, block_tables=bt, scale=softmax_scale,
        out_dtype=torch.bfloat16, q_descale=q_ds, k_descale=k_ds, v_descale=v_ds,
    )
    return got.float(), want.float().reshape(got.shape)


def _check(got, want):
    assert torch.isfinite(got).all(), "output contains NaN or inf"
    scale = want.abs().max().item()
    diff = (got - want).abs()
    rel = diff.max().item() / scale
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(-1), want.reshape(-1), dim=0
    ).item()
    # Per-row worst error, matching the project's standing gates: an aggregate
    # cosine stays high even when whole sequences or trailing pages are wrong.
    bad_rows = int((diff.amax(dim=-1) > MAX_REL_ERR * scale).sum().item())
    assert bad_rows == 0, f"{bad_rows} rows over threshold (rel err {rel:.4g})"
    assert rel < MAX_REL_ERR, f"max rel err {rel:.4g}"
    assert cos > MIN_COS, f"cosine {cos:.6f}"


def test_build_is_deterministic():
    """Any A/B that rebuilds its inputs is meaningless unless _build is a pure
    function of its seed. Caught a real bug: torch.randn_like ignores the
    generator, so V was redrawn from the global RNG on every call and the two
    sides of a comparison saw different tensors."""
    a = _build([256], [256], 64, 4, seed=11)
    b = _build([256], [256], 64, 4, seed=11)
    names = ["q", "k", "v", "out", "cu_q", "seqused_k", "bt", "q_ds", "k_ds", "v_ds"]
    for name, x, y in zip(names, a, b):
        if name == "out":
            continue  # torch.empty; contents are not part of the input
        if x.dtype == FP8:
            x, y = x.view(torch.int8), y.view(torch.int8)
        assert torch.equal(x, y), f"_build is not deterministic in {name}"


# --- shape regimes -----------------------------------------------------------


@pytest.mark.parametrize("num_heads,num_kv_heads", [(64, 64), (64, 16), (64, 4), (64, 1)])
def test_gqa_ratios(num_heads, num_kv_heads):
    """1:1 (packing auto-disables) through 64:1 (the deepest packed group)."""
    r = _run([256] * 2, [256] * 2, num_heads, num_kv_heads)
    assert r is not None, "adapter declined a supported config"
    _check(*r)


@pytest.mark.parametrize("kv_len", [512, 4096, 16384])
def test_prefill_context_depths(kv_len):
    """Chunked prefill: a short new-token chunk against a deep cache."""
    r = _run([512], [kv_len], 64, 4)
    assert r is not None
    _check(*r)


@pytest.mark.parametrize("num_seqs", [1, 7, 64])
def test_decode_batches(num_seqs):
    """Pure decode: one query token per sequence."""
    r = _run([1] * num_seqs, [4096] * num_seqs, 64, 4)
    assert r is not None
    _check(*r)


def test_mixed_batch():
    """The production shape: one prefill chunk batched with decodes in a single
    launch. This is what unified attention exists for, and the case the
    M-dimension packing was built to serve."""
    r = _run([4023] + [1] * 8, [4023] + [16384] * 8, 64, 4)
    assert r is not None
    _check(*r)


# --- softmax_scale injection -------------------------------------------------


def test_softmax_scale_default():
    """1/sqrt(d): the ratio is exactly 1 and q_descale passes through untouched."""
    r = _run([256], [256], 64, 4, softmax_scale=1.0 / math.sqrt(HEAD_DIM))
    assert r is not None
    _check(*r)


@pytest.mark.parametrize("mult", [0.5, 2.0])
def test_softmax_scale_non_default(mult):
    """A scale the kernel cannot bake in, folded into q_descale instead. Far
    enough from 1.0 that a mis-applied factor cannot pass."""
    r = _run([256], [256], 64, 4, softmax_scale=mult / math.sqrt(HEAD_DIM))
    assert r is not None
    _check(*r)


def test_softmax_scale_actually_changes_output():
    """Guard against the injection being silently dropped: two different scales
    must not produce the same result."""
    a = _run([256], [256], 64, 4, softmax_scale=1.0 / math.sqrt(HEAD_DIM))
    b = _run([256], [256], 64, 4, softmax_scale=4.0 / math.sqrt(HEAD_DIM))
    assert a is not None and b is not None
    assert not torch.allclose(a[0], b[0], atol=1e-3), \
        "softmax_scale had no effect -- the q_descale injection is not reaching the kernel"


# --- edge cases --------------------------------------------------------------


def test_ragged_lengths():
    """Uneven query lengths: grid.y is sized from the max, so short sequences
    depend on the per-sequence active guard to skip their surplus blocks."""
    r = _run([1, 37, 256, 900], [1024] * 4, 64, 4)
    assert r is not None
    _check(*r)


def test_one_sequence_dominates():
    """A 100x length skew, maximising the number of inactive workgroups."""
    r = _run([4096] + [1] * 4, [4096] + [512] * 4, 64, 4)
    assert r is not None
    _check(*r)


def test_non_multiple_of_page():
    """KV lengths that do not fill their last page.

    Every kv_len stays >= its query_len: under bottom-right causal alignment a
    shorter KV would leave the leading query rows with no unmasked key at all,
    and the reference softmaxes an all -inf row to NaN. That is an invalid call,
    not a kernel limit.
    """
    r = _run([64, 64, 100], [130, 65, 191], 64, 4)
    assert r is not None
    _check(*r)


def test_out_not_overwritten_past_active_rows():
    """The last Q block of a sequence overhangs its real rows; only the buffer
    descriptor's num_records bound stops it writing past them. Poison `out` and
    confirm every row that should be written was, and nothing beyond."""
    from aiter.ops.flydsl.unified_attention_kernels import flydsl_unified_attention

    query_lens, kv_lens = [37, 5], [512, 512]
    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, 64, 4, seed=3
    )
    pad = 64
    padded = torch.full(
        (out.shape[0] + pad, out.shape[1], out.shape[2]),
        float("nan"), device=DEV, dtype=torch.bfloat16,
    )
    view = padded[: out.shape[0]]

    got = flydsl_unified_attention(
        q, k, v, view, cu_q, max(query_lens), seqused_k, max(kv_lens),
        1.0 / math.sqrt(HEAD_DIM), True, (-1, -1), bt, 0, q_ds, k_ds, v_ds,
        num_kv_heads=4, block_size=PAGE, num_queries_per_kv=16,
        num_seqs=len(query_lens),
    )
    assert got is not None
    assert torch.isfinite(view).all(), "an active output row was left unwritten"
    assert torch.isnan(padded[out.shape[0]:]).all(), \
        "the kernel wrote past the end of the output tensor"


# --- the invariant the softmax_scale injection rests on ----------------------


def test_q_descale_has_no_other_consumer():
    """`_scaled_q_descale` folds softmax_scale into q_descale, which is valid
    only while q_descale reaches c_logit_scale and nothing else. Two occurrences
    are expected: the load and the multiply. A third means someone gave the Q
    descale a second consumer, and the injection is now silently wrong."""
    from pathlib import Path

    import aiter.ops.flydsl.kernels.flash_attn_dualwave_common as common

    src = Path(common.__file__).read_text()
    n = sum(1 for line in src.splitlines() if "_qd" in line and "_qd_" not in line)
    assert n == 2, (
        f"expected 2 references to the loaded q descale, found {n}. "
        "The softmax_scale injection in unified_attention_kernels.py assumes "
        "q_descale feeds only c_logit_scale -- re-verify before trusting it."
    )
