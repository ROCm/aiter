# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the FlyDSL fp8 unified-attention adapter, called directly.

Checks the adapter against `ref_paged_attn` (the same fp32 reference the Triton
unified-attention suite uses), so a failure here is the FlyDSL path's, not a
disagreement between two fp8 kernels.

Routing through the public `unified_attention` entry point is covered separately
in op_tests/flydsl_tests/test_unified_attention_routing.py.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available
from op_tests.flydsl_tests._common import assert_attn_close
from op_tests.flydsl_tests._common import q8 as _q8
from op_tests.triton_tests.attention.test_unified_attention import (
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

# Vectorization width of the shuffled 5D KV cache (see _KV_VEC_SIZE in
# unified_attention_kernels.py): 16 fp8 elements = one 128-bit dwordx4.
_KV_VEC = 16

# Project numeric gate is `err < 1e-1, cos > 0.99, bad_rows == 0` against an
# fp32 reference; bad_rows catches whole-sequence failures an aggregate
# cosine would hide.
#
# Error is thresholded relative to the reference's own magnitude rather than
# absolute, because fp8 rounding error scales with output magnitude: a larger
# softmax_scale sharpens the softmax and grows outputs, so the same kernel
# measures 0.06 abs error at 1x scale and 0.19 at 4x while cosine *improves*.
# 8e-2 gives ~1.7x headroom over the worst observed relative error (0.0478,
# 64-way decode) across GQA ratios, prefill/decode/mixed/ragged shapes, and a
# 4x softmax scale; a 5e-2 bound sat on that boundary and failed intermittently.
MAX_REL_ERR = 8e-2
MIN_COS = 0.99


def _build(
    query_lens, kv_lens, num_heads, num_kv_heads, seed=0, out_dtype=torch.bfloat16
):
    """Build a unified-attention call: packed Q, a scattered page pool, and a
    block table whose pages are deliberately non-contiguous."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    total_q = sum(query_lens)
    q_bf = torch.randn(
        total_q, num_heads, HEAD_DIM, generator=g, device=DEV, dtype=torch.bfloat16
    )

    pages_per = [(ln + PAGE - 1) // PAGE for ln in kv_lens]
    stride = max(max(pages_per), 1)
    n_pages = max(sum(pages_per), 1)
    k_bf = torch.randn(
        n_pages,
        PAGE,
        num_kv_heads,
        HEAD_DIM,
        generator=g,
        device=DEV,
        dtype=torch.bfloat16,
    )
    # randn_like ignores `g` and draws from the global RNG, which would make
    # _build non-deterministic and silently invalidate any A/B that rebuilds.
    v_bf = torch.randn(
        n_pages,
        PAGE,
        num_kv_heads,
        HEAD_DIM,
        generator=g,
        device=DEV,
        dtype=torch.bfloat16,
    )

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
    out = torch.empty(total_q, num_heads, HEAD_DIM, device=DEV, dtype=out_dtype)
    return q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds


def _shuffle_k_to_vectorized(kpool, vec=_KV_VEC):
    """[nb, PAGE, kv_heads, D] fp8 -> [nb, kv_heads, D//vec, PAGE, vec]. This is
    the exact permutation the decode kernel's vectorized K loader assumes."""
    nb, page, hkv, d = kpool.shape
    x = kpool.permute(0, 2, 3, 1).contiguous()  # [nb,hkv,d,page]
    x = x.view(nb, hkv, d // vec, vec, page)
    return x.permute(0, 1, 2, 4, 3).contiguous()  # [nb,hkv,d//vec,page,vec]


def _shuffle_v_to_vectorized(vpool, vec=_KV_VEC):
    """[nb, PAGE, kv_heads, D] fp8 -> [nb, kv_heads, PAGE//vec, D, vec]. This is
    the exact permutation the decode kernel's vectorized V loader assumes."""
    nb, page, hkv, d = vpool.shape
    x = vpool.permute(0, 2, 3, 1).contiguous()  # [nb,hkv,d,page]
    x = x.view(nb, hkv, d, page // vec, vec)
    return x.permute(0, 1, 3, 2, 4).contiguous()  # [nb,hkv,page//vec,d,vec]


def _build_padded(
    query_lens,
    kv_lens,
    num_heads,
    num_kv_heads,
    seed=0,
    out_dtype=torch.bfloat16,
    pool_blocks=None,
    shuffled=False,
):
    """Like `_build`, but (a) the page pool can be padded far past the pages
    actually used (`pool_blocks`, e.g. 65536 -- the production 2 GB pool whose
    high page ids exercise the per-page BufferDesc rebasing rather than a
    flat whole-pool offset) and (b) the pool can be handed to the kernel in
    the shuffled 5D layout (`shuffled=True`).

    Returns the SAME linear (unshuffled) 4D k/v alongside whatever the kernel
    actually consumes, so the caller can still validate against
    `ref_paged_attn`, which only understands the linear 4D layout.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    total_q = sum(query_lens)
    q_bf = torch.randn(
        total_q, num_heads, HEAD_DIM, generator=g, device=DEV, dtype=torch.bfloat16
    )

    pages_per = [(ln + PAGE - 1) // PAGE for ln in kv_lens]
    stride = max(max(pages_per), 1)
    n_pages = max(sum(pages_per), 1)
    nblocks = n_pages if pool_blocks is None else pool_blocks

    # Only the pages actually referenced by the block table carry real data;
    # the rest of a padded pool stays zero (ref_paged_attn never reads them,
    # and the kernel only rebases into pages the block table names).
    k_used = torch.randn(
        n_pages,
        PAGE,
        num_kv_heads,
        HEAD_DIM,
        generator=g,
        device=DEV,
        dtype=torch.bfloat16,
    )
    v_used = torch.randn(
        n_pages,
        PAGE,
        num_kv_heads,
        HEAD_DIM,
        generator=g,
        device=DEV,
        dtype=torch.bfloat16,
    )
    perm = torch.randperm(nblocks, generator=torch.Generator().manual_seed(seed + 1))
    used_ids = perm[:n_pages]
    k_bf = torch.zeros(
        nblocks, PAGE, num_kv_heads, HEAD_DIM, device=DEV, dtype=torch.bfloat16
    )
    v_bf = torch.zeros_like(k_bf)
    k_bf[used_ids] = k_used
    v_bf[used_ids] = v_used

    bt = torch.zeros(len(kv_lens), stride, device=DEV, dtype=torch.int32)
    c = 0
    for i, np_ in enumerate(pages_per):
        for t in range(np_):
            bt[i, t] = int(used_ids[c])
            c += 1

    q, q_ds = _q8(q_bf)
    k, k_ds = _q8(k_bf)
    v, v_ds = _q8(v_bf)

    cu_q = torch.zeros(len(query_lens) + 1, device=DEV, dtype=torch.int32)
    cu_q[1:] = torch.tensor(query_lens, device=DEV, dtype=torch.int32).cumsum(0)
    seqused_k = torch.tensor(kv_lens, device=DEV, dtype=torch.int32)
    out = torch.empty(total_q, num_heads, HEAD_DIM, device=DEV, dtype=out_dtype)

    if shuffled:
        k_arg = _shuffle_k_to_vectorized(k)
        v_arg = _shuffle_v_to_vectorized(v)
    else:
        k_arg, v_arg = k, v
    return q, k_arg, v_arg, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds, k, v


def _run(
    query_lens,
    kv_lens,
    num_heads,
    num_kv_heads,
    softmax_scale=None,
    seed=0,
    out_dtype=torch.bfloat16,
    shuffled_kv_cache=False,
    pool_blocks=None,
):
    """Call the adapter and the reference; return (got, want) or None if declined.

    ``shuffled_kv_cache``/``pool_blocks`` route through `_build_padded` instead
    of `_build` (production 5D-shuffled cache and/or a page pool padded past
    what's actually used); the reference is always computed from the
    equivalent LINEAR 4D pool, since `ref_paged_attn` only understands that
    layout.
    """
    from aiter.ops.flydsl.unified_attention_kernels import flydsl_unified_attention

    if shuffled_kv_cache or pool_blocks is not None:
        q, k_arg, v_arg, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds, k_lin, v_lin = (
            _build_padded(
                query_lens,
                kv_lens,
                num_heads,
                num_kv_heads,
                seed,
                out_dtype,
                pool_blocks,
                shuffled_kv_cache,
            )
        )
    else:
        q, k_arg, v_arg, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
            query_lens, kv_lens, num_heads, num_kv_heads, seed, out_dtype
        )
        k_lin, v_lin = k_arg, v_arg
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

    got = flydsl_unified_attention(
        q,
        k_arg,
        v_arg,
        out,
        cu_q,
        max(query_lens),
        seqused_k,
        max(kv_lens),
        softmax_scale,
        True,
        (-1, -1),
        bt,
        0,
        q_ds,
        k_ds,
        v_ds,
        num_kv_heads=num_kv_heads,
        block_size=PAGE,
        num_queries_per_kv=num_heads // num_kv_heads,
        num_seqs=len(query_lens),
        shuffled_kv_cache=shuffled_kv_cache,
    )
    if got is None:
        return None

    want = ref_paged_attn(
        query=q,
        key_cache=k_lin,
        value_cache=v_lin,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=bt,
        scale=softmax_scale,
        out_dtype=out_dtype,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
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
    assert_attn_close(rel, cos, bad_rows, max_err=MAX_REL_ERR, min_cos=MIN_COS)


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


@pytest.mark.parametrize("num_heads,num_kv_heads", [(64, 4), (64, 64)])
def test_gqa_ratios(num_heads, num_kv_heads):
    """16:1 (the decode-specialized kernel's own ratio) and 1:1/MHA (packing
    auto-disables)."""
    r = _run([256] * 2, [256] * 2, num_heads, num_kv_heads)
    assert r is not None, "adapter declined a supported config"
    _check(*r)


# --- prefill / decode / mixed correctness, all modes -------------------------


_CORRECTNESS_CASES = [
    pytest.param(
        {
            "query_lens": [512],
            "kv_lens": [4096],
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.bfloat16,
        },
        id="prefill-512to4096-bf16",
    ),
    pytest.param(
        {
            "query_lens": [512],
            "kv_lens": [4096],
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.float16,
        },
        id="prefill-512to4096-f16",
    ),
    pytest.param(
        {
            "query_lens": [1] * 8,
            "kv_lens": [4096] * 8,
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.bfloat16,
        },
        id="decode-1to4096x8-gqa16-bf16",
    ),
    pytest.param(
        {
            "query_lens": [1] * 8,
            "kv_lens": [4096] * 8,
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.float16,
        },
        id="decode-1to4096x8-gqa16-f16",
    ),
    pytest.param(
        {
            "query_lens": [1] * 8,
            "kv_lens": [4096] * 8,
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.bfloat16,
            "shuffled_kv_cache": True,
        },
        id="decode-1to4096x8-gqa16-bf16-shuffled",
    ),
    pytest.param(
        # Production pool: 65536 blocks = 2 GB fp8 = exactly 2**31 elements --
        # the per-page-rebasing stress case (page ids up to 65535 => 2**31-byte
        # page-base offsets) that whole-pool addressing faulted on. Shuffled
        # (production layout).
        {
            "query_lens": [1] * 8,
            "kv_lens": [4096] * 8,
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.bfloat16,
            "shuffled_kv_cache": True,
            "pool_blocks": 65536,
        },
        id="decode-production-pool-65536-bf16",
    ),
    pytest.param(
        # Ragged batch=8, deep+shallow mixed depths: with num_kv_heads=4 the
        # host plan (plan_num_kv_splits) picks S=8, exercising the split-K
        # cross-split LSE combine on unequal per-sequence depths.
        {
            "query_lens": [1] * 8,
            "kv_lens": [4096, 8192, 16384, 1024, 6000, 300, 12000, 2048],
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.bfloat16,
        },
        id="decode-splitk-ragged-b8",
    ),
    pytest.param(
        {
            "query_lens": [4023] + [1] * 8,
            "kv_lens": [4023] + [16384] * 8,
            "num_heads": 64,
            "num_kv_heads": 4,
            "out_dtype": torch.bfloat16,
        },
        id="mixed-4023-plus-1x8",
    ),
]


@pytest.mark.parametrize("kwargs", _CORRECTNESS_CASES)
def test_correctness_all_modes(kwargs):
    """Prefill/decode/mixed correctness against `ref_paged_attn`, across output
    dtype, the production shuffled 5D KV-cache layout, a padded production-scale
    page pool, and a split-K-triggering ragged decode batch. Subsumes the former
    per-axis sweeps (prefill context depth, decode batch size, mixed batch) plus
    three regression guards (production pool, split-K, shuffled layout)."""
    r = _run(**kwargs)
    assert r is not None, "adapter declined a supported config"
    _check(*r)


# --- softmax_scale injection -------------------------------------------------


@pytest.mark.parametrize(
    "mult",
    [
        pytest.param(1.0, id="default-ratio-1"),
        pytest.param(0.5, id="half"),
        pytest.param(2.0, id="double"),
    ],
)
def test_softmax_scale(mult):
    """A softmax_scale the kernel folds into q_descale (mult=1.0 is the
    near-universal 1/sqrt(d) case, where the ratio is exactly 1 and q_descale
    passes through untouched; 0.5/2.0 are scales the kernel cannot bake in,
    far enough from 1.0 that a mis-applied factor cannot pass)."""
    r = _run([256], [256], 64, 4, softmax_scale=mult / math.sqrt(HEAD_DIM))
    assert r is not None
    _check(*r)


def test_softmax_scale_actually_changes_output():
    """Guard against the injection being silently dropped: two different scales
    must not produce the same result."""
    a = _run([256], [256], 64, 4, softmax_scale=1.0 / math.sqrt(HEAD_DIM))
    b = _run([256], [256], 64, 4, softmax_scale=4.0 / math.sqrt(HEAD_DIM))
    assert a is not None and b is not None
    assert not torch.allclose(
        a[0], b[0], atol=1e-3
    ), "softmax_scale had no effect -- the q_descale injection is not reaching the kernel"


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


def _run_decode_with_empty_seq(query_lens, kv_lens, seed=9):
    """All-decode call whose last sequence has `seqused_k == 0` (empty KV),
    mixed into an otherwise-normal batch. Returns (got, want); `want` is
    `ref_paged_attn`'s own zero result for the empty row (a zero-length KV
    softmax sums zero terms, no special-casing needed -- see the einsum in
    `ref_paged_attn`)."""
    from aiter.ops.flydsl.unified_attention_kernels import flydsl_unified_attention

    num_heads, num_kv_heads = 64, 4
    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, num_heads, num_kv_heads, seed
    )
    got = flydsl_unified_attention(
        q,
        k,
        v,
        out,
        cu_q,
        max(query_lens),
        seqused_k,
        max(kv_lens),
        1.0 / math.sqrt(HEAD_DIM),
        True,
        (-1, -1),
        bt,
        0,
        q_ds,
        k_ds,
        v_ds,
        num_kv_heads=num_kv_heads,
        block_size=PAGE,
        num_queries_per_kv=num_heads // num_kv_heads,
        num_seqs=len(query_lens),
    )
    assert got is not None, "adapter declined a supported all-decode config"
    want = ref_paged_attn(
        query=q,
        key_cache=k,
        value_cache=v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=bt,
        scale=1.0 / math.sqrt(HEAD_DIM),
        out_dtype=out.dtype,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
    )
    return got.float(), want.float().reshape(got.shape)


def test_empty_kv_direct_store():
    """seqused_k==0 for the last of 8 decode sequences, shallow enough
    (max_seqlen_k=64 -> 1 page -> plan_num_kv_splits==1) to hit the decode
    kernel's direct-store epilogue rather than the split-K combine.

    regression: empty-KV (den==0) must yield a defined zero result, not NaN,
    in both the direct-store epilogue and the split-K combine (PR #4676
    review P2).
    """
    got, want = _run_decode_with_empty_seq([1] * 8, [64] * 7 + [0])
    assert torch.isfinite(got).all(), "empty-KV row produced NaN/inf"
    assert torch.all(got[-1] == 0), "empty-KV row must be exactly zero, not garbage"
    _check(got, want)


def test_empty_kv_splitk_combine():
    """Same seqused_k==0 mix, but deep context (max_seqlen_k=8192 -> 128
    pages) drives `plan_num_kv_splits` to S=8, exercising the split-K
    combine kernel's den==0 guard instead of the direct-store one.

    regression: empty-KV (den==0) must yield a defined zero result, not NaN,
    in both the direct-store epilogue and the split-K combine (PR #4676
    review P2).
    """
    got, want = _run_decode_with_empty_seq([1] * 8, [8192] * 7 + [0])
    assert torch.isfinite(got).all(), "empty-KV row produced NaN/inf"
    assert torch.all(got[-1] == 0), "empty-KV row must be exactly zero, not garbage"
    _check(got, want)


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
        float("nan"),
        device=DEV,
        dtype=torch.bfloat16,
    )
    view = padded[: out.shape[0]]

    got = flydsl_unified_attention(
        q,
        k,
        v,
        view,
        cu_q,
        max(query_lens),
        seqused_k,
        max(kv_lens),
        1.0 / math.sqrt(HEAD_DIM),
        True,
        (-1, -1),
        bt,
        0,
        q_ds,
        k_ds,
        v_ds,
        num_kv_heads=4,
        block_size=PAGE,
        num_queries_per_kv=16,
        num_seqs=len(query_lens),
    )
    assert got is not None
    assert torch.isfinite(view).all(), "an active output row was left unwritten"
    assert torch.isnan(
        padded[out.shape[0] :]
    ).all(), "the kernel wrote past the end of the output tensor"
