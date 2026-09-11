# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of compact and ragged next_n paged MQA logits."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx950 import (
    flydsl_fp8_paged_mqa_logits,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.flydsl_tests.ragged_nn import (
    MAX_NN,
    live_row_mask,
    padded_logits,
    ref_padded_ragged,
    sample_next_n_lens,
)
from op_tests.flydsl_tests.test_flydsl_fp8_paged_mqa_logits import (
    calc_diff,
    kv_cache_cast_to_fp8,
    preshuffle_kv_data,
)

torch.set_default_device("cuda")

HEADS = 32
HEAD_DIM = 128
KVB = 64


def _batch(nq, ctx, max_nn=MAX_NN, seed=0):
    torch.manual_seed(seed)
    fp8 = get_fp8_e4m3_dtype()
    pages = ctx // KVB
    q = torch.randn((nq, max_nn, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    weights = torch.randn((nq * max_nn, HEADS), dtype=torch.float32)
    context_lens = torch.full((nq,), ctx, dtype=torch.int32)
    kv = torch.randn((max(nq * pages, 1), KVB, 1, HEAD_DIM), dtype=torch.bfloat16)
    kv_fp8 = kv_cache_cast_to_fp8(kv, fp8)
    ids = torch.arange(nq * pages, device="cuda", dtype=torch.int32)
    tables = torch.zeros((nq, pages), dtype=torch.int32)
    tables[:, :pages] = ids.reshape(nq, pages)
    return q, weights, context_lens, tables, kv_fp8, fp8


def _run_and_check(
    q,
    weights,
    context_lens,
    tables,
    kv_fp8,
    fp8,
    next_n_lens,
    ctx,
    *,
    split_kv=None,
):
    max_nn = q.shape[1]
    ref = ref_padded_ragged(
        q,
        kv_fp8,
        weights,
        context_lens,
        tables,
        next_n_lens,
        ctx,
        fp8,
        max_nn=max_nn,
        block_size=KVB,
    )
    out = padded_logits(q.shape[0], max_nn, ctx)
    got = flydsl_fp8_paged_mqa_logits(
        q.to(fp8),
        preshuffle_kv_data(kv_fp8, HEAD_DIM),
        weights,
        out,
        context_lens,
        tables,
        ctx,
        next_n_lens=next_n_lens,
        SplitKV=split_kv,
    )
    ref_inf = ref == float("-inf")
    got_inf = got == float("-inf")
    assert torch.equal(got_inf, ref_inf), "ragged -inf mask mismatch"
    diff = calc_diff(got.masked_fill(got_inf, 0), ref.masked_fill(ref_inf, 0))
    assert diff < 1e-3, diff
    return got, ref


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
@pytest.mark.parametrize("nn", [1, 2, 4, 8])
def test_compact_uniform_next_n(nn):
    nq, ctx, max_nn = 4, 256, nn
    next_n_lens = torch.full((nq,), nn, dtype=torch.int32, device="cuda")
    q, weights, context_lens, tables, kv_fp8, fp8 = _batch(nq, ctx, max_nn)
    ref = ref_padded_ragged(
        q,
        kv_fp8,
        weights,
        context_lens,
        tables,
        next_n_lens,
        ctx,
        fp8,
        max_nn=max_nn,
        block_size=KVB,
    )
    out = padded_logits(nq, max_nn, ctx)
    got = flydsl_fp8_paged_mqa_logits(
        q.to(fp8),
        preshuffle_kv_data(kv_fp8, HEAD_DIM),
        weights,
        out,
        context_lens,
        tables,
        ctx,
    )
    ref_inf = ref == float("-inf")
    got_inf = got == float("-inf")
    assert torch.equal(got_inf, ref_inf)
    diff = calc_diff(got.masked_fill(got_inf, 0), ref.masked_fill(ref_inf, 0))
    assert diff < 1e-3, diff


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
@pytest.mark.parametrize("max_nn", [4, MAX_NN])
def test_ragged_mix_unused_stay_neg_inf(max_nn):
    nq, ctx = 8, 192
    next_n_lens = sample_next_n_lens(nq, max_nn, seed=1079).cuda()
    assert int(next_n_lens.min()) >= 1 and int(next_n_lens.max()) <= max_nn
    assert len(next_n_lens.unique()) >= 2
    q, weights, context_lens, tables, kv_fp8, fp8 = _batch(nq, ctx, max_nn, seed=1079)
    got, ref = _run_and_check(
        q,
        weights,
        context_lens,
        tables,
        kv_fp8,
        fp8,
        next_n_lens,
        ctx,
    )
    mask = live_row_mask(next_n_lens, max_nn, ctx)
    unused_rows = ~mask[:, 0]
    assert unused_rows.any(), "fixture should include padded rows"
    assert torch.all(ref[unused_rows] == float("-inf"))
    assert torch.all(got[unused_rows] == float("-inf"))

    q2 = torch.randn_like(q)
    for b, n in enumerate(next_n_lens.tolist()):
        q2[b, :n].copy_(q[b, :n])
    ref2 = ref_padded_ragged(
        q2,
        kv_fp8,
        weights,
        context_lens,
        tables,
        next_n_lens,
        ctx,
        fp8,
        max_nn=max_nn,
        block_size=KVB,
    )
    assert torch.equal(ref == float("-inf"), ref2 == float("-inf"))
    inf = ref == float("-inf")
    diff = calc_diff(ref.masked_fill(inf, 0), ref2.masked_fill(inf, 0))
    assert diff < 1e-3, diff


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
@pytest.mark.parametrize("nq", [1, 24])
def test_nq_and_short_pages(nq):
    ctx, max_nn = 128, MAX_NN
    next_n_lens = sample_next_n_lens(nq, max_nn, seed=3).cuda()
    q, weights, context_lens, tables, kv_fp8, fp8 = _batch(nq, ctx, max_nn, seed=3)
    got, ref = _run_and_check(
        q,
        weights,
        context_lens,
        tables,
        kv_fp8,
        fp8,
        next_n_lens,
        ctx,
    )
    assert ref.shape == (nq * max_nn, ctx)
    assert got.shape == ref.shape
    empty = padded_logits(nq, max_nn, ctx)
    mask = live_row_mask(next_n_lens, max_nn, ctx)
    assert torch.equal(ref[~mask[:, 0]], empty[~mask[:, 0]])


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
def test_odd_page_split_starts():
    nq, ctx = 3, 7 * KVB
    next_n_lens = sample_next_n_lens(nq, MAX_NN, seed=19).cuda()
    q, weights, context_lens, tables, kv_fp8, fp8 = _batch(nq, ctx, seed=19)
    _run_and_check(
        q,
        weights,
        context_lens,
        tables,
        kv_fp8,
        fp8,
        next_n_lens,
        ctx,
        split_kv=3,
    )
