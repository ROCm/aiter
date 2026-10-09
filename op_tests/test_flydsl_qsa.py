# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA oracle, family A plumbing, live vLLM AMD, #4882 Triton/Gluon, FlyDSL K1/K2.

Two layers:
  * Correctness (pytest gate): ``test_*`` unit cases.
  * Perf sweep (``__main__``): ``bench_qsa_family_a_plumbing``,
    ``bench_qsa_family_a_vllm_amd``, ``bench_qsa_family_a_4882_triton``,
    ``bench_qsa_family_b_4882_triton``, ``bench_qsa_family_b_4882_gluon``,
    ``bench_qsa_family_a_k1`` (decode ``M<=8`` and a separate prefill table;
    HIP-graph select columns for FlyDSL and live AMD when the table is
    wider than 20000 columns and ``M<=64``),
    ``bench_qsa_family_b_k1`` (emit; long-``L`` uses family A scorer; published point),
    ``bench_qsa_family_a_k2`` (3d decode ``M<=8`` and a separate prefill table),
    ``bench_qsa_family_a_e2e`` / ``bench_qsa_family_b_e2e`` (indexer through GQA),
    ``bench_qsa_family_a_e2e_graph`` (HIP graph replay at decode).

Usage::

    pytest -q op_tests/test_flydsl_qsa.py
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py --rotate 0 1
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py --pad-pages 4096
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py --cache-layout wide

Perf rows default to cold weights (``--rotate 0``); pass ``--rotate 1`` to
reproduce the older hot-cache 1047 tables. ``--pad-pages 0`` (the default)
keeps each K1 block table packed to the context. A positive width zero-fills
it out to that many pages, which is how vLLM sizes the table for
``max_model_len``. ``--cache-layout packed`` (the default) keeps each cache
contiguous. ``wide`` is vLLM's per-layer view: the same pages, with a page
stride that puts the byte span just past 4 GiB, which is the compile K1
and K2 use for a serving KV pool.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.qsa import k1 as k1_kernel
from aiter.ops.flydsl.kernels.qsa.k2 import qsa_k2_serves
from aiter.ops.flydsl.qsa import (
    gather_paged_cache,
    gather_qsa_caches,
    normalize_qsa_backend,
    pack_paged_cache,
    qsa_auto_uses_flydsl,
    qsa_expand_tail,
    qsa_indexer_scores,
    qsa_k1_block_ids,
    qsa_k2,
    qsa_layer,
    qsa_oracle,
    qsa_sparse_gqa,
    qsa_topk_blocks,
    qsa_visible_blocks,
)
from aiter.ops.triton.attention.qsa_4882 import (
    AITER_4882_QSA_PIN,
    gluon_qsa_available,
    qsa_sparse_paged_gqa,
)
from aiter.ops.triton.attention.qsa_4882 import (
    qsa_select_paged_tokens as qsa_4882_select_paged_tokens,
)
from aiter.ops.triton.attention.qsa_vllm_amd import (
    VLLM_AMD_QSA_PIN,
    expand_qsa_block_indices_cuda,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_tests.qsa_shapes import (
    FAMILY_A_GQA,
    FAMILY_A_INDEXER,
    FAMILY_A_SCORE_SCALE,
    FAMILY_B_GQA,
    FAMILY_B_INDEXER,
    FAMILY_B_INDEXER_H8,
)

SUPPORTED_GFX = ["gfx942", "gfx950"]
# Exactly 4 GiB still fits in a V# offset. The wide compile starts past that.
_FOUR_GIB = 1 << 32


def _span_bytes(t: torch.Tensor) -> int:
    """Bytes from ``t``'s first element to one past its last, for any strides."""
    if t.numel() == 0:
        return 0
    last = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return (last + 1) * t.element_size()


def _storage_tensor(view: torch.Tensor) -> torch.Tensor:
    """Contiguous 1-D tensor over ``view``'s storage.

    Cold rotation deep-copies this, which keeps the gaps. Copying the view
    itself packs the pages and the span falls back under 4 GiB.
    """
    elem = view.element_size()
    n_elems = view.untyped_storage().size() // elem - view.storage_offset()
    flat = torch.empty(0, dtype=view.dtype, device=view.device)
    flat.set_(view.untyped_storage(), view.storage_offset(), (n_elems,), (1,))
    return flat


def _widen_cache(cache: torch.Tensor) -> torch.Tensor:
    """The same pages, with a page stride that puts the span just past 4 GiB.

    vLLM allocates one KV pool and hands each layer a strided view, so the
    page stride covers every layer and the span is the pool. Allocating that
    pool is unnecessary: a few pages and a stride that lands the last page
    just past 4 GiB selects the same compile. A one-page cache gains an
    unreferenced trailing page, because a size of 1 ignores its stride.
    The parent holds the span and no more. Page stride is a multiple of 8
    elements, which is the gather alignment K1 and K2 already require.
    """
    if cache.ndim != 4:
        raise ValueError(
            f"paged cache must be [pages, page_size, H, D], got {tuple(cache.shape)}"
        )
    n_pages, page_size, n_heads, head_dim = (int(s) for s in cache.shape)
    if n_pages < 1:
        return cache
    page_elems = page_size * n_heads * head_dim
    token_stride = n_heads * head_dim
    head_stride = head_dim
    view_pages = max(n_pages, 2)
    elem = cache.element_size()
    gaps = view_pages - 1
    # span = (gaps * page_stride + page_elems) * elem, and it must exceed 4 GiB.
    min_elems = (_FOUR_GIB + elem) // elem
    page_stride = (min_elems - page_elems + gaps - 1) // gaps
    page_stride = max(page_stride, page_elems)
    if page_stride % 8:
        page_stride += 8 - (page_stride % 8)
    span_elems = gaps * page_stride + page_elems
    parent = cache.new_empty(span_elems)
    parent.zero_()
    flat = cache.contiguous().reshape(n_pages, page_elems)
    for i in range(n_pages):
        parent[i * page_stride : i * page_stride + page_elems] = flat[i]
    view = parent.as_strided(
        (view_pages, page_size, n_heads, head_dim),
        (page_stride, token_stride, head_stride, 1),
    )
    if _span_bytes(view) <= _FOUR_GIB:
        raise RuntimeError(f"wide cache span {_span_bytes(view)} did not pass 4 GiB")
    return view


def _apply_cache_layout(cache: torch.Tensor, layout: str) -> torch.Tensor:
    """``packed`` leaves the cache. ``wide`` is the serving span."""
    if layout == "packed":
        return cache
    if layout != "wide":
        raise ValueError(f"cache_layout must be packed or wide, got {layout}")
    return _widen_cache(cache)


def _span_gib(cache: torch.Tensor) -> float:
    return _span_bytes(cache) / (1 << 30)


def test_wide_cache_layout_spans_past_4gib():
    """The bench view crosses 4 GiB without packing, and a clone does not.

    A one-page cache has to grow a trailing page or the stride is ignored.
    Rotating copies the storage and rebuilds the view; cloning the view
    packs it back under 4 GiB, which would time the narrow compile.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    device = torch.device("cuda")
    free, _total = torch.cuda.mem_get_info(device)
    if free < _FOUR_GIB + (1 << 30):
        aiter.logger.warning(
            "skip wide cache layout test: need %s bytes, %s free",
            _FOUR_GIB + (1 << 30),
            free,
        )
        return
    try:
        _apply_cache_layout(torch.empty(1, 1, 1, 8), "dense")
    except ValueError as exc:
        if "cache_layout" not in str(exc):
            raise
    else:
        raise AssertionError("an unknown cache layout was accepted")
    idx = FAMILY_A_INDEXER
    page_size = 16
    for n_pages in (1, 3):
        n_blocks = n_pages * page_size
        cache, _table = pack_paged_cache(
            torch.randn(n_blocks, 1, idx.head_dim, dtype=dtypes.bf16, device=device),
            page_size,
        )
        packed = _apply_cache_layout(cache, "packed")
        if _span_bytes(packed) > _FOUR_GIB:
            raise AssertionError("packed layout crossed 4 GiB")
        view = _apply_cache_layout(cache, "wide")
        if _span_bytes(view) <= _FOUR_GIB:
            raise AssertionError(f"wide span {_span_bytes(view)} did not pass 4 GiB")
        if n_pages == 1 and view.shape[0] != 2:
            raise AssertionError(f"one-page cache stayed at {view.shape[0]} pages")
        if n_pages > 1 and view.shape[0] != n_pages:
            raise AssertionError(f"wide page count {view.shape[0]} != {n_pages}")
        if any(s % 8 for s in view.stride()[:3]) or view.stride(3) != 1:
            raise AssertionError(
                f"wide strides are not gather-aligned: {view.stride()}"
            )
        if not torch.equal(view[:n_pages], cache):
            raise AssertionError("wide view changed the packed pages")
        q = torch.empty(1, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
        table = torch.zeros(1, 1, dtype=dtypes.i32, device=device)
        reason = k1_kernel.qsa_k1_serves(q, view, table, (4,))
        if reason is not None:
            raise AssertionError(f"wide indexer cache was rejected: {reason}")
        if _span_bytes(view.clone()) > _FOUR_GIB:
            raise AssertionError("cloning the view kept the wide span")
        restored = _storage_tensor(view).clone().as_strided(view.shape, view.stride())
        if _span_bytes(restored) <= _FOUR_GIB:
            raise AssertionError("storage copy dropped the wide span")
        if not torch.equal(restored[:n_pages], cache):
            raise AssertionError("storage copy changed the packed pages")
        del view, restored, cache


def _time(fn, *args, rotate, **kwargs):
    """Time ``fn`` under one cache policy for every candidate in the row.

    ``rotate`` is ``run_perftest`` ``num_rotate_args``: ``0`` (the default)
    auto-sizes extra copies from L2 so each timed call sees cold weights,
    ``1`` reuses one buffer set (hot; the older 1047 tables), ``N>1`` uses
    that many copies. Cold is the default because hot reuse credits a
    backend for inter-iteration L2 residency that serving never has -- on
    ``M=8`` decode it flatters live AMD by ~36% and K2 by ~12%. Callers must
    pass paged caches as ``*args`` so deepcopy clones them -- a zero-arg
    closure cannot rotate closed-over tensors. HIP-graph replay is not
    combined with rotation.

    A wide cache is a strided view over about 4 GiB. ``clone`` packs that
    view and the next iteration would take the narrow compile, and one
    copy is already larger than L2, so auto-rotate keeps a single copy.
    An explicit ``rotate`` still makes that many copies. Each copy is the
    storage, and the view is rebuilt per call.
    """
    wide = [
        i
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor) and _span_bytes(arg) > _FOUR_GIB
    ]
    if not wide:
        return run_perftest(fn, *args, num_rotate_args=rotate, **kwargs)
    rotate = max(rotate, 1)
    call_args = list(args)
    shapes = []
    strides = []
    for i in wide:
        shapes.append(tuple(call_args[i].shape))
        strides.append(tuple(int(s) for s in call_args[i].stride()))
        call_args[i] = _storage_tensor(call_args[i])

    def _restore(*restored, **restored_kwargs):
        restored = list(restored)
        for i, shape, stride in zip(wide, shapes, strides):
            restored[i] = restored[i].as_strided(shape, stride)
        return fn(*restored, **restored_kwargs)

    return run_perftest(_restore, *call_args, num_rotate_args=rotate, **kwargs)


def test_indexer_hand_checked_one_row():
    """Tech report ?2.1: I_ib = sum_h ReLU(q[h] . k_bar[b]), complete blocks only.

    One query at token position 6 (0-based) with r=4 and seq_len=8:
      visible = min((6+1)//4, 8//4) = 1  -> only block 0 (tokens 0..3).
    q heads [1,0] and [2,0]; k_bar[0]=[1,0] -> ReLU(1)+ReLU(2)=3.
    k_bar[1]=[10,0] would score 30 but is incomplete -> -inf.
    """
    r = 4
    q = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])  # [1, 2, 2]
    k_bar = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    qpos = torch.tensor([6], dtype=dtypes.i32)
    slen = torch.tensor([8], dtype=dtypes.i32)
    req = torch.tensor([0], dtype=dtypes.i32)

    assert qsa_visible_blocks(qpos, slen, req, r).tolist() == [1]

    scores = qsa_indexer_scores(q, k_bar, qpos, slen, req, r, score_scale=1.0)
    assert scores.shape == (1, 2)
    assert scores[0, 0].item() == 3.0
    assert math.isinf(scores[0, 1].item()) and scores[0, 1].item() < 0

    block_ids = qsa_topk_blocks(scores, k=1)
    assert block_ids.tolist() == [[0]]

    # token_topk = 1 block * 4; width = 4+4-1 = 7.
    indices = qsa_expand_tail(block_ids, qpos, slen, req, r, token_topk=4)
    # expanded 0..3, tail_start=4, tail_count=3 -> 4,5,6.
    assert indices.tolist() == [[0, 1, 2, 3, 4, 5, 6]]


def test_topk_smaller_index_wins_ties():
    """HIP top_k_per_row_decode: equal finite scores keep the smaller block id."""
    r = 4
    q = torch.tensor([[[1.0, 0.0]]])  # H=1
    k_bar = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    qpos = torch.tensor([11], dtype=dtypes.i32)
    slen = torch.tensor([12], dtype=dtypes.i32)
    req = torch.tensor([0], dtype=dtypes.i32)
    scores = qsa_indexer_scores(q, k_bar, qpos, slen, req, r)
    # blocks 0 and 1 both score 1; block 2 scores 0.
    assert scores[0].tolist() == [1.0, 1.0, 0.0]
    block_ids = qsa_topk_blocks(scores, k=2)
    assert block_ids.tolist() == [[0, 1]]

    scaled = qsa_indexer_scores(
        q, k_bar, qpos, slen, req, r, score_scale=FAMILY_A_SCORE_SCALE
    )
    assert qsa_topk_blocks(scaled, k=2).tolist() == [[0, 1]]


def test_incomplete_blocks_not_selected():
    r = 4
    q = torch.tensor([[[1.0, 0.0]]])
    k_bar = torch.tensor([[0.0, 0.0], [9.0, 0.0]])
    qpos = torch.tensor([3], dtype=dtypes.i32)  # visible = 1
    slen = torch.tensor([8], dtype=dtypes.i32)
    req = torch.tensor([0], dtype=dtypes.i32)
    scores = qsa_indexer_scores(q, k_bar, qpos, slen, req, r)
    # block 1 is incomplete despite a huge potential score.
    ids = qsa_topk_blocks(scores, k=2)
    assert ids[0, 0].item() == 0
    assert ids[0, 1].item() == -1


def test_gqa_matches_dense_on_selected():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2] group=2, Hk=1
    k = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 1.0]]])
    v = torch.tensor([[[2.0, 0.0]], [[0.0, 0.0]], [[0.0, 4.0]]])
    indices = torch.tensor([[0, 2]], dtype=dtypes.i32)
    out = qsa_sparse_gqa(q, k, v, indices, softmax_scale=1.0)

    e = math.exp(1.0)
    p_hi = e / (e + 1.0)
    p_lo = 1.0 / (e + 1.0)
    # head 0 attends k[0]=[1,0] and k[2]=[0,1] -> scores 1 and 0.
    expect_h0 = [p_hi * 2.0, p_lo * 4.0]
    # head 1 scores 0 and 1.
    expect_h1 = [p_lo * 2.0, p_hi * 4.0]
    got = out[0].tolist()
    assert abs(got[0][0] - expect_h0[0]) < 1e-5
    assert abs(got[0][1] - expect_h0[1]) < 1e-5
    assert abs(got[1][0] - expect_h1[0]) < 1e-5
    assert abs(got[1][1] - expect_h1[1]) < 1e-5


def test_family_a_shapes_smoke():
    """Family A ABI with a short context (8 blocks << 512)."""
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    m = 1
    n_blocks = 8
    seq = n_blocks * idx.compress_ratio  # 32
    q_idx = torch.zeros(m, idx.n_heads, idx.head_dim)
    q_idx[0, 0, 0] = 1.0
    k_bar = torch.zeros(n_blocks, idx.head_dim)
    k_bar[2, 0] = 1.0  # only block 2 scores 1
    q_gqa = torch.zeros(m, gqa.n_heads, gqa.head_dim)
    q_gqa[0, 0, 0] = 1.0
    k = torch.zeros(seq, gqa.kv_heads, gqa.head_dim)
    v = torch.zeros(seq, gqa.kv_heads, gqa.head_dim)
    v[:, 0, 0] = torch.arange(seq, dtype=dtypes.fp32)
    qpos = torch.tensor([seq - 1], dtype=dtypes.i32)
    slen = torch.tensor([seq], dtype=dtypes.i32)
    req = torch.tensor([0], dtype=dtypes.i32)

    result = qsa_oracle(
        q_idx,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        req,
        idx,
        gqa,
        score_scale=1.0,
        softmax_scale=1.0,
        out_dtype=dtypes.fp32,
    )
    assert result.block_ids.shape == (m, idx.block_budget)
    assert result.indices.shape == (m, idx.index_width)
    assert result.output.shape == (m, gqa.n_heads, gqa.head_dim)
    assert result.block_ids[0, 0].item() == 2
    assert set(result.block_ids[0, :n_blocks].tolist()) == set(range(n_blocks))
    assert set(result.block_ids[0, n_blocks:].tolist()) == {-1}
    # Highest-scoring block 2 is rank 0 -> tokens 8..11; seq is a multiple of r
    # so there is no tail (remaining slots are -1).
    expanded = [t for t in result.indices[0].tolist() if t >= 0]
    assert expanded[:4] == [8, 9, 10, 11]


def test_family_b_shape_constants():
    assert FAMILY_B_INDEXER.head_dim == 128
    assert FAMILY_B_GQA.group_size == 5
    assert FAMILY_B_GQA.n_heads == 10
    assert FAMILY_A_INDEXER.index_width == 2051
    assert FAMILY_A_GQA.group_size == 12


def test_kernel_constants_cover_every_family():
    """The kernels declare their own shape constants, not a model registry.

    Head count aside, nothing downstream re-derives the block budget or the
    compress ratio from the caller's tensors, so a family drifting on those
    axes would slip past the dispatch gate and be silently mis-served.
    """
    for spec in (FAMILY_A_INDEXER, FAMILY_B_INDEXER, FAMILY_B_INDEXER_H8):
        assert spec.n_heads in k1_kernel._SCORE_HEADS
        assert (
            spec.kv_heads,
            spec.head_dim,
            spec.compress_ratio,
            spec.block_budget,
        ) == (k1_kernel._KV_HEADS, k1_kernel._D, k1_kernel._R, k1_kernel._K)
    assert k1_kernel._SCORE_SCALE == FAMILY_A_SCORE_SCALE


def test_paged_roundtrip_tiny():
    """Shuffled pages still gather back to dense (CPU, no kernel)."""
    dense = torch.arange(48, dtype=dtypes.fp32).view(6, 2, 4)
    physical = torch.tensor([1, 0], dtype=dtypes.i32)
    cache, table = pack_paged_cache(dense, page_size=4, physical=physical)
    assert cache.shape == (2, 4, 2, 4)
    assert table.tolist() == [[1, 0]]
    got = gather_paged_cache(cache, table, n_logical=6)
    assert torch.equal(got, dense)
    identity = torch.arange(2, dtype=dtypes.i32).unsqueeze(0)
    wrong = gather_paged_cache(cache, identity, n_logical=6)
    assert not torch.equal(wrong, dense)


def _query_positions(m: int, seq_len: int, device) -> torch.Tensor:
    return torch.arange(seq_len - m, seq_len, device=device, dtype=dtypes.i32)


def _k2_live_n_sel(
    m: int, seq_len: int, token_budget: int, compress_ratio: int
) -> int:
    """Widest packed prefix expand fills for this bench batch.

    Query positions are ``seq_len - m .. seq_len - 1`` in one request of
    length ``seq_len``. A row's live columns are its selected complete
    blocks plus the incomplete block at its own query position, so the
    batch width is the max of those, not the last row alone.
    """
    if compress_ratio < 1 or token_budget % compress_ratio:
        raise ValueError(
            f"token_budget {token_budget} must be a multiple of "
            f"compress_ratio {compress_ratio}"
        )
    if m < 1 or m > seq_len:
        raise ValueError(f"m must fit in seq_len, got m={m} seq_len={seq_len}")
    block_topk = token_budget // compress_ratio
    widest = 0
    for qpos in range(seq_len - m, seq_len):
        visible = min((qpos + 1) // compress_ratio, seq_len // compress_ratio)
        complete = min(visible, block_topk)
        widest = max(widest, complete * compress_ratio + (qpos + 1) % compress_ratio)
    return widest


def _selected_width(indices: torch.Tensor) -> float:
    """Mean non-padding selection slots per row.

    ``indices`` is always allocated ``index_width`` wide (2051 on both
    families), but the indexer can only fill ``complete_blocks * r + tail``
    of it, capped at the budget. Below ``L = 2048`` the remainder is ``-1``
    padding: a quarter of the row is live at ``L = 512`` decode and an eighth
    at ``M = 512`` prefill. Deriving FLOPS and bytes from ``indices.shape[1]``
    therefore overstates the work by up to 8x on those rows. K2's timed
    call passes that prefix as ``n_sel``. Callers that omit it still walk
    the allocated row.
    """
    return float((indices >= 0).sum().item()) / indices.shape[0]


def _pack_family_a(k_bar, k, v, page_size, device):
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    gen_kv = torch.Generator(device=device)
    gen_kv.manual_seed(2)
    index_k = k_bar.unsqueeze(1)  # [n_blocks, 1, D]
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    k_cache, kv_table = pack_paged_cache(k, page_size, generator=gen_kv)
    v_cache, kv_table_v = pack_paged_cache(v, page_size, physical=kv_table[0])
    if not torch.equal(kv_table, kv_table_v):
        raise RuntimeError("K and V page tables diverged")
    return index_cache, index_table, k_cache, v_cache, kv_table


@benchmark()
def bench_qsa_family_a_plumbing(m, seq_len, page_size, dtype, rotate=0):
    """Paged family A tensors + block tables; oracle on gather vs dense.

    No competitor kernel. ``paged_gather`` is the only timed candidate (copy
    through the page table). The oracle is the reference and is not timed.
    """
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtype, device=device)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device)
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)

    index_cache, index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    assert index_cache.shape[2] == idx.kv_heads
    assert k_cache.shape[2] == gqa.kv_heads
    assert k_cache.shape[-1] == gqa.head_dim

    (k_bar_g, k_g, v_g), us = _time(
        gather_qsa_caches,
        index_cache,
        index_table,
        k_cache,
        v_cache,
        kv_table,
        n_blocks,
        seq_len,
        rotate=rotate,
    )
    err_kbar = checkAllclose(
        k_bar.to(dtypes.fp32),
        k_bar_g.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg="paged gather index-K",
    )
    err_k = checkAllclose(
        k.to(dtypes.fp32), k_g.to(dtypes.fp32), rtol=0, atol=0, msg="paged gather K"
    )
    err_v = checkAllclose(
        v.to(dtypes.fp32), v_g.to(dtypes.fp32), rtol=0, atol=0, msg="paged gather V"
    )

    dense = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=FAMILY_A_SCORE_SCALE,
        out_dtype=dtypes.fp32,
    )
    paged = qsa_oracle(
        q_indexer,
        k_bar_g,
        q_gqa,
        k_g,
        v_g,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=FAMILY_A_SCORE_SCALE,
        out_dtype=dtypes.fp32,
    )
    err_o = checkAllclose(
        dense.output, paged.output, rtol=1e-2, atol=1e-2, msg="oracle dense vs paged"
    )
    blocks_match = torch.equal(dense.block_ids, paged.block_ids)
    indices_match = torch.equal(dense.indices, paged.indices)
    if not blocks_match or not indices_match:
        raise AssertionError("oracle block_ids/indices diverged after paged gather")

    elem = dtype.itemsize
    nbytes = (
        n_blocks * idx.kv_heads * idx.head_dim
        + 2 * seq_len * gqa.kv_heads * gqa.head_dim
    ) * elem
    return {
        "gfx": get_gfx(),
        "n_blocks": n_blocks,
        "index_width": idx.index_width,
        "paged_gather us": us,
        "paged_gather TFLOPS": 0.0,
        "paged_gather TB/s": nbytes / us / 1e6,
        "paged_gather err": max(err_kbar, err_k, err_v, err_o),
    }


def _set_mismatch_ratio(
    ref: torch.Tensor,
    got: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> float:
    """Fraction of rows whose selected ids are not the oracle top-k.

    Without ``scores`` the id sets must be equal. With ``scores`` (the
    oracle row that produced ``ref``), a different id is accepted when
    its score is at least one fp32 ulp below the worst score ``ref``
    kept and every strictly better id is still kept. That is an exact
    tie, or the one-ulp boundary a bf16 dot can land on either side of.
    A block any lower still counts as a miss.
    """
    if scores is None:
        miss = 0
        rows = ref.shape[0]
        for i in range(rows):
            a = set(ref[i].tolist()) - {-1}
            b = set(got[i].tolist()) - {-1}
            if a != b:
                miss += 1
        return miss / rows if rows else 0.0

    rows, n_blocks = scores.shape
    if rows == 0:
        return 0.0
    in_range = ((ref < 0) | (ref < n_blocks)).all(dim=1) & (
        (got < 0) | (got < n_blocks)
    ).all(dim=1)
    last = n_blocks - 1
    ref_idx = ref.clamp(0, last)
    got_idx = got.clamp(0, last)
    ref_scores = scores.gather(1, ref_idx).masked_fill(ref < 0, float("inf"))
    kth = ref_scores.min(dim=1).values
    floor = torch.nextafter(kth, torch.full_like(kth, float("-inf")))
    got_scores = scores.gather(1, got_idx)
    got_valid = got >= 0
    bad_score = got_valid & (
        ~torch.isfinite(got_scores) | (got_scores < floor.unsqueeze(1))
    )
    selected = torch.zeros(rows, n_blocks, dtype=torch.bool, device=scores.device)
    # A padded slot clamps to column 0. Scatter that False and it wipes a
    # real selection of block 0, so only the in-range ids are marked.
    safe = got_valid & (got < n_blocks)
    row_idx = torch.arange(rows, device=scores.device).unsqueeze(1).expand_as(got)
    selected[row_idx[safe], got[safe]] = True
    n_got = got_valid.sum(dim=1)
    n_ref = (ref >= 0).sum(dim=1)
    count_bad = (selected.sum(dim=1) != n_got) | (n_got != n_ref)
    required = torch.isfinite(scores) & (scores > kth.unsqueeze(1))
    missing = (required & ~selected).any(dim=1)
    miss = bad_score.any(dim=1) | ~in_range | count_bad | missing
    return float(miss.sum().item()) / rows


def _k1_row_is_packed(row: torch.Tensor) -> str | None:
    """Valid ids, each once, then only ``-1``. ``None`` when the row is packed."""
    seen = set()
    padding = False
    for value in row.tolist():
        if value == -1:
            padding = True
            continue
        if padding:
            return "valid id after -1"
        if value in seen:
            return "duplicate id"
        seen.add(value)
    return None


def _assert_k1_block_ids(ref: torch.Tensor, got: torch.Tensor, what: str) -> None:
    """Set equality, plus no duplicates and a compact valid prefix on ``got``."""
    for i in range(got.shape[0]):
        reason = _k1_row_is_packed(got[i])
        if reason is not None:
            raise AssertionError(f"{what}: row {i} {reason}")
    if _set_mismatch_ratio(ref, got) != 0.0:
        raise AssertionError(f"{what}: set mismatch")


def test_k1_block_ids_require_packed_prefix():
    """A K1 id row is unique valid ids, then only ``-1``.

    Order may differ from the oracle. A duplicate, a valid id after
    ``-1``, or a different id set fails.
    """
    ref = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    _assert_k1_block_ids(ref, torch.tensor([[1, 0, -1]], dtype=torch.int32), "order")
    _assert_k1_block_ids(ref, torch.tensor([[0, 1, -1, -1]], dtype=torch.int32), "pad")
    _assert_k1_block_ids(
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.full((1, 4), -1, dtype=torch.int32),
        "empty",
    )
    same = torch.tensor([[0, -1]], dtype=torch.int32)
    try:
        _assert_k1_block_ids(same, torch.tensor([[0, 0]], dtype=torch.int32), "dup")
    except AssertionError as exc:
        if "duplicate" not in str(exc):
            raise
    else:
        raise AssertionError("duplicate ids passed the K1 check")
    try:
        _assert_k1_block_ids(ref, torch.tensor([[0, -1, 1]], dtype=torch.int32), "hole")
    except AssertionError as exc:
        if "after -1" not in str(exc):
            raise
    else:
        raise AssertionError("a valid id after -1 passed the K1 check")
    try:
        _assert_k1_block_ids(ref, torch.tensor([[2, 3, -1]], dtype=torch.int32), "sets")
    except AssertionError as exc:
        if "set mismatch" not in str(exc):
            raise
    else:
        raise AssertionError("a set mismatch passed the K1 check")


def test_k1_family_a_set_equality_short_decode():
    """FlyDSL K1 block-id sets match the oracle on short family A decode."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 4, 512, 16
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
    )
    _assert_k1_block_ids(ref_ids, got, "K1 block-id set diverged from the oracle")


def test_k1_family_a_set_equality_two_tiles():
    """FlyDSL K1 still matches the oracle when n_blocks exceeds one 512-slot tile."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 2, 4096, 16
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks > 512
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
    )
    _assert_k1_block_ids(
        ref_ids, got, "K1 two-tile block-id set diverged from the oracle"
    )


def test_k1_family_a_set_equality_wide_stream():
    """FlyDSL K1 matches the oracle on 128k rows that take streaming radix."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 1, 131072, 16
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks >= 32768
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
    )
    _assert_k1_block_ids(
        ref_ids, got, "K1 wide-row block-id set diverged from the oracle"
    )


def test_k1_family_a_set_equality_prefill():
    """The 16-row single-request scorer matches the oracle at prefill M=512."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 512, 4096, 16
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks > 512
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
    )
    _assert_k1_block_ids(
        ref_ids, got, "K1 prefill block-id set diverged from the oracle"
    )


def test_k1_prefill_padded_page_table():
    """Page-table entries past the context must not be loaded.

    One request, M=32, context 4096. Sixty-four pages of 16 cover the
    1024 visible blocks; four more entries are padding. Pad 0 matches
    the oracle. Pad -1 and a huge page id must not fault or change
    the selected set.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, context, page_size, n_real_pages, n_pad = 32, 4096, 16, 64, 4
    n_blocks = context // idx.compress_ratio
    if n_real_pages * page_size != n_blocks:
        raise AssertionError("the real pages must cover the context exactly")
    torch.manual_seed(0)
    q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, context, device)
    slen = torch.full((1,), context, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    index_cache, index_table = pack_paged_cache(
        k_bar.unsqueeze(1), page_size, generator=gen
    )
    if index_table.shape[1] != n_real_pages:
        raise AssertionError(f"expected {n_real_pages} pages, got {index_table.shape}")
    ref_scores = qsa_indexer_scores(
        q,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    for name, pad_value in (("zero", 0), ("negative", -1), ("huge", 100000)):
        pad = torch.full((1, n_pad), pad_value, dtype=dtypes.i32, device=device)
        table = torch.cat((index_table, pad), dim=1)
        got = qsa_k1_block_ids(
            q, index_cache, table, token_to_req, qpos, slen, heads=(4,)
        )
        _assert_k1_block_ids(
            ref_ids, got, f"K1 prefill pad {name} changed the selected set"
        )


def test_k1_table_padded_to_max_model_len():
    """A block table far wider than any request selects what the trimmed one does.

    vLLM sizes the block table for max_model_len and zero-fills past each
    request's pages, so K1 sees tens of thousands of columns no row can see.
    The table here is 4096 pages of 16 wide, which also crosses into the
    streaming selector. Every request shares one packed cache and differs in
    its length. Covers the one-row scorer (decode, and decode mixed with a
    prefill chunk) and the 16-row scorer (one request).
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    page_size, max_pages, context = 16, 4096, 65536
    n_blocks = context // idx.compress_ratio
    assert max_pages * page_size >= 32768
    torch.manual_seed(0)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    index_cache, index_table = pack_paged_cache(
        k_bar.unsqueeze(1), page_size, generator=gen
    )

    def run(seq_lens, token_to_req, qpos, what):
        n_req = len(seq_lens)
        slen = torch.tensor(seq_lens, dtype=dtypes.i32, device=device)
        m = token_to_req.shape[0]
        q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
        ref_ids = qsa_topk_blocks(
            qsa_indexer_scores(
                q,
                k_bar,
                qpos,
                slen,
                token_to_req,
                idx.compress_ratio,
                score_scale=FAMILY_A_SCORE_SCALE,
            ),
            idx.block_budget,
        )
        table = torch.zeros(n_req, max_pages, dtype=dtypes.i32, device=device)
        table[:, : index_table.shape[1]] = index_table
        got = qsa_k1_block_ids(
            q, index_cache, table, token_to_req, qpos, slen, heads=(4,)
        )
        _assert_k1_block_ids(ref_ids, got, what)

    decode_lens = [3000, 9000, 30000, context]
    t2r = torch.arange(len(decode_lens), dtype=dtypes.i32, device=device)
    run(
        decode_lens,
        t2r,
        torch.tensor(decode_lens, dtype=dtypes.i32, device=device) - 1,
        "K1 decode on a max_model_len table diverged from the oracle",
    )

    chunk = 64
    mixed_lens = [chunk] + decode_lens
    mixed_t2r = torch.cat(
        (torch.zeros(chunk, dtype=dtypes.i32, device=device), t2r + 1)
    )
    mixed_pos = torch.cat(
        (
            torch.arange(chunk, dtype=dtypes.i32, device=device),
            torch.tensor(decode_lens, dtype=dtypes.i32, device=device) - 1,
        )
    )
    run(
        mixed_lens,
        mixed_t2r,
        mixed_pos,
        "K1 mixed prefill+decode on a max_model_len table diverged",
    )

    # Within a 16-row block the last row sees 4 blocks more than the first.
    # Starting at 2559 puts the first row of every eighth block exactly on a
    # 32-column tile edge (640 + 32k visible), so a tile skip keyed on the
    # wrong row drops blocks the later rows must select from (> 512 visible).
    m, start = 256, 2559
    run(
        [start + m],
        torch.zeros(m, dtype=dtypes.i32, device=device),
        torch.arange(start, start + m, dtype=dtypes.i32, device=device),
        "K1 16-row prefill on a max_model_len table diverged",
    )


def test_k1_padded_decode_graph_capture():
    """Decode on a max_model_len table captures into a HIP graph and replays.

    vLLM captures decode into full graphs, where the padded-table width
    readback is not allowed. Replays twice, the second time with new queries
    written into the captured buffer, against the oracle.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    page_size, max_pages, context = 16, 4096, 65536
    n_blocks = context // idx.compress_ratio
    torch.manual_seed(0)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    index_cache, index_table = pack_paged_cache(
        k_bar.unsqueeze(1), page_size, generator=gen
    )
    decode_lens = [3000, 9000, 30000, context]
    m = len(decode_lens)
    slen = torch.tensor(decode_lens, dtype=dtypes.i32, device=device)
    t2r = torch.arange(m, dtype=dtypes.i32, device=device)
    qpos = slen - 1
    table = torch.zeros(m, max_pages, dtype=dtypes.i32, device=device)
    table[:, : index_table.shape[1]] = index_table
    q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    out = torch.empty(m, k1_kernel._K, dtype=dtypes.i32, device=device)

    def oracle():
        return qsa_topk_blocks(
            qsa_indexer_scores(
                q,
                k_bar,
                qpos,
                slen,
                t2r,
                idx.compress_ratio,
                score_scale=FAMILY_A_SCORE_SCALE,
            ),
            idx.block_budget,
        )

    def launch():
        qsa_k1_block_ids(q, index_cache, table, t2r, qpos, slen, out=out, heads=(4,))

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for replay in range(2):
        if replay:
            q.copy_(torch.randn_like(q))
        out.fill_(-2)
        graph.replay()
        torch.cuda.synchronize()
        _assert_k1_block_ids(
            oracle(), out, f"K1 padded decode graph replay {replay} diverged"
        )


def test_k1_decode_rejects_invalid_page_ids():
    """The one-row scorer must not load a page id that is not in the cache.

    Both cases use M=4 so the one-row scorer runs, and a table wider
    than 512 columns so the emit kernel does not. A request with no
    live blocks still reads entry 0; that entry is ``-1``. A second
    request has ``-1`` on a page inside the visible range.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, page_size = 4, 16

    # No live blocks. Entry 0 is -1, and the table is wide enough that
    # dead columns still index it.
    n_pages = 33
    q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_cache = torch.zeros(
        1, page_size, 1, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    table = torch.full((1, n_pages), -1, dtype=dtypes.i32, device=device)
    qpos = torch.zeros(m, dtype=dtypes.i32, device=device)
    slen = torch.zeros(1, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    got = qsa_k1_block_ids(q, k_cache, table, token_to_req, qpos, slen, heads=(4,))
    _assert_k1_block_ids(
        torch.full_like(got, -1),
        got,
        "K1 decode with no live pages selected a block",
    )

    # -1 inside the visible range. The other pages stay real.
    context = 4096
    n_blocks = context // idx.compress_ratio
    torch.manual_seed(0)
    q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, context, device)
    slen = torch.full((1,), context, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    index_cache, index_table = pack_paged_cache(
        k_bar.unsqueeze(1), page_size, generator=gen
    )
    bad_page = 3
    index_table = index_table.clone()
    index_table[0, bad_page] = -1
    ref_scores = qsa_indexer_scores(
        q,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    block0 = bad_page * page_size
    ref_scores = ref_scores.clone()
    ref_scores[:, block0 : block0 + page_size] = float("-inf")
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q, index_cache, index_table, token_to_req, qpos, slen, heads=(4,)
    )
    _assert_k1_block_ids(ref_ids, got, "K1 decode kept a block whose page id is -1")


def test_k1_gfx942_h8_skips_prefill_tile():
    """gfx942 H=8 prefill must not launch the 16-row tile.

    That tile is 73792 bytes at H=8, past gfx942's 65536-byte budget.
    H=4 still fits, and gfx950 H=8 keeps the 16-by-32 tile. This is a
    dispatch decision: GPU 6 is gfx950 and does not execute the gfx942 path.
    """
    h8 = k1_kernel._k1_prefill_lds_bytes(8)
    h4 = k1_kernel._k1_prefill_lds_bytes(4)
    if h8 != 73792 or h8 <= 65536:
        raise AssertionError(f"H=8 prefill LDS should be 73792, got {h8}")
    if h4 > 65536:
        raise AssertionError(f"H=4 prefill LDS should fit gfx942, got {h4}")
    if k1_kernel._k1_uses_prefill_scorer(1, 16, 8, "gfx942"):
        raise AssertionError("gfx942 H=8 still dispatches the 16-row tile")
    if k1_kernel._k1_uses_prefill_scorer(1, 16, 8, "gfx942:sramecc+:xnack-"):
        raise AssertionError(
            "gfx942 H=8 with an arch suffix still dispatches the 16-row tile"
        )
    if not k1_kernel._k1_uses_prefill_scorer(1, 16, 8, "gfx950"):
        raise AssertionError("gfx950 H=8 left the 16-row tile")
    if not k1_kernel._k1_uses_prefill_scorer(1, 32, 4, "gfx942"):
        raise AssertionError("gfx942 H=4 left the 16-row tile")
    if k1_kernel._k1_uses_prefill_scorer(2, 32, 8, "gfx950"):
        raise AssertionError("multi-request prefill entered the 16-row tile")
    if k1_kernel._k1_uses_prefill_scorer(1, 8, 8, "gfx950"):
        raise AssertionError("short M entered the 16-row tile")
    if not k1_kernel._k1_uses_prefill_scorer(1, 8, 4, "gfx942"):
        raise AssertionError("gfx942 H=4 left the 16-row tile at 8 rows")
    if k1_kernel._k1_uses_prefill_scorer(1, 7, 4, "gfx942"):
        raise AssertionError("gfx942 H=4 entered the 16-row tile below 8 rows")
    if k1_kernel._k1_uses_prefill_scorer(1, 8, 8, "gfx942"):
        raise AssertionError("gfx942 H=8 entered the 16-row tile at 8 rows")
    if k1_kernel._k1_uses_prefill_scorer(1, 8, 4, "gfx950"):
        raise AssertionError("gfx950 H=4 entered the 16-row tile at 8 rows")


def test_qsa_arch_allowlist():
    """qsa_device_arch accepts gfx942 and gfx950, including an ISA suffix.

    A name that does not start with gfx950 used to take the gfx942 tile.
    Anything else raises.
    """
    from aiter.ops.flydsl.kernels.qsa.arch import qsa_device_arch

    accepted = {
        "gfx942": "gfx942",
        "gfx950": "gfx950",
        "gfx942:sramecc+:xnack-": "gfx942",
        "gfx950:sramecc+:xnack-": "gfx950",
    }
    for raw, want in accepted.items():
        got = qsa_device_arch(raw)
        if got != want:
            raise AssertionError(f"{raw!r} resolved to {got!r}, want {want!r}")
    for bad in ("gfx1100", "gfx1250", "gfx90a", "gfx9420", "GFX950", ""):
        try:
            qsa_device_arch(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")


def test_k1_page_past_4gib():
    """A physical indexer page at byte offset 2^32 must not alias page 0.

    A page is 4096 bytes (page 16, one KV head, D=128, bf16), so physical
    page 1048576 starts at 4 GiB. The table is 33 pages wide so the
    scorers run, not emit. Logical page 32 is that far page and holds
    ones; every other logical page is physical page 0 and holds zeros.
    Q is ones, so only blocks 512..527 score above zero. M=4 uses the
    one-row scorer and M=32 the prefill scorer.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    page_size = 16
    page_bytes = page_size * 1 * idx.head_dim * dtypes.bf16.itemsize
    alias = (1 << 32) // page_bytes
    if alias * page_bytes != 1 << 32:
        raise AssertionError(f"page of {page_bytes} bytes does not divide 4 GiB")
    n_pages = alias + 1
    n_logical = 33
    far_logical = n_logical - 1
    n_columns = n_logical * page_size
    need = n_pages * page_bytes
    free, _total = torch.cuda.mem_get_info(device)
    if free < need + (1 << 30):
        aiter.logger.warning(
            "skip K1 4GiB page test: need %s bytes, %s free", need, free
        )
        return
    k_cache = torch.empty(
        n_pages, page_size, 1, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_cache[0].zero_()
    k_cache[alias].fill_(1)
    table = torch.zeros(1, n_logical, dtype=dtypes.i32, device=device)
    table[0, far_logical] = alias
    context = n_columns * idx.compress_ratio
    k_bar = torch.zeros(n_columns, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar[far_logical * page_size : n_columns] = 1
    for m in (4, 32):
        q = torch.ones(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
        qpos = torch.full((m,), context - 1, dtype=dtypes.i32, device=device)
        slen = torch.full((1,), context, dtype=dtypes.i32, device=device)
        token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
        ref_scores = qsa_indexer_scores(
            q,
            k_bar,
            qpos,
            slen,
            token_to_req,
            idx.compress_ratio,
            score_scale=FAMILY_A_SCORE_SCALE,
        )
        ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
        got = qsa_k1_block_ids(q, k_cache, table, token_to_req, qpos, slen, heads=(4,))
        _assert_k1_block_ids(
            ref_ids, got, f"K1 page past 4 GiB aliased page 0 at M={m}"
        )


def test_expand_rejects_out_that_cannot_hold_the_stores():
    """A caller ``out`` is stored across ``token_topk + compress_ratio - 1`` columns.

    The kernel's store mask uses that width, not ``out.shape``. A narrower
    buffer, a non-int32 buffer, or a buffer on another device is rejected
    before launch.
    """
    if not torch.cuda.is_available():
        return
    rows, compress_ratio, token_topk = 2, 4, 8
    output_width = token_topk + compress_ratio - 1
    device = torch.device("cuda")
    block_indices = torch.zeros(
        (rows, token_topk // compress_ratio), dtype=torch.int32, device=device
    )
    query_positions = torch.zeros(rows, dtype=torch.int32, device=device)
    sequence_lengths = torch.zeros(1, dtype=torch.int32, device=device)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)

    def expand(out):
        return expand_qsa_block_indices_cuda(
            block_indices,
            query_positions,
            sequence_lengths,
            token_to_req,
            compress_ratio,
            token_topk,
            out,
        )

    rejected = (
        torch.empty((rows, 1), dtype=torch.int32, device=device),
        torch.empty((rows, output_width), dtype=torch.int64, device=device),
        torch.empty((rows, output_width), dtype=torch.int32),
    )
    for out in rejected:
        try:
            expand(out)
        except ValueError as exc:
            if "QSA expand out" not in str(exc):
                raise
        else:
            raise AssertionError(
                f"accepted out {out.dtype} {tuple(out.shape)} {out.device}"
            )
    empty_rows = torch.zeros(
        (0, token_topk // compress_ratio), dtype=torch.int32, device=device
    )
    empty_out = torch.empty((0, output_width), dtype=torch.int32, device=device)
    got = expand_qsa_block_indices_cuda(
        empty_rows,
        query_positions[:0],
        sequence_lengths,
        token_to_req[:0],
        compress_ratio,
        token_topk,
        empty_out,
    )
    if got.data_ptr() != empty_out.data_ptr():
        raise AssertionError("a matching caller out was replaced")


def test_sparse_attention_rejects_out_that_does_not_match_q():
    """The GQA store writes ``HEAD_DIM`` elements at unit stride.

    A caller ``out`` has to be the contiguous ``q`` tensor: same shape, dtype,
    and GPU. ``[M, Hq, 1]`` is rejected before launch.
    """
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    rows, n_heads, head_dim = 2, 2, 16
    q = torch.empty((rows, n_heads, head_dim), dtype=torch.bfloat16, device=device)
    k_cache = torch.empty((1, 16, 1, head_dim), dtype=torch.bfloat16, device=device)
    indices = torch.zeros((rows, 4), dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    strided = torch.empty(
        (rows, n_heads, head_dim * 2), dtype=torch.bfloat16, device=device
    )[..., ::2]

    def attend(query, idx, req, out):
        return qsa_sparse_paged_attention(
            query, k_cache, k_cache.clone(), idx, block_table, req, out
        )

    rejected = (
        torch.empty((rows, n_heads, 1), dtype=torch.bfloat16, device=device),
        torch.empty((rows, n_heads, head_dim), dtype=torch.float32, device=device),
        torch.empty((rows, n_heads, head_dim), dtype=torch.bfloat16),
        strided,
    )
    for out in rejected:
        try:
            attend(q, indices, token_to_req, out)
        except ValueError as exc:
            if "QSA sparse attention out" not in str(exc):
                raise
        else:
            raise AssertionError(
                f"accepted out {out.dtype} {tuple(out.shape)} {out.device}"
            )
    empty_out = torch.empty((0, n_heads, head_dim), dtype=torch.bfloat16, device=device)
    got = attend(q[:0], indices[:0], token_to_req[:0], empty_out)
    if got.data_ptr() != empty_out.data_ptr():
        raise AssertionError("a matching caller out was replaced")


def test_k1_serves_padded_page_and_rejects_misaligned():
    """A padded page stride is in contract; a 16-byte gather that cannot align is not.

    vLLM keeps D contiguous and the token stride at D, and pads the page
    stride. ``numel()`` undercounts that view. A D stride other than 1, or a
    page/token/head stride that is not a multiple of 8 elements, is rejected
    so ``auto`` stays on Triton.
    """
    page_size, d = 16, 128
    n_pages = 3
    page_elems = page_size * d
    q = torch.zeros(1, 4, d, dtype=torch.bfloat16)
    table = torch.zeros(1, 1, dtype=torch.int32)
    dense = torch.zeros(n_pages, page_size, 1, d, dtype=torch.bfloat16)
    if k1_kernel.qsa_k1_serves(q, dense, table, (4,)) is not None:
        raise AssertionError("a packed indexer cache was rejected")
    page_stride = page_elems + 256
    parent = torch.empty(n_pages, page_stride, dtype=torch.bfloat16)
    view = parent.as_strided((n_pages, page_size, 1, d), (page_stride, d, d, 1))
    if view.is_contiguous():
        raise AssertionError("padded page stride should not be contiguous")
    if k1_kernel.qsa_k1_serves(q, view, table, (4,)) is not None:
        raise AssertionError(f"padded page stride was rejected: {view.stride()}")
    span = k1_kernel._span_bytes(view)
    packed = view.numel() * view.element_size()
    if span <= packed:
        raise AssertionError(f"span {span} did not exceed packed bytes {packed}")
    bad = parent.as_strided((n_pages, page_size, 1, d), (page_elems + 4, d, d, 1))
    reason = k1_kernel.qsa_k1_serves(q, bad, table, (4,))
    if reason is None or "multiples of 8" not in reason:
        raise AssertionError(f"misaligned page stride served: {reason}")
    wide = torch.empty(n_pages, page_size, 1, d * 2, dtype=torch.bfloat16)
    strided_d = wide.as_strided(
        dense.shape, (wide.stride(0), wide.stride(1), wide.stride(2), 2)
    )
    reason = k1_kernel.qsa_k1_serves(q, strided_d, table, (4,))
    if reason is None or "unit D" not in reason:
        raise AssertionError(f"strided D was served: {reason}")


def _indexer_cache_with_page_gap(cache, gap_elems: int) -> torch.Tensor:
    """Same pages, with ``gap_elems`` unused elements between them."""
    n_pages, page_size, n_heads, head_dim = cache.shape
    page_elems = page_size * n_heads * head_dim
    page_stride = page_elems + gap_elems
    parent = cache.new_empty(n_pages, page_stride)
    parent.zero_()
    parent[:, :page_elems] = cache.reshape(n_pages, page_elems)
    return parent.as_strided(
        cache.shape, (page_stride, n_heads * head_dim, head_dim, 1)
    )


def test_k1_padded_page_stride_matches_packed():
    """K1 reads a padded page stride in place, on both scorers.

    The one-row scorer (M=4) and the 16-row scorer (M=16) must match the
    packed cache. ``n_columns`` is past 512 so this is the gather, not emit.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    page_size = 16
    n_blocks = 528
    seq_len = n_blocks * idx.compress_ratio
    torch.manual_seed(0)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    index_k = k_bar.unsqueeze(1)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen)
    gapped = _indexer_cache_with_page_gap(index_cache, 256)
    if gapped.is_contiguous():
        raise AssertionError("gapped indexer cache is contiguous")
    if (
        k1_kernel.qsa_k1_serves(
            torch.empty(1, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device),
            gapped,
            index_table,
            (4,),
        )
        is not None
    ):
        raise AssertionError("gapped indexer cache was rejected")
    for m in (4, 16):
        q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
        qpos = _query_positions(m, seq_len, device)
        slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
        token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
        packed = qsa_k1_block_ids(
            q, index_cache, index_table, token_to_req, qpos, slen, heads=(4,)
        )
        got = qsa_k1_block_ids(
            q, gapped, index_table, token_to_req, qpos, slen, heads=(4,)
        )
        if not torch.equal(packed, got):
            raise AssertionError(f"padded page stride diverged at M={m}")


def test_k1_wide_padded_page_does_not_alias():
    """A page past 4 GiB is addressed by its real stride, not by numel.

    Doubling the page stride puts view page ``alias / 2`` on the byte at
    2^32. The view's packed byte count stays under 4 GiB, which is the
    decision ``numel()`` would have made.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    page_size = 16
    page_elems = page_size * idx.head_dim
    page_bytes = page_elems * dtypes.bf16.itemsize
    alias = (1 << 32) // page_bytes
    if alias * page_bytes != 1 << 32 or alias % 2:
        raise AssertionError(f"page of {page_bytes} bytes does not split 4 GiB")
    n_storage = alias + 1
    view_pages = alias // 2 + 1
    if (view_pages - 1) * 2 != alias:
        raise AssertionError("doubled stride does not land on the 4 GiB page")
    need = n_storage * page_bytes
    free, _total = torch.cuda.mem_get_info(device)
    if free < need + (1 << 30):
        aiter.logger.warning(
            "skip K1 wide padded page test: need %s bytes, %s free", need, free
        )
        return
    parent = torch.empty(
        n_storage, page_size, 1, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    parent[0].zero_()
    parent[alias].fill_(1)
    view = parent.as_strided(
        (view_pages, page_size, 1, idx.head_dim),
        (2 * page_elems, idx.head_dim, idx.head_dim, 1),
    )
    span = k1_kernel._span_bytes(view)
    packed = view.numel() * view.element_size()
    if span <= (1 << 32) or packed > (1 << 32):
        raise AssertionError(f"span {span} packed {packed} is not the wide-view case")
    n_logical = 33
    far_logical = n_logical - 1
    n_columns = n_logical * page_size
    table = torch.zeros(1, n_logical, dtype=dtypes.i32, device=device)
    table[0, far_logical] = view_pages - 1
    context = n_columns * idx.compress_ratio
    k_bar = torch.zeros(n_columns, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar[far_logical * page_size : n_columns] = 1
    for m in (4, 16):
        q = torch.ones(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
        qpos = torch.full((m,), context - 1, dtype=dtypes.i32, device=device)
        slen = torch.full((1,), context, dtype=dtypes.i32, device=device)
        token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
        ref_scores = qsa_indexer_scores(
            q,
            k_bar,
            qpos,
            slen,
            token_to_req,
            idx.compress_ratio,
            score_scale=FAMILY_A_SCORE_SCALE,
        )
        ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
        got = qsa_k1_block_ids(q, view, table, token_to_req, qpos, slen, heads=(4,))
        _assert_k1_block_ids(
            ref_ids, got, f"K1 wide padded page aliased page 0 at M={m}"
        )


def test_k2_family_a_decode_matches_oracle():
    """Family A FlyDSL K2 decode matches qsa_sparse_gqa on paged K/V."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    m, seq_len, page_size, width = 2, 64, 16, 8
    torch.manual_seed(0)
    q = torch.randn(m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    k = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    v = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    indices = torch.randint(0, seq_len, (m, width), dtype=dtypes.i32, device=device)
    indices[:, -1] = -1
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(2)
    k_cache, kv_table = pack_paged_cache(k, page_size, generator=gen)
    v_cache, kv_table_v = pack_paged_cache(v, page_size, physical=kv_table[0])
    assert torch.equal(kv_table, kv_table_v)
    ref = qsa_sparse_gqa(q, k, v, indices)
    out = qsa_k2(q, k_cache, v_cache, indices, kv_table, token_to_req)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="flydsl K2 vs oracle GQA",
    )
    if err != 0:
        raise AssertionError(f"K2 decode diverged from the oracle (err={err})")


def test_k2_live_prefix_matches_full_width():
    """Walking the packed live prefix matches walking the allocated -1 tail."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    page_size = 16
    for m, seq_len in ((1, 512), (1, 2048), (1, 8192), (8, 512), (8, 8192)):
        torch.manual_seed(0)
        n_blocks = seq_len // idx.compress_ratio
        q_indexer = torch.randn(
            m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
        )
        k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
        q = torch.randn(
            m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
        )
        k = torch.randn(
            seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
        )
        v = torch.randn(
            seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
        )
        qpos = _query_positions(m, seq_len, device)
        slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
        token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
        _index_cache, _index_table, k_cache, v_cache, kv_table = _pack_family_a(
            k_bar, k, v, page_size, device
        )
        ref = qsa_oracle(
            q_indexer,
            k_bar,
            q,
            k,
            v,
            qpos,
            slen,
            token_to_req,
            idx,
            gqa,
            score_scale=FAMILY_A_SCORE_SCALE,
            out_dtype=dtypes.fp32,
        )
        indices = ref.indices.contiguous()
        live = _k2_live_n_sel(m, seq_len, idx.token_budget, idx.compress_ratio)
        if m == 1 and seq_len == 512:
            try:
                qsa_k2(
                    q, k_cache, v_cache, indices, kv_table, token_to_req, n_sel=True
                )
            except TypeError:
                pass
            else:
                raise AssertionError("bool n_sel was accepted")
            try:
                qsa_k2(
                    q,
                    k_cache,
                    v_cache,
                    indices,
                    kv_table,
                    token_to_req,
                    n_sel=int(indices.shape[1]) + 1,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("n_sel past the row was accepted")
        if live <= 0 or live > indices.shape[1]:
            raise AssertionError(f"live n_sel {live} outside 1..{indices.shape[1]}")
        if int((indices[:, live:] >= 0).any().item()):
            raise AssertionError(f"live token past n_sel={live} at M={m} L={seq_len}")
        if int((indices >= 0).sum(dim=1).max().item()) > live:
            raise AssertionError(f"a row is wider than n_sel={live} at M={m} L={seq_len}")
        full = qsa_k2(q, k_cache, v_cache, indices, kv_table, token_to_req)
        short = qsa_k2(
            q, k_cache, v_cache, indices, kv_table, token_to_req, n_sel=live
        )
        err = checkAllclose(
            full.to(dtypes.fp32),
            short.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"live prefix vs full width M={m} L={seq_len}",
        )
        if err != 0:
            raise AssertionError(
                f"live prefix diverged from the full row at M={m} L={seq_len} (err={err})"
            )
        err = checkAllclose(
            ref.output,
            short.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"live prefix vs oracle M={m} L={seq_len}",
        )
        if err != 0:
            raise AssertionError(
                f"live prefix diverged from the oracle at M={m} L={seq_len} (err={err})"
            )


def test_k2_family_a_prefill_matches_oracle():
    """The BLOCK_N=64/two-wave K2 specialization matches at prefill M=512."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    m, seq_len, page_size, width = 512, 64, 16, 8
    torch.manual_seed(0)
    q = torch.randn(m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    k = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    v = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    indices = torch.randint(0, seq_len, (m, width), dtype=dtypes.i32, device=device)
    indices[:, -1] = -1
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(2)
    k_cache, kv_table = pack_paged_cache(k, page_size, generator=gen)
    v_cache, kv_table_v = pack_paged_cache(v, page_size, physical=kv_table[0])
    assert torch.equal(kv_table, kv_table_v)
    ref = qsa_sparse_gqa(q, k, v, indices)
    out = qsa_k2(q, k_cache, v_cache, indices, kv_table, token_to_req)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="flydsl K2 prefill vs oracle GQA",
    )
    if err != 0:
        raise AssertionError(f"K2 prefill diverged from the oracle (err={err})")


def test_k2_page_past_4gib():
    """A physical page at byte offset 2^32 must not alias page 0.

    Family A pages are 16384 bytes (page 16, 2 KV heads, D=256, bf16), so
    physical page 262144 starts at 4 GiB. Q is zero, so each live token
    contributes its V with equal weight. Page 0 is ones and the far page
    is twos. One row gathers both in a single tile.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    page_size = 16
    page_bytes = page_size * gqa.kv_heads * gqa.head_dim * dtypes.bf16.itemsize
    alias = (1 << 32) // page_bytes
    if alias * page_bytes != 1 << 32:
        raise AssertionError(f"page of {page_bytes} bytes does not divide 4 GiB")
    n_pages = alias + 1
    need = n_pages * page_bytes * 2
    free, _total = torch.cuda.mem_get_info(device)
    if free < need + (1 << 30):
        aiter.logger.warning(
            "skip K2 4GiB page test: need %s bytes, %s free", need, free
        )
        return
    q = torch.zeros(3, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    k_cache = torch.empty(
        n_pages,
        page_size,
        gqa.kv_heads,
        gqa.head_dim,
        dtype=dtypes.bf16,
        device=device,
    )
    v_cache = torch.empty_like(k_cache)
    k_cache[0].zero_()
    k_cache[alias].zero_()
    v_cache[0].fill_(1)
    v_cache[alias].fill_(2)
    page_table = torch.zeros(1, n_pages, dtype=dtypes.i32, device=device)
    page_table[0, 0] = 0
    page_table[0, alias] = alias
    far_tok = alias * page_size
    indices = torch.tensor(
        [[far_tok, 0], [far_tok, -1], [0, -1]], dtype=dtypes.i32, device=device
    )
    token_to_req = torch.zeros(3, dtype=dtypes.i32, device=device)
    out = qsa_k2(q, k_cache, v_cache, indices, page_table, token_to_req)
    got = out.float()
    expect = (
        torch.tensor([1.5, 2.0, 1.0], dtype=torch.float32, device=device)
        .view(3, 1, 1)
        .expand_as(got)
    )
    if not torch.equal(got, expect):
        raise AssertionError(
            "K2 page past 4 GiB aliased or collapsed: "
            f"row means {got.mean(dim=(1, 2)).tolist()}"
        )


def _interleave_kv(k_cache, v_cache):
    """vLLM's paged layout: one ``[pages, page_size, H, 2 * D]`` buffer, K|V."""
    d = k_cache.shape[-1]
    kv = torch.cat((k_cache, v_cache), dim=-1)
    return kv[..., :d], kv[..., d:]


def test_k2_interleaved_kv_view_matches_contiguous():
    """K2 reads K|V-interleaved cache views in place, bit-equal to copies.

    Covers family A and its TP2 shard (12 query heads over 1 KV head), at
    decode and prefill M.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    seq_len, page_size, width = 256, 16, 64
    for n_heads, kv_heads in (
        (gqa.n_heads, gqa.kv_heads),
        (gqa.n_heads // 2, gqa.kv_heads // 2),
    ):
        for m in (2, 512):
            torch.manual_seed(m)
            q = torch.randn(m, n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
            k = torch.randn(
                seq_len, kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
            )
            v = torch.randn_like(k)
            indices = torch.randint(
                0, seq_len, (m, width), dtype=dtypes.i32, device=device
            )
            indices[:, -3:] = -1
            token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
            gen = torch.Generator(device=device)
            gen.manual_seed(2)
            k_cache, kv_table = pack_paged_cache(k, page_size, generator=gen)
            v_cache, _ = pack_paged_cache(v, page_size, physical=kv_table[0])
            k_view, v_view = _interleave_kv(k_cache, v_cache)
            if k_view.is_contiguous():
                raise AssertionError("interleaved K view should be strided")
            ref = qsa_k2(q, k_cache, v_cache, indices, kv_table, token_to_req)
            got = qsa_k2(q, k_view, v_view, indices, kv_table, token_to_req)
            if not torch.equal(ref, got):
                diff = (ref.float() - got.float()).abs().max().item()
                raise AssertionError(
                    f"K2 on interleaved K|V diverged at Hq={n_heads} "
                    f"Hk={kv_heads} M={m}: max |diff| {diff}"
                )


def test_k2_interleaved_kv_page_past_4gib():
    """The wide path honours the view's page stride, not a dense one.

    An interleaved K view of 2 KV heads at D=256 has a 32768-byte page
    stride, so page 131072 starts at 4 GiB. The view's numel alone would
    stay under 4 GiB and pick the narrow descriptor.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    page_size = 16
    page_bytes = 2 * page_size * gqa.kv_heads * gqa.head_dim * dtypes.bf16.itemsize
    alias = (1 << 32) // page_bytes
    if alias * page_bytes != 1 << 32:
        raise AssertionError(f"page of {page_bytes} bytes does not divide 4 GiB")
    n_pages = alias + 1
    need = n_pages * page_bytes
    free, _total = torch.cuda.mem_get_info(device)
    if free < need + (1 << 30):
        aiter.logger.warning(
            "skip K2 interleaved 4GiB page test: need %s bytes, %s free", need, free
        )
        return
    kv = torch.empty(
        n_pages,
        page_size,
        gqa.kv_heads,
        2 * gqa.head_dim,
        dtype=dtypes.bf16,
        device=device,
    )
    k_view, v_view = kv[..., : gqa.head_dim], kv[..., gqa.head_dim :]
    k_view[0].zero_()
    k_view[alias].zero_()
    v_view[0].fill_(1)
    v_view[alias].fill_(2)
    q = torch.zeros(3, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    page_table = torch.zeros(1, n_pages, dtype=dtypes.i32, device=device)
    page_table[0, alias] = alias
    far_tok = alias * page_size
    indices = torch.tensor(
        [[far_tok, 0], [far_tok, -1], [0, -1]], dtype=dtypes.i32, device=device
    )
    token_to_req = torch.zeros(3, dtype=dtypes.i32, device=device)
    got = qsa_k2(q, k_view, v_view, indices, page_table, token_to_req).float()
    expect = (
        torch.tensor([1.5, 2.0, 1.0], dtype=torch.float32, device=device)
        .view(3, 1, 1)
        .expand_as(got)
    )
    if not torch.equal(got, expect):
        raise AssertionError(
            "K2 interleaved page past 4 GiB aliased or collapsed: "
            f"row means {got.mean(dim=(1, 2)).tolist()}"
        )


def _k2_one_live_token(m, width, live_col):
    """Q/K zero, V one, every index -1 except ``live_col``."""
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    q = torch.zeros(m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    k_cache = torch.zeros(
        1, 16, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    v_cache = torch.ones_like(k_cache)
    page_table = torch.zeros(1, 1, dtype=dtypes.i32, device=device)
    indices = torch.full((m, width), -1, dtype=dtypes.i32, device=device)
    indices[:, live_col] = 0
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    return qsa_k2(q, k_cache, v_cache, indices, page_table, token_to_req)


def test_k2_empty_first_tile_keeps_later_token():
    """A masked tile ahead of the only live token must not zero the output.

    The live token's score is zero and its V is one, so the output is one.
    These are the review's columns: 16 inside M=1/W=2048, and 64 inside
    M=512/W=128.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    cases = ((1, 2048, 16), (512, 128, 64))
    for m, width, live_col in cases:
        out = _k2_one_live_token(m, width, live_col).float()
        if not torch.allclose(out, torch.ones_like(out)):
            raise AssertionError(
                f"K2 empty-to-valid M={m} W={width} column {live_col} "
                f"mean {out.mean().item()} expected 1"
            )


def test_k2_default_out_ignores_query_strides():
    """Default output matches the oracle for both review layouts.

    Shape ``[2, 24, 256]``. The first query is contiguous, strides
    ``(6144, 256, 1)``. The second holds the same values at strides
    ``(6144, 1, 24)``. The plan cache keeps the first compile.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    m, seq_len, page_size, width = 2, 64, 16, 8
    torch.manual_seed(0)
    q_contig = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    if q_contig.stride() != (6144, 256, 1):
        raise AssertionError(f"contiguous strides {q_contig.stride()}")
    q_heads_last = q_contig.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if q_heads_last.stride() != (6144, 1, 24):
        raise AssertionError(f"transposed strides {q_heads_last.stride()}")
    if not torch.equal(q_contig, q_heads_last):
        raise AssertionError("the two query layouts do not hold the same values")
    k = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    v = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    indices = torch.randint(0, seq_len, (m, width), dtype=dtypes.i32, device=device)
    indices[:, -1] = -1
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(2)
    k_cache, kv_table = pack_paged_cache(k, page_size, generator=gen)
    v_cache, _kv_table_v = pack_paged_cache(v, page_size, physical=kv_table[0])
    ref = qsa_sparse_gqa(q_contig, k, v, indices)
    for name, q in (("contiguous", q_contig), ("heads-last", q_heads_last)):
        out = qsa_k2(q, k_cache, v_cache, indices, kv_table, token_to_req)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"flydsl K2 default out vs oracle ({name})",
        )
        if err != 0:
            raise AssertionError(
                f"K2 default out diverged for {name} strides {tuple(q.stride())} "
                f"(err={err})"
            )


def test_k2_empty_cache_or_table_returns_zeros():
    """No pages and a nonempty index list returns zeros without launching.

    The cache geometry has zero physical pages. The table geometry has
    zero logical pages. Either one used to clamp the index to 0 and load.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    from aiter.ops.flydsl.kernels.qsa import k2 as k2_kernel

    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    q = torch.randn(1, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    indices = torch.zeros(1, 4, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(1, dtype=dtypes.i32, device=device)
    pages = torch.zeros(
        1, 16, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    empty_cache = torch.empty(
        0, 16, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    one_page = torch.zeros(1, 1, dtype=dtypes.i32, device=device)
    no_pages = torch.empty(1, 0, dtype=dtypes.i32, device=device)
    cases = (
        ("empty cache", empty_cache, empty_cache, one_page),
        ("empty table", pages, pages, no_pages),
    )
    launched = []

    def _record_launch(*_args, **_kwargs):
        launched.append(1)
        raise AssertionError("qsa_k2 launched with no pages")

    original = k2_kernel._run_compiled
    k2_kernel._run_compiled = _record_launch
    try:
        for name, k_cache, v_cache, page_table in cases:
            launched.clear()
            out = qsa_k2(q, k_cache, v_cache, indices, page_table, token_to_req)
            if launched:
                raise AssertionError(f"K2 launched for {name}")
            if not torch.equal(out, torch.zeros_like(out)):
                raise AssertionError(f"K2 {name} output was not zeros")
    finally:
        k2_kernel._run_compiled = original


def test_k2_caller_workspace_is_the_only_partial_buffer():
    """A caller-owned split workspace is the buffer the kernel writes.

    The default launch still allocates that pair itself. A workspace
    launch writes the caller's tensors and does not allocate another
    pair. One split still aliases the output and allocates nothing.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    from aiter.ops.flydsl.kernels.qsa.k2 import _launch_config

    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    m, seq_len, page_size, width = 1, 64, 16, 32
    torch.manual_seed(0)
    q = torch.randn(m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    k = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    v = torch.randn(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    indices = torch.randint(0, seq_len, (m, width), dtype=dtypes.i32, device=device)
    indices[:, -1] = -1
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    k_cache, kv_table = pack_paged_cache(k, page_size)
    v_cache, _ = pack_paged_cache(v, page_size, physical=kv_table[0])
    _block_n, _threads, n_splits = _launch_config(m, width, gqa.kv_heads, gqa.head_dim)
    if n_splits <= 1:
        raise AssertionError(f"workspace shape did not split (n_splits={n_splits})")
    out_shape = (n_splits, m, gqa.n_heads, gqa.head_dim)
    lse_shape = (n_splits, m, gqa.n_heads)
    partial_out = torch.full(out_shape, 7, dtype=torch.float32, device=device)
    partial_lse = torch.full(lse_shape, 3, dtype=torch.float32, device=device)
    out_default = torch.empty_like(q)
    out_ws = torch.empty_like(q)
    ref = qsa_k2(q, k_cache, v_cache, indices, kv_table, token_to_req, out=out_default)
    try:
        got = qsa_k2(
            q,
            k_cache,
            v_cache,
            indices,
            kv_table,
            token_to_req,
            out=out_ws,
            workspace=(partial_out, partial_lse),
        )
    except TypeError as error:
        raise AssertionError(
            "qsa_k2 does not accept a caller-owned split workspace"
        ) from error
    if not torch.equal(got, ref):
        raise AssertionError("workspace launch diverged from the default launch")
    if torch.all(partial_out == 7):
        raise AssertionError("caller partial_out was not written")

    def _peak_bytes(use_workspace):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        qsa_k2(
            q,
            k_cache,
            v_cache,
            indices,
            kv_table,
            token_to_req,
            out=out_default,
            workspace=(partial_out, partial_lse) if use_workspace else None,
        )
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) - start

    one = partial_out.nbytes
    default_bytes = _peak_bytes(False)
    workspace_bytes = _peak_bytes(True)
    if default_bytes < one or default_bytes >= 2 * one:
        raise AssertionError(
            f"default launch allocated {default_bytes} bytes for a {one}-byte partial"
        )
    if workspace_bytes >= one:
        raise AssertionError(
            f"workspace launch allocated another {workspace_bytes} bytes"
        )
    bad = torch.empty(
        (n_splits + 1, *out_shape[1:]), dtype=torch.float32, device=device
    )
    try:
        qsa_k2(
            q,
            k_cache,
            v_cache,
            indices,
            kv_table,
            token_to_req,
            out=out_ws,
            workspace=(bad, partial_lse),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("wrong workspace shape was accepted")

    m1, width1 = 2, 8
    q1 = torch.randn(m1, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    indices1 = torch.randint(0, seq_len, (m1, width1), dtype=dtypes.i32, device=device)
    indices1[:, -1] = -1
    token1 = torch.zeros(m1, dtype=dtypes.i32, device=device)
    splits1 = _launch_config(m1, width1, gqa.kv_heads, gqa.head_dim)[2]
    if splits1 != 1:
        raise AssertionError(f"one-split shape used {splits1} splits")
    out1 = torch.empty_like(q1)
    ref1 = qsa_k2(q1, k_cache, v_cache, indices1, kv_table, token1, out=out1)
    ignored = (
        torch.empty(
            4, m1, gqa.n_heads, gqa.head_dim, dtype=torch.float32, device=device
        ),
        torch.empty(4, m1, gqa.n_heads, dtype=torch.float32, device=device),
    )
    out1b = torch.empty_like(q1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.memory_allocated(device)
    got1 = qsa_k2(
        q1,
        k_cache,
        v_cache,
        indices1,
        kv_table,
        token1,
        out=out1b,
        workspace=ignored,
    )
    torch.cuda.synchronize()
    one_split_bytes = torch.cuda.max_memory_allocated(device) - start
    if not torch.equal(got1, ref1):
        raise AssertionError("one-split workspace launch diverged")
    if one_split_bytes >= ignored[0].nbytes:
        raise AssertionError(f"one-split launch allocated {one_split_bytes} bytes")


def test_family_a_k1_bench_times_expand():
    """``_flydsl_k1_select`` returns block ids and the vendored expand.

    Block ids match ``qsa_k1_block_ids``. Indices match
    ``expand_qsa_block_indices_cuda`` on those ids.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 1, 512, 16
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q = torch.randn(m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_cache, index_table = pack_paged_cache(k_bar.unsqueeze(1), page_size)
    indices, block_ids = _flydsl_k1_select(
        q,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        (4,),
    )
    direct = qsa_k1_block_ids(
        q,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
    )
    if not torch.equal(block_ids, direct):
        raise AssertionError("timed select block ids differ from qsa_k1_block_ids")
    expanded = expand_qsa_block_indices_cuda(
        direct,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        idx.token_budget,
    )
    if indices.shape != expanded.shape or not torch.equal(indices, expanded):
        raise AssertionError("timed select did not expand with the vendored kernel")


def test_k1_k2_sweep_keeps_requested_m_and_fails_on_mismatch():
    """Requested M stays in the decode or prefill list, and a mismatch raises.

    Decode is ``M<=8``. Every larger requested M, including 64, 2048, and
    8192, is prefill. A nonzero K1 set mismatch and a K2 err above the
    unit tolerance (``checkAllclose`` rtol=1e-2 atol=1e-2, so ``err!=0``)
    raise. ``M > L`` is skipped.
    """
    if _skip_m_past_seq(8, 512, "family A K1 decode"):
        raise AssertionError("M <= L was skipped")
    if not _skip_m_past_seq(8192, 512, "family A K1 prefill"):
        raise AssertionError("M > L was not skipped")
    decode, prefill = _k1_k2_sweep_batches([1, 8, 64, 512, 2048, 8192])
    if decode != [1, 8] or prefill != [64, 512, 2048, 8192]:
        raise AssertionError(
            f"requested M was dropped: decode={decode} prefill={prefill}"
        )
    _raise_if_k1_mismatch(0, 1, 32768)
    _raise_if_k2_above_tolerance(0, 512, 8192)
    try:
        _raise_if_k1_mismatch(1, 64, 8192)
    except AssertionError as exc:
        if "set mismatch" not in str(exc):
            raise
    else:
        raise AssertionError("nonzero K1 set mismatch did not fail the sweep")
    try:
        _raise_if_k2_above_tolerance(2, 2048, 8192)
    except AssertionError as exc:
        if "unit tolerance" not in str(exc):
            raise
    else:
        raise AssertionError("K2 err above the unit tolerance did not fail the sweep")


def _flydsl_k1_select(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    token_topk,
    compress_ratio,
    heads,
):
    """Block ids plus the vendored Triton expand live AMD select includes.

    The vLLM and #4882 columns time ``qsa_select_paged_tokens``, which
    expands inside the call. This is that same span for FlyDSL. Set
    equality uses the block ids.
    """
    block_ids = qsa_k1_block_ids(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        heads=heads,
    )
    indices = expand_qsa_block_indices_cuda(
        block_ids,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
    )
    return indices, block_ids


def _pad_index_table(index_table, pad_pages):
    """Zero-fill a one-request block table out to ``pad_pages``.

    ``pad_pages <= 0`` leaves the packed table. A wider table is what vLLM
    allocates for ``max_model_len``: real pages in front, zeros after them.
    A pad narrower than the packed table is an error; the sweep skips that
    row before calling.
    """
    if pad_pages <= 0:
        return index_table
    n_pages = index_table.shape[1]
    if pad_pages < n_pages:
        raise ValueError(
            f"pad_pages={pad_pages} is narrower than the packed table ({n_pages} pages)"
        )
    if pad_pages == n_pages:
        return index_table
    wide = torch.zeros(
        index_table.shape[0],
        pad_pages,
        dtype=index_table.dtype,
        device=index_table.device,
    )
    wide[:, : index_table.shape[1]] = index_table
    return wide


@benchmark()
def bench_qsa_family_a_k1(
    m, seq_len, page_size, dtype, rotate=0, pad_pages=0, cache_layout="packed"
):
    """Family A FlyDSL K1 vs oracle set equality; us vs live AMD and #4882 Triton.

    2d: short rows use fused emit. Long rows use BLOCK_N=32 BF16 MFMA scoring
    into an fp32 score matrix. Selection is decode radix below 32768 columns
    and streaming radix at or above that width. Single-request prefill scores
    16 query rows per workgroup. The FlyDSL column times block ids and then
    the vendored Triton expand, the same expand live AMD select includes.
    The #4882 column times its own select, which also expands. Set equality
    stays on block ids. Same ``rotate`` on every select column. ``pad_pages``
    widens the block table every column sees; ``0`` keeps it packed to the
    context. ``cache_layout="wide"`` gives every column the serving cache
    span. Family A GQA is group 12 / D=256, so #4882 Gluon does not dispatch
    and is not a column here. The layer bench takes the same layout.

    When ``M`` is at most 64 and the table is wider than 20000 columns, two
    more columns replay the select under HIP-graph capture. FlyDSL capture
    selects the one-workgroup radix; the eager FlyDSL column readbacks the
    live width and streams past 32768 columns. Live AMD is the same
    ``qsa_select_paged_tokens`` call as the eager column, captured and
    replayed hot. Both replays use one hot buffer set and do not follow
    ``rotate``. Shorter tables leave both columns out.
    ``--batch 4 16 64 --seq 131072 263424 --page-size 392`` is the long
    decode: 263424 tokens fill 65856 columns, 168 pages of 392. Batches
    above 8 print in the prefill table.
    """
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    index_cache = _apply_cache_layout(index_cache.contiguous(), cache_layout)
    index_table = _pad_index_table(index_table.contiguous(), pad_pages)

    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=FAMILY_A_SCORE_SCALE,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)

    (_indices, block_ids), k1_us = _time(
        _flydsl_k1_select,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
        heads=(4,),
    )
    k1_err = _set_mismatch_ratio(ref_ids, block_ids, ref_scores)

    (_indices, vllm_ids), vllm_us = _time(
        qsa_select_paged_tokens,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
    )
    vllm_err = _set_mismatch_ratio(ref_ids, vllm_ids, ref_scores)

    (_triton_indices, triton_ids), triton_us = _time(
        qsa_4882_select_paged_tokens,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
        backend="triton",
    )
    triton_err = _set_mismatch_ratio(ref_ids, triton_ids, ref_scores)

    flops = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    nbytes = (m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim) * dtype.itemsize
    # Capture takes the one-workgroup radix on a padded decode table.
    # The eager column above readbacks and streams once the live width
    # crosses 32768 columns, so the two FlyDSL times are the two selectors.
    n_columns = int(index_table.shape[1]) * int(page_size)
    graph_select = (
        m <= k1_kernel._DECODE_MAX_ROWS
        and n_columns > k1_kernel._ONE_WORKGROUP_MAX_ROW_WIDTH
    )
    graph_us = None
    graph_err = None
    amd_graph_us = None
    amd_graph_err = None
    if graph_select:
        block_ids_g = torch.empty(m, idx.block_budget, dtype=dtypes.i32, device=device)
        indices_g = torch.empty(m, idx.index_width, dtype=dtypes.i32, device=device)

        def launch():
            qsa_k1_block_ids(
                q_indexer,
                index_cache,
                index_table,
                token_to_req,
                qpos,
                slen,
                out=block_ids_g,
                heads=(4,),
            )
            expand_qsa_block_indices_cuda(
                block_ids_g,
                qpos,
                slen,
                token_to_req,
                idx.compress_ratio,
                idx.token_budget,
                out=indices_g,
            )

        graph_us = _capture_replay_us(launch)
        graph_err = _set_mismatch_ratio(ref_ids, block_ids_g, ref_scores)

        amd_indices = torch.empty(m, idx.index_width, dtype=dtypes.i32, device=device)
        amd_blocks: dict[str, torch.Tensor] = {}

        def amd_launch():
            _out, blocks = qsa_select_paged_tokens(
                q_indexer,
                index_cache,
                index_table,
                token_to_req,
                qpos,
                slen,
                idx.token_budget,
                idx.compress_ratio,
                out=amd_indices,
            )
            amd_blocks["ids"] = blocks

        amd_graph_us = _capture_replay_us(amd_launch)
        amd_graph_err = _set_mismatch_ratio(ref_ids, amd_blocks["ids"], ref_scores)
    ret = {
        "gfx": get_gfx(),
        "n_blocks": n_blocks,
        "n_pages": int(index_table.shape[1]),
        "cache_span_gib": _span_gib(index_cache),
        "flydsl_k1 us": k1_us,
        "flydsl_k1 TFLOPS": flops / k1_us / 1e6,
        "flydsl_k1 TB/s": nbytes / k1_us / 1e6,
        "flydsl_k1 err": k1_err,
        "vllm_amd_select us": vllm_us,
        "vllm_amd_select TFLOPS": flops / vllm_us / 1e6,
        "vllm_amd_select TB/s": nbytes / vllm_us / 1e6,
        "vllm_amd_select err": vllm_err,
        "4882_triton_select us": triton_us,
        "4882_triton_select TFLOPS": flops / triton_us / 1e6,
        "4882_triton_select TB/s": nbytes / triton_us / 1e6,
        "4882_triton_select err": triton_err,
    }
    if graph_us is not None:
        ret["flydsl_k1_graph us"] = graph_us
        ret["flydsl_k1_graph TFLOPS"] = flops / graph_us / 1e6
        ret["flydsl_k1_graph TB/s"] = nbytes / graph_us / 1e6
        ret["flydsl_k1_graph err"] = graph_err
        ret["vllm_amd_select_graph us"] = amd_graph_us
        ret["vllm_amd_select_graph TFLOPS"] = flops / amd_graph_us / 1e6
        ret["vllm_amd_select_graph TB/s"] = nbytes / amd_graph_us / 1e6
        ret["vllm_amd_select_graph err"] = amd_graph_err
    return ret


@benchmark()
def bench_qsa_family_a_k2(
    m, seq_len, page_size, dtype, rotate=0, cache_layout="packed"
):
    """Family A FlyDSL K2 vs oracle GQA; us vs live AMD and #4882 Triton.

    3d: live-AMD-shaped BLOCK_N/threads/split policy, tiled MFMA QK/PV,
    log2 online softmax, direct output at one split, and a two-wave merge.
    Expand and sigmoid stay unfused. The FlyDSL column passes the packed
    live prefix as ``n_sel`` so the split grid follows filled columns.
    Same ``rotate`` on every GQA column.
    ``cache_layout="wide"`` gives K and V the serving cache span. K2 reads
    V with K's strides, so the two views share one page stride.
    """
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    _index_cache, _index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    k_cache = _apply_cache_layout(k_cache.contiguous(), cache_layout)
    v_cache = _apply_cache_layout(v_cache.contiguous(), cache_layout)
    if k_cache.stride() != v_cache.stride():
        raise RuntimeError(
            f"K and V strides diverged: {k_cache.stride()} vs {v_cache.stride()}"
        )
    kv_table = kv_table.contiguous()
    ref = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=FAMILY_A_SCORE_SCALE,
        out_dtype=dtypes.fp32,
    )
    indices = ref.indices.contiguous()

    out, k2_us = _time(
        qsa_k2,
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        rotate=rotate,
        n_sel=_k2_live_n_sel(m, seq_len, idx.token_budget, idx.compress_ratio),
    )
    k2_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="flydsl K2 vs oracle GQA",
    )

    vllm_out, vllm_us = _time(
        qsa_sparse_paged_attention,
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        rotate=rotate,
    )
    vllm_err = checkAllclose(
        ref.output,
        vllm_out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="vllm_amd GQA vs oracle",
    )

    t4882_out, t4882_us = _time(
        qsa_sparse_paged_gqa,
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        rotate=rotate,
        backend="triton",
    )
    t4882_err = checkAllclose(
        ref.output,
        t4882_out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="4882 Triton GQA vs oracle",
    )

    w_alloc = indices.shape[1]
    w = _selected_width(indices)
    flops = 4 * m * gqa.n_heads * gqa.head_dim * w
    nbytes = (
        m * gqa.n_heads * gqa.head_dim * 2 + 2 * w * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    return {
        "gfx": get_gfx(),
        "n_blocks": n_blocks,
        "width": w_alloc,
        "valid%": 100.0 * w / w_alloc,
        "cache_span_gib": _span_gib(k_cache),
        "flydsl_k2 us": k2_us,
        "flydsl_k2 TFLOPS": flops / k2_us / 1e6,
        "flydsl_k2 TB/s": nbytes / k2_us / 1e6,
        "flydsl_k2 err": k2_err,
        "vllm_amd_gqa us": vllm_us,
        "vllm_amd_gqa TFLOPS": flops / vllm_us / 1e6,
        "vllm_amd_gqa TB/s": nbytes / vllm_us / 1e6,
        "vllm_amd_gqa err": vllm_err,
        "4882_triton_gqa us": t4882_us,
        "4882_triton_gqa TFLOPS": flops / t4882_us / 1e6,
        "4882_triton_gqa TB/s": nbytes / t4882_us / 1e6,
        "4882_triton_gqa err": t4882_err,
    }


def test_k1_family_b_set_equality_short_decode():
    """FlyDSL family B K1 (H=4) block-id sets match the oracle on short decode."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_B_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 4, 512, 16
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=idx.head_dim**-0.5,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    _assert_k1_block_ids(
        ref_ids, got, "family B K1 block-id set diverged from the oracle"
    )


def test_k1_family_b_set_equality_short_decode_h8():
    """FlyDSL family B K1 (H=8) block-id sets match the oracle on short decode."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_B_INDEXER_H8
    device = torch.device("cuda")
    m, seq_len, page_size = 4, 512, 16
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=idx.head_dim**-0.5,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    _assert_k1_block_ids(
        ref_ids, got, "family B K1 H=8 block-id set diverged from the oracle"
    )


def test_k1_family_b_set_equality_two_tiles():
    """Family B K1 H=4 still matches the oracle when n_blocks exceeds one tile."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_B_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 2, 4096, 16
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks > 512
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=idx.head_dim**-0.5,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    _assert_k1_block_ids(
        ref_ids, got, "family B K1 two-tile block-id set diverged from the oracle"
    )


def test_k1_family_b_set_equality_two_tiles_h8():
    """Family B K1 H=8 still matches the oracle when n_blocks exceeds one tile."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_B_INDEXER_H8
    device = torch.device("cuda")
    m, seq_len, page_size = 2, 4096, 16
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks > 512
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=idx.head_dim**-0.5,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    _assert_k1_block_ids(
        ref_ids, got, "family B K1 H=8 two-tile block-id set diverged from the oracle"
    )


def test_k1_family_b_set_equality_published_indexer_point():
    """#4882 published indexer point: M=32, H=4, D=128, page_size=8, n_blocks=512.

    ``pages=512`` is 512 compressed keys packed at ``page_size=8`` (64 pages).
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_B_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 32, 2048, 8
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks == 512
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device
    )
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtypes.bf16, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    assert index_cache.shape[0] == 64
    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=idx.head_dim**-0.5,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_k1_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    _assert_k1_block_ids(
        ref_ids,
        got,
        "family B K1 published-indexer block-id set diverged from the oracle",
    )


@benchmark()
def bench_qsa_family_b_k1(
    m,
    seq_len,
    page_size,
    dtype,
    index_heads,
    rotate=0,
    pad_pages=0,
    cache_layout="packed",
):
    """Family B FlyDSL K1 vs oracle set equality; us vs #4882 select.

    2e/2f: emit on ``n_blocks <= 512``. Longer rows use family A's MFMA
    scorer plus radix (``H=8`` is a second compile). ``H`` 4 and 8 emit
    share one kernel. Separate table from family A. Expand is not fused.
    ``pad_pages`` widens the block table every column sees; ``0`` keeps it
    packed to the context. ``cache_layout="wide"`` gives the indexer cache
    the serving span.
    """
    idx = _family_b_indexer(index_heads)
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_k = k_bar.unsqueeze(1)
    gen_i = torch.Generator(device=device)
    gen_i.manual_seed(1)
    index_cache, index_table = pack_paged_cache(index_k, page_size, generator=gen_i)
    index_cache = _apply_cache_layout(index_cache.contiguous(), cache_layout)
    index_table = _pad_index_table(index_table.contiguous(), pad_pages)
    score_scale = idx.head_dim**-0.5

    ref_scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=score_scale,
    )
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)

    block_ids, k1_us = _time(
        qsa_k1_block_ids,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        rotate=rotate,
        score_scale=score_scale,
    )
    k1_err = _set_mismatch_ratio(ref_ids, block_ids, ref_scores)

    (_indices, triton_ids), triton_us = _time(
        qsa_4882_select_paged_tokens,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
        backend="triton",
    )
    triton_err = _set_mismatch_ratio(ref_ids, triton_ids, ref_scores)

    flops = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    nbytes = (m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim) * dtype.itemsize
    ret = {
        "gfx": get_gfx(),
        "index_heads": idx.n_heads,
        "n_blocks": n_blocks,
        "n_pages": int(index_table.shape[1]),
        "cache_span_gib": _span_gib(index_cache),
        "flydsl_k1 us": k1_us,
        "flydsl_k1 TFLOPS": flops / k1_us / 1e6,
        "flydsl_k1 TB/s": nbytes / k1_us / 1e6,
        "flydsl_k1 err": k1_err,
        "4882_triton_select us": triton_us,
        "4882_triton_select TFLOPS": flops / triton_us / 1e6,
        "4882_triton_select TB/s": nbytes / triton_us / 1e6,
        "4882_triton_select err": triton_err,
    }
    # Gluon indexer dispatches on family B H in {4, 8}, D=128; skip off gfx950.
    if get_gfx() == "gfx950" and gluon_qsa_available():

        (_g_indices, gluon_ids), gluon_us = _time(
            qsa_4882_select_paged_tokens,
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
            rotate=rotate,
            backend="gluon",
        )
        ret["4882_gluon_select us"] = gluon_us
        ret["4882_gluon_select TFLOPS"] = flops / gluon_us / 1e6
        ret["4882_gluon_select TB/s"] = nbytes / gluon_us / 1e6
        ret["4882_gluon_select err"] = _set_mismatch_ratio(
            ref_ids, gluon_ids, ref_scores
        )
    return ret


@benchmark()
def bench_qsa_family_a_vllm_amd(m, seq_len, page_size, dtype, rotate=0):
    """Live AMD path (vLLM Triton MQA + HIP top-k + Triton GQA) vs the oracle.

    Indexer chain and sparse GQA are timed separately. Oracle is not timed.
    Same ``rotate`` on select and GQA.
    """
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)

    index_cache, index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    index_cache = index_cache.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    index_table = index_table.contiguous()
    kv_table = kv_table.contiguous()

    ref = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=FAMILY_A_SCORE_SCALE,
        out_dtype=dtypes.fp32,
    )

    (indices, block_ids), select_us = _time(
        qsa_select_paged_tokens,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
    )
    block_err = _set_mismatch_ratio(ref.block_ids, block_ids)
    index_err = _set_mismatch_ratio(ref.indices, indices)

    out, gqa_us = _time(
        qsa_sparse_paged_attention,
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        rotate=rotate,
    )
    gqa_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="vllm_amd GQA vs oracle",
    )

    w = _selected_width(indices)
    flops_select = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    flops_gqa = 4 * m * gqa.n_heads * gqa.head_dim * w
    bytes_select = (
        m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim
    ) * dtype.itemsize
    bytes_gqa = (
        m * gqa.n_heads * gqa.head_dim * 2 + 2 * seq_len * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    return {
        "gfx": get_gfx(),
        "vllm_pin": VLLM_AMD_QSA_PIN,
        "n_blocks": n_blocks,
        "valid%": 100.0 * w / idx.index_width,
        "vllm_amd_select us": select_us,
        "vllm_amd_select TFLOPS": flops_select / select_us / 1e6,
        "vllm_amd_select TB/s": bytes_select / select_us / 1e6,
        "vllm_amd_select err": max(block_err, index_err),
        "vllm_amd_gqa us": gqa_us,
        "vllm_amd_gqa TFLOPS": flops_gqa / gqa_us / 1e6,
        "vllm_amd_gqa TB/s": bytes_gqa / gqa_us / 1e6,
        "vllm_amd_gqa err": gqa_err,
    }


@benchmark()
def bench_qsa_family_a_4882_triton(m, seq_len, page_size, dtype, rotate=0):
    """#4882 portable Triton QSA vs the oracle. Gluon is not launched.

    Indexer chain and sparse GQA are timed separately. Oracle is not timed.
    Same ``rotate`` on select and GQA.
    """
    idx = FAMILY_A_INDEXER
    gqa = FAMILY_A_GQA
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device).contiguous()
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)

    index_cache, index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    index_cache = index_cache.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    index_table = index_table.contiguous()
    kv_table = kv_table.contiguous()

    ref = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=FAMILY_A_SCORE_SCALE,
        out_dtype=dtypes.fp32,
    )

    (indices, block_ids), select_us = _time(
        qsa_4882_select_paged_tokens,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
        backend="triton",
    )
    block_err = _set_mismatch_ratio(ref.block_ids, block_ids)
    index_err = _set_mismatch_ratio(ref.indices, indices)

    out, gqa_us = _time(
        qsa_sparse_paged_gqa,
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        rotate=rotate,
        backend="triton",
    )
    gqa_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="4882 Triton GQA vs oracle",
    )

    w = _selected_width(indices)
    flops_select = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    flops_gqa = 4 * m * gqa.n_heads * gqa.head_dim * w
    bytes_select = (
        m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim
    ) * dtype.itemsize
    bytes_gqa = (
        m * gqa.n_heads * gqa.head_dim * 2 + 2 * seq_len * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    return {
        "gfx": get_gfx(),
        "aiter_4882_pin": AITER_4882_QSA_PIN,
        "n_blocks": n_blocks,
        "valid%": 100.0 * w / idx.index_width,
        "4882_triton_select us": select_us,
        "4882_triton_select TFLOPS": flops_select / select_us / 1e6,
        "4882_triton_select TB/s": bytes_select / select_us / 1e6,
        "4882_triton_select err": max(block_err, index_err),
        "4882_triton_gqa us": gqa_us,
        "4882_triton_gqa TFLOPS": flops_gqa / gqa_us / 1e6,
        "4882_triton_gqa TB/s": bytes_gqa / gqa_us / 1e6,
        "4882_triton_gqa err": gqa_err,
    }


def _family_b_indexer(index_heads):
    if index_heads == FAMILY_B_INDEXER.n_heads:
        return FAMILY_B_INDEXER
    if index_heads == FAMILY_B_INDEXER_H8.n_heads:
        return FAMILY_B_INDEXER_H8
    raise ValueError(f"family B indexer heads must be 4 or 8, got {index_heads}")


def _bench_qsa_family_b_4882(
    m, seq_len, page_size, dtype, index_heads, backend, rotate=0
):
    """Family B #4882 vs oracle. ``backend`` is ``triton`` or ``gluon``."""
    idx = _family_b_indexer(index_heads)
    gqa = FAMILY_B_GQA
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device).contiguous()
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)

    index_cache, index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    index_cache = index_cache.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    index_table = index_table.contiguous()
    kv_table = kv_table.contiguous()

    ref = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=idx.head_dim**-0.5,
        out_dtype=dtypes.fp32,
    )

    (indices, block_ids), select_us = _time(
        qsa_4882_select_paged_tokens,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
        rotate=rotate,
        backend=backend,
    )
    block_err = _set_mismatch_ratio(ref.block_ids, block_ids)
    index_err = _set_mismatch_ratio(ref.indices, indices)
    if indices.shape[1] != idx.index_width:
        raise AssertionError(
            f"family B selection width {indices.shape[1]} != {idx.index_width}"
        )

    gqa_tol = 2e-2 if backend == "gluon" else 1e-2
    out, gqa_us = _time(
        qsa_sparse_paged_gqa,
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        rotate=rotate,
        backend=backend,
    )
    gqa_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=gqa_tol,
        atol=gqa_tol,
        msg=f"4882 {backend} GQA vs oracle",
    )

    tag = backend
    w = _selected_width(indices)
    flops_select = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    flops_gqa = 4 * m * gqa.n_heads * gqa.head_dim * w
    bytes_select = (
        m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim
    ) * dtype.itemsize
    bytes_gqa = (
        m * gqa.n_heads * gqa.head_dim * 2 + 2 * seq_len * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    return {
        "gfx": get_gfx(),
        "aiter_4882_pin": AITER_4882_QSA_PIN,
        "n_blocks": n_blocks,
        "valid%": 100.0 * w / idx.index_width,
        f"4882_{tag}_select us": select_us,
        f"4882_{tag}_select TFLOPS": flops_select / select_us / 1e6,
        f"4882_{tag}_select TB/s": bytes_select / select_us / 1e6,
        f"4882_{tag}_select err": max(block_err, index_err),
        f"4882_{tag}_gqa us": gqa_us,
        f"4882_{tag}_gqa TFLOPS": flops_gqa / gqa_us / 1e6,
        f"4882_{tag}_gqa TB/s": bytes_gqa / gqa_us / 1e6,
        f"4882_{tag}_gqa err": gqa_err,
    }


@benchmark()
def bench_qsa_family_b_4882_triton(m, seq_len, page_size, dtype, index_heads, rotate=0):
    """#4882 portable Triton QSA vs the oracle on family B shapes.

    Separate table from family A Triton and from family B Gluon. Oracle is
    not timed.
    """
    return _bench_qsa_family_b_4882(
        m, seq_len, page_size, dtype, index_heads, backend="triton", rotate=rotate
    )


@benchmark()
def bench_qsa_family_b_4882_gluon(m, seq_len, page_size, dtype, index_heads, rotate=0):
    """#4882 gfx950 Gluon QSA vs the oracle on family B (Gluon-validated) shapes.

    Family A GQA (group 12 / D=256) is not launched here. Oracle is not timed.
    """
    return _bench_qsa_family_b_4882(
        m, seq_len, page_size, dtype, index_heads, backend="gluon", rotate=rotate
    )


def _policy_args(m, hq, d_gqa, n_columns, page_size, n_heads, d_idx, kv_heads=2):
    """Host-only tensors for the auto-backend predicate. Nothing is launched."""
    n_pages = n_columns // page_size
    q_indexer = torch.zeros(m, n_heads, d_idx, dtype=torch.bfloat16)
    q_gqa = torch.zeros(m, hq, d_gqa, dtype=torch.bfloat16)
    index_cache = torch.zeros(n_pages, page_size, 1, d_idx, dtype=torch.bfloat16)
    index_table = torch.zeros(1, n_pages, dtype=torch.int32)
    k_cache = torch.zeros(n_pages, page_size, kv_heads, d_gqa, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    kv_table = torch.zeros(1, n_pages, dtype=torch.int32)
    indices = torch.zeros(m, 2051, dtype=torch.int32)
    return (
        q_indexer,
        index_cache,
        index_table,
        q_gqa,
        k_cache,
        v_cache,
        kv_table,
        indices,
    )


def test_qsa_backend_default_is_auto():
    """The opt-in defaults to auto, and unknown names are rejected."""
    assert normalize_qsa_backend(None) == "auto"
    assert normalize_qsa_backend("TRITON") == "triton"
    assert normalize_qsa_backend("FlyDSL") == "flydsl"
    assert normalize_qsa_backend("auto") == "auto"
    try:
        normalize_qsa_backend("gluon")
    except ValueError:
        return
    raise AssertionError("gluon is not a qsa_layer backend")


def test_qsa_auto_admits_only_measured_pairs():
    """auto admits the swept (GQA query, indexer heads) pairs and no others.

    Loosened, auto serves an untuned shape at whatever speed the K2 band
    table happens to give; narrowed, it drops a measured shape back to
    Triton. Neither is a wrong result. K2 serves any structurally valid
    shape, so the table in ``qsa.py`` is the only thing doing the rejecting.
    """
    page = 16
    n_columns = 128
    for hq, d_gqa, kv_heads, heads in (
        (24, 256, 2, 4),
        (24, 256, 2, 8),
        (12, 256, 1, 4),
        (12, 256, 1, 8),
        (6, 256, 1, 4),
        (6, 256, 1, 8),
        (3, 256, 1, 4),
        (3, 256, 1, 8),
        (10, 128, 2, 4),
        (10, 128, 2, 8),
    ):
        swept = _policy_args(1, hq, d_gqa, n_columns, page, heads, 128, kv_heads)
        assert qsa_auto_uses_flydsl(*swept) is True
    # An untuned GQA query stays on Triton however it is indexed.
    untuned = _policy_args(1, 16, 128, n_columns, page, 4, 128)
    assert qsa_auto_uses_flydsl(*untuned) is False


def test_qsa_auto_rejects_a_selection_k1_cannot_build():
    """K1 writes 512 block ids and divides positions by 4.

    ``token_topk=1024`` at ratio 4 asks expand for 256 columns.
    ``token_topk=4096`` at ratio 8 still has 512 columns, so expand would
    accept K1's ratio-4 ids. auto stays on Triton. Explicit flydsl raises.
    """
    measured = _policy_args(1, 24, 256, 128, 16, 4, 128)
    if not qsa_auto_uses_flydsl(*measured, token_topk=2048, compress_ratio=4):
        raise AssertionError("the K1 selection contract was rejected")
    rows = torch.zeros(1, dtype=torch.int32)
    for token_topk, compress_ratio in ((1024, 4), (4096, 8)):
        if qsa_auto_uses_flydsl(
            *measured, token_topk=token_topk, compress_ratio=compress_ratio
        ):
            raise AssertionError(
                f"auto accepted token_topk={token_topk} compress_ratio={compress_ratio}"
            )
        (
            q_indexer,
            index_cache,
            index_table,
            q_gqa,
            k_cache,
            v_cache,
            kv_table,
            indices,
        ) = measured
        try:
            qsa_layer(
                q_indexer,
                index_cache,
                index_table,
                q_gqa,
                k_cache,
                v_cache,
                kv_table,
                rows,
                rows,
                torch.ones(1, dtype=torch.int32),
                indices=indices,
                token_topk=token_topk,
                compress_ratio=compress_ratio,
                backend="flydsl",
            )
        except ValueError as exc:
            if "FlyDSL K1 selects" not in str(exc):
                raise
        else:
            raise AssertionError(
                f"flydsl accepted token_topk={token_topk} compress_ratio={compress_ratio}"
            )


def test_qsa_layer_triton_threads_scales():
    """Triton must apply the same multipliers FlyDSL does.

    MQA divides, so a layer ``score_scale`` of 0.5 arrives as the divisor 2.
    ``softmax_scale=0`` stays zero instead of falling back to ``D**-0.5``.
    """
    import aiter.ops.triton.attention.qsa_vllm_amd as amd

    seen = {}

    def select(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
        out=None,
        score_scale=None,
    ):
        seen["divisor"] = score_scale
        rows = q.shape[0]
        width = token_topk + compress_ratio - 1
        blocks = token_topk // compress_ratio
        return (
            torch.zeros(rows, width, dtype=torch.int32),
            torch.zeros(rows, blocks, dtype=torch.int32),
        )

    def attend(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
        softmax_scale=None,
    ):
        seen["softmax"] = softmax_scale
        return q

    measured = _policy_args(1, 24, 256, 128, 16, 4, 128)
    rows = torch.zeros(1, dtype=torch.int32)
    old_select = amd.qsa_select_paged_tokens
    old_attend = amd.qsa_sparse_paged_attention
    amd.qsa_select_paged_tokens = select
    amd.qsa_sparse_paged_attention = attend
    try:
        (
            q_indexer,
            index_cache,
            index_table,
            q_gqa,
            k_cache,
            v_cache,
            kv_table,
            indices,
        ) = measured

        def run(score_scale, softmax_scale):
            seen.clear()
            qsa_layer(
                q_indexer,
                index_cache,
                index_table,
                q_gqa,
                k_cache,
                v_cache,
                kv_table,
                rows,
                rows,
                torch.ones(1, dtype=torch.int32),
                indices=indices,
                score_scale=score_scale,
                softmax_scale=softmax_scale,
                backend="triton",
            )

        run(0.5, 0.0)
        if seen["divisor"] != 2.0 or seen["softmax"] != 0.0:
            raise AssertionError(f"scales arrived as {seen}")
        run(None, None)
        if seen["divisor"] is not None or seen["softmax"] is not None:
            raise AssertionError(f"defaults arrived as {seen}")
        run(0.0, None)
        if seen["divisor"] != float("inf"):
            raise AssertionError(f"zero indexer scale arrived as {seen['divisor']}")
        run(-0.25, None)
        if seen["divisor"] != -4.0:
            raise AssertionError(f"negative indexer scale arrived as {seen['divisor']}")
    finally:
        amd.qsa_select_paged_tokens = old_select
        amd.qsa_sparse_paged_attention = old_attend


def test_qsa_metadata_strided_view_matches_packed():
    """Request and position vectors are loaded as ``ptr + index``.

    A view such as ``base[::2]`` has to score the same request as its
    packed copy. A non-int32 vector is rejected before launch.
    """
    from aiter.ops.triton.attention.qsa_vllm_amd import qsa_mqa_paged

    q = torch.zeros(2, 1, 16, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 16, dtype=torch.bfloat16)
    table = torch.zeros(2, 1, dtype=torch.int32)
    positions = torch.zeros(2, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)
    try:
        qsa_mqa_paged(
            q,
            cache,
            table,
            torch.zeros(2, dtype=torch.int64),
            positions,
            lengths,
            4,
        )
    except ValueError as exc:
        if "token_to_req" not in str(exc):
            raise
    else:
        raise AssertionError("int64 token_to_req was accepted")
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    q = torch.zeros(2, 1, 16, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(1, 16, 1, 16, dtype=torch.bfloat16, device=device)
    table = torch.zeros(2, 1, dtype=torch.int32, device=device)
    requests = torch.tensor([0, 9, 1, 9], dtype=torch.int32, device=device)[::2]
    positions = torch.tensor([7, 0, 3, 0], dtype=torch.int32, device=device)[::2]
    lengths = torch.tensor([20, 1, 8, 1], dtype=torch.int32, device=device)[::2]
    if requests.is_contiguous():
        raise AssertionError("the metadata fixture is already packed")
    _logits, visible = qsa_mqa_paged(q, cache, table, requests, positions, lengths, 4)
    _logits, visible_ref = qsa_mqa_paged(
        q,
        cache,
        table,
        requests.contiguous(),
        positions.contiguous(),
        lengths.contiguous(),
        4,
    )
    if not torch.equal(visible, visible_ref) or not torch.equal(
        visible, torch.tensor([2, 1], dtype=torch.int32, device=device)
    ):
        raise AssertionError(f"strided metadata scored {visible.tolist()}")

    blocks = torch.zeros(2, 1, dtype=torch.int32, device=device)
    expanded = expand_qsa_block_indices_cuda(blocks, positions, lengths, requests, 4, 4)
    expanded_ref = expand_qsa_block_indices_cuda(
        blocks,
        positions.contiguous(),
        lengths.contiguous(),
        requests.contiguous(),
        4,
        4,
    )
    if not torch.equal(expanded, expanded_ref):
        raise AssertionError("strided expand disagreed with the packed metadata")


def test_k2_serves_rejects_shapes_the_builder_rejects():
    """A geometry ``build_qsa_k2_module`` raises on is a reason here.

    ``auto`` selects FlyDSL only when this predicate accepts the tensors.
    An indivisible group, a group past the MFMA M=16 tile, a head_dim that
    is not a multiple of 32, or a zero page size must stay on Triton.
    """

    def tensors(hq, hkv, d, page):
        q = torch.zeros(1, hq, d, dtype=torch.bfloat16)
        k = torch.zeros(1, page, hkv, d, dtype=torch.bfloat16)
        indices = torch.zeros(1, 8, dtype=torch.int32)
        table = torch.zeros(1, 1, dtype=torch.int32)
        return q, k, torch.zeros_like(k), indices, table

    if qsa_k2_serves(*tensors(24, 2, 256, 16)) is not None:
        raise AssertionError("family A GQA was rejected")
    if qsa_k2_serves(*tensors(16, 1, 32, 16)) is not None:
        raise AssertionError("a group of 16 at head_dim 32 was rejected")
    rejected = (
        (24, 5, 256, 16, "do not group"),
        (24, 0, 256, 16, "do not group"),
        (24, 1, 256, 16, "MFMA"),
        (17, 1, 128, 16, "MFMA"),
        (24, 2, 80, 16, "multiple of 32"),
        (24, 2, 256, 0, "page_size"),
    )
    for hq, hkv, d, page, needle in rejected:
        reason = qsa_k2_serves(*tensors(hq, hkv, d, page))
        if reason is None or needle not in reason:
            raise AssertionError(f"Hq={hq} Hkv={hkv} D={d} page={page} -> {reason!r}")
    measured = _policy_args(1, 24, 256, 128, 16, 4, 128)
    q_indexer, index_cache, index_table, q_gqa, _, _, kv_table, indices = measured
    indivisible = torch.zeros(8, 16, 5, 256, dtype=torch.bfloat16)
    if qsa_auto_uses_flydsl(
        q_indexer,
        index_cache,
        index_table,
        q_gqa,
        indivisible,
        indivisible.clone(),
        kv_table,
        indices,
    ):
        raise AssertionError(
            "auto selected FlyDSL for a GQA group the kernel cannot build"
        )


def test_qsa_auto_logs_unmeasured_query_once():
    """auto logs a table miss once per GQA query, then stays on Triton.

    The predicate itself stays quiet: only ``qsa_layer`` falls back.
    A measured query, an M mismatch on a measured query, and an explicit
    backend do not log. The Triton and FlyDSL launches are replaced so
    this stays on the host.
    """
    import logging

    from aiter.ops.flydsl import qsa as qsa_mod

    messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    def _call(args, backend="auto", indexer_rows=None):
        (
            q_indexer,
            index_cache,
            index_table,
            q_gqa,
            k_cache,
            v_cache,
            kv_table,
            indices,
        ) = args
        if indexer_rows is not None:
            q_indexer = torch.zeros(
                indexer_rows,
                *q_indexer.shape[1:],
                dtype=q_indexer.dtype,
            )
        rows = q_gqa.shape[0]
        return qsa_layer(
            q_indexer,
            index_cache,
            index_table,
            q_gqa,
            k_cache,
            v_cache,
            kv_table,
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            indices=indices,
            backend=backend,
        )

    paths = []

    def _triton(*args, **kwargs):
        paths.append("triton")
        return args[3]

    def _flydsl(*args, **kwargs):
        paths.append("flydsl")
        return args[3]

    page = 16
    n_columns = 128
    handler = _Capture()
    old_triton = qsa_mod._qsa_layer_triton
    old_flydsl = qsa_mod._qsa_layer_flydsl
    qsa_mod._log_unmeasured_gqa_query.cache_clear()
    qsa_mod._qsa_layer_triton = _triton
    qsa_mod._qsa_layer_flydsl = _flydsl
    aiter.logger.addHandler(handler)
    try:
        untuned = _policy_args(1, 16, 128, n_columns, page, 4, 128)
        _call(untuned)
        _call(untuned)
        other = _policy_args(1, 32, 64, n_columns, page, 4, 128)
        _call(other)
        if messages != [
            "QSA auto: GQA query shape (16, 128) is not in the measured table; using Triton",
            "QSA auto: GQA query shape (32, 64) is not in the measured table; using Triton",
        ]:
            raise AssertionError(f"table-miss log was {messages}")
        if paths != ["triton", "triton", "triton"]:
            raise AssertionError(f"unmeasured auto took {paths}")

        messages.clear()
        paths.clear()
        swept = _policy_args(1, 24, 256, n_columns, page, 4, 128)
        _call(swept)
        _call(untuned, backend="triton")
        _call(untuned, backend="flydsl")
        _call(swept, indexer_rows=2)
        if messages:
            raise AssertionError(f"measured or explicit backend logged {messages}")
        if paths != ["flydsl", "triton", "flydsl", "triton"]:
            raise AssertionError(f"backend choice was {paths}")
    finally:
        aiter.logger.removeHandler(handler)
        qsa_mod._qsa_layer_triton = old_triton
        qsa_mod._qsa_layer_flydsl = old_flydsl
        qsa_mod._log_unmeasured_gqa_query.cache_clear()


def test_qsa_aot_collector_lists_family_a_launches():
    """The AOT collector lists the family A compiles and nothing else.

    K1 is the page-16 emit, the M=1 long-row scorer, and the M=512
    prefill scorer. M=8 decode uses that same scorer, so it is not a
    second K1 job. K2 is the three bar launches. The JIT wrappers do
    not import this collector.
    """
    import sys

    if "aiter.aot.flydsl.qsa" in sys.modules:
        raise AssertionError("QSA JIT import loaded the AOT collector")
    from aiter.aot.flydsl.common import (
        OpKind,
        _collect_aot_jobs_for,
        _compile_one_config_for,
    )

    try:
        kind = OpKind.QSA
    except AttributeError:
        raise AssertionError("QSA is not an AOT kind") from None
    jobs = _collect_aot_jobs_for(kind)
    if any(job is None for job in jobs):
        raise AssertionError("QSA AOT job list contains None")
    by_name = {job["kernel_name"]: job for job in jobs}
    if len(by_name) != len(jobs):
        raise AssertionError("QSA AOT jobs repeat a kernel_name")
    expect = {
        "qsa_k1_emit_family_a": ("k1", 1, 512),
        "qsa_k1_long_row_family_a_decode": ("k1", 1, 32768),
        "qsa_k1_long_row_family_a_prefill": ("k1", 512, 8192),
        "qsa_k2_family_a_m1_l32768": ("k2", 1, 32768),
        "qsa_k2_family_a_m8_l32768": ("k2", 8, 32768),
        "qsa_k2_family_a_m512_l8192": ("k2", 512, 8192),
    }
    if set(by_name) != set(expect):
        raise AssertionError(f"QSA AOT jobs {sorted(by_name)} != {sorted(expect)}")
    for name, (op, rows, seq_len) in expect.items():
        job = by_name[name]
        if (job["op"], job["m"], job["seq_len"]) != (op, rows, seq_len):
            raise AssertionError(f"{name} collected as {job}")
        if job["page_size"] != 16:
            raise AssertionError(f"{name} page_size is {job['page_size']}")
    decode = by_name["qsa_k1_long_row_family_a_decode"]
    if decode["heads"] != 4 or decode["compress_ratio"] != 4:
        raise AssertionError(f"K1 decode job is not family A: {decode}")
    prefill_k2 = by_name["qsa_k2_family_a_m512_l8192"]
    if (prefill_k2["hq"], prefill_k2["hkv"], prefill_k2["width"]) != (24, 2, 2051):
        raise AssertionError(f"K2 prefill job is not a bar launch: {prefill_k2}")
    compile_one = _compile_one_config_for(kind)
    if not callable(compile_one):
        raise TypeError("QSA AOT has no compile_one_config")


def test_qsa_aot_targets_follow_gpu_archs_without_a_device(monkeypatch):
    """The wheel build lists gfx942 and gfx950 and has no GPU visible.

    Those two archs are the compile targets. A GPU_ARCHS value with no
    QSA arch is an error. With the variable unset and no device, both
    supported archs are still compiled.
    """
    import torch

    from aiter.aot.flydsl.qsa import _qsa_aot_targets

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("GPU_ARCHS", "gfx942;gfx950")
    monkeypatch.delenv("CU_NUM", raising=False)
    assert _qsa_aot_targets() == [("gfx942", 304), ("gfx950", 256)]
    monkeypatch.setenv("GPU_ARCHS", "gfx1250")
    try:
        _qsa_aot_targets()
    except RuntimeError as error:
        if "gfx1250" not in str(error):
            raise AssertionError(str(error)) from error
    else:
        raise AssertionError("GPU_ARCHS with no QSA arch was accepted")
    monkeypatch.delenv("GPU_ARCHS")
    assert _qsa_aot_targets() == [("gfx942", 304), ("gfx950", 256)]


def test_qsa_aot_empty_launch_list_has_no_jobs():
    """An empty QSA launch list collects no jobs.

    The result is ``[]``. ``[None]`` is not the stand-in for no configs.
    One real launch still collects one job.
    """
    from aiter.aot.flydsl.qsa import default_jobs

    jobs = default_jobs(())
    if jobs != []:
        raise AssertionError(f"empty QSA launches collected as {jobs!r}")
    one = default_jobs((("qsa_k1_emit_family_a", "k1", 1, 512),))
    if len(one) != 1 or one[0] is None:
        raise AssertionError(f"one QSA launch collected as {one!r}")
    if one[0]["op"] != "k1" or one[0]["m"] != 1 or one[0]["seq_len"] != 512:
        raise AssertionError(f"one QSA launch collected as {one[0]!r}")


def test_qsa_symbols_export_lazily():
    """K1, K2, and the layer are on ``aiter.ops.flydsl`` without a side import."""
    from aiter.ops import flydsl

    assert flydsl.qsa_k1_block_ids is qsa_k1_block_ids
    assert flydsl.qsa_k2 is qsa_k2
    assert flydsl.qsa_layer is qsa_layer
    assert flydsl.normalize_qsa_backend is normalize_qsa_backend


def _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout="packed"):
    device = torch.device("cuda")
    n_blocks = seq_len // idx.compress_ratio
    torch.manual_seed(0)
    q_indexer = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    qpos = _query_positions(m, seq_len, device).contiguous()
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    index_cache, index_table, k_cache, v_cache, kv_table = _pack_family_a(
        k_bar, k, v, page_size, device
    )
    index_cache = _apply_cache_layout(index_cache.contiguous(), cache_layout)
    index_table = index_table.contiguous()
    k_cache = _apply_cache_layout(k_cache.contiguous(), cache_layout)
    v_cache = _apply_cache_layout(v_cache.contiguous(), cache_layout)
    if k_cache.stride() != v_cache.stride():
        raise RuntimeError(
            f"K and V strides diverged: {k_cache.stride()} vs {v_cache.stride()}"
        )
    kv_table = kv_table.contiguous()
    ref = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=idx.head_dim**-0.5,
        out_dtype=dtypes.fp32,
    )
    return {
        "n_blocks": n_blocks,
        "q_indexer": q_indexer,
        "index_cache": index_cache,
        "index_table": index_table,
        "q_gqa": q_gqa,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_table": kv_table,
        "token_to_req": token_to_req,
        "qpos": qpos,
        "slen": slen,
        "ref": ref,
    }


def _layer_args(case):
    return (
        case["q_indexer"],
        case["index_cache"],
        case["index_table"],
        case["q_gqa"],
        case["k_cache"],
        case["v_cache"],
        case["kv_table"],
        case["token_to_req"],
        case["qpos"],
        case["slen"],
    )


def _layer_kwargs(idx, backend, **extra):
    return {
        "token_topk": idx.token_budget,
        "compress_ratio": idx.compress_ratio,
        "score_scale": idx.head_dim**-0.5,
        "backend": backend,
        **extra,
    }


def test_qsa_layer_family_a_matches_oracle():
    """FlyDSL, the default auto path, and named Triton all match the oracle.

    This family A shape is a measured pair, so auto selects FlyDSL.
    """
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx, gqa = FAMILY_A_INDEXER, FAMILY_A_GQA
    case = _prepare_qsa_layer(idx, gqa, 2, 128, 16, dtypes.bf16)
    ref = case["ref"].output
    fly = qsa_layer(*_layer_args(case), **_layer_kwargs(idx, "flydsl"))
    err = checkAllclose(
        ref, fly.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="flydsl layer vs oracle"
    )
    if err != 0:
        raise AssertionError(f"FlyDSL QSA layer diverged from the oracle (err={err})")
    default = qsa_layer(*_layer_args(case), **_layer_kwargs(idx, None))
    named = qsa_layer(*_layer_args(case), **_layer_kwargs(idx, "triton"))
    for label, out in (("default", default), ("triton", named)):
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{label} layer vs oracle",
        )
        if err != 0:
            raise AssertionError(f"{label} QSA layer diverged (err={err})")


def test_qsa_layer_family_b_matches_oracle():
    """Family B FlyDSL layer matches the oracle (group 5, D=128)."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx, gqa = FAMILY_B_INDEXER, FAMILY_B_GQA
    case = _prepare_qsa_layer(idx, gqa, 2, 128, 16, dtypes.bf16)
    out = qsa_layer(*_layer_args(case), **_layer_kwargs(idx, "flydsl"))
    err = checkAllclose(
        case["ref"].output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="family B flydsl layer vs oracle",
    )
    if err != 0:
        raise AssertionError(f"family B QSA layer diverged (err={err})")


def _capture_replay_us(fn) -> float:
    """Warm up, capture one HIP graph, and time ``replay``. Replay stays hot."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            graph.capture_begin()
            fn()
            graph.capture_end()
    except RuntimeError:
        torch.cuda.current_stream().wait_stream(stream)
        raise
    torch.cuda.current_stream().wait_stream(stream)
    _ignored, us = run_perftest(graph.replay, num_rotate_args=1)
    graph.replay()
    torch.cuda.synchronize()
    return us


def test_qsa_layer_decode_graph_replays():
    """Decode capture of K1 + expand + K2 replays and still matches the oracle."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx, gqa = FAMILY_A_INDEXER, FAMILY_A_GQA
    # n_blocks=1024 forces the long-row score buffer, not the emit fast path.
    case = _prepare_qsa_layer(idx, gqa, 1, 4096, 16, dtypes.bf16)
    device = case["q_gqa"].device
    indices = torch.empty((1, idx.index_width), dtype=dtypes.i32, device=device)
    block_ids = torch.empty((1, idx.block_budget), dtype=dtypes.i32, device=device)
    out = torch.empty_like(case["q_gqa"])

    def launch():
        return qsa_layer(
            *_layer_args(case),
            indices=indices,
            block_ids=block_ids,
            out=out,
            **_layer_kwargs(idx, "flydsl"),
        )

    _capture_replay_us(launch)
    err = checkAllclose(
        case["ref"].output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="flydsl graph replay vs oracle",
    )
    if err != 0:
        raise AssertionError(f"graph replay diverged from the oracle (err={err})")


def _qsa_4882_layer(
    q_indexer,
    index_cache,
    index_table,
    q_gqa,
    k_cache,
    v_cache,
    kv_table,
    token_to_req,
    qpos,
    slen,
    token_topk,
    compress_ratio,
    backend,
):
    indices, _blocks = qsa_4882_select_paged_tokens(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        token_topk,
        compress_ratio,
        backend=backend,
    )
    return qsa_sparse_paged_gqa(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        backend=backend,
    )


def _e2e_counts(case, idx, gqa, dtype):
    indices = case["ref"].indices
    w_alloc = indices.shape[1]
    w = _selected_width(indices)
    m = case["q_gqa"].shape[0]
    flops = (
        2 * m * idx.n_heads * idx.head_dim * case["n_blocks"]
        + 4 * m * gqa.n_heads * gqa.head_dim * w
    )
    nbytes = (
        m * idx.n_heads * idx.head_dim
        + case["n_blocks"] * idx.head_dim
        + m * gqa.n_heads * gqa.head_dim * 2
        + 2 * w * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    return w_alloc, w, flops, nbytes


def _e2e_cells(name, us, err, flops, nbytes):
    return {
        f"{name} us": us,
        f"{name} TFLOPS": flops / us / 1e6,
        f"{name} TB/s": nbytes / us / 1e6,
        f"{name} err": err,
    }


def _time_layer(case, idx, backend, rotate):
    out, us = _time(
        qsa_layer,
        *_layer_args(case),
        rotate=rotate,
        **_layer_kwargs(idx, backend),
    )
    return out, us


@benchmark()
def bench_qsa_family_a_e2e(
    m, seq_len, page_size, dtype, rotate=0, cache_layout="packed"
):
    """One family A QSA layer: FlyDSL K1+expand+K2 vs live AMD and #4882 Triton.

    Expand stays the vendored Triton kernel. Oracle is not timed. Same
    ``rotate`` on every column. HIP graph replay is a separate table.
    ``cache_layout="wide"`` is the serving span on the indexer cache and
    on K and V.
    """
    idx, gqa = FAMILY_A_INDEXER, FAMILY_A_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout)
    ref = case["ref"].output
    w_alloc, w, flops, nbytes = _e2e_counts(case, idx, gqa, dtype)

    fly, fly_us = _time_layer(case, idx, "flydsl", rotate)
    fly_err = checkAllclose(
        ref, fly.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="flydsl e2e vs oracle"
    )
    amd, amd_us = _time_layer(case, idx, "triton", rotate)
    amd_err = checkAllclose(
        ref, amd.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="vllm amd e2e vs oracle"
    )
    t4882, t4882_us = _time(
        _qsa_4882_layer,
        *_layer_args(case),
        idx.token_budget,
        idx.compress_ratio,
        "triton",
        rotate=rotate,
    )
    t4882_err = checkAllclose(
        ref,
        t4882.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="4882 triton e2e vs oracle",
    )
    ret = {
        "gfx": get_gfx(),
        "n_blocks": case["n_blocks"],
        "width": w_alloc,
        "valid%": 100.0 * w / w_alloc,
        "cache_span_gib": _span_gib(case["k_cache"]),
    }
    ret.update(_e2e_cells("flydsl_e2e", fly_us, fly_err, flops, nbytes))
    ret.update(_e2e_cells("vllm_amd_e2e", amd_us, amd_err, flops, nbytes))
    ret.update(_e2e_cells("4882_triton_e2e", t4882_us, t4882_err, flops, nbytes))
    return ret


@benchmark()
def bench_qsa_family_b_e2e(
    m, seq_len, page_size, dtype, index_heads, rotate=0, cache_layout="packed"
):
    """One family B QSA layer vs live AMD, #4882 Triton and, on gfx950, Gluon.

    Separate table from family A. Live AMD is the column ``auto`` decides
    against, so it is the one that governs the gate; the others are parity.
    """
    idx = _family_b_indexer(index_heads)
    gqa = FAMILY_B_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout)
    ref = case["ref"].output
    w_alloc, w, flops, nbytes = _e2e_counts(case, idx, gqa, dtype)
    fly, fly_us = _time_layer(case, idx, "flydsl", rotate)
    fly_err = checkAllclose(
        ref, fly.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="flydsl family B e2e"
    )
    amd, amd_us = _time_layer(case, idx, "triton", rotate)
    amd_err = checkAllclose(
        ref, amd.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="vllm amd family B e2e"
    )
    triton, triton_us = _time(
        _qsa_4882_layer,
        *_layer_args(case),
        idx.token_budget,
        idx.compress_ratio,
        "triton",
        rotate=rotate,
    )
    triton_err = checkAllclose(
        ref,
        triton.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="4882 triton family B e2e",
    )
    ret = {
        "gfx": get_gfx(),
        "index_heads": idx.n_heads,
        "n_blocks": case["n_blocks"],
        "width": w_alloc,
        "valid%": 100.0 * w / w_alloc,
        "cache_span_gib": _span_gib(case["k_cache"]),
    }
    ret.update(_e2e_cells("flydsl_e2e", fly_us, fly_err, flops, nbytes))
    ret.update(_e2e_cells("vllm_amd_e2e", amd_us, amd_err, flops, nbytes))
    ret.update(_e2e_cells("4882_triton_e2e", triton_us, triton_err, flops, nbytes))
    if get_gfx() == "gfx950" and gluon_qsa_available():
        gluon, gluon_us = _time(
            _qsa_4882_layer,
            *_layer_args(case),
            idx.token_budget,
            idx.compress_ratio,
            "gluon",
            rotate=rotate,
        )
        gluon_err = checkAllclose(
            ref,
            gluon.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg="4882 gluon family B e2e",
        )
        ret.update(_e2e_cells("4882_gluon_e2e", gluon_us, gluon_err, flops, nbytes))
    return ret


@benchmark()
def bench_qsa_family_a_e2e_graph(m, seq_len, page_size, dtype, cache_layout="packed"):
    """HIP graph replay of one family A decode layer. Not combined with rotate.

    Each candidate is captured once, then ``replay`` is timed hot. The
    output buffer after replay is the correctness check.
    """
    idx, gqa = FAMILY_A_INDEXER, FAMILY_A_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout)
    ref = case["ref"].output
    _w_alloc, w, flops, nbytes = _e2e_counts(case, idx, gqa, dtype)
    device = case["q_gqa"].device
    width = idx.index_width

    def _replay(backend, out):
        indices = torch.empty((m, width), dtype=dtypes.i32, device=device)
        block_ids = torch.empty((m, idx.block_budget), dtype=dtypes.i32, device=device)

        def launch():
            return qsa_layer(
                *_layer_args(case),
                indices=indices,
                block_ids=block_ids,
                out=out,
                **_layer_kwargs(idx, backend),
            )

        us = _capture_replay_us(launch)
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{backend} graph replay vs oracle",
        )
        return us, err

    fly_us, fly_err = _replay("flydsl", torch.empty_like(case["q_gqa"]))
    amd_us, amd_err = _replay("triton", torch.empty_like(case["q_gqa"]))
    ret = {
        "gfx": get_gfx(),
        "n_blocks": case["n_blocks"],
        "valid%": 100.0 * w / width,
        "cache_span_gib": _span_gib(case["k_cache"]),
    }
    ret.update(_e2e_cells("flydsl_graph", fly_us, fly_err, flops, nbytes))
    ret.update(_e2e_cells("vllm_amd_graph", amd_us, amd_err, flops, nbytes))
    return ret


def _run_unit_cases():
    test_indexer_hand_checked_one_row()
    test_topk_smaller_index_wins_ties()
    test_incomplete_blocks_not_selected()
    test_gqa_matches_dense_on_selected()
    test_family_a_shapes_smoke()
    test_family_b_shape_constants()
    test_paged_roundtrip_tiny()
    test_k1_family_a_set_equality_short_decode()
    test_k1_block_ids_require_packed_prefix()
    test_k1_family_a_set_equality_two_tiles()
    test_k1_family_a_set_equality_wide_stream()
    test_k1_family_a_set_equality_prefill()
    test_k1_prefill_padded_page_table()
    test_k1_decode_rejects_invalid_page_ids()
    test_k1_gfx942_h8_skips_prefill_tile()
    test_family_a_k1_bench_times_expand()
    test_k1_k2_sweep_keeps_requested_m_and_fails_on_mismatch()
    test_qsa_arch_allowlist()
    test_k1_page_past_4gib()
    test_expand_rejects_out_that_cannot_hold_the_stores()
    test_sparse_attention_rejects_out_that_does_not_match_q()
    test_k1_serves_padded_page_and_rejects_misaligned()
    test_k1_padded_page_stride_matches_packed()
    test_k1_wide_padded_page_does_not_alias()
    test_wide_cache_layout_spans_past_4gib()
    test_k1_family_b_set_equality_short_decode()
    test_k1_family_b_set_equality_short_decode_h8()
    test_k1_family_b_set_equality_two_tiles()
    test_k1_family_b_set_equality_two_tiles_h8()
    test_k1_family_b_set_equality_published_indexer_point()
    test_k2_family_a_decode_matches_oracle()
    test_k2_live_prefix_matches_full_width()
    test_k2_family_a_prefill_matches_oracle()
    test_k2_page_past_4gib()
    test_k2_interleaved_kv_view_matches_contiguous()
    test_k2_interleaved_kv_page_past_4gib()
    test_k2_empty_first_tile_keeps_later_token()
    test_k2_default_out_ignores_query_strides()
    test_k2_empty_cache_or_table_returns_zeros()
    test_k2_caller_workspace_is_the_only_partial_buffer()
    test_qsa_backend_default_is_auto()
    test_qsa_auto_admits_only_measured_pairs()
    test_qsa_auto_rejects_a_selection_k1_cannot_build()
    test_qsa_layer_triton_threads_scales()
    test_qsa_metadata_strided_view_matches_packed()
    test_k2_serves_rejects_shapes_the_builder_rejects()
    test_qsa_auto_logs_unmeasured_query_once()
    test_qsa_aot_collector_lists_family_a_launches()
    test_qsa_aot_empty_launch_list_has_no_jobs()
    test_qsa_symbols_export_lazily()
    test_qsa_layer_family_a_matches_oracle()
    test_qsa_layer_family_b_matches_oracle()
    test_qsa_layer_decode_graph_replays()
    aiter.logger.info("QSA oracle + K1 + K2 + layer unit cases passed")


def _k1_k2_sweep_batches(batches):
    """Split requested M into the family A K1/K2 decode and prefill tables.

    Decode is ``M<=8``. Every larger M is prefill, so 64, 2048, and 8192
    are run instead of dropped. Family B is not split this way: its tables
    are emit versus long-L, and every requested M runs in one of them.
    """
    decode = [m for m in batches if m <= 8]
    prefill = [m for m in batches if m > 8]
    return decode, prefill


def _skip_m_past_seq(m, seq_len, where):
    if m <= seq_len:
        return False
    aiter.logger.warning("skip %s M=%s L=%s (M must fit in L)", where, m, seq_len)
    return True


def _skip_pad_narrower_than_context(
    seq_len, page_size, pad_pages, compress_ratio, where
):
    """A pad that cannot hold the context would drop real pages."""
    if pad_pages <= 0:
        return False
    n_blocks = seq_len // compress_ratio
    packed_pages = (n_blocks + page_size - 1) // page_size
    if packed_pages <= pad_pages:
        return False
    aiter.logger.warning(
        "skip %s L=%s page=%s pad_pages=%s (context needs %s pages)",
        where,
        seq_len,
        page_size,
        pad_pages,
        packed_pages,
    )
    return True


def _raise_if_k1_mismatch(err, m, seq_len, where="FlyDSL K1"):
    if err != 0:
        raise AssertionError(f"{where} set mismatch at M={m} L={seq_len} (err={err})")


def _raise_if_k2_above_tolerance(err, m, seq_len):
    if err != 0:
        raise AssertionError(
            f"FlyDSL K2 err={err} at M={m} L={seq_len} is above the unit "
            "tolerance rtol=1e-2 atol=1e-2"
        )


def main():
    _run_unit_cases()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Family A/B QSA sweeps, FlyDSL K1/K2, and the end-to-end layer opt-in",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="activation dtype (family A is BF16)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 8, 512],
        help="flattened query tokens M. Family A K1/K2 report M<=8 as\n"
        "decode and every larger M as prefill.",
    )
    parser.add_argument(
        "-s",
        "--seq",
        type=int,
        nargs="*",
        default=[512, 2048, 8192, 32768],
        help="context length L in tokens (32k default; pass 131072 for 128k).\n"
        "Bar shapes are budget-saturated: decode M in {1,8} at L=32768,\n"
        "prefill M=512 at L=8192. L=512 is a fast smoke row -- only ~25%%\n"
        "of the 2051 selection slots are live there (12.5%% at M=512), so\n"
        "it measures the masked path more than the gather. See valid%%.",
    )
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        nargs="*",
        default=[16],
        help="vLLM-style page size (indexer slots and GQA tokens)",
    )
    parser.add_argument(
        "--pad-pages",
        type=int,
        nargs="*",
        default=[0],
        help="K1 block-table width in pages. 0 keeps the packed table.\n"
        "A positive value zero-fills every K1 column's table out to that\n"
        "many pages, which is how vLLM sizes it for max_model_len. A row\n"
        "whose context already needs more pages is skipped. Examples:\n"
        "--pad-pages 4096 with page size 16 is 65536 columns;\n"
        "--page-size 392 --pad-pages 168 is the Flash-Next TP2 table\n"
        "(65856 columns). Pass 0 and a width to sweep both.",
    )
    parser.add_argument(
        "--cache-layout",
        nargs="*",
        default=["packed"],
        choices=["packed", "wide"],
        help="K and V cache layout. packed is one contiguous page after\n"
        "another and stays under 4 GiB. wide keeps those pages and inserts\n"
        "a page stride so the byte span is just past 4 GiB, which is how\n"
        "vLLM's per-layer view selects the wide K1/K2 compile. A one-page\n"
        "cache gains an unreferenced trailing page so the stride counts.\n"
        "Auto-rotate keeps one copy of a wide cache: it is already larger\n"
        "than L2. Pass both names to sweep them.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        nargs="*",
        default=[0],
        help="run_perftest num_rotate_args (copies of timed tensors).\n"
        "0 = cold cache (default; auto-size copies from L2, matches serving).\n"
        "1 = hot cache, one reused buffer set (older 1047 tables).\n"
        "N>1 = that many copies.\n"
        "Same value on every named backend in a row. Not combined with HIP graphs.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        aiter.logger.warning("no CUDA; skipping family A plumbing sweep")
        return
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("QSA plumbing unsupported on %s; skipping", get_gfx())
        return

    for dtype in args.dtype:
        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            args.batch, args.seq, args.page_size, args.rotate
        ):
            if m > seq_len:
                aiter.logger.warning(
                    "skip m=%s seq_len=%s (M must fit in L)", m, seq_len
                )
                continue
            rows.append(
                bench_qsa_family_a_plumbing(m, seq_len, page_size, dtype, rotate)
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family A plumbing summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            args.batch, args.seq, args.page_size, args.rotate
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_a_vllm_amd(m, seq_len, page_size, dtype, rotate)
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family A vLLM AMD summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            args.batch, args.seq, args.page_size, args.rotate
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_a_4882_triton(m, seq_len, page_size, dtype, rotate)
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family A #4882 Triton summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size, index_heads, rotate in itertools.product(
            args.batch, args.seq, args.page_size, (4, 8), args.rotate
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_4882_triton(
                    m, seq_len, page_size, dtype, index_heads, rotate
                )
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family B #4882 Triton summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        decode_m, prefill_m = _k1_k2_sweep_batches(args.batch)

        rows = []
        for (
            m,
            seq_len,
            page_size,
            rotate,
            pad_pages,
            cache_layout,
        ) in itertools.product(
            decode_m,
            args.seq,
            args.page_size,
            args.rotate,
            args.pad_pages,
            args.cache_layout,
        ):
            if _skip_m_past_seq(m, seq_len, "family A K1 decode"):
                continue
            if _skip_pad_narrower_than_context(
                seq_len,
                page_size,
                pad_pages,
                FAMILY_A_INDEXER.compress_ratio,
                "family A K1 decode",
            ):
                continue
            row = bench_qsa_family_a_k1(
                m,
                seq_len,
                page_size,
                dtype,
                rotate,
                pad_pages,
                cache_layout,
            )
            _raise_if_k1_mismatch(row["flydsl_k1 err"], m, seq_len)
            if "flydsl_k1_graph err" in row:
                _raise_if_k1_mismatch(
                    row["flydsl_k1_graph err"], m, seq_len, where="FlyDSL K1 graph"
                )
                _raise_if_k1_mismatch(
                    row["vllm_amd_select_graph err"],
                    m,
                    seq_len,
                    where="live AMD select graph",
                )
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K1 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate, cache_layout in itertools.product(
            decode_m, args.seq, args.page_size, args.rotate, args.cache_layout
        ):
            if _skip_m_past_seq(m, seq_len, "family A K2 decode"):
                continue
            row = bench_qsa_family_a_k2(
                m, seq_len, page_size, dtype, rotate, cache_layout
            )
            _raise_if_k2_above_tolerance(row["flydsl_k2 err"], m, seq_len)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K2 decode summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for (
            m,
            seq_len,
            page_size,
            rotate,
            pad_pages,
            cache_layout,
        ) in itertools.product(
            prefill_m,
            args.seq,
            args.page_size,
            args.rotate,
            args.pad_pages,
            args.cache_layout,
        ):
            if _skip_m_past_seq(m, seq_len, "family A K1 prefill"):
                continue
            if _skip_pad_narrower_than_context(
                seq_len,
                page_size,
                pad_pages,
                FAMILY_A_INDEXER.compress_ratio,
                "family A K1 prefill",
            ):
                continue
            row = bench_qsa_family_a_k1(
                m,
                seq_len,
                page_size,
                dtype,
                rotate,
                pad_pages,
                cache_layout,
            )
            _raise_if_k1_mismatch(row["flydsl_k1 err"], m, seq_len)
            if "flydsl_k1_graph err" in row:
                _raise_if_k1_mismatch(
                    row["flydsl_k1_graph err"], m, seq_len, where="FlyDSL K1 graph"
                )
                _raise_if_k1_mismatch(
                    row["vllm_amd_select_graph err"],
                    m,
                    seq_len,
                    where="live AMD select graph",
                )
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K1 prefill summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate, cache_layout in itertools.product(
            prefill_m, args.seq, args.page_size, args.rotate, args.cache_layout
        ):
            if _skip_m_past_seq(m, seq_len, "family A K2 prefill"):
                continue
            row = bench_qsa_family_a_k2(
                m, seq_len, page_size, dtype, rotate, cache_layout
            )
            _raise_if_k2_above_tolerance(row["flydsl_k2 err"], m, seq_len)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K2 prefill summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for (
            m,
            seq_len,
            page_size,
            rotate,
            pad_pages,
            cache_layout,
        ) in itertools.product(
            args.batch,
            [s for s in args.seq if s // FAMILY_B_INDEXER.compress_ratio <= 512],
            args.page_size,
            args.rotate,
            args.pad_pages,
            args.cache_layout,
        ):
            if _skip_m_past_seq(m, seq_len, "family B K1 H=4"):
                continue
            if _skip_pad_narrower_than_context(
                seq_len,
                page_size,
                pad_pages,
                FAMILY_B_INDEXER.compress_ratio,
                "family B K1 H=4",
            ):
                continue
            row = bench_qsa_family_b_k1(
                m, seq_len, page_size, dtype, 4, rotate, pad_pages, cache_layout
            )
            _raise_if_k1_mismatch(row["flydsl_k1 err"], m, seq_len)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 H=4 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for (
            m,
            seq_len,
            page_size,
            rotate,
            pad_pages,
            cache_layout,
        ) in itertools.product(
            args.batch,
            [s for s in args.seq if s // FAMILY_B_INDEXER_H8.compress_ratio <= 512],
            args.page_size,
            args.rotate,
            args.pad_pages,
            args.cache_layout,
        ):
            if _skip_m_past_seq(m, seq_len, "family B K1 H=8"):
                continue
            if _skip_pad_narrower_than_context(
                seq_len,
                page_size,
                pad_pages,
                FAMILY_B_INDEXER_H8.compress_ratio,
                "family B K1 H=8",
            ):
                continue
            row = bench_qsa_family_b_k1(
                m, seq_len, page_size, dtype, 8, rotate, pad_pages, cache_layout
            )
            _raise_if_k1_mismatch(row["flydsl_k1 err"], m, seq_len)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 H=8 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        # #4882 published indexer point: M=32, H=4, page_size=8, n_blocks=512.
        rows = []
        for rotate, pad_pages, cache_layout in itertools.product(
            args.rotate, args.pad_pages, args.cache_layout
        ):
            if _skip_pad_narrower_than_context(
                2048,
                8,
                pad_pages,
                FAMILY_B_INDEXER.compress_ratio,
                "family B K1 published",
            ):
                continue
            row = bench_qsa_family_b_k1(
                32, 2048, 8, dtype, 4, rotate, pad_pages, cache_layout
            )
            _raise_if_k1_mismatch(row["flydsl_k1 err"], 32, 2048)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 published indexer point (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for (
            m,
            seq_len,
            page_size,
            index_heads,
            rotate,
            pad_pages,
            cache_layout,
        ) in itertools.product(
            args.batch,
            [s for s in args.seq if s // FAMILY_B_INDEXER.compress_ratio > 512],
            args.page_size,
            (4, 8),
            args.rotate,
            args.pad_pages,
            args.cache_layout,
        ):
            if _skip_m_past_seq(m, seq_len, "family B K1 long-L"):
                continue
            if _skip_pad_narrower_than_context(
                seq_len,
                page_size,
                pad_pages,
                FAMILY_B_INDEXER.compress_ratio,
                "family B K1 long-L",
            ):
                continue
            row = bench_qsa_family_b_k1(
                m,
                seq_len,
                page_size,
                dtype,
                index_heads,
                rotate,
                pad_pages,
                cache_layout,
            )
            _raise_if_k1_mismatch(row["flydsl_k1 err"], m, seq_len)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 long-L summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate, cache_layout in itertools.product(
            args.batch, args.seq, args.page_size, args.rotate, args.cache_layout
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_a_e2e(
                    m, seq_len, page_size, dtype, rotate, cache_layout
                )
            )
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A end-to-end summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, cache_layout in itertools.product(
            [b for b in args.batch if b <= 8],
            args.seq,
            args.page_size,
            args.cache_layout,
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_a_e2e_graph(m, seq_len, page_size, dtype, cache_layout)
            )
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A decode HIP-graph summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for (
            m,
            seq_len,
            page_size,
            index_heads,
            rotate,
            cache_layout,
        ) in itertools.product(
            args.batch,
            args.seq,
            args.page_size,
            (4, 8),
            args.rotate,
            args.cache_layout,
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_e2e(
                    m,
                    seq_len,
                    page_size,
                    dtype,
                    index_heads,
                    rotate,
                    cache_layout,
                )
            )
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B end-to-end summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        if get_gfx() != "gfx950" or not gluon_qsa_available():
            aiter.logger.warning(
                "skip family B #4882 Gluon on %s (need gfx950 + Gluon import)",
                get_gfx(),
            )
            continue
        rows = []
        for m, seq_len, page_size, index_heads, rotate in itertools.product(
            args.batch, args.seq, args.page_size, (4, 8), args.rotate
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_4882_gluon(
                    m, seq_len, page_size, dtype, index_heads, rotate
                )
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family B #4882 Gluon summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
