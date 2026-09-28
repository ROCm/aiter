# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA oracle, family A plumbing, live vLLM AMD, #4882 Triton/Gluon, FlyDSL K1/K2.

Two layers:
  * Correctness (pytest gate): ``test_*`` unit cases.
  * Perf sweep (``__main__``): ``bench_qsa_family_a_plumbing``,
    ``bench_qsa_family_a_vllm_amd``, ``bench_qsa_family_a_4882_triton``,
    ``bench_qsa_family_b_4882_triton``, ``bench_qsa_family_b_4882_gluon``,
    ``bench_qsa_family_a_k1`` (decode ``M<=8`` and a separate prefill table),
    ``bench_qsa_family_b_k1`` (emit; long-``L`` uses family A scorer; published point),
    ``bench_qsa_family_a_k2`` (3d decode ``M<=8`` and a separate prefill table),
    ``bench_qsa_family_a_e2e`` / ``bench_qsa_family_b_e2e`` (indexer through GQA),
    ``bench_qsa_family_a_e2e_graph`` (HIP graph replay at decode).

Usage::

    pytest -q op_tests/test_flydsl_qsa.py
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py --rotate 0 1

Perf rows default to cold weights (``--rotate 0``); pass ``--rotate 1`` to
reproduce the older hot-cache 1047 tables.
"""

from __future__ import annotations

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.qsa import k1 as k1_kernel
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
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.qsa_shapes import (
    FAMILY_A_GQA,
    FAMILY_A_INDEXER,
    FAMILY_A_SCORE_SCALE,
    FAMILY_B_GQA,
    FAMILY_B_INDEXER,
    FAMILY_B_INDEXER_H8,
)

SUPPORTED_GFX = ["gfx942", "gfx950"]


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
    """
    return run_perftest(fn, *args, num_rotate_args=rotate, **kwargs)


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


def _selected_width(indices: torch.Tensor) -> float:
    """Mean non-padding selection slots per row.

    ``indices`` is always allocated ``index_width`` wide (2051 on both
    families), but the indexer can only fill ``complete_blocks * r + tail``
    of it, capped at the budget. Below ``L = 2048`` the remainder is ``-1``
    padding: a quarter of the row is live at ``L = 512`` decode and an eighth
    at ``M = 512`` prefill. Deriving FLOPS and bytes from ``indices.shape[1]``
    therefore overstates the work by up to 8x on those rows. The kernels still
    walk all ``index_width`` columns -- that part is faithful to serving -- so
    only the *derived* columns need the live count.
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


def _set_mismatch_ratio(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Fraction of rows whose non-(-1) id sets differ."""
    miss = 0
    rows = ref.shape[0]
    for i in range(rows):
        a = set(ref[i].tolist()) - {-1}
        b = set(got[i].tolist()) - {-1}
        if a != b:
            miss += 1
    return miss / rows if rows else 0.0


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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 block-id set diverged from the oracle")


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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 two-tile block-id set diverged from the oracle")


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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 wide-row block-id set diverged from the oracle")


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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 prefill block-id set diverged from the oracle")


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


@benchmark()
def bench_qsa_family_a_k1(m, seq_len, page_size, dtype, rotate=0):
    """Family A FlyDSL K1 vs oracle set equality; us vs live AMD and #4882 Triton.

    2d: short rows use fused emit. Long rows use BLOCK_N=32 BF16 MFMA scoring
    into an fp32 score matrix. Selection is decode radix below 32768 columns
    and streaming radix at or above that width. Single-request prefill scores
    16 query rows per workgroup. Expand is not fused. Same ``rotate`` on every
    select column. Family A GQA is group 12 / D=256, so #4882 Gluon does not
    dispatch and is not a column here.
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
    index_cache = index_cache.contiguous()
    index_table = index_table.contiguous()

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

    block_ids, k1_us = _time(
        qsa_k1_block_ids,
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        rotate=rotate,
        heads=(4,),
    )
    k1_err = _set_mismatch_ratio(ref_ids, block_ids)

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
    vllm_err = _set_mismatch_ratio(ref_ids, vllm_ids)

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
    triton_err = _set_mismatch_ratio(ref_ids, triton_ids)

    flops = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    nbytes = (m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim) * dtype.itemsize
    return {
        "gfx": get_gfx(),
        "n_blocks": n_blocks,
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


@benchmark()
def bench_qsa_family_a_k2(m, seq_len, page_size, dtype, rotate=0):
    """Family A FlyDSL K2 vs oracle GQA; us vs live AMD and #4882 Triton.

    3d: live-AMD-shaped BLOCK_N/threads/split policy, tiled MFMA QK/PV,
    log2 online softmax, direct output at one split, and a two-wave merge.
    Expand and sigmoid stay unfused. Same ``rotate`` on every GQA column.
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
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("family B K1 block-id set diverged from the oracle")


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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("family B K1 H=8 block-id set diverged from the oracle")


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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError(
            "family B K1 two-tile block-id set diverged from the oracle"
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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError(
            "family B K1 H=8 two-tile block-id set diverged from the oracle"
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
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError(
            "family B K1 published-indexer block-id set diverged from the oracle"
        )


@benchmark()
def bench_qsa_family_b_k1(m, seq_len, page_size, dtype, index_heads, rotate=0):
    """Family B FlyDSL K1 vs oracle set equality; us vs #4882 select.

    2e/2f: emit on ``n_blocks <= 512``. Longer rows use family A's MFMA
    scorer plus radix (``H=8`` is a second compile). ``H`` 4 and 8 emit
    share one kernel. Separate table from family A. Expand is not fused.
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
    index_cache = index_cache.contiguous()
    index_table = index_table.contiguous()
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
    k1_err = _set_mismatch_ratio(ref_ids, block_ids)

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
    triton_err = _set_mismatch_ratio(ref_ids, triton_ids)

    flops = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    nbytes = (m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim) * dtype.itemsize
    ret = {
        "gfx": get_gfx(),
        "index_heads": idx.n_heads,
        "n_blocks": n_blocks,
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
        ret["4882_gluon_select err"] = _set_mismatch_ratio(ref_ids, gluon_ids)
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


def _policy_args(m, hq, d_gqa, n_columns, page_size, n_heads, d_idx):
    """Host-only tensors for the auto-backend predicate. Nothing is launched."""
    n_pages = n_columns // page_size
    q_indexer = torch.zeros(m, n_heads, d_idx, dtype=torch.bfloat16)
    q_gqa = torch.zeros(m, hq, d_gqa, dtype=torch.bfloat16)
    index_cache = torch.zeros(n_pages, page_size, 1, d_idx, dtype=torch.bfloat16)
    index_table = torch.zeros(1, n_pages, dtype=torch.int32)
    k_cache = torch.zeros(n_pages, page_size, 2, d_gqa, dtype=torch.bfloat16)
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


def test_qsa_backend_default_is_live_amd():
    """The opt-in defaults to Triton, and unknown names are rejected."""
    assert normalize_qsa_backend(None) == "triton"
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

    Both ways this can break are silent. Loosened, auto serves an untuned
    shape at whatever speed the K2 band table happens to give; narrowed, it
    drops a measured shape back to Triton. Neither is a wrong result, so no
    other test in this file would notice. K2 serves any structurally valid
    shape, so the table in ``qsa.py`` is the only thing doing the rejecting.
    """
    page = 16
    n_columns = 128
    for hq, d_gqa, heads in ((24, 256, 4), (10, 128, 4), (10, 128, 8)):
        swept = _policy_args(1, hq, d_gqa, n_columns, page, heads, 128)
        assert qsa_auto_uses_flydsl(*swept) is True
    # Family A was only ever swept with a 4-head indexer.
    wide_indexer = _policy_args(1, 24, 256, n_columns, page, 8, 128)
    assert qsa_auto_uses_flydsl(*wide_indexer) is False
    # An untuned GQA query stays on Triton however it is indexed.
    untuned = _policy_args(1, 16, 128, n_columns, page, 4, 128)
    assert qsa_auto_uses_flydsl(*untuned) is False


def test_qsa_symbols_export_lazily():
    """K1, K2, and the layer are on ``aiter.ops.flydsl`` without a side import."""
    from aiter.ops import flydsl

    assert flydsl.qsa_k1_block_ids is qsa_k1_block_ids
    assert flydsl.qsa_k2 is qsa_k2
    assert flydsl.qsa_layer is qsa_layer
    assert flydsl.normalize_qsa_backend is normalize_qsa_backend


def _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype):
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
    index_table = index_table.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
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
    """FlyDSL and the default Triton opt-in both match the oracle on one layer."""
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
def bench_qsa_family_a_e2e(m, seq_len, page_size, dtype, rotate=0):
    """One family A QSA layer: FlyDSL K1+expand+K2 vs live AMD and #4882 Triton.

    Expand stays the vendored Triton kernel. Oracle is not timed. Same
    ``rotate`` on every column. HIP graph replay is a separate table.
    """
    idx, gqa = FAMILY_A_INDEXER, FAMILY_A_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype)
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
    }
    ret.update(_e2e_cells("flydsl_e2e", fly_us, fly_err, flops, nbytes))
    ret.update(_e2e_cells("vllm_amd_e2e", amd_us, amd_err, flops, nbytes))
    ret.update(_e2e_cells("4882_triton_e2e", t4882_us, t4882_err, flops, nbytes))
    return ret


@benchmark()
def bench_qsa_family_b_e2e(m, seq_len, page_size, dtype, index_heads, rotate=0):
    """One family B QSA layer vs live AMD, #4882 Triton and, on gfx950, Gluon.

    Separate table from family A. Live AMD is the column ``auto`` decides
    against, so it is the one that governs the gate; the others are parity.
    """
    idx = _family_b_indexer(index_heads)
    gqa = FAMILY_B_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype)
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
def bench_qsa_family_a_e2e_graph(m, seq_len, page_size, dtype):
    """HIP graph replay of one family A decode layer. Not combined with rotate.

    Each candidate is captured once, then ``replay`` is timed hot. The
    output buffer after replay is the correctness check.
    """
    idx, gqa = FAMILY_A_INDEXER, FAMILY_A_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype)
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
    test_k1_family_a_set_equality_two_tiles()
    test_k1_family_a_set_equality_wide_stream()
    test_k1_family_a_set_equality_prefill()
    test_k1_family_b_set_equality_short_decode()
    test_k1_family_b_set_equality_short_decode_h8()
    test_k1_family_b_set_equality_two_tiles()
    test_k1_family_b_set_equality_two_tiles_h8()
    test_k1_family_b_set_equality_published_indexer_point()
    test_k2_family_a_decode_matches_oracle()
    test_k2_family_a_prefill_matches_oracle()
    test_k2_page_past_4gib()
    test_k2_empty_first_tile_keeps_later_token()
    test_k2_default_out_ignores_query_strides()
    test_qsa_backend_default_is_live_amd()
    test_qsa_auto_admits_only_measured_pairs()
    test_qsa_symbols_export_lazily()
    test_qsa_layer_family_a_matches_oracle()
    test_qsa_layer_family_b_matches_oracle()
    test_qsa_layer_decode_graph_replays()
    aiter.logger.info("QSA oracle + K1 + K2 + layer unit cases passed")


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
        help="flattened query tokens M (decode 1..8, prefill 512)",
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

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            [b for b in args.batch if b <= 8],
            args.seq,
            args.page_size,
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_k1(m, seq_len, page_size, dtype, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K1 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            [b for b in args.batch if b <= 8],
            args.seq,
            args.page_size,
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_k2(m, seq_len, page_size, dtype, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K2 decode summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            [b for b in args.batch if b == 512],
            args.seq,
            args.page_size,
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_k1(m, seq_len, page_size, dtype, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K1 prefill summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            [b for b in args.batch if b == 512],
            args.seq,
            args.page_size,
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_k2(m, seq_len, page_size, dtype, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K2 prefill summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            [b for b in args.batch if b <= 8],
            [s for s in args.seq if s // FAMILY_B_INDEXER.compress_ratio <= 512],
            args.page_size,
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_b_k1(m, seq_len, page_size, dtype, 4, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 H=4 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            [b for b in args.batch if b <= 8],
            [s for s in args.seq if s // FAMILY_B_INDEXER_H8.compress_ratio <= 512],
            args.page_size,
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_b_k1(m, seq_len, page_size, dtype, 8, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 H=8 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        # #4882 published indexer point: M=32, H=4, page_size=8, n_blocks=512.
        rows = [
            bench_qsa_family_b_k1(32, 2048, 8, dtype, 4, rotate)
            for rotate in args.rotate
        ]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family B FlyDSL K1 published indexer point (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size, index_heads, rotate in itertools.product(
            [b for b in args.batch if b <= 8],
            [s for s in args.seq if s // FAMILY_B_INDEXER.compress_ratio > 512],
            args.page_size,
            (4, 8),
            args.rotate,
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_k1(m, seq_len, page_size, dtype, index_heads, rotate)
            )
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family B FlyDSL K1 long-L summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, rotate in itertools.product(
            args.batch, args.seq, args.page_size, args.rotate
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_e2e(m, seq_len, page_size, dtype, rotate))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A end-to-end summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size in itertools.product(
            [b for b in args.batch if b <= 8], args.seq, args.page_size
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_e2e_graph(m, seq_len, page_size, dtype))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A decode HIP-graph summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size, index_heads, rotate in itertools.product(
            args.batch, args.seq, args.page_size, (4, 8), args.rotate
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_e2e(
                    m, seq_len, page_size, dtype, index_heads, rotate
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
