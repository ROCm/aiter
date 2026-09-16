# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA oracle, family A plumbing, live vLLM AMD, #4882 Triton/Gluon, FlyDSL K1.

Two layers:
  * Correctness (pytest gate): ``test_*`` unit cases.
  * Perf sweep (``__main__``): ``bench_qsa_family_a_plumbing``,
    ``bench_qsa_family_a_vllm_amd``, ``bench_qsa_family_a_4882_triton``,
    ``bench_qsa_family_b_4882_triton``, ``bench_qsa_family_b_4882_gluon``,
    ``bench_qsa_family_a_k1`` (decode ``M<=8`` and a separate prefill table).

Usage::

    pytest -q op_tests/test_flydsl_qsa.py
    HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py
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
from aiter.ops.flydsl.qsa import (
    FAMILY_A_GQA,
    FAMILY_A_INDEXER,
    FAMILY_A_SCORE_SCALE,
    FAMILY_B_GQA,
    FAMILY_B_INDEXER,
    FAMILY_B_INDEXER_H8,
    gather_paged_cache,
    gather_qsa_family_a_caches,
    pack_paged_cache,
    qsa_expand_tail,
    qsa_indexer_scores,
    qsa_k1_family_a_block_ids,
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

SUPPORTED_GFX = ["gfx942", "gfx950"]


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
def bench_qsa_family_a_plumbing(m, seq_len, page_size, dtype):
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

    def paged_gather():
        return gather_qsa_family_a_caches(
            index_cache,
            index_table,
            k_cache,
            v_cache,
            kv_table,
            n_blocks,
            seq_len,
        )

    (k_bar_g, k_g, v_g), us = run_perftest(paged_gather)
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
    got = qsa_k1_family_a_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
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
    got = qsa_k1_family_a_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 two-tile block-id set diverged from the oracle")


def test_k1_family_a_set_equality_column_splits():
    """FlyDSL K1 matches the oracle when columns split across workgroups."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 2, 32768, 16
    n_blocks = seq_len // idx.compress_ratio
    assert n_blocks > 8 * 512
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
    got = qsa_k1_family_a_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 split-merge block-id set diverged from the oracle")


def test_k1_family_a_set_equality_prefill():
    """Same K1 instantiation matches the oracle at prefill M=512."""
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        return
    idx = FAMILY_A_INDEXER
    device = torch.device("cuda")
    m, seq_len, page_size = 512, 512, 16
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
    got = qsa_k1_family_a_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    if _set_mismatch_ratio(ref_ids, got) != 0.0:
        raise AssertionError("K1 prefill block-id set diverged from the oracle")


@benchmark()
def bench_qsa_family_a_k1(m, seq_len, page_size, dtype):
    """Family A FlyDSL K1 vs oracle set equality; us vs live AMD select.

    2d: eight waves per row, one column per thread per tile. Decode and
    prefill share one instantiation. Expand is not fused. Short rows emit
    ids; decode rows with more than one 512-slot tile split columns and
    pair-merge local heaps; prefill streams tiles in one workgroup (no
    score matrix).
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

    def k1():
        return qsa_k1_family_a_block_ids(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
        )

    block_ids, k1_us = run_perftest(k1)
    k1_err = _set_mismatch_ratio(ref_ids, block_ids)

    def select():
        return qsa_select_paged_tokens(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
        )

    (_indices, vllm_ids), vllm_us = run_perftest(select)
    vllm_err = _set_mismatch_ratio(ref_ids, vllm_ids)

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
    }


@benchmark()
def bench_qsa_family_a_vllm_amd(m, seq_len, page_size, dtype):
    """Live AMD path (vLLM Triton MQA + HIP top-k + Triton GQA) vs the oracle.

    Indexer chain and sparse GQA are timed separately. Oracle is not timed.
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

    def select():
        return qsa_select_paged_tokens(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
        )

    (indices, block_ids), select_us = run_perftest(select)
    block_err = _set_mismatch_ratio(ref.block_ids, block_ids)
    index_err = _set_mismatch_ratio(ref.indices, indices)

    def attend():
        return qsa_sparse_paged_attention(
            q_gqa, k_cache, v_cache, indices, kv_table, token_to_req
        )

    out, gqa_us = run_perftest(attend)
    gqa_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="vllm_amd GQA vs oracle",
    )

    w = idx.index_width
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
def bench_qsa_family_a_4882_triton(m, seq_len, page_size, dtype):
    """#4882 portable Triton QSA vs the oracle. Gluon is not launched.

    Indexer chain and sparse GQA are timed separately. Oracle is not timed.
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

    def select():
        return qsa_4882_select_paged_tokens(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
            backend="triton",
        )

    (indices, block_ids), select_us = run_perftest(select)
    block_err = _set_mismatch_ratio(ref.block_ids, block_ids)
    index_err = _set_mismatch_ratio(ref.indices, indices)

    def attend():
        return qsa_sparse_paged_gqa(
            q_gqa,
            k_cache,
            v_cache,
            indices,
            kv_table,
            token_to_req,
            backend="triton",
        )

    out, gqa_us = run_perftest(attend)
    gqa_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="4882 Triton GQA vs oracle",
    )

    w = idx.index_width
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


def _bench_qsa_family_b_4882(m, seq_len, page_size, dtype, index_heads, backend):
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

    def select():
        return qsa_4882_select_paged_tokens(
            q_indexer,
            index_cache,
            index_table,
            token_to_req,
            qpos,
            slen,
            idx.token_budget,
            idx.compress_ratio,
            backend=backend,
        )

    (indices, block_ids), select_us = run_perftest(select)
    block_err = _set_mismatch_ratio(ref.block_ids, block_ids)
    index_err = _set_mismatch_ratio(ref.indices, indices)
    if indices.shape[1] != idx.index_width:
        raise AssertionError(
            f"family B selection width {indices.shape[1]} != {idx.index_width}"
        )

    def attend():
        return qsa_sparse_paged_gqa(
            q_gqa,
            k_cache,
            v_cache,
            indices,
            kv_table,
            token_to_req,
            backend=backend,
        )

    gqa_tol = 2e-2 if backend == "gluon" else 1e-2
    out, gqa_us = run_perftest(attend)
    gqa_err = checkAllclose(
        ref.output,
        out.to(dtypes.fp32),
        rtol=gqa_tol,
        atol=gqa_tol,
        msg=f"4882 {backend} GQA vs oracle",
    )

    tag = backend
    w = idx.index_width
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
def bench_qsa_family_b_4882_triton(m, seq_len, page_size, dtype, index_heads):
    """#4882 portable Triton QSA vs the oracle on family B shapes.

    Separate table from family A Triton and from family B Gluon. Oracle is
    not timed.
    """
    return _bench_qsa_family_b_4882(
        m, seq_len, page_size, dtype, index_heads, backend="triton"
    )


@benchmark()
def bench_qsa_family_b_4882_gluon(m, seq_len, page_size, dtype, index_heads):
    """#4882 gfx950 Gluon QSA vs the oracle on family B (Gluon-validated) shapes.

    Family A GQA (group 12 / D=256) is not launched here. Oracle is not timed.
    """
    return _bench_qsa_family_b_4882(
        m, seq_len, page_size, dtype, index_heads, backend="gluon"
    )


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
    test_k1_family_a_set_equality_prefill()
    aiter.logger.info("QSA oracle + K1 unit cases passed")


def main():
    _run_unit_cases()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Family A QSA + vLLM AMD + #4882 + FlyDSL K1; family B #4882",
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
        help="context length L in tokens (32k default; pass 131072 for 128k)",
    )
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        nargs="*",
        default=[16],
        help="vLLM-style page size (indexer slots and GQA tokens)",
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
        for m, seq_len, page_size in itertools.product(
            args.batch, args.seq, args.page_size
        ):
            if m > seq_len:
                aiter.logger.warning(
                    "skip m=%s seq_len=%s (M must fit in L)", m, seq_len
                )
                continue
            rows.append(bench_qsa_family_a_plumbing(m, seq_len, page_size, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family A plumbing summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size in itertools.product(
            args.batch, args.seq, args.page_size
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_vllm_amd(m, seq_len, page_size, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family A vLLM AMD summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size in itertools.product(
            args.batch, args.seq, args.page_size
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_4882_triton(m, seq_len, page_size, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family A #4882 Triton summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size, index_heads in itertools.product(
            args.batch, args.seq, args.page_size, (4, 8)
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_4882_triton(
                    m, seq_len, page_size, dtype, index_heads
                )
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family B #4882 Triton summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        rows = []
        for m, seq_len, page_size in itertools.product(
            [b for b in args.batch if b <= 8],
            args.seq,
            args.page_size,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_k1(m, seq_len, page_size, dtype))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K1 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        rows = []
        for m, seq_len, page_size in itertools.product(
            [b for b in args.batch if b == 512],
            args.seq,
            args.page_size,
        ):
            if m > seq_len:
                continue
            rows.append(bench_qsa_family_a_k1(m, seq_len, page_size, dtype))
        if rows:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "QSA family A FlyDSL K1 prefill summary (markdown):\n%s",
                df.to_markdown(index=False),
            )

        if get_gfx() != "gfx950" or not gluon_qsa_available():
            aiter.logger.warning(
                "skip family B #4882 Gluon on %s (need gfx950 + Gluon import)",
                get_gfx(),
            )
            continue
        rows = []
        for m, seq_len, page_size, index_heads in itertools.product(
            args.batch, args.seq, args.page_size, (4, 8)
        ):
            if m > seq_len:
                continue
            rows.append(
                bench_qsa_family_b_4882_gluon(m, seq_len, page_size, dtype, index_heads)
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "QSA family B #4882 Gluon summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
