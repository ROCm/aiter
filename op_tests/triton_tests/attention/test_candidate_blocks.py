# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the hierarchical-indexer candidate selection ops.

The reference is a direct transcription of ``select_candidate_blocks`` from
DeepSeek-V4.1-Flash's ``inference/model.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.attention.candidate_blocks import (
    apply_candidate_mask,
    candidate_block_scores,
    candidate_mask_from_blocks,
    select_candidate_blocks,
)

_INF = float("inf")


def torch_block_scores(logits, compress_lens, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-_INF)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    arange = torch.arange(num_blocks, device=logits.device)
    return scores.masked_fill(arange == last, _INF)


def torch_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = torch_block_scores(logits, compress_lens, block_size)
    num_blocks = scores.size(-1)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -_INF
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def make_logits(T, L, compress_lens, seed=0, distinct=False):
    """Random logits with positions past each query's compress_len set to -inf.

    ``distinct=True`` draws a permutation instead, so no two block scores tie and
    a genuinely discriminating top-k has exactly one correct answer.
    """
    torch.manual_seed(seed)
    if distinct:
        assert T * L <= 1 << 24, "fp32 cannot hold that many distinct integers"
        logits = torch.randperm(T * L, device="cuda").to(torch.float32).reshape(T, L)
        logits = logits / (T * L) - 0.5
    else:
        logits = torch.randn(T, L, device="cuda", dtype=torch.float32)
    pos = torch.arange(L, device="cuda")
    logits.masked_fill_(pos >= compress_lens.reshape(T, 1), -_INF)
    return logits


def make_compress_lens(T, L, mode):
    """compress_lens covering the interesting boundaries."""
    if mode == "zero":
        vals = torch.zeros(T, dtype=torch.int32, device="cuda")
    elif mode == "one":
        vals = torch.ones(T, dtype=torch.int32, device="cuda")
    elif mode == "boundary":  # exactly on a block boundary
        vals = torch.full((T,), (L // 8) * 8, dtype=torch.int32, device="cuda")
    elif mode == "full":
        vals = torch.full((T,), L, dtype=torch.int32, device="cuda")
    elif mode == "mixed":
        torch.manual_seed(1234)
        vals = torch.randint(0, L + 1, (T,), dtype=torch.int32, device="cuda")
    else:
        raise ValueError(mode)
    return vals


@pytest.mark.parametrize("T", [1, 7, 512, 4096])
@pytest.mark.parametrize("L", [64, 999, 8192])
@pytest.mark.parametrize("mode", ["zero", "one", "boundary", "full", "mixed"])
def test_candidate_block_scores(T, L, mode):
    cl = make_compress_lens(T, L, mode)
    logits = make_logits(T, L, cl)
    ref = torch_block_scores(logits, cl.reshape(T, 1).long(), 8)
    out = candidate_block_scores(logits, cl, block_size=8)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("T", [1, 7, 512, 4096])
@pytest.mark.parametrize("L", [64, 999, 8192])
@pytest.mark.parametrize("mode", ["zero", "one", "boundary", "full", "mixed"])
def test_select_candidate_blocks(T, L, mode):
    """num_blocks (<= 1024 here) is below topk_blocks, so every reachable block
    is kept and the -inf fillers are the only thing dropped."""
    cl = make_compress_lens(T, L, mode)
    logits = make_logits(T, L, cl)
    ref = torch_select_candidate_blocks(logits, cl.reshape(T, 1).long(), 2048, 8)
    out = select_candidate_blocks(logits, cl, block_size=8, topk_blocks=2048)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("T", [1, 7, 512])
@pytest.mark.parametrize("topk_blocks", [1, 7, 64, 2048])
def test_select_candidate_blocks_discriminating(T, topk_blocks):
    """num_blocks (2048) >= topk_blocks: the top-k actually has to choose."""
    L = 16384
    cl = make_compress_lens(T, L, "mixed")
    logits = make_logits(T, L, cl, seed=7, distinct=True)
    ref = torch_select_candidate_blocks(logits, cl.reshape(T, 1).long(), topk_blocks, 8)
    out = select_candidate_blocks(logits, cl, block_size=8, topk_blocks=topk_blocks)
    assert torch.equal(out, ref)


def test_decode_scalar_compress_lens():
    """Decode passes a plain int rather than a per-row tensor."""
    T, L, cl = 1, 8192, 4097
    lens = torch.full((T,), cl, device="cuda")
    logits = make_logits(T, L, lens, seed=3, distinct=True)
    scores = candidate_block_scores(logits, cl, 8)
    assert torch.equal(scores, torch_block_scores(logits, cl, 8))
    ref = torch_select_candidate_blocks(logits, cl, 512, 8)
    assert torch.equal(select_candidate_blocks(logits, cl, 8, 512), ref)


@pytest.mark.parametrize("block_size", [4, 6, 8, 16])
def test_block_size_variants(block_size):
    T, L = 7, 999
    cl = make_compress_lens(T, L, "mixed")
    logits = make_logits(T, L, cl, seed=5, distinct=True)
    ref = torch_select_candidate_blocks(logits, cl.reshape(T, 1).long(), 32, block_size)
    out = select_candidate_blocks(logits, cl, block_size=block_size, topk_blocks=32)
    assert torch.equal(out, ref)


def test_inf_fillers_are_dropped():
    """Rows with fewer reachable blocks than k keep only the reachable ones."""
    T, L, block_size, topk = 4, 256, 8, 2048
    cl = torch.tensor([0, 1, 8, 100], dtype=torch.int32, device="cuda")
    logits = make_logits(T, L, cl, seed=11)
    ref = torch_select_candidate_blocks(
        logits, cl.reshape(T, 1).long(), topk, block_size
    )
    out = select_candidate_blocks(logits, cl, block_size, topk)
    assert torch.equal(out, ref)
    # row 0 sees nothing at all -> nothing kept, not even a pinned block
    assert out[0].sum().item() == 0
    # row 1 sees position 0 only -> exactly its one (pinned) block
    assert out[1].sum().item() == block_size
    # row 2's compress_len sits on a boundary: block 0 is pinned and is the only one
    assert out[2].sum().item() == block_size
    # row 3 sees 100 positions -> blocks 0..12, block 12 pinned
    assert out[3].sum().item() == 13 * block_size


def test_ties_do_not_straddle_the_cut():
    """Exact ties, arranged so any correct top-k yields the same mask."""
    T, L, block_size = 3, 128, 8
    num_blocks = L // block_size
    cl = torch.full((T,), L, dtype=torch.int32, device="cuda")
    logits = torch.zeros(T, L, device="cuda", dtype=torch.float32)
    logits[:, : L // 2] = 1.0  # blocks 0..7 all tie at 1.0, blocks 8..15 at 0.0
    topk = num_blocks // 2 + 1  # the 8 tied blocks + the pinned last block
    ref = torch_select_candidate_blocks(
        logits, cl.reshape(T, 1).long(), topk, block_size
    )
    out = select_candidate_blocks(logits, cl, block_size, topk)
    assert torch.equal(out, ref)
    assert out.sum().item() == T * topk * block_size


def test_all_equal_scores():
    """Every block ties and k == num_blocks: everything is kept."""
    T, L, block_size = 5, 64, 8
    cl = torch.full((T,), L, dtype=torch.int32, device="cuda")
    logits = torch.full((T, L), 0.5, device="cuda", dtype=torch.float32)
    ref = torch_select_candidate_blocks(
        logits, cl.reshape(T, 1).long(), 2048, block_size
    )
    out = select_candidate_blocks(logits, cl, block_size, 2048)
    assert torch.equal(out, ref)
    assert out.all()


def test_candidate_mask_from_blocks_standalone():
    """Hand-built picks, including an -inf pick that must be dropped."""
    T, width, block_size = 2, 20, 8  # 3 blocks, the last one partial (4 positions)
    scores = torch.tensor(
        [[1.0, -_INF, 2.0], [_INF, 0.5, -_INF]], device="cuda", dtype=torch.float32
    )
    idx = torch.tensor([[2, 1], [0, 2]], dtype=torch.int32, device="cuda")
    out = candidate_mask_from_blocks(idx, scores, width, block_size)
    ref = torch.zeros(T, width, dtype=torch.bool, device="cuda")
    ref[0, 16:20] = True  # block 2 kept, block 1 is -inf
    ref[1, 0:8] = True  # block 0 (+inf) kept, block 2 is -inf
    assert torch.equal(out, ref)


def test_out_arguments_are_reused():
    T, L = 7, 999
    cl = make_compress_lens(T, L, "mixed")
    logits = make_logits(T, L, cl, seed=9)
    num_blocks = (L + 7) // 8
    scores = torch.empty(T, num_blocks, dtype=torch.float32, device="cuda")
    assert candidate_block_scores(logits, cl, 8, out=scores) is scores
    assert torch.equal(scores, torch_block_scores(logits, cl.reshape(T, 1).long(), 8))

    mask = torch.ones(T, L, dtype=torch.bool, device="cuda")  # must be re-zeroed
    idx = scores.topk(32, dim=-1).indices.to(torch.int32)
    assert candidate_mask_from_blocks(idx, scores, L, 8, out=mask) is mask
    keep = torch.zeros(T, num_blocks, dtype=torch.bool, device="cuda")
    keep.scatter_(-1, idx.long(), scores.gather(-1, idx.long()) > -_INF)
    ref = keep.repeat_interleave(8, dim=-1)[..., :L]
    assert torch.equal(mask, ref)
    assert mask.sum().item() < T * L


@pytest.mark.parametrize("T,L", [(1, 8192), (512, 999)])
def test_apply_candidate_mask(T, L):
    cl = make_compress_lens(T, L, "mixed")
    logits = make_logits(T, L, cl, seed=13)
    mask = select_candidate_blocks(logits, cl, 8, 32)
    ref = logits.masked_fill(~mask, -_INF)
    assert apply_candidate_mask(logits, mask) is logits
    assert torch.equal(logits, ref)
