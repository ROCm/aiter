# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact sampled decode top-k against ordered-key and baseline oracles."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.topk.topk_per_row import (
    flydsl_radix_topk_one_block,
    flydsl_sampled_topk_decode,
)

K = 2048


@pytest.fixture(autouse=True)
def gfx950_only():
    if get_gfx_runtime() != "gfx950":
        pytest.skip("sampled selector requires gfx950")


def ordered_keys(values):
    bits = values.view(torch.int32).long()
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return torch.where(values.isnan(), 0x7FFFFFFF, ordered)


def check_selection(logits, seq, nn, output, base):
    lengths = seq[:, None] - nn + torch.arange(nn, device="cuda") + 1
    lengths = lengths.clamp(0, logits.shape[1]).flatten()
    for row, length in enumerate(lengths.tolist()):
        idx = output[row].long()
        valid = idx >= 0
        assert valid.sum().item() == min(length, K)
        assert (idx[valid] < length).all()
        assert idx[valid].unique().numel() == min(length, K)
        if length <= K:
            expected = torch.arange(K, device="cuda")
            expected[length:] = -1
            assert torch.equal(idx, expected)
        else:
            keys = ordered_keys(logits[row])
            selected = keys[idx].sort().values
            assert torch.equal(selected, keys[base[row].long()].sort().values)
            assert torch.equal(selected, keys[:length].topk(K).values.sort().values)


def run_case(logits, seq, nn, graphs=False):
    rows = logits.shape[0]
    base = torch.empty((rows, K), dtype=torch.int32, device="cuda")
    output = torch.full_like(base, -99)
    counts = torch.empty(rows, dtype=torch.int32, device="cuda")

    def run():
        flydsl_radix_topk_one_block(
            logits,
            None,
            seq,
            base,
            None,
            rows,
            logits.stride(0),
            logits.stride(1),
            is_decode=True,
            next_n=nn,
        )
        flydsl_sampled_topk_decode(logits, nn, seq, output, candidate_counts=counts)

    run()
    check_selection(logits, seq, nn, output, base)
    if graphs:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for _ in range(5):
            graph.replay()
            check_selection(logits, seq, nn, output, base)
    return counts


@pytest.mark.parametrize(
    "kind",
    [
        "equal",
        "inf_many",
        "inf_few",
        "exceptional",
        "ascending",
        "descending",
    ],
)
def test_adversarial(kind):
    torch.manual_seed(42)
    logits = torch.randn((4, 128000), device="cuda")
    if kind == "equal":
        logits.fill_(1)
    elif kind.startswith("inf"):
        logits[:, : 4096 if kind == "inf_many" else 128] = float("inf")
    elif kind == "exceptional":
        values = torch.tensor(
            [float("nan"), float("inf"), 0.0, -0.0, float("-inf"), -1.0], device="cuda"
        )
        logits = values.repeat(4, 21334)[:, :128000].contiguous()
    elif kind in ("ascending", "descending"):
        logits = torch.arange(128000, device="cuda", dtype=torch.float32).repeat(4, 1)
        if kind == "descending":
            logits = logits.flip(1)
    seq = torch.full((1,), 128000, device="cuda", dtype=torch.int32)
    counts = run_case(logits, seq, 4, graphs=True)
    if kind == "equal":
        # A tied plateau past the candidate capacity forces the exact fallback.
        assert (counts > 9000).all()


def sampled_columns(row, length):
    """Host mirror of the kernel's sampling positions for one row (length > 8192)."""
    m = 0xFFFFFFFF
    stratum = max(2048, ((length + 4095) // 4096) * 32)
    chunks = (length + stratum - 1) // stratum
    cols = []
    for chunk in range(min(chunks, 128)):
        begin = chunk * stratum
        extent = min(stratum, length - begin)
        h = (chunk + row * 4099 + 17) & m
        h = ((h ^ (h >> 16)) * 0x7FEB352D) & m
        h = ((h ^ (h >> 15)) * 0x846CA68B) & m
        h ^= h >> 16
        base = begin + (h % max(extent // 32, 1)) * 32
        cols += [base + t * 4 + j for t in range(8) for j in range(4)]
    return [c for c in cols if c < length]


def test_underfill_falls_back():
    # Every sampled column holds a huge value, so the sampled threshold sits far
    # above the true kth value and fewer than k elements pass it.
    torch.manual_seed(3)
    width = 128000
    logits = torch.randn((4, width), device="cuda")
    for row in range(4):
        cols = torch.tensor(sampled_columns(row, width), device="cuda")
        logits[row, cols] = 100.0 + torch.rand(cols.numel(), device="cuda")
    seq = torch.full((1,), width, device="cuda", dtype=torch.int32)
    counts = run_case(logits, seq, 4)
    # Count reports the candidate total; underfill means it is below k.
    assert (counts < K).all()


def test_row_shorter_than_k():
    seq = torch.full((1,), K - 5, device="cuda", dtype=torch.int32)
    logits = torch.randn((4, 128000), device="cuda")
    counts = run_case(logits, seq, 4)
    assert (counts == -1).all()


def test_short_and_ragged():
    torch.manual_seed(9)
    seq = torch.tensor(
        [0, 7, K - 1, K, K + 3, 8192, 11000, 128001], device="cuda", dtype=torch.int32
    )
    logits = torch.randn((seq.numel() * 4, 128000), device="cuda")
    run_case(logits, seq, 4, graphs=True)


@pytest.mark.parametrize(
    "ctx, nn", [(8192, 1), (8192, 2), (8192, 4), (128000, 1), (128000, 2), (128000, 4)]
)
def test_paged_mqa_logits(ctx, nn):
    from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx950 import (
        flydsl_fp8_paged_mqa_logits,
    )
    from op_tests.test_flydsl_fp8_paged_mqa_logits import (
        kv_cache_cast_to_fp8,
        preshuffle_kv_data,
    )

    b = 64
    pages = (ctx + 63) // 64
    nblocks = b * pages
    torch.manual_seed(0)
    tb = min(nblocks, 8192)
    raw = kv_cache_cast_to_fp8(
        torch.randn((tb, 64, 1, 128), device="cuda", dtype=torch.bfloat16),
        torch.float8_e4m3fn,
    )
    template = preshuffle_kv_data(raw, 128)
    del raw
    pool = torch.empty((nblocks, 64, 1, 132), device="cuda", dtype=torch.uint8)
    for start in range(0, nblocks, tb):
        count = min(tb, nblocks - start)
        pool[start : start + count].copy_(template[:count])
    del template
    table = torch.randperm(nblocks, device="cuda", dtype=torch.int32).view(b, pages)
    seq = torch.full((b,), ctx, device="cuda", dtype=torch.int32)
    torch.manual_seed(100)
    q = torch.randn((b, nn, 32, 128), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn((b * nn, 32), device="cuda")
    logits = torch.full((b * nn, ctx), float("-inf"), device="cuda")
    flydsl_fp8_paged_mqa_logits(q, pool, weights, logits, seq, table, ctx)
    torch.cuda.synchronize()
    del pool
    counts = run_case(logits, seq, nn, graphs=True)
    accepted = ((counts >= K) & (counts <= 9000)).sum().item()
    print(f"ACCEPT ctx={ctx} B={b} nn={nn}: {accepted}/{b * nn}", flush=True)


if __name__ == "__main__":
    # CI runs op_tests/test_*.py as scripts; hand off to pytest so cases execute.
    import sys

    sys.exit(pytest.main(["-q", __file__]))
