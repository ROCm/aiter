# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark harness for bounded FP4 prefill score + TopK.

Usage:
    python op_tests/test_flydsl_pa_mqa_fp4_prefill_topk.py
    python op_tests/test_flydsl_pa_mqa_fp4_prefill_topk.py --bench
"""

import argparse

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    FP4_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE,
    allocate_fp4_prefill_topk_workspace,
    flydsl_pa_mqa_fp4_prefill_topk,
    flydsl_pa_mqa_fp4_score_tile_topk,
    flydsl_pa_mqa_logits_fp4_prefill,
)
from aiter.ops.topk import top_k_per_row_prefill
from aiter.test_common import run_perftest

try:
    from op_tests.test_flydsl_pa_mqa_logits_fp4_prefill import (
        indexer_k_fp4_paged_preshuffle,
        quant_q_fp4_preshuffle,
    )
except ModuleNotFoundError:
    # Direct ``python op_tests/<this file>`` puts op_tests, not the repository
    # root, at sys.path[0].
    from test_flydsl_pa_mqa_logits_fp4_prefill import (  # type: ignore[no-redef]
        indexer_k_fp4_paged_preshuffle,
        quant_q_fp4_preshuffle,
    )

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
TILE_TOKENS = 4096


def _make_case(seed=7, zero_q=False):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    batch = 2
    max_seq_len = 5120
    blocks_per_seq = max_seq_len // KV_BLOCK_SIZE
    num_blocks = batch * blocks_per_seq

    # Deliberately non-identity page tables so raw-index correctness cannot hide
    # a page-mapping bug.
    block_tables = torch.randperm(num_blocks, device=device, dtype=torch.int32).reshape(
        batch, blocks_per_seq
    )
    kv = torch.randn(batch, max_seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device)
    logical_batch = torch.arange(batch, device=device).repeat_interleave(max_seq_len)
    logical_token = torch.arange(max_seq_len, device=device).repeat(batch)
    physical_page = block_tables[logical_batch, logical_token // KV_BLOCK_SIZE].to(
        torch.int64
    )
    slot_mapping = (physical_page * KV_BLOCK_SIZE + logical_token % KV_BLOCK_SIZE).to(
        torch.int32
    )
    kv_cache = torch.zeros(
        num_blocks,
        1,
        4,
        KV_BLOCK_SIZE,
        16,
        dtype=torch.uint8,
        device=device,
    )
    kv_scale = torch.zeros(
        num_blocks,
        1,
        4,
        KV_BLOCK_SIZE,
        dtype=torch.uint8,
        device=device,
    )
    indexer_k_fp4_paged_preshuffle(
        kv.reshape(-1, HEAD_DIM),
        slot_mapping,
        kv_cache,
        kv_scale,
        KV_BLOCK_SIZE,
    )

    row_to_batch = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32, device=device)
    # Covers non-zero starts, short/empty rows, a tile boundary, and a long row
    # that must merge candidates from both score tiles.
    local_starts = torch.tensor(
        [37, 0, 4011, 1700, 2200, 900], dtype=torch.int32, device=device
    )
    local_ends = torch.tensor(
        [5097, 777, 4411, 1700, 4900, 5050], dtype=torch.int32, device=device
    )
    rows = row_to_batch.numel()
    if zero_q:
        q = torch.zeros(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    else:
        q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    weights = (torch.randn(rows, HEADS, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    return (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
    )


def _stable_reference(full_logits, block_tables, rb, starts, ends, k):
    rows = full_logits.shape[0]
    values = torch.full(
        (rows, k), -float("inf"), dtype=torch.float32, device=full_logits.device
    )
    raw_indices = torch.full(
        (rows, k), -1, dtype=torch.int32, device=full_logits.device
    )
    kv_indices = torch.full_like(raw_indices, -1)
    valid_counts = torch.empty(rows, dtype=torch.int32, device=full_logits.device)

    for row in range(rows):
        start, end = int(starts[row]), int(ends[row])
        count = min(k, max(end - start, 0))
        valid_counts[row] = count
        if count == 0:
            continue
        scores = full_logits[row, start:end]
        # Stable descending rank makes the smaller raw index win a score tie.
        ranked = torch.argsort(scores, descending=True, stable=True)[:count] + start
        # The stable aiter emitter returns the winner set in raw-index order.
        raw = torch.sort(ranked, stable=True).values
        raw_indices[row, :count] = raw.to(torch.int32)
        values[row, :count] = full_logits[row, raw]
        batch = int(rb[row])
        kv_indices[row, :count] = (
            block_tables[batch, raw // KV_BLOCK_SIZE] * KV_BLOCK_SIZE
            + raw % KV_BLOCK_SIZE
        ).to(torch.int32)
    return values, raw_indices, kv_indices, valid_counts


def _full_logits(case, weight_scale):
    (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        rb,
        starts,
        ends,
        max_seq_len,
    ) = case
    return flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        rb,
        starts,
        ends,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=256,
        kv_block_size=KV_BLOCK_SIZE,
    )


def check_case(k, *, zero_q=False, check_tile=True):
    case = _make_case(seed=17 + k, zero_q=zero_q)
    weight_scale = 1.25
    full = _full_logits(case, weight_scale)
    (
        q_fp4,
        _q_scale,
        _kv_cache,
        _kv_scale,
        block_tables,
        _weights,
        rb,
        starts,
        ends,
        max_seq_len,
    ) = case
    workspace = allocate_fp4_prefill_topk_workspace(
        q_fp4.shape[0], k, q_fp4.device, tile_tokens=TILE_TOKENS
    )
    result = flydsl_pa_mqa_fp4_prefill_topk(
        *case[:6],
        rb,
        starts,
        ends,
        max_seq_len,
        k=k,
        tile_tokens=TILE_TOKENS,
        weight_scale=weight_scale,
        workspace=workspace,
    )
    expected = _stable_reference(full, block_tables, rb, starts, ends, k)
    torch.cuda.synchronize()
    torch.testing.assert_close(result.raw_indices, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(result.kv_indices, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(result.valid_counts, expected[3], rtol=0, atol=0)
    torch.testing.assert_close(result.values, expected[0], rtol=0, atol=0)

    if check_tile:
        tile_start = TILE_TOKENS
        candidates = flydsl_pa_mqa_fp4_score_tile_topk(
            *case[:6],
            rb,
            starts,
            ends,
            max_seq_len,
            tile_start,
            k=k,
            tile_tokens=TILE_TOKENS,
            weight_scale=weight_scale,
            workspace=workspace,
        )
        tile_starts = torch.maximum(starts, torch.full_like(starts, tile_start))
        tile_ends = torch.minimum(
            ends, torch.full_like(ends, min(tile_start + TILE_TOKENS, max_seq_len))
        )
        tile_ref = _stable_reference(full, block_tables, rb, tile_starts, tile_ends, k)
        torch.cuda.synchronize()
        torch.testing.assert_close(candidates.raw_indices, tile_ref[1], rtol=0, atol=0)
        torch.testing.assert_close(candidates.valid_counts, tile_ref[3], rtol=0, atol=0)
        torch.testing.assert_close(candidates.values, tile_ref[0], rtol=0, atol=0)

    full_bytes = full.numel() * full.element_size()
    print(
        f"[pass] k={k} zero_q={zero_q} workspace={workspace.nbytes / 2**20:.2f} MiB "
        f"full_logits={full_bytes / 2**20:.2f} MiB"
    )
    return case, workspace


def benchmark_case(k, case, workspace, iters, warmup):
    (
        q_fp4,
        _q_scale,
        _kv_cache,
        _kv_scale,
        _block_tables,
        _weights,
        rb,
        starts,
        ends,
        max_seq_len,
    ) = case
    full = torch.full(
        (q_fp4.shape[0], max_seq_len),
        -float("inf"),
        dtype=torch.float32,
        device=q_fp4.device,
    )
    full_idx = torch.empty((q_fp4.shape[0], k), dtype=torch.int32, device=q_fp4.device)
    full_val = torch.empty(
        (q_fp4.shape[0], k), dtype=torch.float32, device=q_fp4.device
    )

    def bounded():
        return flydsl_pa_mqa_fp4_prefill_topk(
            *case[:6],
            rb,
            starts,
            ends,
            max_seq_len,
            k=k,
            tile_tokens=TILE_TOKENS,
            weight_scale=1.25,
            workspace=workspace,
        )

    def full_logits_topk():
        flydsl_pa_mqa_logits_fp4_prefill(
            *case[:6],
            rb,
            starts,
            ends,
            max_seq_len,
            weight_scale=1.25,
            block_k=256,
            kv_block_size=KV_BLOCK_SIZE,
            out=full,
        )
        top_k_per_row_prefill(
            full,
            starts,
            ends,
            full_idx,
            full_val,
            q_fp4.shape[0],
            full.stride(0),
            full.stride(1),
            k=k,
            stable=True,
        )

    _, bounded_us = run_perftest(bounded, num_iters=iters, num_warmup=warmup)
    _, full_us = run_perftest(full_logits_topk, num_iters=iters, num_warmup=warmup)
    print(
        f"[bench] k={k} bounded={bounded_us:.2f} us "
        f"full-logits+stable-topk={full_us:.2f} us "
        f"ratio={full_us / bounded_us:.3f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        print(
            f"[skip] requires gfx950 (current: {get_gfx() if torch.cuda.is_available() else 'none'})"
        )
        return

    assert not FP4_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE
    for k in (512, 1024):
        case, workspace = check_case(k)
        check_case(k, zero_q=True, check_tile=False)
        if args.bench:
            benchmark_case(k, case, workspace, args.iters, args.warmup)


if __name__ == "__main__":
    main()
