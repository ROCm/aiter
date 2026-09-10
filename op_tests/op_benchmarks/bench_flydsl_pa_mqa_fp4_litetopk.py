# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark FP4 paged MQA + stable TopK against FP4 LiteTopK on gfx950."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from aiter.ops.flydsl import (
    FP4LiteTopKResult,
    allocate_fp4_litetopk_workspace,
    flydsl_pa_mqa_litetopk_fp4_prefill,
    flydsl_pa_mqa_logits_fp4_prefill,
    fp4_litetopk_workspace_nbytes,
)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    compute_prefill_schedule,
)
from aiter.ops.topk import (
    _top_k_per_row_prefill,
    topk_ob_workspace_size,
)
from op_tests.test_flydsl_pa_mqa_logits_fp4_prefill import (
    indexer_k_fp4_paged_preshuffle,
    quant_q_fp4_preshuffle,
)

_MODEL_TOPKS = {
    "deepseek-v4-flash": 512,
    "deepseek-v4-pro": 1024,
}


def _paired_time_ms(
    first_fn, second_fn, warmup: int, iterations: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    functions = (first_fn, second_fn)
    for iteration in range(warmup):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for index in order:
            functions[index]()
    torch.cuda.synchronize()
    times = ([], [])
    for iteration in range(iterations):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        events = []
        for index in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[index]()
            end.record()
            events.append((index, start, end))
        events[-1][2].synchronize()
        for index, start, end in events:
            times[index].append(start.elapsed_time(end))

    summaries = []
    for samples in times:
        samples.sort()
        p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
        summaries.append((statistics.median(samples), p90))
    return summaries[0], summaries[1]


def benchmark(
    model: str, rows: int, raw_context: int, warmup: int, iterations: int
) -> None:
    heads, head_dim, page_size = 64, 128, 64
    topk = _MODEL_TOPKS[model]
    if raw_context % 4:
        raise ValueError("raw context must be divisible by the C4 compression ratio")
    context = raw_context // 4
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(71)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(context, head_dim, dtype=torch.bfloat16, device=device)
    weights = (torch.randn(rows, heads, device=device) * 0.1).to(torch.bfloat16)
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)

    pages = (context + page_size - 1) // page_size
    block_tables = torch.arange(pages, dtype=torch.int32, device=device)[None]
    guarded_tables = torch.cat(
        (block_tables, torch.zeros((1, 4), dtype=torch.int32, device=device)),
        dim=1,
    )
    kv_cache = torch.zeros(pages, 1, 4, page_size, 16, dtype=torch.uint8, device=device)
    kv_scale = torch.zeros(pages, 1, 4, page_size, dtype=torch.uint8, device=device)
    indices = torch.arange(context, dtype=torch.int32, device=device)
    indexer_k_fp4_paged_preshuffle(k, indices, kv_cache, kv_scale, page_size)

    row_to_batch = torch.zeros(rows, dtype=torch.int32, device=device)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=device)
    row_ends = torch.full((rows,), context, dtype=torch.int32, device=device)
    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch,
        row_starts,
        row_ends,
        256,
        max(1024, rows),
        context,
    )
    dense = torch.empty((rows, context), dtype=torch.float32, device=device)
    dense_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    dense_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    dense_topk_workspace = torch.empty(
        topk_ob_workspace_size(rows, context, topk, False),
        dtype=torch.uint8,
        device=device,
    )

    def run_legacy():
        flydsl_pa_mqa_logits_fp4_prefill(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            guarded_tables,
            weights,
            row_to_batch,
            row_starts,
            row_ends,
            context,
            out=dense,
            cta_info=cta_info,
            n_ctas=n_ctas,
            parallel_unit_num=max(1024, rows),
        )
        _top_k_per_row_prefill(
            dense,
            row_starts,
            row_ends,
            dense_indices,
            dense_values,
            rows,
            dense.stride(0),
            dense.stride(1),
            topk,
            dense_topk_workspace,
            True,
        )

    workspace = allocate_fp4_litetopk_workspace(rows, device, topk=topk)
    result = FP4LiteTopKResult(
        values=torch.empty((rows, topk), dtype=torch.float32, device=device),
        raw_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
        physical_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
        counts=workspace.output_counts,
        candidate_counts=workspace.candidate_counts,
        status=workspace.status,
    )

    def run_litetopk():
        flydsl_pa_mqa_litetopk_fp4_prefill(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            guarded_tables,
            weights,
            row_to_batch,
            row_starts,
            row_ends,
            context,
            topk=topk,
            workspace=workspace,
            out=result,
        )

    run_legacy()
    run_litetopk()
    torch.cuda.synchronize()
    for row in range(rows):
        assert torch.equal(
            torch.sort(result.raw_indices[row]).values,
            torch.sort(dense_indices[row]).values,
        )
    assert result.status.eq(0).all()
    (legacy_p50, legacy_p90), (lite_p50, lite_p90) = _paired_time_ms(
        run_legacy, run_litetopk, warmup, iterations
    )
    print(
        {
            "model": model,
            "query_rows": rows,
            "heads": heads,
            "head_dim": head_dim,
            "topk": topk,
            "raw_context": raw_context,
            "c4_context": context,
            "warmup": warmup,
            "iterations": iterations,
            "legacy_ms_p50": legacy_p50,
            "legacy_ms_p90": legacy_p90,
            "litetopk_ms_p50": lite_p50,
            "litetopk_ms_p90": lite_p90,
            "ratio": lite_p50 / legacy_p50,
            "workspace_mib": fp4_litetopk_workspace_nbytes(workspace) / 2**20,
            "candidate_mean": result.candidate_counts.float().mean().item(),
            "candidate_max": result.candidate_counts.max().item(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=tuple(_MODEL_TOPKS), default="deepseek-v4-flash"
    )
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--raw-context", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.model, args.rows, args.raw_context, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
