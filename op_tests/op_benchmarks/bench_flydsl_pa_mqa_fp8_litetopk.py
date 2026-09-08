# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark paged FP8 MQA plus stable TopK against FP8 LiteTopK on gfx950."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from aiter import dtypes
from aiter.ops.flydsl import (
    FP8LiteTopKResult,
    allocate_fp8_litetopk_workspace,
    flydsl_pa_mqa_litetopk_fp8_prefill,
    fp8_litetopk_workspace_nbytes,
)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    _flydsl_pa_mqa_logits_fp8_prefill,
)
from aiter.ops.topk import _top_k_per_row_prefill, topk_ob_workspace_size
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
)

_MODEL = "glm-5.2"


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


def _make_preshuffled_cache(keys: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    page_size, head_dim = 64, 128
    pages = keys.shape[0] // page_size
    payload = (
        keys.view(pages, page_size // 16, 16, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(pages, page_size * head_dim)
    )
    cache = torch.empty(
        (pages, page_size * (head_dim + 4)),
        dtype=torch.uint8,
        device=keys.device,
    )
    cache[:, : page_size * head_dim] = payload.view(torch.uint8)
    cache[:, page_size * head_dim :] = scales.view(pages, page_size).view(torch.uint8)
    return cache.view(pages, page_size, 1, head_dim + 4)


def benchmark(
    model: str, query_rows: int, raw_context: int, warmup: int, iterations: int
) -> None:
    if model != _MODEL:
        raise ValueError(f"unsupported model profile: {model}")
    rows, context = query_rows, raw_context
    heads, head_dim, topk, page_size = 32, 128, 2048, 64
    if context % page_size:
        raise ValueError("context must be divisible by 64")
    if rows <= 0 or rows > context - topk:
        raise ValueError("rows must be positive and leave at least topk live tokens")
    device = torch.device("cuda", torch.cuda.current_device())
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"benchmark requires gfx950, got {arch}")

    torch.manual_seed(71)
    q_source = torch.randn((rows, heads, head_dim), dtype=torch.float32, device=device)
    q_scale = q_source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
    q_fp8 = (q_source / q_scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    weights = (
        torch.rand((rows, heads), dtype=torch.float32, device=device) * q_scale[:, :, 0]
    )

    key_source = torch.randn((context, head_dim), dtype=torch.float32, device=device)
    key_scales = key_source.abs().amax(dim=-1).clamp_min(1e-6) / 448.0
    keys_fp8 = (key_source / key_scales[:, None]).clamp(-448.0, 448.0).to(dtypes.fp8)
    kv_cache = _make_preshuffled_cache(keys_fp8, key_scales)
    pages = context // page_size
    block_tables = torch.arange(pages, dtype=torch.int32, device=device)[None]
    context_lens = torch.tensor([context], dtype=torch.int32, device=device)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device=device)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=device)
    row_ends = (
        context - rows + torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    )

    dense = torch.empty((rows, context), dtype=torch.float32, device=device)
    dense_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    dense_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    dense_topk_workspace = torch.empty(
        topk_ob_workspace_size(rows, context, topk, False),
        dtype=torch.uint8,
        device=device,
    )
    q_paged = q_fp8.view(1, rows, heads, head_dim)

    def run_dense() -> None:
        deepgemm_fp8_paged_mqa_logits(
            q_paged,
            kv_cache,
            weights,
            dense,
            context_lens,
            block_tables,
            context,
            Preshuffle=True,
            KVBlockSize=page_size,
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

    workspace = allocate_fp8_litetopk_workspace(rows, device)
    result = FP8LiteTopKResult(
        values=torch.empty((rows, topk), dtype=torch.float32, device=device),
        raw_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
        physical_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
        counts=workspace.output_counts,
        candidate_counts=workspace.candidate_counts,
        status=workspace.status,
    )

    def run_litetopk() -> None:
        flydsl_pa_mqa_litetopk_fp8_prefill(
            q_fp8,
            kv_cache,
            block_tables,
            weights,
            row_to_batch,
            row_starts,
            row_ends,
            context,
            workspace=workspace,
            out=result,
        )

    run_dense()
    run_litetopk()
    torch.cuda.synchronize()
    if not result.status.eq(0).all():
        raise AssertionError(f"LiteTopK status: {result.status.cpu().tolist()}")

    oracle_logits = _flydsl_pa_mqa_logits_fp8_prefill(
        q_fp8,
        kv_cache,
        block_tables,
        weights,
        row_to_batch,
        row_starts,
        row_ends,
        context,
        parallel_unit_num=max(1024, rows),
    )
    oracle_indices = torch.empty_like(dense_indices)
    oracle_values = torch.empty_like(dense_values)
    _top_k_per_row_prefill(
        oracle_logits,
        row_starts,
        row_ends,
        oracle_indices,
        oracle_values,
        rows,
        oracle_logits.stride(0),
        oracle_logits.stride(1),
        topk,
        dense_topk_workspace,
        True,
    )
    torch.cuda.synchronize()
    gluon_overlap = 0
    for row in range(rows):
        selected_indices = result.raw_indices[row]
        selected_in_range = (selected_indices >= row_starts[row]) & (
            selected_indices < row_ends[row]
        )
        selected_oracle_values = oracle_logits[row, selected_indices.long()]
        selected_values_match = torch.equal(
            torch.sort(selected_oracle_values).values,
            torch.sort(oracle_values[row]).values,
        )
        selected_unique = selected_indices.unique().numel() == topk
        if not (
            bool(selected_in_range.all()) and selected_unique and selected_values_match
        ):
            lite_set = set(result.raw_indices[row].cpu().tolist())
            oracle_set = set(oracle_indices[row].cpu().tolist())
            missing = sorted(oracle_set - lite_set)
            extra = sorted(lite_set - oracle_set)
            candidate_count = min(
                int(result.candidate_counts[row]), workspace.merge_cap
            )
            candidate_set = set(
                workspace.candidate_indices[row, :candidate_count].cpu().tolist()
            )
            raise AssertionError(
                f"LiteTopK value-set mismatch in row {row}: "
                f"overlap={topk - len(missing)}/{topk}, "
                f"unique={selected_unique}, in_range={bool(selected_in_range.all())}, "
                f"candidate_count={int(result.candidate_counts[row])}, "
                f"oracle_missing_from_candidates={len(oracle_set - candidate_set)}, "
                f"missing={missing[:8]}, extra={extra[:8]}, "
                f"oracle_boundary={float(oracle_values[row].min())}, "
                f"lite_boundary={float(result.values[row].min())}"
            )
        gluon_overlap += len(
            set(result.raw_indices[row].cpu().tolist())
            & set(dense_indices[row].cpu().tolist())
        )
    del oracle_logits

    (dense_p50, dense_p90), (lite_p50, lite_p90) = _paired_time_ms(
        run_dense, run_litetopk, warmup, iterations
    )
    print(
        {
            "model": model,
            "query_rows": rows,
            "heads": heads,
            "head_dim": head_dim,
            "topk": topk,
            "raw_context": context,
            "row_end_min": int(row_ends[0]),
            "row_end_max": int(row_ends[-1]),
            "warmup": warmup,
            "iterations": iterations,
            "gluon_dense_topk_ms_p50": dense_p50,
            "gluon_dense_topk_ms_p90": dense_p90,
            "litetopk_ms_p50": lite_p50,
            "litetopk_ms_p90": lite_p90,
            "speedup": dense_p50 / lite_p50,
            "gluon_topk_recall": gluon_overlap / (rows * topk),
            "workspace_mib": fp8_litetopk_workspace_nbytes(workspace) / 2**20,
            "candidate_mean": result.candidate_counts.float().mean().item(),
            "candidate_max": result.candidate_counts.max().item(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(_MODEL,), default=_MODEL)
    parser.add_argument("--query-rows", type=int, default=8192)
    parser.add_argument("--raw-context", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    benchmark(
        args.model,
        args.query_rows,
        args.raw_context,
        args.warmup,
        args.iterations,
    )


if __name__ == "__main__":
    main()
