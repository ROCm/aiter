# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from aiter import logger
from aiter.ops.triton.attention.mfalcon_attention import (
    build_mfalcon_plan,
    mfalcon_attention,
)

LENGTH_CASES = {
    "dense_512": [512],
    "dense_1024": [1024],
    "ragged_512": [511],
    "ragged_1024": [1023],
}


def make_inputs(length_case, num_candidates, head_dim, dtype, num_heads=1):
    history_lengths = LENGTH_CASES[length_case]
    base_candidates = num_candidates // len(history_lengths)
    remainder = num_candidates % len(history_lengths)
    candidate_counts = [
        base_candidates + (1 if index < remainder else 0)
        for index in range(len(history_lengths))
    ]
    seq_lengths = [
        history + candidates
        for history, candidates in zip(history_lengths, candidate_counts)
    ]
    total_tokens = sum(seq_lengths)
    q = torch.randn((total_tokens, num_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((total_tokens, num_heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((total_tokens, num_heads, head_dim), device="cuda", dtype=dtype)
    seq_offsets = torch.tensor(
        [0] + [sum(seq_lengths[: i + 1]) for i in range(len(seq_lengths))],
        dtype=torch.int64,
        device="cuda",
    )
    num_candidates_tensor = torch.tensor(
        candidate_counts, dtype=torch.int64, device="cuda"
    )
    return q, k, v, seq_offsets, num_candidates_tensor


def baseline_attention(q, k, v, seq_offsets, num_candidates):
    batch_size = num_candidates.numel()
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    total_candidates = int(num_candidates.sum().item())
    out = torch.empty(
        (total_candidates, num_heads, head_dim), dtype=q.dtype, device=q.device
    )
    out_offset = 0
    scale = 1.0 / math.sqrt(head_dim)

    for batch_index in range(batch_size):
        start = int(seq_offsets[batch_index].item())
        end = int(seq_offsets[batch_index + 1].item())
        seq_len = end - start
        candidate_count = int(num_candidates[batch_index].item())
        history_len = seq_len - candidate_count
        q_seq = q[start:end].permute(1, 0, 2)
        k_seq = k[start:end].permute(1, 0, 2)
        v_seq = v[start:end].permute(1, 0, 2)

        mask = torch.zeros((seq_len, seq_len), dtype=q.dtype, device=q.device)
        if history_len > 0:
            future = torch.triu(
                torch.ones(
                    (history_len, history_len), dtype=torch.bool, device=q.device
                ),
                diagonal=1,
            )
            mask[:history_len, :history_len] = future.to(q.dtype) * float("-inf")
        if candidate_count > 0:
            candidate_indices = torch.arange(candidate_count, device=q.device)
            mask[history_len:, :history_len] = 0.0
            mask[history_len:, history_len:] = float("-inf")
            mask[history_len + candidate_indices, history_len + candidate_indices] = 0.0

        out_seq = F.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        out[out_offset : out_offset + candidate_count] = out_seq[
            :, history_len:
        ].permute(1, 0, 2)
        out_offset += candidate_count

    return out


def time_function(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def run_case(length_case, num_candidates, head_dim, dtype, warmup, iterations, check):
    q, k, v, seq_offsets, num_candidates_tensor = make_inputs(
        length_case, num_candidates, head_dim, dtype
    )
    plan = build_mfalcon_plan(q, k, v, seq_offsets, num_candidates_tensor)

    def op_function():
        return mfalcon_attention(q, k, v, seq_offsets, num_candidates_tensor, plan=plan)

    def baseline_function():
        return baseline_attention(q, k, v, seq_offsets, num_candidates_tensor)

    if check:
        op_output = op_function()
        baseline_output = baseline_function()
        torch.testing.assert_close(op_output, baseline_output, atol=2e-2, rtol=2e-2)

    op_ms = time_function(op_function, warmup, iterations)
    baseline_ms = time_function(baseline_function, warmup, iterations)
    return {
        "length_case": length_case,
        "num_candidates": num_candidates,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "op_ms": op_ms,
        "baseline_ms": baseline_ms,
        "speedup": baseline_ms / op_ms,
    }


def run_benchmark(args):
    torch.manual_seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    all_results = []

    for run_index in range(1, args.runs + 1):
        results = []
        for length_case in LENGTH_CASES:
            for num_candidates in (1, 8, 32, 128):
                for head_dim in (64, 128, 256):
                    for dtype in (torch.bfloat16, torch.float16):
                        results.append(
                            run_case(
                                length_case,
                                num_candidates,
                                head_dim,
                                dtype,
                                args.warmup,
                                args.iterations,
                                args.check,
                            )
                        )
        geomean = math.exp(
            sum(math.log(row["speedup"]) for row in results) / len(results)
        )
        worst_case = min(row["speedup"] for row in results)
        run_output = output_path.with_name(
            f"{output_path.stem}_run{run_index}{output_path.suffix}"
        )
        run_payload = {
            "device": torch.cuda.get_device_name(),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "geomean_speedup": geomean,
            "worst_case_speedup": worst_case,
            "results": results,
            "run": run_index,
        }
        run_output.write_text(json.dumps(run_payload, indent=2) + "\n")
        run_summaries.append(
            {
                "run": run_index,
                "geomean_speedup": geomean,
                "worst_case_speedup": worst_case,
            }
        )
        all_results.extend(results)
        logger.info(
            "run %d/%d: %d rows: geomean=%.3fx, worst=%.3fx, output=%s",
            run_index,
            args.runs,
            len(results),
            geomean,
            worst_case,
            run_output.resolve(),
        )

    median_geomean = statistics.median(row["geomean_speedup"] for row in run_summaries)
    median_worst = statistics.median(row["worst_case_speedup"] for row in run_summaries)
    geomean_values = [row["geomean_speedup"] for row in run_summaries]
    worst_values = [row["worst_case_speedup"] for row in run_summaries]
    summary_payload = {
        "device": torch.cuda.get_device_name(),
        "rows_per_run": len(all_results) // args.runs,
        "runs": run_summaries,
        "median_geomean_speedup": median_geomean,
        "geomean_range": [min(geomean_values), max(geomean_values)],
        "median_worst_case_speedup": median_worst,
        "worst_case_range": [min(worst_values), max(worst_values)],
        "geomean_spread_fraction": (max(geomean_values) - min(geomean_values))
        / min(geomean_values),
        "worst_case_spread_fraction": (max(worst_values) - min(worst_values))
        / min(worst_values),
    }
    output_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    logger.info(
        "%d runs: median geomean=%.3fx, median worst=%.3fx, output=%s",
        args.runs,
        median_geomean,
        median_worst,
        output_path.resolve(),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/mfalcon_benchmark_final.json")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
