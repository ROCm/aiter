# SPDX-License-Identifier: MIT
"""Benchmark large-row BF16-to-FP8 per-token quantization on gfx1250."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aiter.ops.quant import dynamic_per_token_scaled_quant


def _time_us(fn, warmup: int, iterations: int, kernels_per_call: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iterations * kernels_per_call)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="16,2048,4536,5611")
    parser.add_argument("--cols", default="2048")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--mode", choices=("graph", "cold"), default="graph")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--cold-mib", type=int, default=256)
    parser.add_argument("--rw-roofline-gbps", type=float, default=12750.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    rows_to_test = [int(value) for value in args.rows.split(",")]
    cols_to_test = [int(value) for value in args.cols.split(",")]
    generator = torch.Generator(device="cuda").manual_seed(2048)
    results = []

    for cols in cols_to_test:
        for rows in rows_to_test:
            minimum_bytes = rows * cols * 3 + rows * 4
            if args.mode == "graph":
                graph_batch = max(4, min(512, 65536 // rows))
                source = torch.randn(
                    (graph_batch, rows, cols),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=generator,
                )
                output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
                scale = torch.empty((graph_batch, rows), dtype=torch.float32, device="cuda")

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for index in range(graph_batch):
                        dynamic_per_token_scaled_quant(
                            output[index], source[index], scale[index]
                        )

                elapsed = _time_us(
                    graph.replay, args.warmup, args.iterations, graph_batch
                )
                elapsed_samples = [elapsed]
                pool_span_mib = (
                    source.numel() * source.element_size() / (1024 * 1024)
                )
            else:
                cold_bytes = args.cold_mib * 1024 * 1024
                shape_elems = rows * cols
                # Two complete non-overlapping slots for large tensors; for
                # small tensors retain a >L2 address span and distribute the
                # sampled views across it.
                pool_elems = max(
                    math.ceil(cold_bytes / torch.bfloat16.itemsize),
                    2 * shape_elems,
                )
                source_pool = torch.empty(
                    pool_elems, dtype=torch.bfloat16, device="cuda"
                )
                source_pool.uniform_(-2.0, 2.0, generator=generator)
                output_pool = torch.empty(
                    max(math.ceil(cold_bytes / 2), 2 * shape_elems),
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                )
                scale_pool = torch.empty(
                    max(args.samples, 2) * rows,
                    dtype=torch.float32,
                    device="cuda",
                )
                flush = torch.zeros(
                    math.ceil(cold_bytes / torch.int32.itemsize),
                    dtype=torch.int32,
                    device="cuda",
                )
                slots = max(1, pool_elems // shape_elems)
                output_slots = max(1, output_pool.numel() // shape_elems)

                def view_at(sample: int):
                    slot = min(slots - 1, sample * slots // args.samples)
                    output_slot = min(
                        output_slots - 1, sample * output_slots // args.samples
                    )
                    source = source_pool.narrow(
                        0, slot * shape_elems, shape_elems
                    ).view(rows, cols)
                    output = output_pool.narrow(
                        0, output_slot * shape_elems, shape_elems
                    ).view(rows, cols)
                    scale = scale_pool.narrow(
                        0, sample * rows, rows
                    )
                    return source, output, scale

                for _ in range(args.warmup):
                    source, output, scale = view_at(0)
                    dynamic_per_token_scaled_quant(output, source, scale)
                torch.cuda.synchronize()

                starts = []
                ends = []
                for sample in range(args.samples):
                    source, output, scale = view_at(sample)
                    flush.add_(1)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    dynamic_per_token_scaled_quant(output, source, scale)
                    end.record()
                    starts.append(start)
                    ends.append(end)
                torch.cuda.synchronize()
                elapsed_samples = [
                    start.elapsed_time(end) * 1000
                    for start, end in zip(starts, ends)
                ]
                elapsed = sorted(elapsed_samples)[len(elapsed_samples) // 2]
                graph_batch = 1
                pool_span_mib = (
                    source_pool.numel()
                    * source_pool.element_size()
                    / (1024 * 1024)
                )

            min_gbps = minimum_bytes / elapsed / 1000
            expected_geometry = (
                "tdm8x1024"
                if rows >= 16384 and cols == 1024
                else (
                    "tdm2x64"
                    if (
                        (rows >= 4096 and cols == 2048)
                        or (rows >= 8192 and cols == 4096)
                    )
                    else (
                        "rows4x64"
                        if rows > 4096 and cols == 1024
                        else (
                            "block128"
                            if rows >= 1024
                            and cols <= 2048
                            and cols % 128 == 0
                            else "block256"
                        )
                    )
                )
            )
            result = {
                "label": args.label,
                "mode": args.mode,
                "rows": rows,
                "cols": cols,
                "expected_geometry": expected_geometry,
                "graph_batch": graph_batch,
                "us": elapsed,
                "samples_us": elapsed_samples,
                "minimum_bytes": minimum_bytes,
                "min_gbps": min_gbps,
                "rw_roofline_pct": 100 * min_gbps / args.rw_roofline_gbps,
                "pool_span_mib": pool_span_mib,
                "flush_mib": args.cold_mib if args.mode == "cold" else 0,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
