# SPDX-License-Identifier: MIT
"""Benchmark large-row BF16-to-FP8 per-token quantization on gfx1250."""

from __future__ import annotations

import argparse

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
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    rows_to_test = [int(value) for value in args.rows.split(",")]
    generator = torch.Generator(device="cuda").manual_seed(2048)

    for rows in rows_to_test:
        graph_batch = max(4, min(512, 65536 // rows))
        source = torch.randn(
            (graph_batch, rows, args.cols),
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

        elapsed = _time_us(graph.replay, args.warmup, args.iterations, graph_batch)
        minimum_bytes = rows * args.cols * 3 + rows * 4
        expected_geometry = (
            "rows4x64"
            if rows > 4096 and args.cols == 1024
            else (
                "block128"
                if rows >= 1024 and args.cols <= 2048 and args.cols % 128 == 0
                else "block256"
            )
        )
        print(
            f"rows={rows} cols={args.cols} expected_geometry={expected_geometry} "
            f"graph_batch={graph_batch} us={elapsed:.3f} "
            f"min_gbps={minimum_bytes / elapsed / 1000:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
