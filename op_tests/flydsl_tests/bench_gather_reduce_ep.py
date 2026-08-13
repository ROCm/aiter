#!/usr/bin/env python3
"""Microbenchmark the EP gather-reduce kernel with dropped routes."""

from __future__ import annotations

import argparse
import statistics

import torch

from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    _get_compiled_gather_reduce,
    _get_compiled_gather_reduce_row,
    flydsl_moe_gather_reduce,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg


def make_inputs(
    tokens: int,
    valid_tokens: int,
    model_dim: int,
    topk: int,
    local_probability: float,
):
    generator = torch.Generator(device="cuda").manual_seed(2026)
    keep = torch.rand(
        (valid_tokens, topk), generator=generator, device="cuda"
    ) < local_probability
    # Dispatch sends a token to this rank only when at least one route is local.
    empty = ~keep.any(dim=1)
    keep[empty, 0] = True
    valid_routes = int(keep.sum().item())

    grouped = torch.randn(
        (1, valid_routes, model_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rows = torch.full((tokens, topk), -1, dtype=torch.int32, device="cuda")
    rows[:valid_tokens][keep] = torch.randperm(
        valid_routes, generator=generator, device="cuda", dtype=torch.int64
    ).to(torch.int32)
    weights = torch.rand(
        (tokens, topk),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weights[:valid_tokens].masked_fill_(~keep, 0)
    weights[valid_tokens:].zero_()
    out = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    num_valid_tokens = torch.tensor(
        [valid_tokens], dtype=torch.int32, device="cuda"
    )
    return grouped, rows, weights, out, num_valid_tokens, valid_routes


def make_launch(tensors, kind: str):
    grouped, rows, weights, out, num_valid_tokens, _ = tensors
    tokens, topk = rows.shape
    _, max_m, model_dim = grouped.shape
    if kind == "auto":
        def launch():
            flydsl_moe_gather_reduce(
                grouped,
                rows,
                weights,
                out=out,
                num_valid_tokens=num_valid_tokens,
            )

        return launch
    if kind == "row":
        kernel = _get_compiled_gather_reduce_row(
            model_dim,
            topk,
            "bf16",
            1,
            4,
            "bf16",
        )
    else:
        kernel = _get_compiled_gather_reduce(
            model_dim, topk, "bf16", 1, 4, "bf16"
        )
    slice_stride_dw = max_m * (model_dim // 2)

    def launch():
        kernel(
            ptr_arg(grouped),
            ptr_arg(rows),
            ptr_arg(weights),
            ptr_arg(out),
            tokens,
            slice_stride_dw,
            ptr_arg(num_valid_tokens),
            stream=torch.cuda.current_stream(),
        )

    return launch


def time_graph(launch, launches_per_graph: int, samples: int) -> float:
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches_per_graph):
            launch()

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / launches_per_graph)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--valid-tokens", type=int, default=1700)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--local-probability", type=float, default=0.25)
    parser.add_argument("--launches-per-graph", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    tensors = make_inputs(
        args.tokens,
        args.valid_tokens,
        args.model_dim,
        args.topk,
        args.local_probability,
    )
    reference = None
    timings = {}
    for kind in ("vec4", "row", "auto"):
        launch = make_launch(tensors, kind)
        launch()
        torch.cuda.synchronize()
        output = tensors[3].clone()
        if reference is None:
            reference = output
        else:
            torch.testing.assert_close(
                output, reference, rtol=0.02, atol=0.1
            )
        timings[kind] = time_graph(
            launch, args.launches_per_graph, args.samples
        )
    valid_routes = tensors[-1]
    print(
        f"tokens={args.tokens} valid_tokens={args.valid_tokens} "
        f"valid_routes={valid_routes}/{args.valid_tokens * args.topk} "
        f"vec4={timings['vec4']:.3f}us "
        f"row={timings['row']:.3f}us auto={timings['auto']:.3f}us",
        flush=True,
    )


if __name__ == "__main__":
    main()
