# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark IQ2R route-reduce + residual-add RMSNorm boundary fusion.

The baseline is the exact production pair:

1. ``iq2r_route_reduce_indexed_out``
2. ``aiter.rmsnorm2d_fwd_with_add``

The candidate preserves the route-reduction BF16 rounding boundary while
combining both operations into one launch.  Fixed-address buffers and HIP graph
replay make the result representative of ATOM decode graph execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Callable

import torch

import aiter
from aiter.ops.iq2r import (
    iq2r_route_reduce_add_rmsnorm_indexed_out,
    iq2r_route_reduce_indexed_out,
)

HIDDEN = 2880
TOPK = 4


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("IQ2R qualification requires gfx950")


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _benchmark(
    graph: torch.cuda.CUDAGraph, warmup: int, iterations: int, samples: int
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    return median(values), values


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    device = torch.device("cuda")
    records: list[dict[str, object]] = []

    for tokens in args.m:
        generator = torch.Generator(device=device).manual_seed(args.seed + tokens)
        routes = tokens * TOPK
        route_output = torch.randn(
            (routes, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        route_weights = torch.softmax(
            torch.randn(
                (tokens, TOPK),
                dtype=torch.float32,
                device=device,
                generator=generator,
            ),
            dim=-1,
        )
        scatter_indices = torch.randperm(
            routes, dtype=torch.int32, device=device, generator=generator
        )
        residual = torch.randn(
            (tokens, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        norm_weight = torch.randn(
            (HIDDEN,),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        reduced = torch.empty_like(residual)
        baseline_output = torch.empty_like(residual)
        baseline_residual = torch.empty_like(residual)

        def baseline_call() -> None:
            iq2r_route_reduce_indexed_out(
                route_output,
                route_weights,
                scatter_indices,
                reduced,
                topk=TOPK,
            )
            aiter.rmsnorm2d_fwd_with_add(
                baseline_output,
                reduced,
                residual,
                baseline_residual,
                norm_weight,
                args.epsilon,
            )

        baseline_graph = _capture(baseline_call)
        baseline_ms, baseline_samples = _benchmark(
            baseline_graph, args.warmup, args.iterations, args.samples
        )
        baseline_graph.replay()
        torch.cuda.synchronize()

        for block_size in args.block_size:
            fused_output = torch.empty_like(residual)
            fused_residual = torch.empty_like(residual)

            def fused_call() -> None:
                iq2r_route_reduce_add_rmsnorm_indexed_out(
                    route_output,
                    route_weights,
                    scatter_indices,
                    residual,
                    norm_weight,
                    fused_output,
                    fused_residual,
                    topk=TOPK,
                    epsilon=args.epsilon,
                    block_size=block_size,
                )

            fused_graph = _capture(fused_call)
            fused_ms, fused_samples = _benchmark(
                fused_graph, args.warmup, args.iterations, args.samples
            )
            fused_graph.replay()
            torch.cuda.synchronize()

            output_delta = (fused_output.float() - baseline_output.float()).abs()
            residual_delta = (fused_residual.float() - baseline_residual.float()).abs()
            output_close = torch.allclose(
                fused_output.float(),
                baseline_output.float(),
                rtol=args.rtol,
                atol=args.atol,
            )
            residual_equal = torch.equal(fused_residual, baseline_residual)
            records.append(
                {
                    "M": tokens,
                    "block_size": block_size,
                    "baseline_ms": baseline_ms,
                    "baseline_samples_ms": baseline_samples,
                    "fused_ms": fused_ms,
                    "fused_samples_ms": fused_samples,
                    "speedup": baseline_ms / fused_ms,
                    "saved_us_per_layer": (baseline_ms - fused_ms) * 1000.0,
                    "saved_ms_per_36_layers": (baseline_ms - fused_ms) * 36.0,
                    "output_max_abs": output_delta.max().item(),
                    "output_mean_abs": output_delta.mean().item(),
                    "residual_max_abs": residual_delta.max().item(),
                    "output_close": output_close,
                    "residual_exact": residual_equal,
                }
            )

    return {
        "benchmark": "iq2r-route-reduce-plus-add-rmsnorm-vs-fused",
        "device": torch.cuda.get_device_name(0),
        "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "hidden": HIDDEN,
        "topk": TOPK,
        "epsilon": args.epsilon,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 5, 8, 16])
    parser.add_argument("--block-size", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
