# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the Kimi-K3 B1 row-scaled-FP8 latent-MoE tail."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from aiter.ops.flydsl.latent_moe_tail_fp8 import (
    latent_moe_tail_fp8,
    quantize_latent_moe_tail_weight,
)

LATENT = 3584
HIDDEN = 7168
EPSILON = 1.0e-6
ROTATIONS = 8

Case = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
Operation = Callable[[Case], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--operations-per-graph", type=int, default=100)
    parser.add_argument("--warmup-operations", type=int, default=100_000)
    parser.add_argument("--replays-per-trial", type=int, default=10)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_cases(seed: int) -> list[Case]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = []
    for _ in range(ROTATIONS):
        routed = torch.randn(
            (1, LATENT), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        shared = torch.randn(
            (1, HIDDEN), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        rms_weight = torch.randn(
            (LATENT,), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        bf16_weight = torch.randn(
            (HIDDEN, LATENT),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(LATENT**-0.5)
        fp8_weight, scale = quantize_latent_moe_tail_weight(bf16_weight)
        result.append((routed, shared, rms_weight, bf16_weight, fp8_weight, scale))
    return result


def dequantized_weight(case: Case):
    return case[4].float() * case[5][:, None]


def load_control() -> Operation:
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    import aiter

    def control(case: Case):
        routed, shared, rms_weight, bf16_weight, _, _ = case
        normalized = aiter.rmsnorm2d_fwd(routed, rms_weight, EPSILON)
        return rocm_unquantized_gemm_impl(normalized, bf16_weight).add_(shared)

    return control


def candidate(case: Case):
    routed, shared, rms_weight, _, fp8_weight, scale = case
    return latent_moe_tail_fp8(routed, shared, rms_weight, fp8_weight, scale, EPSILON)


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return (error / scale).item()


def capture(operation: Operation, cases: list[Case], operations: int):
    for case in cases:
        operation(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(operations):
            operation(cases[index % len(cases)])
    return graph


def elapsed_us(graph: torch.cuda.CUDAGraph, replays: int, operations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * operations)


def main() -> None:
    args = parse_args()
    if (
        min(
            args.operations_per_graph,
            args.warmup_operations,
            args.replays_per_trial,
            args.trials,
        )
        <= 0
    ):
        raise ValueError("all benchmark counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not torch.version.hip or arch != "gfx950":
        raise RuntimeError(f"this benchmark requires ROCm gfx950, got {arch!r}")

    cases = make_cases(args.seed)
    control = load_control()
    routed, shared, rms_weight, _, _, _ = cases[0]
    inverse_rms = torch.rsqrt(
        routed.float().square().mean(dim=-1, keepdim=True) + EPSILON
    )
    normalized = (routed.float() * inverse_rms * rms_weight.float()).to(torch.bfloat16)
    expected = (
        (normalized.float() @ dequantized_weight(cases[0]).T)
        .to(torch.bfloat16)
        .float()
        .add_(shared.float())
        .to(torch.bfloat16)
    )
    actual = candidate(cases[0])
    torch.cuda.synchronize()
    error = relative_rmse(actual, expected)
    if error >= 5.0e-4:
        raise AssertionError(f"relative RMSE exceeds 5e-4: {error}")

    graphs = {
        "bf16_control": capture(control, cases, args.operations_per_graph),
        "fp8_candidate": capture(candidate, cases, args.operations_per_graph),
    }
    warmups = math.ceil(args.warmup_operations / args.operations_per_graph)
    for index in range(warmups * len(graphs)):
        name = ("bf16_control", "fp8_candidate")[index % 2]
        graphs[name].replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in graphs}
    for trial in range(args.trials):
        names = list(graphs)
        if trial % 2:
            names.reverse()
        for name in names:
            samples[name].append(
                elapsed_us(
                    graphs[name], args.replays_per_trial, args.operations_per_graph
                )
            )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    rotating_weight_bytes = sum(
        sum(tensor.numel() * tensor.element_size() for tensor in case[3:])
        for case in cases
    )
    result = {
        "shape": "B1 Kimi-K3 row-scaled-FP8 latent-MoE tail",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
        },
        "seed": args.seed,
        "rotations": ROTATIONS,
        "rotating_weight_bytes": rotating_weight_bytes,
        "cache_valid_rotation": rotating_weight_bytes > 256 * 1024 * 1024,
        "operations_per_graph": args.operations_per_graph,
        "warmup_operations": args.warmup_operations,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "relative_rmse": error,
        "p50_us": medians,
        "speedup": medians["bf16_control"] / medians["fp8_candidate"],
        "samples_us": samples,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
