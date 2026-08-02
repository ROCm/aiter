# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the Kimi-K3 B1 BF16 latent-MoE local tail."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from aiter.ops.flydsl.latent_moe_tail import latent_moe_tail

LATENT = 3584
HIDDEN = 7168
EPSILON = 1.0e-6
ROTATIONS = 8

Case = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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
        up_weight = torch.randn(
            (HIDDEN, LATENT),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(LATENT**-0.5)
        result.append((routed, shared, rms_weight, up_weight))
    return result


def load_control() -> Operation:
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    import aiter

    def control(case: Case):
        routed, shared, rms_weight, up_weight = case
        normalized = aiter.rmsnorm2d_fwd(routed, rms_weight, EPSILON)
        return rocm_unquantized_gemm_impl(normalized, up_weight).add_(shared)

    return control


def candidate(case: Case):
    routed, shared, rms_weight, up_weight = case
    return latent_moe_tail(routed, shared, rms_weight, up_weight, EPSILON)


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
    expected = control(cases[0])
    actual = candidate(cases[0])
    torch.cuda.synchronize()
    error = relative_rmse(actual, expected)
    if error >= 2.0e-4:
        raise AssertionError(f"relative RMSE exceeds 2e-4: {error}")

    graphs = {
        "control": capture(control, cases, args.operations_per_graph),
        "candidate": capture(candidate, cases, args.operations_per_graph),
    }
    warmups = math.ceil(args.warmup_operations / args.operations_per_graph)
    for index in range(warmups * len(graphs)):
        graphs[("control", "candidate")[index % 2]].replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in graphs}
    for trial in range(args.trials):
        order = ("control", "candidate") if trial % 2 == 0 else ("candidate", "control")
        for name in order:
            samples[name].append(
                elapsed_us(
                    graphs[name], args.replays_per_trial, args.operations_per_graph
                )
            )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    rotating_weight_bytes = sum(
        case[3].numel() * case[3].element_size() for case in cases
    )
    result = {
        "shape": "B1 Kimi-K3 latent-MoE BF16 local tail",
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
        "speedup": medians["control"] / medians["candidate"],
        "samples_us": samples,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
