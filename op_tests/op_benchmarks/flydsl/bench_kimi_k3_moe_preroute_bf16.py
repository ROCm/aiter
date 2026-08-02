# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the Kimi-K3 B1 BF16 MoE pre-route production boundary."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
    kimi_k3_moe_preroute_bf16,
)

HIDDEN_SIZE = 7168
ROUTED_SIZE = 3584
SHARED_GATE_UP_SIZE = 1536
SHARED_INTERMEDIATE_SIZE = 768
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

Weights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Operation = Callable[[torch.Tensor, Weights], tuple[torch.Tensor, torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--weight-sets", type=int, default=8)
    parser.add_argument("--operations-per-graph", type=int, default=96)
    parser.add_argument("--warmup-replays", type=int, default=100)
    parser.add_argument("--replays-per-trial", type=int, default=10)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def situ_and_mul(value: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "situ_and_mul"):
        import vllm._C  # noqa: F401

    output = torch.empty(
        value.shape[:-1] + (value.shape[-1] // 2,),
        dtype=value.dtype,
        device=value.device,
    )
    torch.ops._C.situ_and_mul(output, value, SITU_BETA, SITU_LINEAR_BETA)
    return output


def load_control() -> Operation:
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    def control(hidden: torch.Tensor, weights: Weights):
        routed_weight, shared_gate_up_weight, shared_down_weight = weights
        routed = rocm_unquantized_gemm_impl(hidden, routed_weight)
        shared_gate_up = rocm_unquantized_gemm_impl(hidden, shared_gate_up_weight)
        activated = situ_and_mul(shared_gate_up)
        shared = rocm_unquantized_gemm_impl(activated, shared_down_weight)
        return routed, shared

    return control


def candidate(hidden: torch.Tensor, weights: Weights):
    return kimi_k3_moe_preroute_bf16(hidden, *weights)


def make_inputs(seed: int, weight_sets: int) -> tuple[torch.Tensor, list[Weights]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        (1, HIDDEN_SIZE),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    banks = []
    for _ in range(weight_sets):
        routed = torch.randn(
            (ROUTED_SIZE, HIDDEN_SIZE),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(HIDDEN_SIZE**-0.5)
        shared_gate_up = torch.randn(
            (SHARED_GATE_UP_SIZE, HIDDEN_SIZE),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(HIDDEN_SIZE**-0.5)
        shared_down = torch.randn(
            (HIDDEN_SIZE, SHARED_INTERMEDIATE_SIZE),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(SHARED_INTERMEDIATE_SIZE**-0.5)
        banks.append((routed, shared_gate_up, shared_down))
    return hidden, banks


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return (error / scale).item()


def check_correctness(control: Operation, hidden: torch.Tensor, weights: Weights):
    expected = control(hidden, weights)
    actual = candidate(hidden, weights)
    torch.cuda.synchronize()
    errors = [relative_rmse(got, want) for got, want in zip(actual, expected)]
    if max(errors) >= 2.0e-4:
        raise AssertionError(f"relative RMSE exceeds 2e-4: {errors}")
    return errors


def capture(
    operation: Operation,
    hidden: torch.Tensor,
    banks: list[Weights],
    operations: int,
) -> torch.cuda.CUDAGraph:
    operation(hidden, banks[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(operations):
            operation(hidden, banks[index % len(banks)])
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
            args.weight_sets,
            args.operations_per_graph,
            args.warmup_replays,
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

    control = load_control()
    hidden, banks = make_inputs(args.seed, args.weight_sets)
    errors = check_correctness(control, hidden, banks[0])
    graphs = {
        "control": capture(control, hidden, banks, args.operations_per_graph),
        "candidate": capture(candidate, hidden, banks, args.operations_per_graph),
    }
    for index in range(args.warmup_replays * len(graphs)):
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
        tensor.numel() * tensor.element_size() for tensor in banks[0]
    ) * len(banks)
    result = {
        "shape": "B1 Kimi-K3 MoE pre-route BF16",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
        },
        "seed": args.seed,
        "weight_sets": args.weight_sets,
        "rotating_weight_bytes": rotating_weight_bytes,
        "cache_valid_rotation": rotating_weight_bytes > 256 * 1024 * 1024,
        "operations_per_graph": args.operations_per_graph,
        "warmup_replays": args.warmup_replays,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "relative_rmse": errors,
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
