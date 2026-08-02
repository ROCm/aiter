# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark separate versus fused f_b projection in Kimi-K3 KDA decode."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from aiter.ops.flydsl.kimi_k3_kda_decode import (
    flydsl_kimi_k3_kda_decode,
    flydsl_kimi_k3_kda_decode_with_f_b,
)

HEADS = 12
DIM = 128
CHANNELS = 3 * HEADS * DIM
CONV_WIDTH = 4
LOWER_BOUND = -5.0
NORM_EPS = 1.0e-5
ROTATIONS = 8


@dataclass
class Case:
    f_a: torch.Tensor
    f_b_weight: torch.Tensor
    x: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor
    out: torch.Tensor


Operation = Callable[[Case], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--operations-per-graph", type=int, default=100)
    parser.add_argument("--warmup-replays", type=int, default=100)
    parser.add_argument("--replays-per-trial", type=int, default=10)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_case(generator: torch.Generator) -> Case:
    slots = 3
    return Case(
        f_a=torch.randn(
            (1, DIM), generator=generator, device="cuda", dtype=torch.bfloat16
        ),
        f_b_weight=torch.randn(
            (HEADS, DIM, DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(0.05),
        x=torch.randn(
            (1, CHANNELS),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        conv_weight=torch.randn(
            (CHANNELS, CONV_WIDTH),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ).mul_(0.1),
        conv_state=torch.randn(
            (slots, CHANNELS, CONV_WIDTH - 1),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        raw_beta=torch.randn(
            (1, 1, HEADS),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        A_log=torch.randn(
            (HEADS,), generator=generator, device="cuda", dtype=torch.float32
        ).mul_(0.5),
        dt_bias=torch.randn(
            (HEADS * DIM,),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ).mul_(0.1),
        state=torch.randn(
            (slots, HEADS, DIM, DIM),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ).mul_(0.01),
        state_indices=torch.tensor([1], device="cuda", dtype=torch.int32),
        output_gate=torch.randn(
            (1, HEADS, DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        norm_weight=torch.randn(
            (DIM,), generator=generator, device="cuda", dtype=torch.bfloat16
        ),
        out=torch.empty((1, 1, HEADS, DIM), device="cuda", dtype=torch.bfloat16),
    )


def clone_case(case: Case) -> Case:
    return Case(
        **{name: getattr(case, name).clone() for name in Case.__dataclass_fields__}
    )


def control(case: Case):
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    raw_g = rocm_unquantized_gemm_impl(case.f_a, case.f_b_weight.view(HEADS * DIM, DIM))
    return flydsl_kimi_k3_kda_decode(
        case.x,
        case.conv_weight,
        None,
        case.conv_state,
        raw_g.view(1, 1, HEADS, DIM),
        case.raw_beta,
        case.A_log,
        case.dt_bias,
        LOWER_BOUND,
        case.state,
        case.state_indices,
        case.output_gate,
        case.norm_weight,
        NORM_EPS,
        case.out,
    )


def candidate(case: Case):
    return flydsl_kimi_k3_kda_decode_with_f_b(
        case.f_a,
        case.f_b_weight,
        case.x,
        case.conv_weight,
        None,
        case.conv_state,
        case.raw_beta,
        case.A_log,
        case.dt_bias,
        LOWER_BOUND,
        case.state,
        case.state_indices,
        case.output_gate,
        case.norm_weight,
        NORM_EPS,
        case.out,
    )


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-8)
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


def elapsed_us(graph, replays: int, operations: int):
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

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    seed_case = make_case(generator)
    control_case = clone_case(seed_case)
    candidate_case = clone_case(seed_case)
    expected = control(control_case).clone()
    actual = candidate(candidate_case).clone()
    torch.cuda.synchronize()
    output_error = relative_rmse(actual, expected)
    state_error = relative_rmse(candidate_case.state, control_case.state)
    if max(output_error, state_error) >= 1.0e-3:
        raise AssertionError(
            f"correctness gate failed: output={output_error}, state={state_error}"
        )

    control_cases = [make_case(generator) for _ in range(ROTATIONS)]
    candidate_cases = [clone_case(case) for case in control_cases]
    graphs = {
        "separate": capture(control, control_cases, args.operations_per_graph),
        "fused": capture(candidate, candidate_cases, args.operations_per_graph),
    }
    for index in range(args.warmup_replays * len(graphs)):
        graphs[("separate", "fused")[index % 2]].replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    for trial in range(args.trials):
        order = ("separate", "fused") if trial % 2 == 0 else ("fused", "separate")
        for name in order:
            samples[name].append(
                elapsed_us(
                    graphs[name], args.replays_per_trial, args.operations_per_graph
                )
            )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    rotating_weight_bytes = sum(
        case.f_b_weight.numel() * case.f_b_weight.element_size()
        for case in control_cases
    )
    result = {
        "shape": "B1 Kimi-K3 TP8-local KDA decode plus f_b",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
        },
        "seed": args.seed,
        "rotations": ROTATIONS,
        "rotating_f_b_weight_bytes": rotating_weight_bytes,
        "operations_per_graph": args.operations_per_graph,
        "warmup_replays": args.warmup_replays,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "relative_rmse": {"output": output_error, "state": state_error},
        "p50_us": medians,
        "speedup": medians["separate"] / medians["fused"],
        "samples_us": samples,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
