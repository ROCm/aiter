# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare FP8 pre-route kernels with the BF16 parent from AITER PR 4498."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
    kimi_k3_moe_dual_projection_fp8,
    kimi_k3_moe_tri_projection_fp8,
    kimi_k3_shared_down_fp8,
)

HIDDEN = 7168
ROUTED = 3584
SHARED_GATE_UP = 1536
SHARED_INTERMEDIATE = 768
ROUTER = 896
FP8_MAX = 448.0


@dataclass
class Case:
    hidden: torch.Tensor
    routed_bf16: torch.Tensor
    shared_up_bf16: torch.Tensor
    shared_down_bf16: torch.Tensor
    router_bf16: torch.Tensor
    routed_fp8: torch.Tensor
    routed_scale: torch.Tensor
    shared_up_fp8: torch.Tensor
    shared_up_scale: torch.Tensor
    shared_down_fp8: torch.Tensor
    shared_down_scale: torch.Tensor


Operation = Callable[[Case], tuple[torch.Tensor, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--weight-sets", type=int, default=8)
    parser.add_argument("--operations-per-graph", type=int, default=96)
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--replays-per-trial", type=int, default=10)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def quantize_rows(weight: torch.Tensor):
    source = weight.float()
    amax = source.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    packed = (
        (source / scale[:, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return packed, scale.contiguous()


def make_cases(seed: int, count: int) -> list[Case]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = []
    for _ in range(count):
        hidden = torch.randn(
            (1, HIDDEN), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        routed = torch.randn(
            (ROUTED, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(HIDDEN**-0.5)
        shared_up = torch.randn(
            (SHARED_GATE_UP, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(HIDDEN**-0.5)
        shared_down = torch.randn(
            (HIDDEN, SHARED_INTERMEDIATE),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(SHARED_INTERMEDIATE**-0.5)
        router = torch.randn(
            (ROUTER, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        routed_fp8, routed_scale = quantize_rows(routed)
        shared_up_fp8, shared_up_scale = quantize_rows(shared_up)
        shared_down_fp8, shared_down_scale = quantize_rows(shared_down)
        result.append(
            Case(
                hidden,
                routed,
                shared_up,
                shared_down,
                router,
                routed_fp8,
                routed_scale,
                shared_up_fp8,
                shared_up_scale,
                shared_down_fp8,
                shared_down_scale,
            )
        )
    return result


def load_parent() -> Operation:
    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
            kimi_k3_moe_preroute_bf16,
        )
    except ImportError as error:
        raise RuntimeError(
            "the BF16 control is AITER PR 4498; apply that PR before running this benchmark"
        ) from error

    def parent(case: Case):
        return kimi_k3_moe_preroute_bf16(
            case.hidden,
            case.routed_bf16,
            case.shared_up_bf16,
            case.shared_down_bf16,
        )

    return parent


def load_tri_parent() -> Operation:
    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
            _compiled_dual_projection,
        )
    except ImportError as error:
        raise RuntimeError(
            "the BF16 control is AITER PR 4498; apply that PR before running this benchmark"
        ) from error

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    def parent(case: Case):
        routed = case.hidden.new_empty((1, ROUTED))
        shared_gate_up = case.hidden.new_empty((1, SHARED_GATE_UP))
        _compiled_dual_projection()(
            ptr_arg(case.hidden),
            ptr_arg(case.routed_bf16),
            ptr_arg(case.shared_up_bf16),
            ptr_arg(routed),
            ptr_arg(shared_gate_up),
            stream=torch.cuda.current_stream(case.hidden.device),
        )
        router_logits = rocm_unquantized_gemm_impl(
            case.hidden, case.router_bf16
        ).float()
        return routed, shared_gate_up, router_logits

    return parent


def candidate(case: Case):
    routed, gate_up = kimi_k3_moe_dual_projection_fp8(
        case.hidden,
        case.routed_fp8,
        case.routed_scale,
        case.shared_up_fp8,
        case.shared_up_scale,
    )
    shared = kimi_k3_shared_down_fp8(
        gate_up,
        case.shared_down_fp8,
        case.shared_down_scale,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    return routed, shared


def tri_candidate(case: Case):
    return kimi_k3_moe_tri_projection_fp8(
        case.hidden,
        case.routed_fp8,
        case.routed_scale,
        case.shared_up_fp8,
        case.shared_up_scale,
        case.router_bf16,
    )


def metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = actual_f32 - expected_f32
    scale = expected_f32.square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "relative_rmse": (delta.square().mean().sqrt() / scale).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            actual_f32.flatten(), expected_f32.flatten(), dim=0
        ).item(),
    }


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


def measure_pair(
    parent: Operation,
    fp8_candidate: Operation,
    cases: list[Case],
    args: argparse.Namespace,
):
    graphs = {
        "bf16_parent": capture(parent, cases, args.operations_per_graph),
        "fp8_candidate": capture(fp8_candidate, cases, args.operations_per_graph),
    }
    for index in range(args.warmup_replays * len(graphs)):
        name = ("bf16_parent", "fp8_candidate")[index % 2]
        graphs[name].replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    for trial in range(args.trials):
        order = (
            ("bf16_parent", "fp8_candidate")
            if trial % 2 == 0
            else ("fp8_candidate", "bf16_parent")
        )
        for name in order:
            samples[name].append(
                elapsed_us(
                    graphs[name], args.replays_per_trial, args.operations_per_graph
                )
            )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "p50_us": medians,
        "speedup": medians["bf16_parent"] / medians["fp8_candidate"],
        "samples_us": samples,
    }


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
    if args.operations_per_graph % args.weight_sets:
        raise ValueError("operations per graph must be divisible by weight sets")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not torch.version.hip or arch != "gfx950":
        raise RuntimeError(f"this benchmark requires ROCm gfx950, got {arch!r}")

    cases = make_cases(args.seed, args.weight_sets)
    parent = load_parent()
    tri_parent = load_tri_parent()
    expected = parent(cases[0])
    actual = candidate(cases[0])
    torch.cuda.synchronize()
    dual_shared_accuracy = {
        name: metrics(got, want)
        for name, got, want in zip(("routed", "shared"), actual, expected)
    }
    if (
        dual_shared_accuracy["routed"]["relative_rmse"] >= 0.035
        or dual_shared_accuracy["shared"]["relative_rmse"] >= 0.06
        or dual_shared_accuracy["routed"]["cosine_similarity"] <= 0.999
        or dual_shared_accuracy["shared"]["cosine_similarity"] <= 0.998
    ):
        raise AssertionError(
            f"dual/shared accuracy gate failed: {dual_shared_accuracy}"
        )

    tri_expected = tri_parent(cases[0])
    tri_actual = tri_candidate(cases[0])
    torch.cuda.synchronize()
    tri_accuracy = {
        name: metrics(got, want)
        for name, got, want in zip(
            ("routed", "shared_gate_up", "router_logits"),
            tri_actual,
            tri_expected,
        )
    }
    reference_topk = tri_expected[2].topk(17, dim=-1)
    candidate_topk = tri_actual[2].topk(16, dim=-1)
    reference_boundary = (
        reference_topk.values[0, 15].item(),
        reference_topk.values[0, 16].item(),
    )
    membership_match = torch.equal(
        candidate_topk.indices.sort(dim=-1).values,
        reference_topk.indices[:, :16].sort(dim=-1).values,
    )
    if (
        tri_accuracy["routed"]["relative_rmse"] >= 0.035
        or tri_accuracy["shared_gate_up"]["relative_rmse"] >= 0.035
        or tri_accuracy["router_logits"]["relative_rmse"] >= 0.01
        or reference_boundary[0] <= reference_boundary[1]
        or not membership_match
    ):
        raise AssertionError(
            "tri-projection accuracy gate failed: "
            f"metrics={tri_accuracy}, boundary={reference_boundary}, "
            f"membership_match={membership_match}"
        )

    dual_shared_result = measure_pair(parent, candidate, cases, args)
    tri_result = measure_pair(tri_parent, tri_candidate, cases, args)
    rotating_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for field, tensor in vars(cases[0]).items()
        if field != "hidden"
    ) * len(cases)
    result = {
        "shape": "B1 Kimi-K3 MoE pre-route FP8 vs BF16 parent",
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
        "boundaries": {
            "dual_projection_plus_shared_down": {
                "accuracy_vs_bf16_parent": dual_shared_accuracy,
                **dual_shared_result,
            },
            "dual_projection_plus_router": {
                "accuracy_vs_bf16_parent": tri_accuracy,
                "reference_top16_boundary": {
                    "rank_16": reference_boundary[0],
                    "rank_17": reference_boundary[1],
                    "strict": reference_boundary[0] > reference_boundary[1],
                },
                "top16_membership_match": membership_match,
                **tri_result,
            },
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
