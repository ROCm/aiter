# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the Kimi-K3 B1 KDA input projection on real model weights."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from aiter.ops.flydsl.kimi_k3_kda_input_group64 import (
    kimi_k3_kda_input_group64,
    quantize_kimi_k3_kda_input_group64,
)

HIDDEN = 7168
PADDED_OUTPUT = 6288
LOGICAL_OUTPUT = 6284
GROUP = 64
ROTATIONS = 8

Case = dict[str, torch.Tensor]
Operation = Callable[[Case], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="directory containing the Kimi-K3 safetensors checkpoint",
    )
    parser.add_argument("--tp-rank", type=int, default=0, choices=range(8))
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--operations-per-graph", type=int, default=96)
    parser.add_argument("--warmup-operations", type=int, default=100_000)
    parser.add_argument("--replays-per-trial", type=int, default=10)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def checkpoint_tensor(
    model: Path, weight_map: dict[str, str], name: str, selection=None
):
    with safe_open(model / weight_map[name], framework="pt", device="cpu") as source:
        tensor = source.get_slice(name)
        return tensor[:] if selection is None else tensor[selection]


def load_real_case(model: Path, tp_rank: int) -> Case:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = "language_model.model.layers.0"
    row_start = tp_rank * 1536
    row_end = row_start + 1536
    parts = [
        checkpoint_tensor(
            model,
            weight_map,
            f"{prefix}.self_attn.{name}_proj.weight",
            (slice(row_start, row_end), slice(None)),
        )
        for name in ("q", "k", "v", "g")
    ]
    parts.append(
        checkpoint_tensor(model, weight_map, f"{prefix}.self_attn.f_a_proj.weight")
    )
    beta_start = tp_rank * 12
    parts.append(
        checkpoint_tensor(
            model,
            weight_map,
            f"{prefix}.self_attn.b_proj.weight",
            (slice(beta_start, beta_start + 12), slice(None)),
        )
    )
    parts.append(torch.zeros((4, HIDDEN), dtype=torch.bfloat16))
    weight = torch.cat(parts).contiguous().cuda()

    hidden = checkpoint_tensor(
        model,
        weight_map,
        "language_model.model.embed_tokens.weight",
        (slice(91561, 91562), slice(None)),
    )
    norm = checkpoint_tensor(model, weight_map, f"{prefix}.input_layernorm.weight")
    hidden_f32 = hidden.float()
    hidden = (
        hidden_f32
        * torch.rsqrt(hidden_f32.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * norm.float()
    ).to(device="cuda", dtype=torch.bfloat16)
    packed, scale = quantize_kimi_k3_kda_input_group64(weight)
    return {"hidden": hidden, "bf16": weight, "packed": packed, "scale": scale}


def make_rotations(seed: int) -> list[Case]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = []
    for _ in range(ROTATIONS):
        hidden = torch.randn(
            (1, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            (PADDED_OUTPUT, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(HIDDEN**-0.5)
        weight[LOGICAL_OUTPUT:].zero_()
        packed, scale = quantize_kimi_k3_kda_input_group64(weight)
        result.append(
            {"hidden": hidden, "bf16": weight, "packed": packed, "scale": scale}
        )
    return result


def load_control() -> Operation:
    from vllm import _custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    def control(case: Case):
        return ops.wvSplitK(case["bf16"], case["hidden"], num_compute_units(), None)

    return control


def candidate(case: Case):
    return kimi_k3_kda_input_group64(case["hidden"], case["packed"], case["scale"])


def correctness(case: Case) -> dict[str, float]:
    dequantized = (
        case["packed"].float().reshape(LOGICAL_OUTPUT, HIDDEN // GROUP, GROUP)
        * case["scale"].reshape(LOGICAL_OUTPUT, HIDDEN // GROUP, 1)
    ).reshape(LOGICAL_OUTPUT, HIDDEN)
    actual = candidate(case)
    expected = torch.zeros((1, PADDED_OUTPUT), device="cuda", dtype=torch.bfloat16)
    expected[:, :LOGICAL_OUTPUT] = (case["hidden"].float() @ dequantized.float().T).to(
        torch.bfloat16
    )
    torch.cuda.synchronize()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-8)
    relative_rmse = (error / scale).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    if relative_rmse > 5.0e-4 or cosine < 0.99999:
        raise AssertionError(
            f"correctness gate failed: relative_rmse={relative_rmse}, cosine={cosine}"
        )
    return {"relative_rmse": relative_rmse, "cosine_similarity": cosine}


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
    counts = (
        args.operations_per_graph,
        args.warmup_operations,
        args.replays_per_trial,
        args.trials,
    )
    if min(counts) <= 0:
        raise ValueError("all benchmark counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not torch.version.hip or arch != "gfx950":
        raise RuntimeError(f"this benchmark requires ROCm gfx950, got {arch!r}")

    real_case = load_real_case(args.model, args.tp_rank)
    accuracy = correctness(real_case)
    rotations = make_rotations(args.seed)
    operations = {"control": load_control(), "candidate": candidate}
    graphs = {
        name: capture(operation, rotations, args.operations_per_graph)
        for name, operation in operations.items()
    }
    warmup_replays = math.ceil(args.warmup_operations / args.operations_per_graph)
    for index in range(warmup_replays * len(graphs)):
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
        tensor.numel() * tensor.element_size()
        for name, tensor in rotations[0].items()
        if name != "hidden"
    ) * len(rotations)
    result = {
        "shape": "M1 N6288/6284 K7168 Kimi-K3 KDA input",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
        },
        "tp_rank": args.tp_rank,
        "seed": args.seed,
        "timed_input_provenance": "deterministic synthetic rotating weights",
        "rotations": ROTATIONS,
        "rotating_weight_bytes": rotating_weight_bytes,
        "cache_valid_rotation": rotating_weight_bytes > 256 * 1024 * 1024,
        "operations_per_graph": args.operations_per_graph,
        "warmup_operations": args.warmup_operations,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "accuracy_input_provenance": "mounted Kimi-K3 checkpoint",
        "real_weight_accuracy": accuracy,
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
