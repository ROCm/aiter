# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Matched GPT-OSS MoE graph benchmark: dynamic A8W4 versus IQ2R.

Both timed paths begin with the same BF16 hidden states and FP32 router logits
and end with a BF16 token output. The timed graph includes top-k/routing,
dispatch metadata, dynamic per-1x32 MXFP8 activation packing, gate/up GEMM,
GPT-OSS SwiGLU, dynamic MXFP8 intermediate packing, down GEMM, and weighted
route reduction. The common BF16 router projection that produces the logits is
outside both paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

# fused_moe_triton conditionally imports its AITER kernels at module import.
os.environ.setdefault("ATOM_USE_TRITON_MOE", "1")

import torch

from atom.model_ops.fused_moe_triton import triton_kernel_moe_forward
from atom.model_ops.moe import MoEActivationQuant
from aiter import ActivationType
from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.moe_op import topk_softmax

from bench_iq2r_gpt_oss import _load_mxfp4_baseline, _relative_metrics, _routes

DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
EXPERTS = 128
TOPK = 4
HIDDEN = 2880


def _load_launch_choices(
    path: Path | None,
) -> dict[tuple[int, str, str], tuple[str, int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    choices: dict[tuple[int, str, str], tuple[str, int]] = {}
    for row in payload["aggregate"]:
        key = (int(row["M"]), str(row["routing"]), str(row["projection"]))
        candidate = (str(row["family"]), int(row["grid_multiplier"]))
        previous = choices.get(key)
        if previous is None:
            choices[key] = candidate
            continue
        previous_row = next(
            item
            for item in payload["aggregate"]
            if (int(item["M"]), str(item["routing"]), str(item["projection"])) == key
            and (str(item["family"]), int(item["grid_multiplier"])) == previous
        )
        if float(row["median_latency_ms"]) < float(previous_row["median_latency_ms"]):
            choices[key] = candidate
    return choices


def _apply_launch_choices(
    choices: dict[tuple[int, str, str], tuple[str, int]], tokens: int, routing: str
) -> dict[str, object]:
    selected: dict[str, object] = {}
    for projection, prefix in (
        ("gate_up", "IQ2R_GEMM_GATE_UP"),
        ("down", "IQ2R_GEMM_DOWN"),
    ):
        choice = choices.get((tokens, routing, projection))
        if choice is None:
            os.environ.pop(f"{prefix}_FAMILY", None)
            os.environ.pop(f"{prefix}_GRID_MULTIPLIER", None)
            selected[projection] = "production-default"
        else:
            family, grid = choice
            os.environ[f"{prefix}_FAMILY"] = family
            os.environ[f"{prefix}_GRID_MULTIPLIER"] = str(grid)
            selected[projection] = {"family": family, "grid_multiplier": grid}
    return selected


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("this GPT-OSS qualification requires gfx950")


def _capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    torch.cuda.synchronize()
    return graph, output


def _sample(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark_pair(
    iq2r_graph: torch.cuda.CUDAGraph,
    a8w4_graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        iq2r_graph.replay()
        a8w4_graph.replay()
    torch.cuda.synchronize()

    iq2r_samples: list[float] = []
    a8w4_samples: list[float] = []
    for sample in range(samples):
        if sample % 2 == 0:
            iq2r_samples.append(_sample(iq2r_graph, iterations))
            a8w4_samples.append(_sample(a8w4_graph, iterations))
        else:
            a8w4_samples.append(_sample(a8w4_graph, iterations))
            iq2r_samples.append(_sample(iq2r_graph, iterations))
    return iq2r_samples, a8w4_samples


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    device = torch.device("cuda")
    iq2r = load_iq2r_layer_checkpoint(args.iq2r_checkpoint, args.layer, device=device)
    if iq2r.expert_count != EXPERTS or iq2r.expert_start != 0:
        raise ValueError("benchmark requires the full 128-expert IQ2R layer")
    a8w4 = _load_mxfp4_baseline(args.model_dir, args.layer, device)
    launch_choices = _load_launch_choices(args.iq2r_launch_sweep)

    records: list[dict[str, object]] = []
    for pattern in args.routing:
        for tokens in args.m:
            selected_launches = _apply_launch_choices(launch_choices, tokens, pattern)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + tokens * 1009 + sum(map(ord, pattern))
            )
            hidden = torch.randn(
                (tokens, HIDDEN),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            expected_ids, _expected_weights, logits = _routes(tokens, pattern, device)

            workspace = IQ2RMoeWorkspace.allocate(
                tokens, TOPK, device=device, task_rows=args.task_rows
            )
            iq2r_output = torch.empty_like(hidden)
            iq2r_ids = torch.empty((tokens, TOPK), dtype=torch.int32, device=device)
            iq2r_weights = torch.empty(
                (tokens, TOPK), dtype=torch.float32, device=device
            )
            token_expert_indices = torch.empty_like(iq2r_ids)

            def iq2r_call() -> torch.Tensor:
                topk_softmax(
                    iq2r_weights,
                    iq2r_ids,
                    token_expert_indices,
                    logits,
                    True,
                )
                iq2r_fused_moe_out(
                    hidden,
                    iq2r.gate_up_data,
                    iq2r.gate_up_auxiliary,
                    iq2r.down_data,
                    iq2r.down_auxiliary,
                    iq2r_weights,
                    iq2r_ids,
                    iq2r_output,
                    gate_up_metadata=iq2r.gate_up_metadata,
                    down_metadata=iq2r.down_metadata,
                    gate_up_tile_n=iq2r.gate_up_tile_n,
                    down_tile_n=iq2r.down_tile_n,
                    gate_up_bias=iq2r.gate_up_bias,
                    down_bias=iq2r.down_bias,
                    workspace=workspace,
                )
                return iq2r_output

            def a8w4_call() -> torch.Tensor:
                return triton_kernel_moe_forward(
                    hidden,
                    a8w4["gate_weight"],
                    a8w4["down_weight"],
                    logits,
                    topk=TOPK,
                    renormalize=True,
                    activation=ActivationType.Swiglu,
                    w13_scale=a8w4["gate_scale"],
                    w2_scale=a8w4["down_scale"],
                    w13_swizzle_layout=a8w4["gate_layout"],
                    w2_swizzle_layout=a8w4["down_layout"],
                    w1_bias=a8w4["gate_bias"],
                    w2_bias=a8w4["down_bias"],
                    swiglu_limit=7.0,
                    global_num_experts=EXPERTS,
                    act_quant=MoEActivationQuant.FP8,
                )

            iq2r_graph, iq2r_result = _capture(iq2r_call)
            a8w4_graph, a8w4_result = _capture(a8w4_call)
            iq2r_samples, a8w4_samples = _benchmark_pair(
                iq2r_graph,
                a8w4_graph,
                warmup=args.warmup,
                iterations=args.iterations,
                samples=args.samples,
            )
            iq2r_graph.replay()
            a8w4_graph.replay()
            torch.cuda.synchronize()

            iq2r_ms = median(iq2r_samples)
            a8w4_ms = median(a8w4_samples)
            record = {
                "M": tokens,
                "routing": pattern,
                "task_rows": args.task_rows,
                "iq2r_launches": selected_launches,
                "iq2r_ms": iq2r_ms,
                "iq2r_samples_ms": iq2r_samples,
                "a8w4_dynamic_mxfp8_ms": a8w4_ms,
                "a8w4_samples_ms": a8w4_samples,
                "speedup_a8w4_over_iq2r": a8w4_ms / iq2r_ms,
                "latency_reduction_percent": (a8w4_ms - iq2r_ms) / a8w4_ms * 100.0,
                "iq2r_selected_expected_experts": torch.equal(
                    torch.sort(iq2r_ids, dim=1).values,
                    torch.sort(expected_ids, dim=1).values,
                ),
                **_relative_metrics(iq2r_result, a8w4_result),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    summary: list[dict[str, object]] = []
    for tokens in args.m:
        selected = [row for row in records if row["M"] == tokens]
        summary.append(
            {
                "M": tokens,
                "patterns": len(selected),
                "iq2r_median_ms": median(float(row["iq2r_ms"]) for row in selected),
                "a8w4_dynamic_median_ms": median(
                    float(row["a8w4_dynamic_mxfp8_ms"]) for row in selected
                ),
                "median_speedup": median(
                    float(row["speedup_a8w4_over_iq2r"]) for row in selected
                ),
                "min_speedup": min(
                    float(row["speedup_a8w4_over_iq2r"]) for row in selected
                ),
                "max_speedup": max(
                    float(row["speedup_a8w4_over_iq2r"]) for row in selected
                ),
            }
        )

    return {
        "benchmark": "gpt-oss-tp1-full-moe-graph-iq2r-vs-dynamic-mxfp8-mxfp4",
        "gpu": torch.cuda.get_device_name(0),
        "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "device_binding": "HIP_VISIBLE_DEVICES=6",
        "layer": args.layer,
        "timed_boundary": (
            "FP32 router logits + BF16 hidden input through top-k/routing, "
            "dynamic MXFP8 packing, both expert GEMMs, SwiGLU, and reduction"
        ),
        "a8w4_activation": "dynamic MXFP8 E4M3 values with E8M0 per-1x32 scales",
        "iq2r_launch_selection": (
            "per-M/per-routing oracle from launch sweep"
            if args.iq2r_launch_sweep is not None
            else "production default"
        ),
        "records": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iq2r-checkpoint",
        type=Path,
        default=Path("/models/openai/gpt-oss-120b-o0-e132-profile"),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/models/openai/gpt-oss-120b")
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_M_VALUES)
    parser.add_argument(
        "--routing",
        choices=("uniform", "hot", "single-group"),
        nargs="+",
        default=("uniform", "hot", "single-group"),
    )
    parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument(
        "--iq2r-launch-sweep",
        type=Path,
        help="optional launch-sweep JSON; uses the best candidate per M/routing",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0xA8F2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
