# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measure IQ2R MoE stage boundaries on captured GPT-OSS routes.

The benchmark uses fixed-address output buffers and HIP graph replay.  It can
also reproduce GPT-OSS's padded 3072-element hidden-state stride while keeping
the logical hidden width at 2880.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Callable

import torch

from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.iq2r_moe import (
    IQ2R_DIRECT_ROUTE_MAX,
    IQ2RMoeWorkspace,
    iq2r_fused_moe_out,
)
from aiter.ops.iq2r import (
    iq2r_route_direct_gather_quant_out,
    iq2r_route_gather_quant_out,
    iq2r_route_reduce_indexed_out,
    iq2r_route_sort_tasks_out,
    iq2r_route_topk_direct_gather_quant_out,
    iq2r_route_topk_sort_gather_quant_out,
    iq2r_swiglu_quant_out,
    iq2r_task_gemm_out,
)
from aiter.ops.moe_op import topk_softmax

EXPERTS = 128
TOPK = 4
HIDDEN = 2880


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("captured-route qualification requires gfx950")


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _benchmark(
    call: Callable[[], None], warmup: int, iterations: int, samples: int
) -> tuple[float, list[float]]:
    graph = _capture(call)
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


def _select_files(root: Path, tokens: int, limit: int) -> list[Path]:
    files = sorted(root.glob(f"*-m{tokens}-*.pt"))
    if not files:
        raise FileNotFoundError(f"no m={tokens} route captures under {root}")
    if len(files) <= limit:
        return files
    if limit == 1:
        return [files[0]]
    indices = [round(i * (len(files) - 1) / (limit - 1)) for i in range(limit)]
    return [files[index] for index in indices]


def _logits_from_capture(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    logits = torch.full(
        (topk_ids.shape[0], EXPERTS),
        -30.0,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    logits.scatter_(1, topk_ids.to(torch.int64), topk_weights.clamp_min(1e-12).log())
    return logits


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    if args.input_stride < HIDDEN:
        raise ValueError(f"input stride must be at least {HIDDEN}")

    device = torch.device("cuda")
    if args.gate_up_family:
        os.environ["IQ2R_GEMM_GATE_UP_FAMILY"] = args.gate_up_family
    if args.down_family:
        os.environ["IQ2R_GEMM_DOWN_FAMILY"] = args.down_family
    checkpoint = load_iq2r_layer_checkpoint(
        args.iq2r_checkpoint, args.layer, device=device
    )
    records: list[dict[str, object]] = []

    for tokens in args.m:
        routes = tokens * TOPK
        task_rows = 32 if routes == 4096 else args.task_rows
        workspace = IQ2RMoeWorkspace.allocate(
            tokens,
            TOPK,
            device=device,
            task_rows=args.task_rows,
            scale_layout=args.scale_layout,
        )
        task_capacity = workspace.tasks.shape[0]
        tasks = workspace.tasks[:task_capacity]
        sorted_ids = workspace.sorted_expert_ids[:routes]
        gather = workspace.gather_indices[:routes]
        scatter = workspace.scatter_indices[:routes]
        route_input_fp8 = workspace.route_input_fp8[:routes]
        route_input_scales = (
            workspace.route_input_scales[:routes]
            if args.scale_layout == "row_major"
            else workspace.route_input_scales
        )
        gate_up = workspace.gate_up[:routes]
        intermediate_fp8 = workspace.intermediate_fp8[:routes]
        intermediate_scales = (
            workspace.intermediate_scales[:routes]
            if args.scale_layout == "row_major"
            else workspace.intermediate_scales
        )
        route_output = workspace.route_output[:routes]
        output = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        selected_ids = torch.empty((tokens, TOPK), dtype=torch.int32, device=device)
        selected_weights = torch.empty(
            (tokens, TOPK), dtype=torch.float32, device=device
        )
        token_expert_indices = torch.empty_like(selected_ids)
        fused_selected_ids = workspace.topk_ids[:tokens]
        fused_selected_weights = workspace.topk_weights[:tokens]

        for index, route_file in enumerate(
            _select_files(args.route_dir, tokens, args.max_route_files)
        ):
            capture = torch.load(route_file, map_location="cpu", weights_only=True)
            captured_ids = capture["topk_ids"].to(device=device, dtype=torch.int32)
            captured_weights = capture["topk_weights"].to(
                device=device, dtype=torch.float32
            )
            logits = _logits_from_capture(captured_ids, captured_weights)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + tokens * 1009 + index
            )
            hidden_storage = torch.randn(
                (tokens, args.input_stride),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            hidden = hidden_storage[:, :HIDDEN]
            direct_routes = routes <= IQ2R_DIRECT_ROUTE_MAX

            def topk_call() -> None:
                topk_softmax(
                    selected_weights,
                    selected_ids,
                    token_expert_indices,
                    logits,
                    True,
                )

            def route_call() -> None:
                if direct_routes:
                    iq2r_route_direct_gather_quant_out(
                        hidden,
                        selected_ids.reshape(-1),
                        sorted_ids,
                        gather,
                        scatter,
                        tasks,
                        workspace.task_count,
                        route_input_fp8,
                        route_input_scales,
                        topk=TOPK,
                        expert_count=EXPERTS,
                    )
                else:
                    iq2r_route_sort_tasks_out(
                        selected_ids.reshape(-1),
                        sorted_ids,
                        gather,
                        scatter,
                        tasks,
                        workspace.task_count,
                        expert_count=EXPERTS,
                        task_rows=task_rows,
                    )
                    iq2r_route_gather_quant_out(
                        hidden,
                        gather,
                        route_input_fp8,
                        route_input_scales,
                        topk=TOPK,
                    )

            def fused_frontend_call() -> None:
                if direct_routes:
                    iq2r_route_topk_direct_gather_quant_out(
                        hidden,
                        logits,
                        fused_selected_weights,
                        fused_selected_ids,
                        sorted_ids,
                        gather,
                        scatter,
                        tasks,
                        workspace.task_count,
                        route_input_fp8,
                        route_input_scales,
                        renormalize=True,
                    )
                else:
                    iq2r_route_topk_sort_gather_quant_out(
                        hidden,
                        logits,
                        fused_selected_weights,
                        fused_selected_ids,
                        sorted_ids,
                        gather,
                        scatter,
                        tasks,
                        workspace.task_count,
                        route_input_fp8,
                        route_input_scales,
                        task_rows=task_rows,
                        renormalize=True,
                    )

            def gate_up_call() -> None:
                iq2r_task_gemm_out(
                    route_input_fp8,
                    route_input_scales,
                    checkpoint.gate_up_data,
                    checkpoint.gate_up_auxiliary,
                    tasks,
                    workspace.task_count,
                    checkpoint.gate_up_metadata,
                    gate_up,
                    tile_n=checkpoint.gate_up_tile_n,
                    bias=checkpoint.gate_up_bias,
                )

            def swiglu_call() -> None:
                iq2r_swiglu_quant_out(
                    gate_up,
                    intermediate_fp8,
                    intermediate_scales,
                )

            def down_call() -> None:
                iq2r_task_gemm_out(
                    intermediate_fp8,
                    intermediate_scales,
                    checkpoint.down_data,
                    checkpoint.down_auxiliary,
                    tasks,
                    workspace.task_count,
                    checkpoint.down_metadata,
                    route_output,
                    tile_n=checkpoint.down_tile_n,
                    bias=checkpoint.down_bias,
                )

            def reduce_call() -> None:
                iq2r_route_reduce_indexed_out(
                    route_output,
                    selected_weights,
                    scatter,
                    output,
                    topk=TOPK,
                )

            def experts_call() -> None:
                gate_up_call()
                swiglu_call()
                down_call()

            def post_topk_call() -> None:
                route_call()
                experts_call()
                reduce_call()

            def full_call() -> None:
                topk_call()
                post_topk_call()

            def fused_full_call() -> None:
                iq2r_fused_moe_out(
                    hidden,
                    checkpoint.gate_up_data,
                    checkpoint.gate_up_auxiliary,
                    checkpoint.down_data,
                    checkpoint.down_auxiliary,
                    fused_selected_weights,
                    fused_selected_ids,
                    output,
                    gate_up_metadata=checkpoint.gate_up_metadata,
                    down_metadata=checkpoint.down_metadata,
                    gate_up_tile_n=checkpoint.gate_up_tile_n,
                    down_tile_n=checkpoint.down_tile_n,
                    gate_up_bias=checkpoint.gate_up_bias,
                    down_bias=checkpoint.down_bias,
                    workspace=workspace,
                    router_logits=logits,
                    renormalize=True,
                )

            topk_call()
            route_call()
            experts_call()
            reduce_call()
            torch.cuda.synchronize()

            timings: dict[str, float | list[float]] = {}
            for name, call in (
                ("topk", topk_call),
                ("route", route_call),
                ("gate_up", gate_up_call),
                ("swiglu_quant", swiglu_call),
                ("down", down_call),
                ("reduce", reduce_call),
                ("experts", experts_call),
                ("post_topk", post_topk_call),
                ("full", full_call),
                ("fused_frontend", fused_frontend_call),
                ("fused_full", fused_full_call),
            ):
                latency, samples = _benchmark(
                    call, args.warmup, args.iterations, args.samples
                )
                timings[f"{name}_ms"] = latency
                timings[f"{name}_samples_ms"] = samples

            records.append(
                {
                    "M": tokens,
                    "routes": routes,
                    "route_file": route_file.name,
                    "input_stride": hidden.stride(0),
                    "task_rows": task_rows,
                    "scale_layout": args.scale_layout,
                    "gate_up_family": args.gate_up_family or "auto",
                    "down_family": args.down_family or "auto",
                    "task_count": int(workspace.task_count.item()),
                    "direct_routes": direct_routes,
                    "unique_experts": int(captured_ids.unique().numel()),
                    "max_expert_load": int(
                        torch.bincount(captured_ids.reshape(-1), minlength=EXPERTS)
                        .max()
                        .item()
                    ),
                    **timings,
                }
            )

    aggregate = []
    for tokens in args.m:
        selected = [record for record in records if record["M"] == tokens]
        row: dict[str, object] = {"M": tokens, "route_files": len(selected)}
        for name in (
            "topk",
            "route",
            "gate_up",
            "swiglu_quant",
            "down",
            "reduce",
            "experts",
            "post_topk",
            "full",
            "fused_frontend",
            "fused_full",
        ):
            row[f"{name}_median_ms"] = median(
                float(record[f"{name}_ms"]) for record in selected
            )
        row["summed_stage_median_ms"] = sum(
            float(row[f"{name}_median_ms"])
            for name in ("topk", "route", "gate_up", "swiglu_quant", "down", "reduce")
        )
        aggregate.append(row)

    return {
        "benchmark": "gpt-oss-iq2r-captured-route-stage-boundaries",
        "device_binding": "HIP_VISIBLE_DEVICES=6",
        "direct_route_max": IQ2R_DIRECT_ROUTE_MAX,
        "scale_layout": args.scale_layout,
        "gate_up_family": args.gate_up_family or "auto",
        "down_family": args.down_family or "auto",
        "swiglu_family": os.environ.get("IQ2R_SWIGLU_QUANT_FAMILY", "group32"),
        "aggregate": aggregate,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iq2r-checkpoint",
        type=Path,
        default=Path("/models/openai/gpt-oss-120b-o0-e132-profile"),
    )
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=Path("artifacts/iq2r-production-routes-c2-c4"),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[16, 1024])
    parser.add_argument(
        "--task-rows", type=int, choices=(16, 32, 64, 128, 256), default=16
    )
    parser.add_argument(
        "--scale-layout",
        choices=("row_major", "tile16"),
        default="row_major",
    )
    parser.add_argument("--gate-up-family")
    parser.add_argument("--down-family")
    parser.add_argument("--input-stride", type=int, default=3072)
    parser.add_argument("--max-route-files", type=int, default=36)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0x1709)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["aggregate"]:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
