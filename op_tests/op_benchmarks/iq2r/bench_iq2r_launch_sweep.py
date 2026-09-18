# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sweep GPT-OSS IQ2R cooperative launch families.

The sweep accepts either captured serving routes or deterministic synthetic
uniform/hot/single-group routes.  Synthetic routes make it possible to cover
decode token counts that have not yet been captured from ATOM while retaining
the production task-creation path, including direct dispatch at M=4.
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
from aiter.iq2r_moe import IQ2R_DIRECT_ROUTE_MAX, IQ2RMoeWorkspace
from aiter.ops.iq2r import (
    iq2r_route_direct_gather_quant_out,
    iq2r_route_gather_quant_out,
    iq2r_route_sort_tasks_out,
    iq2r_swiglu_quant_out,
    iq2r_task_capacity,
    iq2r_task_gemm_out,
)
from bench_iq2r_gpt_oss import _routes

EXPERTS = 128
TOPK = 4
HIDDEN = 2880
FAMILIES = ("3x4", "3x8", "6x4", "6x8")


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("IQ2R launch sweep requires gfx950")


def _benchmark_graph(
    call: Callable[[], None], warmup: int, iterations: int, samples: int
) -> tuple[float, list[float]]:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
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


def _set_launch(projection: str, family: str, grid: int) -> None:
    prefix = "IQ2R_GEMM_GATE_UP" if projection == "gate_up" else "IQ2R_GEMM_DOWN"
    os.environ[f"{prefix}_FAMILY"] = family
    os.environ[f"{prefix}_GRID_MULTIPLIER"] = str(grid)


def _clear_launches() -> None:
    for projection in ("GATE_UP", "DOWN"):
        os.environ.pop(f"IQ2R_GEMM_{projection}_FAMILY", None)
        os.environ.pop(f"IQ2R_GEMM_{projection}_GRID_MULTIPLIER", None)


def _select_files(root: Path, tokens: int, limit: int) -> list[Path]:
    files = sorted(root.glob(f"*-m{tokens}-*.pt"))
    if not files:
        raise FileNotFoundError(f"no m={tokens} route captures under {root}")
    if len(files) <= limit:
        return files
    # Even spacing retains examples from early/middle/late layers and decode steps.
    indices = [round(i * (len(files) - 1) / (limit - 1)) for i in range(limit)]
    return [files[i] for i in indices]


def _route_cases(
    args: argparse.Namespace, tokens: int, device: torch.device
) -> list[tuple[str, torch.Tensor]]:
    if args.routing:
        return [
            (f"synthetic-{pattern}", _routes(tokens, pattern, device)[0])
            for pattern in args.routing
        ]
    if args.route_dir is None:
        raise ValueError("provide --routing or --route-dir")
    cases = []
    for route_file in _select_files(args.route_dir, tokens, args.max_route_files):
        capture = torch.load(route_file, map_location="cpu", weights_only=True)
        cases.append(
            (
                route_file.name,
                capture["topk_ids"].to(device=device, dtype=torch.int32),
            )
        )
    return cases


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    device = torch.device("cuda")
    checkpoint = load_iq2r_layer_checkpoint(
        args.iq2r_checkpoint, args.layer, device=device
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    candidates = [(family, grid) for family in FAMILIES for grid in args.grid]
    records: list[dict[str, object]] = []

    for tokens in args.m:
        workspace = IQ2RMoeWorkspace.allocate(
            tokens, TOPK, device=device, task_rows=args.task_rows
        )
        routes = tokens * TOPK
        task_capacity = iq2r_task_capacity(routes, EXPERTS, args.task_rows)
        tasks = workspace.tasks[:task_capacity]
        sorted_ids = workspace.sorted_expert_ids[:routes]
        gather = workspace.gather_indices[:routes]
        scatter = workspace.scatter_indices[:routes]
        route_input_fp8 = workspace.route_input_fp8[:routes]
        route_input_scales = workspace.route_input_scales[:routes]
        gate_up = workspace.gate_up[:routes]
        intermediate_fp8 = workspace.intermediate_fp8[:routes]
        intermediate_scales = workspace.intermediate_scales[:routes]
        route_output = workspace.route_output[:routes]

        for route_name, topk_ids in _route_cases(args, tokens, device):
            hidden = torch.randn(
                (tokens, HIDDEN),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            if routes <= IQ2R_DIRECT_ROUTE_MAX and tasks.shape[0] >= routes:
                iq2r_route_direct_gather_quant_out(
                    hidden,
                    topk_ids.reshape(-1),
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
                    topk_ids.reshape(-1),
                    sorted_ids,
                    gather,
                    scatter,
                    tasks,
                    workspace.task_count,
                    expert_count=EXPERTS,
                    task_rows=args.task_rows,
                )
                iq2r_route_gather_quant_out(
                    hidden, gather, route_input_fp8, route_input_scales, topk=TOPK
                )

            def gate_call():
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

            _clear_launches()
            gate_call()
            gate_reference = gate_up.clone()
            for family, grid in candidates:
                _set_launch("gate_up", family, grid)
                latency, samples = _benchmark_graph(
                    gate_call, args.warmup, args.iterations, args.samples
                )
                error = float((gate_up.float() - gate_reference.float()).abs().max())
                records.append(
                    {
                        "M": tokens,
                        "routing": route_name.removeprefix("synthetic-")
                        if route_name.startswith("synthetic-")
                        else "captured",
                        "route_file": route_name,
                        "projection": "gate_up",
                        "family": family,
                        "grid_multiplier": grid,
                        "latency_ms": latency,
                        "samples_ms": samples,
                        "max_abs_error_vs_default": error,
                        "task_count": int(workspace.task_count.item()),
                        "unique_experts": int(topk_ids.unique().numel()),
                    }
                )

            # Produce one stable intermediate for the independent down sweep.
            _clear_launches()
            gate_call()
            iq2r_swiglu_quant_out(gate_up, intermediate_fp8, intermediate_scales)

            def down_call():
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

            _clear_launches()
            down_call()
            down_reference = route_output.clone()
            for family, grid in candidates:
                _set_launch("down", family, grid)
                latency, samples = _benchmark_graph(
                    down_call, args.warmup, args.iterations, args.samples
                )
                error = float(
                    (route_output.float() - down_reference.float()).abs().max()
                )
                records.append(
                    {
                        "M": tokens,
                        "routing": route_name.removeprefix("synthetic-")
                        if route_name.startswith("synthetic-")
                        else "captured",
                        "route_file": route_name,
                        "projection": "down",
                        "family": family,
                        "grid_multiplier": grid,
                        "latency_ms": latency,
                        "samples_ms": samples,
                        "max_abs_error_vs_default": error,
                        "task_count": int(workspace.task_count.item()),
                        "unique_experts": int(topk_ids.unique().numel()),
                    }
                )
            _clear_launches()

    aggregate = []
    route_groups = args.routing if args.routing else ["captured"]
    for tokens in args.m:
        for routing_pattern in route_groups:
            for projection in ("gate_up", "down"):
                for family, grid in candidates:
                    selected = [
                        float(record["latency_ms"])
                        for record in records
                        if record["M"] == tokens
                        and record["routing"] == routing_pattern
                        and record["projection"] == projection
                        and record["family"] == family
                        and record["grid_multiplier"] == grid
                    ]
                    errors = [
                        float(record["max_abs_error_vs_default"])
                        for record in records
                        if record["M"] == tokens
                        and record["routing"] == routing_pattern
                        and record["projection"] == projection
                        and record["family"] == family
                        and record["grid_multiplier"] == grid
                    ]
                    aggregate.append(
                        {
                            "M": tokens,
                            "routing": routing_pattern,
                            "projection": projection,
                            "family": family,
                            "grid_multiplier": grid,
                            "median_latency_ms": median(selected),
                            "min_latency_ms": min(selected),
                            "max_latency_ms": max(selected),
                            "max_abs_error_vs_default": max(errors),
                            "route_files": len(selected),
                        }
                    )
    return {"aggregate": aggregate, "records": records}


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
        default=None,
    )
    parser.add_argument(
        "--routing",
        choices=("uniform", "hot", "single-group"),
        nargs="+",
        default=None,
        help="use deterministic synthetic routes instead of captured files",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--grid", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-route-files", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0x1709)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    route_groups = args.routing if args.routing else ["captured"]
    for tokens in args.m:
        for routing_pattern in route_groups:
            for projection in ("gate_up", "down"):
                rows = [
                    row
                    for row in result["aggregate"]
                    if row["M"] == tokens
                    and row["routing"] == routing_pattern
                    and row["projection"] == projection
                ]
                print(json.dumps(min(rows, key=lambda row: row["median_latency_ms"])))


if __name__ == "__main__":
    main()
