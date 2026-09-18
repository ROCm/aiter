# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare IQ2R with dynamic A8W4 on captured GPT-OSS production routes.

Both paths start from the same BF16 hidden states and FP32 router logits.  The
A8W4 path dynamically quantizes activations to blockwise MXFP8 for both MoE
GEMMs; IQ2R performs its corresponding dynamic MXFP8 preparation internally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Callable

# fused_moe_triton conditionally imports its AITER routing kernels at import.
os.environ.setdefault("ATOM_USE_TRITON_MOE", "1")

import torch

from atom.model_ops.fused_moe_triton import triton_kernel_moe_forward
from atom.model_ops.moe import MoEActivationQuant
from aiter import ActivationType
from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.moe_op import topk_softmax

from bench_iq2r_gpt_oss import _load_mxfp4_baseline, _relative_metrics

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


def _capture(call: Callable[[], torch.Tensor]):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    torch.cuda.synchronize()
    return graph, output


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


def _select_files(root: Path, tokens: int, limit: int) -> list[Path]:
    files = sorted(root.glob(f"*-m{tokens}-*.pt"))
    if not files:
        raise FileNotFoundError(f"no m={tokens} route captures under {root}")
    if len(files) <= limit:
        return files
    if limit == 1:
        return [files[0]]
    indices = [round(i * (len(files) - 1) / (limit - 1)) for i in range(limit)]
    return [files[i] for i in indices]


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
    device = torch.device("cuda")
    iq2r = load_iq2r_layer_checkpoint(args.iq2r_checkpoint, args.layer, device=device)
    a8w4 = _load_mxfp4_baseline(args.model_dir, args.layer, device)
    records: list[dict[str, object]] = []

    for tokens in args.m:
        workspace = IQ2RMoeWorkspace.allocate(
            tokens, TOPK, device=device, task_rows=args.task_rows
        )
        iq2r_output = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        iq2r_ids = torch.empty((tokens, TOPK), dtype=torch.int32, device=device)
        iq2r_weights = torch.empty((tokens, TOPK), dtype=torch.float32, device=device)
        token_expert_indices = torch.empty_like(iq2r_ids)

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
            hidden = torch.randn(
                (tokens, HIDDEN),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )

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

            def a8w4_dynamic_call() -> torch.Tensor:
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

            iq_graph, iq_output = _capture(iq2r_call)
            a8_graph, a8_output = _capture(a8w4_dynamic_call)
            if index % 2 == 0:
                iq_latency, iq_samples = _benchmark(
                    iq_graph, args.warmup, args.iterations, args.samples
                )
                a8_latency, a8_samples = _benchmark(
                    a8_graph, args.warmup, args.iterations, args.samples
                )
            else:
                a8_latency, a8_samples = _benchmark(
                    a8_graph, args.warmup, args.iterations, args.samples
                )
                iq_latency, iq_samples = _benchmark(
                    iq_graph, args.warmup, args.iterations, args.samples
                )
            iq_graph.replay()
            a8_graph.replay()
            torch.cuda.synchronize()
            ids_equal = torch.equal(
                torch.sort(iq2r_ids, dim=1).values,
                torch.sort(captured_ids, dim=1).values,
            )
            records.append(
                {
                    "M": tokens,
                    "route_file": route_file.name,
                    "unique_experts": int(captured_ids.unique().numel()),
                    "max_expert_load": int(
                        torch.bincount(captured_ids.reshape(-1), minlength=EXPERTS)
                        .max()
                        .item()
                    ),
                    "routing_ids_reproduced": ids_equal,
                    "iq2r_post_router_ms": iq_latency,
                    "iq2r_samples_ms": iq_samples,
                    "a8w4_dynamic_post_router_ms": a8_latency,
                    "a8w4_dynamic_samples_ms": a8_samples,
                    "speedup": a8_latency / iq_latency,
                    **_relative_metrics(iq_output, a8_output),
                }
            )

    aggregate = []
    for tokens in args.m:
        selected = [record for record in records if record["M"] == tokens]
        speedups = [float(record["speedup"]) for record in selected]
        iq_latencies = [float(record["iq2r_post_router_ms"]) for record in selected]
        a8_latencies = [
            float(record["a8w4_dynamic_post_router_ms"]) for record in selected
        ]
        aggregate.append(
            {
                "M": tokens,
                "route_files": len(selected),
                "iq2r_median_ms": median(iq_latencies),
                "a8w4_dynamic_median_ms": median(a8_latencies),
                "median_speedup": median(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "win_fraction": sum(speedup > 1.0 for speedup in speedups)
                / len(speedups),
                "routing_ids_reproduced": all(
                    bool(record["routing_ids_reproduced"]) for record in selected
                ),
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
        "--model-dir", type=Path, default=Path("/models/openai/gpt-oss-120b")
    )
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=Path("artifacts/iq2r-production-routes-c2-c4"),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=16)
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
