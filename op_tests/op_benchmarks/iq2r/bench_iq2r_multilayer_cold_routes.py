# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay captured GPT-OSS routes with distinct weights for every layer.

The single-layer captured-route benchmark intentionally keeps one layer's
weights hot while varying route distributions.  This benchmark exercises the
complementary production question: whether IQ2R still beats dynamic A8W4 when
all 36 GPT-OSS MoE layers execute in sequence and each layer reads distinct
expert weights.  Forward and reverse layer orders reduce cache-order bias.

Both paths start at FP32 router logits and BF16 hidden states.  The A8W4 path
dynamically quantizes activations to blockwise MXFP8; IQ2R performs its matched
MXFP8 activation preparation internally.
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
LAYERS = 36


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("multi-layer cold-route qualification requires gfx950")


def _route_file(root: Path, tokens: int, layer: int, route_index: int) -> Path:
    matches = sorted(
        root.glob(f"model_layers_{layer}_mlp_experts_fused_moe-m{tokens}-*.pt")
    )
    if not matches:
        raise RuntimeError(
            f"expected at least one m={tokens} capture for layer {layer} under {root}"
        )
    if route_index >= len(matches):
        raise RuntimeError(
            f"m={tokens} layer {layer} has {len(matches)} captures under {root}, "
            f"so route index {route_index} is unavailable"
        )
    return matches[route_index]


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


def _capture(call: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object]:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    torch.cuda.synchronize()
    return graph, output


def _time_graph(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
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
    for index in range(warmup):
        if index % 2:
            a8w4_graph.replay()
            iq2r_graph.replay()
        else:
            iq2r_graph.replay()
            a8w4_graph.replay()
    torch.cuda.synchronize()

    iq2r_samples = []
    a8w4_samples = []
    for index in range(samples):
        # Alternating AB/BA order avoids consistently giving either format the
        # cache state left by the other format's distinct, much larger weights.
        if index % 2:
            iq2r_graph.replay()
            a8w4_samples.append(_time_graph(a8w4_graph, iterations))
            a8w4_graph.replay()
            iq2r_samples.append(_time_graph(iq2r_graph, iterations))
        else:
            a8w4_graph.replay()
            iq2r_samples.append(_time_graph(iq2r_graph, iterations))
            iq2r_graph.replay()
            a8w4_samples.append(_time_graph(a8w4_graph, iterations))
    return iq2r_samples, a8w4_samples


def _load_routes(
    route_dir: Path,
    tokens: int,
    device: torch.device,
    seed: int,
    route_index: int,
    input_stride: int,
) -> list[dict[str, object]]:
    records = []
    for layer in range(LAYERS):
        path = _route_file(route_dir, tokens, layer, route_index)
        capture = torch.load(path, map_location="cpu", weights_only=True)
        captured_ids = capture["topk_ids"].to(device=device, dtype=torch.int32)
        captured_weights = capture["topk_weights"].to(
            device=device, dtype=torch.float32
        )
        generator = torch.Generator(device=device).manual_seed(seed + layer * 1009)
        records.append(
            {
                "layer": layer,
                "route_file": path.name,
                "captured_ids": captured_ids,
                "logits": _logits_from_capture(captured_ids, captured_weights),
                "hidden": torch.randn(
                    (tokens, input_stride),
                    generator=generator,
                    dtype=torch.bfloat16,
                    device=device,
                )[:, :HIDDEN],
            }
        )
    return records


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    if args.input_stride < HIDDEN:
        raise ValueError(f"input stride must be at least {HIDDEN}")
    device = torch.device("cuda")
    routes = _load_routes(
        args.route_dir,
        args.tokens,
        device,
        args.seed,
        args.route_index,
        args.input_stride,
    )

    iq2r_layers = []
    a8w4_layers = []
    for layer in range(LAYERS):
        print(f"loading layer {layer + 1}/{LAYERS}", flush=True)
        iq2r_layers.append(
            load_iq2r_layer_checkpoint(args.iq2r_checkpoint, layer, device=device)
        )
        a8w4_layers.append(_load_mxfp4_baseline(args.model_dir, layer, device))

    records = []
    for order_name in args.order:
        layer_order = list(range(LAYERS))
        if order_name == "reverse":
            layer_order.reverse()

        workspace = IQ2RMoeWorkspace.allocate(
            args.tokens, TOPK, device=device, task_rows=args.task_rows
        )
        iq2r_outputs = [
            torch.empty((args.tokens, HIDDEN), dtype=torch.bfloat16, device=device)
            for _ in range(LAYERS)
        ]
        iq2r_ids = [
            torch.empty((args.tokens, TOPK), dtype=torch.int32, device=device)
            for _ in range(LAYERS)
        ]
        iq2r_weights = [
            torch.empty((args.tokens, TOPK), dtype=torch.float32, device=device)
            for _ in range(LAYERS)
        ]
        token_expert_indices = [torch.empty_like(ids) for ids in iq2r_ids]

        def iq2r_sequence() -> list[torch.Tensor]:
            for layer in layer_order:
                route = routes[layer]
                checkpoint = iq2r_layers[layer]
                fused_router = args.tokens <= 16
                if not fused_router:
                    topk_softmax(
                        iq2r_weights[layer],
                        iq2r_ids[layer],
                        token_expert_indices[layer],
                        route["logits"],
                        True,
                    )
                iq2r_fused_moe_out(
                    route["hidden"],
                    checkpoint.gate_up_data,
                    checkpoint.gate_up_auxiliary,
                    checkpoint.down_data,
                    checkpoint.down_auxiliary,
                    iq2r_weights[layer],
                    iq2r_ids[layer],
                    iq2r_outputs[layer],
                    gate_up_metadata=checkpoint.gate_up_metadata,
                    down_metadata=checkpoint.down_metadata,
                    gate_up_tile_n=checkpoint.gate_up_tile_n,
                    down_tile_n=checkpoint.down_tile_n,
                    gate_up_bias=checkpoint.gate_up_bias,
                    down_bias=checkpoint.down_bias,
                    workspace=workspace,
                    router_logits=route["logits"] if fused_router else None,
                    renormalize=True,
                )
            return iq2r_outputs

        def a8w4_sequence() -> list[torch.Tensor]:
            outputs = []
            for layer in layer_order:
                route = routes[layer]
                weights = a8w4_layers[layer]
                outputs.append(
                    triton_kernel_moe_forward(
                        route["hidden"],
                        weights["gate_weight"],
                        weights["down_weight"],
                        route["logits"],
                        topk=TOPK,
                        renormalize=True,
                        activation=ActivationType.Swiglu,
                        w13_scale=weights["gate_scale"],
                        w2_scale=weights["down_scale"],
                        w13_swizzle_layout=weights["gate_layout"],
                        w2_swizzle_layout=weights["down_layout"],
                        w1_bias=weights["gate_bias"],
                        w2_bias=weights["down_bias"],
                        swiglu_limit=7.0,
                        global_num_experts=EXPERTS,
                        act_quant=MoEActivationQuant.FP8,
                    )
                )
            return outputs

        print(f"capturing {order_name} IQ2R graph", flush=True)
        iq2r_graph, iq2r_graph_outputs = _capture(iq2r_sequence)
        print(f"capturing {order_name} A8W4 graph", flush=True)
        a8w4_graph, a8w4_graph_outputs = _capture(a8w4_sequence)
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

        layer_metrics = []
        for layer in range(LAYERS):
            captured_ids = routes[layer]["captured_ids"]
            ids_equal = torch.equal(
                torch.sort(iq2r_ids[layer], dim=1).values,
                torch.sort(captured_ids, dim=1).values,
            )
            layer_metrics.append(
                {
                    "layer": layer,
                    "route_file": routes[layer]["route_file"],
                    "routing_ids_reproduced": ids_equal,
                    **_relative_metrics(
                        iq2r_graph_outputs[layer], a8w4_graph_outputs[layer]
                    ),
                }
            )

        iq2r_latency = median(iq2r_samples)
        a8w4_latency = median(a8w4_samples)
        records.append(
            {
                "order": order_name,
                "layers": LAYERS,
                "tokens_per_layer": args.tokens,
                "routes_per_layer": args.tokens * TOPK,
                "iq2r_sequence_ms": iq2r_latency,
                "a8w4_dynamic_sequence_ms": a8w4_latency,
                "iq2r_per_layer_ms": iq2r_latency / LAYERS,
                "a8w4_dynamic_per_layer_ms": a8w4_latency / LAYERS,
                "speedup": a8w4_latency / iq2r_latency,
                "iq2r_samples_ms": iq2r_samples,
                "a8w4_dynamic_samples_ms": a8w4_samples,
                "routing_ids_reproduced": all(
                    bool(metric["routing_ids_reproduced"]) for metric in layer_metrics
                ),
                "layer_metrics": layer_metrics,
            }
        )

    return {
        "protocol": {
            "layers": LAYERS,
            "tokens_per_layer": args.tokens,
            "routes_per_layer": args.tokens * TOPK,
            "task_rows": args.task_rows,
            "route_index": args.route_index,
            "input_stride": args.input_stride,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "samples": args.samples,
            "orders": args.order,
            "device": "HIP_VISIBLE_DEVICES=6",
            "cache_model": (
                "36 distinct layer-weight pairs per graph; alternating A8W4/IQ2R "
                "timing order; forward/reverse layer order"
            ),
        },
        "aggregate": {
            "median_speedup": median(float(record["speedup"]) for record in records),
            "min_speedup": min(float(record["speedup"]) for record in records),
            "max_speedup": max(float(record["speedup"]) for record in records),
            "routing_ids_reproduced": all(
                bool(record["routing_ids_reproduced"]) for record in records
            ),
        },
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
        "--model-dir", type=Path, default=Path("/models/openai/gpt-oss-120b")
    )
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--input-stride", type=int, default=3072)
    parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument(
        "--order",
        choices=("forward", "reverse"),
        nargs="+",
        default=["forward", "reverse"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0x1709)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"]), flush=True)
    for record in result["records"]:
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "layer_metrics"}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
