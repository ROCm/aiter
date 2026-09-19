# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare two IQ2R launch variants in one process on captured GPT-OSS routes.

Both variants use identical hidden states, routes, compressed checkpoints, and
workspace sizes.  Separate HIP graphs are captured while the profiling-only
family overrides select each kernel.  Timings alternate AB/BA so clock, cache,
and measurement-order drift cannot be mistaken for a kernel improvement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch

from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.moe_op import topk_softmax

from bench_iq2r_gpt_oss import _relative_metrics
from bench_iq2r_multilayer_cold_routes import (
    HIDDEN,
    LAYERS,
    TOPK,
    _benchmark_pair,
    _capture,
    _load_routes,
    _require_device_contract,
)


def _select_families(gate_up_family: str, down_family: str) -> None:
    for name, family in (
        ("IQ2R_GEMM_GATE_UP_FAMILY", gate_up_family),
        ("IQ2R_GEMM_DOWN_FAMILY", down_family),
    ):
        if family:
            os.environ[name] = family
        else:
            os.environ.pop(name, None)


def _select_swiglu_family(family: str) -> None:
    if family:
        os.environ["IQ2R_SWIGLU_QUANT_FAMILY"] = family
    else:
        os.environ.pop("IQ2R_SWIGLU_QUANT_FAMILY", None)


def _make_sequence(
    *,
    layer_order: list[int],
    layers: list[object],
    routes: list[dict[str, object]],
    workspace: IQ2RMoeWorkspace,
    outputs: list[torch.Tensor],
    topk_ids: list[torch.Tensor],
    topk_weights: list[torch.Tensor],
    token_expert_indices: list[torch.Tensor],
    tokens: int,
):
    def sequence() -> list[torch.Tensor]:
        for layer in layer_order:
            route = routes[layer]
            checkpoint = layers[layer]
            fused_router = tokens <= 16
            if not fused_router:
                topk_softmax(
                    topk_weights[layer],
                    topk_ids[layer],
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
                topk_weights[layer],
                topk_ids[layer],
                outputs[layer],
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
        return outputs

    return sequence


def _allocate_variant_state(
    tokens: int, task_rows: int, scale_layout: str, device: torch.device
):
    return {
        "workspace": IQ2RMoeWorkspace.allocate(
            tokens,
            TOPK,
            device=device,
            task_rows=task_rows,
            scale_layout=scale_layout,
        ),
        "outputs": [
            torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
            for _ in range(LAYERS)
        ],
        "topk_ids": [
            torch.empty((tokens, TOPK), dtype=torch.int32, device=device)
            for _ in range(LAYERS)
        ],
        "topk_weights": [
            torch.empty((tokens, TOPK), dtype=torch.float32, device=device)
            for _ in range(LAYERS)
        ],
    }


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

    layers = []
    for layer in range(LAYERS):
        print(f"loading layer {layer + 1}/{LAYERS}", flush=True)
        layers.append(
            load_iq2r_layer_checkpoint(args.iq2r_checkpoint, layer, device=device)
        )

    records = []
    for order_name in args.order:
        layer_order = list(range(LAYERS))
        if order_name == "reverse":
            layer_order.reverse()

        candidate = _allocate_variant_state(
            args.tokens, args.task_rows, args.candidate_scale_layout, device
        )
        control = _allocate_variant_state(
            args.tokens, args.task_rows, args.control_scale_layout, device
        )
        candidate_indices = [torch.empty_like(ids) for ids in candidate["topk_ids"]]
        control_indices = [torch.empty_like(ids) for ids in control["topk_ids"]]
        candidate_sequence = _make_sequence(
            layer_order=layer_order,
            layers=layers,
            routes=routes,
            token_expert_indices=candidate_indices,
            tokens=args.tokens,
            **candidate,
        )
        control_sequence = _make_sequence(
            layer_order=layer_order,
            layers=layers,
            routes=routes,
            token_expert_indices=control_indices,
            tokens=args.tokens,
            **control,
        )

        print(f"capturing {order_name} candidate graph", flush=True)
        _select_swiglu_family(args.candidate_swiglu_family)
        _select_families(
            args.candidate_gate_up_family,
            args.candidate_down_family,
        )
        candidate_graph, candidate_outputs = _capture(candidate_sequence)
        print(f"capturing {order_name} control graph", flush=True)
        _select_swiglu_family(args.control_swiglu_family)
        _select_families(args.control_gate_up_family, args.control_down_family)
        control_graph, control_outputs = _capture(control_sequence)

        candidate_samples, control_samples = _benchmark_pair(
            candidate_graph,
            control_graph,
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
        )
        candidate_graph.replay()
        control_graph.replay()
        torch.cuda.synchronize()

        layer_metrics = []
        for layer in range(LAYERS):
            layer_metrics.append(
                {
                    "layer": layer,
                    "route_file": routes[layer]["route_file"],
                    "candidate_routing_ids_reproduced": torch.equal(
                        torch.sort(candidate["topk_ids"][layer], dim=1).values,
                        torch.sort(routes[layer]["captured_ids"], dim=1).values,
                    ),
                    "control_routing_ids_reproduced": torch.equal(
                        torch.sort(control["topk_ids"][layer], dim=1).values,
                        torch.sort(routes[layer]["captured_ids"], dim=1).values,
                    ),
                    **_relative_metrics(
                        candidate_outputs[layer], control_outputs[layer]
                    ),
                }
            )

        candidate_latency = median(candidate_samples)
        control_latency = median(control_samples)
        records.append(
            {
                "order": order_name,
                "layers": LAYERS,
                "tokens_per_layer": args.tokens,
                "routes_per_layer": args.tokens * TOPK,
                "candidate_sequence_ms": candidate_latency,
                "control_sequence_ms": control_latency,
                "candidate_per_layer_ms": candidate_latency / LAYERS,
                "control_per_layer_ms": control_latency / LAYERS,
                "candidate_speedup": control_latency / candidate_latency,
                "candidate_samples_ms": candidate_samples,
                "control_samples_ms": control_samples,
                "routing_ids_reproduced": all(
                    bool(metric["candidate_routing_ids_reproduced"])
                    and bool(metric["control_routing_ids_reproduced"])
                    for metric in layer_metrics
                ),
                "layer_metrics": layer_metrics,
            }
        )

    os.environ.pop("IQ2R_GEMM_GATE_UP_FAMILY", None)
    os.environ.pop("IQ2R_GEMM_DOWN_FAMILY", None)
    os.environ.pop("IQ2R_SWIGLU_QUANT_FAMILY", None)
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
            "candidate": {
                "gate_up_family": args.candidate_gate_up_family,
                "down_family": args.candidate_down_family,
                "scale_layout": args.candidate_scale_layout,
                "swiglu_family": args.candidate_swiglu_family,
            },
            "control": {
                "gate_up_family": args.control_gate_up_family,
                "down_family": args.control_down_family,
                "scale_layout": args.control_scale_layout,
                "swiglu_family": args.control_swiglu_family,
            },
            "cache_model": (
                "same process and compressed weights; separate HIP graphs and "
                "workspaces; alternating candidate/control timing order; "
                "forward/reverse layer order"
            ),
        },
        "aggregate": {
            "median_candidate_speedup": median(
                float(record["candidate_speedup"]) for record in records
            ),
            "min_candidate_speedup": min(
                float(record["candidate_speedup"]) for record in records
            ),
            "max_candidate_speedup": max(
                float(record["candidate_speedup"]) for record in records
            ),
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
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--input-stride", type=int, default=3072)
    parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument(
        "--order",
        choices=("forward", "reverse"),
        nargs="+",
        default=["forward", "reverse"],
    )
    parser.add_argument("--candidate-gate-up-family", default="6x4")
    parser.add_argument("--candidate-down-family", default="3x4")
    parser.add_argument("--candidate-swiglu-family", default="")
    parser.add_argument(
        "--candidate-scale-layout",
        choices=("row_major", "tile16"),
        default="row_major",
    )
    parser.add_argument("--control-gate-up-family", default="6x4scalar")
    parser.add_argument("--control-down-family", default="3x4scalar")
    parser.add_argument("--control-swiglu-family", default="")
    parser.add_argument(
        "--control-scale-layout",
        choices=("row_major", "tile16"),
        default="row_major",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=9)
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
