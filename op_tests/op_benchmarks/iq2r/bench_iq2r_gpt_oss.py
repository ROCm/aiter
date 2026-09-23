# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPT-OSS TP1 IQ2R versus production-selected AITER MXFP4 benchmark."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from typing import Callable

import torch
from safetensors import safe_open

from aiter.iq2r_checkpoint import IQ2RLayerCheckpoint, load_iq2r_layer_checkpoint
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
    iq2r_swiglu_quant_out,
    iq2r_task_capacity,
    iq2r_task_gemm_out,
)
from aiter.ops.moe_op import topk_softmax
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    _is_gluon_available,
    moe_gemm_a16w4,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
EXPERTS = 128
TOPK = 4
HIDDEN = 2880


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("initial IQ2R qualification requires gfx950")


def _source_keys(layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.mlp.experts"
    return {
        "gate_up_blocks": f"{prefix}.gate_up_proj_blocks",
        "gate_up_scales": f"{prefix}.gate_up_proj_scales",
        "gate_up_bias": f"{prefix}.gate_up_proj_bias",
        "down_blocks": f"{prefix}.down_proj_blocks",
        "down_scales": f"{prefix}.down_proj_scales",
        "down_bias": f"{prefix}.down_proj_bias",
    }


def _load_mxfp4_baseline(
    model_dir: Path, layer: int, device: torch.device
) -> dict[str, object]:
    with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    keys = _source_keys(layer)
    files = {name: model_dir / weight_map[key] for name, key in keys.items()}
    with ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in sorted(set(files.values()))
        }
        raw = {name: handles[files[name]].get_tensor(key) for name, key in keys.items()}

    # GPT-OSS stores gate/up rows in the native adjacent order
    # [gate_0, up_0, gate_1, up_1, ...].  This is the order consumed by the
    # official ``GptOssExperts._apply_gate`` implementation and by AITER's
    # fused GUGU SwiGLU epilogue.  Do not apply ``interleave_gate_up_rows``:
    # that helper converts a split [all gate, all up] tensor to adjacent order
    # and would therefore permute this checkpoint a second time.
    gate_weight = raw["gate_up_blocks"].reshape(EXPERTS, 5760, -1)
    gate_weight = gate_weight.to(device).transpose(-2, -1)
    down_weight = raw["down_blocks"].reshape(EXPERTS, HIDDEN, -1).to(device)
    down_weight = down_weight.transpose(-2, -1)

    def prepare_scale(scale: torch.Tensor, n: int, k: int):
        scale = scale.to(device).transpose(-2, -1)
        if n % 32 or k % (32 * 8):
            return scale, None
        return shuffle_scale_moe(scale, return_layout=True, scale_kwidth=8)

    gate_scale, gate_layout = prepare_scale(raw["gate_up_scales"], 5760, HIDDEN)
    down_scale, down_layout = prepare_scale(raw["down_scales"], HIDDEN, HIDDEN)
    return {
        "gate_weight": gate_weight,
        "down_weight": down_weight,
        "gate_scale": gate_scale,
        "down_scale": down_scale,
        "gate_layout": gate_layout,
        "down_layout": down_layout,
        "gate_bias": raw["gate_up_bias"].to(device=device, dtype=torch.float32),
        "down_bias": raw["down_bias"].to(device=device, dtype=torch.float32),
    }


def _routes(tokens: int, pattern: str, device: torch.device):
    token = torch.arange(tokens, device=device, dtype=torch.int64).unsqueeze(1)
    slot = torch.arange(TOPK, device=device, dtype=torch.int64).unsqueeze(0)
    if pattern == "uniform":
        ids = (token * 17 + slot * 29) % EXPERTS
    elif pattern == "hot":
        ids = (token % 2 + slot) % 8
    elif pattern == "single-group":
        ids = slot.expand(tokens, -1)
    else:
        raise ValueError(f"unsupported routing pattern {pattern!r}")
    scores = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device).expand(tokens, -1)
    weights = torch.softmax(scores, dim=-1).float().contiguous()
    logits = torch.full((tokens, EXPERTS), -20.0, dtype=torch.float32, device=device)
    logits.scatter_(1, ids, scores)
    return ids.to(torch.int32).contiguous(), weights, logits


def _sample(call: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark(call, warmup: int, iterations: int, samples: int):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = [_sample(call, iterations) for _ in range(samples)]
    return median(values), values


def _capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    torch.cuda.synchronize()
    return graph, result


def _mxfp4_call(hidden, weights, routing_data, gather_index, scatter_index, backend):
    intermediate = moe_gemm_a16w4(
        hidden,
        weights["gate_weight"],
        None,
        weights["gate_scale"],
        bias=weights["gate_bias"],
        routing_data=routing_data,
        gather_indx=gather_index,
        swizzle_mx_scale=weights["gate_layout"],
        out_dtype=torch.bfloat16,
        apply_swiglu=True,
        alpha=1.702,
        limit=7.0,
        swiglu_add_residual=True,
        backend=backend,
    )
    return moe_gemm_a16w4(
        intermediate,
        weights["down_weight"],
        None,
        weights["down_scale"],
        bias=weights["down_bias"],
        routing_data=routing_data,
        scatter_indx=scatter_index,
        gammas=routing_data.gate_scal,
        swizzle_mx_scale=weights["down_layout"],
        out_dtype=torch.bfloat16,
        backend=backend,
    )


def _iq2r_call(hidden, ids, weights, checkpoint, workspace, output):
    iq2r_fused_moe_out(
        hidden,
        checkpoint.gate_up_data,
        checkpoint.gate_up_auxiliary,
        checkpoint.down_data,
        checkpoint.down_auxiliary,
        weights,
        ids,
        output,
        gate_up_metadata=checkpoint.gate_up_metadata,
        down_metadata=checkpoint.down_metadata,
        gate_up_tile_n=checkpoint.gate_up_tile_n,
        down_tile_n=checkpoint.down_tile_n,
        gate_up_bias=checkpoint.gate_up_bias,
        down_bias=checkpoint.down_bias,
        workspace=workspace,
    )


def _iq2r_from_logits(
    hidden,
    logits,
    ids,
    weights,
    token_expert_indices,
    checkpoint,
    workspace,
    output,
):
    topk_softmax(weights, ids, token_expert_indices, logits, True)
    _iq2r_call(hidden, ids, weights, checkpoint, workspace, output)


def _relative_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    return {
        "relative_rmse_vs_mxfp4": (
            delta.square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-12)
        ).item(),
        "mean_cosine_vs_mxfp4": torch.nn.functional.cosine_similarity(
            actual, expected, dim=-1
        )
        .mean()
        .item(),
    }


def run(args: argparse.Namespace) -> list[dict]:
    _require_device_contract()
    device = torch.device("cuda")
    checkpoint: IQ2RLayerCheckpoint = load_iq2r_layer_checkpoint(
        args.iq2r_checkpoint, args.layer, device=device
    )
    if checkpoint.expert_count != EXPERTS or checkpoint.expert_start != 0:
        raise ValueError("benchmark requires the full 128-expert IQ2R layer")
    baseline = _load_mxfp4_baseline(args.model_dir, args.layer, device)
    workspace = IQ2RMoeWorkspace.allocate(
        max(args.m), TOPK, device=device, task_rows=args.task_rows
    )
    results = []

    for tokens in args.m:
        generator = torch.Generator(device=device).manual_seed(args.seed + tokens)
        hidden = torch.randn(
            (tokens, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
        )
        topk_ids, topk_weights, logits = _routes(tokens, args.routing, device)
        routes = tokens * TOPK
        output = torch.empty_like(hidden)
        logits_ids = torch.empty_like(topk_ids)
        logits_weights = torch.empty_like(topk_weights)
        token_expert_indices = torch.empty_like(topk_ids)
        routing_data, gather_index, scatter_index = routing(logits, TOPK)

        _iq2r_call(hidden, topk_ids, topk_weights, checkpoint, workspace, output)
        iq2r_latency, iq2r_samples = _benchmark(
            lambda: _iq2r_call(
                hidden, topk_ids, topk_weights, checkpoint, workspace, output
            ),
            args.warmup,
            args.iterations,
            args.samples,
        )
        iq2r_graph, _ = _capture(
            lambda: _iq2r_call(
                hidden, topk_ids, topk_weights, checkpoint, workspace, output
            )
        )
        iq2r_graph_latency, iq2r_graph_samples = _benchmark(
            iq2r_graph.replay, args.warmup, args.iterations, args.samples
        )
        iq2r_logits_latency, iq2r_logits_samples = _benchmark(
            lambda: _iq2r_from_logits(
                hidden,
                logits,
                logits_ids,
                logits_weights,
                token_expert_indices,
                checkpoint,
                workspace,
                output,
            ),
            args.warmup,
            args.iterations,
            args.samples,
        )

        baseline_timings = {}
        baseline_outputs = {}
        for backend in args.baseline_backend:
            if backend == "gluon" and not _is_gluon_available():
                baseline_timings[backend] = {"available": False}
                continue
            try:
                baseline_outputs[backend] = _mxfp4_call(
                    hidden,
                    baseline,
                    routing_data,
                    gather_index,
                    scatter_index,
                    backend,
                )
                latency, samples = _benchmark(
                    lambda backend=backend: _mxfp4_call(
                        hidden,
                        baseline,
                        routing_data,
                        gather_index,
                        scatter_index,
                        backend,
                    ),
                    args.warmup,
                    args.iterations,
                    args.samples,
                )
                graph, graph_output = _capture(
                    lambda backend=backend: _mxfp4_call(
                        hidden,
                        baseline,
                        routing_data,
                        gather_index,
                        scatter_index,
                        backend,
                    )
                )
                graph_latency, graph_samples = _benchmark(
                    graph.replay, args.warmup, args.iterations, args.samples
                )
                baseline_outputs[backend] = graph_output
                baseline_timings[backend] = {
                    "available": True,
                    "latency_ms": latency,
                    "samples_ms": samples,
                    "graph_latency_ms": graph_latency,
                    "graph_samples_ms": graph_samples,
                }
            except (AssertionError, RuntimeError) as error:
                baseline_timings[backend] = {
                    "available": True,
                    "error": str(error),
                }
        valid = {
            name: timing
            for name, timing in baseline_timings.items()
            if "latency_ms" in timing
        }
        if not valid:
            raise RuntimeError(f"all MXFP4 backends failed: {baseline_timings}")
        best = min(valid, key=lambda name: valid[name]["latency_ms"])
        best_graph = min(valid, key=lambda name: valid[name]["graph_latency_ms"])

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
        expert_ids = topk_ids.reshape(-1)

        direct_routes = routes <= IQ2R_DIRECT_ROUTE_MAX and tasks.shape[0] >= routes

        def route_quant_call():
            if direct_routes:
                iq2r_route_direct_gather_quant_out(
                    hidden,
                    expert_ids,
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
                    expert_ids,
                    sorted_ids,
                    gather,
                    scatter,
                    workspace.sort_workspace,
                    tasks,
                    workspace.task_count,
                    expert_count=EXPERTS,
                    task_rows=args.task_rows,
                )
                iq2r_route_gather_quant_out(
                    hidden,
                    gather,
                    route_input_fp8,
                    route_input_scales,
                    topk=TOPK,
                )

        route_quant_call()
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
        iq2r_swiglu_quant_out(
            gate_up,
            intermediate_fp8,
            intermediate_scales,
        )

        stages = {}

        def measure(name, call):
            latency, samples = _benchmark(
                call, args.warmup, args.iterations, args.samples
            )
            stages[f"iq2r_{name}_latency_ms"] = latency
            stages[f"iq2r_{name}_samples_ms"] = samples
            graph, _ = _capture(call)
            graph_latency, graph_samples = _benchmark(
                graph.replay, args.warmup, args.iterations, args.samples
            )
            stages[f"iq2r_{name}_graph_latency_ms"] = graph_latency
            stages[f"iq2r_{name}_graph_samples_ms"] = graph_samples

        if direct_routes:
            measure("direct_route_gather_quant", route_quant_call)
        else:
            measure(
                "sort_tasks",
                lambda: iq2r_route_sort_tasks_out(
                    expert_ids,
                    sorted_ids,
                    gather,
                    scatter,
                    workspace.sort_workspace,
                    tasks,
                    workspace.task_count,
                    expert_count=EXPERTS,
                    task_rows=args.task_rows,
                ),
            )
            measure(
                "gather_quant",
                lambda: iq2r_route_gather_quant_out(
                    hidden,
                    gather,
                    route_input_fp8,
                    route_input_scales,
                    topk=TOPK,
                ),
            )
        measure(
            "gate_up_gemm",
            lambda: iq2r_task_gemm_out(
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
            ),
        )
        measure(
            "swiglu_quant",
            lambda: iq2r_swiglu_quant_out(
                gate_up,
                intermediate_fp8,
                intermediate_scales,
            ),
        )
        measure(
            "down_gemm",
            lambda: iq2r_task_gemm_out(
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
            ),
        )
        measure(
            "reduce",
            lambda: iq2r_route_reduce_indexed_out(
                route_output, topk_weights, scatter, output, topk=TOPK
            ),
        )

        result = {
            "M": tokens,
            "routing": args.routing,
            "task_rows": args.task_rows,
            "task_count": int(workspace.task_count.item()),
            "direct_routes": direct_routes,
            "iq2r_experts_latency_ms": iq2r_latency,
            "iq2r_experts_samples_ms": iq2r_samples,
            "iq2r_graph_latency_ms": iq2r_graph_latency,
            "iq2r_graph_samples_ms": iq2r_graph_samples,
            "iq2r_from_logits_latency_ms": iq2r_logits_latency,
            "iq2r_from_logits_samples_ms": iq2r_logits_samples,
            "mxfp4_best_backend": best,
            "mxfp4_experts_latency_ms": valid[best]["latency_ms"],
            "mxfp4_best_graph_backend": best_graph,
            "mxfp4_graph_latency_ms": valid[best_graph]["graph_latency_ms"],
            "mxfp4_backends": baseline_timings,
            "speedup_vs_best_mxfp4": valid[best]["latency_ms"] / iq2r_latency,
            "speedup_vs_best_mxfp4_graph": (
                valid[best_graph]["graph_latency_ms"] / iq2r_graph_latency
            ),
            **stages,
            **_relative_metrics(output, baseline_outputs[best]),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return results


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
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M_VALUES))
    parser.add_argument(
        "--routing", choices=("uniform", "hot", "single-group"), default="uniform"
    )
    parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument(
        "--baseline-backend", nargs="+", choices=("triton", "gluon"), default=["triton"]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0x1709)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
