# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the GPT-OSS BF16 router projection through IQ2R input quantization.

This measures the production boundary that the expert-only benchmarks omit:

    BF16 [M, 2880] @ BF16 [128, 2880]^T + bias
      -> softmax top-k4
      -> route ordering/task creation
      -> routed-input MXFP8 quantization

Both one-layer-hot and L3-cold router-weight measurements are reported.  The
cold graph rotates enough distinct copies of the router matrix to exceed the
assumed L3 capacity, approximating the per-layer weight residency seen by a
36-layer decoder whose much larger weights evict each router between tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Callable

import torch
from safetensors import safe_open

from aiter.iq2r_moe import IQ2R_DIRECT_ROUTE_MAX, IQ2RMoeWorkspace
from aiter.ops.iq2r import (
    iq2r_route_topk_direct_gather_quant_out,
    iq2r_route_topk_sort_gather_quant_out,
)
from aiter.ops.moe_op import topk_softmax
from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

EXPERTS = 128
TOPK = 4
HIDDEN = 2880


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("GPT-OSS IQ2R qualification requires gfx950")


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _benchmark(
    call: Callable[[], None],
    *,
    divisor: int,
    warmup: int,
    iterations: int,
    samples: int,
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
        values.append(start.elapsed_time(end) / iterations / divisor)
    return median(values), values


def _load_router(
    model: Path, layer: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_name = f"model.layers.{layer}.mlp.router.weight"
    bias_name = f"model.layers.{layer}.mlp.router.bias"
    shard_name = index["weight_map"][weight_name]
    if index["weight_map"][bias_name] != shard_name:
        raise RuntimeError("GPT-OSS router weight and bias are in different shards")
    shard = model / shard_name
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_name).to(device=device)
        bias = handle.get_tensor(bias_name).to(device=device)
    if weight.shape != (EXPERTS, HIDDEN) or bias.shape != (EXPERTS,):
        raise RuntimeError(
            f"unexpected router tensors: weight={tuple(weight.shape)}, "
            f"bias={tuple(bias.shape)}"
        )
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        raise RuntimeError(
            f"expected BF16 router tensors, got {weight.dtype} and {bias.dtype}"
        )
    return weight.contiguous(), bias.contiguous(), shard


def _frontend(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    workspace: IQ2RMoeWorkspace,
    task_rows: int,
    router_bias: torch.Tensor | None = None,
) -> None:
    tokens = hidden.shape[0]
    routes = tokens * TOPK
    topk_weights = workspace.topk_weights[:tokens]
    topk_ids = workspace.topk_ids[:tokens]
    sorted_ids = workspace.sorted_expert_ids[:routes]
    gather = workspace.gather_indices[:routes]
    scatter = workspace.scatter_indices[:routes]
    tasks = workspace.tasks
    route_input_fp8 = workspace.route_input_fp8[:routes]
    route_input_scales = workspace.route_input_scales[:routes]
    if routes <= IQ2R_DIRECT_ROUTE_MAX:
        iq2r_route_topk_direct_gather_quant_out(
            hidden,
            logits,
            topk_weights,
            topk_ids,
            sorted_ids,
            gather,
            scatter,
            tasks,
            workspace.task_count,
            route_input_fp8,
            route_input_scales,
            renormalize=True,
            router_bias=router_bias,
        )
    else:
        iq2r_route_topk_sort_gather_quant_out(
            hidden,
            logits,
            topk_weights,
            topk_ids,
            sorted_ids,
            gather,
            scatter,
            tasks,
            workspace.task_count,
            route_input_fp8,
            route_input_scales,
            task_rows=task_rows,
            renormalize=True,
            router_bias=router_bias,
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    if args.input_stride < HIDDEN:
        raise ValueError(f"input stride must be at least {HIDDEN}")

    device = torch.device("cuda")
    weight, bias, shard = _load_router(args.model, args.layer, device)
    weight_bytes = weight.numel() * weight.element_size()
    cold_count = max(2, math.ceil(args.assumed_l3_mib * 1024**2 / weight_bytes) + 2)
    cold_weights = [weight.clone() for _ in range(cold_count)]
    records: list[dict[str, object]] = []

    for tokens in args.m:
        generator = torch.Generator(device=device).manual_seed(args.seed + tokens)
        hidden_storage = torch.randn(
            (tokens, args.input_stride),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        hidden = hidden_storage[:, :HIDDEN]
        workspace = IQ2RMoeWorkspace.allocate(
            tokens, TOPK, device=device, task_rows=args.task_rows
        )
        logits = tgemm.mm(hidden, weight, bias, otype=torch.bfloat16)
        reference_ids = torch.empty((tokens, TOPK), dtype=torch.int32, device=device)
        reference_weights = torch.empty(
            (tokens, TOPK), dtype=torch.float32, device=device
        )
        token_expert_indices = torch.empty_like(reference_ids)
        topk_softmax(
            reference_weights,
            reference_ids,
            token_expert_indices,
            logits,
            True,
        )
        _frontend(hidden, logits, workspace, args.task_rows)
        torch.cuda.synchronize()

        if not torch.equal(workspace.topk_ids[:tokens], reference_ids):
            raise AssertionError(f"fused router IDs mismatch at M={tokens}")
        torch.testing.assert_close(
            workspace.topk_weights[:tokens], reference_weights, rtol=0, atol=0
        )
        can_fold_bias = tokens <= 8
        if can_fold_bias:
            no_bias_logits = tgemm.mm(hidden, weight, None, otype=torch.bfloat16)
            _frontend(hidden, no_bias_logits, workspace, args.task_rows, bias)
            torch.cuda.synchronize()
            if not torch.equal(workspace.topk_ids[:tokens], reference_ids):
                raise AssertionError(
                    f"bias-folded fused router IDs mismatch at M={tokens}"
                )
            torch.testing.assert_close(
                workspace.topk_weights[:tokens], reference_weights, rtol=0, atol=0
            )

        def router_hot() -> None:
            tgemm.mm(hidden, weight, bias, otype=torch.bfloat16)

        def router_no_bias_hot() -> None:
            tgemm.mm(hidden, weight, None, otype=torch.bfloat16)

        def frontend_hot() -> None:
            _frontend(hidden, logits, workspace, args.task_rows)

        def chain_hot() -> None:
            current_logits = tgemm.mm(hidden, weight, bias, otype=torch.bfloat16)
            _frontend(hidden, current_logits, workspace, args.task_rows)

        def chain_no_bias_hot() -> None:
            current_logits = tgemm.mm(hidden, weight, None, otype=torch.bfloat16)
            _frontend(hidden, current_logits, workspace, args.task_rows, bias)

        def router_cold_batch() -> None:
            for current_weight in cold_weights:
                tgemm.mm(hidden, current_weight, bias, otype=torch.bfloat16)

        def router_hot_batch() -> None:
            for _ in cold_weights:
                tgemm.mm(hidden, weight, bias, otype=torch.bfloat16)

        def router_no_bias_hot_batch() -> None:
            for _ in cold_weights:
                tgemm.mm(hidden, weight, None, otype=torch.bfloat16)

        def chain_cold_batch() -> None:
            for current_weight in cold_weights:
                current_logits = tgemm.mm(
                    hidden, current_weight, bias, otype=torch.bfloat16
                )
                _frontend(hidden, current_logits, workspace, args.task_rows)

        def chain_hot_batch() -> None:
            for _ in cold_weights:
                current_logits = tgemm.mm(hidden, weight, bias, otype=torch.bfloat16)
                _frontend(hidden, current_logits, workspace, args.task_rows)

        benchmark_cases: list[tuple[str, Callable[[], None], int]] = [
            ("router_hot", router_hot, 1),
            ("frontend_hot", frontend_hot, 1),
            ("router_plus_frontend_hot", chain_hot, 1),
            ("router_hot_batched", router_hot_batch, cold_count),
            ("router_cold", router_cold_batch, cold_count),
            ("router_plus_frontend_hot_batched", chain_hot_batch, cold_count),
            ("router_plus_frontend_cold", chain_cold_batch, cold_count),
        ]
        if can_fold_bias:
            benchmark_cases.extend(
                [
                    ("router_no_bias_hot", router_no_bias_hot, 1),
                    (
                        "router_no_bias_plus_frontend_hot",
                        chain_no_bias_hot,
                        1,
                    ),
                    (
                        "router_no_bias_hot_batched",
                        router_no_bias_hot_batch,
                        cold_count,
                    ),
                ]
            )

        timings: dict[str, object] = {}
        for name, call, divisor in benchmark_cases:
            latency, samples = _benchmark(
                call,
                divisor=divisor,
                warmup=args.warmup,
                iterations=args.iterations,
                samples=args.samples,
            )
            timings[f"{name}_ms"] = latency
            timings[f"{name}_samples_ms"] = samples

        config = get_GEMM_A16W16_config(
            tokens,
            EXPERTS,
            HIDDEN,
            True,
            str(torch.bfloat16),
            str(torch.bfloat16),
        )
        no_bias_config = get_GEMM_A16W16_config(
            tokens,
            EXPERTS,
            HIDDEN,
            False,
            str(torch.bfloat16),
            str(torch.bfloat16),
        )
        records.append(
            {
                "M": tokens,
                "input_stride": hidden.stride(0),
                "task_rows": args.task_rows,
                "bias_fold_candidate": can_fold_bias,
                "cold_weight_copies": cold_count,
                "router_config": config,
                "router_no_bias_config": no_bias_config,
                **timings,
            }
        )

    return {
        "benchmark": "gpt-oss-bf16-router-through-iq2r-frontend",
        "device_binding": "HIP_VISIBLE_DEVICES=6",
        "device": torch.cuda.get_device_name(0),
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "model": str(args.model),
        "router_shard": str(shard),
        "router_layer": args.layer,
        "router_weight_shape": list(weight.shape),
        "router_weight_dtype": str(weight.dtype),
        "assumed_l3_mib": args.assumed_l3_mib,
        "gemm_config_file": os.environ.get("AITER_CONFIG_GEMM_BF16", "<default>"),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("/models/openai/gpt-oss-120b")
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 5, 8, 16])
    parser.add_argument("--task-rows", type=int, default=16)
    parser.add_argument("--input-stride", type=int, default=3072)
    parser.add_argument("--assumed-l3-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0x1709)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["records"]:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
