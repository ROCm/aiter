# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare GLM-5.3 indexed route-reduction kernel families.

Each family is captured into a separate HIP graph from the same native module.
The graphs use distinct output buffers and are timed in alternating forward and
reverse order so cache and clock drift do not systematically favor a family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch

import aiter
from aiter.ops.iq2r import iq2r_route_reduce_indexed_out

HIDDEN = 6144
TOPK = 8
FAMILIES = ("generic", "auto", "glm1", "glm2", "glm4")


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise RuntimeError("set HIP_VISIBLE_DEVICES=0 for this benchmark")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("this benchmark requires gfx950")


def _capture(call) -> torch.cuda.CUDAGraph:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return 1000.0 * start.elapsed_time(end) / iterations


def _module_identity() -> dict[str, object]:
    root = Path(aiter.__file__).resolve().parent / "jit"
    candidates = sorted(root.glob("module_iq2r_moe.so*"))
    if not candidates:
        return {"path": None, "sha256": None, "size": None}
    module = candidates[0]
    return {
        "path": str(module),
        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "size": module.stat().st_size,
    }


def _run_tokens(args: argparse.Namespace, tokens: int) -> dict[str, object]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed + tokens * 9176)
    routes = tokens * TOPK
    route_output = torch.randn(
        (routes, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    route_weights = torch.softmax(
        torch.randn(
            (tokens, TOPK),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    ).contiguous()
    scatter_indices = torch.randperm(
        routes, dtype=torch.int32, device=device, generator=generator
    )
    outputs = {
        family: torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        for family in args.families
    }
    graphs: dict[str, torch.cuda.CUDAGraph] = {}

    for family in args.families:
        if family == "auto":
            os.environ.pop("IQ2R_ROUTE_REDUCE_FAMILY", None)
        else:
            os.environ["IQ2R_ROUTE_REDUCE_FAMILY"] = family

        def call(selected: str = family) -> None:
            iq2r_route_reduce_indexed_out(
                route_output,
                route_weights,
                scatter_indices,
                outputs[selected],
                topk=TOPK,
            )

        graphs[family] = _capture(call)

    os.environ["IQ2R_ROUTE_REDUCE_FAMILY"] = "generic"
    for graph in graphs.values():
        for _ in range(args.warmup):
            graph.replay()
    torch.cuda.synchronize()

    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    baseline = outputs["generic"]
    correctness = {}
    for family, output in outputs.items():
        delta = (output.float() - baseline.float()).abs()
        correctness[family] = {
            "exact": bool(torch.equal(output, baseline)),
            "max_abs": float(delta.max().item()),
            "mismatches": int((output != baseline).sum().item()),
        }

    samples = {family: [] for family in args.families}
    forward = list(args.families)
    reverse = list(reversed(forward))
    for sample in range(args.samples):
        order = forward if sample % 2 == 0 else reverse
        for family in order:
            samples[family].append(_time_graph(graphs[family], args.iterations))

    medians = {family: median(values) for family, values in samples.items()}
    generic_us = medians["generic"]
    return {
        "tokens": tokens,
        "routes": routes,
        "correctness": correctness,
        "families": {
            family: {
                "median_us": medians[family],
                "samples_us": samples[family],
                "speedup_vs_generic": generic_us / medians[family],
                "delta_percent_vs_generic": 100.0
                * (generic_us / medians[family] - 1.0),
            }
            for family in args.families
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    unknown = set(args.families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    if "generic" not in args.families:
        raise ValueError("the generic control family is required")

    _require_device_contract()
    records = [_run_tokens(args, tokens) for tokens in args.tokens]
    result = {
        "benchmark": "glm53-iq2r-route-reduce-family-pair",
        "protocol": {
            "hidden": HIDDEN,
            "topk": TOPK,
            "families": args.families,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "samples": args.samples,
            "timing_order": "alternating forward/reverse",
            "separate_output_buffers": True,
            "device": torch.cuda.get_device_name(0),
            "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
            "module": _module_identity(),
        },
        "records": records,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not all(
        family["exact"]
        for record in records
        for family in record["correctness"].values()
    ):
        raise RuntimeError("a route-reduction candidate was not bit-exact")


if __name__ == "__main__":
    main()
