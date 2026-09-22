# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare GPT-OSS IQ2R SwiGLU/MXFP8 launch families in one process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch

from aiter.ops.iq2r import iq2r_swiglu_quant_out

INTERMEDIATE = 2880


def _require_device_contract() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("IQ2R launch sweep requires gfx950")


def _capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark_pair(candidate, control, warmup: int, iterations: int, samples: int):
    for index in range(warmup):
        (candidate if index % 2 == 0 else control).replay()
        (control if index % 2 == 0 else candidate).replay()
    torch.cuda.synchronize()
    candidate_samples = []
    control_samples = []
    for index in range(samples):
        if index % 2 == 0:
            candidate_samples.append(_time_graph(candidate, iterations))
            control_samples.append(_time_graph(control, iterations))
        else:
            control_samples.append(_time_graph(control, iterations))
            candidate_samples.append(_time_graph(candidate, iterations))
    return candidate_samples, control_samples


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_device_contract()
    device = torch.device("cuda")
    records = []
    for rows in args.rows:
        generator = torch.Generator(device=device).manual_seed(args.seed + rows)
        gate_up = (
            torch.randn((rows, 2 * INTERMEDIATE), generator=generator, device=device)
            * 4.0
        ).to(torch.bfloat16)
        candidate_output = torch.empty(
            (rows, INTERMEDIATE), dtype=torch.float8_e4m3fn, device=device
        )
        candidate_scales = torch.empty(
            (rows, INTERMEDIATE // 32), dtype=torch.uint8, device=device
        )
        control_output = torch.empty_like(candidate_output)
        control_scales = torch.empty_like(candidate_scales)

        os.environ["IQ2R_SWIGLU_QUANT_FAMILY"] = args.candidate
        candidate_graph = _capture(
            lambda: iq2r_swiglu_quant_out(gate_up, candidate_output, candidate_scales)
        )
        os.environ["IQ2R_SWIGLU_QUANT_FAMILY"] = args.control
        control_graph = _capture(
            lambda: iq2r_swiglu_quant_out(gate_up, control_output, control_scales)
        )

        candidate_samples, control_samples = _benchmark_pair(
            candidate_graph,
            control_graph,
            args.warmup,
            args.iterations,
            args.samples,
        )
        candidate_graph.replay()
        control_graph.replay()
        torch.cuda.synchronize()
        output_exact = torch.equal(
            candidate_output.view(torch.uint8), control_output.view(torch.uint8)
        )
        scales_exact = torch.equal(candidate_scales, control_scales)
        candidate_latency = median(candidate_samples)
        control_latency = median(control_samples)
        records.append(
            {
                "rows": rows,
                "candidate_ms": candidate_latency,
                "control_ms": control_latency,
                "candidate_speedup": control_latency / candidate_latency,
                "candidate_samples_ms": candidate_samples,
                "control_samples_ms": control_samples,
                "output_exact": output_exact,
                "scales_exact": scales_exact,
            }
        )

    os.environ.pop("IQ2R_SWIGLU_QUANT_FAMILY", None)
    return {
        "candidate": args.candidate,
        "control": args.control,
        "device_binding": "HIP_VISIBLE_DEVICES=6",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[4, 8, 16, 20, 32, 64, 128, 256, 512, 4096],
    )
    parser.add_argument("--candidate", default="parallel8")
    parser.add_argument("--control", default="group32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0x51A6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for record in result["records"]:
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
