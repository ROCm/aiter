#!/usr/bin/env python3
"""Benchmark gfx1250 gather_kv_b_proj cluster variants in one GPU session."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from aiter.ops.flydsl.gather_kv_b_proj_gfx1250 import _run_projection
from aiter.ops.triton.utils.shuffle import shuffle_weight
from flydsl.runtime.device import get_rocm_arch
from op_tests.test_flydsl_gather_kv_b_proj import (
    DIMS_DEEPSEEK,
    _check_output,
    _make_case,
)


def _parse_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--cluster-n", type=_parse_ints, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    case = _make_case(args.m, args.heads, dims=DIMS_DEEPSEEK)
    weight = shuffle_weight(case["weight"], layout=(16, 16))

    def run(cluster_n: int) -> None:
        _run_projection(
            case["k_buffer"],
            case["kv_indices"],
            case["k_scale"],
            weight,
            case["weight_scale"],
            case["k_prefix"],
            case["v_prefix"],
            args.heads,
            args.m,
            cluster_n=cluster_n,
        )

    results: dict[int, dict] = {}
    valid: list[int] = []
    for cluster_n in args.cluster_n:
        try:
            run(cluster_n)
            torch.cuda.synchronize()
            _check_output(case)
            valid.append(cluster_n)
            results[cluster_n] = {"status": "ok", "samples_us": []}
        except Exception as exc:  # noqa: BLE001 - tuner records invalid candidates
            results[cluster_n] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    for cluster_n in valid:
        for _ in range(args.warmup):
            run(cluster_n)
    torch.cuda.synchronize()

    for round_idx in range(args.rounds):
        order = valid if round_idx % 2 == 0 else list(reversed(valid))
        for cluster_n in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.calls):
                run(cluster_n)
            end.record()
            end.synchronize()
            results[cluster_n]["samples_us"].append(
                start.elapsed_time(end) * 1000.0 / args.calls
            )

    flops = 2.0 * args.m * args.heads * 256 * 512
    for cluster_n in valid:
        samples = results[cluster_n]["samples_us"]
        median_us = statistics.median(samples)
        results[cluster_n].update(
            median_us=median_us,
            min_us=min(samples),
            max_us=max(samples),
            tflops=flops / median_us / 1.0e6,
        )

    payload = {
        "arch": str(get_rocm_arch()),
        "torch": torch.__version__,
        "m": args.m,
        "heads": args.heads,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "calls": args.calls,
        "results": {str(key): value for key, value in results.items()},
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
