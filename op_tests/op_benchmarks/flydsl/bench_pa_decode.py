# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the current FlyDSL PA decode FP8/BF16 correctness-test matrix.

Run from the repository root:
    python -m op_tests.op_benchmarks.flydsl.bench_pa_decode --list
    python -m op_tests.op_benchmarks.flydsl.bench_pa_decode --case bf16-decode
    python -m op_tests.op_benchmarks.flydsl.bench_pa_decode --jsonl pa_decode.jsonl

Each case runs the correctness test's setup, FP32 reference checks, dispatch
assertions and, when requested by the case, graph replay checks before timing.
Timing reuses the work plan and input buffers and sums GPU kernel times for
eager decode calls (including reduction when dispatched). Input generation,
quantization, reference checks, plan creation and JIT compilation are excluded.
These test shapes measure latency; they are not a throughput or HBM benchmark.
"""

import argparse
import importlib.util
import json
import math
import statistics
from contextlib import nullcontext

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import run_perftest
from op_tests import test_flydsl_pa_decode as reference_tests


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def benchmark_case(parameter, device, args):
    record = {"id": parameter.id}
    try:
        # Restore the test's compilation spy before entering the timing loop.
        with pytest.MonkeyPatch.context() as monkeypatch:
            case, launch, check, arguments, _, plan = (
                reference_tests._run_pa_decode_case(
                    parameter.values[0], device, monkeypatch
                )
            )
        query, _, _, context, *_ = arguments
        record.update(
            status="passed",
            correct=True,
            gfx=get_gfx_runtime(),
            case={
                name: (
                    str(value).removeprefix("torch.")
                    if isinstance(value, torch.dtype)
                    else value
                )
                for name, value in case.items()
            },
            query_shape=list(query.shape),
            lengths=context.cpu().tolist(),
            plan_capacity=plan.capacity,
            timing="eager_gpu_kernel_sum",
            rounds=args.rounds,
            iters=args.iters,
            warmup=args.warmup,
            num_rotate_args=1,
        )
        latencies = []
        for _ in range(args.rounds):
            _, latency = run_perftest(
                launch,
                num_iters=args.iters,
                num_warmup=args.warmup,
                num_rotate_args=1,
            )
            if not math.isfinite(latency) or latency <= 0:
                raise RuntimeError(f"invalid GPU kernel time: {latency}")
            latencies.append(latency)
        # Repeated launches must still produce the validated output.
        check()
        record.update(
            us=statistics.median(latencies),
            us_rounds=latencies,
            spread_pct=100 * (max(latencies) - min(latencies)) / min(latencies),
        )
    except pytest.skip.Exception as error:
        record.update(status="skipped", correct=None, reason=str(error))
    # Record each failed case so the remaining matrix can still run.
    except (Exception, pytest.fail.Exception) as error:  # noqa: BLE001
        record.update(
            status="failed", correct=False, error=f"{type(error).__name__}: {error}"
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", help="exact test case ID; repeatable"
    )
    parser.add_argument("--filter", help="select case IDs containing this text")
    parser.add_argument(
        "--list", action="store_true", help="list selected IDs and exit"
    )
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--iters", type=positive_int, default=101)
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--jsonl", help="append a JSON record per case to this file")
    args = parser.parse_args()

    parameters = [
        parameter
        for parameter in reference_tests._PA_DECODE_CASES
        if "invalid_input" not in parameter.values[0]
        and parameter.values[0].get("kv_dtype", "bf16") != "mxfp4"
    ]
    if args.case:
        unknown = set(args.case) - {parameter.id for parameter in parameters}
        if unknown:
            parser.error(f"unknown case IDs: {', '.join(sorted(unknown))}")
        parameters = [p for p in parameters if p.id in args.case]
    if args.filter:
        parameters = [p for p in parameters if args.filter in p.id]
    if not parameters:
        parser.error("no cases selected")
    if args.list:
        for parameter in parameters:
            print(parameter.id)
        return
    if args.iters < 2:
        parser.error(
            "--iters must be at least 2; the profiler drops its first iteration"
        )
    if not torch.cuda.is_available():
        parser.error("ROCm is not available")
    if get_gfx_runtime() not in ("gfx942", "gfx950", "gfx1250"):
        parser.error(f"pa_decode is unsupported on {get_gfx_runtime()}")
    if importlib.util.find_spec("flydsl") is None:
        parser.error("FlyDSL is not installed")

    failed = False
    with (
        open(args.jsonl, "a") if args.jsonl else nullcontext() as sink,
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        for parameter in parameters:
            record = benchmark_case(parameter, torch.device("cuda"), args)
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            if sink:
                sink.write(line + "\n")
                sink.flush()
            failed |= record["status"] == "failed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
