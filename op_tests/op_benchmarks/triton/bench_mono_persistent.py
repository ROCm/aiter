# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from aiter import logger
from aiter.ops.triton.quant.fp8_group32 import quantize_fp8_group32
from aiter.ops.triton.quant.fused_mxfp8_quant import fused_dual_rmsnorm_mxfp8_quant
from aiter.ops.triton.quant.mono_utils import require_wave32
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.utils import mono_native as native


def inputs(op, rows):
    if op == "a03":
        storage = torch.randn((rows, 1792), device="cuda", dtype=torch.bfloat16)
        wq = torch.randn((1280,), device="cuda", dtype=torch.bfloat16)
        wk = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
        return ((storage[:, :1280], storage[:, 1280:], wq, wk, 1e-20),)
    return ((torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16),),)


def distribution(values):
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    return {
        "median_us": statistics.median(values),
        "p10_us": deciles[0],
        "p90_us": deciles[8],
        "samples_us": values,
    }


def benchmark(op, rows, cache, trials):
    options = {"min_pool_len": 1, "max_pool_len": 1, "target_rot_pool_sz_mb": 0.0}
    bench = RotatingBufferBench.from_dict(options if cache == "hot" else {})
    bench.build_pool([lambda: inputs(op, rows)], [("args",)])
    if op == "a03":
        baseline = functools.partial(
            native.fused_dual_rmsnorm_mxfp8_quant, amax_floor=1e-10
        )
        candidate = functools.partial(
            fused_dual_rmsnorm_mxfp8_quant, amax_floor=1e-10, persistent=True
        )
    else:
        baseline = native.quantize_fp8
        candidate = functools.partial(quantize_fp8_group32, persistent=True)
    functions = {
        "native": lambda args: baseline(*args),
        "persistent": lambda args: candidate(*args),
    }
    samples = {key: [] for key in functions}
    for trial in range(trials):
        order = ("native", "persistent") if trial % 2 == 0 else ("persistent", "native")
        for provider in order:
            samples[provider].append(1000 * bench.bench(functions[provider]))
    result = {key: distribution(values) for key, values in samples.items()}
    result.update(
        op=op,
        rows=rows,
        cache=cache,
        pool_len=len(bench.rot_bufs),
        kernels_per_graph=max(bench.kernels_per_graph, len(bench.rot_bufs)),
        replays_per_trial=bench.num_replays,
        ratio_ps_native=statistics.median(samples["persistent"])
        / statistics.median(samples["native"]),
    )
    logger.info(
        "%s rows=%s cache=%s PS/native=%s", op, rows, cache, result["ratio_ps_native"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Paired native/P256 graph timings on gfx1250"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--rows", type=int, nargs="+", default=[96, 257, 513])
    parser.add_argument("--cache", choices=["hot", "rotating", "both"], default="both")
    args = parser.parse_args()
    if args.trials < 3 or any(rows <= 0 for rows in args.rows):
        parser.error("Use at least three trials and positive row counts")
    require_wave32()
    torch.manual_seed(451)
    root = Path(__file__).resolve().parents[3]
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "gpu": str(props),
        "aiter_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ),
        "timing": "RotatingBufferBench graph replay; each sample is mean per launch in one trial",
        "results": [],
    }
    caches = ["hot", "rotating"] if args.cache == "both" else [args.cache]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for op in ("a03", "a09"):
        for rows in args.rows:
            for cache in caches:
                report["results"].append(benchmark(op, rows, cache, args.trials))
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
