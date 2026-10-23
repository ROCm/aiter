# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone ABI128 scheduler latency sweep using preallocated output.

Run from the repository root with:
    python3 -m op_tests.op_benchmarks.triton.bench_q256_schedule
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.q256_schedule import q256_make_job_map
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.triton_tests.moe.test_q256_schedule import (
    make_inputs,
    reference_job_map,
)


@benchmark()
def bench_q256_schedule(boundary, tokens):
    cpu_inputs = make_inputs(boundary=boundary, tokens=tokens)
    reference = torch.tensor(
        reference_job_map(*cpu_inputs), dtype=torch.int32, device="cuda"
    )
    sorted_ids, sorted_experts, counts = [
        torch.tensor(values, dtype=torch.int32, device="cuda") for values in cpu_inputs
    ]
    job_map = torch.empty(1024, dtype=torch.int32, device="cuda")
    candidates = {
        "triton": lambda: q256_make_job_map(
            sorted_ids, sorted_experts, counts, job_map
        ),
    }
    # Logical traffic: eight scalar count/suffix reads, routed fragment probes,
    # and the full map write. This is not measured HBM traffic (cache reuse).
    probe_bytes = min(boundary, 1024) * 7 * 4 if tokens == 8192 else 0
    nbytes = 8 * (2 + 33) * 4 + probe_bytes + job_map.numel() * job_map.element_size()
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(reference.float(), out.float(), rtol=0, atol=0)
        if not torch.equal(reference, out):
            raise AssertionError("Q256 job map differs from the independent CPU oracle")
        ret[f"{name} us"] = us
        # Scheduler work is integer-only; there are no floating-point FLOPs.
        ret[f"{name} TFLOPS"] = 0.0
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if not torch.cuda.is_available():
        aiter.logger.warning("Q256 scheduler benchmark requires a GPU; skipping")
        return
    if get_gfx() not in ("gfx942", "gfx950"):
        aiter.logger.warning(
            "Q256 scheduler benchmark unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(description="Q256 ABI128 scheduler sweep")
    parser.add_argument(
        "--boundary", type=int, nargs="+", default=[256, 257, 512, 513, 960, 961]
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 4096])
    args = parser.parse_args()
    if any(boundary < 1 for boundary in args.boundary) or any(
        tokens < 1 for tokens in args.tokens
    ):
        parser.error("boundary and tokens must be positive")
    rows = [
        bench_q256_schedule(boundary, tokens)
        for boundary, tokens in itertools.product(args.boundary, args.tokens)
    ]
    aiter.logger.info(
        "Q256 scheduler summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
