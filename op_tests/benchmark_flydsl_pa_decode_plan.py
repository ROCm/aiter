# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark static versus GPU-planned PA, including each plan refresh.

Run from the repository root with
``python -m op_tests.benchmark_flydsl_pa_decode_plan --output plan_results.json``.
The shared correctness helper supplies sparse caches and checks the final result
against its FP32 causal reference. run_perftest uses allocation rotation.
"""

import argparse
import json
from pathlib import Path

import torch

from aiter.ops.flydsl.pa_decode import (
    get_recommended_splits,
    pa_decode,
    plan_pa_decode,
)
from aiter.test_common import run_perftest
from op_tests import test_flydsl_pa_decode as reference_tests

CASES = {
    "b1_uniform": [200000],
    "b8_uniform": [200000] * 8,
    "b200_uniform": [200000] * 200,
    "b8_one_long": [200003] + [257] * 7,
    "b8_ramp": [25000 * i + 3 for i in range(1, 9)],
    "b32_one_long": [200003] + [257] * 31,
    "b200_one_long": [200003] + [257] * 199,
    "b8_short": [257] * 8,
}


def benchmark_case(name, block_size, trans_v):
    lengths_host = CASES[name]
    result = {}

    def compare(*args, **kwargs):
        output, query, key, value, lengths, tables = args[:6]
        batch, heads, qlen = lengths.numel(), key.shape[1], args[7]
        rows = qlen * query.shape[1] // heads
        static_np = get_recommended_splits(
            batch, heads, 256 // block_size, max_context_length=max(lengths_host)
        )
        plan = plan_pa_decode(lengths, heads)
        saved, times = {}, {}
        for mode in ("static", "plan_each_call", "reuse_plan"):
            shape = (
                (batch, heads, static_np, rows)
                if mode == "static"
                else (heads, plan.capacity, rows)
            )
            pmax = torch.empty(shape, dtype=torch.float32)
            psum = torch.empty_like(pmax)
            pout = torch.empty((*shape, query.shape[2]), dtype=query.dtype)

            def run(out, q, k, v, ctx, bt, ks, vs, pm, ps, po):
                if mode == "plan_each_call":
                    plan_pa_decode(ctx, heads, plan=plan)
                pa_decode(
                    out,
                    q,
                    k,
                    v,
                    ctx,
                    bt,
                    args[6],
                    qlen,
                    static_np if mode == "static" else 256,
                    compute_type=k.dtype,
                    key_scale=ks,
                    value_scale=vs,
                    max_logits=pm,
                    exp_sums=ps,
                    temporary_output=po,
                    work_plan=None if mode == "static" else plan,
                )
                return out

            out, us = run_perftest(
                run,
                output,
                query,
                key,
                value,
                lengths,
                tables,
                args[12],
                args[13],
                pmax,
                psum,
                pout,
            )
            saved[mode] = out.clone()
            times[mode + "_us"] = us
            torch.testing.assert_close(
                out.float(), saved["static"].float(), atol=0.005, rtol=0.005
            )
        _, plan_us = run_perftest(
            lambda ctx: plan_pa_decode(ctx, heads, plan=plan).work_info, lengths
        )
        result.update(
            case=name,
            lengths=lengths_host,
            block=block_size,
            trans_v=trans_v,
            static_np=static_np,
            capacity=plan.capacity,
            actual_partitions=plan.reduce_info[:, 1].cpu().tolist(),
            plan_us=plan_us,
            **times,
            speedup=times["static_us"] / times["plan_each_call_us"],
        )
        output.copy_(saved["plan_each_call"])

    original = torch.ops.aiter.pa_decode_flydsl
    torch.ops.aiter.pa_decode_flydsl = compare
    try:
        reference_tests._run_mtp4_fused_reference_case(
            lengths_host, 1, block_size, trans_v, 256
        )
    finally:
        torch.ops.aiter.pa_decode_flydsl = original
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--block-size", type=int, nargs="+", choices=[16, 128], default=[16, 128]
    )
    parser.add_argument(
        "--trans-v", type=int, nargs="+", choices=[0, 1], default=[0, 1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference_tests._require_gpu()
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    results = []
    try:
        for name in args.cases:
            for block in args.block_size:
                for trans in args.trans_v:
                    result = benchmark_case(name, block, bool(trans))
                    results.append(result)
                    print("COMPARE " + json.dumps(result), flush=True)
                    if args.output is not None:
                        args.output.write_text(json.dumps(results, indent=2))
    finally:
        torch.set_default_device(previous)


if __name__ == "__main__":
    main()
