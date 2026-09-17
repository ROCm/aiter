# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark static versus GPU-planned PA, including each plan refresh.

Run from the repository root with
``python -m op_tests.benchmark_flydsl_pa_decode_plan --output plan_results.json``.
The shared correctness helper supplies sparse caches and checks the final result
against its FP32 causal reference. run_perftest uses allocation rotation.

The main uniform and one-long sweep covers batches 1, 2, 4, 8, 16 and 200:
uniform contexts have 200000 tokens; one-long contexts retain 200003 and 257.
An additional b64_uniform_100k case uses 64 equal 100000-token contexts.
Automatic partition selection uses host-known context work and MTP length.
Use --num-partitions for exact static splits, --workgroup-budget for the
planner's total CTA budget, and --max-partitions for its per-request limit.
The three --num-* timing options configure aiter's allocation-rotation timer.

Each PA mode reports time in us and effective per-rank bandwidth in GB/s and
TB/s (decimal units). Effective bytes count Q + O + valid K/V tokens + their
scales + referenced page-table entries + context lengths once. Sparse cache
holes, page padding, repeated loads, partition scratch and plan metadata are
excluded; KV traffic is not multiplied by MTP or TP. This is logical bandwidth,
not hardware-counter HBM traffic. plan_each_call includes plan refresh in its
time denominator; reuse_plan excludes it. plan_us measures only the refresh.
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
    "b2_uniform": [200000] * 2,
    "b4_uniform": [200000] * 4,
    "b8_uniform": [200000] * 8,
    "b16_uniform": [200000] * 16,
    "b64_uniform_100k": [100000] * 64,
    "b200_uniform": [200000] * 200,
    "b1_one_long": [200003],
    "b2_one_long": [200003] + [257],
    "b4_one_long": [200003] + [257] * 3,
    "b8_one_long": [200003] + [257] * 7,
    "b16_one_long": [200003] + [257] * 15,
    "b32_one_long": [200003] + [257] * 31,
    "b200_one_long": [200003] + [257] * 199,
    "b8_ramp": [25000 * i + 3 for i in range(1, 9)],
    "b8_short": [257] * 8,
}


def benchmark_case(
    name,
    block_size,
    trans_v,
    *,
    num_partitions=None,
    workgroup_budget=None,
    max_partitions=256,
    num_iters=101,
    num_warmup=2,
    num_rotate_args=0,
):
    lengths_host = CASES[name]
    result = {}

    def compare(*args, **kwargs):
        output, query, key, value, lengths, tables = args[:6]
        batch, heads, qlen = lengths.numel(), key.shape[1], args[7]
        rows = qlen * query.shape[1] // heads
        key_scale, value_scale = args[12:14]
        total_tokens = sum(lengths_host)
        total_pages = sum(
            (length + block_size - 1) // block_size for length in lengths_host
        )
        # The reference helper leaves holes between physical pages. Count only
        # useful tokens and table entries, including for heterogeneous lengths.
        # MTP queries share K/V and scales; these bytes are counted once.
        effective_bytes = (
            query.numel() * query.element_size()
            + output.numel() * output.element_size()
            + total_tokens
            * heads
            * query.shape[2]
            * (key.element_size() + value.element_size())
            + sum(
                (total_tokens * heads if scale.numel() > 1 else 1)
                * scale.element_size()
                for scale in (key_scale, value_scale)
            )
            + total_pages * tables.element_size()
            + lengths.numel() * lengths.element_size()
        )
        static_np = num_partitions
        if static_np is None:
            static_np = get_recommended_splits(
                batch,
                heads,
                256 // block_size,
                max_context_length=max(lengths_host),
                query_length=qlen,
            )
        plan = plan_pa_decode(
            lengths,
            heads,
            max_partitions=max_partitions,
            workgroup_budget=workgroup_budget,
            total_context_length=total_tokens,
            query_length=qlen,
        )
        saved, times, bandwidth = {}, {}, {}
        for mode in ("static", "plan_each_call", "reuse_plan"):
            partition_kwargs = (
                {"max_context_partition_num": static_np}
                if mode == "static"
                else {"work_plan": plan}
            )
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
                    compute_type=k.dtype,
                    key_scale=ks,
                    value_scale=vs,
                    max_logits=pm,
                    exp_sums=ps,
                    temporary_output=po,
                    **partition_kwargs,
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
                num_iters=num_iters,
                num_warmup=num_warmup,
                num_rotate_args=num_rotate_args,
            )
            saved[mode] = out.clone()
            times[mode + "_us"] = us
            bandwidth[mode + "_GB_s"] = effective_bytes / us / 1e3
            bandwidth[mode + "_TB_s"] = effective_bytes / us / 1e6
            torch.testing.assert_close(
                out.float(), saved["static"].float(), atol=0.005, rtol=0.005
            )
        _, plan_us = run_perftest(
            lambda ctx: plan_pa_decode(ctx, heads, plan=plan).work_info,
            lengths,
            num_iters=num_iters,
            num_warmup=num_warmup,
            num_rotate_args=num_rotate_args,
        )
        result.update(
            case=name,
            lengths=lengths_host,
            block=block_size,
            trans_v=trans_v,
            static_np=static_np,
            max_partitions=max_partitions,
            workgroup_budget=workgroup_budget,
            capacity=plan.capacity,
            actual_partitions=plan.num_partitions.cpu().tolist(),
            plan_us=plan_us,
            effective_bytes=effective_bytes,
            num_iters=num_iters,
            num_warmup=num_warmup,
            num_rotate_args=num_rotate_args,
            **times,
            **bandwidth,
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


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--block-size", type=int, nargs="+", choices=[16, 128], default=[16, 128]
    )
    parser.add_argument(
        "--trans-v", type=int, nargs="+", choices=[0, 1], default=[0, 1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--num-partitions",
        type=_positive_int,
        help="Exact static partition count (1..256); otherwise use auto selection.",
    )
    parser.add_argument(
        "--workgroup-budget",
        type=_positive_int,
        help="GPU plan CTA budget across KV heads; otherwise use the planner default.",
    )
    parser.add_argument(
        "--max-partitions",
        type=_positive_int,
        default=256,
        help="GPU plan per-sequence upper bound (1..256); independent of static NP.",
    )
    parser.add_argument("--num-iters", type=_positive_int, default=101)
    parser.add_argument("--num-warmup", type=_positive_int, default=2)
    parser.add_argument(
        "--num-rotate-args",
        type=int,
        default=0,
        help="Number of allocation sets; 0 uses aiter's automatic selection.",
    )
    args = parser.parse_args(argv)
    if args.num_partitions is not None and args.num_partitions > 256:
        parser.error("--num-partitions must be in [1, 256]")
    if args.max_partitions > 256:
        parser.error("--max-partitions must be in [1, 256]")
    if args.num_iters < 2:
        parser.error("--num-iters must be at least 2")
    if args.num_rotate_args < 0:
        parser.error("--num-rotate-args must be non-negative")
    return args


def main():
    args = _parse_args()
    reference_tests._require_gpu()
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    results = []
    try:
        for name in args.cases:
            for block in args.block_size:
                for trans in args.trans_v:
                    result = benchmark_case(
                        name,
                        block,
                        bool(trans),
                        num_partitions=args.num_partitions,
                        workgroup_budget=args.workgroup_budget,
                        max_partitions=args.max_partitions,
                        num_iters=args.num_iters,
                        num_warmup=args.num_warmup,
                        num_rotate_args=args.num_rotate_args,
                    )
                    results.append(result)
                    print("COMPARE " + json.dumps(result), flush=True)
                    if args.output is not None:
                        args.output.write_text(json.dumps(results, indent=2))
    finally:
        torch.set_default_device(previous)


if __name__ == "__main__":
    main()
