# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the validated gfx950 QSA sparse paged-GQA geometry."""

import argparse
import statistics

import torch
import triton

from aiter.ops.triton.attention.qsa import qsa_sparse_paged_gqa


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _inputs(index_order: str, seed: int):
    torch.manual_seed(seed)
    rows, query_heads, head_dim = 16, 10, 128
    page_size, pages, kv_heads = 16, 129, 2
    q = torch.randn(rows, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        pages, page_size, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    block_table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    token_to_request = torch.arange(rows, device="cuda", dtype=torch.int32)
    valid = torch.arange(2048, device="cuda", dtype=torch.int32)
    if index_order == "ordered":
        indices = valid.repeat(rows, 1)
    else:
        indices = torch.stack(
            [valid[torch.randperm(valid.numel(), device="cuda")] for _ in range(rows)]
        )
    indices = torch.cat(
        (
            indices,
            torch.full((rows, 3), -1, device="cuda", dtype=torch.int32),
        ),
        dim=1,
    )
    return q, k, v, indices, block_table, token_to_request


def _print_environment(args) -> None:
    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "unknown")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({architecture})")
    print(
        f"ROCm: {torch.version.hip}  PyTorch: {torch.__version__}  "
        f"Triton: {triton.__version__}"
    )
    print(
        "Config: rows=16 query_heads=10 kv_heads=2 head_dim=128 "
        "pages=129 page_size=16 selection_width=2051 "
        f"index_order={args.order} seed={args.seed} runs={args.runs} "
        f"warmup={args.warmup} rep={args.rep}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("ordered", "random"), default="ordered")
    parser.add_argument("--runs", type=_positive_int, default=7)
    parser.add_argument("--warmup", type=_positive_int, default=100)
    parser.add_argument("--rep", type=_positive_int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    _print_environment(args)
    inputs = _inputs(args.order, args.seed)
    results = {"triton": [], "gluon": []}
    for run in range(args.runs):
        backend_order = ("triton", "gluon") if run % 2 == 0 else ("gluon", "triton")
        outputs = {}
        print(f"Run {run + 1}/{args.runs}: {' -> '.join(backend_order)}")
        for backend in backend_order:
            fn = lambda backend=backend: qsa_sparse_paged_gqa(*inputs, backend=backend)
            outputs[backend] = fn()
            p50, p20, p80 = triton.testing.do_bench(
                fn, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
            )
            results[backend].append((p20, p50, p80))
            print(
                f"  {backend}: p20={p20:.6f} ms " f"p50={p50:.6f} ms p80={p80:.6f} ms"
            )
        torch.testing.assert_close(
            outputs["gluon"], outputs["triton"], rtol=2e-2, atol=2e-2
        )

    for backend, samples in results.items():
        p20s, p50s, p80s = zip(*samples)
        mean_p50 = statistics.fmean(p50s)
        cv_p50 = statistics.pstdev(p50s) / mean_p50 * 100
        print(
            f"Aggregate {backend}: p20_mean={statistics.fmean(p20s):.6f} ms "
            f"p50_mean={mean_p50:.6f} ms p80_mean={statistics.fmean(p80s):.6f} ms "
            f"p50_stdev={statistics.pstdev(p50s):.6f} ms p50_cv={cv_p50:.3f}%"
        )

    triton_p50 = statistics.fmean(sample[1] for sample in results["triton"])
    gluon_p50 = statistics.fmean(sample[1] for sample in results["gluon"])
    print(
        f"Aggregate speedup: {triton_p50 / gluon_p50:.3f}x "
        f"({(triton_p50 - gluon_p50) / triton_p50 * 100:.1f}% lower latency)"
    )


if __name__ == "__main__":
    main()
