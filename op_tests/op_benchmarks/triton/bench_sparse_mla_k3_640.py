# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the gfx1250 fp8_k3_mla_640 path: sparse_mla_fwd over the 640 B
record, and the record insert, Q pack and gather kernels.

Usage:
  python op_tests/op_benchmarks/triton/bench_sparse_mla_k3_640.py
  python op_tests/op_benchmarks/triton/bench_sparse_mla_k3_640.py --op attn --num_heads 12 24
  python op_tests/op_benchmarks/triton/bench_sparse_mla_k3_640.py --op insert gather q_pack
  python op_tests/op_benchmarks/triton/bench_sparse_mla_k3_640.py --metric bandwidth
"""

import torch
import triton

from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
from aiter.ops.triton.quant.fused_mxfp8_quant import (
    k3_mla_640_concat_and_cache,
    k3_mla_640_cp_gather_and_upconvert,
    k3_mla_640_q_pack,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

NOPE, ROPE, REC = 512, 64, 640
D_QK = NOPE + ROPE
BLOCK_SIZE = 64
DEV = "cuda"


def rand_bf16(*shape, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * 0.5).bfloat16()


def filled_cache(rows, seed=0):
    cache = torch.zeros(
        triton.cdiv(rows, BLOCK_SIZE), BLOCK_SIZE, REC, dtype=torch.uint8, device=DEV
    )
    kv = rand_bf16(rows, D_QK, seed=seed)
    slots = torch.arange(rows, device=DEV)
    k3_mla_640_concat_and_cache(kv[:, :NOPE], kv[:, NOPE:], cache, slots)
    return cache


def csr(lens):
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32, device=DEV)
    indptr[1:] = torch.as_tensor(lens, device=DEV).cumsum(0)
    return indptr


def attn_case(num_seqs, num_tokens, num_heads, context, topk):
    """One sequence per pool slice; each token selects min(pos + 1, topk) keys."""
    g = torch.Generator(device=DEV).manual_seed(1)
    cache = filled_cache(num_seqs * context)
    rows, lens = [], []
    for s in range(num_seqs):
        for t in range(num_tokens):
            pos = context - 1 if num_tokens == 1 else t
            n = min(pos + 1, topk)
            sel = torch.randperm(pos + 1, generator=g, device=DEV)[:n]
            rows.append((sel + s * context).int())
            lens.append(n)
    q = rand_bf16(num_seqs * num_tokens, num_heads, D_QK, seed=2)
    q = k3_mla_640_q_pack(q[..., :NOPE], q[..., NOPE:])
    return q, cache, csr(lens), torch.cat(rows)


def time_ms(func, args):
    if args.no_cudagraph:
        return triton.testing.do_bench(func, warmup=25, rep=100)
    return triton.testing.do_bench_cudagraph(func, rep=100)


def bench_attn(args, phase, num_seqs, num_tokens, num_heads, metric):
    q, cache, indptr, indices = attn_case(
        num_seqs, num_tokens, num_heads, args.context, args.topk
    )
    out = torch.empty(q.shape[0], num_heads, NOPE, dtype=torch.bfloat16, device=DEV)

    def func():
        sparse_mla_fwd(
            q, cache, indptr, indices, D_QK**-0.5, kv_format="fp8_k3_mla_640", out=out
        )

    ms = time_ms(func, args)
    nnz = int(indptr[-1])
    tokens = q.shape[0]
    moved = nnz * (REC + 4) + tokens * num_heads * (REC + NOPE * 2)
    flops = 2.0 * num_heads * nnz * (D_QK + NOPE)
    if metric == "throughput":
        return flops / (ms * 1e-3) / 1e12
    return moved / (ms * 1e-3) / 1e9 if metric == "bandwidth" else ms


def bench_record(args, op, rows, num_heads, metric):
    if op == "insert":
        kv = rand_bf16(rows, D_QK)
        cache = torch.zeros(
            triton.cdiv(rows, BLOCK_SIZE),
            BLOCK_SIZE,
            REC,
            dtype=torch.uint8,
            device=DEV,
        )
        slots = torch.randperm(rows, device=DEV)
        nope, pe = kv[:, :NOPE], kv[:, NOPE:]

        def func():
            k3_mla_640_concat_and_cache(nope, pe, cache, slots)

        moved = rows * (D_QK * 2 + REC + 8)
    elif op == "q_pack":
        q = rand_bf16(rows, num_heads, D_QK)
        nope, pe = q[..., :NOPE], q[..., NOPE:]
        packed = torch.empty(rows, num_heads, REC, dtype=torch.uint8, device=DEV)

        def func():
            k3_mla_640_q_pack(nope, pe, out=packed)

        moved = rows * num_heads * (D_QK * 2 + REC)
    else:
        cache = filled_cache(rows)
        blocks = cache.shape[0]
        table = torch.randperm(blocks, device=DEV).int().reshape(1, blocks)
        starts = torch.zeros(1, dtype=torch.int32, device=DEV)
        dst = torch.empty(rows, D_QK, dtype=torch.bfloat16, device=DEV)

        def func():
            k3_mla_640_cp_gather_and_upconvert(cache, dst, table, starts, 1)

        moved = rows * (REC + D_QK * 2)

    ms = time_ms(func, args)
    return moved / (ms * 1e-3) / 1e9 if metric == "bandwidth" else ms


def run_benchmark(args):
    ylabel = {"time": "Time (ms)", "throughput": "TFLOPs", "bandwidth": "GB/s"}
    benchmarks = []
    if "attn" in args.op:
        benchmarks.append(
            triton.testing.Benchmark(
                x_names=["phase", "num_seqs", "num_tokens"],
                x_vals=[
                    ["decode" if t == 1 else "prefill", s, t]
                    for s in args.num_seqs
                    for t in args.num_tokens
                ],
                line_arg="num_heads",
                line_vals=args.num_heads,
                line_names=[f"H={h}" for h in args.num_heads],
                ylabel=ylabel[args.metric],
                plot_name=get_caller_name_no_ext() + "_attn",
                args={"metric": args.metric},
            )
        )
    records = [op for op in args.op if op != "attn"]
    if records:
        benchmarks.append(
            triton.testing.Benchmark(
                x_names=["rows"],
                x_vals=args.rows,
                line_arg="op",
                line_vals=records,
                line_names=records,
                ylabel=ylabel[args.metric],
                plot_name=get_caller_name_no_ext() + "_record",
                args={"metric": args.metric, "num_heads": args.num_heads[0]},
            )
        )

    @triton.testing.perf_report(benchmarks)
    def bench_sparse_mla_k3_640(metric, num_heads, op=None, rows=None, **kw):
        if op is None:
            return bench_attn(args, metric=metric, num_heads=num_heads, **kw)
        return bench_record(args, op, rows, num_heads, metric)

    bench_sparse_mla_k3_640.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = get_parser(kernel_name="Sparse MLA fp8_k3_mla_640 Benchmark")
    parser.add_argument(
        "--op",
        nargs="+",
        choices=["attn", "insert", "q_pack", "gather"],
        default=["attn", "insert", "q_pack", "gather"],
    )
    parser.add_argument("--num_seqs", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument(
        "--num_tokens",
        type=int,
        nargs="+",
        default=[1],
        help="query tokens per sequence: 1 is decode, more is prefill",
    )
    parser.add_argument("--num_heads", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1024, 8192, 65536],
        help="rows for the insert, Q pack and gather kernels",
    )
    parser.add_argument(
        "--no_cudagraph",
        action="store_true",
        help="time without CUDA graphs; decode then measures host launch time",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    override_parser_action(
        parser,
        action="--metric",
        choices=["time", "throughput", "bandwidth"],
        default="time",
    )
    args = parser.parse_args()
    if not arch_info.get_arch().startswith("gfx1250"):
        raise SystemExit("fp8_k3_mla_640 runs on gfx1250 only")
    if any(t > args.context for t in args.num_tokens):
        raise SystemExit("--num_tokens cannot exceed --context")
    run_benchmark(args)


if __name__ == "__main__":
    main()
