# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for sparse_mla_fwd (gfx950 gluon, separated-rope MLA).

Both dot precisions are timed side by side, since the fp8 matrix-core path is
the interesting one and only differs by a flag.

A shape is num_seqs sequences of num_tokens query tokens each. Prefill and decode
are the same operator here, one program per query token over that token's own
gathered KV, so there is no phase to pick: num_tokens=1 is a decode step and
anything above that is prefill.

The cache is flushed between iterations by default, which matters: a decode shape
runs up to twice as fast when the loop is allowed to re-read its KV out of cache.

Usage:
  python op_tests/op_benchmarks/triton/bench_sparse_mla.py
  python op_tests/op_benchmarks/triton/bench_sparse_mla.py --num_seqs 1 --num_tokens 8192
  python op_tests/op_benchmarks/triton/bench_sparse_mla.py --num_seqs 64 --metric bandwidth
"""

import argparse

import torch
import triton
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from aiter.ops.triton.attention.sparse_mla import _mla_num_splits, sparse_mla_fwd
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
D_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
E4M3_MAX = 448.0

# Both kernels the launch runs are named after the gluon kernel, the combine as
# _pa_decode_sparse_reduce, so one substring picks up both and nothing else.
KERNEL_MATCH = "_pa_decode_sparse"
FLUSH_BYTES = 512 << 20

_flush_buf = None


def flush_l2():
    """Push a buffer bigger than the last-level cache through it.

    A benchmark loop re-reads the same KV every iteration, so without this the
    gather is served from cache and the numbers describe the cache rather than
    the gather. Any reuse inside a single launch is real and survives this.
    """
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    _flush_buf.fill_(1)


def device_time_ms(func, warmup=25, rep=100, flush=True):
    """Mean GPU time per call, counting only this op's kernels.

    do_bench would measure wall time instead, and a decode launch is small
    enough that the wrapper's own python cost exceeds it, which is enough to make
    the faster kernel look like the slower one. Filtering by kernel name also
    keeps the cache flush out of the total.
    """
    for _ in range(warmup):
        if flush:
            flush_l2()
        func()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(rep):
            if flush:
                flush_l2()
            func()
        torch.cuda.synchronize()
    total = sum(
        e.self_device_time_total
        for e in prof.key_averages()
        if e.device_type == DeviceType.CUDA
        and KERNEL_MATCH in e.key
        and e.self_device_time_total > 0
    )
    if total == 0:
        raise RuntimeError(f"no kernel matching {KERNEL_MATCH!r} was profiled")
    return total / rep / 1e3


def bytes_moved(num_tokens, num_heads, nnz, kv_elem_bytes, num_splits):
    """Bytes the launch actually moves, counting rows as gathered.

    nnz is the summed per-token top-k length, so a row shared by several query
    tokens counts once per token that reads it. That is what the kernel loads,
    not the unique footprint. Split-K partials get written and read back by the
    combine, so they count twice.
    """
    kv = nnz * D_QK * kv_elem_bytes           # the gather, and the bulk of it
    idx = nnz * 4                             # int32 index stream, read once
    q = num_tokens * num_heads * D_QK * kv_elem_bytes
    out = num_tokens * num_heads * KV_LORA_RANK * 2
    partials = 0
    if num_splits > 1:
        # [tokens, splits, heads, lora] bf16 accumulator plus the m/l pair
        partials = 2 * num_tokens * num_splits * num_heads * (KV_LORA_RANK * 2 + 8)
    return kv + idx + q + out + partials


def build_case(num_seqs, num_tokens, num_heads, context, topk, device="cuda"):
    """Build one launch the way vLLM's sparse-MLA path lays it out.

    Every sequence owns its own slice of the pool, so there is no cross-request
    KV sharing to inflate the cache hit rate. Per-token KV length is
    min(pos + 1, topk), which is what the index converter emits.

    A single query token per sequence is a decode step, so it sits at the end of
    the context. More than one means prefill, and they walk positions 0..n-1.
    """
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    pool = num_seqs * context

    kv = torch.randn(pool, D_QK, dtype=torch.bfloat16, device=device) * 0.125
    q = (
        torch.randn(
            num_seqs * num_tokens, num_heads, D_QK, dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )

    rows, lens = [], []
    for s in range(num_seqs):
        base = s * context
        for t in range(num_tokens):
            pos = context - 1 if num_tokens == 1 else t
            if pos + 1 <= topk:
                sel = torch.arange(pos + 1, dtype=torch.int64)  # everything fits
            else:
                sel = torch.randperm(pos + 1, generator=gen)[:topk]
            rows.append((sel + base).to(torch.int32))
            lens.append(sel.numel())

    indices = torch.cat(rows).to(device)
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)

    # fp8 dots want the flat per-tensor format; bf16 dots read kv as it is.
    scale = (kv.float().abs().amax() / E4M3_MAX).clamp_min(1e-30).reshape(1)
    kv_fp8 = (
        (kv.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    ).view(torch.uint8)

    return q, kv, kv_fp8, scale.to(device), indices, indptr.to(device)


def run_benchmark(args):
    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "TFLOPs"
    else:
        ylabel = "GB/s"

    # "phase" is a label, not a knob: it is num_tokens read out loud. It also
    # keeps the row from being all-numeric, which is what makes perf_report print
    # the shape columns as ints instead of floats.
    x_vals_list = [
        ["decode" if tokens == 1 else "prefill", seqs, tokens, args.num_heads,
         args.context, args.topk]
        for seqs in args.num_seqs
        for tokens in args.num_tokens
    ]

    benchmark = triton.testing.Benchmark(
        x_names=["phase", "num_seqs", "num_tokens", "num_heads", "context",
                 "topk"],
        x_vals=x_vals_list,
        line_arg="dots",
        line_vals=["bf16", "fp8"],
        line_names=["bf16 dots", "fp8 dots"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_sparse_mla(phase, num_seqs, num_tokens, num_heads, context, topk,
                         dots, metric, **kwargs):
        q, kv, kv_fp8, kv_scale, indices, indptr = build_case(
            num_seqs, num_tokens, num_heads, context, topk
        )
        sm_scale = D_QK**-0.5
        out = torch.empty(
            q.shape[0], num_heads, KV_LORA_RANK, dtype=torch.bfloat16,
            device=q.device,
        )
        if dots == "fp8":
            # Hand q over already quantized, the way production does. Letting
            # the wrapper do it would put an amax and a cast inside the timed
            # region, which is more than a decode launch costs.
            cache, scale = kv_fp8, kv_scale
            q_scale = (q.float().abs().amax() / E4M3_MAX).clamp_min(1e-30).reshape(1)
            q = (
                (q.float() / q_scale).clamp(-E4M3_MAX, E4M3_MAX)
                .to(torch.float8_e4m3fn)
            )
        else:
            cache, scale, q_scale = kv, None, None

        def func():
            return sparse_mla_fwd(
                q, cache, indptr, indices, sm_scale, kv_scale=scale,
                q_scale=q_scale, dot_precision=dots, out=out,
            )

        time_ms = device_time_ms(func, flush=not args.no_flush_l2)

        nnz = int(indptr[-1].item())
        num_tokens = q.shape[0]
        # QK reads the whole row, PV only the latent half
        flops = 2.0 * num_heads * nnz * (D_QK + KV_LORA_RANK)
        moved = bytes_moved(
            num_tokens, num_heads, nnz, 1 if dots == "fp8" else 2,
            _mla_num_splits(num_tokens, 1, nnz / num_tokens),
        )

        if metric == "time":
            return time_ms
        if metric == "throughput":
            return flops / (time_ms * 1e-3) / 1e12
        return moved / (time_ms * 1e-3) / 1e9

    bench_sparse_mla.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sparse MLA Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_seqs", type=int, nargs="+", default=[1, 8, 64, 256],
        help="sequences in the launch; every value is benchmarked",
    )
    parser.add_argument(
        "--num_tokens", type=int, nargs="+", default=[1],
        help="query tokens per sequence. 1 is a decode step, more is prefill; "
        "every value is benchmarked against every --num_seqs",
    )
    parser.add_argument("--num_heads", type=int, default=16, help="q heads (TP4=16)")
    parser.add_argument("--context", type=int, default=8192, help="context length")
    parser.add_argument("--topk", type=int, default=2048, help="indexer top-k")
    parser.add_argument(
        "--no_flush_l2", action="store_true",
        help="skip the cache flush between iterations, so the gather runs warm",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="time",
        help="metric to plot. bandwidth is the interesting one for a "
        "gather-bound kernel, but an fp8 cache moves half the bytes of a bf16 "
        "one, so only time compares the two precisions directly",
    )
    args = parser.parse_args()
    over = [t for t in args.num_tokens if t > args.context]
    if over:
        raise SystemExit(
            f"--num_tokens {over} exceeds --context {args.context}: a sequence "
            "cannot prefill more tokens than its context holds"
        )
    if arch_info.get_arch() != "gfx950":
        raise SystemExit(f"sparse_mla_fwd is gfx950-only (got {arch_info.get_arch()})")
    run_benchmark(args)


if __name__ == "__main__":
    main()
