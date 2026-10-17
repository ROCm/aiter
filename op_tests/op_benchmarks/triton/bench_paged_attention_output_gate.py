# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark paged_attention_output_gate_group_fp8_quant.

The op is bandwidth-bound: a decode step streams each sequence's K and V exactly
once and does O(context) work per output element, so the roofline is pure HBM.
Traffic is `tokens * context * HEAD_DIM * 2` bytes (fp8, K and V), and the
`bandwidth` metric reports that over the measured time. Below tokens ~8 the
kernel is latency-bound rather than bandwidth-bound and the number will be a
small fraction of peak; that is the kernel having only a few MB to move, not a
defect.

`--compare` additionally times the three launches this op replaces --
`paged_attention_ragged` + a sigmoid-gate multiply + a group-128 activation
quant -- so the speedup can be read directly.

NOTE on the comparison: on the FP8 KV path `paged_attention_ragged` is not
numerically equivalent to this op; it casts already-normalized softmax
probabilities to e4m3 where they are subnormal. The work (KV bytes gathered,
mfma count) is the same, so the timing is a fair performance baseline, but do
not read it as two implementations computing the same thing.
"""

import argparse
import functools
import sys

import torch
import triton

from aiter.ops.triton.attention.paged_attention_output_gate import (
    paged_attention_output_gate_group_fp8_quant,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.attention.test_paged_attention_output_gate import (
    _make_inputs,
)

HEAD_DIM = 256
_PARTITION_SIZE = 256


def _generate_stock_pool_inputs(tokens, context, heads, dtype, seed):
    """Buffers for the three launches `_stock_composition`'s `_run` replaces.

    `out`/`ws`/`gated_buf` are write destinations, not random inputs -- they
    still need a fresh instance per rotating-buffer pool entry so repeated
    writes across the pool don't collapse onto one hot cache line.
    """
    q, kc, vc, indptr, indices, gate = _make_inputs(
        tokens, context, heads, dtype, seed=seed
    )
    out = torch.empty(tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    parts = (context + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    nbytes = (tokens * heads * parts * HEAD_DIM) * 4 + 2 * (tokens * heads * parts) * 4
    ws = torch.empty(nbytes, dtype=torch.uint8, device=q.device)
    last_page = torch.ones(tokens, dtype=torch.int32, device=q.device)
    one = torch.ones(1, dtype=torch.float32, device=q.device)
    flat = out.view(tokens, heads * HEAD_DIM)
    # Separate destination: fused_sigmoid_mul defaults to writing in place, which
    # would compound across the invocations captured in one graph.
    gated_buf = torch.empty_like(flat)
    return q, kc, vc, indptr, indices, gate, out, ws, last_page, one, flat, gated_buf


def bench_fn(tokens, context, heads, provider, args):
    dtype = torch.float8_e4m3fn
    scale = HEAD_DIM**-0.5
    impl, metric = provider.rsplit("_", 1)

    if impl == "fused":
        k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        v_scale = torch.ones(1, dtype=torch.float32, device="cuda")

        def _run(q, kc, vc, indptr, indices, gate):
            paged_attention_output_gate_group_fp8_quant(
                q,
                kc,
                vc,
                indptr,
                indices,
                gate,
                scale=scale,
                max_context=context,
                k_scale=k_scale,
                v_scale=v_scale,
                quant_dtype=dtype,
            )

        _p_gen = functools.partial(
            _make_inputs, tokens, context, heads, dtype, seed=tokens
        )
        _out_val_keys = ["q", "kc", "vc", "indptr", "indices", "gate"]
    else:
        # The gate step uses aiter's own `fused_sigmoid_mul`, NOT `flat *
        # torch.sigmoid(gate)`. The torch spelling is two kernels rather than
        # one, and the caller this op targets (SGLang's Qwen3-Next decode)
        # runs a single fused sigmoid-gate multiply -- so timing the torch
        # spelling would inflate the baseline by a launch this op does not
        # actually remove, and overstate the speedup (measured: +4% at bs=64
        # and +21% at bs=1, ctx 8192).
        from aiter import paged_attention_ragged
        from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul
        from aiter.ops.triton.quant.fused_fp8_quant import (
            fused_flatten_fp8_group_quant,
        )

        parts = (context + _PARTITION_SIZE - 1) // _PARTITION_SIZE

        def _run(
            q, kc, vc, indptr, indices, gate, out, ws, last_page, one, flat, gated_buf
        ):
            paged_attention_ragged(
                out,
                ws,
                q,
                kc.view(-1, 1, 1, HEAD_DIM),
                vc.view(-1, 1, 1, HEAD_DIM),
                scale,
                indptr,
                indices,
                last_page,
                1,
                parts,
                None,
                "fp8_e4m3",
                "NHD",
                0.0,
                one,
                one,
                None,
                _PARTITION_SIZE,
            )
            gated = fused_sigmoid_mul(flat, gate, out=gated_buf)
            fused_flatten_fp8_group_quant(gated.view(tokens, heads, HEAD_DIM), 128)

        _p_gen = functools.partial(
            _generate_stock_pool_inputs, tokens, context, heads, dtype, tokens
        )
        _out_val_keys = [
            "q",
            "kc",
            "vc",
            "indptr",
            "indices",
            "gate",
            "out",
            "ws",
            "last_page",
            "one",
            "flat",
            "gated_buf",
        ]

    if args.do_bench:
        bufs = _p_gen()

        def call():
            return _run(*bufs)

        us = triton.testing.do_bench(call, warmup=args.warmup, rep=args.rep) * 1e3
    else:
        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], target_mb=256, min_pool_len=4
        )
        us = (
            do_bench_aiter_triton(
                _run, rot_bufs, kernels_per_graph=args.reps, num_replays=args.iters
            )
            * 1e3
        )

    if metric == "time":
        return us
    if metric == "bandwidth":
        # K and V, one byte per element, streamed once per decode step.
        kv_bytes = tokens * context * HEAD_DIM * 2
        return kv_bytes / (us * 1e-6) * 1e-9
    raise ValueError(f"Unknown metric: {metric}")


def get_benchmark_shapes(args):
    if args.tokens is not None and args.context is not None:
        return [(args.tokens, args.context, args.heads)]
    # Decode batches a CUDA graph capture actually asks for, including the
    # non-power-of-two sizes, crossed with contexts either side of the
    # short/long body crossover at 32768.
    tokens = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    contexts = [args.context] if args.context else [1024, 8192, 65536]
    return [(t, c, args.heads) for c in contexts for t in tokens]


def run_benchmark(args):
    impls = ("fused", "stock") if args.compare else ("fused",)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{i}_{m}" for m in metrics for i in impls]

    benchmark = triton.testing.Benchmark(
        x_names=["tokens", "context", "heads"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="us / GB/s",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def _run(tokens, context, heads, provider):
        return bench_fn(tokens, context, heads, provider, args)

    _run.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark paged attention with output gate",
        description=(
            "Benchmark the Gluon paged-decode attention with sigmoid output "
            "gate and group-128 FP8 epilogue"
        ),
    )
    parser.add_argument("--tokens", type=int, default=None, help="Decode batch size")
    parser.add_argument(
        "--context", type=int, default=None, help="Context length per sequence"
    )
    parser.add_argument("--heads", type=int, default=4, help="Query heads (1..16)")
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["time", "bandwidth", "all"],
        help="time is microseconds; bandwidth is KV GB/s",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="also time the three stock launches this op replaces",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--do-bench",
        action="store_true",
        default=False,
        help=(
            "use triton.testing.do_bench instead of rotating-buffer graph "
            "capture; launch-overhead dominated at decode shapes, see bench_fn"
        ),
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--reps", type=int, default=50, help="invocations per graph")
    parser.add_argument(
        "--print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for paged_attention_output_gate kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
