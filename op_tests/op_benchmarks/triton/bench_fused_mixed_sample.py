# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for ``fused_mixed_sample``.

The ``triton`` line is the fused op: two launches, tiled over the vocab. The
``hip`` line is the path that it replaces: ``mixed_sample_outer_exponential``,
then a greedy override (``topk_select`` or ``torch.argmax``) and ``where``.
Both lines use the same noise. The noise draw is not timed.
``--greedy-frac`` sets the part of the rows that are greedy (default 0: all
rows sample, as in a speculative-decoding verify batch).
"""

import torch
import triton

import aiter
from aiter import dtypes
from aiter.ops.triton.fusions.fused_mixed_sample import fused_mixed_sample
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

EPS = 1e-10


def _greedy_ids(logits):
    if hasattr(aiter, "topk_select"):
        return aiter.topk_select(logits, 1, tie="low")[1].view(-1)
    return torch.argmax(logits, dim=-1).to(torch.int32)


def _generate_inputs(M, V, dtype, greedy_frac):
    logits = (torch.randn((M, V), dtype=torch.float32, device="cuda") * 4).to(dtype)
    noise = torch.empty((M, V), dtype=torch.float32, device="cuda").exponential_(1)
    temps = torch.rand(M, dtype=torch.float32, device="cuda") + 0.3
    temps[: int(M * greedy_frac)] = 0.0
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    return logits, noise, temps, out


def get_x_vals():
    """(M rows, V vocab) pairs. 163840 is the Kimi-K3 vocab."""
    return [(m, v) for v in (163840, 129280) for m in (1, 4, 16, 64, 128, 256, 512)]


def run_benchmark(args):
    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    x_vals_list = [tuple(args.shape)] if args.shape is not None else get_x_vals()
    dtype = dtypes.str2Dtype(args.dtype)
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    benchmark = triton.testing.Benchmark(
        x_names=["M", "V"],
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=["triton", "hip"],
        line_names=["triton", "hip"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_mixed_sample(M, V, provider, metric, **kwargs):
        keys = ["logits", "noise", "temps", "out"]
        rot_bufs = generate_rotating_buffers_pool(
            [lambda: _generate_inputs(M, V, dtype, args.greedy_frac)],
            [keys],
            **kwargs_rot_bufs,
        )

        if provider == "triton":

            def _run(logits, noise, temps, out):
                fused_mixed_sample(logits, noise, temps, eps=EPS, out=out)

        else:

            def _run(logits, noise, temps, out):
                aiter.mixed_sample_outer_exponential(out, logits, noise, temps, eps=EPS)
                torch.where(temps <= 0.0, _greedy_ids(logits), out)

        ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
        sample = rot_bufs[0]
        # Read logits and noise once; write one id per row.
        mem = (
            M * V * (sample["logits"].element_size() + sample["noise"].element_size())
            + M * 4
        )
        if metric == "time":
            return ms
        return mem / (ms * 1e-3) * 1e-9

    bench_fused_mixed_sample.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_mixed_sample")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices={"fp16", "bf16", "fp32"},
        help="dtype of the logits",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "V"),
        help="one (M rows, V vocab) shape",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    parser.add_argument(
        "--greedy-frac",
        type=float,
        default=0.0,
        help="part of the rows that are greedy (temperature 0), from 0 to 1",
    )
    override_parser_action(parser, action="--metric", default="time")
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    main()
