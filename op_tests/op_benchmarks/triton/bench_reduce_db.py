import argparse
import sys
import torch
import triton
import math
from aiter.ops.triton.reduce_db import reduce_db
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr
)


def get_shape_benchmark_object(plot_name, args, x_names=None):
    """
    Utility function for returning a triton.testing.Benchmark object to populate.

    Note: This is for benchmarking GEMM kernels without the --model flag. The distinction
    comes in the x_names and x_vals: For models, we use hidden_dim and intermediate_dim
    as args, but if we're just given a shape, we use M, N, K.
    """
    if x_names is None:
        x_names = ["M", "N"]

    if len(x_names) == len(args.shape):
        x_vals_list = [args.shape]
    else:
        raise ValueError(
            f"Incompatible --shape provided: {args.shape}. Expected a shape that matches {x_names}."
        )

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="provider",
        line_vals=["Triton"],
        line_names=["Triton"],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=plot_name,
        args={"metric": args.metric},
    )
    return benchmark


def bench_reduce_db_fn(
    M: int, N: int, metric: str, **kwargs
):
    xdtype = torch.bfloat16
    ydtype = torch.float32
    x = torch.randn((M, N), dtype=xdtype).cuda()
    y = torch.zeros((N,), dtype=ydtype).cuda()
    # flops
    flops = M * N
    # memory transfer
    mem_read = (M * N) * x.element_size()
    mem_write = N * y.element_size()
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: reduce_db(x, ydtype, y), warmup=25, rep=100  # noqa: E731
    )

    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops
    elif metric == "bandwidth":
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        return bandwidth
    else:
        raise ValueError("Unknown metric: " + metric)


def run_shape_benchmark(args):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object("REDUCE DB Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_reduce_db(M, N, metric, **kwargs):
        return bench_reduce_db_fn(M, N, metric)

    bench_reduce_db.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args):
    run_shape_benchmark(args)


def parse_args():
    parser = get_parser(kernel_name="REDUCE DB")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)  # noqa: E731
        print_vgpr(fun, "GEMM")
        return 0
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
