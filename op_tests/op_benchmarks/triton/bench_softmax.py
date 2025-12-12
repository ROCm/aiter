import argparse
import sys
import torch
import triton
from aiter.ops.triton.softmax import softmax as triton_softmax
from aiter.ops.triton.gluon.softmax import softmax as gluon_softmax
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)


def get_x_vals(dim=0):
    if dim == 0:
        x_vals = [
            (1, 8192),
            (32, 8192),
            (64, 8192),
            (128, 8192),
            (192, 8192),
            (256, 8192),
            (320, 8192),
            (512, 8192),
            (1024, 8192),
            (2048, 8192),
            (4096, 8192),
            (8192, 8192),
            (16384, 8192),
        ]
    elif dim == 1:
        x_vals = [
            (8192, 1),
            (8192, 32),
            (8192, 64),
            (8192, 128),
            (8192, 192),
            (8192, 256),
            (8192, 320),
            (8192, 512),
            (8192, 1024),
            (8192, 2048),
            (8192, 4096),
            (8192, 8192),
            (8192, 16384),
            (8192, 32768),
            (8192, 49152),
            (8192, 65536),
            (8192, 81920),
            (8192, 98304),
            (8192, 131072),
        ]
    return x_vals


def run_shape_benchmark(args, impl):
    x_names = ["M", "N"]
    if args.shape:
        if len(x_names) == len(args.shape):
            x_vals_list = [args.shape]
        else:
            raise ValueError(
                f"Incompatible --shape provided: {args.shape}. Expected a shape that matches {x_names}."
            )
    else:
        x_vals_list = get_x_vals(dim=(1 if args.N else 0))

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_softmax(M, N, metric, **kwargs):
        dtype = torch.bfloat16
        x = torch.randn(M, N, dtype=dtype, device="cuda")

        # Memory transfer
        mem_read = (
            2 * M * N * x.element_size()
        )  # one load in loop 1, one load in loop 2
        mem_write = M * N * x.element_size()  # one write in loop 2
        mem = mem_read + mem_write

        ms = triton.testing.do_bench(lambda: impl(x), warmup=25, rep=100)

        if metric == "time":
            return ms
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_softmax.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.N), "User can specify --shape or -N exclusively"

    if args.gluon:
        impl = gluon_softmax
    else:
        impl = triton_softmax

    unsupported_args = []
    for arg in unsupported_args:
        if getattr(args, arg, None) != getattr(defaults, arg, None):
            raise Exception(
                f"Argument '{arg}' is not supported for benchmarking with the --model flag."
            )
    run_shape_benchmark(args, impl)


def parse_args():
    parser = argparse.ArgumentParser(
        prog=f"Benchmark Softmax",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "-gluon",
        action="store_true",
        help="Use Gluon implementation (experimental, requires latest Triton from main)",
    )
    parser.add_argument(
        "-N",
        action="store_true",
        help=f"Benchmark shapes with increasing N dimension, with M = {get_x_vals(dim=1)[0][0]}",
    )
    args = parser.parse_args()
    defaults = parser.parse_args([])
    return args, defaults


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
