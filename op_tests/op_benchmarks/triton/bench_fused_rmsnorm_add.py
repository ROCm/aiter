import argparse

import triton

from aiter.ops.triton.normalization.fused_rmsnorm_add import fused_rmsnorm_add
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.normalization.test_fused_rmsnorm_add import (
    generate_fused_rmsnorm_add_inputs,
)


def get_x_vals():
    x_vals = [
        # M scaling at a fixed width
        (1, 4096),
        (1024, 4096),
        (16384, 4096),
        # BLOCK_SIZE_N buckets
        (8192, 4),
        (8192, 128),
        (8192, 512),
        (8192, 2048),
        (8192, 8192),
        (8192, 16384),
        (256, 32768),
        # model_shapes hidden sizes
        (8192, 2880),
        (8192, 7168),
        (4096, 5120),
        (16380, 1536),
        # N % 16 != 0 synthetic probe
        (8192, 4095),
    ]
    return x_vals


def run_benchmark(args):
    if args.shape is not None and min(args.shape) <= 0:
        raise ValueError("--shape M N requires positive M and N")

    x_names = ["M", "N"]
    if args.shape is not None:
        x_vals_list = [list(args.shape)]
    else:
        x_vals_list = [list(shape) for shape in get_x_vals()]

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    elif args.metric == "throughput":
        ylabel = "Throughput_(TFLOPS)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = [""]  # prevents doubled bandwidth text
    line_vals = [ylabel]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    add_residual = args.add_residual
    c_dtype = str_to_torch_dtype[args.dtype]

    @triton.testing.perf_report([benchmark])
    def bench_fused_rmsnorm_add(M, N, metric, **kwargs):
        x, weight, res1 = generate_fused_rmsnorm_add_inputs(M, N, c_dtype, add_residual)
        eps = 1e-5

        fn = lambda: fused_rmsnorm_add(
            x,
            weight,
            eps,
            res1=res1,
        )

        es = x.element_size()
        mem_read = M * N * es + N * es
        mem_write = M * N * es
        if add_residual:
            # res1 is read and the pre-norm sum is written back out
            mem_read += M * N * es
            mem_write += M * N * es
        mem = mem_read + mem_write

        flops = 4 * M * N + (M * N if add_residual else 0)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        if metric == "time":
            return ms
        elif metric == "bandwidth":
            return mem / (ms * 1e-3) * 1e-9  # GB/s
        elif metric == "throughput":
            return flops / ms * 1e-9  # TFLOP/s
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_fused_rmsnorm_add.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark FusedRMSNormAdd",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth", "throughput"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument(
        "--add-residual",
        action="store_true",
        default=False,
        help="Fuse a residual add ahead of the norm (FIRST_INPUT_RES path).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Input dtype.",
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
    args = parser.parse_args(args=args)
    return args


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
