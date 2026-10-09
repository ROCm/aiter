import argparse

import triton

from aiter.ops.triton.normalization.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.normalization.test_fused_add_rmsnorm_pad import (
    generate_inputs,
)


def get_x_vals():
    x_vals = [
        # M scaling at fixed rate
        (1, 2880),
        (32, 2880),
        (256, 2880),
        (1024, 2880),
        (4096, 2880),
        (8192, 2880),
        (16384, 2880),
        # diff BLOCK_SIZE_N buckets
        (8192, 4),
        (8192, 16),
        (8192, 320),
        (8192, 640),
        # w/o padding
        (8192, 512),
        (8192, 1024),
        (8192, 2048),
        (8192, 4096),
    ]
    return x_vals


def run_benchmark(args):
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

    backend = args.backend
    add_residual = args.add_residual
    pad_to_multiple = args.pad_to_multiple
    c_dtype = str_to_torch_dtype[args.dtype]

    @triton.testing.perf_report([benchmark])
    def bench_fused_add_rmsnorm_pad(M, N, metric, **kwargs):
        x, weight, res = generate_inputs(M, N, add_residual, c_dtype)
        eps = 1e-6

        fn = lambda: fused_add_rmsnorm_pad(
            x,
            weight,
            eps,
            res=res,
            x_pad_to_multiple=pad_to_multiple,
            backend=backend,
        )

        n_out = (
            triton.cdiv(N, pad_to_multiple) * pad_to_multiple
            if pad_to_multiple > 0
            else N
        )
        es = x.element_size()
        mem_read = M * N * es + N * es
        mem_write = M * n_out * es
        if add_residual:
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

    bench_fused_add_rmsnorm_pad.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark FusedAddRMSNormPad",
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
        "--backend",
        type=str,
        choices=["triton", "gluon"],
        default=None,
        help="Kernel backend. Default follows the arch: gluon on gfx1250, triton othwerise.",
    )
    parser.add_argument(
        "--add-residual",
        action="store_true",
        default=False,
        help="Fuse a residual add ahead of the norm (HAS_RES path).",
    )
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=0,
        help="Pad the output's last dim up to a multiple of this value (0 disables).",
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
