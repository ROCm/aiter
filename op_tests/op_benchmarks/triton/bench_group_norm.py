import argparse

import torch
import triton

from aiter.ops.triton.normalization.group_norm import group_norm
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)


def get_x_vals():
    """(N, C, H, W) of the GroupNorm layers of a UNet and a VAE decoder."""
    x_vals = [
        (2, 320, 64, 64),  # SD1.5 UNet, 512px
        (2, 640, 32, 32),
        (2, 1280, 16, 16),
        (2, 320, 128, 128),  # SDXL UNet, 1024px
        (2, 640, 64, 64),
        (1, 512, 64, 64),  # VAE decoder, 512px output
        (1, 256, 256, 256),
        (1, 128, 512, 512),
    ]
    return x_vals


def run_benchmark(args):
    x_names = ["N", "C", "H", "W"]
    if args.shape is not None:
        x_vals_list = [list(args.shape)]
    else:
        x_vals_list = [list(shape) for shape in get_x_vals()]

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[""],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    c_dtype = str_to_torch_dtype[args.dtype]
    activation = None if args.activation == "none" else args.activation
    num_groups = args.num_groups
    with_temb = args.temb

    @triton.testing.perf_report([benchmark])
    def bench_group_norm(N, C, H, W, metric, **kwargs):
        torch.manual_seed(0)
        x = torch.randn(N, C, H, W, device="cuda", dtype=c_dtype)
        weight = torch.randn(C, device="cuda", dtype=c_dtype)
        bias = torch.randn(C, device="cuda", dtype=c_dtype)
        temb = torch.randn(N, C, device="cuda", dtype=c_dtype) if with_temb else None
        out = torch.empty_like(x)
        fn = lambda: group_norm(
            x, num_groups, weight, bias, 1e-5, temb=temb, activation=activation, out=out
        )
        # x in, out out; weight, bias and temb are small
        mem = 2 * x.numel() * x.element_size()

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if metric == "time":
            return ms
        return mem / (ms * 1e-3) * 1e-9  # GB/s

    bench_group_norm.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark GroupNorm",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("N", "C", "H", "W"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument("--num-groups", type=int, default=32, help="number of groups")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["none", "silu", "gelu"],
        default="silu",
        help="activation after the norm",
    )
    parser.add_argument(
        "--temb",
        action="store_true",
        default=False,
        help="add an (N, C) tensor to the input before the norm",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="fp16",
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
