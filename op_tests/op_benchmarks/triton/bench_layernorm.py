import argparse

import torch
import triton

from aiter.ops.triton.normalization.norm import (
    layer_norm,
    layernorm2d_fwd_with_add,
    layernorm2d_fwd_with_add_dynamicquant,
    layernorm2d_fwd_with_add_smoothquant,
    layernorm2d_fwd_with_dynamicquant,
    layernorm2d_fwd_with_smoothquant,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)


def get_x_vals():
    x_vals = [
        # M scaling at the most common hidden size
        (1, 4096),
        (128, 4096),
        (1024, 4096),
        (4096, 4096),
        (16384, 4096),
        # model hidden sizes
        (8192, 512),
        (8192, 1536),
        (8192, 2880),
        (8192, 5120),
        (8192, 7168),
        (8192, 8192),
        # edge case in tester, odd widths, wide row
        (1, 4),
        (128, 2),
        (359, 1),
        (1, 131072),
        # BLOCK_SIZE caps at 65536//element_size; past it the kernel loops over N
        (256, 32768),
        (256, 65536),
    ]
    return x_vals


def run_benchmark(args):
    if args.backward and args.quant != "none":
        raise ValueError("--backward cannot be combined with --quant")
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

    line_names = [""]  # prevents double bandwidth text
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

    quant = args.quant
    add_residual = args.add_residual
    do_backward = args.backward
    c_dtype = str_to_torch_dtype[args.dtype]

    @triton.testing.perf_report([benchmark])
    def bench_layernorm(M, N, metric, **kwargs):
        torch.manual_seed(0)
        x = torch.randn(M, N, device="cuda", dtype=c_dtype)
        weight = torch.rand(N, device="cuda", dtype=c_dtype)
        bias = torch.rand(N, device="cuda", dtype=c_dtype)
        eps = 1e-5
        es = x.element_size()

        if do_backward:
            # build y once, time backward only
            x = x.requires_grad_(True)
            weight = weight.requires_grad_(True)
            bias = bias.requires_grad_(True)
            if add_residual:
                out = torch.empty_like(x)
                res_in = torch.randn(M, N, device="cuda", dtype=c_dtype)
                res_out = torch.empty_like(x)
                y = layernorm2d_fwd_with_add(out, x, res_in, res_out, weight, bias, eps)
            else:
                y = layer_norm(x, weight, bias, eps)
            dy = torch.randn_like(y)

            def fn():
                x.grad, weight.grad, bias.grad = None, None, None
                y.backward(dy, retain_graph=True)

            # x, dy in; dx, dw, db out
            mem_read = 2 * M * N * es + 2 * M * 4 + 2 * N * es
            mem_write = M * N * es + 2 * N * es
            mem = mem_read + mem_write
            flops = 12 * M * N
        else:
            res_in = res_out = None
            if add_residual:
                res_in = torch.randn(M, N, device="cuda", dtype=c_dtype)
                res_out = torch.empty_like(x)

            if quant == "none":
                if add_residual:
                    out = torch.empty_like(x)
                    fn = lambda: layernorm2d_fwd_with_add(
                        out, x, res_in, res_out, weight, bias, eps
                    )
                else:
                    fn = lambda: layer_norm(x, weight, bias, eps)
                out_bytes = M * N * es + 2 * M * 4  # y + mean/rstd
            else:
                out_i8 = torch.empty((M, N), dtype=torch.int8, device="cuda")
                yscale = torch.empty((M, 1), dtype=torch.float32, device="cuda")
                xscale = torch.rand(N, device="cuda", dtype=torch.float32)
                if quant == "smooth" and add_residual:
                    fn = lambda: layernorm2d_fwd_with_add_smoothquant(
                        out_i8, x, res_in, res_out, xscale, yscale, weight, bias, eps
                    )
                elif quant == "smooth":
                    fn = lambda: layernorm2d_fwd_with_smoothquant(
                        out_i8, x, xscale, yscale, weight, bias, eps
                    )
                elif add_residual:
                    fn = lambda: layernorm2d_fwd_with_add_dynamicquant(
                        out_i8, x, res_in, res_out, yscale, weight, bias, eps
                    )
                else:
                    fn = lambda: layernorm2d_fwd_with_dynamicquant(
                        out_i8, x, yscale, weight, bias, eps
                    )
                out_bytes = M * N + M * 4  # int8 out + per-row scale

            mem_read = M * N * es + 2 * N * es
            mem_write = out_bytes
            if add_residual:
                mem_read += M * N * es
                mem_write += M * N * es
            if quant == "smooth":
                mem_read += N * 4
            mem = mem_read + mem_write

            flops = 8 * M * N
            if add_residual:
                flops += M * N
            if quant != "none":
                flops += 2 * M * N

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        if metric == "time":
            return ms
        elif metric == "bandwidth":
            return mem / (ms * 1e-3) * 1e-9  # GB/s
        elif metric == "throughput":
            return flops / ms * 1e-9  # TFLOP/s
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_layernorm.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark LayerNorm",
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
        "--quant",
        type=str,
        choices=["none", "dynamic", "smooth"],
        default="none",
        help="Fuse a quantization epilogue (int8 output, IS_SMOOTH selects the variant).",
    )
    parser.add_argument(
        "--add-residual",
        action="store_true",
        default=False,
        help="Fuse a residual add ahead of the norm.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        default=False,
        help="Benchmark the backward pass instead of forward. Cannot be used with --quant.",
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
