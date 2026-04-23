import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant import dynamic_mxfp4_quant as triton_dynamic_mxfp4_quant
from aiter.utility.fp4_utils import dynamic_mxfp4_quant as fp4_utils_dynamic_mxfp4_quant
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext


def get_default_shapes() -> list[list[int]]:
    return [
        [1, 4],
        [1, 32],
        [1, 64],
        [2, 32],
        [128, 32],
        [128, 64],
        [256, 32],
        [4096, 4096],
        [4096, 8192],
    ]


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_provider(provider: str):
    if provider == "triton":
        return triton_dynamic_mxfp4_quant
    if provider == "fp4_utils":
        return fp4_utils_dynamic_mxfp4_quant
    raise ValueError(f"Unknown provider: {provider}")


def parse_shape_args(args) -> list[list[int]]:
    if args.shape is not None:
        return [args.shape]
    return get_default_shapes()


def run_benchmark(args):
    x_vals = parse_shape_args(args)
    providers = args.provider.split(",")

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        x_log=True,
        y_log=True,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("green", "-"), ("blue", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric, "dtype": args.dtype},
    )

    @triton.testing.perf_report([benchmark])
    def bench_quant_mxfp4(M, N, metric, provider, dtype, **kwargs):
        dtype = get_dtype(dtype)
        x = torch.randn((M, N), dtype=dtype, device="cuda")
        quant_fn = get_provider(provider)

        def fn():
            quant_fn(x)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Read x and write quantized output + block scales.
        x_bytes = x.numel() * x.element_size()
        x_fp4_bytes = M * (N // 2)
        x_scale_bytes = M * ((N + 31) // 32)
        total_bytes = x_bytes + x_fp4_bytes + x_scale_bytes

        if metric == "time":
            return ms
        if metric == "bandwidth":
            return total_bytes / (ms * 1e-3) * 1e-9
        raise ValueError("Unknown metric: " + metric)

    bench_quant_mxfp4.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 Quant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="Single shape to benchmark.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="triton",
        help="Provider(s) to benchmark. Comma-separated values from: triton,fp4_utils.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="Metric to plot.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write performance results to CSV file.",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None):
    parsed_args = parse_args(args)
    run_benchmark(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
