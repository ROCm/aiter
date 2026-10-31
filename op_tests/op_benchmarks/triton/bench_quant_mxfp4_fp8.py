# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from functools import partial

import torch
import triton

from aiter.ops.triton.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant
from aiter.utility.fp4_utils import dynamic_mxfp4_quant as fp4_utils_dynamic_mxfp4_quant
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
    get_model_configs,
)


def get_default_shapes() -> list[list[int]]:
    M = [8, 32, 256, 2048, 8192, 16384]
    N = [1024, 3072, 7168]
    return [[m, n] for n in N for m in M]


def model_benchmark_shapes(args) -> list[tuple[str, int, int]]:
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    M_list = [args.M] if args.model == "all" else [2**i for i in range(15)]
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            N = config["hidden_size"]
            shapes.append((model_name, M, N))

    return shapes


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_provider(fmt: str, provider: str):
    if fmt == "mxfp4" and provider == "fp4_utils":
        return fp4_utils_dynamic_mxfp4_quant
    if provider in ("triton", "gluon"):
        quant = dynamic_mxfp4_quant if fmt == "mxfp4" else dynamic_mxfp8_quant
        return partial(quant, backend=provider)
    raise ValueError(f"Unknown provider: {provider}")


def get_line_vals(args) -> list[str]:
    """Each line is a "format-provider" combo; fp4_utils only applies to mxfp4."""
    formats = args.format.split(",")
    providers = args.provider.split(",")
    line_vals = []
    for fmt in formats:
        for provider in providers:
            if fmt == "mxfp8" and provider == "fp4_utils":
                continue
            line_vals.append(f"{fmt}-{provider}")
    return line_vals


def parse_shape_args(args) -> list[tuple[str, int, int]]:
    if args.shape is not None:
        M, N = args.shape
        return [("custom", M, N)]
    if args.model is not None:
        return model_benchmark_shapes(args)
    shapes = get_default_shapes()
    if args.use_sr:
        shapes = [(M, N) for M, N in shapes if N % 32 == 0]
    return [("default", M, N) for M, N in shapes]


def _generate_quant_pool_entry(dtype, M, N):
    return (torch.randn((M, N), dtype=dtype, device="cuda"),)


_QUANT_POOL_FIELDS = ["x"]


def _make_quant_kernel_fn(quant_fn, use_sr):
    if use_sr:

        def kernel_fn(x, **_):
            quant_fn(x, use_sr=True, philox_seed=1234)

    else:

        def kernel_fn(x, **_):
            quant_fn(x)

    return kernel_fn


def run_benchmark(args):
    if args.shape is not None and args.model is not None:
        raise ValueError("Use either --shape or --model, not both")

    x_vals = parse_shape_args(args)
    line_vals = get_line_vals(args)
    if args.use_sr and line_vals != ["mxfp4-triton"]:
        raise ValueError("--use-sr currently requires --format mxfp4 --provider triton")
    if args.use_sr and args.dtype not in ("bf16", "fp32"):
        raise ValueError("--use-sr requires --dtype bf16 or fp32")
    if args.use_sr and any(M <= 0 or N <= 0 or N % 32 != 0 for _, M, N in x_vals):
        raise ValueError("--use-sr requires positive shapes with N divisible by 32")

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    colors = ["green", "blue", "red", "orange", "purple"]
    benchmark = triton.testing.Benchmark(
        x_names=["model_name", "M", "N"],
        x_vals=x_vals,
        x_log=True,
        y_log=True,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[(colors[i % len(colors)], "-") for i in range(len(line_vals))],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric, "dtype": args.dtype, "use_sr": args.use_sr},
    )

    @triton.testing.perf_report([benchmark])
    def bench_quant_mx(
        M, N, metric, provider, dtype, use_sr, model_name=None, **kwargs
    ):
        fmt, provider = provider.split("-", 1)
        dtype = get_dtype(dtype)
        quant_fn = get_provider(fmt, provider)

        pool_gen = partial(_generate_quant_pool_entry, dtype, M, N)
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_QUANT_POOL_FIELDS], **kwargs_rot_bufs
        )
        kernel_fn = _make_quant_kernel_fn(quant_fn, use_sr)

        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

        # Read x and write quantized output + block scales.
        x_bytes = M * N * rot_bufs[0]["x"].element_size()
        x_quant_bytes = M * (N // 2) if fmt == "mxfp4" else M * N
        x_scale_bytes = M * ((N + 31) // 32)
        total_bytes = x_bytes + x_quant_bytes + x_scale_bytes

        if metric == "time":
            return ms
        if metric == "bandwidth":
            return total_bytes / (ms * 1e-3) * 1e-9
        raise ValueError("Unknown metric: " + metric)

    bench_quant_mx.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="MX Quant")
    parser.add_argument(
        "--format",
        type=str,
        default="mxfp4,mxfp8",
        help="Comma-separated quantized output format(s) to benchmark, from: mxfp4,mxfp8.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="Single shape to benchmark.",
    )
    parser.add_argument(
        "-M",
        type=int,
        default=4096,
        help="M dim of model benchmark if --model=all.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="triton",
        help="Provider(s) to benchmark, applied to each --format. Comma-separated "
        "values from: triton,gluon,fp4_utils (triton and gluon force that backend; "
        "fp4_utils only valid for mxfp4; silently skipped for other formats).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--use-sr",
        action="store_true",
        help="Benchmark gfx950 stochastic rounding with a fixed Philox seed.",
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="bandwidth"
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
