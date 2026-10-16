"""Benchmark the gfx1250 gluon MXFP8 GEMM (e4m3 x e4m3, 1x32 e8m0 scales): gemm_afp8wfp8_mx_preshuffle."""

import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_mx_preshuffle
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8_mx import (
    generate_inputs,
    prepare,
)


def bench_gemm_fn(
    M: int, N: int, K: int, metric: str, x_preshuffled: bool, scale_layout: str
):
    c_dtype = torch.bfloat16
    x, w, xs, ws = generate_inputs(M, N, K)
    # input preparation is not timed
    x_k, w_k, xs_k, ws_k = prepare(x, w, xs, ws, scale_layout, x_preshuffled)
    y = torch.empty((M, N), dtype=c_dtype, device=x.device)
    flops = 2.0 * M * N * K
    mem_read = x.numel() + w.numel() + xs_k.numel() + ws_k.numel()  # 1-byte elements
    mem = mem_read + M * N * 2
    ms = triton.testing.do_bench(
        lambda: gemm_afp8wfp8_mx_preshuffle(
            x_k,
            w_k,
            xs_k,
            ws_k,
            c_dtype,
            y,
            x_preshuffled=x_preshuffled,
            scale_layout=scale_layout,
        ),
        warmup=25,
        rep=100,
    )
    if metric == "time":
        return ms
    if metric == "throughput":
        return flops / ms * 1e-9  # TFLOPS
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError("Unknown metric: " + metric)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(
        "GEMM MXFP8 x MXFP8 (1x32 scales) Benchmark", args
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp8wfp8_mx(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
    ):
        if layer == "fc1":
            N, K = (
                intermediate_dim if args.no_glu else intermediate_dim * 2
            ), hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, math.ceil(intermediate_dim / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.x_preshuffled, args.scale_layout)

    bench_gemm_afp8wfp8_mx.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(
        "GEMM MXFP8 x MXFP8 (1x32 scales) Benchmark", args
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp8wfp8_mx(M, N, K, metric, model_name=None, **kwargs):
        return bench_gemm_fn(M, N, K, metric, args.x_preshuffled, args.scale_layout)

    bench_gemm_afp8wfp8_mx.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        run_model_benchmark(args)
    else:
        for arg in ("fc1", "fc2", "no_glu"):
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser("MXFP8 x MXFP8 GEMM (1x32 scales)")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--x-preshuffled",
        action="store_true",
        help="pass x through shuffle_weight (x_preshuffled=True); needs M % 16 == 0",
    )
    parser.add_argument(
        "--scale-layout",
        choices=["n32k4", "128row"],
        default="n32k4",
        help="scale layout of x and w scales (scale_layout=)",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        print_vgpr(lambda: run_benchmark(parsed_args, defaults), "GEMM")
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
