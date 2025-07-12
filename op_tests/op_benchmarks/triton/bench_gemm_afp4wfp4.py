import sys
import os
import torch
import triton
import math
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_scales,
)
from op_tests.triton_tests.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)

TRITON_HIP_PRESHUFFLE_SCALES = (
    os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES", "0") == "1"
)


def bench_gemm_fn(M, N, K, metric):
    c_dtype = torch.bfloat16
    x, w, _, _, x_scale, w_scale, _, _ = generate_gemm_afp4wfp4_inputs(M, N, K, c_dtype)
    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
    mem_read += (
        x_scale.numel() * x_scale.element_size()
        + w_scale.numel() * w_scale.element_size()
    )
    mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write
    out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=c_dtype)

    if TRITON_HIP_PRESHUFFLE_SCALES:
        ms = triton.testing.do_bench(
            lambda: gemm_afp4wfp4_preshuffled_scales(
                x, w, x_scale, w_scale, c_dtype, out
            ),
            warmup=25,
            rep=100,
        )
    else:
        ms = triton.testing.do_bench(
            lambda: gemm_afp4wfp4(x, w, x_scale, w_scale, c_dtype, out),
            warmup=25,
            rep=100,
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


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        unsupported_args = [
            "layout",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object("GEMM MXFP4 x MXFP4 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            # Divide N by tensor parallel
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            # Divide K by tensor parallel
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(M, N, K, metric)

    bench_gemm_afp4wfp4.run(save_path=".", print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object("GEMM MXFP4 x MXFP4 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4(M, N, K, metric, **kwargs):
        return bench_gemm_fn(M, N, K, metric)

    bench_gemm_afp4wfp4.run(save_path=".", print_data=True)


def parse_args():
    parser = get_parser("MXFP4 x MXFP4 GEMM")
    parser = add_argparse_ff(parser)

    parser.add_argument(
        "--print_vgpr",
        action="store_true",
        help="Print VGPR usage for Triton kernels.",
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args)  # noqa: E731
        print_vgpr(fun, "GEMM")
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
