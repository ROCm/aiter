import argparse
import sys
import torch
import triton
import math
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.test_gemm_a16w16 import generate_gemm_a16w16_inputs
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)

from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
)


def bench_gemm_fn(
    M: int, N: int, K: int, metric: str, layout: str, atomic: bool = False, **kwargs
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = torch.bfloat16
    x, w, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, c_dtype, layout=layout, output=True
    )
    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N) * x.element_size()
    mem = mem_read + mem_write

    if atomic:
        # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
        y = y.to(torch.float32).zero_()
        ms = triton.testing.do_bench(
            lambda: gemm_a16w16_atomic(x, w, torch.float32, y),
            warmup=25,
            rep=100,  # noqa: E731
        )
    else:
        ms = triton.testing.do_bench(
            lambda: gemm_a16w16(x, w, c_dtype, y), warmup=25, rep=100  # noqa: E731
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


def run_model_benchmark(args):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object("GEMM A16W16 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        """
        Fc1:
             M      K                  K           N          M       N
        A = (B, hidden_dim) @ W = (hidden_dim, 2*int_dim) -> (B, 2*int_dim) -> gating -> (B, int_dim)

        Fc2:
             M     K               K          N          M       N
        A = (B, int_dim) @ W = (int_dim, hidden_dim) -> (B, hidden_dim)

        Tensor parallel splits across int_dim (N for fc1, K for fc2)
        """
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
        # print(f"Layer: {layer}, M: {M}, N: {N}, K: {K}, hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

        return bench_gemm_fn(M, N, K, metric, args.layout, atomic=args.atomic)

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object("GEMM A16W16 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, metric, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.layout, atomic=args.atomic)

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        unsupported_args = []
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


def parse_args():
    parser = get_parser(kernel_name="A16W16 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Use the atomic kernel (split-k with atomic_add) instead of the standard a16w16 kernel.",
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
