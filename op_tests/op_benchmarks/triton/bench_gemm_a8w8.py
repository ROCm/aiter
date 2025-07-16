import sys
import triton
import math
from aiter.ops.triton.gemm_a8w8 import gemm_a8w8
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.triton_tests.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)

from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
)


def bench_gemm_fn(M, N, K, metric):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = str_to_torch_dtype["bf16"]
    x, weight, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M, N, K, str_to_torch_dtype["fp8e4m3"], c_dtype, output=True
    )

    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * weight.element_size()
    mem_write = (M * N) * bias.element_size()
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: gemm_a8w8(x, weight, x_scale, w_scale, bias, c_dtype, y),  # noqa: E731
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


def run_model_benchmark(args):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object("GEMM A8W8 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
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

        return bench_gemm_fn(M, N, K, metric)

    bench_gemm_a8w8.run(save_path=".", print_data=True)


def run_shape_benchmark(args):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object("GEMM A8W8 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(M, N, K, metric, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric)

    bench_gemm_a8w8.run(save_path=".", print_data=True)


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


def parse_args():
    parser = get_parser(kernel_name="A8W8 GEMM")
    parser = add_argparse_ff(parser)
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
