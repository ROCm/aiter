import sys
import torch
import triton
import math
from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
    generate_gemm_a16w16_gated_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
    get_caller_name_no_ext,
)


def bench_gemm_fn(M: int, N: int, K: int, metric: str, layout: str, activation: str = None):
    c_dtype = torch.bfloat16
    x, w, _, y = generate_gemm_a16w16_gated_inputs(
        M, N, K,
        c_dtype,
        layout=layout,
        output=True,
    )
    # flops: N is pre-gating dimension, output is N//2
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N // 2) * 2
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: gemm_a16w16_gated(x, w, c_dtype, y, activation=activation),
        warmup=25,
        rep=100,
    )

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    else:
        raise ValueError("Unknown metric: " + metric)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        run_model_benchmark(args)
    else:
        run_shape_benchmark(args)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(M, N, K, metric, args.layout, activation=args.activation)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, K, metric, model_name=None, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.layout, activation=args.activation)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args(args=None):
    parser = get_parser(kernel_name="A16W16 Gated GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function for gating (silu, gelu, relu, etc.)",
    )
    return get_ff_args(parser, args=args)


def main(args=None):
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
