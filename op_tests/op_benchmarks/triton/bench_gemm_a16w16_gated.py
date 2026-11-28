import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    config_launch,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    metric_value,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
    generate_gemm_a16w16_gated_inputs,
)


def make_bench_case(
    bencher: RotatingBufferBench,
    M: int,
    N: int,
    K: int,
    layout: str,
    activation: str | None = None,
):
    """(launch, rot_bufs, flops, mem) of one shape.

    ``launch(config, **bufs)`` takes a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_gemm_a16w16_gated_inputs, M, N, K, c_dtype, layout=layout, output=True
    )
    _out_vals_keys = ["x", "w", "_", "y"]

    rot_bufs = bencher.build_pool([_p_gen], [_out_vals_keys])

    launch = config_launch(
        functools.partial(gemm_a16w16_gated, dtype=c_dtype, activation=activation)
    )

    # flops: N is pre-gating dimension, output is N//2
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N // 2) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    return launch, rot_bufs, flops, mem


def bench_gemm_fn(
    bencher: RotatingBufferBench,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    activation: str | None = None,
):
    launch, _, flops, mem = make_bench_case(
        bencher, M, N, K, layout, activation=activation
    )
    ms = bencher.bench(functools.partial(launch, None))
    return metric_value(metric, ms, flops, mem)


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

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

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

        return bench_gemm_fn(
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            activation=args.activation,
        )

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, K, metric, model_name=None, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            activation=args.activation,
        )

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
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
