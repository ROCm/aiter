import sys
import torch
import triton
import math
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
    gemm_a16w8_blockscale,
    gemm_a16w8_blockscale_preshuffle,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
    generate_gemm_a16w8_blockscale_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_shape_benchmark_object,
    print_vgpr,
    get_caller_name_no_ext,
)

block_shape = (128, 128)


def bench_gemm_fn(M: int, N: int, K: int, metric: str, layout: str, preshuffle: bool = False):
    block_shape_n, block_shape_k = block_shape
    c_dtype = torch.bfloat16

    x, weight, weight_shuffled, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
        M, N, K, block_shape_n, block_shape_k,
        dtype=c_dtype, layout=layout, output=True, shuffle=preshuffle,
    )
    if preshuffle:
        bench_weight = weight_shuffled
    else:
        bench_weight = weight

    flops = 2.0 * M * N * K
    mem_read = (M * K) * x.element_size() + (N * K) * weight.element_size()
    mem_write = (M * N) * 2
    mem = mem_read + mem_write

    if preshuffle:
        ms = triton.testing.do_bench(
            lambda: gemm_a16w8_blockscale_preshuffle(
                x, bench_weight, w_scale, c_dtype, y, prequant=False
            ),
            warmup=25, rep=100,
        )
    else:
        ms = triton.testing.do_bench(
            lambda: gemm_a16w8_blockscale(
                x, bench_weight, w_scale, c_dtype, y, prequant=False
            ),
            warmup=25, rep=100,
        )

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    else:
        raise ValueError("Unknown metric: " + metric)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, K, metric, model_name=None, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.layout, preshuffle=args.preshuffle)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    run_shape_benchmark(args)


def parse_args(args=None):
    parser = get_parser(kernel_name="A16W8 Blockscale GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-preshuffle",
        action="store_true",
        help="Use preshuffle implementation",
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
