import functools
import math

import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16x3 import gemm_a16w16x3
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16x3 import (
    generate_gemm_a16w16x3_inputs,
)


def bench_gemm_fn(bencher: RotatingBufferBench, M: int, N: int, K: int, metric: str):
    _p_gen = functools.partial(generate_gemm_a16w16x3_inputs, M, N, K, output=True)
    # (x, weight, w, y): the kernel takes the bf16 terms w, not the fp32 weight
    _out_vals_keys = ["x", "_", "w", "y"]
    rot_bufs = bencher.build_pool([_p_gen], [_out_vals_keys])

    # three bf16 MFMA dots for each output, one for each weight term
    flops = 3 * 2.0 * M * N * K
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + 3 * (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    ms = bencher.bench(gemm_a16w16x3)

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

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, K, metric, model_name=None, **kwargs):
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(bencher, M, N, K, metric)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    run_shape_benchmark(args)


def parse_args(args=None):
    parser = get_parser(kernel_name="A16W16X3 GEMM (bf16x3 fp32-weight GEMM)")
    parser = add_argparse_ff(parser)
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
