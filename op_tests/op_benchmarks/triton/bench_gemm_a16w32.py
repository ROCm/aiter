import functools
import math

import triton

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w32 import _get_config
from aiter.ops.triton.gemm.basic.gemm_a16w32 import gemm_a16w32
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
from op_tests.triton_tests.gemm.basic.test_gemm_a16w32 import (
    generate_gemm_a16w32_inputs,
)


def _gen_inputs(M, N, K, keys):
    names = ("x", "w", "w_bf16x3", "y")
    inputs = dict(zip(names, generate_gemm_a16w32_inputs(M, N, K, output=True)))
    return tuple(inputs[key] for key in keys)


def bench_gemm_fn(bencher: RotatingBufferBench, M: int, N: int, K: int, metric: str):
    # The config of M selects one kernel, which reads only w (fp32 FMA) or only
    # w_bf16x3 (bf16x3). That weight rotates, and the other one is fixed.
    config, _ = _get_config(M, N, K)
    fma = config.get("kernel_type") == "bandwidth_bound"
    read, unread = ("w", "w_bf16x3") if fma else ("w_bf16x3", "w")
    rot_keys = ["x", read, "y"]
    rot_bufs = bencher.build_pool(
        [functools.partial(_gen_inputs, M, N, K, rot_keys)],
        [rot_keys],
        inp_gen_fixed_fns=[functools.partial(_gen_inputs, M, N, K, [unread])],
        inp_fixed_keys=[[unread]],
    )

    flops = 2.0 * M * N * K
    # the fp32 weight size: the FMA kernel reads it, the bf16x3 kernel reads 1.5x
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    ms = bencher.bench(gemm_a16w32)

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
    parser = get_parser(kernel_name="A16W32 GEMM (bf16 activations, fp32 weights)")
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
