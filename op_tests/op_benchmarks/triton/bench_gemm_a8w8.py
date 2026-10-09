import functools
import math
import sys
from collections.abc import Callable

import torch
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8
from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
    gemm_a8w8_preshuffle as gluon_gemm_a8w8_preshuffle,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
    generate_rotating_buffers_pool,
    do_bench_aiter_triton
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)

def bench_gemm_fn(M: int, N: int, K: int, metric: str, layout: str, shuffle: bool, kernel_fn: Callable):
    c_dtype = str_to_torch_dtype["bf16"]

    #Input generator callable partial
    _p_gen = functools.partial(generate_gemm_a8w8_inputs,
        M,
        N,
        K,
        str_to_torch_dtype["fp8e4m3"],
        c_dtype,
        layout=layout,
        output=True,
        shuffle=shuffle
    )
    _out_vals_keys = ["x", "_", "w", "x_scale", "w_scale", "bias", "y"] #keys for outputs from the gen function

    #create rotating buffers pool
    rot_bufs = generate_rotating_buffers_pool(_p_gen, _out_vals_keys, target_mb=256, min_pool_len=4)

    #Kernel FLOPS and Mem Accesses(bytes)
    flops = 2.0 * M * N * K
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0]["w"].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs)

    #Return metric
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

    
def run_model_benchmark(args, impl):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
    ):
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

        return bench_gemm_fn(M, N, K, metric, args.layout, args.shuffle, impl)

    bench_gemm_a8w8.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, impl):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(M, N, K, metric, model_name=None, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.layout, args.shuffle, impl)

    bench_gemm_a8w8.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    
    if args.backend == "gluon":
        if args.shuffle:
            impl = gluon_gemm_a8w8_preshuffle
        else:
            impl = functools.partial(triton_gemm_a8w8, backend="gluon")
    else:
        if args.shuffle:
            raise RuntimeError(
                "Argument --shuffle is only supported with --gluon flag."
            )
        impl = triton_gemm_a8w8

    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args, impl)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args, impl)


def parse_args():
    parser = get_parser(kernel_name="A8W8 GEMM")
    parser = add_argparse_ff(parser)

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Preshuffle weight",
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
