import functools
import math
import sys
from collections.abc import Callable

import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8_preshuffle
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)

# generate_gemm_a8w8_inputs returns (x, weight, weight_shuffled, x_scale, w_scale,
# bias, y); the kernels take the (possibly shuffled) weight as `w`, and the plain
# `weight` return value is unused here.
_OUT_VALS_KEYS = ["x", "_", "w", "x_scale", "w_scale", "bias", "y"]


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    shuffle: bool,
    impl: Callable,
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = str_to_torch_dtype["bf16"]
    _p_gen = functools.partial(
        generate_gemm_a8w8_inputs,
        M,
        N,
        K,
        str_to_torch_dtype["fp8e4m3"],
        c_dtype,
        layout=layout,
        output=True,
        shuffle=shuffle,
    )

    # create rotating buffers pool
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_OUT_VALS_KEYS], **kwargs_rot_bufs
    )

    # Kernel FLOPS and Mem Accesses(bytes)
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    kernel_fn = functools.partial(impl, dtype=c_dtype)
    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

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


def run_model_benchmark(args, impl):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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

        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
            args.shuffle,
            impl,
        )

    bench_gemm_a8w8.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, impl):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(M, N, K, metric, model_name=None, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            kwargs_rot_bufs,
            kwargs_do_bench_graph,
            M,
            N,
            K,
            metric,
            args.layout,
            args.shuffle,
            impl,
        )

    bench_gemm_a8w8.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    backend = "gluon" if args.gluon else "triton"
    if args.shuffle:
        impl = functools.partial(gemm_a8w8_preshuffle, backend=backend)
    elif args.gluon:
        impl = functools.partial(triton_gemm_a8w8, backend="gluon")
    else:
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
        "--gluon",
        action="store_true",
        help="Use Gluon implementation",
    )
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
