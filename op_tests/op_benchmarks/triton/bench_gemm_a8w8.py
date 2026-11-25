import functools
import math
import sys

import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8_preshuffle
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.types import str_to_torch_dtype
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
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)

# generate_gemm_a8w8_inputs returns (x, weight, weight_shuffled, x_scale, w_scale,
# bias, y); the kernels take the (possibly shuffled) weight as `w`. The plain
# `weight` is unused, and the bias too: most production linears have no bias.
_OUT_VALS_KEYS = ["x", "_", "w", "x_scale", "w_scale", "_", "y"]


def make_bench_case(
    bencher: RotatingBufferBench,
    M: int,
    N: int,
    K: int,
    layout: str,
    shuffle: bool,
    backend: str,
):
    """(launch, rot_bufs, flops, mem) of one shape, without bias.

    ``launch(config, **bufs)`` takes a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
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
    rot_bufs = bencher.build_pool([_p_gen], [_OUT_VALS_KEYS])

    # Kernel FLOPS and Mem Accesses(bytes)
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    impl = gemm_a8w8_preshuffle if shuffle else triton_gemm_a8w8
    launch = config_launch(
        functools.partial(impl, dtype=c_dtype, backend=backend),
        (
            (lambda config: compute_splitk_params(config, K))
            if backend == "triton"
            else None
        ),
    )

    return launch, rot_bufs, flops, mem


def bench_gemm_fn(
    bencher: RotatingBufferBench,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    shuffle: bool,
    backend: str,
):
    launch, _, flops, mem = make_bench_case(bencher, M, N, K, layout, shuffle, backend)
    ms = bencher.bench(functools.partial(launch, None))
    return metric_value(metric, ms, flops, mem)


def run_model_benchmark(args, backend):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

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
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            args.shuffle,
            backend,
        )

    bench_gemm_a8w8.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, backend):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(M, N, K, metric, model_name=None, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            args.shuffle,
            backend,
        )

    bench_gemm_a8w8.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    backend = "gluon" if args.gluon else "triton"
    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args, backend)
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
        run_shape_benchmark(args, backend)


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
