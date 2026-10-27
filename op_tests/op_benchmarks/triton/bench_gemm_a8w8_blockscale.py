import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale_preshuffle as triton_gemm_a8w8_blockscale_preshuffle,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    config_launch,
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    metric_value,
    output_check,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
    run_torch,
)

block_shape = (128, 128)


def make_bench_case(
    kwargs_rot_bufs: dict,
    M: int,
    N: int,
    K: int,
    layout: str,
    shuffle: bool,
    backend: str | None,
):
    """(launch, rot_bufs, flops, mem, check) of one shape; backend None lets the wrapper pick.

    ``launch(config, **bufs)`` and ``check(config)`` take a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
    block_shape_n, block_shape_k = block_shape
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_gemm_a8w8_blockscale_inputs,
        M,
        N,
        K,
        block_shape_n,
        block_shape_k,
        layout=layout,
        output=True,
        shuffle=shuffle,
    )

    # (x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y); keep
    # only the weight/scale variant actually consumed by the kernel
    if shuffle:
        _out_vals_keys = ["x", "_", "w", "_", "x_scale", "w_scale", "y"]
    else:
        _out_vals_keys = ["x", "w", "_", "x_scale", "_", "w_scale", "y"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    impl = (
        triton_gemm_a8w8_blockscale_preshuffle
        if shuffle
        else triton_gemm_a8w8_blockscale
    )
    launch = config_launch(functools.partial(impl, dtype=c_dtype, backend=backend))

    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

    def reference():
        x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y = _p_gen()
        bufs = {
            "x": x,
            "w": weight_shuffled if shuffle else weight,
            "x_scale": x_scale_shuffled if shuffle else x_scale,
            "w_scale": w_scale,
            "y": y,
        }
        return bufs, run_torch(x, weight, x_scale, w_scale, torch.float32)

    check = output_check(launch, reference, "y", atol=0.01, rtol=1e-2)
    return launch, rot_bufs, flops, mem, check


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    backend: str | None,
    shuffle: bool = False,
    test: bool = False,
):
    launch, rot_bufs, flops, mem, check = make_bench_case(
        kwargs_rot_bufs, M, N, K, layout, shuffle, backend
    )
    if test:
        check(None)
    ms = do_bench_aiter_triton(
        functools.partial(launch, None), rot_bufs, **kwargs_do_bench_graph
    )
    return metric_value(metric, ms, flops, mem)


def run_model_benchmark(args, backend):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8_blockscale(
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
            backend,
            shuffle=args.preshuffle,
            test=args.test,
        )

    bench_gemm_a8w8_blockscale.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, backend):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8_blockscale(M, N, K, metric, model_name=None, **kwargs):
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
            backend,
            shuffle=args.preshuffle,
            test=args.test,
        )

    bench_gemm_a8w8_blockscale.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    backend = "gluon" if args.gluon else None
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


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A8W8 GEMM Blockscale")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-gluon",
        action="store_true",
        help="Use Gluon implementation (experimental, requires latest Triton from main)",
    )
    parser.add_argument(
        "-preshuffle",
        action="store_true",
        help="Use preshuffle implementation",
    )
    parser.add_argument(
        "-test",
        action="store_true",
        help="Run a correctness check for each benchmarked shape against a "
        "torch reference (mirrors op_tests/test_gemm_a8w8_blockscale.py).",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
