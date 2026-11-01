import functools
import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16 import _is_gluon_available, gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
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
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)


def _generate_atomic_inputs(M, N, K, c_dtype, layout):
    # Accumulation in bf16/fp16 loses precision; the atomic kernel accumulates in fp32.
    x, w, _, _, y = generate_gemm_a16w16_inputs(
        M, N, K, c_dtype, layout=layout, output=True, bias=True
    )
    y = y.to(torch.float32).zero_()
    return x, w, y


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    backend: str,
    atomic: bool = False,
    activation: str | None = None,
    persistent: bool = False,
    kernel_type: str = "bandwidth_bound",
    **kwargs,
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = torch.bfloat16

    if atomic:
        # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
        assert backend != "gluon", "Atomic kernel is triton-only"
        assert (
            activation is None
        ), "Atomic kernel does not currently support fused activation"
        _p_gen = functools.partial(_generate_atomic_inputs, M, N, K, c_dtype, layout)
        _out_vals_keys = ["x", "w", "y"]  # matches gemm_a16w16_atomic's kwargs
        kernel_fn = functools.partial(gemm_a16w16_atomic, dtype=torch.float32)
    else:
        if persistent:
            assert not (backend == "gluon" and layout not in ("TN", "TT")), (
                f"--persistent with --backend gluon requires --layout TN or TT, got "
                f"'{layout}' (the gluon persistent kernel has no 'N' path for A)"
            )
        _p_gen = functools.partial(
            generate_gemm_a16w16_inputs,
            M,
            N,
            K,
            c_dtype,
            layout=layout,
            output=True,
            bias=True,
        )
        _out_vals_keys = [
            "x",
            "w",
            "bias",
            "dtype",
            "y",
        ]  # matches gemm_a16w16's kwargs
        kernel_fn = functools.partial(
            gemm_a16w16,
            activation=activation,
            kernel_type=kernel_type,
            backend=backend,
            persistent=persistent,
        )

    # create rotating buffers pool
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    # flops
    flops = 2.0 * M * N * K
    if activation is not None:
        flops += M * N  # elementwise ops on the GEMM output
    # memory transfer
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K) * rot_bufs[0][
        "w"
    ].element_size()
    mem_write = (M * N) * rot_bufs[0]["y"].element_size()
    mem = mem_read + mem_write

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


def run_model_benchmark(args, backend):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
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
            atomic=args.atomic,
            activation=args.activation,
            persistent=args.persistent,
            kernel_type=args.kernel_type,
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, backend):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, metric, **kwargs):
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
            atomic=args.atomic,
            persistent=args.persistent,
            kernel_type=args.kernel_type,
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    backend = args.backend or ("gluon" if _is_gluon_available() else "triton")
    print(f"Using backend: {backend}")

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
    parser = get_parser(kernel_name="A16W16 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Use the atomic kernel (split-k with atomic_add) instead of the standard a16w16 kernel.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function to apply to the output. One of ('gelu', 'gelu_tanh', 'silu', 'silu_exp2', 'relu').",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        default=False,
        help="Use the persistent kernel (gemm_a16w16(..., persistent=True)) instead of "
        "the standard a16w16 kernel",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["bandwidth_bound", "compute_bound"],
        default="bandwidth_bound",
        help="Kernel variant to use (gluon only). Default: bandwidth_bound.",
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
