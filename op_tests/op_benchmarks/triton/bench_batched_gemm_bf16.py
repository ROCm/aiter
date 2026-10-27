import functools
import math
import sys

import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    batched_model_benchmark_shapes,
    config_launch,
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    metric_value,
    output_check,
    print_vgpr,
    run_input_generator,
    splitk_block_size,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
    generate_batched_gemm_a16w16_inputs as generate_batched_gemm_bf16_inputs,
)


def make_bench_case(
    kwargs_rot_bufs: dict,
    batch: int,
    M: int,
    N: int,
    K: int,
    layout: str,
    backend: str | None,
):
    """(launch, rot_bufs, flops, mem, check) of one shape; backend None lets the wrapper pick.

    ``launch(config, **bufs)`` and ``check(config)`` take a config; None is the installed one.
    The tuning scripts in aiter/ops/triton/utils/_triton/tuning use this function.
    """
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_batched_gemm_bf16_inputs,
        batch,
        M,
        N,
        K,
        dtype=c_dtype,
        layout=layout,
        output=True,
    )
    # batched_gemm_bf16 names its activation/weight/output kwargs XQ/WQ/YQ
    _out_vals_keys = ["XQ", "WQ", "bias", "YQ"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    launch = config_launch(
        functools.partial(batched_gemm_bf16, dtype=c_dtype, backend=backend),
        splitk_block_size(K) if backend == "triton" else None,
    )

    # FLOPs for batched GEMM:
    # C[B, M, N] = A[B, M, K] @ W[B, N, K]^T
    flops = 2.0 * batch * M * N * K

    # Memory traffic. bias keeps its own numel ((N,)-shaped, which doesn't
    # reduce to a clean M/N/K product).
    bias_buf = rot_bufs[0]["bias"]
    mem_read = (batch * M * K) * rot_bufs[0]["XQ"].element_size() + (
        batch * N * K
    ) * rot_bufs[0]["WQ"].element_size()
    mem_read += (
        bias_buf.numel() * bias_buf.element_size() if bias_buf is not None else 0
    )
    mem_write = (batch * M * N) * rot_bufs[0]["YQ"].element_size()
    mem = mem_read + mem_write

    def reference():
        bufs = run_input_generator(_p_gen, _out_vals_keys)
        out = bufs["XQ"].float() @ bufs["WQ"].float().transpose(1, 2)
        return bufs, out + bufs["bias"].float()

    check = output_check(launch, reference, "YQ", atol=1e-2, rtol=1e-2)
    return launch, rot_bufs, flops, mem, check


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    batch: int,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
):
    launch, rot_bufs, flops, mem, _ = make_bench_case(
        kwargs_rot_bufs, batch, M, N, K, layout, None
    )
    ms = do_bench_aiter_triton(
        functools.partial(launch, None), rot_bufs, **kwargs_do_bench_graph
    )
    return metric_value(metric, ms, flops, mem)


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["M", "hidden_dim", "intermediate_dim", "batch", "model_name"],
        model_benchmark_shapes_fn=batched_model_benchmark_shapes,
    )

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_bf16(
        M,
        hidden_dim,
        intermediate_dim,
        batch,
        metric,
        layer,
        **kwargs,
    ):
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

        else:
            raise ValueError(f"Unsupported layer: {layer}")

        return bench_gemm_fn(
            kwargs_rot_bufs, kwargs_do_bench_graph, batch, M, N, K, metric, args.layout
        )

    bench_batched_gemm_bf16.run(
        save_path="." if args.o else None,
        print_data=True,
    )


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["batch", "M", "N", "K"],
    )

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_bf16(batch, M, N, K, metric, **kwargs):
        return bench_gemm_fn(
            kwargs_rot_bufs, kwargs_do_bench_graph, batch, M, N, K, metric, args.layout
        )

    bench_batched_gemm_bf16.run(
        save_path="." if args.o else None,
        print_data=True,
    )


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
            "tp",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args():
    parser = get_parser("Batched BF16 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-B",
        type=int,
        required=False,
        help="Batch size to be used when using --model flag.",
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
