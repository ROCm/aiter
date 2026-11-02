import functools
import math
import sys

import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
    batched_gemm_a16wfp4,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    batched_model_benchmark_shapes,
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16wfp4 import (
    generate_batched_gemm_a16wfp4_inputs,
)


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
    c_dtype = torch.bfloat16

    _p_gen = functools.partial(
        generate_batched_gemm_a16wfp4_inputs,
        batch,
        M,
        N,
        K,
        c_dtype,
        layout=layout,
        output=True,
    )
    # (x, w, x_scales, w_scales, y); kernel's weight-scale kwarg is "w_scales"
    _out_vals_keys = ["x", "w", "_", "w_scales", "y"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    kernel_fn = functools.partial(batched_gemm_a16wfp4, dtype=c_dtype)

    # flops
    flops = 2.0 * M * N * K * batch
    # memory transfer. w is e2m1, packed two values per byte (K // 2 wide);
    # w_scales keeps its own numel (one scale per 32 K-elements, which doesn't
    # reduce to a clean M/N/K product).
    mem_read = (batch * M * K) * rot_bufs[0]["x"].element_size() + (
        batch * N * K // 2
    ) * rot_bufs[0]["w"].element_size()
    mem_read += rot_bufs[0]["w_scales"].numel() * rot_bufs[0]["w_scales"].element_size()
    mem_write = (batch * M * N) * rot_bufs[0]["y"].element_size()
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
    def bench_batched_gemm_a16wfp4(
        M, hidden_dim, intermediate_dim, batch, metric, layer, **kwargs
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
        # print(f"Layer: {layer}, B: {batch}, M: {M}, N: {N}, K: {K}, hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

        return bench_gemm_fn(
            kwargs_rot_bufs, kwargs_do_bench_graph, batch, M, N, K, metric, args.layout
        )

    bench_batched_gemm_a16wfp4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["batch", "M", "N", "K"],
    )

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a16wfp4(
        batch,
        M,
        N,
        K,
        metric,
        **kwargs,
    ):
        return bench_gemm_fn(
            kwargs_rot_bufs, kwargs_do_bench_graph, batch, M, N, K, metric, args.layout
        )

    bench_batched_gemm_a16wfp4.run(save_path="." if args.o else None, print_data=True)


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


def parse_args(args: list[str] | None = None):
    parser = get_parser("Batched MXFP4 x MXFP4 GEMM, Pre Quant")
    parser = add_argparse_ff(parser)

    # Overriding to reduce the runtime of the benchmark
    override_parser_action(parser, action="--graph-kernels-per-graph", default=200)

    parser.add_argument(
        "-B",
        type=int,
        required=False,
        help="Batch size to be used when using --model flag.",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None):
    if not (arch_info.is_fp4_avail()):
        print("MXFP4 is not available on this architecture")
        sys.exit()

    args, defaults = parse_args(args=args)
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(args, defaults)


if __name__ == "__main__":
    main()
