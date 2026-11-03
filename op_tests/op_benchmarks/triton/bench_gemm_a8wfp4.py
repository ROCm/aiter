import functools
import math
import sys

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import (
    gemm_a8wfp4,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes
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
from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
    generate_gemm_a8wfp4_inputs,
)


def bench_gemm_fn(
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
):
    _e5m2_type, e4m3_type = get_fp8_dtypes()
    a_dtype = e4m3_type
    out_dtype = torch.float16

    _p_gen = functools.partial(
        generate_gemm_a8wfp4_inputs,
        M=M,
        N=N,
        K=K,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        layout=layout,
        output=True,
    )
    # (x, w, x_scales, w_scales, x_fp32, w_fp32, y)
    _out_vals_keys = ["x", "w", "x_scales", "w_scales", "_", "_", "y"]

    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_vals_keys], **kwargs_rot_bufs
    )

    kernel_fn = functools.partial(gemm_a8wfp4, dtype=out_dtype)

    # flops
    flops = 2.0 * M * N * K
    # memory transfer. w is e2m1, packed two values per byte (K // 2 wide); the
    # scale tensors keep their own numel (one scale per 32 K-elements, which
    # doesn't reduce to a clean M/N/K product).
    mem_read = (M * K) * rot_bufs[0]["x"].element_size() + (N * K // 2) * rot_bufs[0][
        "w"
    ].element_size()
    mem_read += (
        rot_bufs[0]["x_scales"].numel() * rot_bufs[0]["x_scales"].element_size()
        + rot_bufs[0]["w_scales"].numel() * rot_bufs[0]["w_scales"].element_size()
    )
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


def run_model_benchmark(args):
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8wfp4(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
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

        return bench_gemm_fn(
            kwargs_rot_bufs, kwargs_do_bench_graph, M, N, K, metric, args.layout
        )

    bench_gemm_a8wfp4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_af8wfp4(M, N, K, metric, model_name=None, **kwargs):
        return bench_gemm_fn(
            kwargs_rot_bufs, kwargs_do_bench_graph, M, N, K, metric, args.layout
        )

    bench_gemm_af8wfp4.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser("8-bit x 4-bit GEMM")
    parser = add_argparse_ff(parser)
    return get_ff_args(parser)


def main():
    from aiter.ops.triton.utils._triton import arch_info

    if not (arch_info.is_fp4_avail()):
        print("MXFP4 is not available on this architecture")
        sys.exit()

    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
