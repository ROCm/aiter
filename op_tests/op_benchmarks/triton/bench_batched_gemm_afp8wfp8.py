import math
import sys

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import _is_gluon_available
from aiter.ops.triton.gemm.batched.batched_gemm_afp8wfp8 import (
    batched_gemm_afp8wfp8_preshuffle as batched_gemm_afp8wfp8_preshuffle,  # noqa: PLC0414  explicit re-export (PEP 484 redundant-alias form)
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    batched_model_benchmark_shapes,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_afp8wfp8 import (
    generate_batched_gemm_afp8wfp8_preshuffle_inputs,
)


def bench_gemm_fn(batch: int, M: int, N: int, K: int, metric: str, model_name=None):
    c_dtype = torch.bfloat16

    # input preparation is not timed
    x, _, w, x_scales, w_scales, y = generate_batched_gemm_afp8wfp8_preshuffle_inputs(
        batch, M, N, K, c_dtype, output=True
    )

    # flops
    flops = 2.0 * M * N * K * batch
    # memory transfer (1-byte inputs)
    mem_read = x.numel() + w.numel() + x_scales.numel() + w_scales.numel()
    mem_write = (batch * M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: batched_gemm_afp8wfp8_preshuffle(
            x,
            w,
            x_scales,
            w_scales,
            c_dtype,
            y,
            is_x_scale_transposed=False,
        ),
        warmup=25,
        rep=100,
    )

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

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_afp8wfp8_preshuffle(
        M, hidden_dim, intermediate_dim, batch, metric, layer, model_name=None, **kwargs
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

        return bench_gemm_fn(batch, M, N, K, metric)

    bench_batched_gemm_afp8wfp8_preshuffle.run(
        save_path="." if args.o else None, print_data=True
    )


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["batch", "M", "N", "K"],
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_afp8wfp8_preshuffle(batch, M, N, K, metric, **kwargs):
        return bench_gemm_fn(batch, M, N, K, metric)

    bench_batched_gemm_afp8wfp8_preshuffle.run(
        save_path="." if args.o else None, print_data=True
    )


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    if args.model:
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
    parser = get_parser("Batched MXFP8 x MXFP8 GEMM (1x128 x scales, 128x128 w scales)")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-B",
        type=int,
        required=False,
        help="Batch size to be used when using --model flag.",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    if not _is_gluon_available():
        print("batched_gemm_afp8wfp8_preshuffle is gfx1250 (gluon) only")
        sys.exit()

    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
