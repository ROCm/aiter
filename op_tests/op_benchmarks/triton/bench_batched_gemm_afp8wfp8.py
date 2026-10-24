import math
import sys

import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_afp8wfp8 import (
    batched_gemm_afp8wfp8 as batched_gemm_afp8wfp8,  # noqa: PLC0414  explicit re-export (PEP 484 redundant-alias form)
)
from aiter.ops.triton.utils._triton import arch_info
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
    generate_batched_gemm_afp8wfp8_inputs,
)


def bench_gemm_fn(
    batch: int,
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    w_scale_group_size: tuple[int, int],
    model_name=None,
):
    c_dtype = torch.bfloat16
    x, w, x_scale, w_scale, y = generate_batched_gemm_afp8wfp8_inputs(
        batch,
        M,
        N,
        K,
        c_dtype,
        layout=layout,
        output=True,
        w_scale_group_size=w_scale_group_size,
    )
    flops = 2.0 * M * N * K * batch
    mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
    mem_read += (
        x_scale.numel() * x_scale.element_size()
        + w_scale.numel() * w_scale.element_size()
    )
    mem_write = y.numel() * y.element_size()
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: batched_gemm_afp8wfp8(
            x,
            w,
            x_scale,
            w_scale,
            c_dtype,
            y,
            w_scale_group_size=w_scale_group_size,
        ),
        warmup=25,
        rep=100,
    )

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
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
    def bench_batched_gemm_afp8wfp8(
        M, hidden_dim, intermediate_dim, batch, metric, layer, model_name=None, **kwargs
    ):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(batch, M, N, K, metric, args.layout, (args.w_scale_n, 32))

    bench_batched_gemm_afp8wfp8.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(
        plot_name=get_caller_name_no_ext(),
        args=args,
        x_names=["batch", "M", "N", "K"],
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_afp8wfp8(batch, M, N, K, metric, **kwargs):
        return bench_gemm_fn(batch, M, N, K, metric, args.layout, (args.w_scale_n, 32))

    bench_batched_gemm_afp8wfp8.run(save_path="." if args.o else None, print_data=True)


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
    parser = get_parser("MXFP8 x MXFP8 batched GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-B",
        type=int,
        required=False,
        help="Batch size to be used when using --model flag.",
    )
    parser.add_argument(
        "--w-scale-n",
        type=int,
        default=1,
        choices=(1, 32),
        help="Weight rows for each E8M0 scale: 1 (1x32 blocks) or 32 (32x32 blocks).",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    if arch_info.get_arch() != "gfx1250":
        print("batched_gemm_afp8wfp8 has configs only for gfx1250")
        sys.exit()

    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(parsed_args, defaults)

        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
