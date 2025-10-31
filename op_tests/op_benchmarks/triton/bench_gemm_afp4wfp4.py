import sys
import os
import torch
import triton
import math
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_weight_scales,
)
from op_tests.triton_tests.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
    run_torch,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_gemm_model_benchmark_object,
    get_gemm_shape_benchmark_object,
    print_vgpr,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    shuffle: bool,
    use_torch: bool = False,
):
    c_dtype = torch.bfloat16
    x, w, w_triton, x_scale, w_scale, x_scale_triton, w_scale_triton, _, y = (
        generate_gemm_afp4wfp4_inputs(
            M,
            N,
            K,
            c_dtype,
            layout=layout,
            output=True,
            shuffle_scales_fg=shuffle,
            shuffle_weight_fg=shuffle,
        )
    )
    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
    mem_read += (
        x_scale.numel() * x_scale.element_size()
        + w_scale.numel() * w_scale.element_size()
    )
    mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write

    if use_torch:
        ms = triton.testing.do_bench(
            lambda: run_torch(x, w, x_scale, w_scale, c_dtype),
            warmup=25,
            rep=100,  # noqa: E731
        )
    else:
        if shuffle:
            ms = triton.testing.do_bench(
                lambda: gemm_afp4wfp4_preshuffled_weight_scales(
                    x, w_triton, x_scale_triton, w_scale_triton, c_dtype, y
                ),
                warmup=25,
                rep=100,
            )
        else:
            ms = triton.testing.do_bench(
                lambda: gemm_afp4wfp4(
                    x, w_triton, x_scale_triton, w_scale_triton, c_dtype, y
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


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def run_model_benchmark(args):
    benchmark = get_gemm_model_benchmark_object("GEMM MXFP4 x MXFP4 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4(
        M, hidden_dim, intermediate_dim, metric, provider, model_name=None, **kwargs
    ):
        if provider[1] == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            # Divide N by tensor parallel
            N = math.ceil(N / args.tp)
        elif provider[1] == "fc2":
            N, K = hidden_dim, intermediate_dim
            # Divide K by tensor parallel
            K = math.ceil(K / args.tp)

        return bench_gemm_fn(
            M,
            N,
            K,
            metric,
            args.layout,
            args.shuffle,
            use_torch=(provider[0] == "torch"),
        )

    bench_gemm_afp4wfp4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_gemm_shape_benchmark_object("GEMM MXFP4 x MXFP4 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4(M, N, K, metric, provider, model_name=None, **kwargs):
        return bench_gemm_fn(
            M, N, K, metric, args.layout, args.shuffle, use_torch=(provider == "torch")
        )

    bench_gemm_afp4wfp4.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser("MXFP4 x MXFP4 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--shuffle", action="store_true", help="Preshuffle weight and scales"
    )
    return get_ff_args(parser)


def main():
    if not (arch_info.is_fp4_avail()):
        print("MXFP4 is not available on this architecture")
        sys.exit()

    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)  # noqa: E731
        print_vgpr(fun, "GEMM")
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
