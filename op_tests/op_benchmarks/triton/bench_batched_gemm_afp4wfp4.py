import sys
import torch
import triton
import math
from op_tests.triton_tests.test_batched_gemm_afp4wfp4 import (
    generate_batched_gemm_afp4wfp4_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    get_model_configs,
)
from aiter.ops.triton.batched_gemm_afp4wfp4 import (
    batched_gemm_afp4wfp4 as batched_gemm_afp4wfp4,
)
import aiter.ops.triton.utils.arch_info as arch_info


def model_benchmark_shapes(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    M_list = [4096] if args.model == "all" else [2**i for i in range(0, 15)]
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            N = config["intermediate_size"]
            K = config["hidden_size"]

            shapes.append(
                (model_name, M, N, K, 16)
            )  # rearrange batch to last dim so M is graph x-axis

    return shapes


def bench_gemm_fn(batch: int, M: int, N: int, K: int, metric: str, layout: str,  model_name=None):
    c_dtype = torch.bfloat16
    x, w, x_scale, w_scale, y = generate_batched_gemm_afp4wfp4_inputs(
        batch,
        M,
        N,
        K,
        c_dtype,
        layout=layout,
        output=True,
    )
    # print(f"M: {M}, N: {N}, K: {K}, x.shape: {x.shape}, x.stride(): {x.stride()}, w.shape: {w.shape}, w.stride(): {w.stride()}")
    # flops
    flops = 2.0 * M * N * K * batch
    # memory transfer
    mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
    mem_read += (
        x_scale.numel() * x_scale.element_size()
        + w_scale.numel() * w_scale.element_size()
    )
    mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: batched_gemm_afp4wfp4(x, w, x_scale, w_scale, c_dtype, y),
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
        plot_name="Batched GEMM MXFP4 x MXFP4 Benchmark",
        args=args,
        x_names=["model_name", "M", "hidden_dim", "intermediate_dim", "batch"],
        model_benchmark_shapes_fn=model_benchmark_shapes,
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_afp4wfp4(
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
        #print(f"Layer: {layer}, B: {batch}, M: {M}, N: {N}, K: {K}, hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

        return bench_gemm_fn(batch, M, N, K, metric, layout=args.layout)

    bench_batched_gemm_afp4wfp4.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    benchmark = get_shape_benchmark_object(
        plot_name="Batched GEMM MXFP4 x MXFP4 Benchmark",
        args=args,
        x_names=["M", "N", "K", "batch"],
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_afp4wfp4(M, N, K, batch, metric, provider, model_name=None):
        return bench_gemm_fn(batch, M, N, K, metric, layout=args.layout)

    bench_batched_gemm_afp4wfp4.run(save_path="." if args.o else None, print_data=True)


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


def parse_args():
    parser = get_parser("MXFP4 x MXFP4 GEMM")
    parser = add_argparse_ff(parser)
    return get_ff_args(parser)


def main():
    if not (arch_info.is_fp4_avail()):
        print("MXFP4 is not available on this architecture")
        sys.exit()

    args, defaults = parse_args()
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
