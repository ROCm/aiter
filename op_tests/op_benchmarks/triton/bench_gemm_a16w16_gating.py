import sys
import torch
import triton
import math
from aiter.ops.triton.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.test_gemm_a16w16_gated import (
    generate_gemm_a16w16_gated_inputs,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    get_ff_args,
)

from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    model_benchmark_shapes,
    get_shape_benchmark_object,
    print_vgpr,
)
import matplotlib.pyplot as plt


def get_model_benchmark_object(
    plot_name,
    args,
    x_names=None,
):
    """
    Utility function for returning a triton.testing.Benchmark object to populate.

    Note: This is for benchmarking models (e.g with the --model arg).
    """
    if x_names is None:
        x_names = ["M", "hidden_dim", "intermediate_dim", "model_name"]
    x_vals_list = model_benchmark_shapes(args)

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = ["fc1"]
    line_vals = line_names

    mpl_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="layer",
        line_vals=line_vals,
        line_names=line_names,
        styles=[
            (mpl_colors[i], "-") for i in range(len(line_names))
        ],  # match line names to colors
        ylabel=ylabel,
        plot_name=plot_name,
        args={"metric": args.metric},
    )
    return benchmark


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    activation: str = None,
    **kwargs,
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = torch.bfloat16
    x, w, out_dtype, y = generate_gemm_a16w16_gated_inputs(
        M, N, K, c_dtype, layout=layout, output=True
    )

    # flops
    flops = 2.0 * M * N * K + M * N  # GEMM + gating
    if activation is not None:
        flops += M * N  # elementwise ops on the GEMM output

    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N // 2) * x.element_size()
    mem = mem_read + mem_write
    ms = triton.testing.do_bench(
        lambda: gemm_a16w16_gated(x, w, c_dtype, y, activation=activation),
        warmup=25,
        rep=100,  # noqa: E731
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
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object("Fused-act-gate GEMM A16W16 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, hidden_dim, intermediate_dim, metric, **kwargs):
        N, K = intermediate_dim * 2, hidden_dim
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        # print(f"Layer: {layer}, M: {M}, N: {N}, K: {K}, hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

        return bench_gemm_fn(M, N, K, metric, args.layout, activation=args.activation)

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object("Fused-act-gate GEMM A16W16 Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, metric, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(M, N, K, metric, args.layout, activation=args.activation)

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


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
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise Exception(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args)


def parse_args():
    parser = get_parser(kernel_name="Fused-act-gate A16W16 GEMM")
    parser.add_argument(
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel (divides intermediate_size)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["TT", "TN", "NT", "NN"],
        default="TN",
        help="Layout of input and weight matrix",
    )
    parser.add_argument(
        "-M",
        type=int,
        default=None,
        help="M dim of model benchmark if only one model is under test",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="user-defined shape to benchmark. Can be 3D (M, N, K) or 4D (B, M, N, K) for batched kernels.",
    )
    parser.add_argument(
        "--print_vgpr",
        action="store_true",
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Optional activation function to apply to the output. One of ('gelu', 'gelu_tanh', 'silu', 'silu_exp2', 'relu').",
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)  # noqa: E731
        print_vgpr(fun, "Fused-act-gate")
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
