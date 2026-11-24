import functools
import math
import sys

import matplotlib.pyplot as plt
import torch
import triton

from aiter.ops.triton.gemm.feed_forward.ff_a16w16 import (
    ff_a16w16_gated,
    ff_a16w16_nogate,
)
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
    ff_a16w16_fused_gated,
)
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_ungated import (
    ff_a16w16_fused_ungated,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_do_bench_graph_kwargs,
    get_ff_args,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_shape_benchmark_object,
    model_benchmark_shapes,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import (
    generate_ff_inputs,
)


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

    evaluation_metric_to_unit = {
        "throughput": "TFLOPS",
        "time": "Time_(ms)",
        "bandwidth": "Bandwidth_(GB/s)",  # spaces break prettytable parsing
    }
    line_names = [evaluation_metric_to_unit[args.metric]]
    line_vals = line_names

    mpl_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="unit",
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


# generate_ff_inputs returns (x, w1, w2, out_dtype, intermediate, y). out_dtype is
# not a tensor and the wrappers do not read intermediate, so drop both.
_POOL_KEYS = ["x", "w1", "w2", "_", "_", "y"]


def bench_fn(
    bencher: RotatingBufferBench,
    batch: int,
    hidden_dim: int,
    intermediate_dim: int,
    metric: str,
    layout: str,
    gating: bool,
    activation: str | None = None,
    e2e_fused: bool = False,
    **kwargs,
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = torch.bfloat16
    bencher.build_pool(
        [
            functools.partial(
                generate_ff_inputs,
                batch,
                hidden_dim,
                intermediate_dim,
                c_dtype,
                layout=layout,
                gating=gating,
                output=True,
            )
        ],
        [_POOL_KEYS],
    )
    elem = c_dtype.itemsize

    # flops
    if gating:
        flops_gemm1 = 2.0 * batch * hidden_dim * intermediate_dim * 2
        flops_gate = batch * intermediate_dim
        flops_gemm2 = 2.0 * batch * intermediate_dim * hidden_dim
        flops = flops_gemm1 + flops_gate + flops_gemm2
        if activation is not None:
            flops += batch * intermediate_dim
    else:
        flops = 4.0 * batch * hidden_dim * intermediate_dim
        if activation is not None:
            flops += batch * intermediate_dim

    # memory transfer
    if gating:
        mem_read = (batch * intermediate_dim) * elem + (
            hidden_dim * intermediate_dim * 2
        ) * elem
    else:
        mem_read = (
            batch * intermediate_dim * elem + (hidden_dim * intermediate_dim) * elem
        )
    mem_write = (batch * hidden_dim) * elem
    mem = mem_read + mem_write

    if e2e_fused:
        fn = ff_a16w16_fused_gated if gating else ff_a16w16_fused_ungated
    else:
        fn = ff_a16w16_gated if gating else ff_a16w16_nogate

    def kernel_fn(x, w1, w2, y):
        fn(x, w1, w2, c_dtype, y=y, activation=activation)

    ms = bencher.bench(kernel_fn)

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


def run_model_benchmark(args, bencher):
    """
    Runs benchmark given a --model argument.
    """
    if args.e2e:
        label = "E2E"
    else:
        label = "Act+Gate+GEMM" if not args.ungated else "Act+GEMM"
    benchmark = get_model_benchmark_object(f"Fused FF A16W16 {label} Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_a16w16(M, hidden_dim, intermediate_dim, metric, **kwargs):
        intermediate_dim = math.ceil(intermediate_dim / args.tp)

        return bench_fn(
            bencher,
            M,
            hidden_dim,
            intermediate_dim,
            metric,
            args.layout,
            gating=not args.ungated,
            activation=args.activation,
            e2e_fused=args.e2e,
        )

    bench_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, bencher):
    """
    Runs a benchmark with given tensor shapes.
    """
    if args.e2e:
        label = "E2E"
    else:
        label = "Act+Gate+GEMM" if not args.ungated else "Act+GEMM"
    benchmark = get_shape_benchmark_object(f"Fused FF A16W16 {label} Benchmark", args)

    @triton.testing.perf_report([benchmark])
    def bench_a16w16(M, N, K, metric, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_fn(
            bencher,
            M,
            N,
            K,
            metric,
            args.layout,
            gating=not args.ungated,
            activation=args.activation,
            e2e_fused=args.e2e,
        )

    bench_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    if args.model:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking with the --model flag."
                )
        run_model_benchmark(args, bencher)
    else:
        unsupported_args = []
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args, bencher)


def parse_args():
    parser = get_parser(kernel_name="Fused FF")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Optional activation function to apply to the output. One of ('gelu', 'gelu_tanh', 'silu', 'silu_exp2', 'relu').",
    )
    parser.add_argument(
        "-ungated",
        action="store_true",
        help="Use an ungated FF (e.g silu instead of swiglu).",
    )
    parser.add_argument(
        "-e2e", action="store_true", help="Bench end-to-end (E2E) fused kernel."
    )
    return get_ff_args(parser)


def main():
    args, defaults = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args, defaults)
        print_vgpr(fun, "Fused FF")
        return 0
    run_benchmark(args, defaults)


if __name__ == "__main__":
    sys.exit(main())
