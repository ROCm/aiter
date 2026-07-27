import torch
import triton

from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
from op_tests.triton_tests.fusions.test_fused_mul_add import (
    generate_fused_mul_add_inputs,
)

eval_metrics_to_units = {
    "throughput": "TFLOPS",
    "time": "Time_(ms)",
    "bandwidth": "Bandwidth_(GB/s)",
}

default_unused_args = ("fc1", "fc2", "no_glu", "tp", "layout")


def metric_to_scalar(metric, ms, flops, mem):
    """Return the single scalar for the active metric."""
    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9  # TFLOPS
    elif metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError("Unknown metric: " + metric)


def parse_fused_args(kernel_name, args=None):
    """Build the standard FF parser and return (parsed_args, defaults).

    get_ff_args destructures a 4-element --shape as (B, M, N, K); the fused
    benches read args.shape directly, so that mapping is unused here.
    """
    parser = get_parser(kernel_name=kernel_name)
    parser = add_argparse_ff(parser)
    return get_ff_args(parser, args=args)


def get_fused_shape_benchmark_object(plot_name, args, x_names, get_x_vals):
    """triton.testing.Benchmark for a fused shape sweep with len(x_names) dims.

    Unlike benchmark_utils.get_shape_benchmark_object (which models (M, N, K) /
    (B, M, N, K)), this enforces an exact len(x_names)-tuple --shape and does NOT
    apply the (B, M, N, K) remapping: a 4-tuple here is (M, N8, N16, K), etc.
    """
    if args.shape:
        if len(args.shape) != len(x_names):
            raise ValueError(
                f"--shape expects {len(x_names)} ints "
                f"({' '.join(x_names)}); got {len(args.shape)}: {args.shape}"
            )
        x_vals_list = [args.shape]
    else:
        x_vals_list = get_x_vals(args=args)

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="unit",
        line_vals=[eval_metrics_to_units[args.metric]],
        line_names=[eval_metrics_to_units[args.metric]],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=plot_name,
        args={"metric": args.metric},
    )
    return benchmark


def run_fused_shape_benchmark(args, x_names, bench_fn, get_x_vals, plot_name):
    """Wrap bench_fn in a perf_report over the fused Benchmark object and run it."""
    benchmark = get_fused_shape_benchmark_object(plot_name, args, x_names, get_x_vals)

    @triton.testing.perf_report([benchmark])
    def _run(**kwargs):
        shape = tuple(kwargs[name] for name in x_names)
        return bench_fn(*shape, kwargs["metric"])

    _run.run(save_path="." if args.o else None, print_data=True)


def run_fused_benchmark(
    args,
    defaults,
    kernel_label,
    shape,
    x_names,
    bench_fn,
    get_x_vals,
    plot_name,
    unsupported_args=default_unused_args,
):
    """Shared dispatcher: reject --model and unsupported FF args, else run sweep."""
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    # --model has no meaning here since we will not run models & single family shape
    if args.model:
        raise NotImplementedError(
            f"--model is not supported for {kernel_label}; it targets a single "
            f"fixed shape family {shape}. Use --shape "
            f"{' '.join(x_names)} or -M."
        )
    for arg in unsupported_args:
        if getattr(args, arg, None) != getattr(defaults, arg, None):
            raise Exception(f"Argument '{arg}' is not supported for {kernel_label}.")
    run_fused_shape_benchmark(args, x_names, bench_fn, get_x_vals, plot_name)


def make_mul_add_ab(M, N, dtype, a_kind="tensor", b_kind="tensor"):
    """Build the (a, b) mul/add operands for the fused mul_add benches.

    a_kind/b_kind: "tensor" -> full (M, N) tensor; "scalar" -> python float scalar.
    Delegates to the unit test's generator so bench inputs match the test exactly.
    """
    a_spec = (torch.Tensor, False) if a_kind == "tensor" else (float, True)
    b_spec = (torch.Tensor, False) if b_kind == "tensor" else (float, True)
    _, a, b = generate_fused_mul_add_inputs([M, N], a_spec, b_spec, dtype)
    return a, b
