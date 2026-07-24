"""Triton perf_report benchmark for (M, N) softmax over the last dim.
"""
import argparse
import torch
import torch.nn.functional as F
import triton
from aiter.ops.triton.softmax import softmax as aiter_softmax
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    get_caller_name_no_ext,
    print_vgpr,
)


def _n_values_from_model_cfg(cfg: dict, config_n: str) -> list[int]:
    """Last-dim sizes N to try when benchmarking from ``model_configs.json``.

    Softmax runs along N; ``hidden_size`` / ``intermediate_size`` are the usual
    vocabulary for transformer blocks (attention vs FFN-like widths).
    """
    hidden = int(cfg["hidden_size"])
    inter = cfg.get("intermediate_size")
    inter_i = int(inter) if inter is not None else None

    if config_n == "hidden":
        return [hidden]
    if config_n == "intermediate":
        if inter_i is None:
            return [hidden]
        return [inter_i]
    if inter_i is None or inter_i == hidden:
        return [hidden]
    return [hidden, inter_i]


def model_benchmark_shapes(args):
    """Build (model_name, M, N) rows for the x-axis of the Triton Benchmark.

    - Default ``--model`` omitted → single config family ``llama3`` (same as RMSNorm bench).
    - ``--model all`` → every model in JSON; M fixed to ``-M`` for each.
    - Otherwise → M sweeps 2**0 .. 2**14 for the selected model(s).
    """
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file, models="llama3" if args.model is None else args.model
    )
    M_list = [args.M] if args.model == "all" else [2**i for i in range(0, 15)]
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            for N in _n_values_from_model_cfg(config, args.config_n):
                shapes.append((model_name, M, N))

    return shapes


def get_x_vals():
    """Fixed (M, N) grid; kept for parity with ``bench_rmsnorm.get_x_vals``."""
    x_vals = [
        (1, 1280),
        (32, 1280),
        (64, 1280),
        (128, 1280),
        (192, 1280),
        (256, 1280),
        (320, 1280),
        (512, 1280),
        (1024, 1280),
        (2048, 1280),
        (4096, 1280),
        (8192, 1280),
        (16384, 1280),
    ]
    return x_vals


def run_benchmark(args):
    # Disallow mixing --shape with --model (and implicit -M default), matching RMSNorm bench.
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    act_dtype = str_to_torch_dtype[args.dtype]

    # perf_report maps each tuple entry to these names (order must match x_vals rows).
    x_names = ["model_name", "M", "N"]
    if args.shape is not None:
        M, N = args.shape
        x_vals_list = [("custom", M, N)]
    else:
        x_vals_list = model_benchmark_shapes(args)

    if args.metric == "time":
        ylabel = "Time_(ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth_(GB/s)"
    elif args.metric == "throughput":
        ylabel = "Throughput_(TFLOPS)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    # Single line in the plot; actual quantity is selected inside bench_softmax via ``metric``.
    line_names = [ylabel]
    line_vals = [ylabel]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        # Passed into each bench invocation alongside (model_name, M, N) from x_vals.
        args={"metric": args.metric, "kernel": args.kernel},
    )

    @triton.testing.perf_report([benchmark])
    def bench_softmax(M, N, metric, model_name=None, kernel=None, **kwargs):
        # Fresh tensor each call so we do not bench an in-place-only steady state.
        x = torch.randn(M, N, dtype=act_dtype, device="cuda")

        # Bandwidth: treat softmax as read full x + write full output (same dtype).
        mem_read = M * N * x.element_size()
        mem_write = M * N * x.element_size()
        mem = mem_read + mem_write

        if kernel == "aiter":

            def fn():
                aiter_softmax(x)

        elif kernel == "torch":

            def fn():
                F.softmax(x, dim=-1)

        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        if metric == "time":
            return ms
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9
            return bandwidth
        elif metric == "throughput":
            # Rough elementwise work: ~5 passes over the row (shift, exp, sum, normalize, etc.).
            flops = 5 * M * N
            tflops = flops / ms * 1e-9
            return tflops
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_softmax.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark Softmax",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    available_models = get_available_models()
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for the default benchmark script."
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument("--model", type=str, help=model_help)
    parser.add_argument(
        "-M",
        type=int,
        default=4096,
        help="M dim of model benchmark if only one model is under test",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--config-n",
        choices=("hidden", "intermediate", "both"),
        default="both",
        help=(
            "When using model configs: which dims to use as softmax N "
            "(hidden_size, intermediate_size, or both)."
        ),
    )
    parser.add_argument(
        "--kernel",
        choices=("aiter", "torch"),
        default="aiter",
        help="aiter: Triton softmax; torch: torch.nn.functional.softmax.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Input dtype (str_to_torch_dtype), e.g. bf16, fp16.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth", "throughput"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parsed = parser.parse_args(args=args)
    return parsed


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        # Re-run benchmark under Triton instrumentation to report register pressure.
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
