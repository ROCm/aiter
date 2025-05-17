import argparse
import sys
import torch
import triton
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from op_tests.triton_tests.test_gemm_a16w16 import generate_gemm_a16w16_inputs
from utils.benchmark_utils import get_model_configs, get_available_models


def model_benchmark_shapes(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    M_list = [args.M] if args.model == "all" else [2**i for i in range(0, 15)]
    shapes = []
    for M in M_list:
        for _, config in configs.items():
            N = config["intermediate_size"]
            K = config["hidden_size"]

            shapes.append((M, N, K, "TN"))

    return shapes


def get_x_vals():
    x_vals = [
        (1, 1280, 8192, "TN"),
        (32, 1280, 8192, "TN"),
        (64, 1280, 8192, "TN"),
        (128, 1280, 8192, "TN"),
        (192, 1280, 8192, "TN"),
        (256, 1280, 8192, "TN"),
        (320, 1280, 8192, "TN"),
        (512, 1280, 8192, "TN"),
        (1024, 1280, 8192, "TN"),
        (2048, 1280, 8192, "TN"),
        (4096, 1280, 8192, "TN"),
        (8192, 1280, 8192, "TN"),
        (16384, 1280, 8192, "TN"),
        (8192, 7168, 20480, "NT"),
        (1024, 20480, 8192, "NT"),
        (1024, 8192, 20480, "NT"),
        (8192, 7168, 20480, "TN"),
        (1024, 20480, 8192, "TN"),
        (1024, 8192, 20480, "TN"),
    ]
    return x_vals


def run_benchmark(args):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    x_names = ["M", "N", "K", "layout"]
    if args.model:
        x_vals_list = model_benchmark_shapes(args)
    elif args.shape:
        x_vals_list = [args.shape + [args.layout]]
    else:
        x_vals_list = get_x_vals()

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = ["Triton"]
    line_vals = ["triton"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name="GEMM A16W16 Benchmark",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, layout, metric, provider):
        # NOTE: Assume bias and output has the same dtype
        c_dtype = torch.bfloat16
        x, w, y = generate_gemm_a16w16_inputs(M, N, K, c_dtype, layout, output=True)
        # flops
        flops = 2.0 * M * N * K
        # memory transfer
        mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
        mem_write = (M * N) * x.element_size()
        mem = mem_read + mem_write

        ms = triton.testing.do_bench(
            lambda: gemm_a16w16(x, w, c_dtype, y), warmup=25, rep=100  # noqa: E731
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

    bench_gemm_a16w16.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark A16W16 GEMM",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    available_models = get_available_models()  # Dynamically load model names
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
        nargs=3,
        metavar=("M", "N", "K"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["TT", "TN", "NT", "NN"],
        default="TN",
        help="Layout of input and weight matrix",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
