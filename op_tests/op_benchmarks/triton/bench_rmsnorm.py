import argparse
import sys
import torch
import triton
from aiter.ops.triton.rmsnorm import rms_norm
from op_tests.triton_tests.test_rmsnorm import generate_rmsnorm_inputs
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
)


def model_benchmark_shapes(args):
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file, models="llama3" if args.model is None else args.model
    )
    M_list = [args.M] if args.model == "all" else [2**i for i in range(0, 15)]
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            N = config["hidden_size"]
            shapes.append((model_name, M, N))

    return shapes


def get_x_vals():
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
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    x_names = ["model_name", "M", "N"]
    x_vals_list = model_benchmark_shapes(args)

    if args.metric == "time":
        ylabel = "Time (ms)"
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
        plot_name="RMSNorm Fwd Benchmark",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_rmsnorm(M, N, metric, provider, model_name=None):
        c_dtype = torch.bfloat16
        x, w = generate_rmsnorm_inputs(M, N, c_dtype)

        # memory transfer
        mem_read = (M * 1) * N * x.element_size()  # x is (M,N) and g/weight is (N)
        mem_write = M * N * x.element_size()  # output
        mem = mem_read + mem_write

        eps = 1e-6
        ms = triton.testing.do_bench(lambda: rms_norm(x, w, eps), warmup=25, rep=100)

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_rmsnorm.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RMSNorm",
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
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
