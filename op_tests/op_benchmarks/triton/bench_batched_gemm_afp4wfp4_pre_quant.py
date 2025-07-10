import argparse
import sys
import torch
import triton
from op_tests.triton_tests.test_batched_gemm_afp4wfp4_pre_quant import (
    generate_batched_gemm_afp4wfp4_pre_quant_inputs,
)
from utils.benchmark_utils import get_model_configs, get_available_models

from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import (
    batched_gemm_afp4wfp4_pre_quant as batched_gemm_afp4wfp4_pre_quant,
)


def model_benchmark_shapes(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    M_list = [args.M] if args.model == "all" else [2**i for i in range(0, 15)]
    shapes = []
    for M in M_list:
        for _, config in configs.items():
            N = config["intermediate_size"]
            K = config["hidden_size"]

            shapes.append((16, M, N, K))

    return shapes


def get_x_vals():
    x_vals = [
        (16, 1, 1280, 8192),
        (16, 32, 1280, 8192),
        (16, 64, 1280, 8192),
        (16, 128, 1280, 8192),
        (16, 192, 1280, 8192),
        (16, 256, 1280, 8192),
        (16, 320, 1280, 8192),
        (16, 512, 1280, 8192),
        (16, 1024, 1280, 8192),
        (16, 2048, 1280, 8192),
        (16, 4096, 1280, 8192),
        (16, 8192, 1280, 8192),
        (16, 16384, 1280, 8192),
    ]
    return x_vals


def run_benchmark(args):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    x_names = ["batch", "M", "N", "K"]
    if args.model:
        x_vals_list = model_benchmark_shapes(args)
    elif args.shape:
        x_vals_list = [args.shape]
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
        plot_name="GEMM MXFP4 x MXFP4 Benchmark",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4_blockscale(batch, M, N, K, metric, provider):
        c_dtype = torch.bfloat16
        x, w, x_scale, w_scale = generate_batched_gemm_afp4wfp4_pre_quant_inputs(
            batch, M, N, K
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
        out = torch.empty(
            x.shape[0], x.shape[1], w.shape[2], device=x.device, dtype=c_dtype
        )

        ms = triton.testing.do_bench(
            lambda: batched_gemm_afp4wfp4_pre_quant(
                x, w, x_scale, w_scale, c_dtype, out
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

    bench_gemm_afp4wfp4_blockscale.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 x MXFP4 GEMM",
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
        nargs=4,
        metavar=("batch", "M", "N", "K"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
