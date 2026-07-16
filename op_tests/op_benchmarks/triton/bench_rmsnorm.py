import argparse
import torch
import triton
from aiter.ops.triton.normalization.rmsnorm import rms_norm
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_rms_mxfp4_quant
from op_tests.triton_tests.normalization.test_rmsnorm import (
    generate_rmsnorm_inputs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    get_caller_name_no_ext,
    print_vgpr,
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
        args={"metric": args.metric},
    )

    quant = args.quant
    add_residual = args.add_residual

    @triton.testing.perf_report([benchmark])
    def bench_rmsnorm(M, N, metric, model_name=None, **kwargs):
        c_dtype = torch.bfloat16
        x, w = generate_rmsnorm_inputs(M, N, c_dtype)
        eps = 1e-6

        if quant == "mxfp4":
            # Fused RMSNorm (+ optional residual add) + MXFP4 quant epilogue.
            assert N % 2 == 0, "fused mxfp4 quant requires an even N (two fp4 -> uint8)"
            res = torch.randn_like(x) if add_residual else None
            fn = lambda: fused_rms_mxfp4_quant(x, w, eps, res1=res)  # noqa: E731
            # Bytes moved: read x (+ residual), write packed fp4 (N/2 bytes/row)
            # + e8m0 block scales (cdiv(N,32) bytes/row) (+ residual passthrough).
            mxfp4_block = 32
            mem_read = (M * 1) * N * x.element_size() * (2 if add_residual else 1)
            mem_write = M * (N // 2) + M * triton.cdiv(N, mxfp4_block)
            if add_residual:
                mem_write += M * N * x.element_size()
            mem = mem_read + mem_write
            flops = 4 * M * N  # dominated by the norm; quant is elementwise
        else:
            fn = lambda: rms_norm(x, w, eps)  # noqa: E731
            # memory transfer
            mem_read = (M * 1) * N * x.element_size()  # x is (M,N) and g/weight is (N)
            mem_write = M * N * x.element_size()  # output
            mem = mem_read + mem_write
            flops = 4 * M * N

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        elif metric == "throughput":
            tflops = flops / ms * 1e-9  # TFLOP/s
            return tflops
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_rmsnorm.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None):
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
        choices=["time", "bandwidth", "throughput"],
        default="bandwidth",
        help="metric to plot",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "mxfp4"],
        default="none",
        help=(
            "Fuse a quantization epilogue onto RMSNorm. 'mxfp4' benchmarks "
            "fused_rms_mxfp4_quant (RMSNorm + optional residual add + MXFP4 quant)."
        ),
    )
    parser.add_argument(
        "--add-residual",
        action="store_true",
        default=False,
        help="With --quant mxfp4, fuse a residual add (res1) ahead of the norm.",
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
    args = parser.parse_args(args=args)
    return args


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
