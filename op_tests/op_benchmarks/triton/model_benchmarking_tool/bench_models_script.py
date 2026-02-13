from contextlib import redirect_stdout, redirect_stderr
from typing import Callable
import io
import logging
import shlex
import os
import pandas as pd
import json
import re
import aiter.ops.triton.utils._triton.arch_info as arch_info
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bench_gemm_a16w16 import (
    main as bench_gemm_a16w16_main,
)
from bench_gemm_a8w8_per_token_scale import (
    main as bench_gemm_a8w8_per_token_scale_main,
)
from bench_gemm_a8w8_blockscale import (
    main as bench_gemm_a8w8_blockscale_main,
)
from bench_gemm_afp4wfp4 import (
    main as bench_gemm_afp4wfp4_main,
)
from bench_batched_gemm_a8w8 import main as bench_batched_gemm_a8w8_main
from bench_batched_gemm_afp4wfp4 import main as bench_batched_gemm_afp4wfp4_main
from bench_moe_gemm_a8w8 import main as bench_moe_gemm_a8w8_main
from bench_moe_gemm_a8w8_blockscale import main as bench_moe_gemm_a8w8_blockscale_main
from bench_moe_gemm_a8w4 import main as bench_moe_gemm_a8w4_main
from bench_moe_gemm_a4w4 import main as bench_moe_gemm_a4w4_main


def disable_aiter_logs() -> None:
    logging.getLogger("aiter").disabled = True


disable_aiter_logs()


kernel_dict = {
    "gemm_a16w16": bench_gemm_a16w16_main,
    "gemm_a8w8_per_token_scale": bench_gemm_a8w8_per_token_scale_main,
    "gemm_a8w8_blockscale": bench_gemm_a8w8_blockscale_main,
    "gemm_afp4wfp4": bench_gemm_afp4wfp4_main,
    "batched_gemm_a8w8": bench_batched_gemm_a8w8_main,
    "batched_gemm_afp4wfp4": bench_batched_gemm_afp4wfp4_main,
    "moe_op_gemm_a8w8": bench_moe_gemm_a8w8_main,
    "moe_op_gemm_a8w8_blockscale": bench_moe_gemm_a8w8_blockscale_main,
    "moe_op_gemm_a8w4": bench_moe_gemm_a8w4_main,
    "moe_op_gemm_a4w4": bench_moe_gemm_a4w4_main,
}


def parse_M_values(raw_values: list[str], parser: argparse.ArgumentParser) -> list[int]:
    result = []
    for value in raw_values:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 3:
                parser.error(
                    f"Invalid range '{value}'. " "Ranges must be start:stop:step."
                )

            try:
                start, stop, step = map(int, parts)
            except ValueError:
                parser.error(f"Invalid integers in range '{value}'.")

            if start <= 0 or step <= 0:
                parser.error(f"Values must be positive in range '{value}'.")

            if start > stop:
                parser.error(f"Start must be <= stop in range '{value}'.")

            result.extend(range(start, stop + 1, step))

        else:
            try:
                val = int(value)
            except ValueError:
                parser.error(f"Invalid integer value '{value}'.")

            if val <= 0:
                parser.error("Input size M must be positive.")

            result.append(val)

    # Remove duplicates and sort
    return sorted(set(result))


def parse_args(available_models: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model benchmarking tool",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--M",
        type=str,
        nargs="+",
        default=["512"],
        help=(
            "Input size M. Accepts:\n"
            "  Single value:            --M 512\n"
            "  Multiple values:         --M 256 512 1024\n"
            "  Range start:stop:step:   --M 128:1024:128\n"
            "  Combinations of values and ranges are also accepted.\n"
            "Default: 512."
        ),
    )
    parser.add_argument(
        "--TP",
        type=int,
        choices=[1, 2, 4, 8],
        default=8,
        help="Tensor parallel size. Default: 8.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["throughput", "bandwidth", "time"],
        default="throughput",
        help="Metric to report (throughput=TFLOPS, bandwidth=GB/s, time=ms). Default: throughput.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["all"] + available_models,
        default=["all"],
        help=(
            "Model(s) to benchmark. "
            "Use 'all' to benchmark all available models. Default: all."
        ),
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["TN", "TT", "NN", "NT"],
        default="TN",
        help="GEMM layout. Default: TN.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bench_results",
        help="Name for the CSV output file. Default: bench_results.",
    )

    args = parser.parse_args()

    args.M = parse_M_values(args.M, parser)

    if "all" in args.models and len(args.models) > 1:
        parser.error("'all' cannot be combined with other models.")

    return args


def read_json(json_path: str) -> dict:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(script_dir + json_path, "r") as f:
        data = json.load(f)

    return data


def call_function(
    bench_fn: Callable[[list[str] | None], None], args_str: str
) -> tuple[str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            bench_fn(shlex.split(args_str))
        except SystemExit as e:
            print(f"SystemExit caught: {e}", file=stderr)

    # Close matplotlib figures to silence errors and avoid memory leaks.
    plt.close("all")
    return stdout.getvalue(), stderr.getvalue()


def parse_gemm_bench_stdout(stdout: str) -> float:
    # Get last non-empty line (the data row)
    lines = [line for line in stdout.splitlines() if line.strip()]
    data_line = lines[-1]

    last_row_values = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", data_line)))

    if len(last_row_values) not in (5, 6):
        raise ValueError(f"Unexpected GEMM bench output format: {last_row_values}")

    bench_result = last_row_values[-1]
    return round(bench_result, 4)


def parse_moe_bench_stdout(stdout: str, metric: str) -> float:
    # Get last non-empty line (the data row)
    lines = [line for line in stdout.splitlines() if line.strip()]
    data_line = lines[-1]

    data = {}
    for result in data_line.split("|"):
        key, value = result.split(":")
        data[key.strip()] = float(value.strip())

    if metric == "time":
        bench_result = data["Kernel latency (us)"] * 1e-3  # Convert from us to ms
    elif metric == "throughput":
        bench_result = data["TFLOPS"]
    elif metric == "bandwidth":
        bench_result = data["TBPS"] * 1e3  # Convert from TBps to GBps
    return round(bench_result, 4)


def gemm_args(
    shape: dict[str, int | str], M: int, metric: str, layout: str
) -> tuple[str, dict[str, int | str]]:

    N = shape["N"]
    K = shape["K"]
    if "B" in shape:
        B = shape["B"]
        args_str = f"--shape {B} {M} {N} {K} --metric {metric} --layout {layout}"
    else:
        args_str = f"--shape {M} {N} {K} --metric {metric} --layout {layout}"
    return args_str, shape


def moe_args(
    shape: dict[str, int | str], M: int, metric: str
) -> tuple[str, dict[str, int | str]]:

    E = shape["E"]
    dim1 = shape["Dim1"]
    dim2 = shape["Dim2"]
    topk = shape["TopK"]
    args_str = f"--M {M} --shape {dim1} {dim2} --experts {E} {topk}"
    return args_str, shape


def get_tp_shapes(
    shapes: list[dict[str, int | str]], kernel: str, TP: int
) -> list[dict[str, int | str]]:
    def transform(shape):
        s = shape.copy()
        if "moe" in kernel:
            s["Dim2"] //= TP
        else:
            if s["TP_dim"] in ("N", "K", "B"):
                s[s["TP_dim"]] //= TP
        return s

    return [transform(shape) for shape in shapes]


def run_benchmarks(
    data: dict[str, dict[str, dict[str, int | str]]],
    M_values: list[int],
    TP: int,
    layout: str,
    metric: str,
) -> list[dict[str, int | float | str]]:

    results = []
    for model, kernels in data.items():
        print(f"Running benchmarks for {model}...")
        for kernel, shapes in kernels.items():

            if "fp4" in kernel and not arch_info.is_fp4_avail():
                continue

            bench_fn = kernel_dict[kernel]

            tp_shapes = get_tp_shapes(shapes, kernel, TP)
            for shape in tp_shapes:
                for M in M_values:
                    if "moe" in kernel:
                        args_str, shape = moe_args(shape, M, metric)
                    else:
                        args_str, shape = gemm_args(shape, M, metric, layout)

                    stdout, stderr = call_function(bench_fn, args_str)

                    if "moe" in kernel:
                        bench_result = parse_moe_bench_stdout(stdout, metric)
                        results.append(
                            {
                                "Model": model,
                                "Kernel": kernel,
                                "E": shape["E"],
                                "M": M,
                                "Dim1": shape["Dim1"],
                                "Dim2": shape["Dim2"],
                                "TopK": shape["TopK"],
                                metric: bench_result,
                            }
                        )
                    else:
                        bench_result = parse_gemm_bench_stdout(stdout)
                        results.append(
                            {
                                "Model": model,
                                "Kernel": kernel,
                                "B": shape["B"] if "B" in shape else None,
                                "M": M,
                                "N": shape["N"],
                                "K": shape["K"],
                                metric: bench_result,
                            }
                        )
    return results


def main() -> None:
    data = read_json("/model_shapes.json")
    available_models = list(data.keys())
    args = parse_args(available_models)

    models = args.models
    M_values = args.M
    TP = args.TP
    metric = args.metric
    layout = args.layout
    output_file = args.output_file

    if "all" not in models:
        data = {m: data[m] for m in models}

    results = run_benchmarks(data, M_values, TP, layout, metric)

    # Prints results grouped by model and kernel and saves them to disk
    metric = args.metric
    df = pd.DataFrame(results)
    cols = df.select_dtypes(include="number").columns.difference([metric])
    df[cols] = df[cols].astype("Int64")

    unit = {"time": "ms", "throughput": "tflops", "bandwidth": "GBps"}
    df[f"{metric}({unit[metric]})"] = df.pop(metric)

    for model, idf in df.groupby("Model"):
        print(f"\n=== Model: {model} ===")
        for kernel, jdf in idf.groupby("Kernel"):
            print(f"\nKernel: {kernel}")
            print(
                jdf.drop(columns=["Model", "Kernel"])
                .dropna(axis=1)
                .to_string(index=False)
            )

    output_path = f"{os.path.dirname(os.path.realpath(__file__))}/{output_file}.csv"
    print(f"\nSaving results to {output_path}...\n")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
