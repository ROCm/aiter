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
from bench_rmsnorm import main as bench_rmsnorm_main
from bench_rope import main as bench_rope_main


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
    "rmsnorm": bench_rmsnorm_main,
    "rope": bench_rope_main,
}


ROPE_METRIC_NOTE = (
    "Note: RoPE reports only total flops (TFLOP), i.e. total floating-point operations, not throughput (FLOPS). "
    "Time and bandwidth are not available because short-running kernels cannot be measured accurately "
    "through triton.testing.do_bench; use rocprof for accurate runtime."
)


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
        help=(
            "Metric to report (throughput=TFLOPS, bandwidth=GB/s, time=ms). Default: throughput. "
            "RoPE reports total flops (TFLOP) in a separate column (see note in output)."
        ),
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


def parse_rmsnorm_bench_stdout(stdout: str) -> float:
    lines = [line for line in stdout.splitlines() if line.strip()]
    data_line = lines[-1]
    last_row_values = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", data_line)))
    if not last_row_values:
        raise ValueError(f"Unexpected RMSNorm bench output format: {data_line}")
    return round(last_row_values[-1], 4)


def parse_rope_bench_stdout(stdout: str) -> str:
    bench_result = None
    for line in stdout.splitlines():
        if "Total flops" in line:
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if nums:
                val = float(nums[0])
                bench_result = f"{val:.6e}"
                break
    if bench_result is None:
        raise ValueError(f"Unexpected RoPE bench output format: {stdout[:200]!r}")
    return bench_result


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


def rmsnorm_args(
    shape: dict[str, int | str], M: int, metric: str
) -> tuple[str, dict[str, int | str]]:
    N = shape["N"]
    args_str = f"--shape {M} {N} --metric {metric}"
    return args_str, shape


def rope_args(
    shape: dict[str, int | str], M: int, metric: str
) -> tuple[str, dict[str, int | str]]:
    num_heads = int(shape["num_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    two_inputs = str(shape["two_inputs"]).lower()
    positions = str(shape["positions"]).lower()
    style = str(shape["style"]).lower()
    Q = num_heads // num_kv_heads
    args_str = (
        f"-B 1 -S {M} -H {num_kv_heads} -Q {Q} -D {head_dim} "
        f"--rotate_style {style} --two_inputs {two_inputs} --pos {positions} -l thd"
    )
    return args_str, shape


def get_tp_shapes(
    shapes: list[dict[str, int | str]], kernel: str, TP: int
) -> list[dict[str, int | str]]:
    def transform(shape):
        s = shape.copy()
        if "moe" in kernel:
            s["Dim2"] = max(s["Dim2"] // TP, 1)
        elif "rope" == kernel:
            s["num_heads"] = max(s["num_heads"] // TP, 1)
            s["num_kv_heads"] = max(s["num_kv_heads"] // TP, 1)
        elif "gemm" in kernel and "moe" not in kernel:
            if s["TP_dim"] in ("N", "K", "B"):
                key = s["TP_dim"]
                s[key] = max(s[key] // TP, 1)
        return s

    if kernel == "rmsnorm":
        return shapes
    else:
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
                    elif kernel == "rmsnorm":
                        args_str, shape = rmsnorm_args(shape, M, metric)
                    elif kernel == "rope":
                        args_str, shape = rope_args(shape, M, metric)
                    elif "gemm" in kernel and "moe" not in kernel:
                        args_str, shape = gemm_args(shape, M, metric, layout)
                    else:
                        raise ValueError(f"Kernel {kernel} not supported")

                    stdout, _ = call_function(bench_fn, args_str)

                    if "moe" in kernel:
                        bench_result = parse_moe_bench_stdout(stdout, metric)
                        results.append(
                            {
                                "Model": model,
                                "Kernel": kernel,
                                "E": shape["E"],
                                "M/S": M,
                                "Dim1": shape["Dim1"],
                                "Dim2": shape["Dim2"],
                                "TopK": shape["TopK"],
                                metric: bench_result,
                            }
                        )
                    elif kernel == "rmsnorm":
                        bench_result = parse_rmsnorm_bench_stdout(stdout)
                        results.append(
                            {
                                "Model": model,
                                "Kernel": kernel,
                                "M/S": M,
                                "N": shape["N"],
                                metric: bench_result,
                            }
                        )
                    elif kernel == "rope":
                        bench_result = parse_rope_bench_stdout(stdout)
                        row = {
                            "Model": model,
                            "Kernel": kernel,
                            "M/S": M,
                            "Q": shape["num_heads"] // shape["num_kv_heads"],
                            "H": shape["num_kv_heads"],
                            "D": shape["head_dim"],
                            "style": shape["style"],
                            "rope_total_flops": bench_result,
                        }
                        results.append(row)
                    elif "gemm" in kernel and "moe" not in kernel:
                        bench_result = parse_gemm_bench_stdout(stdout)
                        results.append(
                            {
                                "Model": model,
                                "Kernel": kernel,
                                "B": shape["B"] if "B" in shape else None,
                                "M/S": M,
                                "N": shape["N"],
                                "K": shape["K"],
                                metric: bench_result,
                            }
                        )
                    else:
                        raise ValueError(f"Kernel {kernel} not supported")
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
    # Exclude metric and rope_total_flops from Int64 conversion
    cols = df.select_dtypes(include="number").columns.difference(
        [metric, "rope_total_flops"]
    )
    df[cols] = df[cols].astype("Int64")

    unit = {"time": "ms", "throughput": "tflops", "bandwidth": "GBps"}
    df[f"{metric}({unit[metric]})"] = df.pop(metric)
    if "rope_total_flops" in df.columns:
        df["total_flops(TFLOP)"] = df.pop("rope_total_flops")

    for model, idf in df.groupby("Model"):
        print(f"\n=== Model: {model} ===")
        for kernel, jdf in idf.groupby("Kernel"):
            print(f"\nKernel: {kernel}")
            print(
                jdf.drop(columns=["Model", "Kernel"])
                .dropna(axis=1)
                .to_string(index=False)
            )

    if (df["Kernel"] == "rope").any():
        print(f"\n{ROPE_METRIC_NOTE}")

    output_path = f"{os.path.dirname(os.path.realpath(__file__))}/{output_file}.csv"
    print(f"\nSaving results to {output_path}...\n")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
