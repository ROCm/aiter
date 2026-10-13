# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the Triton Conv3D implementations against PyTorch."""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.conv._prepack import (
    clear_conv3d_weight_pack_caches,
    prepack_ncdhw_to_cblocked,
)
from aiter.ops.triton.conv._utils import BLOCK_K, _out_dhw
from aiter.ops.triton.conv.conv3d import (
    _resolve_route,
    conv3d,
    conv3d_1x1x1,
    conv3d_general,
    conv3d_ncdhw_cblocked,
    conv3d_ndhwc_3x3x3,
    conv3d_winograd_hw_f4x3,
    conv3d_winograd_hw_f4x3_cblocked,
)
from op_tests.triton_tests.conv._helpers import (
    _winograd_tolerances,
    apply_activation,
    dynamic_conv_tolerances,
)

METHODS = (
    "auto",
    "general",
    "1x1x1",
    "cblocked",
    "ndhwc_3x3x3",
    "winograd",
    "winograd_cblocked",
)
_DEFAULT_MODEL = "wan22-a14b-vae"
_MODEL_SHAPES_PATH = Path(__file__).with_name("conv_shapes.json")


@dataclass(frozen=True)
class Conv3DCase:
    name: str
    dims: tuple[int, int, int, int, int, int, int, int, int]
    stride: tuple[int, int, int] = (1, 1, 1)
    padding: tuple[int, int, int] = (0, 0, 0)
    dilation: tuple[int, int, int] = (1, 1, 1)
    bias: bool = True
    encoder_calls: int = 0
    decoder_calls: int = 0


def _case(name, dims, *, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    return Conv3DCase(name, dims, stride, padding, dilation)


EDGE_CASES = (
    _case("1x1x1", (1, 64, 4, 16, 16, 128, 1, 1, 1)),
    _case("1x1x1 stride2", (1, 96, 6, 32, 32, 96, 1, 1, 1), stride=(2, 2, 2)),
    _case(
        "5x5x5 general",
        (1, 16, 8, 20, 20, 16, 5, 5, 5),
        padding=(2, 2, 2),
    ),
    _case(
        "3x3x3 stride2",
        (1, 64, 8, 24, 24, 64, 3, 3, 3),
        stride=(2, 2, 2),
        padding=(1, 1, 1),
    ),
    _case(
        "3x3x3 dilation2",
        (1, 16, 8, 24, 24, 16, 3, 3, 3),
        padding=(2, 2, 2),
        dilation=(2, 2, 2),
    ),
    replace(
        _case(
            "single input channel",
            (1, 1, 5, 15, 17, 16, 3, 3, 3),
            padding=(1, 1, 1),
        ),
        bias=False,
    ),
    _case(
        "asymmetric general",
        (2, 32, 7, 17, 23, 64, 3, 5, 3),
        stride=(1, 2, 1),
        padding=(1, 2, 1),
    ),
)


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _load_model_cases(pattern: str) -> tuple[str, list[Conv3DCase]]:
    with _MODEL_SHAPES_PATH.open() as file:
        data = json.load(file)
    matches = [
        model
        for model, operations in data.items()
        if pattern.lower() in model.lower() and "conv3d" in operations
    ]
    if not matches:
        available = sorted(model for model, ops in data.items() if "conv3d" in ops)
        raise ValueError(f"No Conv3D model matches {pattern!r}; available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Pattern {pattern!r} matched multiple models: {matches}")

    model = matches[0]
    cases = []
    for index, shape in enumerate(data[model]["conv3d"]):
        cases.append(
            Conv3DCase(
                name=shape.get("name", f"{model} L{index}"),
                dims=tuple(
                    shape[key] for key in ("N", "C", "D", "H", "W", "K", "T", "R", "S")
                ),
                stride=tuple(shape.get(f"stride_{axis}", 1) for axis in "dhw"),
                padding=tuple(shape.get(f"pad_{axis}", 0) for axis in "dhw"),
                dilation=tuple(shape.get(f"dilation_{axis}", 1) for axis in "dhw"),
                bias=shape.get("bias", True),
                encoder_calls=shape.get("encoder_calls", 0),
                decoder_calls=shape.get("decoder_calls", 0),
            )
        )
    return model, cases


def _load_cases(args) -> tuple[str, list[Conv3DCase]]:
    if args.shape:
        return "custom", [
            Conv3DCase(
                "custom",
                tuple(args.shape),
                tuple(args.stride),
                tuple(args.padding),
                tuple(args.dilation),
            )
        ]
    if args.smoke:
        return "smoke", list(EDGE_CASES)
    model, cases = _load_model_cases(args.model or _DEFAULT_MODEL)
    if args.batch_size is not None:
        cases = [
            replace(case, dims=(args.batch_size, *case.dims[1:])) for case in cases
        ]
    return model, cases


def _selected_kernel(case: Conv3DCase, method: str, layout: str) -> str:
    if method != "auto":
        return method
    N, C, D, H, W, K, T, R, S = case.dims
    return _resolve_route(
        T,
        R,
        S,
        case.stride,
        case.dilation,
        N,
        C,
        D,
        H,
        W,
        K,
        layout,
        padding=case.padding,
    ).value


def _run_triton(method, x, w, bias, case: Conv3DCase, layout: str, activation: str):
    kwargs = {
        "bias": bias,
        "stride": case.stride,
        "padding": case.padding,
        "dilation": case.dilation,
        "activation": activation,
    }
    if method == "auto":
        return conv3d(x, w, layout=layout, **kwargs)
    if method == "general":
        return conv3d_general(x, w, layout=layout, **kwargs)
    if method == "1x1x1":
        return conv3d_1x1x1(x, w, layout=layout, **kwargs)
    if method == "cblocked":
        return conv3d_ncdhw_cblocked(x, w, **kwargs)
    if method == "ndhwc_3x3x3":
        return conv3d_ndhwc_3x3x3(x, w, **kwargs)
    if method == "winograd":
        return conv3d_winograd_hw_f4x3(x, w, **kwargs)
    if method == "winograd_cblocked":
        return conv3d_winograd_hw_f4x3_cblocked(x, w, **kwargs)
    raise ValueError(f"unknown method: {method}")


def _torch_reference(x, w, bias, case: Conv3DCase, activation: str):
    result = F.conv3d(
        x,
        w,
        bias,
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
    )
    return apply_activation(result, activation)


def _make_inputs(case: Conv3DCase, dtype: torch.dtype, layout: str, use_bias: bool):
    N, C, D, H, W, K, T, R, S = case.dims
    x = torch.randn((N, C, D, H, W), device="cuda", dtype=dtype)
    if layout == "ndhwc":
        x = x.contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn((K, C, T, R, S), device="cuda", dtype=dtype)
    bias = torch.randn((K,), device="cuda", dtype=dtype) if use_bias else None
    return x, w, bias


def _assert_correct(
    output,
    reference,
    case: Conv3DCase,
    dtype: torch.dtype,
    kernel_name: str,
):
    _, C, _, _, _, _, T, R, S = case.dims
    reduction = C * T * R * S
    if "winograd" in kernel_name.lower() or "wino" in kernel_name.lower():
        rtol, atol = _winograd_tolerances(dtype, reduction)
    else:
        rtol, atol = dynamic_conv_tolerances(dtype, reduction)
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        rtol=rtol,
        atol=atol,
        msg=lambda msg: f"Conv3D benchmark mismatch for {case.name}\n\n{msg}",
    )


def _flops(case: Conv3DCase) -> float:
    N, C, D, H, W, K, T, R, S = case.dims
    OD, OH, OW = _out_dhw(D, H, W, T, R, S, case.stride, case.padding, case.dilation)
    return 2.0 * N * OD * OH * OW * K * C * T * R * S


def _effective_bytes(
    case: Conv3DCase,
    dtype: torch.dtype,
    use_bias: bool,
) -> int:
    N, C, D, H, W, K, T, R, S = case.dims
    OD, OH, OW = _out_dhw(D, H, W, T, R, S, case.stride, case.padding, case.dilation)
    elements = N * C * D * H * W + K * C * T * R * S + N * K * OD * OH * OW
    if use_bias:
        elements += K
    return elements * torch.empty((), dtype=dtype).element_size()


def _metric_value(metric: str, milliseconds: float, flops: float, moved: int):
    if metric == "time":
        return milliseconds
    if metric == "throughput":
        return flops / milliseconds * 1e-9
    return moved / milliseconds * 1e-6


def _shape_label(case: Conv3DCase) -> str:
    N, C, D, H, W, K, T, R, S = case.dims
    return f"({N},{C},{D},{H},{W})→{K}/{T}x{R}x{S}"


def _kernel_type(case: Conv3DCase) -> str:
    _, _, _, _, _, _, T, R, S = case.dims
    if (T, R, S) == (1, 1, 1) and case.dilation == (1, 1, 1):
        return "[1x1x1]"
    if (T, R, S) == (3, 3, 3):
        return "[3x3x3]"
    return "[general]"


def _box_table(headers, rows, align=None) -> str:
    column_count = len(headers)
    if align is None:
        align = ["l"] * column_count
    widths = [
        max(
            len(headers[column]),
            max((len(str(row[column])) for row in rows), default=0),
        )
        for column in range(column_count)
    ]

    def format_row(values):
        cells = []
        for column, value in enumerate(values):
            value = str(value)
            if align[column] == "r":
                cells.append(f" {value:>{widths[column]}} ")
            else:
                cells.append(f" {value:<{widths[column]}} ")
        return "│" + "│".join(cells) + "│"

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    middle = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"
    lines = [top, format_row(headers), middle]
    for index, row in enumerate(rows):
        lines.append(format_row(row))
        if index < len(rows) - 1:
            lines.append(middle)
    lines.append(bottom)
    return "\n".join(lines)


def _metric_labels(metric: str) -> tuple[str, str]:
    if metric == "throughput":
        return "TF/s", "TFLOPS"
    if metric == "bandwidth":
        return "GB/s", "GB/s"
    return "ms", "latency [ms]"


def _is_better(metric: str, triton_value: float, torch_value: float) -> bool:
    if metric == "time":
        return triton_value < torch_value
    return triton_value > torch_value


def _print_layer_table(layers, has_any_repack, metric):
    short_unit, _ = _metric_labels(metric)
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER BENCHMARK")
    print("=" * 80)

    headers = ["#", "Layer", "Type", "Shape", "Triton Kernel"]
    headers.append(f"Tri Kernel {short_unit}")
    if has_any_repack:
        headers.append(f"Tri Kernel+Repack {short_unit}")
    headers.extend([f"Torch {short_unit}", "Winner"])

    rows = []
    for index, layer in enumerate(layers):
        row = [
            str(index),
            layer["name"],
            layer["type"],
            layer["shape"],
            layer["kernel_name"],
            f"{layer['values']['triton_kernel']:.2f}",
        ]
        if has_any_repack:
            e2e_value = layer["values"].get("triton_e2e")
            row.append(f"{e2e_value:.2f}" if e2e_value is not None else "—")
        torch_value = layer["values"]["torch"]
        row.append(f"{torch_value:.2f}")
        row.append(
            "Triton"
            if _is_better(metric, layer["values"]["triton_kernel"], torch_value)
            else "Torch"
        )
        rows.append(row)
    print(_box_table(headers, rows))


def _aggregate_value(layers, provider, metric):
    timing_key = {
        "triton_kernel": "ms_tri",
        "triton_e2e": "ms_tri_e2e",
        "torch": "ms_torch",
    }[provider]
    total_ms = 0.0
    total_work = 0.0
    for layer in layers:
        milliseconds = layer[timing_key]
        if milliseconds is None:
            milliseconds = layer["ms_tri"]
        total_ms += milliseconds
        if metric == "throughput":
            total_work += layer["flops"]
        elif metric == "bandwidth":
            total_work += layer["bytes"].get(provider, layer["bytes"]["triton_kernel"])
    if metric == "throughput":
        return total_work / total_ms * 1e-9
    if metric == "bandwidth":
        return total_work / total_ms * 1e-6
    return total_ms


def _print_overall_perf_table(layers, has_any_repack, metric):
    _, long_unit = _metric_labels(metric)
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    tri_values = [layer["values"]["triton_kernel"] for layer in layers]
    torch_values = [layer["values"]["torch"] for layer in layers]
    e2e_values = [
        layer["values"].get("triton_e2e", layer["values"]["triton_kernel"])
        for layer in layers
    ]
    count = len(layers)
    tri_wins = sum(
        _is_better(metric, tri, torch) for tri, torch in zip(tri_values, torch_values)
    )
    e2e_wins = sum(
        _is_better(metric, e2e, torch) for e2e, torch in zip(e2e_values, torch_values)
    )

    rows = [
        [
            f"Mean {long_unit} (kernel)",
            f"{statistics.mean(tri_values):.2f}",
            f"{statistics.mean(torch_values):.2f}",
        ]
    ]
    if has_any_repack:
        rows.append(
            [
                f"Mean {long_unit} (kernel+repack)",
                f"{statistics.mean(e2e_values):.2f}",
                f"{statistics.mean(torch_values):.2f}",
            ]
        )
    rows.append(
        [
            f"Median {long_unit} (kernel)",
            f"{statistics.median(tri_values):.2f}",
            f"{statistics.median(torch_values):.2f}",
        ]
    )
    if has_any_repack:
        rows.append(
            [
                f"Median {long_unit} (kernel+repack)",
                f"{statistics.median(e2e_values):.2f}",
                f"{statistics.median(torch_values):.2f}",
            ]
        )
    if metric != "time":
        rows.append(
            [
                f"Aggregate {long_unit} (kernel)",
                f"{_aggregate_value(layers, 'triton_kernel', metric):.2f}",
                f"{_aggregate_value(layers, 'torch', metric):.2f}",
            ]
        )
        if has_any_repack:
            rows.append(
                [
                    f"Aggregate {long_unit} (kernel+repack)",
                    f"{_aggregate_value(layers, 'triton_e2e', metric):.2f}",
                    f"{_aggregate_value(layers, 'torch', metric):.2f}",
                ]
            )
    rows.append(
        [
            "Total kernel time (ms)",
            f"{sum(layer['ms_tri'] for layer in layers):.2f}",
            f"{sum(layer['ms_torch'] for layer in layers):.2f}",
        ]
    )
    if has_any_repack:
        rows.append(
            [
                "Total kernel+repack time (ms)",
                f"{_aggregate_value(layers, 'triton_e2e', 'time'):.2f}",
                f"{sum(layer['ms_torch'] for layer in layers):.2f}",
            ]
        )
    rows.append(
        ["Layer wins (kernel)", f"{tri_wins}/{count}", f"{count - tri_wins}/{count}"]
    )
    if has_any_repack:
        rows.append(
            [
                "Layer wins (kernel+repack)",
                f"{e2e_wins}/{count}",
                f"{count - e2e_wins}/{count}",
            ]
        )
    rows.append(["Correctness", f"{count}/{count} passed", "—"])
    print(_box_table(("Metric", "Triton", "PyTorch (MIOpen)"), rows))


def _print_weighted_totals(cases, timings, include_e2e):
    rows = []
    for workload, field in (("Encoder", "encoder_calls"), ("Decoder", "decoder_calls")):
        calls = sum(getattr(case, field) for case in cases)
        if not calls:
            continue
        kernel_ms = sum(
            getattr(case, field) * timings[(index, "triton_kernel")]
            for index, case in enumerate(cases)
        )
        torch_ms = sum(
            getattr(case, field) * timings[(index, "torch")]
            for index, case in enumerate(cases)
        )
        if include_e2e:
            e2e_ms = sum(
                getattr(case, field)
                * timings.get((index, "triton_e2e"), timings[(index, "triton_kernel")])
                for index, case in enumerate(cases)
            )
        else:
            e2e_ms = kernel_ms
        rows.append((workload, calls, kernel_ms / 1e3, e2e_ms / 1e3, torch_ms / 1e3))
    if not rows:
        return

    print("\nCall-count-weighted model totals:")
    print("workload calls triton_kernel_s triton_e2e_s torch_s speedup")
    for workload, calls, kernel_s, e2e_s, torch_s in rows:
        speedup = torch_s / e2e_s if e2e_s else float("nan")
        print(
            f"{workload} {calls} {kernel_s:.4f} {e2e_s:.4f} "
            f"{torch_s:.4f} {speedup:.2f}x"
        )


def run_benchmark(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Conv3D benchmarking requires a CUDA/ROCm GPU")

    source, cases = _load_cases(args)
    dtype = _torch_dtype(args.dtype)
    kernel_names = [_selected_kernel(case, args.method, args.layout) for case in cases]
    packs_input = ["cblocked" in name.lower() for name in kernel_names]
    include_e2e = any(packs_input)
    providers = ["triton_kernel"]
    if include_e2e:
        providers.append("triton_e2e")
    providers.append("torch")

    unit = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}[args.metric]
    timings: dict[tuple[int, str], float] = {}
    values: dict[tuple[int, str], float] = {}
    moved_bytes: dict[tuple[int, str], int] = {}
    validated: set[int] = set()
    result_rows = []

    def benchmark_case(case_id, provider):
        case = cases[case_id]
        kernel_name = kernel_names[case_id]
        needs_pack = packs_input[case_id]
        if provider == "triton_e2e" and not needs_pack:
            return None

        torch.manual_seed(case_id)
        use_bias = case.bias and not args.no_bias
        x, w, bias = _make_inputs(case, dtype, args.layout, use_bias)

        def run_triton():
            return _run_triton(
                args.method, x, w, bias, case, args.layout, args.activation
            )

        def run_torch():
            return _torch_reference(x, w, bias, case, args.activation)

        try:
            if case_id not in validated:
                output = run_triton()
                reference = run_torch()
                torch.cuda.synchronize()
                _assert_correct(output, reference, case, dtype, kernel_name)
                validated.add(case_id)
                del output, reference
                torch.cuda.empty_cache()

            if provider == "torch":
                function = run_torch
            elif provider == "triton_e2e" or not needs_pack:
                function = run_triton
            else:
                x_blocked, _ = prepack_ncdhw_to_cblocked(x, BLOCK_K)
                cblocked_fn = (
                    conv3d_winograd_hw_f4x3_cblocked
                    if "winograd" in kernel_name.lower()
                    else conv3d_ncdhw_cblocked
                )

                def function():
                    return cblocked_fn(
                        x,
                        w,
                        bias,
                        case.stride,
                        case.padding,
                        case.dilation,
                        activation=args.activation,
                        x_blocked=x_blocked,
                    )

            milliseconds = triton.testing.do_bench(function, warmup=15, rep=50)
            timings[(case_id, provider)] = milliseconds
            moved = _effective_bytes(case, dtype, use_bias)
            moved_bytes[(case_id, provider)] = moved
            return _metric_value(args.metric, milliseconds, _flops(case), moved)
        finally:
            clear_conv3d_weight_pack_caches()
            torch.cuda.empty_cache()

    print(
        f"# source={source} shapes={len(cases)} dtype={args.dtype} "
        f"layout={args.layout} method={args.method} metric={args.metric}"
    )
    for case_id, case in enumerate(cases):
        for provider in providers:
            value = benchmark_case(case_id, provider)
            if value is None:
                continue
            values[(case_id, provider)] = value
            row = {
                "case": case_id,
                "provider": provider,
                "value": value,
                "unit": unit,
                "kernel": kernel_names[case_id],
                "shape": _shape_label(case),
            }
            result_rows.append(row)

    layers = []
    for case_id, case in enumerate(cases):
        layer_values = {
            provider: values[(case_id, provider)]
            for provider in providers
            if (case_id, provider) in values
        }
        layer_bytes = {
            provider: moved_bytes[(case_id, provider)]
            for provider in providers
            if (case_id, provider) in moved_bytes
        }
        layers.append(
            {
                "name": case.name,
                "type": _kernel_type(case),
                "shape": _shape_label(case),
                "kernel_name": kernel_names[case_id],
                "values": layer_values,
                "bytes": layer_bytes,
                "ms_tri": timings[(case_id, "triton_kernel")],
                "ms_tri_e2e": timings.get((case_id, "triton_e2e")),
                "ms_torch": timings[(case_id, "torch")],
                "flops": _flops(case),
            }
        )

    _print_layer_table(layers, include_e2e, args.metric)
    _print_overall_perf_table(layers, include_e2e, args.metric)
    _print_weighted_totals(cases, timings, include_e2e)
    if args.output:
        safe_source = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in source
        )
        output_path = Path(
            f"conv3d_{safe_source}_{args.dtype}_{args.layout}_{args.method}.csv"
        )
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result_rows[0])
            writer.writeheader()
            writer.writerows(result_rows)
        print(f"Saved {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_conv3d",
        description="Benchmark Triton Conv3D against PyTorch.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--shape",
        type=int,
        nargs=9,
        metavar=("N", "C", "D", "H", "W", "K", "T", "R", "S"),
        help="benchmark one NCDHW/OIDHW shape",
    )
    source.add_argument(
        "--model",
        help="benchmark Conv3D shapes matching this model name",
    )
    source.add_argument(
        "--smoke",
        action="store_true",
        help="benchmark a compact edge-case sweep",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--layout", choices=("ncdhw", "ndhwc"), default="ncdhw")
    parser.add_argument("--method", choices=METHODS, default="auto")
    parser.add_argument(
        "--metric",
        choices=("time", "throughput", "bandwidth"),
        default="throughput",
    )
    parser.add_argument(
        "--activation",
        choices=("none", "relu", "relu6", "gelu"),
        default="none",
    )
    parser.add_argument("--stride", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--padding", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--dilation", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--batch-size", type=int, help="override N in a sweep")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="save CSV output in the current directory",
    )
    args = parser.parse_args(argv)

    if args.shape and any(value <= 0 for value in args.shape):
        parser.error("all shape dimensions must be positive")
    if any(value <= 0 for value in args.stride):
        parser.error("stride values must be positive")
    if any(value < 0 for value in args.padding):
        parser.error("padding values must be non-negative")
    if any(value <= 0 for value in args.dilation):
        parser.error("dilation values must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.shape and args.batch_size is not None:
        parser.error("--batch-size cannot be combined with --shape")
    if args.layout == "ndhwc" and args.method in {
        "cblocked",
        "winograd",
        "winograd_cblocked",
    }:
        parser.error(f"--method {args.method} requires --layout ncdhw")
    if args.layout == "ncdhw" and args.method == "ndhwc_3x3x3":
        parser.error("--method ndhwc_3x3x3 requires --layout ndhwc")
    return args


def main(argv: list[str] | None = None) -> None:
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
