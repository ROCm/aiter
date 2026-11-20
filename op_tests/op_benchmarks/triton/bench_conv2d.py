# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark Triton Conv2D with rotating buffers and graph replay.

Channel-blocked routes report input packing excluded and included by default.
Nonzero padding modes and --end-to-end time the full GPU call.
Weight preparation is excluded. Correctness checks belong in unit tests.
Single-shape mode emits key=value fields and a final primary metric.
"""

import argparse
import functools
import json
import os
import statistics

import torch

from aiter import logger
from aiter.ops.triton.conv._prepack import (
    clear_conv2d_weight_pack_caches,
    prepack_nchw_to_cblocked,
)
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _conv2d_padding_geometry,
    _is_1x1_conv,
    _is_3x3_conv,
    _out_hw,
    _validate_conv2d_padding_mode,
)
from aiter.ops.triton.conv.conv2d import (
    _resolve_route,
    conv2d,
    conv2d_nchw,
    conv2d_nchw_cblocked,
    conv2d_nhwc,
    conv2d_winograd_f4x3,
    conv2d_winograd_f4x3_cblocked,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import BASE_DIR
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.conv._helpers import (
    generate_conv_input,
    generate_conv_weights,
)


def flops_conv(N, C, K_out, R, S, P, Q):
    return 2.0 * N * P * Q * K_out * C * R * S


def which_kernel(
    x,
    w_oihw,
    stride=(1, 1),
    dilation=(1, 1),
    layout="nchw",
    padding=(0, 0),
    padding_mode="zeros",
):
    """Predict the routed kernel without launching it.

    Use shared padding geometry and dispatch rules for labels.
    """
    N, C, H, W_in = x.shape
    K_out, _, R, S = w_oihw.shape
    _validate_conv2d_padding_mode(padding_mode, H, W_in, padding)
    H, W_in, padding = _conv2d_padding_geometry(H, W_in, padding, padding_mode)
    route = _resolve_route(
        R,
        S,
        stride,
        dilation,
        N,
        C,
        H,
        W_in,
        K_out,
        layout.lower(),
        padding=padding,
    )
    return route.value


METHODS = {
    "auto": conv2d,
    "default": conv2d_nchw,
    "cblocked": conv2d_nchw_cblocked,
    "nhwc": conv2d_nhwc,
    "winograd_f4x3": conv2d_winograd_f4x3,
    "winograd_f4x3_cblocked": conv2d_winograd_f4x3_cblocked,
}


# Edge-case smoke set — same shapes as the unit-test edge cases (see
# _helpers.get_edge_case_shapes). These exercise degenerate paths (C=1,
# dilation>1, asymmetric dims, stride>1, etc.) that real production
# models don't hit. Used only when --smoke is passed; otherwise the sweep
# defaults to a real model from conv_shapes.json.
EDGE_CASE_SHAPES = [
    # (N, C, H, W, K, R, S, stride, padding, dilation, desc)
    (1, 3, 7, 7, 8, 3, 3, (1, 1), (1, 1), (1, 1), "3x3 same padding"),
    (1, 3, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 stride1"),
    (2, 16, 32, 32, 32, 3, 3, (2, 2), (1, 1), (1, 1), "stride2"),
    (2, 32, 17, 23, 64, 5, 5, (2, 2), (2, 2), (1, 1), "odd dims + pad"),
    (4, 64, 28, 28, 128, 3, 3, (1, 1), (0, 0), (2, 2), "dilation2"),
    (2, 512, 7, 7, 1024, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 large channels"),
    (1, 3, 112, 112, 64, 7, 7, (2, 2), (3, 3), (1, 1), "7x7 large spatial"),
    (1, 1, 16, 16, 16, 3, 3, (1, 1), (1, 1), (1, 1), "single input channel"),
    (2, 64, 8, 8, 64, 3, 3, (1, 1), (1, 1), (1, 1), "small spatial 3x3"),
    (1, 128, 4, 4, 256, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 tiny spatial"),
    (2, 32, 32, 32, 32, 3, 3, (1, 1), (0, 0), (1, 1), "3x3 no padding"),
    (2, 64, 28, 28, 128, 3, 3, (2, 2), (1, 1), (1, 1), "3x3 stride2 standard"),
]


def _torch_dtype(s: str) -> torch.dtype:
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {s}")


def _kernel_type_tag(R: int, S: int, dilation: tuple) -> str:
    if _is_1x1_conv(R, S, dilation):
        return "[1x1]"
    if _is_3x3_conv(R, S):
        return "[3x3]"
    return "[general]"


def _shape_str(N, C, H, W, K, R, S) -> str:
    return f"({N},{C},{H},{W})→{K}/{R}x{S}"


def bench_one_shape(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    dtype: torch.dtype,
    method: str,
    layout: str,
    bias: bool = True,
    measure_repack: bool = True,
    padding_mode: str = "zeros",
    end_to_end: bool = False,
    bencher: RotatingBufferBench | None = None,
) -> dict:
    """Time one shape with fixed weights and rotating activations.

    Channel-blocked routes report input packing excluded and included by default.
    Nonzero padding modes or end_to_end time padding, layout, and input packing.
    All measurements exclude weight preparation and CPU dispatch.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; conv2d bench requires a GPU.")
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}; choices: {list(METHODS)}")
    _validate_conv2d_padding_mode(padding_mode, H, W, padding)
    end_to_end = end_to_end or padding_mode != "zeros"
    output_height, output_width = _out_hw(H, W, R, S, stride, padding, dilation)
    if output_height < 1 or output_width < 1:
        raise ValueError("output spatial dimensions must be positive")

    effective_layout = "nhwc" if layout == "nhwc" or method == "nhwc" else "nchw"
    kernel_fn = conv2d_nhwc if effective_layout == "nhwc" else METHODS[method]
    if method in ("auto", "default", "nhwc") or effective_layout == "nhwc":
        routed_height, routed_width, routed_padding = _conv2d_padding_geometry(
            H, W, padding, padding_mode
        )
        kernel_name = _resolve_route(
            R,
            S,
            stride,
            dilation,
            N,
            C,
            routed_height,
            routed_width,
            K,
            effective_layout,
            padding=routed_padding,
        ).value
    else:
        kernel_name = method
    packs_input = "cblocked" in kernel_name.lower()
    has_repack = not end_to_end and measure_repack and packs_input
    if bencher is None:
        bencher = RotatingBufferBench()

    input_generator = functools.partial(
        generate_conv_input, (N, C, H, W), dtype, effective_layout
    )
    weight_generator = functools.partial(
        generate_conv_weights, (K, C, R, S), dtype, bias
    )
    timings = {}
    for include_packing in ([False, True] if has_repack else [False]):
        prepacked = packs_input and not (include_packing or end_to_end)
        selected_fn = kernel_fn
        if prepacked:
            selected_fn = (
                conv2d_winograd_f4x3_cblocked
                if "winograd" in kernel_name.lower()
                else conv2d_nchw_cblocked
            )

        def invoke(
            input_tensor,
            weight,
            bias,
            x_blocked=None,
            prepacked=prepacked,
            selected_fn=selected_fn,
        ):
            kwargs = {"x_blocked": x_blocked} if prepacked else {}
            return selected_fn(
                input_tensor,
                weight,
                bias,
                stride,
                padding,
                dilation,
                activation="none",
                padding_mode=padding_mode,
                **kwargs,
            )

        fixed_generators = [weight_generator]
        fixed_keys = [("weight", "bias")]
        rotating_generator = input_generator
        rotating_keys = ("input_tensor",)
        if prepacked:
            fixed_generators.append(input_generator)
            fixed_keys.append(("input_tensor",))

            def rotating_generator():
                (input_tensor,) = input_generator()
                packed_input, _ = prepack_nchw_to_cblocked(input_tensor, BLOCK_K)
                return (packed_input,)

            rotating_keys = ("x_blocked",)

        try:
            bencher.build_pool(
                [rotating_generator],
                [rotating_keys],
                inp_gen_fixed_fns=fixed_generators,
                inp_fixed_keys=fixed_keys,
            )
            timings[include_packing] = bencher.bench(invoke)
        finally:
            bencher.rot_bufs = None
            clear_conv2d_weight_pack_caches()
            torch.cuda.empty_cache()

    milliseconds = timings[False]
    milliseconds_with_packing = timings.get(True)
    flops = flops_conv(N, C, K, R, S, output_height, output_width)
    return {
        "ms_tri": milliseconds,
        "ms_tri_e2e": milliseconds_with_packing,
        "tflops_tri": flops / milliseconds * 1e-9,
        "tflops_tri_e2e": (
            flops / milliseconds_with_packing * 1e-9
            if milliseconds_with_packing is not None
            else None
        ),
        "kernel_name": kernel_name,
        "has_repack": has_repack,
        "flops": flops,
        "weight_policy": "fixed",
        "padding_mode": padding_mode,
        "timing_scope": "end_to_end" if end_to_end else "kernel",
    }


def _format_single_shape_line(args, result: dict) -> str:
    """Format key=value fields followed by the primary metric."""
    primary = result["ms_tri"] if args.metric == "time" else result["tflops_tri"]
    parts = [
        f"N={args.N}",
        f"C={args.C}",
        f"H={args.H}",
        f"W={args.W}",
        f"K={args.K}",
        f"R={args.R}",
        f"S={args.S}",
        f"method={args.method}",
        f"layout={args.layout}",
        f"padding_mode={result['padding_mode']}",
        f"timing_scope={result['timing_scope']}",
        f"ms_tri={result['ms_tri']:.4f}",
        f"tflops_tri={result['tflops_tri']:.4f}",
    ]
    parts.append("weight_policy=fixed")
    if result["has_repack"]:
        parts.extend(
            [
                f"ms_tri_e2e={result['ms_tri_e2e']:.4f}",
                f"tflops_tri_e2e={result['tflops_tri_e2e']:.4f}",
            ]
        )
    if args.show_kernel_name:
        parts.append(f"kernel={result['kernel_name'] or 'unknown'}")
    parts.append(f"{primary:.4f}")
    return " ".join(parts)


def run_single_shape(args) -> None:
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    result = bench_one_shape(
        args.N,
        args.C,
        args.H,
        args.W,
        args.K,
        args.R,
        args.S,
        (args.stride_h, args.stride_w),
        (args.pad_h, args.pad_w),
        (args.dilation_h, args.dilation_w),
        _torch_dtype(args.dtype),
        args.method,
        args.layout,
        bias=not args.no_bias,
        padding_mode=args.padding_mode,
        end_to_end=args.end_to_end,
        bencher=bencher,
    )
    logger.info("%s", _format_single_shape_line(args, result))


# ----------------------------------------------------------------------------
# Box-drawn table printers
# ----------------------------------------------------------------------------


def _box_table(headers, rows, align: list | None = None) -> str:
    """Render a list of header-string + row-tuples into a box-drawn table.

    align: per-column alignment, "l" (left, default) or "r" (right).
    """
    n = len(headers)
    if align is None:
        align = ["l"] * n
    widths = [
        max(len(headers[j]), max((len(str(r[j])) for r in rows), default=0))
        for j in range(n)
    ]

    def fmt_row(vals):
        cells = []
        for j, v in enumerate(vals):
            s = str(v)
            if align[j] == "r":
                cells.append(f" {s:>{widths[j]}} ")
            else:
                cells.append(f" {s:<{widths[j]}} ")
        return "│" + "│".join(cells) + "│"

    sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    lines = [sep_top, fmt_row(headers), sep_mid]
    for i, row in enumerate(rows):
        lines.append(fmt_row(row))
        if i < len(rows) - 1:
            lines.append(sep_mid)
    lines.append(sep_bot)
    return "\n".join(lines)


def _print_layer_table(layers: list, has_any_repack: bool) -> None:
    logger.info("\nLAYER-BY-LAYER BENCHMARK")
    primary_label = (
        "GPU end-to-end"
        if layers[0]["timing_scope"] == "end_to_end"
        else "Packing excluded"
    )
    headers = ["#", "Layer", "Type", "Shape", "Triton Kernel", f"{primary_label} TF/s"]
    if has_any_repack:
        headers.append("Packing included TF/s")
    rows = []
    for index, layer in enumerate(layers):
        row = [
            str(index),
            layer["name"],
            layer["type"],
            layer["shape"],
            layer["kernel_name"],
            f"{layer['tflops_tri']:.2f}",
        ]
        if has_any_repack:
            value = layer["tflops_tri_e2e"]
            row.append(f"{value:.2f}" if value is not None else "—")
        rows.append(row)
    logger.info("\n%s", _box_table(headers, rows))


def _print_overall_perf_table(layers: list, has_any_repack: bool) -> None:
    logger.info("\nOVERALL PERFORMANCE")
    total_flops = sum(layer["flops"] for layer in layers)
    rows = []
    primary_label = (
        "GPU end-to-end"
        if layers[0]["timing_scope"] == "end_to_end"
        else "Packing excluded"
    )
    modes = [(primary_label, "ms_tri", "tflops_tri")]
    if has_any_repack:
        modes.append(("Packing included", "ms_tri_e2e", "tflops_tri_e2e"))
    for label, time_key, throughput_key in modes:
        times = [
            layer[time_key] if layer[time_key] is not None else layer["ms_tri"]
            for layer in layers
        ]
        throughputs = [
            (
                layer[throughput_key]
                if layer[throughput_key] is not None
                else layer["tflops_tri"]
            )
            for layer in layers
        ]
        total_ms = sum(times)
        aggregate = total_flops / total_ms * 1e-9 if total_ms else 0.0
        rows.append(
            [
                label,
                f"{statistics.mean(throughputs):.2f}",
                f"{statistics.median(throughputs):.2f}",
                f"{aggregate:.2f}",
                f"{total_ms:.4f}",
            ]
        )
    logger.info(
        "\n%s",
        _box_table(
            (
                "Measurement",
                "Mean TFLOPS",
                "Median TFLOPS",
                "Aggregate TFLOPS",
                "Total ms",
            ),
            rows,
        ),
    )


# ----------------------------------------------------------------------------
# Sweep mode
# ----------------------------------------------------------------------------


def _load_model_shapes(model_pattern: str, config_path: str) -> tuple[str, list]:
    """Load conv2d shapes for a model from --model-configs (conv_shapes.json).

    model_pattern: case-insensitive substring matched against model keys.
    config_path: path to the shapes JSON, resolved relative to BASE_DIR
        (same convention as --model-configs for the attention benchmarks).
    Returns (matched_model_name, list of shape tuples in the same form as
    EDGE_CASE_SHAPES — desc is "<model> L<i>").
    """
    with open(os.path.join(BASE_DIR, config_path)) as f:
        data = json.load(f)

    matches = [
        m for m in data if model_pattern.lower() in m.lower() and "conv2d" in data[m]
    ]
    if not matches:
        avail = sorted(m for m, k in data.items() if "conv2d" in k)
        raise ValueError(
            f"No model with 'conv2d' shapes matches {model_pattern!r}. "
            f"Available: {avail}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Pattern {model_pattern!r} matched multiple models: {matches}. "
            f"Use a more specific pattern."
        )

    model = matches[0]
    shapes = []
    for i, s in enumerate(data[model]["conv2d"]):
        shapes.append(
            (
                s["N"],
                s["C"],
                s["H"],
                s["W"],
                s["K"],
                s["R"],
                s["S"],
                (s.get("stride_h", 1), s.get("stride_w", 1)),
                (s.get("pad_h", 0), s.get("pad_w", 0)),
                (s.get("dilation_h", 1), s.get("dilation_w", 1)),
                f"{model} L{i}",
            )
        )
    return model, shapes


def run_sweep(args) -> None:
    """Measure a shape sweep and report Triton-only results."""
    dtype = _torch_dtype(args.dtype)
    if args.smoke:
        shapes = EDGE_CASE_SHAPES
        logger.info("Sweep source: edge cases (%s shapes)", len(shapes))
    else:
        model, shapes = _load_model_shapes(args.model or "resnet50", args.model_configs)
        logger.info(
            "Sweep source: %s :: %s (%s layers)", args.model_configs, model, len(shapes)
        )
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        shapes = [(args.batch_size,) + entry[1:] for entry in shapes]
    logger.info(
        "dtype=%s method=%s layout=%s padding_mode=%s timing_scope=%s "
        "weight_policy=fixed; "
        "GPU graph timings exclude weight preparation",
        args.dtype,
        args.method,
        args.layout,
        args.padding_mode,
        "end_to_end" if args.end_to_end or args.padding_mode != "zeros" else "kernel",
    )
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    layers = []
    for entry in shapes:
        (
            batch,
            channels,
            height,
            width,
            outputs,
            rows,
            columns,
            stride,
            padding,
            dilation,
            name,
        ) = entry
        try:
            result = bench_one_shape(
                batch,
                channels,
                height,
                width,
                outputs,
                rows,
                columns,
                stride,
                padding,
                dilation,
                dtype,
                args.method,
                args.layout,
                bias=not args.no_bias,
                padding_mode=args.padding_mode,
                end_to_end=args.end_to_end,
                bencher=bencher,
            )
        except (RuntimeError, ValueError, AssertionError):
            logger.exception("Failed to benchmark %s", name)
            continue
        layers.append(
            {
                "name": name,
                "type": _kernel_type_tag(rows, columns, dilation),
                "shape": _shape_str(
                    batch, channels, height, width, outputs, rows, columns
                ),
                **result,
            }
        )
    if not layers:
        raise RuntimeError("No Conv2D shapes were benchmarked")
    has_any_repack = any(layer["has_repack"] for layer in layers)
    _print_layer_table(layers, has_any_repack)
    _print_overall_perf_table(layers, has_any_repack)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = get_parser(kernel_name="bench_conv2d")
    override_parser_action(
        p,
        action="--model-configs",
        default="utils/conv_shapes.json",
        help="Conv2D shapes JSON file (sweep mode).",
    )
    p.add_argument(
        "--dtype",
        "--conv_dtype",
        "--conv-dtype",
        type=str.lower,
        choices=["fp16", "bf16"],
        default="fp16",
    )
    p.add_argument(
        "--method",
        type=str.lower,
        choices=list(METHODS.keys()),
        default="auto",
        help="kernel to bench. 'auto' uses the conv2d router.",
    )
    p.add_argument(
        "--layout",
        "--conv_layout",
        "--conv-layout",
        type=str.lower,
        choices=["nchw", "nhwc"],
        default="nchw",
    )
    override_parser_action(
        p, action="--metric", choices=["time", "throughput"], default="throughput"
    )
    p.add_argument(
        "--padding-mode",
        choices=["zeros", "reflect", "replicate", "circular"],
        default="zeros",
        help="Conv2D padding mode; nonzero modes always use end-to-end timing",
    )
    p.add_argument(
        "--end-to-end",
        action="store_true",
        help="include padding, layout conversion, and activation packing in the "
        "primary timing; enabled automatically for nonzero padding modes",
    )
    p.add_argument(
        "--no-bias",
        "--no_bias",
        action="store_true",
        help="bench the bias=None code path",
    )
    p.add_argument(
        "--show-kernel-name",
        "--show_kernel_name",
        action="store_true",
        help="include the routed Triton kernel name in single-shape output",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="sweep mode: use the EDGE_CASE_SHAPES set instead of a real model. "
        "Exercises degenerate paths (C=1, dilation>1, asymmetric dims, etc.) "
        "for regression smoke-testing. NOT representative of production perf.",
    )
    p.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        help="sweep mode: override the batch dim N on every swept shape "
        "(the JSON shapes ship with N=1). Ignored in single-shape mode.",
    )

    # Single-shape mode (used by bench_models.py and one-off measurements).
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--R", type=int, default=None)
    p.add_argument("--S", type=int, default=None)
    p.add_argument("--stride-h", "--stride_h", type=int, default=1)
    p.add_argument("--stride-w", "--stride_w", type=int, default=1)
    p.add_argument("--pad-h", "--pad_h", type=int, default=0)
    p.add_argument("--pad-w", "--pad_w", type=int, default=0)
    p.add_argument("--dilation-h", "--dilation_h", type=int, default=1)
    p.add_argument("--dilation-w", "--dilation_w", type=int, default=1)

    args = p.parse_args(argv)

    single = [args.N, args.C, args.H, args.W, args.K, args.R, args.S]
    if any(v is not None for v in single):
        if any(v is None for v in single):
            p.error("single-shape mode requires all of --N --C --H --W --K --R --S")
        args.single_shape = True
    else:
        args.single_shape = False
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.single_shape:
        run_single_shape(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
