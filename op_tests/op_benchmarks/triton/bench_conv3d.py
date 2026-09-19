# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark aiter.ops.triton.conv.conv3d.

Two modes are available:

- Single shape: pass --N --C --D --H --W --K --T --R --S and the optional
  stride, padding, and dilation flags. This prints one key=value result line.
- Sweep: omit the shape dimensions. This benchmarks either a model from
  conv_shapes.json or the edge-case smoke set, then prints layer-by-layer,
  optional MIOpen-solver, and overall-performance tables.

Model code and weights are not loaded at runtime. For causal wrappers, each
database row describes the inner Conv3D call after explicit wrapper padding
has already been materialized.

TFLOPS uses the direct-convolution operation count:

    2 * N * K * C * T * R * S * OD * P * Q

This keeps effective throughput comparable across Triton routes and MIOpen,
including Winograd routes whose physical operation count differs.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.conv._prepack import (
    clear_conv3d_weight_pack_caches,
    prepack_ncdhw_to_cblocked,
)
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _is_1x1x1_conv,
    _is_3x3x3_conv,
    _out_dhw,
)
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


def flops_conv3d(N, C, K_out, T, R, S, OD, P, Q):
    return 2.0 * N * OD * P * Q * K_out * C * T * R * S


def which_kernel(
    x,
    w_oidhw,
    stride=(1, 1, 1),
    dilation=(1, 1, 1),
    layout="ncdhw",
    padding=(0, 0, 0),
):
    """Return the kernel name selected by the production Conv3D router."""
    N, C, D, H, W = x.shape
    K_out, _, T, R, S = w_oidhw.shape
    route = _resolve_route(
        T,
        R,
        S,
        stride,
        dilation,
        N,
        C,
        D,
        H,
        W,
        K_out,
        layout.lower(),
        padding=padding,
    )
    return route.value


METHODS = (
    "auto",
    "general",
    "1x1x1",
    "cblocked",
    "ndhwc_3x3x3",
    "winograd",
    "winograd_cblocked",
)


# Shape tuple fields:
# (N, C, D, H, W, K, T, R, S, stride, padding, dilation,
#  bias, encoder_calls, decoder_calls, name)
EDGE_CASE_SHAPES = [
    (
        1,
        64,
        4,
        16,
        16,
        128,
        1,
        1,
        1,
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        True,
        0,
        0,
        "1x1x1",
    ),
    (
        1,
        96,
        6,
        32,
        32,
        96,
        1,
        1,
        1,
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        True,
        0,
        0,
        "1x1x1 stride2",
    ),
    (
        1,
        16,
        8,
        20,
        20,
        16,
        5,
        5,
        5,
        (1, 1, 1),
        (2, 2, 2),
        (1, 1, 1),
        True,
        0,
        0,
        "5x5x5 general",
    ),
    (
        1,
        64,
        8,
        24,
        24,
        64,
        3,
        3,
        3,
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
        True,
        0,
        0,
        "3x3x3 stride2",
    ),
    (
        1,
        16,
        8,
        24,
        24,
        16,
        3,
        3,
        3,
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        True,
        0,
        0,
        "3x3x3 dilation2",
    ),
    (
        1,
        1,
        5,
        15,
        17,
        16,
        3,
        3,
        3,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        False,
        0,
        0,
        "single input channel",
    ),
    (
        2,
        32,
        7,
        17,
        23,
        64,
        3,
        5,
        3,
        (1, 2, 1),
        (1, 2, 1),
        (1, 1, 1),
        True,
        0,
        0,
        "asymmetric general",
    ),
]


MIOPEN_ALGO_MAP = {
    "ConvWinoFuryRxS<2-3>": "Winograd Fury F(2,3)",
    "ConvBinWinogradRxSf3x2": "Winograd F(3x3,2x2) binary",
    "GemmFwd1x1_0_1": "GEMM (no workspace)",
    "GemmFwdRest": "GEMM fallback",
}


_MODEL_SHAPES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "conv_shapes.json",
)
_DEFAULT_MODEL = "wan22-a14b-vae"


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _check_close(got, ref, dtype, K_red, is_winograd: bool) -> bool:
    if is_winograd:
        rtol, atol = _winograd_tolerances(dtype, K_red)
    else:
        rtol, atol = dynamic_conv_tolerances(dtype, K_red)
    try:
        torch.testing.assert_close(got.float(), ref.float(), rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def _torch_reference(x, w, bias, stride, padding, dilation, activation):
    """Run the benchmark's PyTorch/MIOpen reference in the requested dtype.

    Keeping the input dtype is important for production-sized Conv3D shapes.
    Promoting a large NCDHW case to fp32 can select a MIOpen solver whose
    workspace exceeds device memory (and, on some stacks, yields an invalid
    output instead of a clean allocation failure).  The Conv2D benchmark uses
    the same-dtype backend result for this reason as well.
    """
    result = F.conv3d(
        x,
        w,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return apply_activation(result, activation)


def _kernel_type_tag(T: int, R: int, S: int, dilation: tuple) -> str:
    if _is_1x1x1_conv(T, R, S, dilation):
        return "[1x1x1]"
    if _is_3x3x3_conv(T, R, S):
        return "[3x3x3]"
    return "[general]"


def _shape_str(N, C, D, H, W, K, T, R, S) -> str:
    return f"({N},{C},{D},{H},{W})→{K}/{T}x{R}x{S}"


def _load_model_shapes(model_pattern: str) -> tuple[str, list]:
    """Load Conv3D shape tuples using a case-insensitive model substring."""
    with open(_MODEL_SHAPES_PATH) as file:
        data = json.load(file)

    matches = [
        model
        for model, operations in data.items()
        if model_pattern.lower() in model.lower() and "conv3d" in operations
    ]
    if not matches:
        available = sorted(model for model, ops in data.items() if "conv3d" in ops)
        raise ValueError(
            f"No model with 'conv3d' shapes matches {model_pattern!r}. "
            f"Available: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Pattern {model_pattern!r} matched multiple models: {matches}. "
            "Use a more specific pattern."
        )

    model = matches[0]
    shapes = []
    for index, shape in enumerate(data[model]["conv3d"]):
        shapes.append(
            (
                shape["N"],
                shape["C"],
                shape["D"],
                shape["H"],
                shape["W"],
                shape["K"],
                shape["T"],
                shape["R"],
                shape["S"],
                (
                    shape.get("stride_d", 1),
                    shape.get("stride_h", 1),
                    shape.get("stride_w", 1),
                ),
                (
                    shape.get("pad_d", 0),
                    shape.get("pad_h", 0),
                    shape.get("pad_w", 0),
                ),
                (
                    shape.get("dilation_d", 1),
                    shape.get("dilation_h", 1),
                    shape.get("dilation_w", 1),
                ),
                shape.get("bias", True),
                shape.get("encoder_calls", 0),
                shape.get("decoder_calls", 0),
                shape.get("name", f"{model} L{index}"),
            )
        )
    return model, shapes


_miopen_solver_cache: dict = {}


def precompute_miopen_solvers(
    shapes, dtype: torch.dtype, layout: str = "ncdhw"
) -> None:
    """Detect the MIOpen solver for each unique shape in one subprocess."""
    layout = layout.lower()
    if layout not in ("ncdhw", "ndhwc"):
        raise ValueError(f"layout must be 'ncdhw' or 'ndhwc', got {layout!r}")
    _miopen_solver_cache.clear()
    unique = []
    seen = set()
    for entry in shapes:
        N, C, D, H, W, K, T, R, S, stride, padding, dilation = entry[:12]
        key = (
            N,
            C,
            D,
            H,
            W,
            K,
            T,
            R,
            S,
            *stride,
            *padding,
            *dilation,
        )
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if not unique:
        return

    dtype_str = {
        torch.float16: "torch.float16",
        torch.bfloat16: "torch.bfloat16",
    }.get(dtype, "torch.float16")
    lines = [
        "import os, sys",
        "os.environ['MIOPEN_LOG_LEVEL']='6'",
        "import torch, torch.nn.functional as F",
    ]
    for index, key in enumerate(unique):
        (
            N,
            C,
            D,
            H,
            W,
            K,
            T,
            R,
            S,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            dilation_d,
            dilation_h,
            dilation_w,
        ) = key
        lines.extend(
            [
                f"sys.stderr.write('SHAPE_BEGIN:{index}\\n');sys.stderr.flush()",
                "try:",
                f"    x=torch.randn({N},{C},{D},{H},{W},device='cuda',dtype={dtype_str})",
                *(
                    ["    x=x.contiguous(memory_format=torch.channels_last_3d)"]
                    if layout == "ndhwc"
                    else []
                ),
                f"    w=torch.randn({K},{C},{T},{R},{S},device='cuda',dtype={dtype_str})",
                (
                    "    y=F.conv3d(x,w,None,"
                    f"stride=({stride_d},{stride_h},{stride_w}),"
                    f"padding=({pad_d},{pad_h},{pad_w}),"
                    f"dilation=({dilation_d},{dilation_h},{dilation_w}))"
                ),
                "    torch.cuda.synchronize()",
                f"    sys.stderr.write('SHAPE_DONE:{index}\\n');sys.stderr.flush()",
                "except Exception as error:",
                (
                    f"    sys.stderr.write('SHAPE_ERROR:{index}:'"
                    "+type(error).__name__+'\\n');sys.stderr.flush()"
                ),
                "finally:",
                "    globals().pop('y',None);globals().pop('x',None);globals().pop('w',None)",
                "    torch.cuda.empty_cache()",
            ]
        )
    script = "\n".join(lines)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MIOPEN_LOG_LEVEL": "6"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[miopen-detect] WARNING: subprocess timed out after 300s; "
            f"solver names are unavailable for {len(unique)} shape(s).",
            file=sys.stderr,
        )
        return
    except Exception as error:  # noqa: BLE001
        print(
            f"[miopen-detect] WARNING: subprocess failed ({error!r}); "
            "solver names are unavailable.",
            file=sys.stderr,
        )
        return

    current_index = None
    pending_solver = None
    attributed = {}
    failed = set()
    begin_re = re.compile(r"^SHAPE_BEGIN:(\d+)\s*$")
    done_re = re.compile(r"^SHAPE_DONE:(\d+)\s*$")
    error_re = re.compile(r"^SHAPE_ERROR:(\d+):")
    chosen_re = re.compile(r"Chosen Algorithm:\s*(\S+)")
    for line in result.stderr.splitlines():
        match = begin_re.match(line)
        if match:
            current_index = int(match.group(1))
            pending_solver = None
            continue
        match = chosen_re.search(line)
        if match and current_index is not None:
            pending_solver = match.group(1).strip(" ,")
            continue
        match = error_re.match(line)
        if match:
            failed.add(int(match.group(1)))
            current_index = None
            pending_solver = None
            continue
        match = done_re.match(line)
        if match:
            index = int(match.group(1))
            if pending_solver is not None:
                attributed[index] = pending_solver
            current_index = None
            pending_solver = None

    for index, solver in attributed.items():
        _miopen_solver_cache[(layout, *unique[index])] = solver

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        print(
            f"[miopen-detect] WARNING: subprocess exited with code "
            f"{result.returncode}; keeping the solver names detected before "
            f"the failure.\n  Last stderr lines:\n{tail}",
            file=sys.stderr,
        )

    missing = len(unique) - len(attributed)
    if missing:
        print(
            f"[miopen-detect] WARNING: {missing}/{len(unique)} shape(s) have no "
            f"MIOpen solver ({len(failed)} shape(s) failed in the subprocess).",
            file=sys.stderr,
        )


def _get_miopen_solver(
    N,
    C,
    D,
    H,
    W,
    K,
    T,
    R,
    S,
    stride,
    padding,
    dilation,
    layout="ncdhw",
) -> str:
    return _miopen_solver_cache.get(
        (
            layout.lower(),
            N,
            C,
            D,
            H,
            W,
            K,
            T,
            R,
            S,
            *stride,
            *padding,
            *dilation,
        ),
        "",
    )


def _run_triton(method, x, w, bias, stride, padding, dilation, layout, activation):
    kwargs = {
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "activation": activation,
    }
    if method == "auto":
        return conv3d(x, w, layout=layout, **kwargs)
    if method == "general":
        return conv3d_general(x, w, layout=layout, **kwargs)
    if method == "1x1x1":
        return conv3d_1x1x1(x, w, layout=layout, **kwargs)
    if method == "cblocked":
        if layout != "ncdhw":
            raise ValueError("cblocked is an NCDHW-only method")
        return conv3d_ncdhw_cblocked(x, w, **kwargs)
    if method == "ndhwc_3x3x3":
        if layout != "ndhwc":
            raise ValueError("ndhwc_3x3x3 requires layout=ndhwc")
        return conv3d_ndhwc_3x3x3(x, w, **kwargs)
    if method == "winograd":
        if layout != "ncdhw":
            raise ValueError("winograd is an NCDHW-only method")
        return conv3d_winograd_hw_f4x3(x, w, **kwargs)
    if method == "winograd_cblocked":
        if layout != "ncdhw":
            raise ValueError("winograd_cblocked is an NCDHW-only method")
        return conv3d_winograd_hw_f4x3_cblocked(x, w, **kwargs)
    raise ValueError(f"unknown method: {method}; choices: {list(METHODS)}")


def bench_one_shape(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    K: int,
    T: int,
    R: int,
    S: int,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    dtype: torch.dtype,
    method: str,
    layout: str,
    bias: bool = True,
    activation: str = "none",
    measure_repack: bool = True,
) -> dict:
    """Time and correctness-check one Conv3D shape."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; conv3d bench requires a GPU.")

    OD, P, Q = _out_dhw(D, H, W, T, R, S, stride, padding, dilation)
    if OD < 1 or P < 1 or Q < 1:
        raise ValueError(
            "output spatial dims < 1 for shape "
            f"N={N} C={C} D={D} H={H} W={W} K={K} T={T} R={R} S={S} "
            f"stride={stride} padding={padding} dilation={dilation}"
        )

    x = torch.randn((N, C, D, H, W), device="cuda", dtype=dtype)
    if layout == "ndhwc":
        x = x.contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn((K, C, T, R, S), device="cuda", dtype=dtype)
    b = torch.randn((K,), device="cuda", dtype=dtype) if bias else None

    def run_triton():
        return _run_triton(
            method, x, w, b, stride, padding, dilation, layout, activation
        )

    def run_torch():
        return _torch_reference(x, w, b, stride, padding, dilation, activation)

    output = run_triton()
    torch.cuda.synchronize()
    if method == "auto":
        kernel_name = which_kernel(
            x,
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
            layout=layout,
        )
    else:
        kernel_name = method
    is_winograd = "winograd" in kernel_name.lower() or "wino" in kernel_name.lower()

    reference = run_torch()
    K_red = C * T * R * S
    correct = _check_close(output, reference, dtype, K_red, is_winograd)
    del output, reference
    torch.cuda.empty_cache()

    packs_input = "cblocked" in kernel_name.lower()
    if packs_input:
        x_blocked_pre, _ = prepack_ncdhw_to_cblocked(x, BLOCK_K)
        cblocked_fn = (
            conv3d_winograd_hw_f4x3_cblocked
            if "winograd" in kernel_name.lower() or "wino" in kernel_name.lower()
            else conv3d_ncdhw_cblocked
        )

        def run_triton_kernel_only():
            return cblocked_fn(
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation=activation,
                x_blocked=x_blocked_pre,
            )

        ms_tri = triton.testing.do_bench(run_triton_kernel_only, warmup=15, rep=50)
    else:
        ms_tri = triton.testing.do_bench(run_triton, warmup=15, rep=50)

    ms_torch = triton.testing.do_bench(run_torch, warmup=15, rep=50)
    has_repack = measure_repack and packs_input
    if has_repack:
        ms_tri_e2e = triton.testing.do_bench(run_triton, warmup=15, rep=50)
    else:
        ms_tri_e2e = None

    flops = flops_conv3d(N, C, K, T, R, S, OD, P, Q)
    tflops_tri = flops / (ms_tri * 1e-3) / 1e12
    tflops_torch = flops / (ms_torch * 1e-3) / 1e12
    tflops_tri_e2e = flops / (ms_tri_e2e * 1e-3) / 1e12 if ms_tri_e2e else None

    return {
        "ms_tri": ms_tri,
        "ms_torch": ms_torch,
        "ms_tri_e2e": ms_tri_e2e,
        "tflops_tri": tflops_tri,
        "tflops_torch": tflops_torch,
        "tflops_tri_e2e": tflops_tri_e2e,
        "correct": correct,
        "kernel_name": kernel_name,
        "has_repack": has_repack,
        "flops": flops,
    }


def _format_single_shape_line(args, result: dict) -> str:
    """Return the single-line key=value output used by benchmark tooling."""
    primary = result["ms_tri"] if args.metric == "time" else result["tflops_tri"]
    parts = [
        f"N={args.N}",
        f"C={args.C}",
        f"D={args.D}",
        f"H={args.H}",
        f"W={args.W}",
        f"K={args.K}",
        f"T={args.T}",
        f"R={args.R}",
        f"S={args.S}",
        f"method={args.method}",
        f"layout={args.layout}",
        f"ms_tri={result['ms_tri']:.4f}",
        f"ms_torch={result['ms_torch']:.4f}",
        f"tflops_tri={result['tflops_tri']:.4f}",
        f"tflops_torch={result['tflops_torch']:.4f}",
        f"correct={int(result['correct'])}",
    ]
    if args.show_kernel_name:
        parts.append(f"kernel={result['kernel_name'] or 'unknown'}")
    parts.append(f"{primary:.4f}")
    return " ".join(parts)


def run_single_shape(args) -> None:
    dtype = _torch_dtype(args.dtype)
    stride = (args.stride_d, args.stride_h, args.stride_w)
    padding = (args.pad_d, args.pad_h, args.pad_w)
    dilation = (args.dilation_d, args.dilation_h, args.dilation_w)
    try:
        result = bench_one_shape(
            args.N,
            args.C,
            args.D,
            args.H,
            args.W,
            args.K,
            args.T,
            args.R,
            args.S,
            stride,
            padding,
            dilation,
            dtype,
            args.method,
            args.layout,
            bias=not args.no_bias,
            activation=args.activation,
            measure_repack=False,
        )
        print(_format_single_shape_line(args, result))
    finally:
        clear_conv3d_weight_pack_caches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _box_table(headers, rows, align: list | None = None) -> str:
    """Render headers and rows as a box-drawn table."""
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


def _print_layer_table(
    layers: list, has_any_repack: bool, miopen_enabled: bool
) -> None:
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER BENCHMARK")
    print("=" * 80)

    headers = ["#", "Layer", "Type", "Shape"]
    if miopen_enabled:
        headers.append("MIOpen Solver")
    headers.extend(["Triton Kernel", "Tri Kernel TF/s"])
    if has_any_repack:
        headers.append("Tri Kernel+Repack TF/s")
    headers.extend(["Torch TF/s", "Correct", "Winner"])

    rows = []
    for index, layer in enumerate(layers):
        row = [
            str(layer.get("shape_index", index)),
            layer["name"],
            layer["type"],
            layer["shape"],
        ]
        if miopen_enabled:
            row.append(layer["miopen_solver"] or "—")
        row.extend([layer["kernel_name"] or "—", f"{layer['tflops_tri']:.2f}"])
        if has_any_repack:
            row.append(
                f"{layer['tflops_tri_e2e']:.2f}"
                if layer["tflops_tri_e2e"] is not None
                else "—"
            )
        row.extend(
            [
                f"{layer['tflops_torch']:.2f}",
                "yes" if layer["correct"] else "NO",
                "Triton" if layer["tflops_tri"] > layer["tflops_torch"] else "Torch",
            ]
        )
        rows.append(row)

    print(_box_table(headers, rows))


def _print_miopen_solver_table(layers: list) -> None:
    from collections import OrderedDict

    solver_layers: dict = OrderedDict()
    for index, layer in enumerate(layers):
        solver = layer.get("miopen_solver") or "unknown"
        solver_layers.setdefault(solver, []).append(
            f"L{layer.get('shape_index', index)}"
        )
    if not any(solver != "unknown" for solver in solver_layers):
        return

    print("\n" + "=" * 80)
    print("MIOpen SOLVER SUMMARY")
    print("=" * 80)
    rows = []
    for solver, layer_names in solver_layers.items():
        algorithm = MIOPEN_ALGO_MAP.get(solver, solver)
        used_for = ", ".join(layer_names)
        if len(used_for) > 80:
            used_for = (
                ", ".join(layer_names[:10]) + f" ... ({len(layer_names)} layers total)"
            )
        rows.append([solver, algorithm, used_for])
    print(_box_table(("MIOpen Solver", "Algorithm Type", "Used For"), rows))


def _print_overall_perf_table(
    layers: list, has_any_repack: bool, requested_count: int | None = None
) -> None:
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    completed_count = len(layers)
    if requested_count is None:
        requested_count = completed_count
    errored_count = requested_count - completed_count
    incorrect_count = sum(not layer["correct"] for layer in layers)
    if errored_count:
        print(
            f"WARNING: partial sweep: {completed_count}/{requested_count} shapes "
            f"completed; aggregates exclude {errored_count} errored shape(s)."
        )
    if incorrect_count:
        print(
            f"WARNING: {incorrect_count} completed shape(s) failed correctness; "
            "their performance results are not valid."
        )

    tri_tflops = [layer["tflops_tri"] for layer in layers]
    torch_tflops = [layer["tflops_torch"] for layer in layers]
    tri_ms = [layer["ms_tri"] for layer in layers]
    torch_ms = [layer["ms_torch"] for layer in layers]
    sum_flops = sum(layer["flops"] for layer in layers)
    sum_time_tri = sum(value * 1e-3 for value in tri_ms)
    sum_time_torch = sum(value * 1e-3 for value in torch_ms)
    aggregate_tri = sum_flops / sum_time_tri / 1e12 if sum_time_tri else 0.0
    aggregate_torch = sum_flops / sum_time_torch / 1e12 if sum_time_torch else 0.0
    layer_count = completed_count
    tri_wins = sum(1 for layer in layers if layer["tflops_tri"] > layer["tflops_torch"])

    rows = [
        [
            "Mean TFLOPS (kernel)",
            f"{statistics.mean(tri_tflops):.2f}",
            f"{statistics.mean(torch_tflops):.2f}",
        ]
    ]
    if has_any_repack:
        e2e_tflops = [
            (
                layer["tflops_tri_e2e"]
                if layer["tflops_tri_e2e"] is not None
                else layer["tflops_tri"]
            )
            for layer in layers
        ]
        e2e_ms = [
            layer["ms_tri_e2e"] if layer["ms_tri_e2e"] is not None else layer["ms_tri"]
            for layer in layers
        ]
        sum_time_e2e = sum(value * 1e-3 for value in e2e_ms)
        aggregate_e2e = sum_flops / sum_time_e2e / 1e12 if sum_time_e2e else 0.0
        e2e_wins = sum(
            1
            for layer, e2e_tflop in zip(layers, e2e_tflops)
            if e2e_tflop > layer["tflops_torch"]
        )
        rows.append(
            [
                "Mean TFLOPS (kernel+repack)",
                f"{statistics.mean(e2e_tflops):.2f}",
                f"{statistics.mean(torch_tflops):.2f}",
            ]
        )
    rows.append(
        [
            "Median TFLOPS (kernel)",
            f"{statistics.median(tri_tflops):.2f}",
            f"{statistics.median(torch_tflops):.2f}",
        ]
    )
    if has_any_repack:
        rows.append(
            [
                "Median TFLOPS (kernel+repack)",
                f"{statistics.median(e2e_tflops):.2f}",
                f"{statistics.median(torch_tflops):.2f}",
            ]
        )
    rows.append(
        ["Aggregate TFLOPS (kernel)", f"{aggregate_tri:.2f}", f"{aggregate_torch:.2f}"]
    )
    if has_any_repack:
        rows.append(
            [
                "Aggregate TFLOPS (kernel+repack)",
                f"{aggregate_e2e:.2f}",
                f"{aggregate_torch:.2f}",
            ]
        )
    rows.append(
        ["Total kernel time (ms)", f"{sum(tri_ms):.2f}", f"{sum(torch_ms):.2f}"]
    )
    if has_any_repack:
        rows.append(
            [
                "Total kernel+repack time (ms)",
                f"{sum(e2e_ms):.2f}",
                f"{sum(torch_ms):.2f}",
            ]
        )
    rows.append(
        [
            "Layer wins (kernel)",
            f"{tri_wins}/{layer_count}",
            f"{layer_count - tri_wins}/{layer_count}",
        ]
    )
    if has_any_repack:
        rows.append(
            [
                "Layer wins (kernel+repack)",
                f"{e2e_wins}/{layer_count}",
                f"{layer_count - e2e_wins}/{layer_count}",
            ]
        )
    correct_count = completed_count - incorrect_count
    correctness = f"{correct_count}/{requested_count} passed"
    details = []
    if incorrect_count:
        details.append(f"{incorrect_count} incorrect")
    if errored_count:
        details.append(f"{errored_count} errored")
    if details:
        correctness += f" ({', '.join(details)})"
    rows.append(["Correctness", correctness, "—"])
    print(_box_table(("Metric", "Triton", "PyTorch (MIOpen)"), rows))


def _print_weighted_model_totals(
    layers: list, has_any_repack: bool, requested_count: int | None = None
) -> None:
    rows = []
    for workload, count_key in (
        ("Encoder", "encoder_calls"),
        ("Decoder", "decoder_calls"),
    ):
        call_count = sum(layer[count_key] for layer in layers)
        if not call_count:
            continue
        tri_ms = sum(layer[count_key] * layer["ms_tri"] for layer in layers)
        torch_ms = sum(layer[count_key] * layer["ms_torch"] for layer in layers)
        row = [workload, str(call_count), f"{tri_ms / 1000:.3f}"]
        if has_any_repack:
            e2e_ms = sum(
                layer[count_key]
                * (
                    layer["ms_tri_e2e"]
                    if layer["ms_tri_e2e"] is not None
                    else layer["ms_tri"]
                )
                for layer in layers
            )
            row.append(f"{e2e_ms / 1000:.3f}")
        row.extend(
            [
                f"{torch_ms / 1000:.3f}",
                f"{torch_ms / tri_ms:.2f}x" if tri_ms else "—",
            ]
        )
        rows.append(row)
    if not rows:
        return

    print("\n" + "=" * 80)
    title = "CALL-COUNT-WEIGHTED MODEL TOTALS"
    if requested_count is not None and requested_count != len(layers):
        title += " (INCOMPLETE — ERRORED LAYERS EXCLUDED)"
    print(title)
    print("=" * 80)
    headers = ["Workload", "Calls", "Tri Kernel (s)"]
    if has_any_repack:
        headers.append("Tri Kernel+Repack (s)")
    headers.extend(["PyTorch (s)", "Kernel Speedup"])
    print(_box_table(headers, rows))


def run_sweep(args) -> None:
    dtype = _torch_dtype(args.dtype)
    if args.smoke:
        shapes = EDGE_CASE_SHAPES
        print(
            f"# Sweep source: EDGE_CASE_SHAPES ({len(shapes)} shapes) — smoke / "
            "degenerate-path coverage, NOT representative of production workloads"
        )
    else:
        model_name = args.model if args.model else _DEFAULT_MODEL
        try:
            model, shapes = _load_model_shapes(model_name)
        except (FileNotFoundError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        label = f":: {model} ({len(shapes)} layers)"
        if args.model is None:
            label += "  (default — pass --model X or --smoke to change)"
        print(f"# Sweep source: conv_shapes.json {label}")

    if args.batch_size is not None:
        if args.batch_size < 1:
            print("ERROR: --batch-size must be >= 1", file=sys.stderr)
            raise SystemExit(1)
        shapes = [(args.batch_size,) + entry[1:] for entry in shapes]
        print(f"# batch-size override: N={args.batch_size} on all swept shapes")

    print(
        f"# dtype={args.dtype} method={args.method} layout={args.layout} "
        f"activation={args.activation} "
        f"miopen_solvers={'on' if args.miopen_solvers else 'off'}"
    )

    if args.miopen_solvers:
        print("# Detecting MIOpen solvers (subprocess; this can take a minute)...")
        precompute_miopen_solvers(shapes, dtype, args.layout)
        print("# MIOpen solver detection complete.")

    requested_count = len(shapes)
    layers = []
    failures = []
    for shape_index, entry in enumerate(shapes):
        (
            N,
            C,
            D,
            H,
            W,
            K,
            T,
            R,
            S,
            stride,
            padding,
            dilation,
            shape_bias,
            encoder_calls,
            decoder_calls,
            name,
        ) = entry
        try:
            result = bench_one_shape(
                N,
                C,
                D,
                H,
                W,
                K,
                T,
                R,
                S,
                stride,
                padding,
                dilation,
                dtype,
                args.method,
                args.layout,
                bias=shape_bias and not args.no_bias,
                activation=args.activation,
                measure_repack=True,
            )
        except Exception as error:  # noqa: BLE001
            failures.append((shape_index, name, type(error).__name__, str(error)))
            print(
                f"  {name:<24} ERROR: {type(error).__name__}: {error}", file=sys.stderr
            )
            continue
        finally:
            clear_conv3d_weight_pack_caches()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        miopen_solver = (
            _get_miopen_solver(
                N,
                C,
                D,
                H,
                W,
                K,
                T,
                R,
                S,
                stride,
                padding,
                dilation,
                args.layout,
            )
            if args.miopen_solvers
            else ""
        )
        layers.append(
            {
                "shape_index": shape_index,
                "name": name,
                "type": _kernel_type_tag(T, R, S, dilation),
                "shape": _shape_str(N, C, D, H, W, K, T, R, S),
                "kernel_name": result["kernel_name"],
                "miopen_solver": miopen_solver,
                "tflops_tri": result["tflops_tri"],
                "tflops_tri_e2e": result["tflops_tri_e2e"],
                "tflops_torch": result["tflops_torch"],
                "ms_tri": result["ms_tri"],
                "ms_tri_e2e": result["ms_tri_e2e"],
                "ms_torch": result["ms_torch"],
                "correct": result["correct"],
                "flops": result["flops"],
                "encoder_calls": encoder_calls,
                "decoder_calls": decoder_calls,
            }
        )

    if not layers:
        print(
            f"No layers benched: all {requested_count} requested shapes errored.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    has_any_repack = any(layer["ms_tri_e2e"] is not None for layer in layers)
    _print_layer_table(layers, has_any_repack, miopen_enabled=args.miopen_solvers)
    if args.miopen_solvers:
        _print_miopen_solver_table(layers)
    _print_overall_perf_table(layers, has_any_repack, requested_count)
    _print_weighted_model_totals(layers, has_any_repack, requested_count)

    incorrect_count = sum(not layer["correct"] for layer in layers)
    if failures or incorrect_count:
        print(
            "# Sweep FAILED: "
            f"requested={requested_count} completed={len(layers)} "
            f"correct={len(layers) - incorrect_count} "
            f"incorrect={incorrect_count} errored={len(failures)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_conv3d",
        description="Benchmark aiter.ops.triton.conv.conv3d (single shape or sweep).",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dtype",
        "--conv_dtype",
        "--conv-dtype",
        type=str.lower,
        choices=["fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument(
        "--method",
        type=str.lower,
        choices=METHODS,
        default="auto",
        help="kernel to bench. 'auto' uses the conv3d router.",
    )
    parser.add_argument(
        "--layout",
        "--conv_layout",
        "--conv-layout",
        type=str.lower,
        choices=["ncdhw", "ndhwc"],
        default="ncdhw",
    )
    parser.add_argument(
        "--metric",
        type=str.lower,
        choices=["time", "throughput"],
        default="throughput",
    )
    parser.add_argument(
        "--activation",
        type=str.lower,
        choices=["none", "relu", "relu6", "gelu"],
        default="none",
    )
    parser.add_argument(
        "--no-bias",
        "--no_bias",
        action="store_true",
        help="bench the bias=None code path",
    )
    parser.add_argument(
        "--show-kernel-name",
        "--show_kernel_name",
        action="store_true",
        help="include the routed Triton kernel name in single-shape output",
    )
    parser.add_argument(
        "--miopen-solvers",
        "--miopen_solvers",
        action="store_true",
        help="detect MIOpen solver names via a subprocess (sweep mode only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="sweep mode: load Conv3D shapes for this model from "
        "conv_shapes.json using a case-insensitive substring. If omitted, "
        f"defaults to {_DEFAULT_MODEL} unless --smoke is passed.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="sweep mode: use edge cases instead of a model workload",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        help="sweep mode: override N on every swept shape",
    )

    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--C", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--S", type=int, default=None)
    parser.add_argument("--stride-d", "--stride_d", type=int, default=1)
    parser.add_argument("--stride-h", "--stride_h", type=int, default=1)
    parser.add_argument("--stride-w", "--stride_w", type=int, default=1)
    parser.add_argument("--pad-d", "--pad_d", type=int, default=0)
    parser.add_argument("--pad-h", "--pad_h", type=int, default=0)
    parser.add_argument("--pad-w", "--pad_w", type=int, default=0)
    parser.add_argument("--dilation-d", "--dilation_d", type=int, default=1)
    parser.add_argument("--dilation-h", "--dilation_h", type=int, default=1)
    parser.add_argument("--dilation-w", "--dilation_w", type=int, default=1)

    args = parser.parse_args(argv)
    dimensions = [
        args.N,
        args.C,
        args.D,
        args.H,
        args.W,
        args.K,
        args.T,
        args.R,
        args.S,
    ]
    if any(value is not None for value in dimensions):
        if any(value is None for value in dimensions):
            parser.error(
                "single-shape mode requires all of --N --C --D --H --W --K --T --R --S"
            )
        if any(value <= 0 for value in dimensions):
            parser.error("all single-shape dimensions must be positive")
        args.single_shape = True
    else:
        args.single_shape = False

    if min(args.stride_d, args.stride_h, args.stride_w) <= 0:
        parser.error("stride values must be positive")
    if min(args.pad_d, args.pad_h, args.pad_w) < 0:
        parser.error("padding values must be non-negative")
    if min(args.dilation_d, args.dilation_h, args.dilation_w) <= 0:
        parser.error("dilation values must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.single_shape:
        run_single_shape(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
