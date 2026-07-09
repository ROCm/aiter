# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark aiter.ops.triton.conv.conv2d.

Two modes:
- Single-shape (pass --N --C --H --W --K --R --S [--stride ...] etc.)
  Bench one shape and emit a single key=value result line. Useful for
  ad-hoc one-off measurements or scripting around a specific shape.
- Sweep (no --N). Iterates either the built-in default shape list or the
  conv2d shapes for a model in conv_shapes.json (--model NAME), and
  prints three box-drawn tables at the end:
    1. LAYER-BY-LAYER BENCHMARK   (per-layer Tri vs Torch + correctness)
    2. MIOpen SOLVER SUMMARY      (only when --miopen-solvers is passed)
    3. OVERALL PERFORMANCE        (mean/median/aggregate TFLOPS, layer wins)

Each shape is timed with triton.testing.do_bench against
torch.nn.functional.conv2d as the reference backend (MIOpen on AMD),
and a correctness check compares the Triton output against F.conv2d
within the same tolerance model the test suite uses.

The selected Triton kernel name is captured for every shape. The MIOpen
solver name is captured only when --miopen-solvers is passed, because
detection requires a separate subprocess with MIOPEN_LOG_LEVEL=6 (~60s
fixed startup cost).

For NCHW non-1x1 shapes, two Triton numbers are reported: kernel-only (the
cblocked input pack is done once up front and handed to the kernel via
x_blocked=) and kernel+repack (the public call, which packs the input on the
host every time — the real inference cost when the input layout changes per
call).

No model loading at runtime — model shapes come from the pre-extracted
conv_shapes.json: each conv layer's (N,C,H,W,K,R,S,stride,pad,dilation) was
captured offline once per model via forward hooks, deduped, and frozen here.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _out_hw,
    _is_1x1_conv,
    _is_3x3_conv,
)
from aiter.ops.triton.conv._prepack import prepack_nchw_to_cblocked
from aiter.ops.triton.conv.conv2d import (
    conv2d,
    conv2d_nchw,
    conv2d_nchw_cblocked,
    conv2d_nhwc,
    conv2d_winograd_f4x3,
    conv2d_winograd_f4x3_cblocked,
    _resolve_route,
)
from op_tests.triton_tests.conv._helpers import (
    dynamic_conv_tolerances,
    _winograd_tolerances,
)


def flops_conv(N, C, K_out, R, S, P, Q):
    return 2.0 * N * P * Q * K_out * C * R * S


def which_kernel(x, w_oihw, stride=(1, 1), dilation=(1, 1), layout="nchw"):
    """Name of the Triton kernel ``conv2d`` would route to for these shapes,
    without launching anything. Delegates to the production ``_resolve_route``
    (the same decision the router uses), so the label can never drift from
    dispatch. Bench-only: used to label rows and pick correctness tolerances."""
    N, C, H, W_in = x.shape
    K_out, _, R, S = w_oihw.shape
    route = _resolve_route(R, S, stride, dilation, N, C, H, W_in, K_out, layout.lower())
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


# MIOpen solver names → human-readable algorithm types (matches old suite.py).
MIOPEN_ALGO_MAP = {
    "ConvWinoFuryRxS<2-3>": "Winograd Fury F(2,3)",
    "ConvBinWinogradRxSf3x2": "Winograd F(3x3,2x2) binary",
    "GemmFwd1x1_0_1": "GEMM (no workspace)",
    "GemmFwdRest": "GEMM fallback",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _torch_dtype(s: str) -> torch.dtype:
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {s}")


def _check_close(got, ref, dtype, K_red, is_winograd: bool) -> bool:
    """Tolerance-aware correctness check (same model as the pytest suite)."""
    if is_winograd:
        rtol, atol = _winograd_tolerances(dtype, K_red, "f4x3")
    else:
        rtol, atol = dynamic_conv_tolerances(dtype, K_red)
    try:
        torch.testing.assert_close(got.float(), ref.float(), rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def _kernel_type_tag(R: int, S: int, dilation: tuple) -> str:
    if _is_1x1_conv(R, S, dilation):
        return "[1x1]"
    if _is_3x3_conv(R, S):
        return "[3x3]"
    return "[general]"


def _shape_str(N, C, H, W, K, R, S) -> str:
    return f"({N},{C},{H},{W})→{K}/{R}x{S}"


# ----------------------------------------------------------------------------
# MIOpen solver detection (subprocess-based, opt-in via --miopen-solvers)
# ----------------------------------------------------------------------------


_miopen_solver_cache: dict = {}


def precompute_miopen_solvers(shapes, dtype: torch.dtype) -> None:
    """Detect MIOpen solver per shape via a single subprocess.

    Spawns Python with MIOPEN_LOG_LEVEL=6, runs F.conv2d for each shape,
    and parses stderr for "Chosen Algorithm:" lines. SHAPE_DONE markers
    on stderr disambiguate which "Chosen Algorithm" line belongs to which
    shape (positional alignment is unreliable when MIOpen logs vary).

    Cache populated as a side effect. Use _get_miopen_solver to read.
    """
    global _miopen_solver_cache

    unique = []
    seen = set()
    for entry in shapes:
        N, C, H, W, K, R, S, stride, padding, dilation = entry[:10]
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, tuple) else (padding, padding)
        d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        key = (N, C, H, W, K, R, S, s_h, s_w, p_h, p_w, d_h, d_w)
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
    for i, (N, C, H, W, K, R, S, s_h, s_w, p_h, p_w, d_h, d_w) in enumerate(unique):
        lines.append(f"# shape {i}")
        lines.append(f"x=torch.randn({N},{C},{H},{W},device='cuda',dtype={dtype_str})")
        lines.append(f"w=torch.randn({K},{C},{R},{S},device='cuda',dtype={dtype_str})")
        lines.append(
            f"F.conv2d(x,w,None,stride=({s_h},{s_w}),padding=({p_h},{p_w}),dilation=({d_h},{d_w}))"
        )
        lines.append("torch.cuda.synchronize()")
        lines.append(f"sys.stderr.write('SHAPE_DONE:{i}\\n');sys.stderr.flush()")
    script = "\n".join(lines)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "MIOPEN_LOG_LEVEL": "6"},
        )
    except subprocess.TimeoutExpired:
        print(
            f"[miopen-detect] WARNING: subprocess timed out after 120s; "
            f"MIOpen solver column will be empty for {len(unique)} shape(s).",
            file=sys.stderr,
        )
        return
    except Exception as e:
        print(
            f"[miopen-detect] WARNING: subprocess failed ({e!r}); "
            f"MIOpen solver column will be empty.",
            file=sys.stderr,
        )
        return

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().split("\n")[-5:])
        print(
            f"[miopen-detect] WARNING: subprocess exited with code "
            f"{result.returncode}; MIOpen solver column will be empty.\n"
            f"  Last stderr lines:\n{tail}",
            file=sys.stderr,
        )
        return

    pending: Optional[str] = None
    attributed: dict = {}
    orphan = 0
    shape_done_re = re.compile(r"^SHAPE_DONE:(\d+)\s*$")
    chosen_re = re.compile(r"Chosen Algorithm:\s*(\S+)")
    for line in result.stderr.split("\n"):
        m = chosen_re.search(line)
        if m:
            pending = m.group(1).strip(" ,")
            continue
        m = shape_done_re.match(line)
        if m:
            idx = int(m.group(1))
            if pending is not None:
                attributed[idx] = pending
            else:
                orphan += 1
            pending = None
    for idx, solver in attributed.items():
        _miopen_solver_cache[unique[idx]] = solver

    missing = len(unique) - len(attributed)
    if missing > 0:
        print(
            f"[miopen-detect] WARNING: {missing}/{len(unique)} shape(s) have no "
            f"MIOpen solver detected ({orphan} marker(s) had no preceding "
            f"'Chosen Algorithm' line). Common causes: MIOpen log format changed, "
            f"MIOPEN_LOG_LEVEL was overridden, or the shape failed in the subprocess.",
            file=sys.stderr,
        )


def _get_miopen_solver(N, C, H, W, K, R, S, stride, padding, dilation) -> str:
    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
    p_h, p_w = padding if isinstance(padding, tuple) else (padding, padding)
    d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    return _miopen_solver_cache.get(
        (N, C, H, W, K, R, S, s_h, s_w, p_h, p_w, d_h, d_w), ""
    )


# ----------------------------------------------------------------------------
# Per-shape bench (returns rich dict)
# ----------------------------------------------------------------------------


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
) -> dict:
    """Time + correctness-check one shape. Returns a dict with full metadata.

    Keys: ms_tri, ms_torch, ms_tri_e2e (or None), tflops_tri, tflops_torch,
    tflops_tri_e2e (or None), correct, kernel_name, has_repack, flops.

    measure_repack: if True (default), additionally times the kernel+input-repack
    path for NCHW non-1x1 shapes by timing the public call (which packs the input
    on the host each time).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; conv2d bench requires a GPU.")

    P, Q = _out_hw(H, W, R, S, stride, padding, dilation)
    if P < 1 or Q < 1:
        raise ValueError(
            f"output spatial dims < 1 for shape "
            f"N={N} C={C} H={H} W={W} K={K} R={R} S={S} "
            f"stride={stride} padding={padding} dilation={dilation}"
        )

    device = "cuda"
    x = torch.randn((N, C, H, W), device=device, dtype=dtype)
    w = torch.randn((K, C, R, S), device=device, dtype=dtype)
    b = torch.randn((K,), device=device, dtype=dtype) if bias else None

    if layout == "nhwc":
        x_in = x.to(memory_format=torch.channels_last)
        kernel_fn = conv2d_nhwc
    else:
        x_in = x
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}; choices: {list(METHODS)}")
        kernel_fn = METHODS[method]

    def run_triton():
        return kernel_fn(
            x_in,
            w,
            b,
            stride,
            padding,
            dilation,
            activation="none",
        )

    def run_torch():
        return F.conv2d(x_in, w, b, stride=stride, padding=padding, dilation=dilation)

    # One run for correctness. The routed-kernel name is queried separately via
    # which_kernel (no global side-channel); forced --methods name the kernel
    # directly, so for those we use the method string (its winograd/cblocked
    # substrings drive the checks below).
    y_tri = run_triton()
    torch.cuda.synchronize()
    if method in ("auto", "default") or layout == "nhwc":
        kernel_name = which_kernel(x_in, w, stride, dilation, layout=layout)
    else:
        kernel_name = method
    is_winograd = "winograd" in kernel_name.lower() or "wino" in kernel_name.lower()

    y_ref = run_torch()
    correct = _check_close(
        y_tri, y_ref, dtype, K_red=C * R * S, is_winograd=is_winograd
    )

    # Kernel-only timing: the cblocked routes pack the input on the host. To
    # time just the kernel we pre-pack once outside the loop and hand the packed
    # input back via x_blocked=. Non-cblocked routes do no host pack, so the
    # plain call already is kernel-only.
    packs_input = "cblocked" in kernel_name.lower()
    if packs_input:
        x_blocked_pre, _ = prepack_nchw_to_cblocked(x_in, BLOCK_K)
        cblk_fn = (
            conv2d_winograd_f4x3_cblocked
            if "winograd" in kernel_name.lower()
            else conv2d_nchw_cblocked
        )

        def run_triton_kernel_only():
            return cblk_fn(
                x_in,
                w,
                b,
                stride,
                padding,
                dilation,
                activation="none",
                x_blocked=x_blocked_pre,
            )

        ms_tri = triton.testing.do_bench(run_triton_kernel_only, warmup=15, rep=50)
    else:
        ms_tri = triton.testing.do_bench(run_triton, warmup=15, rep=50)

    ms_th = triton.testing.do_bench(run_torch, warmup=15, rep=50)

    # Kernel+repack timing: the public call now always packs the input on the
    # host, so timing it directly includes the repack. NCHW non-1x1 only (1x1
    # takes raw weights; NHWC has no host-side input pack). For non-cblocked
    # routes there's no host pack, so this matches the kernel-only number.
    has_repack = (
        measure_repack and layout != "nhwc" and not _is_1x1_conv(R, S, dilation)
    )
    if has_repack:
        ms_tri_e2e = triton.testing.do_bench(run_triton, warmup=15, rep=50)
    else:
        ms_tri_e2e = None

    flops = flops_conv(N, C, K, R, S, P, Q)
    tflops_tri = flops / (ms_tri * 1e-3) / 1e12
    tflops_th = flops / (ms_th * 1e-3) / 1e12
    tflops_tri_e2e = flops / (ms_tri_e2e * 1e-3) / 1e12 if ms_tri_e2e else None

    return {
        "ms_tri": ms_tri,
        "ms_torch": ms_th,
        "ms_tri_e2e": ms_tri_e2e,
        "tflops_tri": tflops_tri,
        "tflops_torch": tflops_th,
        "tflops_tri_e2e": tflops_tri_e2e,
        "correct": correct,
        "kernel_name": kernel_name,
        "has_repack": has_repack,
        "flops": flops,
    }


# ----------------------------------------------------------------------------
# Single-shape mode (used by bench_models.py)
# ----------------------------------------------------------------------------


def _format_single_shape_line(args, result: dict) -> str:
    """Single-line key=value output for bench_models.py to parse.

    Last whitespace-separated token is the primary metric value.
    """
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
    stride = (args.stride_h, args.stride_w)
    padding = (args.pad_h, args.pad_w)
    dilation = (args.dilation_h, args.dilation_w)
    # Single-shape mode (bench_models.py consumer): skip kernel+repack timing
    # to keep per-call cost predictable for the framework.
    result = bench_one_shape(
        args.N,
        args.C,
        args.H,
        args.W,
        args.K,
        args.R,
        args.S,
        stride,
        padding,
        dilation,
        dtype,
        args.method,
        args.layout,
        bias=not args.no_bias,
        measure_repack=False,
    )
    print(_format_single_shape_line(args, result))


# ----------------------------------------------------------------------------
# Box-drawn table printers
# ----------------------------------------------------------------------------


def _box_table(headers, rows, align: Optional[list] = None) -> str:
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


def _print_layer_table(
    layers: list, has_any_repack: bool, miopen_enabled: bool
) -> None:
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER BENCHMARK")
    print("=" * 80)

    headers = ["#", "Layer", "Type", "Shape"]
    if miopen_enabled:
        headers.append("MIOpen Solver")
    headers.append("Triton Kernel")
    headers.append("Tri Kernel TF/s")
    if has_any_repack:
        headers.append("Tri Kernel+Repack TF/s")
    headers.extend(["Torch TF/s", "Winner"])

    rows = []
    for i, lr in enumerate(layers):
        row = [str(i), lr["name"], lr["type"], lr["shape"]]
        if miopen_enabled:
            row.append(lr["miopen_solver"] or "—")
        row.append(lr["kernel_name"] or "—")
        row.append(f"{lr['tflops_tri']:.2f}")
        if has_any_repack:
            row.append(
                f"{lr['tflops_tri_e2e']:.2f}"
                if lr["tflops_tri_e2e"] is not None
                else "—"
            )
        row.append(f"{lr['tflops_torch']:.2f}")
        # Winner uses kernel TF/s (not kernel+repack) for consistency with old code.
        row.append("Triton" if lr["tflops_tri"] > lr["tflops_torch"] else "Torch")
        rows.append(row)

    print(_box_table(headers, rows))


def _print_miopen_solver_table(layers: list) -> None:
    """Group layers by MIOpen solver, print one row per solver."""
    from collections import OrderedDict

    solver_layers: dict = OrderedDict()
    for i, lr in enumerate(layers):
        s = lr.get("miopen_solver") or "unknown"
        solver_layers.setdefault(s, []).append(f"L{i}")

    if not any(s != "unknown" for s in solver_layers):
        return  # Nothing detected; skip the table entirely.

    print("\n" + "=" * 80)
    print("MIOpen SOLVER SUMMARY")
    print("=" * 80)

    rows = []
    for solver, ls in solver_layers.items():
        algo = MIOPEN_ALGO_MAP.get(solver, solver)
        layer_str = ", ".join(ls)
        if len(layer_str) > 80:
            layer_str = ", ".join(ls[:10]) + f" ... ({len(ls)} layers total)"
        rows.append([solver, algo, layer_str])
    print(_box_table(("MIOpen Solver", "Algorithm Type", "Used For"), rows))


def _print_overall_perf_table(layers: list, has_any_repack: bool) -> None:
    """Mean/median/aggregate TFLOPS, total time, layer wins."""
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    tri_tf = [lr["tflops_tri"] for lr in layers]
    th_tf = [lr["tflops_torch"] for lr in layers]
    tri_ms = [lr["ms_tri"] for lr in layers]
    th_ms = [lr["ms_torch"] for lr in layers]

    # Aggregate = sum(flops) / sum(time)
    sum_flops = sum(lr["flops"] for lr in layers)
    sum_time_tri = sum(lr["ms_tri"] * 1e-3 for lr in layers)
    sum_time_th = sum(lr["ms_torch"] * 1e-3 for lr in layers)
    agg_tri = sum_flops / sum_time_tri / 1e12 if sum_time_tri else 0.0
    agg_th = sum_flops / sum_time_th / 1e12 if sum_time_th else 0.0

    n = len(layers)
    tri_wins = sum(1 for lr in layers if lr["tflops_tri"] > lr["tflops_torch"])

    rows = [
        [
            "Mean TFLOPS (kernel)",
            f"{statistics.mean(tri_tf):.2f}",
            f"{statistics.mean(th_tf):.2f}",
        ],
    ]
    if has_any_repack:
        e2e_tf = [
            (
                lr["tflops_tri_e2e"]
                if lr["tflops_tri_e2e"] is not None
                else lr["tflops_tri"]
            )
            for lr in layers
        ]
        e2e_ms = [
            lr["ms_tri_e2e"] if lr["ms_tri_e2e"] is not None else lr["ms_tri"]
            for lr in layers
        ]
        sum_time_e2e = sum(t * 1e-3 for t in e2e_ms)
        agg_tri_e2e = sum_flops / sum_time_e2e / 1e12 if sum_time_e2e else 0.0
        e2e_wins = sum(1 for lr, ee in zip(layers, e2e_tf) if ee > lr["tflops_torch"])
        rows.append(
            [
                "Mean TFLOPS (kernel+repack)",
                f"{statistics.mean(e2e_tf):.2f}",
                f"{statistics.mean(th_tf):.2f}",
            ]
        )
    rows.append(
        [
            "Median TFLOPS (kernel)",
            f"{statistics.median(tri_tf):.2f}",
            f"{statistics.median(th_tf):.2f}",
        ]
    )
    if has_any_repack:
        rows.append(
            [
                "Median TFLOPS (kernel+repack)",
                f"{statistics.median(e2e_tf):.2f}",
                f"{statistics.median(th_tf):.2f}",
            ]
        )
    rows.append(["Aggregate TFLOPS (kernel)", f"{agg_tri:.2f}", f"{agg_th:.2f}"])
    if has_any_repack:
        rows.append(
            ["Aggregate TFLOPS (kernel+repack)", f"{agg_tri_e2e:.2f}", f"{agg_th:.2f}"]
        )
    rows.append(["Total kernel time (ms)", f"{sum(tri_ms):.2f}", f"{sum(th_ms):.2f}"])
    if has_any_repack:
        rows.append(
            ["Total kernel+repack time (ms)", f"{sum(e2e_ms):.2f}", f"{sum(th_ms):.2f}"]
        )
    rows.append(["Layer wins (kernel)", f"{tri_wins}/{n}", f"{n - tri_wins}/{n}"])
    if has_any_repack:
        rows.append(
            ["Layer wins (kernel+repack)", f"{e2e_wins}/{n}", f"{n - e2e_wins}/{n}"]
        )
    rows.append(
        ["Correctness", f"{sum(1 for lr in layers if lr['correct'])}/{n} passed", "—"]
    )

    print(_box_table(("Metric", "Triton", "PyTorch (MIOpen)"), rows))


# ----------------------------------------------------------------------------
# Sweep mode
# ----------------------------------------------------------------------------


_MODEL_SHAPES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "conv_shapes.json",
)


def _load_model_shapes(model_pattern: str) -> tuple[str, list]:
    """Load conv2d shapes for a model from conv_shapes.json.

    model_pattern: case-insensitive substring matched against model keys.
    Returns (matched_model_name, list of shape tuples in the same form as
    EDGE_CASE_SHAPES — desc is "<model> L<i>").
    """
    with open(_MODEL_SHAPES_PATH) as f:
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
    """Iterate the chosen shape set, then print three summary tables."""
    dtype = _torch_dtype(args.dtype)

    if args.smoke:
        shapes = EDGE_CASE_SHAPES
        print(
            f"# Sweep source: EDGE_CASE_SHAPES ({len(shapes)} shapes) — smoke / "
            "degenerate-path coverage, NOT representative of production workloads"
        )
    else:
        # Default: real-model sweep. --model picks one; absent → resnet50.
        model_name = args.model if args.model else "resnet50"
        try:
            model, shapes = _load_model_shapes(model_name)
            label = f":: {model} ({len(shapes)} layers)"
            if not args.model:
                label += "  (default — pass --model X or --smoke to change)"
            print(f"# Sweep source: conv_shapes.json {label}")
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if args.batch_size is not None:
        if args.batch_size < 1:
            print("ERROR: --batch-size must be >= 1", file=sys.stderr)
            sys.exit(1)
        shapes = [(args.batch_size,) + entry[1:] for entry in shapes]
        print(f"# batch-size override: N={args.batch_size} on all swept shapes")

    print(
        f"# dtype={args.dtype} method={args.method} layout={args.layout} "
        f"miopen_solvers={'on' if args.miopen_solvers else 'off'}"
    )

    # Optional MIOpen solver detection (subprocess; ~60-120s startup)
    if args.miopen_solvers:
        print("# Detecting MIOpen solvers (subprocess; this can take a minute)...")
        precompute_miopen_solvers(shapes, dtype)
        print("# MIOpen solver detection complete.")

    # Bench each shape, collect rows.
    layers = []
    for entry in shapes:
        N, C, H, W, K, R, S, stride, padding, dilation, name = entry
        try:
            r = bench_one_shape(
                N,
                C,
                H,
                W,
                K,
                R,
                S,
                stride,
                padding,
                dilation,
                dtype,
                args.method,
                args.layout,
                bias=not args.no_bias,
                measure_repack=True,
            )
        except Exception as e:
            print(f"  {name:<24} ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        miopen = (
            _get_miopen_solver(N, C, H, W, K, R, S, stride, padding, dilation)
            if args.miopen_solvers
            else ""
        )
        layers.append(
            {
                "name": name,
                "type": _kernel_type_tag(R, S, dilation),
                "shape": _shape_str(N, C, H, W, K, R, S),
                "kernel_name": r["kernel_name"],
                "miopen_solver": miopen,
                "tflops_tri": r["tflops_tri"],
                "tflops_tri_e2e": r["tflops_tri_e2e"],
                "tflops_torch": r["tflops_torch"],
                "ms_tri": r["ms_tri"],
                "ms_tri_e2e": r["ms_tri_e2e"],
                "ms_torch": r["ms_torch"],
                "correct": r["correct"],
                "flops": r["flops"],
            }
        )

    if not layers:
        print("No layers benched (all errored?).", file=sys.stderr)
        return

    has_any_repack = any(lr["ms_tri_e2e"] is not None for lr in layers)

    _print_layer_table(layers, has_any_repack, miopen_enabled=args.miopen_solvers)
    if args.miopen_solvers:
        _print_miopen_solver_table(layers)
    _print_overall_perf_table(layers, has_any_repack)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench_conv2d",
        description="Benchmark aiter.ops.triton.conv.conv2d (single shape or sweep).",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    p.add_argument(
        "--metric",
        type=str.lower,
        choices=["time", "throughput"],
        default="throughput",
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
        "--miopen-solvers",
        "--miopen_solvers",
        action="store_true",
        help="detect MIOpen solver names via a subprocess (sweep mode only; "
        "adds ~60-120s upfront cost)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="sweep mode: load conv2d shapes for this model from "
        "conv_shapes.json (case-insensitive substring match). If omitted, "
        "defaults to resnet50 unless --smoke is passed.",
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


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.single_shape:
        run_single_shape(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
