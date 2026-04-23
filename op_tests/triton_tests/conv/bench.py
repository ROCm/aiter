# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from aiter.ops.triton.conv._utils import (
    flops_conv,
    _out_hw,
    _is_1x1_conv,
    _is_3x3_conv,
)
from aiter.ops.triton.conv.conv2d import conv2d_nhwc
from aiter.ops.triton.conv._prepack import _PACK_CACHE_CBLOCKED
from ._registry import METHOD_REGISTRY, ORDERED_METHODS
import aiter.ops.triton.conv.conv2d as _ops_module

_miopen_solver_cache: Dict[tuple, str] = {}


def precompute_miopen_solvers(shapes, dtype=None):
    """Run a single subprocess to detect MIOpen solvers for all shapes at once.

    Args:
        shapes: list of (N, C, H, W, K, R, S, stride, padding, dilation) tuples
        dtype: torch dtype to use for solver detection (default: torch.float16)
    """
    import os
    import re
    import subprocess
    import sys

    global _miopen_solver_cache

    # Deduplicate shapes
    unique = []
    seen = set()
    for N, C, H, W, K, R, S, stride, padding, dilation in shapes:
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, tuple) else (padding, padding)
        d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        key = (N, C, H, W, K, R, S, s_h, s_w, p_h, p_w, d_h, d_w)
        if key not in seen:
            seen.add(key)
            unique.append(key)

    if not unique:
        return

    import torch as _torch

    dtype_str = {
        _torch.float16: "torch.float16",
        _torch.bfloat16: "torch.bfloat16",
    }.get(dtype, "torch.float16")
    # Emit SHAPE_DONE markers to stderr so they interleave with MIOpen's stderr
    # logging — this is what lets us attribute each "Chosen Algorithm:" line
    # to the correct shape index, instead of relying on positional alignment.
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

    # Walk stderr line-by-line. The last "Chosen Algorithm:" seen since the
    # previous SHAPE_DONE marker belongs to the shape whose marker we just hit.
    pending: Optional[str] = None
    attributed: Dict[int, str] = {}
    orphan_markers = 0
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
                orphan_markers += 1
            pending = None

    for idx, solver in attributed.items():
        _miopen_solver_cache[unique[idx]] = solver

    missing = len(unique) - len(attributed)
    if missing > 0:
        print(
            f"[miopen-detect] WARNING: {missing}/{len(unique)} shape(s) have no "
            f"MIOpen solver detected ({orphan_markers} marker(s) had no preceding "
            f"'Chosen Algorithm' line). Common causes: MIOpen log format changed, "
            f"MIOPEN_LOG_LEVEL was overridden, or the shape failed in the subprocess.",
            file=sys.stderr,
        )


def _get_miopen_solver(x, w, b, stride, padding, dilation):
    """Look up MIOpen solver from pre-computed cache."""
    N, C, H, W = x.shape
    K, _, R, S = w.shape
    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
    p_h, p_w = padding if isinstance(padding, tuple) else (padding, padding)
    d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    key = (N, C, H, W, K, R, S, s_h, s_w, p_h, p_w, d_h, d_w)
    return _miopen_solver_cache.get(key, "")


def _bench_ms(fn, warmup=25, rep=100):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        if torch.cuda.is_available():
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            import timeit

            times.append(timeit.timeit(fn, number=1) * 1e3)
    times.sort()
    return times[len(times) // 2]


def run_bench_case(
    suite,
    x,
    w,
    b,
    stride,
    padding,
    dilation,
    activation,
    name,
    layout="nchw",
    method="default",
):
    N, C, H, W = x.shape
    K_out, _, R, S = w.shape
    P, Q = _out_hw(H, W, R, S, stride, padding, dilation)
    total_flops = flops_conv(N, C, K_out, R, S, P, Q)

    if _is_1x1_conv(R, S, dilation):
        kernel_type = "[1x1]"
    elif _is_3x3_conv(R, S):
        kernel_type = "[3x3]"
    else:
        kernel_type = "[general]"

    if layout == "nhwc":
        # Use channels_last input for BOTH Torch and Triton so we compare kernels fairly
        x_bench = x.to(memory_format=torch.channels_last)

        def fn_torch():
            _ = F.conv2d(
                x_bench,
                w,
                b.to(dtype=suite.dtype) if b is not None else None,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

    else:
        x_bench = x

        def fn_torch():
            _ = F.conv2d(
                x_bench,
                w,
                b.to(dtype=suite.dtype) if b is not None else None,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

    # `--method all` only has a meaningful comparison for 3x3 shapes
    # (the alternate kernels — cblocked / Winograd — are 3x3-only). For
    # non-3x3 layers (1x1, 5x5, 7x7, etc.) downgrade to "default" so the
    # per-layer bench row is still produced instead of crashing on
    # METHOD_REGISTRY["all"].
    if method == "all" and not _is_3x3_conv(R, S):
        method = "default"

    if method == "all" and _is_3x3_conv(R, S):
        # Run all 3x3 methods side-by-side and collect for comparison table
        ms_th = _bench_ms(fn_torch, warmup=15, rep=50)
        tf_th = total_flops / (ms_th * 1e-3) / 1e12

        methods = []

        def _make_bench_fn(e):
            def fn():
                return e.kernel_fn(
                    x,
                    w,
                    b,
                    stride,
                    padding,
                    dilation,
                    activation="none",
                    out_dtype=suite.dtype,
                )

            return fn

        for mname in ORDERED_METHODS:
            if mname == "default":
                continue
            entry = METHOD_REGISTRY[mname]
            if entry.guard_fn and not entry.guard_fn(R, S, stride, dilation, C):
                continue
            methods.append((mname, _make_bench_fn(entry)))
        if layout == "nhwc":

            def _fn_nhwc():
                return conv2d_nhwc(
                    x_bench,
                    w,
                    b,
                    stride,
                    padding,
                    dilation,
                    activation="none",
                    out_dtype=suite.dtype,
                )

            methods.append(("nhwc", _fn_nhwc))

        # Build stride string
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        stride_str = f" s{s_h}" if s_h > 1 else ""
        channels_str = f"{C}->{K_out}{stride_str}"

        method_tfs = {}
        for mname, fn in methods:
            ms_tri = _bench_ms(fn, warmup=15, rep=50)
            tf_tri = total_flops / (ms_tri * 1e-3) / 1e12

            def fn_e2e(f=fn):
                _PACK_CACHE_CBLOCKED.clear()
                return f()

            ms_tri_e2e = _bench_ms(fn_e2e, warmup=15, rep=50)
            method_tfs[mname] = tf_tri
            tag = f"{name} {kernel_type} [{mname}]"
            suite.add_bench(
                tag,
                total_flops,
                ms_tri,
                ms_th,
                ms_tri_e2e=ms_tri_e2e,
                x_shape=str(tuple(x.shape)),
                y_shape=f"({N},{K_out},{P},{Q})",
            )

        suite.compare3x3_records.append(
            {
                "layer": name.split()[-1] if " " in name else name,
                "channels": channels_str,
                "methods": method_tfs,
                "torch_tf": tf_th,
            }
        )
        return

    if layout == "nhwc":

        def fn_triton():
            _ = conv2d_nhwc(
                x_bench,
                w,
                b,
                stride,
                padding,
                dilation,
                activation="none",
                out_dtype=suite.dtype,
            )

        tag = f"{name} {kernel_type} [NHWC]"
    else:
        entry = METHOD_REGISTRY[method]

        def fn_triton(e=entry):
            _ = e.kernel_fn(
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation="none",
                out_dtype=suite.dtype,
            )

        tag = (
            f"{name} {kernel_type} {entry.bench_tag}"
            if entry.bench_tag
            else f"{name} {kernel_type}"
        )

    # Warm up once so _last_triton_kernel is set
    fn_triton()
    torch.cuda.synchronize()

    # Capture Triton kernel name
    triton_kernel = getattr(_ops_module, "_last_triton_kernel", "") or ""

    miopen_solver = _get_miopen_solver(
        x if layout != "nhwc" else x.to(memory_format=torch.channels_last),
        w,
        b.to(dtype=suite.dtype) if b is not None else None,
        stride,
        padding,
        dilation,
    )

    ms_tri = _bench_ms(fn_triton, warmup=15, rep=50)
    ms_th = _bench_ms(fn_torch, warmup=15, rep=50)
    ms_tri_e2e = None
    # Only measure repack overhead for kernels that actually use prepacking
    # (3x3, general, winograd). 1x1 kernels take raw weights — no repacking.
    # Only clear the INPUT cache (_PACK_CACHE_CBLOCKED) — weight caches stay warm
    # in real inference since weights are constant across calls.
    needs_repack = layout != "nhwc" and not _is_1x1_conv(R, S, dilation)
    if needs_repack:

        def fn_triton_e2e():
            _PACK_CACHE_CBLOCKED.clear()
            return fn_triton()

        ms_tri_e2e = _bench_ms(fn_triton_e2e, warmup=15, rep=50)
    suite.add_bench(
        tag,
        total_flops,
        ms_tri,
        ms_th,
        ms_tri_e2e=ms_tri_e2e,
        triton_kernel=triton_kernel,
        miopen_solver=miopen_solver,
        x_shape=str(tuple(x.shape)),
        y_shape=f"({N},{K_out},{P},{Q})",
    )
