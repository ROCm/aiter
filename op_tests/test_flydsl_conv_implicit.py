#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf for the FlyDSL implicit-GEMM convolution.

No arguments: six representative Wan families (conv_in at real 368x544 for an
exact production-tuned lookup check, the other five at compact 64x64), plus six
required keyword regressions and seven unfiltered key/static checks. The exact
check requires a device-matching FlyDSL row; explicit --wan-res overrides disable
that requirement and apply the requested resolutions to all selected Wan families.
Each CI case runs two warmups and ten direct calls, checking every output.
No profiler, run_perftest, torch performance baseline, or fake throughput columns.

Perf mode retains the full keyword/Wan2.1/Qwen-Image matrix and performance tables
from the tuner's CSVs (wan21_vae_bf16_untuned_conv3d.csv and
qwenimage_vae_bf16_untuned_conv3d.csv). Explicit cases replace the default subset;
resolution overrides preserve selected families and never change perf defaults.

Usage::

    python op_tests/test_flydsl_conv_implicit.py
    python op_tests/test_flydsl_conv_implicit.py --mode perf
    python op_tests/test_flydsl_conv_implicit.py -c down_0_1 res_96_L0
    python op_tests/test_flydsl_conv_implicit.py -c conv_in --wan-res 368x544
    python op_tests/test_flydsl_conv_implicit.py --mode perf --qwen-res 1664x928
"""

import argparse
import functools
import itertools
import time
from contextlib import contextmanager
from unittest.mock import patch


def _load_backend():
    """Load GPU dependencies only after argument validation (including --help)."""
    global pd, torch, F, aiter, dtypes, get_gfx
    global conv_kernels, flydsl_conv_implicit, SUPPORTED_GFX, benchmark, run_perftest
    import pandas as pd
    import torch
    import torch.nn.functional as F

    import aiter
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl import conv_kernels, flydsl_conv_implicit
    from aiter.ops.flydsl.conv_kernels import SUPPORTED_GFX
    from aiter.test_common import benchmark, run_perftest


def _perf_row(fn):
    """Keep the existing benchmark row metadata without eager GPU imports."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return benchmark()(fn)(*args, **kwargs)

    return wrapped


TOL = {"rtol": 2e-2, "atol": 2e-2}

# Collect strict failures and assert after the tables. No mismatched element is
# accepted in either mode; CI also retains per-iteration failure details.
ERR_TOL = 0.0
_FAILED = []
# SUPPORTED_GFX is the op's allow-list; skip rather than restating it.


def _levels(v, n=3):
    """A spatial extent and what it becomes after each stride-2 downsample."""
    out = [v]
    for _ in range(n):
        out.append((out[-1] + 1) // 2)
    return out


def parse_res(text):
    h, _, w = text.lower().partition("x")
    h, w = int(h), int(w)
    if h <= 0 or w <= 0 or h % 8 or w % 8:
        # Both VAEs downsample by 8. Off-multiple sizes still run, but the encoder
        # and decoder then land on different extents and the shape set stops
        # collapsing to one row per level, which these tables assume.
        raise ValueError(f"resolution must be a multiple of 8, got {h}x{w}")
    return h, w


# Wan2.1 encoder cached conv3d. Module pads first (padding=0 here); T is chunk+2.
# Spatial extents scale with input; time extents are fixed by the architecture.
def wan_vae_conv3d(height, width, frames):
    """(case, x, weight, calls) for the T>1/cached conv3d of one encode."""
    h, w = _levels(height), _levels(width)
    chunks = (frames - 1) // 4  # the leading T=1 chunk is reducible, not counted
    # name, level, T of the padded input, Cin, Cout, layers sharing the shape
    layers = [
        ("conv_in", 0, 6, 3, 96, 1),
        ("down_0_1", 0, 6, 96, 96, 4),
        ("down_3_in", 1, 6, 96, 192, 1),
        ("down_3_4", 1, 6, 192, 192, 3),
        ("down_6_in", 2, 4, 192, 384, 1),
        ("down_6_7", 2, 4, 384, 384, 3),
        ("down_9_10_mid", 3, 3, 384, 384, 8),
        ("conv_out", 3, 3, 384, 32, 1),
    ]
    return [
        (
            name,
            (1, cin, t, h[lv] + 2, w[lv] + 2),
            (cout, cin, 3, 3, 3),
            n * chunks,
        )
        for name, lv, t, cin, cout, n in layers
    ]


# Wan2.1 encoder remainder: resamplers (H+1, pad=0), time_conv, first-chunk `_t1`.
# True 1x1 pointwise stays on torch; covered by the keyword surface.
def wan_vae_aux(height, width, frames):
    """(case, bucket, x, weight, stride, padding, calls) for the rest of an encode."""
    h, w = _levels(height), _levels(width)
    c = (frames - 1) // 4
    return [
        # WanResample spatial downsamplers, over T images with time folded into batch
        (
            "resample_96_t1",
            "plain2d",
            (1, 96, h[0] + 1, w[0] + 1),
            (96, 96, 3, 3),
            2,
            0,
            1,
        ),
        (
            "resample_96",
            "plain2d",
            (4, 96, h[0] + 1, w[0] + 1),
            (96, 96, 3, 3),
            2,
            0,
            c,
        ),
        (
            "resample_192_t1",
            "plain2d",
            (1, 192, h[1] + 1, w[1] + 1),
            (192, 192, 3, 3),
            2,
            0,
            1,
        ),
        (
            "resample_192",
            "plain2d",
            (4, 192, h[1] + 1, w[1] + 1),
            (192, 192, 3, 3),
            2,
            0,
            c,
        ),
        (
            "resample_384_t1",
            "plain2d",
            (1, 384, h[2] + 1, w[2] + 1),
            (384, 384, 3, 3),
            2,
            0,
            1,
        ),
        (
            "resample_384",
            "plain2d",
            (2, 384, h[2] + 1, w[2] + 1),
            (384, 384, 3, 3),
            2,
            0,
            c,
        ),
        # the two temporal downsamples
        (
            "time_conv_192",
            "time1x1",
            (1, 192, 5, h[2], w[2]),
            (192, 192, 3, 1, 1),
            (2, 1, 1),
            0,
            c,
        ),
        (
            "time_conv_384",
            "time1x1",
            (1, 384, 3, h[3], w[3]),
            (384, 384, 3, 1, 1),
            (2, 1, 1),
            0,
            c,
        ),
    ]


# Wan2.1 decoder shapes not already in the encoder tables (8 rows).
def wan_vae_decode(height, width, frames):
    """(case, bucket, x, weight, stride, padding, calls). ``w`` prefix avoids Qwen -c clashes."""
    h, w = _levels(height), _levels(width)
    c = (frames - 1) // 4
    return [
        (
            "wdec_conv_in",
            "kernel3d",
            (1, 16, 3, h[3] + 2, w[3] + 2),
            (384, 16, 3, 3, 3),
            1,
            0,
            c,
        ),
        (
            "wdec_up_384_L2_t1",
            "plain2d",
            (1, 384, h[2], w[2]),
            (192, 384, 3, 3),
            1,
            1,
            1,
        ),
        ("wdec_up_384_L2", "plain2d", (2, 384, h[2], w[2]), (192, 384, 3, 3), 1, 1, c),
        (
            "wdec_up_384_L1_t1",
            "plain2d",
            (1, 384, h[1], w[1]),
            (192, 384, 3, 3),
            1,
            1,
            1,
        ),
        ("wdec_up_384_L1", "plain2d", (4, 384, h[1], w[1]), (192, 384, 3, 3), 1, 1, c),
        (
            "wdec_up_192_L0_t1",
            "plain2d",
            (1, 192, h[0], w[0]),
            (96, 192, 3, 3),
            1,
            1,
            1,
        ),
        ("wdec_up_192_L0", "plain2d", (4, 192, h[0], w[0]), (96, 192, 3, 3), 1, 1, c),
        (
            "wdec_conv_out",
            "kernel3d",
            (1, 96, 6, h[0] + 2, w[0] + 2),
            (3, 96, 3, 3, 3),
            1,
            0,
            c,
        ),
    ]


# Qwen-Image VAE at T=1: causal convs are exact conv2d with padding=1 (not pre-padded).
def qwen_vae_conv2d(height, width):
    """(case, x, weight, stride, padding, calls) for the conv2d the T=1 path runs."""
    h, w = _levels(height), _levels(width)
    # name, level, extra input extent, Cin, Cout, stride, padding, calls per
    # encode+decode. Ordered to match the untuned CSV row for row.
    layers = [
        ("enc_conv_in", 0, 0, 3, 96, 1, 1, 1),
        ("res_96_L0", 0, 0, 96, 96, 1, 1, 10),
        ("enc_down_96", 0, 1, 96, 96, 2, 0, 1),
        ("down_96_192", 1, 0, 96, 192, 1, 1, 1),
        ("res_192_L1", 1, 0, 192, 192, 1, 1, 9),
        ("enc_down_192", 1, 1, 192, 192, 2, 0, 1),
        ("down_192_384", 2, 0, 192, 384, 1, 1, 2),
        ("res_384_L2", 2, 0, 384, 384, 1, 1, 8),
        ("enc_down_384", 2, 1, 384, 384, 2, 0, 1),
        ("res_384_L3", 3, 0, 384, 384, 1, 1, 18),
        ("enc_conv_out", 3, 0, 384, 32, 1, 1, 1),
        ("dec_conv_in", 3, 0, 16, 384, 1, 1, 1),
        ("dec_up_384_L2", 2, 0, 384, 192, 1, 1, 1),
        ("dec_up_384_L1", 1, 0, 384, 192, 1, 1, 1),
        ("dec_up_192_L0", 0, 0, 192, 96, 1, 1, 1),
        ("dec_conv_out", 0, 0, 96, 3, 1, 1, 1),
    ]
    return [
        (name, (1, cin, h[lv] + e, w[lv] + e), (cout, cin, 3, 3), st, pad, n)
        for name, lv, e, cin, cout, st, pad, n in layers
    ]


# Keyword surface: one row per feature of the entry point's keyword API.
# 2D reuses the Qwen-Image VAE shape family. splitk needs a ref_kw because torch
# has no such argument.
_X3, _W3 = (1, 32, 4, 16, 16), (48, 32, 3, 3, 3)
_X2, _W2 = (1, 96, 64, 64), (96, 96, 3, 3)
# kH x kW reuse at 8x32 output blocks, forced rather than taken from a table.
_KH8 = {"tile": (256, 96, 4, 1), "block_rows": 8}
_NHWC = {"input_layout": "NHWC", "output_layout": "NHWC"}
_NDHWC = {"input_layout": "NDHWC", "output_layout": "NDHWC"}

# (case, rank, xshape, wshape, kw, ref_kw, bias)
KW_CASES = [
    ("3d_3x3x3_pad1", 3, _X3, _W3, {"padding": 1}, None, False),
    ("3d_bias", 3, _X3, _W3, {"padding": 1}, None, True),
    ("3d_stride2", 3, _X3, _W3, {"stride": 2, "padding": 1}, None, False),
    ("3d_dilation2", 3, _X3, _W3, {"padding": 2, "dilation": 2}, None, False),
    ("3d_same", 3, _X3, _W3, {"padding": "same"}, None, False),
    # One row per padding_mode: each takes its own branch of the kernel's tap
    # coordinate fixup, and only "zeros" routes through the OOB sentinel.
    (
        "3d_pad_reflect",
        3,
        _X3,
        _W3,
        {"padding": 1, "padding_mode": "reflect"},
        None,
        False,
    ),
    (
        "3d_pad_replicate",
        3,
        _X3,
        _W3,
        {"padding": 1, "padding_mode": "replicate"},
        None,
        False,
    ),
    (
        "3d_pad_circular",
        3,
        _X3,
        _W3,
        {"padding": 1, "padding_mode": "circular"},
        None,
        False,
    ),
    ("3d_groups4", 3, _X3, (48, 8, 3, 3, 3), {"padding": 1, "groups": 4}, None, False),
    ("3d_1x1x1", 3, _X3, (48, 32, 1, 1, 1), {}, None, True),
    ("2d_3x3_pad1", 2, _X2, _W2, {"padding": 1}, None, False),
    ("2d_1x1", 2, _X2, (96, 96, 1, 1), {}, None, False),
    (
        "2d_splitk2",
        2,
        (1, 64, 16, 16),
        (64, 64, 3, 3),
        {"padding": 1, "tile": (32, 32, 1, 2), "splitk": 2},
        {"padding": 1},
        False,
    ),
    ("1d_3_pad1", 1, (1, 32, 128), (64, 32, 3), {"padding": 1}, None, False),
    # Independent input/output layouts; torch has no such kwargs (ref_kw).
    (
        "3d_in_ndhwc",
        3,
        _X3,
        _W3,
        {"padding": 1, "input_layout": "NDHWC"},
        {"padding": 1},
        False,
    ),
    (
        "3d_out_ndhwc",
        3,
        _X3,
        _W3,
        {"padding": 1, "output_layout": "NDHWC"},
        {"padding": 1},
        False,
    ),
    (
        "3d_ndhwc",
        3,
        _X3,
        _W3,
        {"padding": 1, "input_layout": "NDHWC", "output_layout": "NDHWC"},
        {"padding": 1},
        True,
    ),
    (
        "2d_nhwc",
        2,
        _X2,
        _W2,
        {"padding": 1, "input_layout": "NHWC", "output_layout": "NHWC"},
        {"padding": 1},
        False,
    ),
    (
        "1d_nwc",
        1,
        (1, 32, 128),
        (64, 32, 3),
        {"padding": 1, "input_layout": "NWC", "output_layout": "NWC"},
        {"padding": 1},
        False,
    ),
    ("3d_valid", 3, _X3, _W3, {"padding": "valid"}, None, False),
    ("3d_unbatched", 3, (32, 4, 16, 16), _W3, {"padding": 1}, None, False),
    # Depthwise: C/groups==1. Slow is expected.
    (
        "3d_depthwise",
        3,
        _X3,
        (32, 1, 3, 3, 3),
        {"padding": 1, "groups": 32},
        None,
        True,
    ),
    # Pin wider tiles; KW shapes otherwise land on the narrowest _pick_tile.
    (
        "3d_tile_128",
        3,
        _X3,
        _W3,
        {"padding": 1, "tile": (128, 128, 2, 4)},
        {"padding": 1},
        False,
    ),
    (
        "2d_tile_256",
        2,
        _X2,
        _W2,
        {"padding": 1, "tile": (256, 256, 2, 4)},
        {"padding": 1},
        False,
    ),
    # B's last DMA round is half full (384 vecs over 256 threads), on the kW
    # reuse path (Wo >= TILE_M).
    (
        "2d_tile_256x96",
        2,
        (1, 96, 20, 300),
        _W2,
        {"padding": 1, "tile": (256, 96, 4, 1)},
        {"padding": 1},
        True,
    ),
    # kH x kW reuse forced (_KH8), independently of the tuned tables. Ho/Wo
    # leave an H and a W tail; stride 2 must be declined by the plan.
    (
        "2d_block_rows8_nhwc",
        2,
        (1, 96, 37, 70),
        _W2,
        {"padding": 1, **_KH8, **_NHWC},
        {"padding": 1},
        True,
    ),
    (
        "3d_block_rows8_ndhwc",
        3,
        (1, 96, 4, 20, 45),
        (96, 96, 3, 3, 3),
        {"padding": 1, **_KH8, **_NDHWC},
        {"padding": 1},
        True,
    ),
    (
        "2d_block_rows8_dil_w2",
        2,
        (1, 96, 37, 70),
        _W2,
        {"padding": (1, 2), "dilation": (1, 2), **_KH8},
        {"padding": (1, 2), "dilation": (1, 2)},
        True,
    ),
    (
        "2d_block_rows8_gated_s2",
        2,
        (1, 96, 41, 81),
        _W2,
        {"stride": 2, "padding": 1, **_KH8},
        {"stride": 2, "padding": 1},
        True,
    ),
]

# -c labels, read back off the sweep builders so the choices cannot drift from the
# shapes. Resolution and clip length scale extents and call counts, never names, so
# any legal argument answers for all of them.
ALL_CASES = (
    [c[0] for c in KW_CASES]
    + [c[0] for c in wan_vae_conv3d(64, 64, 5)]
    + [c[0] for c in wan_vae_aux(64, 64, 5)]
    + [c[0] for c in wan_vae_decode(64, 64, 5)]
    + [c[0] for c in qwen_vae_conv2d(64, 64)]
)
# One namespace across all five tables: a label reused by two of them would make -c
# silently select both, and argparse would list it twice.
_DUPE_CASES = sorted({c for c in ALL_CASES if ALL_CASES.count(c) > 1})
assert not _DUPE_CASES, f"duplicate case labels: {_DUPE_CASES}"


CI_WAN_CASES = (
    "conv_in",
    "down_0_1",
    "resample_96",
    "time_conv_192",
    "wdec_up_192_L0",
    "wdec_conv_out",
)
CI_KW_CASES = (
    "2d_tile_256x96",
    "2d_block_rows8_nhwc",
    "3d_block_rows8_ndhwc",
    "2d_block_rows8_dil_w2",
    "2d_block_rows8_gated_s2",
    "2d_splitk2",
)
CI_WARMUP = 2
CI_ITERATIONS = 10


def parse_args(argv=None):
    """Pure configuration: explicit cases override the mode's default subset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("ci", "perf"),
        default="ci",
        help="ci: representative correctness only; perf: full original matrix",
    )
    p.add_argument(
        "-d",
        "--dtype",
        nargs="+",
        default=["bf16"],
        help="the kernel is bf16-only; other dtypes are warned and dropped",
    )
    p.add_argument(
        "-c",
        "--cases",
        nargs="+",
        choices=ALL_CASES,
        metavar="CASE",
        help="replace the default subset with these cases (including Qwen/KW); "
        "key isolation and static reuse are always run",
    )
    p.add_argument(
        "--wan-res",
        nargs="+",
        help="Wan HxW, positive multiples of 8; ci defaults to conv_in@368x544 "
        "(exact tuned check) and five families@64x64; perf defaults to 480x832 "
        "and 368x544; overrides disable the exact check and keep selected families",
    )
    p.add_argument(
        "--wan-frames",
        type=int,
        nargs="+",
        default=[81],
        help="Wan clip lengths (4n+1, at least 5); only call counts change",
    )
    p.add_argument(
        "--qwen-res",
        nargs="+",
        help="Qwen HxW; explicitly enables all Qwen families unless --cases is "
        "given; defaults to 1024x1024 and 1328x1328 in perf or 64x64 in ci",
    )
    args = p.parse_args(argv)
    explicit_qwen = args.qwen_res is not None
    explicit_wan = args.wan_res is not None
    args.wan_res_explicit = explicit_wan
    if args.cases is None:
        args.cases = (
            list(ALL_CASES) if args.mode == "perf" else list(CI_KW_CASES + CI_WAN_CASES)
        )
        if args.mode == "ci" and explicit_qwen:
            args.cases += [c[0] for c in qwen_vae_conv2d(64, 64)]
    args.cases = list(dict.fromkeys(args.cases))
    if args.wan_res is None:
        args.wan_res = ["64x64"] if args.mode == "ci" else ["480x832", "368x544"]
    if args.qwen_res is None:
        args.qwen_res = ["64x64"] if args.mode == "ci" else ["1024x1024", "1328x1328"]
    try:
        for res in args.wan_res + args.qwen_res:
            parse_res(res)
    except ValueError as exc:
        p.error(str(exc))
    if any(f < 5 or (f - 1) % 4 for f in args.wan_frames):
        p.error("--wan-frames must contain clip lengths 4n+1, at least 5")
    if explicit_qwen and not any(c[0] in args.cases for c in qwen_vae_conv2d(64, 64)):
        p.error("--qwen-res has no selected Qwen cases; include one with --cases")
    wan_names = (
        [c[0] for c in wan_vae_conv3d(64, 64, 5)]
        + [c[0] for c in wan_vae_aux(64, 64, 5)]
        + [c[0] for c in wan_vae_decode(64, 64, 5)]
    )
    if explicit_wan and not any(c in args.cases for c in wan_names):
        p.error("--wan-res has no selected Wan cases; include one with --cases")
    return args


def correctness_runner(
    fn, check, synchronize, warmup=CI_WARMUP, iterations=CI_ITERATIONS
):
    """Direct calls only: reuse a reference and check every measured invocation."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    start = time.perf_counter()
    firstcall_ms = 0.0
    for i in range(warmup):
        before = time.perf_counter()
        fn()
        synchronize()
        if i == 0:
            firstcall_ms = (time.perf_counter() - before) * 1e3
    errors = []
    check_ms = 0.0
    checked = 0
    for i in range(iterations):
        try:
            out = fn()
            synchronize()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"iteration {i + 1}: {type(exc).__name__}: {exc}")
            break
        checked += 1
        before = time.perf_counter()
        try:
            check(out)
        except AssertionError as exc:
            errors.append(f"iteration {i + 1}: {exc}")
        check_ms += (time.perf_counter() - before) * 1e3
    return {
        "mode": "ci",
        "warmup": warmup,
        "iterations": checked,
        "wall ms": (time.perf_counter() - start) * 1e3,
        "firstcall ms": firstcall_ms,
        "check ms": check_ms,
        "failure": "; ".join(errors),
    }


def _ref(x, w, bias, rank, padding_mode="zeros", padding=0, **kw):
    """torch reference.

    The functional convs take no ``padding_mode``; ``nn.Conv*`` materializes the
    pad and then convolves with ``padding=0``, so a non-zero mode does the same
    here. torch's pad takes the axes in reverse, innermost first.
    """
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    dtype = x.dtype
    if isinstance(padding, str):
        p = ()
    elif isinstance(padding, int):
        p = (padding,) * rank
    else:
        p = tuple(padding)
    if padding_mode != "zeros" and any(p):
        pads = [v for axis in reversed(p) for v in (axis, axis)]
        x, padding = F.pad(x.float(), tuple(pads), mode=padding_mode), 0
    out = fn(
        x.float(),
        w.float(),
        None if bias is None else bias.float(),
        padding=padding,
        **kw,
    )
    return out.to(dtype)


_CHANNELS_LAST = {1: "NWC", 2: "NHWC", 3: "NDHWC"}


def _channels_last(t, rank):
    """(N,C,*spatial) -> a tensor that is really (N,*spatial,C) in memory.

    The 1D/2D entries ``reshape`` their input up to 5D and the gather reads a flat
    buffer, so a permuted view would not do -- it has to be materialized.
    """
    return t.permute(0, *range(2, rank + 2), 1).contiguous()


def _channels_first(t, rank):
    """(N,*spatial,C) -> (N,C,*spatial), to compare against the NCDHW reference."""
    return t.permute(0, rank + 1, *range(1, rank + 1))


def _roofline(x, w, ref, rank):
    """The convolution's implicit-GEMM (M, N, K), and the FLOPs and bytes it implies.

    N is the full out-channel count while K is C/groups * prod(filter), so M*N*K
    already accounts for a grouped conv summing only over its own group. An
    unbatched call has no N axis in the output, so M is then the spatial extent
    alone.
    """
    m = ref.shape[0] if ref.dim() == rank + 2 else 1
    for d in ref.shape[-rank:]:
        m *= d
    n = w.shape[0]
    k = w.shape[1]
    for d in w.shape[2:]:
        k *= d
    nbytes = (x.numel() + w.numel() + ref.numel()) * x.element_size()
    return m, n, k, 2 * m * n * k, nbytes


def _prepare(
    rank, xshape, wshape, dtype, kw, ref_kw=None, bias=False, round_reference=True
):
    """Create data once; preserve key-isolation's unrounded fp32 reference."""
    torch.manual_seed(0)
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    b = torch.randn(wshape[0], device="cuda", dtype=dtype) if bias else None
    reference_kw = kw if ref_kw is None else ref_kw
    if round_reference:
        ref = _ref(x, w, b, rank, **reference_kw)
    else:
        ref = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank](
            x.float(), w.float(), None if b is None else b.float(), **reference_kw
        )
    cl = _CHANNELS_LAST[rank]
    xk = _channels_last(x, rank) if kw.get("input_layout") == cl else x
    return x, w, b, ref, xk, kw.get("output_layout") == cl


def _compare(ref, out, case):
    """Zero mismatched elements, with an explicit finite check on both tensors."""
    ref, out = ref.to(dtypes.fp32), out.to(dtypes.fp32)
    assert ref.shape == out.shape, f"{case}: shape {out.shape} != {ref.shape}"
    assert torch.isfinite(ref).all().item(), f"{case}: non-finite reference"
    assert torch.isfinite(out).all().item(), f"{case}: non-finite output"
    # Keep checkAllclose(ref, out)'s original relative-tolerance direction.
    mismatches = int((~torch.isclose(ref, out, **TOL)).sum().item())
    return mismatches / ref.numel()


def _check_output(ref, out, case):
    err = _compare(ref, out, case)
    assert err == 0, f"{case}: {err:.8%} of elements outside {TOL}"


def _assert_host_path(case, param):
    """Inspect the actual runtime compile parameter, not a guessed case name."""
    from aiter.ops.flydsl.kernels.conv.conv3d_gfx950_utils import (
        make_conv_geometry,
        make_launch_grid,
        make_tile_config,
    )
    from aiter.ops.flydsl.kernels.conv.conv3d_im2col import make_im2col_plan

    cfg = make_tile_config(param.tile)
    geom = make_conv_geometry(param)
    plan = make_im2col_plan(param, geom, cfg)
    grid = make_launch_grid(param, geom, cfg, plan.out_block)
    if case == "2d_splitk2":
        assert (
            param.splitk == 2 and grid.use_splitk and grid.splitk == 2
        ), "splitk=2 must survive host resolution and reach the launch grid"
        assert not plan.kw_reuse and not plan.kh_reuse
    elif case == "2d_tile_256x96":
        assert plan.kw_reuse and not plan.kh_reuse, "expected kW reuse"
    elif case == "2d_block_rows8_gated_s2":
        assert not plan.kh_reuse and not plan.kw_reuse, "stride 2 must gate reuse"
    else:
        assert plan.kh_reuse and not plan.kw_reuse, "expected kH x kW reuse"
        assert plan.out_block == (8, 32), "expected an 8x32 output patch"
    if case == "2d_block_rows8_static":
        assert not param.dyn_hw, "static regression must use static input extents"
    return (
        f"kw_reuse={plan.kw_reuse}, kh_reuse={plan.kh_reuse}, "
        f"splitk={grid.splitk}, use_splitk={grid.use_splitk}, dyn_hw={param.dyn_hw}"
    )


@contextmanager
def _capture_host_path(case):
    """Spy without replacing params or clearing/populating artificial cache keys."""
    notes = []
    if case not in CI_KW_CASES and case != "2d_block_rows8_static":
        yield notes
        return
    compile_fn = conv_kernels.compile_conv3d_implicit

    def checked_compile(param):
        note = _assert_host_path(case, param)
        if note not in notes:
            notes.append(note)
        return compile_fn(param)

    with patch.object(conv_kernels, "compile_conv3d_implicit", checked_compile):
        yield notes
        assert notes, f"{case}: no implicit compile parameter was observed"


@contextmanager
def _capture_tuned_path(required):
    """Observe exact production lookup and the compile parameter it selects."""
    notes = []
    if not required:
        yield notes
        return
    log_fn = conv_kernels._log_tuned_lookup
    compile_fn = conv_kernels.compile_conv3d_implicit
    hits = []

    def checked_lookup(table, dev, key, hit, borrowed=False):
        assert hit is not None and not borrowed, (
            f"production conv_in requires an exact tuned row for device={dev}; "
            "use an explicit --wan-res override for generic correctness"
        )
        hits.append(hit)
        note = f"exact tuned hit: device={dev}, tile={hit[0]}, wgm={hit[1]}"
        if note not in notes:
            notes.append(note)
        log_fn(table, dev, key, hit, borrowed=borrowed)

    def checked_compile(param):
        assert hits and tuple(param.tile) == tuple(
            hits[-1][0]
        ), "compile parameter must use the tile returned by the exact lookup"
        return compile_fn(param)

    with patch.object(conv_kernels, "_log_tuned_lookup", checked_lookup), patch.object(
        conv_kernels, "compile_conv3d_implicit", checked_compile
    ):
        yield notes
        assert hits, "production conv_in did not perform a tuned lookup"


def _ci_case(
    case,
    rank,
    xshape,
    wshape,
    dtype,
    kw,
    ref_kw=None,
    bias=False,
    round_reference=True,
    require_tuned=False,
):
    _, w, b, ref, xk, out_cl = _prepare(
        rank, xshape, wshape, dtype, kw, ref_kw, bias, round_reference
    )

    def check(out):
        if out_cl:
            out = _channels_first(out, rank)
        _check_output(ref, out, case)

    with _capture_host_path(case) as notes, _capture_tuned_path(require_tuned) as tuned:
        ret = correctness_runner(
            lambda: flydsl_conv_implicit(xk, w, b, **kw),
            check,
            torch.cuda.synchronize,
        )
    return {
        "case": case,
        "x": str(xshape),
        "weight": str(wshape),
        "host path": "; ".join(notes + tuned) or "not asserted",
        **ret,
    }


@_perf_row
def test_conv_implicit(case, rank, xshape, wshape, dtype, kw, ref_kw=None, bias=False):
    x, w, b, ref, xk, out_cl = _prepare(rank, xshape, wshape, dtype, kw, ref_kw, bias)
    m, n, k, flops, nbytes = _roofline(x, w, ref, rank)

    # Keyword surface, so torch is the reference only. The model-shape sweeps below
    # are where it also runs as a candidate, because there MIOpen is the baseline.
    candidates = {"flydsl": lambda: flydsl_conv_implicit(xk, w, b, **kw)}

    ret = {"gfx": get_gfx(), "M": m, "N": n, "K": k}
    for name, fn in candidates.items():
        with _capture_host_path(case):
            out, us = run_perftest(fn, num_rotate_args=1)
        if out_cl:
            out = _channels_first(out, rank)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = _compare(ref, out, f"{case} {name}")
    return ret


def _bench_vs_torch(
    case, xshape, wshape, dtype, calls, stride=1, padding=0, per="encode"
):
    """One model shape, both kernels, as the row of a summary table.

    Rank comes off the filter. Every VAE convolution here carries a bias. torch is
    a candidate rather than only the reference: MIOpen is the baseline the Lumen
    patch replaces, so its number belongs in the table. ``per`` names the pass the
    call count belongs to, so a decode table does not claim to weight an encode.
    """
    rank = len(wshape) - 2
    kw = {"stride": stride, "padding": padding}
    x, w, b, ref, _, _ = _prepare(rank, xshape, wshape, dtype, kw, bias=True)
    m, n, k, flops, nbytes = _roofline(x, w, ref, rank)

    torch_conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    candidates = {
        "torch": lambda: torch_conv(x, w, b, **kw),
        "flydsl": lambda: flydsl_conv_implicit(x, w, b, **kw),
    }

    ret = {"gfx": get_gfx(), "M": m, "N": n, "K": k}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        # Weight the row by how often one pass runs this shape, so the column sums
        # to that bucket's share of the pass instead of to a single call.
        ret[f"{name} ms/{per}"] = us * calls / 1e3
        ret[f"{name} err"] = _compare(ref, out, f"{case} {name}")
    return ret


@_perf_row
def test_wan_vae_conv3d(case, clip, xshape, wshape, dtype, calls):
    # WanCausalConv3d padded x itself, so padding=0 / stride=1.
    return _bench_vs_torch(case, xshape, wshape, dtype, calls)


@_perf_row
def test_wan_vae_aux(case, clip, bucket, xshape, wshape, stride, padding, dtype, calls):
    return _bench_vs_torch(case, xshape, wshape, dtype, calls, stride, padding)


@_perf_row
def test_wan_vae_decode(
    case, clip, bucket, xshape, wshape, stride, padding, dtype, calls
):
    # calls weight a decode, not an encode, so the column is named for it.
    return _bench_vs_torch(
        case, xshape, wshape, dtype, calls, stride, padding, per="decode"
    )


@_perf_row
def test_qwen_vae_conv2d(case, res, xshape, wshape, stride, padding, dtype, calls):
    # The T=1 rewrite hands conv2d the unpadded input and the module's spatial pad;
    # the resamplers instead get a pre-padded input at stride 2, so both are swept.
    return _bench_vs_torch(case, xshape, wshape, dtype, calls, stride, padding)


# Same compile-time constants, different input extents (stride-2 floor-div).
# StaticInputExtents must keep these as separate artifacts.
_KEY_ISOLATION_PAIRS = (
    ("depth", (1, 32, 8, 16, 16), (1, 32, 7, 16, 16)),
    ("height", (1, 32, 4, 32, 16), (1, 32, 4, 31, 16)),
    ("width", (1, 32, 4, 16, 34), (1, 32, 4, 16, 33)),
)


def _execute_ci(case, *args, **kwargs):
    """Keep failures beside their case and continue to the final strict summary."""
    start = time.perf_counter()
    try:
        return _ci_case(case, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "case": case,
            "mode": "ci",
            "iterations": 0,
            "wall ms": (time.perf_counter() - start) * 1e3,
            "failure": f"{type(exc).__name__}: {exc}",
        }


def test_artifact_key_isolation(dtype, mode="perf"):
    """Preserve the six shapes and the original unrounded fp32 reference."""
    rows = []
    for axis, big, small in _KEY_ISOLATION_PAIRS:
        for which, xshape in (("larger", big), ("smaller", small)):
            kw = {"stride": 2, "padding": 1}
            wshape = (32, xshape[1], 3, 3, 3)
            case = f"{axis}_{which}_{'x'.join(str(v) for v in xshape[2:])}"
            if mode == "ci":
                rows.append(
                    _execute_ci(
                        case, 3, xshape, wshape, dtype, kw, round_reference=False
                    )
                )
            else:
                _, w, b, ref, xk, _ = _prepare(
                    3, xshape, wshape, dtype, kw, round_reference=False
                )
                out = flydsl_conv_implicit(xk, w, b, **kw)
                rows.append(
                    {
                        "case": case,
                        "gfx": get_gfx(),
                        "x": str(xshape),
                        "flydsl err": _compare(ref, out, case),
                    }
                )
    return rows


def test_block_rows_static(dtype, mode="perf"):
    """Force static extents for all calls, restoring the module switch on failure."""
    kw = {"padding": 1, **_KH8}
    case = "2d_block_rows8_static"
    with patch.object(conv_kernels, "AITER_CONV3D_DYN_HW", 0):
        if mode == "ci":
            return [
                _execute_ci(
                    case,
                    2,
                    (1, 96, 37, 70),
                    _W2,
                    dtype,
                    kw,
                    ref_kw={"padding": 1},
                    bias=True,
                )
            ]
        _, w, b, ref, xk, _ = _prepare(
            2, (1, 96, 37, 70), _W2, dtype, kw, ref_kw={"padding": 1}, bias=True
        )
        with _capture_host_path(case):
            out = flydsl_conv_implicit(xk, w, b, **kw)
    return [{"case": case, "gfx": get_gfx(), "flydsl err": _compare(ref, out, case)}]


def summarize(title, rows):
    if not rows:  # every case in this sweep was filtered out by --cases
        return
    for row in rows:
        if row.get("failure"):
            _FAILED.append(f"{title}: {row['case']} -- {row['failure']}")
        err = row.get("flydsl err")
        if err is not None and err > ERR_TOL:
            _FAILED.append(f"{title}: {row['case']} -- {err:.2%} of elements mismatch")
    aiter.logger.info("%s:\n%s", title, pd.DataFrame(rows).to_markdown(index=False))


def selected_model_cases(args):
    """Build model rows with unchanged N/C/K/T/filter/stride/padding semantics."""
    rows = []
    for res, frames in itertools.product(args.wan_res, args.wan_frames):
        h, w = parse_res(res)
        clip = f"{res}@{frames}f"
        for case, xs, ws, calls in wan_vae_conv3d(h, w, frames):
            if case in args.cases:
                case_clip = clip
                if (
                    args.mode == "ci"
                    and not args.wan_res_explicit
                    and case == "conv_in"
                ):
                    _, xs, ws, calls = wan_vae_conv3d(368, 544, frames)[0]
                    case_clip = f"368x544@{frames}f (exact tuned check)"
                rows.append(
                    (
                        case,
                        3,
                        xs,
                        ws,
                        {"stride": 1, "padding": 0},
                        calls,
                        "Wan encode",
                        case_clip,
                    )
                )
        for source, builder in (
            ("Wan auxiliary", wan_vae_aux),
            ("Wan decode", wan_vae_decode),
        ):
            for case, _, xs, ws, stride, pad, calls in builder(h, w, frames):
                if case in args.cases:
                    rows.append(
                        (
                            case,
                            len(ws) - 2,
                            xs,
                            ws,
                            {"stride": stride, "padding": pad},
                            calls,
                            source,
                            clip,
                        )
                    )
    for res in args.qwen_res:
        for case, xs, ws, stride, pad, calls in qwen_vae_conv2d(*parse_res(res)):
            if case in args.cases:
                rows.append(
                    (
                        case,
                        2,
                        xs,
                        ws,
                        {"stride": stride, "padding": pad},
                        calls,
                        "Qwen",
                        res,
                    )
                )
    return rows


def _run_ci(args, dtype):
    start = time.perf_counter()
    rows = [
        _execute_ci(case, rank, xs, ws, dtype, kw, ref_kw, bias)
        for case, rank, xs, ws, kw, ref_kw, bias in KW_CASES
        if case in args.cases
    ]
    for case, rank, xs, ws, kw, calls, source, clip in selected_model_cases(args):
        row = _execute_ci(
            case,
            rank,
            xs,
            ws,
            dtype,
            kw,
            bias=True,
            require_tuned=case == "conv_in" and not args.wan_res_explicit,
        )
        row.update({"source": source, "clip": clip, "calls/pass": calls})
        rows.append(row)
    rows += test_artifact_key_isolation(dtype, mode="ci")
    rows += test_block_rows_static(dtype, mode="ci")
    summarize("FlyDSL convolution correctness (mode=ci)", rows)
    aiter.logger.info(
        "mode=ci executed cases=%d checked iterations=%d warmup/case=%d "
        "iterations/case=%d wall=%.3fs failed cases=%d; no performance measurement",
        len(rows),
        sum(row["iterations"] for row in rows),
        CI_WARMUP,
        CI_ITERATIONS,
        time.perf_counter() - start,
        sum(bool(row.get("failure")) for row in rows),
    )


def main(argv=None):
    args = parse_args(argv)
    _load_backend()
    _FAILED.clear()
    args.dtype = [dtypes.str2Dtype(d) for d in args.dtype]

    # The kernel asserts bf16 on entry, so drop the rest here instead of letting a
    # sweep die halfway through.
    sweep_dtypes = [d for d in args.dtype if d == dtypes.bf16]
    for d in args.dtype:
        if d != dtypes.bf16:
            aiter.logger.warning("flydsl_conv_implicit is bf16-only; skipping %s", d)
    if not sweep_dtypes:
        raise ValueError("no supported dtype selected; convolution requires bf16")
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mode=%s flydsl_conv_implicit unsupported on %s; skipped, "
            "executed cases=0",
            args.mode,
            get_gfx(),
        )
        return

    for dtype in dict.fromkeys(sweep_dtypes):
        name = str(dtype).split(".")[-1]
        if args.mode == "ci":
            aiter.logger.info(
                "mode=ci representative correctness; Wan resolutions=%s; "
                "conv_in exact production-tuned check=%s (368x544 when enabled)",
                args.wan_res,
                "conv_in" in args.cases and not args.wan_res_explicit,
            )
            _run_ci(args, dtype)
            continue
        aiter.logger.info("mode=perf full selected matrix with performance statistics")

        rows = [
            test_conv_implicit(case, rank, xs, ws, dtype, kw, ref_kw, bias)
            for case, rank, xs, ws, kw, ref_kw, bias in KW_CASES
            if case in args.cases
        ]
        summarize(f"flydsl_conv_implicit keyword surface ({name})", rows)

        wan_clips = [
            (f"{r}@{f}f", *parse_res(r), f)
            for r, f in itertools.product(args.wan_res, args.wan_frames)
        ]

        rows = [
            test_wan_vae_conv3d(case, clip, xshape, wshape, dtype, calls)
            for clip, h, w, frames in wan_clips
            for case, xshape, wshape, calls in wan_vae_conv3d(h, w, frames)
            if case in args.cases
        ]
        summarize(f"Wan2.1 VAE encode, T>1/cached conv3d ({name})", rows)

        rows = [
            test_wan_vae_aux(
                case, clip, bucket, xshape, wshape, stride, pad, dtype, calls
            )
            for clip, h, w, frames in wan_clips
            for case, bucket, xshape, wshape, stride, pad, calls in wan_vae_aux(
                h, w, frames
            )
            if case in args.cases
        ]
        summarize(f"Wan2.1 VAE encode, resamplers and time_conv ({name})", rows)

        rows = [
            test_wan_vae_decode(
                case, clip, bucket, xshape, wshape, stride, pad, dtype, calls
            )
            for clip, h, w, frames in wan_clips
            for case, bucket, xshape, wshape, stride, pad, calls in wan_vae_decode(
                h, w, frames
            )
            if case in args.cases
        ]
        summarize(f"Wan2.1 VAE decode, shapes encode does not cover ({name})", rows)

        rows = [
            test_qwen_vae_conv2d(case, res, xshape, wshape, stride, pad, dtype, calls)
            for res in args.qwen_res
            for case, xshape, wshape, stride, pad, calls in qwen_vae_conv2d(
                *parse_res(res)
            )
            if case in args.cases
        ]
        summarize(f"Qwen-Image VAE encode+decode, T=1 rewritten conv2d ({name})", rows)

        # Not shape coverage: this one checks that two shapes which differ only
        # in what the key used to drop still get their own artifact. Unfiltered
        # by --cases, since it is cheap and a regression here is silent.
        summarize(
            f"compile-key isolation, same output extents ({name})",
            test_artifact_key_isolation(dtype),
        )
        # Also unfiltered: the static-extent 2D path is otherwise only reached
        # with AITER_CONV3D_DYN_HW=0 set for the whole process.
        summarize(
            f"kH x kW reuse on the static-extent path ({name})",
            test_block_rows_static(dtype),
        )

    # After the tables, so a failure is read next to the numbers that produced it.
    if _FAILED:
        raise AssertionError(
            f"{len(_FAILED)} case(s) outside rtol={TOL['rtol']} atol={TOL['atol']}:\n  "
            + "\n  ".join(_FAILED)
        )


if __name__ == "__main__":
    main()
