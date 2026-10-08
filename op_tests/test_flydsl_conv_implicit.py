#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf for the FlyDSL implicit-GEMM convolution.

Keyword surface plus Wan2.1 / Qwen-Image VAE shapes from the tuner's CSVs
(`wan21_vae_bf16_untuned_conv3d.csv`, `qwenimage_vae_bf16_untuned_conv3d.csv`).

Usage::

    python op_tests/test_flydsl_conv_implicit.py
    python op_tests/test_flydsl_conv_implicit.py -c down_0_1 res_96_L0
    python op_tests/test_flydsl_conv_implicit.py --wan-res 480x832 --wan-frames 17
    python op_tests/test_flydsl_conv_implicit.py --qwen-res 1664x928
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_conv_implicit
from aiter.ops.flydsl.conv_kernels import SUPPORTED_GFX
from aiter.test_common import benchmark, checkAllclose, run_perftest

TOL = {"rtol": 2e-2, "atol": 2e-2}

# CI runs this as a script: checkAllclose only raises on non-finite, so collect
# flydsl mismatches and assert once at the end of main(). Bar is zero elements.
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
    if h % 8 or w % 8:
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
    ("2d_splitk2", 2, _X2, _W2, {"padding": 1, "splitk": 2}, {"padding": 1}, False),
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


@benchmark()
def test_conv_implicit(case, rank, xshape, wshape, dtype, kw, ref_kw=None, bias=False):
    torch.manual_seed(0)
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    b = torch.randn(wshape[0], device="cuda", dtype=dtype) if bias else None

    ref = _ref(x, w, b, rank, **(kw if ref_kw is None else ref_kw))
    m, n, k, flops, nbytes = _roofline(x, w, ref, rank)

    # The reference stays channels-first; only what the kernel is handed follows the
    # swept layout, and a channels-last result is rotated back before the compare.
    cl = _CHANNELS_LAST[rank]
    xk = _channels_last(x, rank) if kw.get("input_layout") == cl else x
    out_cl = kw.get("output_layout") == cl

    # Keyword surface, so torch is the reference only. The model-shape sweeps below
    # are where it also runs as a candidate, because there MIOpen is the baseline.
    candidates = {"flydsl": lambda: flydsl_conv_implicit(xk, w, b, **kw)}

    ret = {"gfx": get_gfx(), "M": m, "N": n, "K": k}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_rotate_args=1)
        if out_cl:
            out = _channels_first(out, rank)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32), out.to(dtypes.fp32), msg=f"{case} {name}: ", **TOL
        )
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
    torch.manual_seed(0)
    rank = len(wshape) - 2
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    b = torch.randn(wshape[0], device="cuda", dtype=dtype)

    kw = {"stride": stride, "padding": padding}
    ref = _ref(x, w, b, rank, **kw)
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
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32), out.to(dtypes.fp32), msg=f"{case} {name}: ", **TOL
        )
    return ret


@benchmark()
def test_wan_vae_conv3d(case, clip, xshape, wshape, dtype, calls):
    # WanCausalConv3d padded x itself, so padding=0 / stride=1.
    return _bench_vs_torch(case, xshape, wshape, dtype, calls)


@benchmark()
def test_wan_vae_aux(case, clip, bucket, xshape, wshape, stride, padding, dtype, calls):
    return _bench_vs_torch(case, xshape, wshape, dtype, calls, stride, padding)


@benchmark()
def test_wan_vae_decode(
    case, clip, bucket, xshape, wshape, stride, padding, dtype, calls
):
    # calls weight a decode, not an encode, so the column is named for it.
    return _bench_vs_torch(
        case, xshape, wshape, dtype, calls, stride, padding, per="decode"
    )


@benchmark()
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


def test_artifact_key_isolation(dtype):
    """One row per shape; a shape that borrowed another's artifact shows up here."""
    rows = []
    for axis, big, small in _KEY_ISOLATION_PAIRS:
        for which, xshape in (("larger", big), ("smaller", small)):
            torch.manual_seed(0)
            x = torch.randn(xshape, device="cuda", dtype=dtype)
            w = torch.randn((32, xshape[1], 3, 3, 3), device="cuda", dtype=dtype)
            kw = {"stride": 2, "padding": 1}
            out = flydsl_conv_implicit(x, w, **kw)
            ref = F.conv3d(x.to(dtypes.fp32), w.to(dtypes.fp32), **kw)
            case = f"{axis}_{which}_{'x'.join(str(v) for v in xshape[2:])}"
            rows.append(
                {
                    "case": case,
                    "gfx": get_gfx(),
                    "x": "x".join(str(v) for v in xshape),
                    "flydsl err": checkAllclose(
                        ref, out.to(dtypes.fp32), msg=f"{case}: ", **TOL
                    ),
                }
            )
    return rows


def summarize(title, rows):
    if not rows:  # every case in this sweep was filtered out by --cases
        return
    for row in rows:
        err = row.get("flydsl err")
        if err is not None and err > ERR_TOL:
            _FAILED.append(f"{title}: {row['case']} -- {err:.2%} of elements mismatch")
    aiter.logger.info("%s:\n%s", title, pd.DataFrame(rows).to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_conv_implicit unsupported on %s; skipping", get_gfx()
        )
        return

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="the kernel is bf16-only; anything else is dropped with a warning",
    )
    p.add_argument(
        "-c",
        "--cases",
        type=str,
        nargs="*",
        choices=ALL_CASES,
        default=ALL_CASES,
        metavar="CASE",
        help=f"case labels to run (default: all {len(ALL_CASES)}). "
        "An unknown label is rejected with the full list.",
    )
    p.add_argument(
        "--wan-res",
        nargs="*",
        default=["480x832", "368x544"],
        help="Wan clip HxW, multiples of 8. 480x832 is what the integration report "
        "benchmarks and 368x544 what its 8-GPU training run feeds the VAE; "
        "wan21_vae_bf16_tuned_conv3d.csv holds both. Any other size runs the same "
        "shapes on a tile borrowed from the nearer of the two.",
    )
    p.add_argument(
        "--wan-frames",
        type=int,
        nargs="*",
        default=[81],
        help="Wan clip lengths. Only the call counts change with this -- the shapes "
        "are the same at 17 and 81 frames.",
    )
    p.add_argument(
        "--qwen-res",
        nargs="*",
        default=["1024x1024", "1328x1328"],
        help="Qwen-Image HxW, multiples of 8. qwenimage_vae_bf16_tuned_conv3d.csv "
        "holds both defaults; any other legal size (1664x928, ...) runs the same 16 "
        "shapes at different extents, on a tile borrowed from the nearer default.",
    )
    args = p.parse_args()
    _FAILED.clear()

    # The kernel asserts bf16 on entry, so drop the rest here instead of letting a
    # sweep die halfway through.
    sweep_dtypes = [d for d in args.dtype if d == dtypes.bf16]
    for d in args.dtype:
        if d != dtypes.bf16:
            aiter.logger.warning("flydsl_conv_implicit is bf16-only; skipping %s", d)
    if not sweep_dtypes:
        return

    for dtype in sweep_dtypes:
        name = str(dtype).split(".")[-1]

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

    # After the tables, so a failure is read next to the numbers that produced it.
    if _FAILED:
        raise AssertionError(
            f"{len(_FAILED)} case(s) outside rtol={TOL['rtol']} atol={TOL['atol']}:\n  "
            + "\n  ".join(_FAILED)
        )


if __name__ == "__main__":
    main()
