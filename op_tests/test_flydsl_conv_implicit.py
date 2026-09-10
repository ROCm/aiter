#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf for the FlyDSL implicit-GEMM convolution.

Two sweeps, one table each. The first covers the keyword surface: the entry point
dispatches 1D/2D/3D off the filter rank, so every rank is here along with stride,
padding (incl. "same"), dilation, groups, bias and split-K.

The rest are the shapes two real VAEs run, traced rather than assumed:

* Wan2.1, the causal Conv3d calls carrying a feature cache -- 440 of one encode's
  650 convolutions, and the reason a 3-D kernel is needed at all.
* Wan2.1, the remaining resamplers and pointwise layers, so the three tables
  together account for every convolution one encode performs.
* Qwen-Image, which is the same architecture run at T=1. There the causal Conv3d
  collapses to an exact conv2d and the model never calls a 3-D kernel, so those
  rows are conv2d -- testing them as conv3d would measure something else.

See `docs_flydsl_conv_0826/wan21_vae_conv3d_shapes.md` for the derivation.

Both VAEs downsample space by 8 and their shapes are generated from the input
resolution, so the sweeps take one. Time is not a free variable on the Wan side:
the chunk is 4 frames, the cache is 2, and the two stride-2 time_conv fix the rest,
so clip length only changes how often each shape runs.

Usage::

    python op_tests/test_flydsl_conv_implicit.py
    python op_tests/test_flydsl_conv_implicit.py -c down_0_1 res_96_L0   # hot shapes
    python op_tests/test_flydsl_conv_implicit.py --wan-res 480x832 368x544
    python op_tests/test_flydsl_conv_implicit.py --qwen-res 1024x1024 1328x1328
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
from aiter.test_common import benchmark, checkAllclose, run_perftest

TOL = {"rtol": 2e-2, "atol": 2e-2}
SUPPORTED_GFX = ["gfx942", "gfx950"]


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


# Wan2.1 VAE encoder -- the causal Conv3d calls a T=1 rewrite cannot claim.
#
# AutoencoderKLWan encodes a clip in chunks along time: frame 0 on its own, then
# (T-1)//4 chunks of 4 frames. Every chunk after the first prepends 2 cached
# feature frames, so the leading time slices hold real features instead of the
# zeros a causal pad would supply and conv3d(pad(x), w) == conv2d(x, w[:,:,-1])
# stops holding. Those calls are 67.7% of one encode's convolutions and are what
# vae_conv_video routes to this kernel.
#
# WanCausalConv3d leaves nn.Conv3d's own padding at (0,0,0) and applies its causal
# padding itself before calling down, so the model hands the kernel an already
# padded tensor and stride/padding/dilation are all trivial. The x shapes are
# therefore post-cat-post-pad, matching what F.conv3d receives; time is chunk+2 and
# the spatial dims carry the usual +2 from padding=1.
#
# Only the spatial extents and the call counts move with the input. The time
# extents are fixed by the architecture: a chunk is 4 frames because the VAE
# compresses time 4x, the +2 is CACHE_T which is kT-1 for the 3x3x3 filters, and
# 4 -> 2 -> 1 is the two stride-2 time_conv. So 81 and 17 frames give the same
# shapes and differ only in how many times each runs.
#
# Derived from Wan-AI/Wan2.1-T2V-1.3B-Diffusers vae/config.json (base_dim 96,
# dim_mult [1,2,4,4], temperal_downsample [F,T,T]); see
# docs_flydsl_conv_0826/wan21_vae_conv3d_shapes.md.
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


# Wan2.1 VAE encoder -- everything else one encode runs, so the tables cover all
# 650 calls rather than only the 440 that need the 3-D kernel.
#
# `plain2d` are WanResample's spatial downsamplers. WanResample folds time into
# batch, so these are ordinary nn.Conv2d over N*T images, and the ZeroPad2d((0,1,0,1))
# ahead of them is a separate Sequential entry -- hence the odd H+1 input and
# padding=0. vae_conv_video does replace these (1.19-1.86x at Wan resolutions).
#
# `pointwise` have a spatial kernel of 1: the two time_conv (3x1x1), the residual
# conv_shortcut, mid_block attention's qkv/proj, and quant_conv. min_spatial_kernel
# leaves them on torch because they measure 0.88-1.05x through the kernel; they are
# here to keep that decision backed by numbers instead of assumed. time_conv is the
# interesting one: kT=3 makes K = C*3, so it is only pointwise in space.
#
# The `_t1` rows are the first chunk (1 frame, no cache) and so run once per encode
# whatever the clip length -- a different batch/time extent, hence a separate shape.
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
        # residual shortcuts
        (
            "shortcut_96_t1",
            "pointwise",
            (1, 96, 1, h[1], w[1]),
            (192, 96, 1, 1, 1),
            1,
            0,
            1,
        ),
        (
            "shortcut_96",
            "pointwise",
            (1, 96, 4, h[1], w[1]),
            (192, 96, 1, 1, 1),
            1,
            0,
            c,
        ),
        (
            "shortcut_192_t1",
            "pointwise",
            (1, 192, 1, h[2], w[2]),
            (384, 192, 1, 1, 1),
            1,
            0,
            1,
        ),
        (
            "shortcut_192",
            "pointwise",
            (1, 192, 2, h[2], w[2]),
            (384, 192, 1, 1, 1),
            1,
            0,
            c,
        ),
        # the two temporal downsamples
        (
            "time_conv_192",
            "pointwise",
            (1, 192, 5, h[2], w[2]),
            (192, 192, 3, 1, 1),
            (2, 1, 1),
            0,
            c,
        ),
        (
            "time_conv_384",
            "pointwise",
            (1, 384, 3, h[3], w[3]),
            (384, 384, 3, 1, 1),
            (2, 1, 1),
            0,
            c,
        ),
        # mid-block attention runs on every chunk including the T=1 one
        ("attn_qkv", "pointwise", (1, 384, h[3], w[3]), (1152, 384, 1, 1), 1, 0, c + 1),
        ("attn_proj", "pointwise", (1, 384, h[3], w[3]), (384, 384, 1, 1), 1, 0, c + 1),
        # once per encode, over the whole assembled latent
        (
            "quant_conv",
            "pointwise",
            (1, 32, c + 1, h[3], w[3]),
            (32, 32, 1, 1, 1),
            1,
            0,
            1,
        ),
    ]


# Qwen-Image VAE -- the same architecture as Wan's (identical vae/config.json:
# base_dim 96, dim_mult [1,2,4,4], temperal_downsample [F,T,T]), fine-tuned and run
# at T=1. That degeneracy is the whole story: with one frame and no cache, every
# time slice of the filter but the last multiplies zeros, so lumen_vae_conv rebinds
# forward to an exact 2-D convolution
#
#     conv3d(causal_pad(x), w) == conv2d(x[:,:,0], w[:,:,-1])
#
# and the model never runs a 3-D kernel. These are therefore conv2d shapes with
# ordinary padding=1 -- the input is NOT pre-padded, unlike Wan's T>1 calls above.
# Testing them as conv3d would measure something the model does not do.
#
# Encode and decode together: 52 calls collapsing to 10 shapes, since the encoder
# and decoder resnets meet at the same extents. The 6 plain Conv2d resamplers and 9
# pointwise layers are skipped by vae_conv (image mode), so they are not here.
def qwen_vae_conv2d(height, width):
    """(case, x, weight, calls) for the T=1 rewritten conv2d of encode+decode."""
    h, w = _levels(height), _levels(width)
    # name, level, Cin, Cout, calls per encode+decode
    layers = [
        ("enc_conv_in", 0, 3, 96, 1),
        ("res_96_L0", 0, 96, 96, 10),
        ("down_96_192", 1, 96, 192, 1),
        ("res_192_L1", 1, 192, 192, 9),
        ("down_192_384", 2, 192, 384, 2),
        ("res_384_L2", 2, 384, 384, 8),
        ("res_384_L3", 3, 384, 384, 18),
        ("enc_conv_out", 3, 384, 32, 1),
        ("dec_conv_in", 3, 16, 384, 1),
        ("dec_conv_out", 0, 96, 3, 1),
    ]
    return [
        (name, (1, cin, h[lv], w[lv]), (cout, cin, 3, 3), n)
        for name, lv, cin, cout, n in layers
    ]


def _ref(x, w, bias, rank, **kw):
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    out = fn(x.float(), w.float(), None if bias is None else bias.float(), **kw)
    return out.to(x.dtype)


@benchmark()
def test_conv_implicit(case, rank, xshape, wshape, dtype, kw, ref_kw=None, bias=False):
    torch.manual_seed(0)
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    b = torch.randn(wshape[0], device="cuda", dtype=dtype) if bias else None

    ref = _ref(x, w, b, rank, **(kw if ref_kw is None else ref_kw))
    out, us = run_perftest(flydsl_conv_implicit, x, w, b, **kw, num_rotate_args=1)
    err = checkAllclose(ref, out, msg=f"{case}: ", **TOL)
    return {"case": case, "dtype": str(dtype), "us": us, "err": err}


def _bench_vs_torch(case, xshape, wshape, dtype, calls, stride=1, padding=0):
    """One model shape, both kernels, as the row of a summary table.

    Rank comes off the filter. Every VAE convolution here carries a bias. torch is
    a candidate rather than only the reference: MIOpen is the baseline the Lumen
    patch replaces, so its number belongs in the table.
    """
    torch.manual_seed(0)
    rank = len(wshape) - 2
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    b = torch.randn(wshape[0], device="cuda", dtype=dtype)

    kw = {"stride": stride, "padding": padding}
    ref = _ref(x, w, b, rank, **kw)

    m = ref.shape[0]
    for d in ref.shape[2:]:
        m *= d
    n = wshape[0]
    k = wshape[1]
    for d in wshape[2:]:
        k *= d
    flops = 2 * m * n * k
    nbytes = (x.numel() + w.numel() + ref.numel()) * x.element_size()

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
        # Weight the row by how often one encode runs this shape, so the column
        # sums to that bucket's share of an encode instead of to a single call.
        ret[f"{name} ms/encode"] = us * calls / 1e3
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
def test_qwen_vae_conv2d(case, res, xshape, wshape, dtype, calls):
    # The T=1 rewrite hands conv2d the unpadded input and the module's spatial pad.
    return _bench_vs_torch(case, xshape, wshape, dtype, calls, padding=1)


def summarize(title, rows):
    aiter.logger.info("%s:\n%s", title, pd.DataFrame(rows).to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_conv_implicit unsupported on %s; skipping", get_gfx()
        )
        return

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--dtype", nargs="*", default=["bf16"], choices=["bf16", "fp16"]
    )
    p.add_argument(
        "-c",
        "--cases",
        nargs="*",
        default=None,
        help="restrict the model sweeps to these case names (default: all)",
    )
    p.add_argument(
        "--wan-res",
        nargs="*",
        default=["480x832"],
        help="Wan clip HxW, multiples of 8. 480x832 is what the integration report "
        "benchmarks; 368x544 is what its 8-GPU training run actually feeds the VAE.",
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
        default=["1024x1024"],
        help="Qwen-Image HxW, multiples of 8 (1024x1024, 1328x1328, 1664x928, ...)",
    )
    args = p.parse_args()

    def wanted(case):
        return args.cases is None or case in args.cases

    for name in args.dtype:
        dtype = dtypes.bf16 if name == "bf16" else dtypes.fp16
        rows = []

        # 3D
        x3, w3 = (1, 32, 4, 16, 16), (48, 32, 3, 3, 3)
        rows.append(
            test_conv_implicit("3d_3x3x3_pad1", 3, x3, w3, dtype, {"padding": 1})
        )
        rows.append(
            test_conv_implicit("3d_bias", 3, x3, w3, dtype, {"padding": 1}, bias=True)
        )
        rows.append(
            test_conv_implicit(
                "3d_stride2", 3, x3, w3, dtype, {"stride": 2, "padding": 1}
            )
        )
        rows.append(
            test_conv_implicit(
                "3d_dilation2", 3, x3, w3, dtype, {"padding": 2, "dilation": 2}
            )
        )
        rows.append(
            test_conv_implicit("3d_same", 3, x3, w3, dtype, {"padding": "same"})
        )
        rows.append(
            test_conv_implicit(
                "3d_groups4",
                3,
                (1, 32, 4, 16, 16),
                (48, 8, 3, 3, 3),
                dtype,
                {"padding": 1, "groups": 4},
            )
        )

        # 2D -- the Qwen-Image VAE shape family
        x2, w2 = (1, 96, 64, 64), (96, 96, 3, 3)
        rows.append(test_conv_implicit("2d_3x3_pad1", 2, x2, w2, dtype, {"padding": 1}))
        rows.append(test_conv_implicit("2d_1x1", 2, x2, (96, 96, 1, 1), dtype, {}))
        rows.append(
            test_conv_implicit(
                "2d_splitk2",
                2,
                x2,
                w2,
                dtype,
                {"padding": 1, "splitk": 2},
                ref_kw={"padding": 1},
            )
        )

        # 1D
        rows.append(
            test_conv_implicit(
                "1d_3_pad1", 1, (1, 32, 128), (64, 32, 3), dtype, {"padding": 1}
            )
        )
        summarize(f"flydsl_conv_implicit keyword surface ({name})", rows)

        wan_clips = [
            (f"{r}@{f}f", *parse_res(r), f)
            for r, f in itertools.product(args.wan_res, args.wan_frames)
        ]

        rows = [
            test_wan_vae_conv3d(case, clip, xshape, wshape, dtype, calls)
            for clip, h, w, frames in wan_clips
            for case, xshape, wshape, calls in wan_vae_conv3d(h, w, frames)
            if wanted(case)
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
            if wanted(case)
        ]
        summarize(f"Wan2.1 VAE encode, resamplers and pointwise ({name})", rows)

        rows = [
            test_qwen_vae_conv2d(case, res, xshape, wshape, dtype, calls)
            for res in args.qwen_res
            for case, xshape, wshape, calls in qwen_vae_conv2d(*parse_res(res))
            if wanted(case)
        ]
        summarize(f"Qwen-Image VAE encode+decode, T=1 rewritten conv2d ({name})", rows)


if __name__ == "__main__":
    main()
