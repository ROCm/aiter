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

Usage::

    python op_tests/test_flydsl_conv_implicit.py
    python op_tests/test_flydsl_conv_implicit.py -c down_0_1 res_96_1024  # hot shapes
"""

import argparse

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

# Wan2.1 VAE encoder -- the causal Conv3d calls a T=1 rewrite cannot claim.
#
# AutoencoderKLWan encodes a clip in chunks along time: frame 0 on its own, then
# ceil((T-1)/4) chunks of 4 frames. Every chunk after the first prepends 2 cached
# feature frames, so the leading time slices hold real features instead of the
# zeros a causal pad would supply and conv3d(pad(x), w) == conv2d(x, w[:,:,-1])
# stops holding. Those calls are 67.7% of one encode's convolutions and are what
# vae_conv_video routes to this kernel.
#
# WanCausalConv3d leaves nn.Conv3d's own padding at (0,0,0) and applies its causal
# padding itself before calling down, so the model hands the kernel an already
# padded tensor and stride/padding/dilation are all trivial. The x shapes below are
# therefore post-cat-post-pad, matching what F.conv3d receives; time is chunk+2 and
# the spatial dims carry the usual +2 from padding=1.
#
# Traced from Wan-AI/Wan2.1-T2V-1.3B-Diffusers vae/config.json at 81 frames of
# 480x832 (base_dim 96, dim_mult [1,2,4,4], temperal_downsample [F,T,T]). The two
# time_conv (3x1x1) and conv_shortcut (1x1x1) layers are pointwise and stay on
# torch, so they are not here.
#
# name        x post-pad (N,C,T,H,W)   weight (K,C,kT,kH,kW)  calls/encode
WAN_VAE_CONV3D = [
    ("conv_in", (1, 3, 6, 482, 834), (96, 3, 3, 3, 3), 20),
    ("down_0_1", (1, 96, 6, 482, 834), (96, 96, 3, 3, 3), 80),
    ("down_3_in", (1, 96, 6, 242, 418), (192, 96, 3, 3, 3), 20),
    ("down_3_4", (1, 192, 6, 242, 418), (192, 192, 3, 3, 3), 60),
    ("down_6_in", (1, 192, 4, 122, 210), (384, 192, 3, 3, 3), 20),
    ("down_6_7", (1, 384, 4, 122, 210), (384, 384, 3, 3, 3), 60),
    ("down_9_10_mid", (1, 384, 3, 62, 106), (384, 384, 3, 3, 3), 160),
    ("conv_out", (1, 384, 3, 62, 106), (32, 384, 3, 3, 3), 20),
]

# Wan2.1 VAE encoder -- everything else one encode runs, so the table covers all
# 650 calls rather than only the 440 that need the 3-D kernel.
#
# `plain2d` are WanResample's spatial downsamplers. WanResample folds time into
# batch, so these are ordinary nn.Conv2d over N*T images, and the ZeroPad2d((0,1,0,1))
# ahead of them is a separate Sequential entry -- hence the odd 481x833 input and
# padding=0. vae_conv_video does replace these (1.19-1.86x at Wan resolutions).
#
# `pointwise` have a spatial kernel of 1: the two time_conv (3x1x1), the residual
# conv_shortcut, mid_block attention's qkv/proj, and quant_conv. min_spatial_kernel
# leaves them on torch because they measure 0.88-1.05x through the kernel; they are
# here to keep that decision backed by numbers instead of assumed.
#
# The `_t1` rows are the first chunk (1 frame, no cache) and so run once per encode
# against 20 for the rest -- a different batch/time extent, hence a separate shape.
#
# name  bucket  x (N,C[,T],H,W)  weight  stride  padding  calls/encode
WAN_VAE_AUX = [
    ("resample_96_t1", "plain2d", (1, 96, 481, 833), (96, 96, 3, 3), 2, 0, 1),
    ("resample_96", "plain2d", (4, 96, 481, 833), (96, 96, 3, 3), 2, 0, 20),
    ("resample_192_t1", "plain2d", (1, 192, 241, 417), (192, 192, 3, 3), 2, 0, 1),
    ("resample_192", "plain2d", (4, 192, 241, 417), (192, 192, 3, 3), 2, 0, 20),
    ("resample_384_t1", "plain2d", (1, 384, 121, 209), (384, 384, 3, 3), 2, 0, 1),
    ("resample_384", "plain2d", (2, 384, 121, 209), (384, 384, 3, 3), 2, 0, 20),
    ("shortcut_96_t1", "pointwise", (1, 96, 1, 240, 416), (192, 96, 1, 1, 1), 1, 0, 1),
    ("shortcut_96", "pointwise", (1, 96, 4, 240, 416), (192, 96, 1, 1, 1), 1, 0, 20),
    (
        "shortcut_192_t1",
        "pointwise",
        (1, 192, 1, 120, 208),
        (384, 192, 1, 1, 1),
        1,
        0,
        1,
    ),
    ("shortcut_192", "pointwise", (1, 192, 2, 120, 208), (384, 192, 1, 1, 1), 1, 0, 20),
    (
        "time_conv_192",
        "pointwise",
        (1, 192, 5, 120, 208),
        (192, 192, 3, 1, 1),
        (2, 1, 1),
        0,
        20,
    ),
    (
        "time_conv_384",
        "pointwise",
        (1, 384, 3, 60, 104),
        (384, 384, 3, 1, 1),
        (2, 1, 1),
        0,
        20,
    ),
    ("attn_qkv", "pointwise", (1, 384, 60, 104), (1152, 384, 1, 1), 1, 0, 21),
    ("attn_proj", "pointwise", (1, 384, 60, 104), (384, 384, 1, 1), 1, 0, 21),
    ("quant_conv", "pointwise", (1, 32, 21, 60, 104), (32, 32, 1, 1, 1), 1, 0, 1),
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
# Traced from Qwen/Qwen-Image vae/config.json at 1024x1024, encode and decode
# together: 52 calls collapsing to 10 distinct shapes (the encoder and decoder
# resnets share most of them). The 6 plain Conv2d resamplers and 9 pointwise layers
# are skipped by vae_conv (image mode), so they are not here.
#
# name  x (N,C,H,W)  weight (K,C,3,3)  calls/encode+decode
QWEN_VAE_CONV2D = [
    ("enc_conv_in", (1, 3, 1024, 1024), (96, 3, 3, 3), 1),
    ("res_96_1024", (1, 96, 1024, 1024), (96, 96, 3, 3), 10),
    ("down_96_192", (1, 96, 512, 512), (192, 96, 3, 3), 1),
    ("res_192_512", (1, 192, 512, 512), (192, 192, 3, 3), 9),
    ("down_192_384", (1, 192, 256, 256), (384, 192, 3, 3), 2),
    ("res_384_256", (1, 384, 256, 256), (384, 384, 3, 3), 8),
    ("res_384_128", (1, 384, 128, 128), (384, 384, 3, 3), 18),
    ("enc_conv_out", (1, 384, 128, 128), (32, 384, 3, 3), 1),
    ("dec_conv_in", (1, 16, 128, 128), (384, 16, 3, 3), 1),
    ("dec_conv_out", (1, 96, 1024, 1024), (3, 96, 3, 3), 1),
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
def test_wan_vae_conv3d(case, xshape, wshape, dtype, calls):
    # WanCausalConv3d padded x itself, so padding=0 / stride=1.
    return _bench_vs_torch(case, xshape, wshape, dtype, calls)


@benchmark()
def test_wan_vae_aux(case, bucket, xshape, wshape, stride, padding, dtype, calls):
    return _bench_vs_torch(case, xshape, wshape, dtype, calls, stride, padding)


@benchmark()
def test_qwen_vae_conv2d(case, xshape, wshape, dtype, calls):
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

        rows = [
            test_wan_vae_conv3d(case, xshape, wshape, dtype, calls)
            for case, xshape, wshape, calls in WAN_VAE_CONV3D
            if wanted(case)
        ]
        summarize(f"Wan2.1 VAE encode, T>1/cached conv3d ({name})", rows)

        rows = [
            test_wan_vae_aux(case, bucket, xshape, wshape, stride, pad, dtype, calls)
            for case, bucket, xshape, wshape, stride, pad, calls in WAN_VAE_AUX
            if wanted(case)
        ]
        summarize(f"Wan2.1 VAE encode, resamplers and pointwise ({name})", rows)

        rows = [
            test_qwen_vae_conv2d(case, xshape, wshape, dtype, calls)
            for case, xshape, wshape, calls in QWEN_VAE_CONV2D
            if wanted(case)
        ]
        summarize(f"Qwen-Image VAE encode+decode, T=1 rewritten conv2d ({name})", rows)


if __name__ == "__main__":
    main()
