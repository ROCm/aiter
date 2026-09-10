#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf for the FlyDSL implicit-GEMM convolution.

Two sweeps, one table each. The first covers the keyword surface: the entry point
dispatches 1D/2D/3D off the filter rank, so every rank is here along with stride,
padding (incl. "same"), dilation, groups, bias and split-K.

The second is the Wan2.1 video VAE's real encode shapes -- the causal Conv3d calls
that carry a feature cache and so cannot be rewritten as 2-D. Those are 67.7% of
one encode's convolutions and the reason a 3-D kernel is needed at all; see
`docs_flydsl_conv_0826/wan21_vae_conv3d_shapes.md` for how they were derived.

Usage::

    python op_tests/test_flydsl_conv_implicit.py
    python op_tests/test_flydsl_conv_implicit.py -c down_0_1 down_3_4  # hot shapes
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


@benchmark()
def test_wan_vae_conv3d(case, xshape, wshape, dtype, calls):
    torch.manual_seed(0)
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    # Every WanCausalConv3d carries a bias, and the module already padded x, so
    # the real call is padding=0 / stride=1 / dilation=1.
    b = torch.randn(wshape[0], device="cuda", dtype=dtype)

    ref = _ref(x, w, b, 3)
    m = ref.shape[0] * ref.shape[2] * ref.shape[3] * ref.shape[4]
    n = wshape[0]
    k = wshape[1] * wshape[2] * wshape[3] * wshape[4]
    flops = 2 * m * n * k
    nbytes = (x.numel() + w.numel() + ref.numel()) * x.element_size()

    candidates = {
        # torch is a kernel under test here, not just the reference: MIOpen's
        # conv3d is the baseline vae_conv_video replaces.
        "torch": lambda: F.conv3d(x, w, b),
        "flydsl": lambda: flydsl_conv_implicit(x, w, b),
    }

    ret = {"gfx": get_gfx(), "M": m, "N": n, "K": k}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        # Time the whole encode's worth of this shape, so the rows add up to the
        # T>1 share of one encode rather than to a single call.
        ret[f"{name} ms/encode"] = us * calls / 1e3
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32), out.to(dtypes.fp32), msg=f"{case} {name}: ", **TOL
        )
    return ret


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
        default=[c[0] for c in WAN_VAE_CONV3D],
        help="Wan VAE conv3d cases to sweep",
    )
    args = p.parse_args()

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
            if case in args.cases
        ]
        summarize(f"Wan2.1 VAE encode, T>1/cached conv3d ({name})", rows)


if __name__ == "__main__":
    main()
