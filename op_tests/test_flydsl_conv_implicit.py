#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf for the FlyDSL implicit-GEMM convolution.

Five sweeps, one table each. The first covers the keyword surface: the entry point
dispatches 1D/2D/3D off the filter rank, so every rank is here along with stride,
padding (incl. "same"), dilation, groups, bias and split-K.

The rest are the shapes two real VAEs run, traced rather than assumed:

* Wan2.1, the causal Conv3d calls carrying a feature cache -- 440 of one encode's
  650 convolutions, and the reason a 3-D kernel is needed at all.
* Wan2.1, the spatial resamplers and the two time_conv -- the calls that are not
  cached conv3d but do, or plausibly should, reach this kernel. The 125 genuine
  1x1 calls of an encode never do and are left to the keyword surface.
* Wan2.1 decode, only the 8 shapes it adds: 560 of its 600 cached conv3d calls
  reuse encoder shapes, since the channel ladder is mirrored at the same extents.
* Qwen-Image, which is the same architecture run at T=1. There the causal Conv3d
  collapses to an exact conv2d and the model never calls a 3-D kernel, so those
  rows are conv2d -- testing them as conv3d would measure something else.

The coverage the model tables owe is the tuner's own shape set,
`aiter/configs/model_configs/{wan21,qwenimage}_vae_*_bf16_untuned_conv3d.csv`: at the
default resolutions every row of all four files is a row of a table here. The Wan
files hold only the 8 cached conv3d shapes; the Qwen files also hold the encoder
downsamplers and decoder upsamplers, which is why that table carries stride and
padding columns. See `docs_flydsl_conv_0826/wan21_vae_conv3d_shapes.md` for the
derivation.

Both VAEs downsample space by 8 and their shapes are generated from the input
resolution, so the sweeps take one. Time is not a free variable on the Wan side:
the chunk is 4 frames, the cache is 2, and the two stride-2 time_conv fix the rest,
so clip length only changes how often each shape runs.

Usage::

    python op_tests/test_flydsl_conv_implicit.py
    python op_tests/test_flydsl_conv_implicit.py -c down_0_1 res_96_L0   # hot shapes
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
from aiter.test_common import benchmark, checkAllclose, run_perftest

TOL = {"rtol": 2e-2, "atol": 2e-2}
# TILE_K=32 uses mfma_f32_16x16x32_bf16, which is CDNA4/gfx950 only. gfx942
# (MI300X) has no K=32 BF16 MFMA; a positive allow-list keeps unknown cards
# from silently compiling an illegal instruction.
SUPPORTED_GFX = ["gfx950"]


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


# Wan2.1 VAE encoder -- the calls that are not cached conv3d but still reach, or could
# reach, this kernel. One encode's remaining 125 pointwise calls are a genuine 1x1 in
# every dimension (3 conv_shortcut, attention qkv/proj, quant_conv), measure 0.96-1.20x
# here and are never dispatched, so they are not worth a row; the 5D 1x1x1 path is
# covered for correctness by the keyword surface instead.
#
# `plain2d` are WanResample's spatial downsamplers. WanResample folds time into
# batch, so these are ordinary nn.Conv2d over N*T images, and the ZeroPad2d((0,1,0,1))
# ahead of them is a separate Sequential entry -- hence the odd H+1 input and
# padding=0. vae_conv_video does replace these (0.54-0.86x of torch here). They have no
# row in the Wan tuned config, so they run on _pick_tile's heuristic.
#
# `time1x1` is pointwise in space only: kT=3 makes K = C*3, which is why it behaves
# nothing like the true 1x1 layers it used to be bucketed with. min_spatial_kernel=2
# leaves it on torch, but it measures 0.41-0.66x through the kernel, i.e. ~0.8 ms per
# encode at 480x832 and ~1.3 ms at 368x544 left on the table. Both sides here are
# 12-85 us launch-bound kernels timed L2-warm, and torch's 368x544 number is slower
# than its larger 480x832 one, so re-measure cleanly before moving the gate.
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


# Wan2.1 VAE decoder -- the shapes decode adds that encode does not already cover.
#
# Decode is 797 further conv calls per clip, but 560 of its 600 cached conv3d calls
# land on shapes the encoder tables already carry: the channel ladder is mirrored at
# the same extents, and the decoder's channel reduction happens in the upsampler's
# Conv2d rather than in a resnet, so every resnet on the way up is same-channel. What
# is left is these 8, which is also what the untuned config gained for decode.
#
# `conv_in`/`conv_out` are the cached causal convs with no encoder counterpart: 16
# channels in at the latent extent, 3 out at full extent. Their T follows decode's own
# chunking -- one latent frame plus 2 cache for conv_in, a 4-frame group plus 2 for
# conv_out.
#
# The upsamplers are WanResample's 3x3 Conv2d after the interpolate, so unlike the
# encoder's downsamplers they are stride 1 padding 1 on the unpadded extent. Their
# batch is time folded in again, and time grows on the way up (1 -> 2 -> 4), which is
# why L2 pairs N=1 with N=2 while L1 and L0 pair N=1 with N=4.
def wan_vae_decode(height, width, frames):
    """(case, bucket, x, weight, stride, padding, calls) for one decode.

    The ``w`` prefix is not decoration: -c labels are one global namespace and the
    Qwen table already owns ``dec_conv_in``, ``dec_up_384_L2`` and friends.
    """
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


# Qwen-Image VAE -- the same architecture as Wan's (identical vae/config.json:
# base_dim 96, dim_mult [1,2,4,4], temperal_downsample [F,T,T]), fine-tuned and run
# at T=1. That degeneracy is the whole story: with one frame and no cache, every
# time slice of the filter but the last multiplies zeros, so lumen_vae_conv rebinds
# forward to an exact 2-D convolution
#
#     conv3d(causal_pad(x), w) == conv2d(x[:,:,0], w[:,:,-1])
#
# and the model never runs a 3-D kernel. The causal rows are therefore conv2d with
# ordinary padding=1 -- the input is NOT pre-padded, unlike Wan's T>1 calls above.
# Testing them as conv3d would measure something the model does not do.
#
# Encode and decode together: 58 calls collapsing to 16 shapes, since the encoder and
# decoder resnets meet at the same extents. That is row for row the shape set in
# aiter/configs/model_configs/qwenimage_vae_<res>_bf16_untuned_conv3d.csv, which is
# what the tuner enumerates and therefore the coverage this test owes. Two kinds are
# not causal convs and so do not follow the padding=1 form:
#
# * `enc_down_*` are WanResample's spatial downsamplers, stride 2 over a ZeroPad2d'd
#   input -- hence the +1 extent and padding=0, exactly as in the Wan aux table.
# * `dec_up_*` are the decoder upsamplers' 3x3 Conv2d after the interpolate. They halve
#   channels one level later than the encoder raises them, so 384->192 appears at both
#   L2 and L1 and none of them shares a shape with an encoder row.
#
# The 9 remaining calls of an encode+decode are pointwise (3 conv_shortcut 1x1x1, 4
# attention 1x1, quant_conv + post_quant_conv) and are out of both this table and the
# CSV. The 4 (3,1,1) time_conv never run at T=1 -- the resample time branch is skipped
# with one frame, confirmed by tracing encode+decode on meta device.
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
    # The VAEs' pointwise layers stay on torch, so this row is what keeps the 5D
    # 1x1x1 path -- no pad, no tap fixup, K = C -- under test at all.
    ("3d_1x1x1", 3, _X3, (48, 32, 1, 1, 1), {}, None, True),
    ("2d_3x3_pad1", 2, _X2, _W2, {"padding": 1}, None, False),
    ("2d_1x1", 2, _X2, (96, 96, 1, 1), {}, None, False),
    ("2d_splitk2", 2, _X2, _W2, {"padding": 1, "splitk": 2}, {"padding": 1}, False),
    ("1d_3_pad1", 1, (1, 32, 128), (64, 32, 3), {"padding": 1}, None, False),
    # Channels-last is the kernel's own layout, and the two sides are independent
    # keywords, so each direction takes its own row: a channels-last input skips
    # the pre-transpose, a channels-last output skips the split-K epilogue's
    # transpose and, at n == 1 as here, gives up the vectorized store. torch has
    # no such argument, hence the ref_kw.
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
    # "valid" takes its own early return out of _resolve_padding, ahead of the
    # "same" arithmetic, and torch spells it the same way.
    ("3d_valid", 3, _X3, _W3, {"padding": "valid"}, None, False),
    # An input one rank short goes through the entry's unsqueeze/squeeze, which
    # nothing else here exercises. torch's functional convs accept it too.
    ("3d_unbatched", 3, (32, 4, 16, 16), _W3, {"padding": 1}, None, False),
    # True depthwise: C/groups == 1 pads to the gather's 8-wide vector and K/groups
    # leaves all but one column of the N tile masked, which the entry documents as
    # the one ~0.5x case. Slow is expected; wrong is not, and only a row says which.
    (
        "3d_depthwise",
        3,
        _X3,
        (32, 1, 3, 3, 3),
        {"padding": 1, "groups": 32},
        None,
        True,
    ),
    # The launch config is otherwise chosen by problem size, and every shape
    # above is small enough that _pick_tile lands on the narrowest tile, whose
    # single MFMA column block makes the epilogue's row/col mapping degenerate.
    # The VAE sweeps below do exercise the wider tiles, but only at model
    # resolutions; pin them here so this table stands on its own. torch has no
    # `tile`, hence the ref_kw.
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


def summarize(title, rows):
    if not rows:  # every case in this sweep was filtered out by --cases
        return
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
        help="Wan clip HxW, multiples of 8. The two defaults are the ones with a tuned "
        "config in aiter/configs/model_configs: 480x832 is what the integration report "
        "benchmarks, 368x544 what its 8-GPU training run actually feeds the VAE.",
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
        help="Qwen-Image HxW, multiples of 8. The two defaults are the ones with a "
        "tuned config in aiter/configs/model_configs; any other legal size (1664x928, "
        "...) runs the same 16 shapes at different extents.",
    )
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
