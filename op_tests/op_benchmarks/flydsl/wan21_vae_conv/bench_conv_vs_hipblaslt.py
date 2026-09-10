#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter conv3d_implicit vs hipBLASLt on the equivalent GEMM, per Wan2.1 VAE shape.

The video counterpart of ``qwenimage_vae_conv/bench_conv_vs_hipblaslt.py``. Where
Qwen-Image's causal Conv3d degenerates to 2-D at T=1, Wan encodes a clip in chunks
that carry a feature cache, so the leading time slices are real features and a true
3-D kernel is required. Those calls are 440 of one encode's 650 convolutions.

Each convolution runs on its real tensors; hipBLASLt runs the same M/N/K matmul, so
the FLOP count matches. This is not a like-for-like comparison -- hipBLASLt reads an
already-materialised M x K matrix while the conv gathers from the ~27x smaller
source tensor (kT*kH*kW = 27 for a 3x3x3 filter, against 9 for Qwen's 3x3) -- and
the materialisation cost is not charged to hipBLASLt. The gap should therefore be
wider here than in the 2-D figure.

The conv is timed twice: its own kernel, which is what the GEMM comparison is
about, and the whole call, which also pays the NCDHW->NDHWC transpose the kernel
needs when the input is not already channels-last.

Shapes are what F.conv3d actually receives: WanCausalConv3d concatenates the cache
and applies its causal padding itself, then convolves at padding=0. Traced from
Wan-AI/Wan2.1-T2V-1.3B-Diffusers at 81 frames of 480x832; see
`docs_flydsl_conv_0826/wan21_vae_conv3d_shapes.md`. Bias is omitted so the conv and
the matmul do the same arithmetic.

Usage::

    python op_tests/op_benchmarks/flydsl/wan21_vae_conv/bench_conv_vs_hipblaslt.py
"""

import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from aiter.ops.flydsl import flydsl_conv_implicit

# sid, x (post-cache post-pad), weight, stride, padding, calls per encode, kind
#
# The 1x1 pointwise layers (conv_shortcut, attention qkv/proj, quant_conv) are left
# out: their equivalent GEMM is the convolution itself, so the comparison carries no
# information. time_conv is kept -- its spatial kernel is 1 but kT=3 makes K = C*3,
# so it is a real reduction and not pointwise in the sense that matters here.
SHAPES = [
    ("conv_in 3→96", (1, 3, 6, 482, 834), (96, 3, 3, 3, 3), 1, 0, 20, "conv3d"),
    ("96→96 T6@482²", (1, 96, 6, 482, 834), (96, 96, 3, 3, 3), 1, 0, 80, "conv3d"),
    ("96→192 T6@242²", (1, 96, 6, 242, 418), (192, 96, 3, 3, 3), 1, 0, 20, "conv3d"),
    ("192→192 T6@242²", (1, 192, 6, 242, 418), (192, 192, 3, 3, 3), 1, 0, 60, "conv3d"),
    ("192→384 T4@122²", (1, 192, 4, 122, 210), (384, 192, 3, 3, 3), 1, 0, 20, "conv3d"),
    ("384→384 T4@122²", (1, 384, 4, 122, 210), (384, 384, 3, 3, 3), 1, 0, 60, "conv3d"),
    ("384→384 T3@62²", (1, 384, 3, 62, 106), (384, 384, 3, 3, 3), 1, 0, 160, "conv3d"),
    ("384→32 T3@62²", (1, 384, 3, 62, 106), (32, 384, 3, 3, 3), 1, 0, 20, "conv3d"),
    ("96→96 @481² s2", (4, 96, 481, 833), (96, 96, 3, 3), 2, 0, 20, "resample"),
    ("192→192 @241² s2", (4, 192, 241, 417), (192, 192, 3, 3), 2, 0, 20, "resample"),
    ("384→384 @121² s2", (2, 384, 121, 209), (384, 384, 3, 3), 2, 0, 20, "resample"),
    (
        "time_conv 192",
        (1, 192, 5, 120, 208),
        (192, 192, 3, 1, 1),
        (2, 1, 1),
        0,
        20,
        "time",
    ),
    (
        "time_conv 384",
        (1, 384, 3, 60, 104),
        (384, 384, 3, 1, 1),
        (2, 1, 1),
        0,
        20,
        "time",
    ),
]

DEV = torch.device("cuda")


def bench(fn, iters=20, reps=3, key=None):
    """Best-of-N device time. ``key`` restricts to one kernel; None sums them all."""
    best = None
    for _ in range(reps):
        time.sleep(0.4)
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        total = sum(
            ev.self_device_time_total / iters
            for ev in prof.key_averages()
            if ev.device_type == torch.autograd.DeviceType.CUDA
            and ev.self_device_time_total > 0
            and "memcpy" not in ev.key.lower()
            and "memset" not in ev.key.lower()
            and (key is None or key in ev.key)
        )
        if total > 0 and (best is None or total < best):
            best = total
    return best


def time_conv(xshape, wshape, stride, pad):
    x = torch.randn(xshape, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(wshape, device=DEV, dtype=torch.bfloat16)

    def run():
        return flydsl_conv_implicit(x, w, stride=stride, padding=pad)

    return bench(run, key="conv3d_implicit_kernel"), bench(run)


def time_matmul(m, n, k):
    a = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=DEV, dtype=torch.bfloat16)
    return bench(lambda: a @ b)


def gemm_dims(xshape, wshape, stride, pad):
    """(M, N, K) of the implicit GEMM this convolution becomes."""
    rank = len(wshape) - 2
    st = (stride,) * rank if isinstance(stride, int) else tuple(stride)
    pd = (pad,) * rank if isinstance(pad, int) else tuple(pad)
    m = xshape[0]
    for i in range(rank):
        m *= (xshape[2 + i] + 2 * pd[i] - wshape[2 + i]) // st[i] + 1
    k = wshape[1]
    for d in wshape[2:]:
        k *= d
    return m, wshape[0], k


def main():
    rows = []
    print(
        f"{'case':>18s} {'M':>9s} {'N':>5s} {'K':>6s} "
        f"{'kernel':>9s} {'+转置':>9s} {'hipB':>9s} {'比值':>7s}"
    )
    for sid, xshape, wshape, stride, pad, freq, kind in SHAPES:
        m, n, k = gemm_dims(xshape, wshape, stride, pad)
        torch.manual_seed(0)
        rec = {
            "sid": sid,
            "kind": kind,
            "xshape": list(xshape),
            "wshape": list(wshape),
            "freq": freq,
            "M": m,
            "N": n,
            "K": k,
        }
        try:
            t_conv, t_all = time_conv(xshape, wshape, stride, pad)
            torch.cuda.empty_cache()
            t_hip = time_matmul(m, n, k)
            torch.cuda.empty_cache()
            rec.update(
                conv=t_conv,
                conv_all=t_all,
                hip=t_hip,
                ratio=t_hip / t_conv,
                ratio_all=t_hip / t_all,
            )
            print(
                f"{sid:>18s} {m:9d} {n:5d} {k:6d} {t_conv:9.2f} {t_all:9.2f} "
                f"{t_hip:9.2f} {t_hip / t_conv:6.3f}x"
            )
        except Exception as exc:  # noqa: BLE001 -- report and continue the sweep
            msg = (str(exc).strip().splitlines() or ["?"])[0][:70]
            rec.update(
                conv=None,
                conv_all=None,
                hip=None,
                ratio=None,
                ratio_all=None,
                error=msg,
            )
            print(f"{sid:>18s} {m:9d} {n:5d} {k:6d}  failed: {msg}")
            torch.cuda.empty_cache()
        rows.append(rec)

    out = Path(__file__).resolve().parent / "conv_vs_hipblaslt.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
