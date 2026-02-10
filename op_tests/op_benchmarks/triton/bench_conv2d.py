# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for Triton conv2d (NCHW & NHWC) vs. PyTorch conv2d.

Usage:
    python bench_conv2d.py
"""

import sys
import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.conv2d import conv2d as triton_conv2d


# ---------------------------------------------------------------------------
# Benchmark configurations – common conv2d shapes from vision models
# ---------------------------------------------------------------------------
# (batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, stride, padding, groups)
CONFIGS = [
    # ResNet-50 stem
    (1, 3, 224, 224, 64, 7, 7, 2, 3, 1),
    # ResNet-50 blocks
    (1, 64, 56, 56, 64, 3, 3, 1, 1, 1),
    (1, 128, 28, 28, 128, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 512, 7, 7, 512, 3, 3, 1, 1, 1),
    # 1x1 pointwise
    (1, 64, 56, 56, 256, 1, 1, 1, 0, 1),
    (1, 256, 56, 56, 64, 1, 1, 1, 0, 1),
    # Depthwise (MobileNet-style)
    (1, 32, 112, 112, 32, 3, 3, 1, 1, 32),
    (1, 64, 56, 56, 64, 3, 3, 1, 1, 64),
    # Larger batch
    (8, 64, 56, 56, 64, 3, 3, 1, 1, 1),
    (16, 128, 28, 28, 128, 3, 3, 1, 1, 1),
]


def _config_label(cfg):
    B, Ci, H, W, Co, kH, kW, s, p, g = cfg
    return f"B{B}_Ci{Ci}_H{H}xW{W}_Co{Co}_k{kH}x{kW}_s{s}_p{p}_g{g}"


def bench_one(cfg, dtype=torch.float16):
    B, Ci, H, W, Co, kH, kW, s, p, g = cfg

    # NCHW tensors
    x_nchw = torch.randn(B, Ci, H, W, dtype=dtype, device="cuda")
    w = torch.randn(Co, Ci // g, kH, kW, dtype=dtype, device="cuda")
    b = torch.randn(Co, dtype=dtype, device="cuda")

    # NHWC tensors
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    # Compute FLOPs
    Oh = (H + 2 * p - kH) // s + 1
    Ow = (W + 2 * p - kW) // s + 1
    flops = 2.0 * B * Co * Oh * Ow * (Ci // g) * kH * kW

    ms_triton_nchw = triton.testing.do_bench(
        lambda: triton_conv2d(x_nchw, w, b, stride=s, padding=p, groups=g, layout="nchw"),
        warmup=10, rep=50,
    )
    ms_triton_nhwc = triton.testing.do_bench(
        lambda: triton_conv2d(x_nhwc, w, b, stride=s, padding=p, groups=g, layout="nhwc"),
        warmup=10, rep=50,
    )
    ms_torch = triton.testing.do_bench(
        lambda: F.conv2d(x_nchw, w, b, stride=s, padding=p, groups=g),
        warmup=10, rep=50,
    )

    tflops_nchw = flops / ms_triton_nchw * 1e-9
    tflops_nhwc = flops / ms_triton_nhwc * 1e-9
    tflops_torch = flops / ms_torch * 1e-9
    nhwc_vs_nchw = ms_triton_nchw / ms_triton_nhwc

    return ms_triton_nchw, ms_triton_nhwc, ms_torch, tflops_nchw, tflops_nhwc, tflops_torch, nhwc_vs_nchw


def main():
    hdr = (
        f"{'Config':<55} "
        f"{'NCHW ms':>8} {'NHWC ms':>8} {'PT ms':>8} "
        f"{'NCHW TF':>8} {'NHWC TF':>8} {'PT TF':>8} "
        f"{'NHWC/NCHW':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for cfg in CONFIGS:
        label = _config_label(cfg)
        try:
            ms_nc, ms_nh, ms_pt, tf_nc, tf_nh, tf_pt, ratio = bench_one(cfg)
            print(
                f"{label:<55} "
                f"{ms_nc:>8.3f} {ms_nh:>8.3f} {ms_pt:>8.3f} "
                f"{tf_nc:>8.2f} {tf_nh:>8.2f} {tf_pt:>8.2f} "
                f"{ratio:>9.2f}x"
            )
        except Exception as e:
            print(f"{label:<55} ERROR: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
