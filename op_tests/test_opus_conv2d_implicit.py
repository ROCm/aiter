# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Correctness and benchmark tests for opus conv2d implicit GEMM (gfx942).

Usage:
    python test_opus_conv2d_implicit.py           # correctness only
    python test_opus_conv2d_implicit.py --bench   # correctness + benchmark
"""

import sys
import time
import argparse

import torch
import torch.nn.functional as F

# ---- Arch guard ----
from aiter.ops.opus._arch import _detect_arch

_arch_ok, _detected_gfx = _detect_arch({"gfx942"})
if not _arch_ok:
    print(f"[skip] requires gfx942 (detected {_detected_gfx!r})")
    sys.exit(0)

from aiter.ops.opus.conv2d_implicit_op import conv2d_implicit_opus  # noqa: E402


def _torch_ref_nhwc(input_nhwc, weight_nhwc, stride, padding, dilation, groups):
    """Reference: convert NHWC->NCHW, run torch conv2d in fp32, convert back."""
    # input: [N, Hi, Wi, C] -> [N, C, Hi, Wi]
    x = input_nhwc.float().permute(0, 3, 1, 2).contiguous()
    # weight: [K, R, S, Cpg] -> [K, Cpg, R, S]
    w = weight_nhwc.float().permute(0, 3, 1, 2).contiguous()
    out_nchw = F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # [N, K, Ho, Wo] -> [N, Ho, Wo, K]
    return out_nchw.permute(0, 2, 3, 1).contiguous()


def _check_close(got, ref, name, atol=0.2, rtol=0.05):
    """Check correctness using global relative error (same as PoC)."""
    diff = (got.float() - ref.float()).abs()
    maxabs_err = diff.max().item()
    maxabs_ref = ref.float().abs().max().item()
    rel = maxabs_err / max(1e-6, maxabs_ref)
    passed = rel < 8e-3
    print(f"  {name:<24s} rel={rel:.3e}  {'PASS' if passed else 'FAIL'}")
    return passed


def _bench_us(fn, *args, warmup=5, iters=50, **kwargs):
    """Benchmark a function, return average time in microseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms * 1000.0  # microseconds


# ---- Test cases (from PoC) ----
CORRECTNESS_CASES = [
    # (N, C, K, Hi, Wi, R, S, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, group, name)
    (1, 16,  64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, "3x3_basic"),
    (2, 32, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, "3x3_larger"),
    (1, 64,  64,  8,  8, 3, 3, 1, 1, 1, 1, 1, 1, 1, "3x3_small"),
    (2, 32,  64, 16, 16, 3, 3, 1, 1, 2, 2, 1, 1, 1, "3x3_s2"),
    (1, 32,  64, 16, 16, 3, 3, 2, 2, 1, 1, 2, 2, 1, "3x3_d2"),
    (1, 64, 128,  8,  8, 1, 1, 0, 0, 1, 1, 1, 1, 1, "1x1"),
    (2, 16,  32, 32, 32, 5, 5, 2, 2, 1, 1, 1, 1, 1, "5x5"),
    (1, 64,  64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 4, "3x3_group4"),
    (2, 32,  64, 16, 16, 3, 3, 0, 0, 1, 1, 1, 1, 1, "3x3_no_pad"),
]

BENCH_CASES = [
    (2,  64,  64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, "ResNet-3x3-56"),
    (2, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, "ResNet-3x3-28"),
    (2, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, "ResNet-3x3-14"),
    (2, 512, 512,  7,  7, 3, 3, 1, 1, 1, 1, 1, 1, 1, "ResNet-3x3-7"),
    (2,  64, 256, 56, 56, 1, 1, 0, 0, 1, 1, 1, 1, 1, "ResNet-1x1-56"),
    (2, 256, 512, 28, 28, 1, 1, 0, 0, 1, 1, 1, 1, 1, "ResNet-1x1-28"),
]


def run_correctness():
    print("=== Correctness ===\n")
    torch.manual_seed(42)
    passed = 0
    total = len(CORRECTNESS_CASES)

    for N, C, K, Hi, Wi, R, S, ph, pw, sh, sw, dh, dw, g, name in CORRECTNESS_CASES:
        Cpg = C // g
        # NHWC tensors
        x = (torch.randn(N, Hi, Wi, C, device="cuda") * 0.3).to(torch.bfloat16)
        w = (torch.randn(K, R, S, Cpg, device="cuda") * 0.3).to(torch.bfloat16)

        ref = _torch_ref_nhwc(x, w, stride=(sh, sw), padding=(ph, pw),
                              dilation=(dh, dw), groups=g)
        got = conv2d_implicit_opus(x, w, stride=(sh, sw), padding=(ph, pw),
                                   dilation=(dh, dw), groups=g)

        if _check_close(got, ref, name):
            passed += 1

    print(f"\n  correctness: {passed}/{total}\n")
    return passed == total


def run_benchmark():
    print("=== Benchmark (us) ===\n")
    torch.manual_seed(42)

    print(f"  {'TestCase':<20s} {'us':>10s} {'TFLOPS':>10s}")
    print(f"  {'----':<20s} {'------':>10s} {'------':>10s}")

    for N, C, K, Hi, Wi, R, S, ph, pw, sh, sw, dh, dw, g, name in BENCH_CASES:
        Cpg = C // g
        Ho = (Hi + 2 * ph - dh * (R - 1) - 1) // sh + 1
        Wo = (Wi + 2 * pw - dw * (S - 1) - 1) // sw + 1

        x = (torch.randn(N, Hi, Wi, C, device="cuda") * 0.3).to(torch.bfloat16)
        w = (torch.randn(K, R, S, Cpg, device="cuda") * 0.3).to(torch.bfloat16)

        us = _bench_us(conv2d_implicit_opus, x, w,
                       stride=(sh, sw), padding=(ph, pw),
                       dilation=(dh, dw), groups=g)

        flops = 2.0 * N * K * Ho * Wo * Cpg * R * S
        tflops = flops / (us * 1e-6) / 1e12
        print(f"  {name:<20s} {us:10.1f} {tflops:10.2f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmark after correctness")
    args = parser.parse_args()

    all_pass = run_correctness()
    if args.bench:
        run_benchmark()

    sys.exit(0 if all_pass else 1)
