#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter conv3d_implicit vs hipBLASLt on the equivalent GEMM, per VAE shape.

Each convolution runs on its real tensors; hipBLASLt runs the same M/N/K matmul,
so the FLOP count matches. This is not a like-for-like comparison -- hipBLASLt
reads an already-materialised M x K matrix while the conv gathers from the ~9x
smaller source tensor -- and the materialisation cost is not charged to
hipBLASLt.

The conv is timed twice: its own kernel, which is what the GEMM comparison is
about, and the whole call, which also pays the NCHW->NHWC transpose the kernel
needs when the input is not already channels-last.

Usage::

    python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/bench_conv_vs_hipblaslt.py
"""

import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from aiter.ops.flydsl import flydsl_conv_implicit

# sid, cin, cout, hin, stride, pad, calls per VAE decode
SHAPES = [
    ("3→96 @1024²", 3, 96, 1024, 1, 1, 1),
    ("96→96 @1024²", 96, 96, 1024, 1, 1, 10),
    ("96→192 @512²", 96, 192, 512, 1, 1, 1),
    ("192→192 @512²", 192, 192, 512, 1, 1, 9),
    ("192→384 @256²", 192, 384, 256, 1, 1, 2),
    ("384→384 @256²", 384, 384, 256, 1, 1, 8),
    ("384→384 @128²", 384, 384, 128, 1, 1, 18),
    ("384→32 @128²", 384, 32, 128, 1, 1, 1),
    ("16→384 @128²", 16, 384, 128, 1, 1, 1),
    ("96→3 @1024²", 96, 3, 1024, 1, 1, 1),
    ("96→96 @1025² s2", 96, 96, 1025, 2, 0, 1),
    ("192→192 @513² s2", 192, 192, 513, 2, 0, 1),
    ("384→384 @257² s2", 384, 384, 257, 2, 0, 1),
    ("384→192 @256²", 384, 192, 256, 1, 1, 1),
    ("384→192 @512²", 384, 192, 512, 1, 1, 1),
    ("192→96 @1024²", 192, 96, 1024, 1, 1, 1),
    ("384→384 @166²", 384, 384, 166, 1, 1, 18),
    ("96→96 @1328²", 96, 96, 1328, 1, 1, 10),
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


def time_conv(cin, cout, hin, stride, pad):
    x = torch.randn((1, cin, hin, hin), device=DEV, dtype=torch.bfloat16)
    w = torch.randn((cout, cin, 3, 3), device=DEV, dtype=torch.bfloat16)

    def run():
        return flydsl_conv_implicit(x, w, stride=stride, padding=pad)

    return bench(run, key="conv3d_implicit_kernel"), bench(run)


def time_matmul(m, n, k):
    a = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=DEV, dtype=torch.bfloat16)
    return bench(lambda: a @ b)


def main():
    rows = []
    print(
        f"{'case':>20s} {'M':>9s} {'N':>5s} {'K':>6s} "
        f"{'kernel':>9s} {'+转置':>9s} {'hipB':>9s} {'比值':>7s}"
    )
    for sid, cin, cout, hin, stride, pad, freq in SHAPES:
        hout = (hin + 2 * pad - 3) // stride + 1
        m, n, k = hout * hout, cout, cin * 9
        torch.manual_seed(0)
        rec = {
            "sid": sid,
            "cin": cin,
            "cout": cout,
            "hin": hin,
            "stride": stride,
            "freq": freq,
            "M": m,
            "N": n,
            "K": k,
        }
        try:
            t_conv, t_all = time_conv(cin, cout, hin, stride, pad)
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
                f"{sid:>20s} {m:9d} {n:5d} {k:6d} {t_conv:9.2f} {t_all:9.2f} "
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
            print(f"{sid:>20s} {m:9d} {n:5d} {k:6d}  failed: {msg}")
            torch.cuda.empty_cache()
        rows.append(rec)

    out = Path(__file__).resolve().parent / "conv_vs_hipblaslt.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
