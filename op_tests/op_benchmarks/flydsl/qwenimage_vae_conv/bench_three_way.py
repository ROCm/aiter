#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""conv3d_implicit vs the FlyDSL A16W16 GEMM vs hipBLASLt on the VAE shapes.

The three arms only compare if they share a cache regime, so the two GEMM arms
come from gemm_a16w16_tune.py itself: this runs the tuner over the equivalent
M/N/K and reads the per-candidate profile it writes, which makes the FlyDSL arm
its best candidate and the hipBLASLt arm its best solution index, both timed
with rotated inputs. The conv is then timed here with a matching rotation. A
baseline carried over from another session is not comparable -- two earlier
sweeps disagreed by up to 35% on hipBLASLt.

The conv is reported by its own kernel, excluding the NCHW->NHWC transpose, so
it stays comparable with the two GEMM arms, which are handed an already
materialised M x K matrix. Neither GEMM arm is charged for im2col.

Usage::

    python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/bench_three_way.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from aiter.ops.flydsl import flydsl_conv_implicit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TUNER = ROOT / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py"

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
# gfx950 carries 256 MB of last-level cache; four copies of the largest working
# set here clear it comfortably
NROT = 4


def gemm_shape(cin, cout, hin, stride, pad):
    hout = (hin + 2 * pad - 3) // stride + 1
    return hout * hout, cout, cin * 9


def run_tuner(shapes):
    """Tune every GEMM shape, and keep the best time per backend."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        untuned = tmp / "untuned_gemm.csv"
        profiled = tmp / "profile.csv"
        pd.DataFrame(
            [
                {
                    "M": m,
                    "N": n,
                    "K": k,
                    "bias": False,
                    "dtype": "torch.bfloat16",
                    "outdtype": "torch.bfloat16",
                    "scaleAB": False,
                    "bpreshuffle": False,
                }
                for m, n, k in shapes
            ]
        ).to_csv(untuned, index=False)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        subprocess.run(
            [
                sys.executable,
                str(TUNER),
                "-i",
                str(untuned),
                "-o",
                str(tmp / "tuned.csv"),
                "-o2",
                str(profiled),
                "--libtype",
                "flydsl,hipblaslt",
                "--with-hipblaslt",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        df = pd.read_csv(profiled)

    best = {}
    for (m, n, k), group in df.groupby(["M", "N", "K"]):
        best[(int(m), int(n), int(k))] = {
            lib: (float(rows.us.min()) if len(rows) else None)
            for lib, rows in (
                (lib, group[group.libtype == lib]) for lib in ("flydsl", "hipblaslt")
            )
        }
    return best


def bench(pairs, stride, pad, iters=20, reps=3, key="conv3d_implicit_kernel"):
    best = None
    for _ in range(reps):
        time.sleep(0.4)
        for _ in range(4):
            flydsl_conv_implicit(*pairs[0], stride=stride, padding=pad)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(iters):
                flydsl_conv_implicit(*pairs[i % len(pairs)], stride=stride, padding=pad)
            torch.cuda.synchronize()
        total = sum(
            ev.self_device_time_total / iters
            for ev in prof.key_averages()
            if ev.device_type == torch.autograd.DeviceType.CUDA
            and ev.self_device_time_total > 0
            and key in ev.key
        )
        if total > 0 and (best is None or total < best):
            best = total
    return best


def time_conv(cin, cout, hin, stride, pad):
    pairs = [
        (
            torch.randn((1, cin, hin, hin), device=DEV, dtype=torch.bfloat16),
            torch.randn((cout, cin, 3, 3), device=DEV, dtype=torch.bfloat16),
        )
        for _ in range(NROT)
    ]
    try:
        return bench(pairs, stride, pad), bench(pairs[:1], stride, pad)
    finally:
        del pairs
        torch.cuda.empty_cache()


def main():
    shapes = sorted({gemm_shape(*s[1:6]) for s in SHAPES})
    print(f"tuning {len(shapes)} GEMM shapes behind {len(SHAPES)} convolutions")
    gemm = run_tuner(shapes)

    rows = []
    print(
        f"{'case':>20s} {'M':>9s} {'N':>5s} {'K':>6s} "
        f"{'conv':>9s} {'flyGEMM':>9s} {'hipB':>9s} {'conv/hip':>9s} {'gemm/hip':>9s}"
    )
    for sid, cin, cout, hin, stride, pad, freq in SHAPES:
        m, n, k = gemm_shape(cin, cout, hin, stride, pad)
        torch.manual_seed(0)
        found = gemm.get((m, n, k), {})
        t_fly, t_hip = found.get("flydsl"), found.get("hipblaslt")
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
            t_conv, t_warm = time_conv(cin, cout, hin, stride, pad)
        except Exception as exc:  # noqa: BLE001 -- report and continue the sweep
            msg = (str(exc).strip().splitlines() or ["?"])[0][:60]
            rec.update(conv=None, conv_warm=None, error=msg)
            print(f"{sid:>20s} {m:9d} {n:5d} {k:6d}  failed: {msg}", flush=True)
            rows.append(rec)
            continue
        rec.update(
            conv=t_conv,
            conv_warm=t_warm,
            gemm=t_fly,
            hip=t_hip,
            conv_ratio=(t_hip / t_conv) if t_hip else None,
            gemm_ratio=(t_hip / t_fly) if (t_hip and t_fly) else None,
        )
        fg = f"{t_fly:9.2f}" if t_fly else f"{'不支持':>9s}"
        rg = f"{t_hip / t_fly:8.3f}x" if (t_hip and t_fly) else f"{'-':>9s}"
        print(
            f"{sid:>20s} {m:9d} {n:5d} {k:6d} {t_conv:9.2f} {fg} "
            f"{t_hip:9.2f} {t_hip / t_conv:8.3f}x {rg}",
            flush=True,
        )
        rows.append(rec)

    out = HERE / "three_way.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
