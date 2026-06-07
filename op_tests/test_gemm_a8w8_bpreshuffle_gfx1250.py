# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + benchmark for the gfx1250 (WMMA) a8w8 bpreshuffle GEMM.

Main-driven like op_tests/test_gemm_a8w8.py (aiter runs op_tests via python3,
not pytest). Sweeps -mnk x -d, plus a feature pass (ragged M, strided A/C,
split-k, warp/cluster configs, vendored OOB descriptor). Each row times the
WMMA backend with run_perftest(use_cuda_event=True) -- the FlyDSL kernel is
JIT-dispatched (hipModuleLaunchKernel), which torch.profiler can miss -- and
checks correctness vs the dense reference (err / pass columns). Skipped off
gfx1250.
"""

import argparse
import os
import sys

import pandas as pd
import torch

import aiter
from aiter.utility import dtypes
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import run_preshuffle_gemm_a8_gfx1250

TEST_NUM_ITERS = 50
_DTYPE = {"bf16": torch.bfloat16, "f16": torch.float16}


def _dt(dtype):
    return "bf16" if dtype == torch.bfloat16 else "f16"


def _quant(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_scale = aiter.pertoken_quant(a, quant_dtype=dtypes.fp8)
    bq, b_scale = aiter.pertoken_quant(b, quant_dtype=dtypes.fp8)
    return aq, bq, a_scale, b_scale


def _ref(aq, bq, a_scale, b_scale, dtype):
    a_f = aq.to(torch.float32) * a_scale.to(torch.float32)
    b_f = bq.to(torch.float32) * b_scale.to(torch.float32)
    return (a_f @ b_f.t()).to(dtype)


def _bound(split_k):
    return 0.10 if split_k > 1 else 0.05


@perftest(num_iters=TEST_NUM_ITERS, num_rotate_args=1, use_cuda_event=True)
def run_gemm_flydsl(
    xq,
    wq_sh,
    x_scale,
    w_scale,
    out,
    tile,
    num_buffers,
    split_k,
    m_warp,
    n_warp,
    cluster,
):
    run_preshuffle_gemm_a8_gfx1250(
        xq,
        wq_sh,
        x_scale,
        w_scale,
        out,
        *tile,
        num_buffers=num_buffers,
        split_k=split_k,
        m_warp=m_warp,
        n_warp=n_warp,
        cluster_m=cluster[0],
        cluster_n=cluster[1],
    )
    return out


@benchmark()
def test_gemm(
    dtype,
    m,
    n,
    k,
    tile_m=128,
    tile_n=128,
    tile_k=128,
    num_buffers=4,
    split_k=1,
    m_warp=2,
    n_warp=2,
    cluster_m=1,
    cluster_n=1,
    variant="dense",
):
    torch.manual_seed(0)
    xq, wq, x_scale, w_scale = _quant(m, n, k)
    wq_sh = shuffle_weight(wq, layout=(16, 16))
    ref = _ref(xq, wq, x_scale, w_scale, dtype)

    xin = xq
    if variant == "strided_a":
        big = torch.empty(m, k + 64, dtype=xq.dtype, device="cuda")
        big[:, :k].copy_(xq)
        xin = big[:, :k]
    if variant == "strided_c":
        out = torch.full((m, n + 64), -1.0, dtype=dtype, device="cuda")[:, :n]
    else:
        out = torch.empty(m, n, dtype=dtype, device="cuda")

    kmod = None
    if variant == "oob":
        import aiter.ops.flydsl.kernels.gemm_fp8fp4_gfx1250 as kmod

        saved, kmod._TDM_HAS_OOB = kmod._TDM_HAS_OOB, False
    try:
        out, us = run_gemm_flydsl(
            xin,
            wq_sh,
            x_scale,
            w_scale,
            out,
            (tile_m, tile_n, tile_k),
            num_buffers,
            split_k,
            m_warp,
            n_warp,
            (cluster_m, cluster_n),
        )
    finally:
        if kmod is not None:
            kmod._TDM_HAS_OOB = saved

    tol = 2e-2 if split_k > 1 else 1e-2
    err = float(checkAllclose(ref, out, rtol=tol, atol=tol, printLog=False))
    return {
        "us": round(us, 2),
        "tflops": round(2 * m * n * k / us / 1e6, 1),
        "err": round(err, 4),
        "pass": bool(err <= _bound(split_k)),
    }


def _report(title, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s:\n%s", title, df.to_markdown(index=False))
    print(f"\n== {title} ==\n{df.to_markdown(index=False)}")
    return df


def run_sweep(l_dtype, l_mnk, **cfg):
    rows = []
    for dtype in l_dtype:
        for m, n, k in l_mnk:
            try:
                rows.append(test_gemm(dtype, m, n, k, **cfg))
            except Exception as e:  # noqa: BLE001
                rows.append(
                    {
                        "dtype": dtype,
                        "m": m,
                        "n": n,
                        "k": k,
                        "pass": False,
                        "note": str(e)[:50],
                    }
                )
    return _report("shape sweep", rows)


def run_features(l_dtype):
    dt = l_dtype[0]
    cases = [
        dict(m=17, n=256, k=1024, num_buffers=2, variant="dense"),
        dict(m=100, n=256, k=1024, num_buffers=2, split_k=2, variant="dense"),
        dict(m=257, n=256, k=1024, num_buffers=2, split_k=4, variant="dense"),
        dict(m=128, n=256, k=1024, num_buffers=2, variant="strided_a"),
        dict(m=128, n=256, k=1024, num_buffers=2, split_k=2, variant="strided_c"),
        dict(m=100, n=256, k=512, tile_m=64, tile_n=64, num_buffers=2, variant="oob"),
        dict(
            m=1,
            n=256,
            k=512,
            tile_m=16,
            tile_n=32,
            tile_k=256,
            num_buffers=2,
            m_warp=1,
            n_warp=2,
        ),
        dict(
            m=1,
            n=256,
            k=512,
            tile_m=32,
            tile_n=64,
            tile_k=256,
            num_buffers=2,
            m_warp=2,
            n_warp=4,
        ),
        dict(m=512, n=512, k=512, num_buffers=2, cluster_m=2, cluster_n=2),
    ]
    rows = []
    for c in cases:
        try:
            rows.append(test_gemm(dt, **c))
        except Exception as e:  # noqa: BLE001
            rows.append({"dtype": dt, **c, "pass": False, "note": str(e)[:50]})
    return _report("feature checks", rows)


_DEFAULT_MNK = [
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    (16384, 1280, 8192),
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
    (16, 7424, 8192),
    (32, 7424, 8192),
    (48, 7424, 8192),
    (64, 7424, 8192),
    (4096, 7424, 8192),
    (5120, 7424, 8192),
    (8192, 7424, 8192),
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="gfx1250 FlyDSL PTPC FP8 a8w8 bpreshuffle GEMM correctness + benchmark.",
)
parser.add_argument(
    "-d", "--dtype", nargs="*", choices=list(_DTYPE), default=list(_DTYPE)
)
parser.add_argument("-mnk", type=dtypes.str2tuple, nargs="*", default=_DEFAULT_MNK)
parser.add_argument("--tile_m", type=int, default=128)
parser.add_argument("--tile_n", type=int, default=128)
parser.add_argument("--tile_k", type=int, default=128)
parser.add_argument("--num_buffers", type=int, default=4)
parser.add_argument("--split_k", type=int, default=1)
parser.add_argument("--m_warp", type=int, default=2)
parser.add_argument("--n_warp", type=int, default=2)
parser.add_argument(
    "--no-features", action="store_true", help="Skip the feature checks."
)
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("--suffix", type=str, default="results")
args = parser.parse_args()

if not torch.cuda.is_available() or get_gfx() != "gfx1250":
    print(f"Skipping: requires gfx1250 (current: {get_gfx()})")
    sys.exit(0)

l_dtype = [_DTYPE[d] for d in args.dtype]
df = run_sweep(
    l_dtype,
    args.mnk,
    tile_m=args.tile_m,
    tile_n=args.tile_n,
    tile_k=args.tile_k,
    num_buffers=args.num_buffers,
    split_k=args.split_k,
    m_warp=args.m_warp,
    n_warp=args.n_warp,
)
dfs = [df]
if not args.no_features:
    dfs.append(run_features(l_dtype))

if args.output:
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(
        args.output, f"gemm_a8w8_bpreshuffle_gfx1250_{args.suffix}.csv"
    )
    pd.concat(dfs, ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")

n_fail = sum(int((~d["pass"]).sum()) for d in dfs)
sys.exit(1 if n_fail else 0)
