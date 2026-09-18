# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the atomic split-K epilogue into its own config, off the mainline path.

Mainline dispatch keeps using the fused (clustered) rows in
``*_tuned_gemm.csv``; the atomic winners land in ``*_atomic_tuned_gemm.csv``,
which only ``test_gemm_a8w8_blockscale.py --splitk-ab`` reads.

usage: tune_a8w8_splitk_atomic.py -m 512 -nk 6144,7168 7168,16384 --apre 0 1
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
from atomic_splitk_config import atomic_config_path

from aiter import benchmark_data_init as bench_init
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
    kernel_fits_shape,
    kernels_list,
)
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    COMPUTE_WMMA_NAME_PREFIX,
    is_compute_wmma_kernel_name,
    resolve_splitk_mode,
)
from aiter.ops.shuffle import shuffle_mxfp8fp4_a, shuffle_weight
from aiter.test_common import checkAllclose, perftest
from aiter.utility import fp4_utils

BLOCK_K = 128
# sk=8 has deadlocked the box; cluster dims >= 8 hard-hang it.
MAX_SPLIT_K = 4
MAX_CLUSTER_DIM = 4


def flydsl_call(x, w, x_scale, w_scale, name, m, apre):
    """Out is allocated here, as the dispatch does, so perftest rotates the same
    argument set the ubench will."""
    from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250,
    )

    out = torch.empty((m, w.shape[0]), dtype=dtypes.bf16, device=x.device)
    return run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
        x, w, x_scale, w_scale, out, name, a_is_preshuffled=apre
    )


def atomic_name(ki, apre):
    """Tuned name with the explicit _atm opt-in, before any _apre/_ps suffix."""
    return re.sub(r"(_cn\d+)", r"\1_atm", ki.name_for(apre), count=1)


def candidates(m, n, k, apre, cu_num):
    """Legal compute-bound kernelNames that actually resolve to atomic."""
    out, seen = [], set()
    for ki in kernels_list.values():
        if ki.name_prefix != COMPUTE_WMMA_NAME_PREFIX or ki.persistent_n_tiles > 1:
            continue
        if ki.split_k < 2 or ki.split_k > MAX_SPLIT_K:
            continue
        if max(ki.cluster_m, ki.cluster_n) > MAX_CLUSTER_DIM:
            continue
        if not kernel_fits_shape(ki, m, n, k):
            continue
        name = atomic_name(ki, apre)
        if name in seen:
            continue
        mode = resolve_splitk_mode(
            m,
            n,
            ki.tile_m,
            ki.tile_n,
            ki.cluster_m,
            ki.cluster_n,
            ki.split_k,
            is_compute_wmma_kernel_name(name),
            "atomic",
        )
        if mode != "atomic":
            continue
        gx = -(-m // ki.tile_m)
        gx = -(-gx // ki.cluster_m) * ki.cluster_m
        if gx * (-(-n // ki.tile_n)) * ki.split_k > cu_num:
            continue
        seen.add(name)
        out.append((name, ki))
    return out


def make_operands(m, n, k, seed):
    gen = bench_init.make_generator(seed)
    x, x_scale_raw = bench_init.fill_mx_e8m0(
        (m, k), "norm", gen, block_n=1, block_k=BLOCK_K
    )
    w, w_scale_raw = bench_init.fill_mx_e8m0(
        (n, k), "norm", gen, block_n=BLOCK_K, block_k=BLOCK_K, std=1.0 / math.sqrt(k)
    )
    x_scale = x_scale_raw.view(dtypes.fp8_e8m0)
    w_scale = w_scale_raw.view(dtypes.fp8_e8m0)
    ref = (
        (
            x.to(torch.float32).view(m, k // BLOCK_K, BLOCK_K)
            * fp4_utils.e8m0_to_f32(x_scale).unsqueeze(-1)
        )
        .view(m, k)
        .to(torch.float32)
    )
    wf = fp4_utils.e8m0_to_f32(w_scale)
    wref = (
        w.to(torch.float32).view(n // BLOCK_K, BLOCK_K, k // BLOCK_K, BLOCK_K)
        * wf.view(n // BLOCK_K, 1, k // BLOCK_K, 1)
    ).view(n, k)
    ref = (ref @ wref.T).to(dtypes.bf16)
    return x, w, x_scale, w_scale, ref


def tune_one(m, n, k, apre, iters, seed, cu_num):
    x, w, x_scale, w_scale, ref = make_operands(m, n, k, seed)
    gemm_w = shuffle_weight(w, layout=(16, 16))
    gemm_xs = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    xin = shuffle_mxfp8fp4_a(x) if apre else x
    timed = perftest(num_iters=iters)(flydsl_call)
    atol = 1e-2 * ref.to(torch.float32).pow(2).mean().sqrt().item()

    cands = candidates(m, n, k, apre, cu_num)
    print(
        f"  {m}x{n}x{k} apre={int(apre)}: {len(cands)} 个合法 atomic 候选", flush=True
    )
    best = None
    for name, ki in cands:
        args = (xin, gemm_w, gemm_xs, w_scale, name, m, apre)
        try:
            for _ in range(5):
                flydsl_call(*args)
            torch.cuda.synchronize()
            y, us = timed(*args)
        except Exception as exc:  # noqa: BLE001 - a candidate that will not build
            print(f"    skip {name}: {type(exc).__name__}", flush=True)
            continue
        err = checkAllclose(ref, y, msg=f"tune {name}", atol=atol, printLog=False)
        if err > 1e-3:
            print(f"    reject {name}: err={err:.2e}", flush=True)
            continue
        print(f"    {us:8.3f} us  err={err:.1e}  {name}", flush=True)
        if best is None or us < best[1]:
            best = (name, us, ki, err)
    return best


def write_csv(path, rows):
    head = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
    keep = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                f = line.rstrip("\n").split(",")
                if len(f) > 4 and f[0] != "gfx":
                    keep[(f[0], f[1], f[2], f[3], f[4])] = line.rstrip("\n")
    for r in rows:
        f = r.split(",")
        keep[(f[0], f[1], f[2], f[3], f[4])] = r
    with open(path, "w") as fh:
        fh.write(head + "\n")
        for _, line in sorted(
            keep.items(), key=lambda kv: (kv[0][0], *map(int, kv[0][1:]))
        ):
            fh.write(line + "\n")
    print(f"  写入 {path}  ({len(keep)} 行)")


def main(a):
    gfx = get_gfx()
    assert gfx == "gfx1250", f"atomic split-K is gfx1250-only, got {gfx}"
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    for apre in a.apre:
        rows = []
        for m in a.m:
            for n, k in a.nk:
                best = tune_one(m, n, k, apre, a.iters, a.seed, cu_num)
                if best is None:
                    print(f"  {m}x{n}x{k} apre={int(apre)}: 无可用 atomic 候选，跳过")
                    continue
                name, us, ki, err = best
                tflops = m * n * k * 2 / us / 1e6
                bw = (m * k + n * k) / us / 1e3  # GB/s, fp8 operands = 1 B/elem
                rows.append(
                    f"{gfx},{cu_num},{m},{n},{k},flydsl,0,{ki.split_k},"
                    f"{us:.4f},{name},{tflops:.2f},{bw:.2f},{err:.6g}"
                )
                print(f"  => BEST {m}x{n}x{k} apre={int(apre)}: {us:.3f} us  {name}\n")
        if rows:
            write_csv(atomic_config_path(apre), rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("-m", type=int, nargs="+", default=[512])
    p.add_argument(
        "-nk",
        type=str,
        nargs="+",
        required=True,
        help="N,K pairs, e.g. -nk 6144,7168 7168,16384",
    )
    p.add_argument("--apre", type=int, nargs="+", default=[0, 1], choices=[0, 1])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.nk = [tuple(int(v) for v in s.split(",")) for s in args.nk]
    args.apre = [bool(v) for v in args.apre]
    main(args)
