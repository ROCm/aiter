# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test + bench for the FP4 VSA dual-warp-set attention kernel
(``aiter.fp4_vsa_dual_dropB``, gfx950).

Loads the canonical .co from
    $AITER_ASM_DIR/gfx950/vsa/vsa_dual_setprio_dropB.co

If ``/data/fp4_data/{big_q2k_block_sparse_mask.pt, big_variable_block_sizes.pt}``
is present we run on the real dev-box dataset; otherwise a synthetic
banded-sparse mask of the same density is generated.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, perftest


# --------------------------------------------------------------------------- #
# Test-data builders  (self-contained — do NOT depend on /home/VSA_Release)
# --------------------------------------------------------------------------- #
def _map_to_index_torch(block_map: torch.Tensor):
    """``block_map`` (B,H,Qb,Kb) bool → ``(index, num)`` for the kernel ABI."""
    bs, h, num_q, num_kv = block_map.shape
    flat       = block_map.reshape(-1, num_kv).to(torch.int32)
    num_flat   = flat.sum(dim=-1).to(torch.int32)
    sort_keys  = (flat > 0).to(torch.int32) * (num_kv + 1) - torch.arange(
        num_kv, device=flat.device, dtype=torch.int32
    )
    order      = torch.argsort(sort_keys, dim=-1, descending=True).to(torch.int32)
    cols       = torch.arange(num_kv, device=flat.device, dtype=torch.int32)
    mask_valid = cols.unsqueeze(0) < num_flat.unsqueeze(-1)
    index_flat = torch.where(mask_valid, order, torch.full_like(order, -1))
    return (
        index_flat.reshape(bs, h, num_q, num_kv),
        num_flat.reshape(bs, h, num_q),
    )


def _make_synthetic_data(
    seed: int        = 0,
    B: int           = 1,
    H: int           = 4,
    num_q_blks: int  = 1024,
    sparsity: float  = 0.0846,   # ≈ real-dataset density
    dense_frac: float = 0.03,    # ≈ real-dataset n_dense / total
    block_size: int   = 128,
) -> dict:
    """Synthetic banded sliding-window mask + matching FP4/FP8 tensors."""
    T          = num_q_blks * 128
    max_kv     = num_q_blks
    D, FP4_DIM = 128, 64
    SCALE_D    = 4

    torch.manual_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed)

    n_attended = max(1, int(round(max_kv * sparsity)))
    half       = n_attended // 2
    mask = torch.zeros(
        (B, H, num_q_blks, max_kv), dtype=torch.bool, device="cuda"
    )
    for q in range(num_q_blks):
        lo = max(0, q - half)
        hi = min(max_kv, lo + n_attended)
        lo = max(0, hi - n_attended)
        mask[..., q, lo:hi] = True
    n_dense_rows = max(1, int(round(num_q_blks * dense_frac)))
    dense_q = torch.randint(0, num_q_blks, (n_dense_rows,), device="cuda", generator=g)
    mask[..., dense_q, :] = True

    q2k_index, q2k_num = _map_to_index_torch(mask)

    q  = torch.randint(0, 128, (B, H, T, FP4_DIM), device="cuda",
                       dtype=torch.uint8, generator=g).contiguous()
    k  = torch.randint(0, 128, (B, H, T, FP4_DIM), device="cuda",
                       dtype=torch.uint8, generator=g).contiguous()
    v  = torch.randn((B, H, T, D), device="cuda", generator=g).to(
        torch.float8_e4m3fn).contiguous()
    qs = torch.randint(0, 128, (B, H, T, SCALE_D), device="cuda",
                       dtype=torch.uint8, generator=g).contiguous()
    ks = torch.randint(0, 128, (B, H, T, SCALE_D), device="cuda",
                       dtype=torch.uint8, generator=g).contiguous()
    vs = torch.randn((B, H, D), device="cuda", dtype=torch.float32,
                     generator=g).contiguous()
    vm = torch.randn((B, H, D), device="cuda", dtype=torch.float32,
                     generator=g).contiguous()
    vbs = torch.full((num_q_blks,), block_size, dtype=torch.int32,
                     device="cuda").contiguous()

    BH = B * H
    bundle = {
        "B": B, "H": H, "T": T, "D": D,
        "num_q_blks": num_q_blks, "max_kv": max_kv,
        "q":  q.reshape(BH, T, -1).contiguous(),
        "k":  k.reshape(BH, T, -1).contiguous(),
        "v":  v.reshape(BH, T, -1).contiguous(),
        "qs": qs.reshape(BH, T, -1).contiguous(),
        "ks": ks.reshape(BH, T, -1).contiguous(),
        "vm": vm.reshape(BH, -1).contiguous(),
        "vs": vs.reshape(BH, -1).contiguous(),
        "q2k_idx": q2k_index.reshape(BH * num_q_blks, -1).contiguous(),
        "q2k_num": q2k_num.reshape(BH * num_q_blks).contiguous(),
        "vbs": vbs,
        "_synthetic": True,
    }
    bundle["lim"], bundle["n_dense"] = aiter.build_l2_aware_lim_vsa(
        bundle["q2k_idx"], bundle["q2k_num"], max_kv
    )
    return bundle


def _make_real_data(seed: int, data_dir: str) -> dict:
    """Load the live block-sparse mask from the dev-box snapshot."""
    mask_path = os.path.join(data_dir, "big_q2k_block_sparse_mask.pt")
    vbs_path  = os.path.join(data_dir, "big_variable_block_sizes.pt")
    mask = torch.load(mask_path, weights_only=False)
    vbs  = torch.load(vbs_path,  weights_only=False).cuda()
    B, H, num_q_blks, _ = mask.shape
    T = num_q_blks * 128
    D, FP4_DIM, SCALE_D = 128, 64, 4

    q2k_index, q2k_num = _map_to_index_torch(mask)
    max_kv = q2k_index.shape[-1]

    torch.manual_seed(seed)
    q  = torch.randint(0, 128, (B, H, T, FP4_DIM), device="cuda",
                       dtype=torch.uint8).contiguous()
    k  = torch.randint(0, 128, (B, H, T, FP4_DIM), device="cuda",
                       dtype=torch.uint8).contiguous()
    v  = torch.randn((B, H, T, D), device="cuda").to(
        torch.float8_e4m3fn).contiguous()
    qs = torch.randint(0, 128, (B, H, T, SCALE_D), device="cuda",
                       dtype=torch.uint8).contiguous()
    ks = torch.randint(0, 128, (B, H, T, SCALE_D), device="cuda",
                       dtype=torch.uint8).contiguous()
    vs = torch.randn((B, H, D), device="cuda", dtype=torch.float32).contiguous()
    vm = torch.randn((B, H, D), device="cuda", dtype=torch.float32).contiguous()

    BH = B * H
    bundle = {
        "B": B, "H": H, "T": T, "D": D,
        "num_q_blks": num_q_blks, "max_kv": max_kv,
        "q":  q.reshape(BH, T, -1).contiguous(),
        "k":  k.reshape(BH, T, -1).contiguous(),
        "v":  v.reshape(BH, T, -1).contiguous(),
        "qs": qs.reshape(BH, T, -1).contiguous(),
        "ks": ks.reshape(BH, T, -1).contiguous(),
        "vm": vm.reshape(BH, -1).contiguous(),
        "vs": vs.reshape(BH, -1).contiguous(),
        "q2k_idx": q2k_index.reshape(BH * num_q_blks, -1).contiguous(),
        "q2k_num": q2k_num.reshape(BH * num_q_blks).contiguous(),
        "vbs": vbs,
        "_synthetic": False,
    }
    bundle["lim"], bundle["n_dense"] = aiter.build_l2_aware_lim_vsa(
        bundle["q2k_idx"], bundle["q2k_num"], max_kv
    )
    return bundle


def _make_data_or_synthetic(seed: int, data_dir: str, **kw) -> dict:
    mask_path = os.path.join(data_dir, "big_q2k_block_sparse_mask.pt")
    vbs_path  = os.path.join(data_dir, "big_variable_block_sizes.pt")
    if os.path.isfile(mask_path) and os.path.isfile(vbs_path):
        return _make_real_data(seed=seed, data_dir=data_dir)
    return _make_synthetic_data(seed=seed, **kw)


# --------------------------------------------------------------------------- #
# Perf-tested wrapper around the kernel
# --------------------------------------------------------------------------- #
@perftest()
def run_fp4_vsa_dual_dropB(data: dict,
                           out: torch.Tensor,
                           lse: torch.Tensor,
                           counters: torch.Tensor):
    return aiter.fp4_vsa_dual_dropB(
        q=data["q"], k=data["k"], v=data["v"],
        qs=data["qs"], ks=data["ks"], vm=data["vm"], vs=data["vs"],
        q2k_idx=data["q2k_idx"], q2k_num=data["q2k_num"], vbs=data["vbs"],
        lim=data["lim"], n_dense=data["n_dense"],
        B=data["B"], T=data["T"],
        num_q_blks=data["num_q_blks"], max_kv=data["max_kv"],
        out=out, lse=lse, counters=counters,
    )


# --------------------------------------------------------------------------- #
# Top-level test (correctness sanity + perf)
# --------------------------------------------------------------------------- #
@benchmark()
def test_fp4_vsa_dual_dropB(
    data_dir: str    = "/data/fp4_data",
    seed: int        = 42,
    B: int           = 1,
    H: int           = 4,
    num_q_blks: int  = 1024,
) -> dict:
    torch.set_default_device("cuda:0")
    ret = {}

    data = _make_data_or_synthetic(
        seed=seed, data_dir=data_dir,
        B=B, H=H, num_q_blks=num_q_blks,
    )
    BH = data["B"] * data["H"]
    out      = torch.empty((BH, data["T"], data["D"]),
                           dtype=torch.bfloat16, device="cuda")
    lse      = torch.empty((BH, data["T"]),
                           dtype=torch.float32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")

    (out, lse), us = run_fp4_vsa_dual_dropB(data, out, lse, counters)
    torch.cuda.synchronize()

    ret["data_src"]      = "synthetic" if data.get("_synthetic") else "real"
    ret["B"]             = data["B"]
    ret["H"]             = data["H"]
    ret["T"]             = data["T"]
    ret["num_q_blks"]    = data["num_q_blks"]
    ret["max_kv"]        = data["max_kv"]
    ret["n_dense"]       = data["n_dense"]
    ret["has_nan"]       = bool(torch.isnan(out.float()).any().item())
    ret["has_inf"]       = bool(torch.isinf(out.float()).any().item())
    ret["out_first8_l2"] = float(out[0, 0, :8].float().pow(2).sum().sqrt().item())
    ret["us"]            = us
    return ret


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="aiter test for fp4_vsa_dual_dropB (gfx950)",
)
parser.add_argument("--data-dir", type=str, default="/data/fp4_data",
                    help="dir holding big_q2k_block_sparse_mask.pt + big_variable_block_sizes.pt; "
                         "if missing, fall back to a synthetic banded mask")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("-b", "--batch_size", type=int, default=[1], nargs="+")
parser.add_argument("--num_heads", type=int, default=[4], nargs="+")
parser.add_argument("--num_q_blks", type=int, default=[1024], nargs="+",
                    help="only used when synthetic data is generated")
args = parser.parse_args()

if get_gfx() != "gfx950":
    aiter.logger.warning(
        "fp4_vsa_dual_dropB is gfx950-only; current arch is %s — kernel will most "
        "likely fail to load. Skipping.", get_gfx())
    raise SystemExit(0)

df = []
for B in args.batch_size:
    for H in args.num_heads:
        for nqb in args.num_q_blks:
            ret = test_fp4_vsa_dual_dropB(
                data_dir=args.data_dir, seed=args.seed,
                B=B, H=H, num_q_blks=nqb,
            )
            df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("fp4_vsa_dual_dropB summary (markdown):\n%s", df_md)
