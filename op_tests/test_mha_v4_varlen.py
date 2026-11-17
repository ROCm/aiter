# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf for the gfx950 varlen BF16 hd256 ASM kernels (mha_v4_varlen_fwd).

Layout (packed THD, whole documents, seqlen_q == seqlen_k), causal and non-causal:
    q, out : (total, Hq, 256)      k, v : (total, Hkv, 256)      cu_seqlens : int32 [batch + 1]
    lse    : (Hq, total) float32, natural log

O passes when its max error against an exact fp32 reference stays within a small multiple of the
error of rounding that reference to BF16 (the floor for any BF16-output kernel); LSE within 1e-3.
--large adds packs past 4 GiB of Q, checked on sampled rows, which exercise 64-bit addressing.
"""

import argparse

import pandas as pd
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import mha_v4_varlen_fwd
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
HEAD_DIM = 256


def make_varlen(seqlens, hq, hk, seed=0):
    torch.manual_seed(seed)
    total = sum(seqlens)
    q = torch.randn(total, hq, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(total, hk, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(total, hk, HEAD_DIM, dtype=torch.bfloat16)
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32)
    return q, k, v, cu


def reference_doc(q, k, v, scale, causal, rows=None, chunk=1024):
    """Exact fp32 attention for one document; rows selects query rows (default all)."""
    L, hq, _ = q.shape
    g = hq // k.shape[1]
    kf = k.float().repeat_interleave(g, dim=1).permute(1, 2, 0)
    vf = v.float().repeat_interleave(g, dim=1).permute(1, 0, 2)
    rows = torch.arange(L) if rows is None else rows
    kv = torch.arange(L)
    outs, lses = [], []
    for c0 in range(0, rows.numel(), chunk):
        r = rows[c0 : c0 + chunk]
        s = (q[r].float().permute(1, 0, 2) @ kf) * scale
        if causal:
            s.masked_fill_(kv[None, None, :] > r[None, :, None], float("-inf"))
        lse = torch.logsumexp(s, dim=-1)
        outs.append((torch.exp(s - lse[..., None]) @ vf).permute(1, 0, 2))
        lses.append(lse)
    return torch.cat(outs, 0), torch.cat(lses, 1)


def check(q, k, v, cu, out, lse, causal, sample_rows=0):
    """Max O error, its BF16 rounding floor, and max LSE error over every document."""
    scale = HEAD_DIM**-0.5
    o_err = rne = lse_err = 0.0
    bounds = cu.tolist()
    for b in range(len(bounds) - 1):
        s0, s1 = bounds[b], bounds[b + 1]
        L = s1 - s0
        rows = None
        if sample_rows and sample_rows < L:
            edge = torch.cat(
                [torch.arange(min(L, 256)), torch.arange(max(0, L - 256), L)]
            )
            rows = torch.unique(torch.cat([edge, torch.randint(0, L, (sample_rows,))]))
        ro, rl = reference_doc(q[s0:s1], k[s0:s1], v[s0:s1], scale, causal, rows)
        ko = out[s0:s1] if rows is None else out[s0:s1][rows]
        kl = lse[:, s0:s1] if rows is None else lse[:, s0:s1][:, rows]
        o_err = max(
            o_err, (ko.float() - ro).abs().nan_to_num(float("inf")).max().item()
        )
        rne = max(rne, (ro.to(torch.bfloat16).float() - ro).abs().max().item())
        lse_err = max(lse_err, (kl - rl).abs().nan_to_num(float("inf")).max().item())
    return o_err, rne, lse_err


def attention_flops(seqlens, hq, causal):
    return sum(
        4.0 * hq * HEAD_DIM * L * ((L + 1) / 2 if causal else L) for L in seqlens
    )


@benchmark()
def test_mha_v4_varlen(hq, hk, seqlens, causal=True, sample_rows=0, perf=True):
    q, k, v, cu = make_varlen(seqlens, hq, hk)
    max_seqlen = max(seqlens)
    out = torch.full_like(q, float("nan"))
    _, lse = mha_v4_varlen_fwd(
        q, k, v, cu, max_seqlen, causal=causal, out=out, return_lse=True
    )
    torch.cuda.synchronize()
    o_err, rne, lse_err = check(q, k, v, cu, out, lse, causal, sample_rows)
    assert o_err <= max(2e-2, 2.5 * rne), f"O error {o_err:.3e} vs BF16 floor {rne:.3e}"
    assert lse_err < 1e-3, f"LSE error {lse_err:.3e}"
    ret = {
        "gfx": get_gfx(),
        "docs": len(seqlens),
        "total": q.shape[0],
        "err(O)": o_err,
        "bf16 floor": rne,
        "err(LSE)": lse_err,
    }
    if perf:
        _, us = run_perftest(
            mha_v4_varlen_fwd,
            q,
            k,
            v,
            cu,
            max_seqlen,
            causal=causal,
            num_iters=10,
            num_warmup=2,
        )
        ret["us"] = us
        ret["TFLOPS"] = attention_flops(seqlens, hq, causal) / us / 1e6
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--heads",
        type=str,
        nargs="*",
        default=["24,4", "12,2", "6,1"],
        help="Hq,Hkv pairs",
    )
    parser.add_argument(
        "--mask",
        choices=["causal", "none", "both"],
        default="both",
        help="which kernels",
    )
    parser.add_argument("--large", action="store_true", help="add >4 GiB packs (slow)")
    args = parser.parse_args()
    if get_gfx() not in SUPPORTED_GFX:
        print(f"skip: mha_v4_varlen_fwd ships only for {SUPPORTED_GFX}")
        return

    causals = {"causal": [True], "none": [False], "both": [True, False]}[args.mask]
    gen = torch.Generator().manual_seed(1)
    mixed = [1, 31, 33, 255, 256, 257, 1000, 3000, 8192]
    random_docs = torch.randint(1, 6000, (40,), generator=gen, device="cpu").tolist()
    rows = []
    for causal in causals:
        for pair in args.heads:
            hq, hk = map(int, pair.split(","))
            for seqlens in (
                [1024],
                [2048],
                [4096],
                [8192],
                [300, 4096],
                mixed,
                random_docs,
            ):
                rows.append(test_mha_v4_varlen(hq, hk, seqlens, causal))
        if args.large:
            rows.append(
                test_mha_v4_varlen(
                    24, 4, [100000, 262144, 50000], causal, sample_rows=512
                )
            )
            rows.append(
                test_mha_v4_varlen(
                    6, 1, [262144] * 3 + [131072], causal, sample_rows=512
                )
            )
    df = pd.DataFrame(rows)
    df["seqlens"] = df["seqlens"].map(lambda s: s if len(s) <= 4 else f"{len(s)} docs")
    print(df.to_string(index=False, float_format="{:.4g}".format))


if __name__ == "__main__":
    main()
