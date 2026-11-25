# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner of the MXFP8 QKV front tier (``aiter/ops/gemm_a8w8_qkv_front.py``): for each decode row count it times every tier
that serves it through the PUBLIC call chain (``gemm_a8w8_qkv_front`` -> ``fused_qknorm_idxrqknorm``, which runs the deferred
GEMM + epilogue of a fused tier) over the whole front (quantize + GEMM + norm / RoPE / NVFP4 insert), and writes the
winner as a row of ``a8w8_mxfp8_qkv_front_tuned_gemm.csv`` (``kernelName`` = ``flydsl_qkvfront_<tier>``).

The generic GEMM tuner times a GEMM alone; a fused tier owns its epilogue, so its candidates are only comparable as a whole
front, which this script measures. Timing: R calls on R different weight / cache copies captured in one HIP graph
(cold weights), replay median / R.

    python3 op_tests/tune_gemm_a8w8_qkv_front.py --m 17,24,32,... --out rows.csv
"""

import argparse
import statistics

import torch

import aiter
from aiter import dtypes
from aiter.ops import gemm_a8w8_qkv_front as front
from aiter.ops.shuffle import shuffle_weight

N, K = front.N, front.K
NB = 64


class _Set:
    """One weight copy + its cache pages (distinct memory per graph node, so the weight is read cold)."""

    def __init__(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        w = (torch.randn(N, K, generator=g, device="cuda") * 0.05).to(dtypes.fp8)
        self.w_scale = torch.randint(
            120, 128, (N, K // 32), generator=g, dtype=torch.uint8, device="cuda"
        )
        self.weight = shuffle_weight(w, layout=(16, 16))
        self.chain_scale = front.chain_weight_scale(self.w_scale)
        self.kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8, device="cuda")
        self.ix = torch.zeros(NB, 128, 72, dtype=torch.uint8, device="cuda")


def _front_call(s, x, tier, aux, m, a_row=False):
    qkv = front.gemm_a8w8_qkv_front(
        x,
        s.weight,
        s.w_scale,
        s.chain_scale,
        defer_ok=tier != "plain",
        tier=tier,
        a_row=a_row,
    )
    q = torch.empty(m, 2048, dtype=torch.bfloat16, device="cuda")
    iq = torch.empty(m, 128, dtype=dtypes.fp8, device="cuda")
    pages = s.kv.view(NB, -1)
    qw, kw, iqw, ikw, cs, pos, slot, ks, vs = aux
    aiter.fused_qknorm_idxrqknorm(
        qkv, qw, kw, cs, pos, 16, 1, 64, 1e-6, iqw, ikw, 1, slot, pages, pages,
        s.ix, 128, q, iq, slot, kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4",
        k_scale=ks, v_scale=vs, asm_layout=False, skip_index_branch=False,
    )  # fmt: skip
    return q, iq


def _aux(m, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qw, kw, iqw, ikw = [
        (torch.randn(128, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
        for _ in range(4)
    ]
    cs = torch.randn(4096, 64, generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 4000, (m,), generator=g, dtype=torch.int64, device="cuda")
    nreq = (m + 7) // 8
    bases = (
        torch.randperm(NB * 128 // 32, generator=g, device="cuda")[:nreq] * 32
    ).tolist()
    slot = torch.cat([torch.arange(b, b + 8) for b in bases])[:m].to(torch.int64).cuda()
    return (
        qw,
        kw,
        iqw,
        ikw,
        cs,
        pos,
        slot,
        torch.tensor([0.5], device="cuda"),
        torch.tensor([0.7], device="cuda"),
    )


def time_tier(sets, x, tier, reps=30, a_row=False):
    m = x.shape[0]
    aux = _aux(m)
    for s in sets:  # eager warm-up (compile, counter pools)
        _front_call(s, x, tier, aux, m, a_row)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for s in sets:
            _front_call(s, x, tier, aux, m, a_row)
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / len(sets))
    q, _ = _front_call(sets[0], x, tier, aux, m, a_row)
    torch.cuda.synchronize()
    return statistics.median(ts), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="17,24,32,40,48,56,64,72,80,88,96,104,112,120,128")
    ap.add_argument("--out", default="")
    ap.add_argument("--sets", type=int, default=8)
    args = ap.parse_args()
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    sets = [_Set(i) for i in range(args.sets)]
    rows = []
    print("M, " + ", ".join(front.TIERS) + ", winner (us per front, cold weights)")
    for m in [int(v) for v in args.m.split(",")]:
        res, qref = {}, None
        x = (torch.randn(m, K, device="cuda") * 2).to(
            torch.bfloat16
        )  # the same activation for every tier
        for tier in front.TIERS:
            if not front.tier_applies(tier, m):
                continue
            us, q = time_tier(sets, x, tier)
            qref = q if tier == "plain" else qref
            err = (
                0.0
                if qref is None or tier == "plain"
                else (
                    (q.float() - qref.float()).abs().mean() / qref.float().abs().mean()
                ).item()
            )
            res[tier] = (us, err)
        win = min(res, key=lambda t: res[t][0])
        print(
            f"{m}, "
            + ", ".join(f"{res[t][0]:.2f}" if t in res else "-" for t in front.TIERS)
            + f", {win}",
            flush=True,
        )
        us, err = res[win]
        rows.append(
            f"{get_gfx_runtime()},{get_cu_num()},{m},{N},{K},1x32,flydsl,{front.KERNEL_ID},1,{us:.4f},"
            f"{front.tier_kernel_name(win)},{2 * m * N * K / us / 1e6:.2f},{N * K / us / 1e3:.2f},{err:.2e}"
        )
    if args.out:
        with open(args.out, "w") as f:
            f.write(
                "gfx,cu_num,M,N,K,w_scale_block,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
            )
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
