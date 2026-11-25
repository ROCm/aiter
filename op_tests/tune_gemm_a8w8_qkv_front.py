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
import itertools
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


def _front_call(s, x, tier, aux, m):  # tier: a str or a front.Plan
    qkv = front.gemm_a8w8_qkv_front(
        x,
        s.weight,
        s.w_scale,
        s.chain_scale,
        defer_ok=True,
        tier=tier,
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


def time_tier(sets, x, tier, reps=30):
    m = x.shape[0]
    aux = _aux(m)
    for s in sets:  # eager warm-up (compile, counter pools)
        _front_call(s, x, tier, aux, m)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for s in sets:
            _front_call(s, x, tier, aux, m)
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
    q, _ = _front_call(sets[0], x, tier, aux, m)
    torch.cuda.synchronize()
    return statistics.median(ts), q


def split_candidates(m, sets, x):
    """The split plans worth a whole-front timing: per K-split factor the GEMM tile with the fastest slab kernel (kernel-only
    graph timing over every (RT, NI, nt, il) that compiles and fits), so no split factor is skipped.
    """
    from aiter import dtypes
    from aiter.ops.flydsl.kernels import mxfp8_gemm_decode_splitk as sk
    from aiter.ops.quant import per_group_quant_hip

    xq, xs = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    best = {}
    for ks in sk.KS_CHOICES:
        for rt, ni, nt, il in itertools.product(
            (2, 3, 4, 5), (2, 4, 5), (False, True), (False, True)
        ):
            cfg = (rt, ni, nt, il)
            try:
                us = _graph_us(
                    sets,
                    lambda s: sk.mxfp8_gemm_decode_splitk(
                        xq, xs, s.weight, s.chain_scale, cfg=cfg, ks=ks
                    ),
                )
            except (
                Exception
            ):
                continue
            if ks not in best or us < best[ks][0]:
                best[ks] = (us, cfg)
    return [front.Plan(front.TIER_SPLIT, ks, cfg) for ks, (_, cfg) in best.items()]


def _graph_us(sets, fn, reps=20):
    for s in sets:
        fn(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for s in sets:
            fn(s)
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
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="17,24,32,40,48,56,64,72,80,88,96,104,112,120,128")
    ap.add_argument("--out", default="")
    ap.add_argument("--sets", type=int, default=8)
    args = ap.parse_args()
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    sets = [_Set(i) for i in range(args.sets)]
    rows = []
    print("M, winner, us per front (cold weights); other candidates")
    for m in [int(v) for v in args.m.split(",")]:
        x = (torch.randn(m, K, device="cuda") * 2).to(
            torch.bfloat16
        )  # the same activation for every candidate
        plans = [front.Plan(front.TIER_PLAIN)]
        if front.tier_applies(front.TIER_COOP, m):
            plans.append(front.Plan(front.TIER_COOP))
        elif front.tier_applies(front.TIER_SPLIT, m):
            plans += split_candidates(m, sets, x)
        res, qref = {}, None
        for plan in plans:
            us, q = time_tier(sets, x, plan)
            qref = q if plan.tier == front.TIER_PLAIN else qref
            err = (
                0.0
                if plan.tier == front.TIER_PLAIN
                else (
                    (q.float() - qref.float()).abs().mean() / qref.float().abs().mean()
                ).item()
            )
            res[plan] = (us, err)
        win = min(res, key=lambda p: res[p][0])
        print(
            f"{m}, {front.plan_kernel_name(win)}, {res[win][0]:.2f}; "
            + ", ".join(
                f"{front.plan_kernel_name(p)}={u:.2f}" for p, (u, _) in res.items()
            ),
            flush=True,
        )
        us, err = res[win]
        ks = win.ks if win.tier != front.TIER_PLAIN else 1
        rows.append(
            f"{get_gfx_runtime()},{get_cu_num()},{m},{N},{K},1x32,flydsl,{front.KERNEL_ID},{ks},{us:.4f},"
            f"{front.plan_kernel_name(win)},{2 * m * N * K / us / 1e6:.2f},{N * K / us / 1e3:.2f},{err:.2e}"
        )
    if args.out:
        with open(args.out, "w") as f:
            f.write(
                "gfx,cu_num,M,N,K,w_scale_block,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
            )
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
