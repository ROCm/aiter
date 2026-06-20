# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel-level test for FlyDSL bf16 2-stage MoE across inter_dim, incl. >4GiB.

Regression for the weight buffer 32-bit-offset overflow: the per-expert weight
must be addressed via a 64-bit base, otherwise results corrupt once the whole
[E, 2*inter, model_dim] weight exceeds 4 GiB (e.g. MiniMax-M3: inter=3072,
model_dim=6144, E=128 -> 9 GiB). Validates BOTH stage1 paths:
  * k_batch=1  -> stage1 fuses SiLU(gate)*up, writes inter-wide output
  * k_batch>=2 -> split-K, stage1 writes raw 2*inter gate/up, SiLU applied here

Run:  python op_tests/test_flydsl_moe_large_inter.py
      pytest op_tests/test_flydsl_moe_large_inter.py
"""

import torch

import flydsl.compiler as flyc
from aiter.fused_moe import moe_sorting
from aiter.ops.activation import silu_and_mul
from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
)
from aiter.ops.shuffle import shuffle_weight

H = 6144  # model_dim
E = 128
TOPK = 4
DEV = "cuda"


def _ref(x, w1, w2, tw, tid):
    # Loop over experts (one fp32 weight materialized at a time) so the reference
    # stays memory-light even for large token counts.
    xf = x.float()
    out = torch.zeros(x.shape[0], H, device=DEV, dtype=torch.float32)
    for e in range(E):
        sel = tid == e
        if not sel.any():
            continue
        rows, slot = sel.nonzero(as_tuple=True)
        gu = xf[rows] @ w1[e].float().t()
        gate, up = gu.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        down = act @ w2[e].float().t()
        out[rows] += tw[rows, slot].unsqueeze(1) * down
    return out


def _cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0)).item()


def _make(ntok, inter, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(ntok, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.1
    w1 = (
        torch.randn(E, 2 * inter, H, device=DEV, dtype=torch.bfloat16, generator=g)
        * 0.05
    )
    w2 = torch.randn(E, H, inter, device=DEV, dtype=torch.bfloat16, generator=g) * 0.05
    logits = torch.randn(ntok, E, device=DEV, dtype=torch.float32, generator=g)
    tw, tid = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    return x, w1, w2, tw.float().contiguous(), tid.int().contiguous()


def run_flydsl(x, w1, w2, tw, tid, inter, *, tile_m, tile_k, k_batch):
    ntok = x.shape[0]
    split_k = k_batch > 1
    s, sw, se, nv, _ = moe_sorting(tid, tw, E, H, torch.float16, tile_m)
    if nv.numel() > 1:
        nv = nv[:1].contiguous()
    s, se = s.contiguous(), se.contiguous()
    sw1d = sw.contiguous().view(-1)
    blocks = int(se.numel())
    w1s = shuffle_weight(w1, layout=(16, 16)).contiguous().view(-1)
    w2s = shuffle_weight(w2, layout=(16, 16)).contiguous().view(-1)
    sd = torch.empty(0, device=DEV, dtype=torch.float32)
    st = torch.cuda.current_stream()

    e1 = compile_moe_gemm1(
        model_dim=H,
        inter_dim=inter,
        experts=E,
        topk=TOPK,
        in_dtype="bf16",
        group_size=-1,
        tile_m=tile_m,
        tile_n=128,
        tile_k=tile_k,
        doweight_stage1=False,
        use_cshuffle_epilog=False,
        out_dtype="bf16",
        scale_is_bf16=False,
        k_batch=int(k_batch),
    )
    e2 = compile_moe_gemm2(
        model_dim=H,
        inter_dim=inter,
        experts=E,
        topk=TOPK,
        in_dtype="bf16",
        group_size=-1,
        tile_m=tile_m,
        tile_n=256,
        tile_k=tile_k,
        doweight_stage2=True,
        use_cshuffle_epilog=True,
        accumulate=True,
        out_dtype="bf16",
        scale_is_bf16=False,
    )

    s1_w = 2 * inter if split_k else inter
    g1 = torch.zeros(ntok * TOPK, s1_w, device=DEV, dtype=torch.bfloat16)
    a2 = (
        g1
        if not split_k
        else torch.empty(ntok * TOPK, inter, device=DEV, dtype=torch.bfloat16)
    )
    out = torch.zeros(ntok, H, device=DEV, dtype=torch.bfloat16)

    c1 = flyc.compile(
        e1,
        g1.view(-1),
        x.view(-1),
        w1s,
        sd,
        sd,
        s,
        se,
        sw1d,
        nv,
        ntok,
        inter,
        H,
        blocks,
        st,
    )
    c2 = flyc.compile(
        e2,
        out.view(-1),
        a2.view(-1),
        w2s,
        sd,
        sd,
        s,
        se,
        sw1d,
        nv,
        ntok,
        H,
        inter,
        blocks,
        st,
    )

    def go():
        if split_k:
            g1.zero_()
        c1(
            g1.view(-1),
            x.view(-1),
            w1s,
            sd,
            sd,
            s,
            se,
            sw1d,
            nv,
            ntok,
            inter,
            H,
            blocks,
            st,
        )
        if split_k:
            silu_and_mul(a2, g1.view(-1, 2 * inter))
        out.zero_()
        c2(
            out.view(-1),
            a2.view(-1),
            w2s,
            sd,
            sd,
            s,
            se,
            sw1d,
            nv,
            ntok,
            H,
            inter,
            blocks,
            st,
        )
        return out

    return go


def _time_ms(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# Sweep inter across the 4 GiB w13 boundary (E=128, model_dim=6144, bf16):
#   total w13 bytes = E*2*inter*model_dim*2.  4 GiB at inter ~= 1365.
#   inter<=1280 -> <4GiB (worked before the fix);  inter>=1408 -> >4GiB (broken
#   before the fix);  inter=3072 -> M3 (9 GiB).  All must pass after the fix, and
#   the small-inter cases must still pass (no regression).
def _gib(inter):
    return E * 2 * inter * H * 2 / 2**30


CASES = []
# decode (small M), fused (k_batch=1): sweep inter below/at/above the boundary
for it in (384, 768, 1280, 1408, 1536, 2048, 3072):
    CASES.append(
        (16, it, 16, 128, 1, f"decode  fused    inter={it:<4} ({_gib(it):.2f}GiB)")
    )
# decode, split-K (k_batch=6): small + M3
for it in (384, 3072):
    CASES.append(
        (16, it, 16, 128, 6, f"decode  split-K  inter={it:<4} ({_gib(it):.2f}GiB)")
    )
# prefill (large M): boundary + M3, both paths
for it, kb in ((1408, 1), (3072, 1), (3072, 2)):
    tag = "split-K" if kb > 1 else "fused  "
    CASES.append(
        (2048, it, 128, 64, kb, f"prefill {tag} inter={it:<4} ({_gib(it):.2f}GiB)")
    )


def test_flydsl_moe_large_inter():
    print(f"\n{'case':40s} {'inter':>5} {'kb':>3} {'cos_dist':>10} {'ms':>9}  status")
    print("-" * 82)
    bad = []
    for ntok, inter, tm, tk, kb, name in CASES:
        x, w1, w2, tw, tid = _make(ntok, inter)
        ref = _ref(x, w1, w2, tw, tid)
        go = run_flydsl(x, w1, w2, tw, tid, inter, tile_m=tm, tile_k=tk, k_batch=kb)
        out = go()
        cd = _cos(out.float(), ref)
        ms = _time_ms(go)
        ok = cd < 0.01
        if not ok:
            bad.append(name)
        print(
            f"{name:40s} {inter:>5} {kb:>3} {cd:>10.2e} {ms:>9.4f}  {'OK' if ok else 'FAIL'}"
        )
        del x, w1, w2, ref, out
        torch.cuda.empty_cache()
    assert not bad, f"cos_dist >= 0.01 for: {bad}"


if __name__ == "__main__":
    test_flydsl_moe_large_inter()
    print("\nAll cases PASS (cos_dist < 0.01).")
