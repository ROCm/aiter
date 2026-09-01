# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness + performance test for the gfx950 assembly MTP-verify attention
# (aiter.mtp_verify_attn_fwd_asm): 4 query tokens per sequence over an NHD
# page-16 fp8 KV cache, head_dim 256, GQA ratio 16 (TP2 shape) or 8 (TP4).
import argparse
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest

torch.set_default_device("cuda")
SUPPORTED_GFX = ["gfx950"]
QLEN, HD, PAGE = 4, 256, 16


def make_inputs(ctx, num_seqs, num_q_heads, num_kv_heads, k_descale=1.0, v_descale=1.0, replicas=1, seed=0):
    """Random paged KV cache with a shuffled page table (pages of one sequence
    are scattered over the pool, as a real allocator would leave them)."""
    torch.manual_seed(seed)
    kvlen = ctx + QLEN
    npages = (kvlen + PAGE - 1) // PAGE
    perm = torch.randperm(num_seqs * npages)
    bt = perm.to(torch.int32).view(num_seqs, npages)
    caches = []
    for _ in range(replicas):
        k = (torch.randn(num_seqs * npages, PAGE, num_kv_heads, HD) / 4).to(dtypes.fp8)
        v = (torch.randn(num_seqs * npages, PAGE, num_kv_heads, HD) / 4).to(dtypes.fp8)
        k2, v2 = torch.empty_like(k), torch.empty_like(v)
        k2[perm], v2[perm] = k, v
        caches.append((k2, v2))
    q = (torch.randn(num_seqs * QLEN, num_q_heads, HD) / 4).to(dtypes.bf16)
    cu_q = torch.arange(0, (num_seqs + 1) * QLEN, QLEN, dtype=torch.int32)
    seq_lens = torch.full((num_seqs,), kvlen, dtype=torch.int64)
    kd = torch.full((1,), k_descale, dtype=torch.float32)
    vd = torch.full((1,), v_descale, dtype=torch.float32)
    return caches, bt, q, cu_q, seq_lens, kd, vd, kvlen


def torch_ref(k2, v2, bt, q, seq_lens, kd, vd, num_q_heads, num_kv_heads):
    num_seqs = bt.shape[0]
    gqa = num_q_heads // num_kv_heads
    outs = []
    for s in range(num_seqs):
        kvlen = int(seq_lens[s])
        pages = bt[s].long()
        kk = k2[pages].reshape(-1, num_kv_heads, HD)[:kvlen].float() * kd
        vv = v2[pages].reshape(-1, num_kv_heads, HD)[:kvlen].float() * vd
        qq = q[s * QLEN:(s + 1) * QLEN].float()
        o = torch.empty(QLEN, num_q_heads, HD)
        for t in range(QLEN):
            L = kvlen - QLEN + t + 1  # causal: query t sees the first L keys
            for h in range(num_q_heads):
                kvh = h // gqa
                sc = (qq[t, h] @ kk[:L, kvh].T) / math.sqrt(HD)
                o[t, h] = torch.softmax(sc, dim=-1) @ vv[:L, kvh]
        outs.append(o)
    return torch.cat(outs).to(dtypes.bf16)


def test_correctness(ctx, num_seqs, num_q_heads, num_kv_heads, k_descale, v_descale):
    caches, bt, q, cu_q, seq_lens, kd, vd, kvlen = make_inputs(
        ctx, num_seqs, num_q_heads, num_kv_heads, k_descale, v_descale)
    k2, v2 = caches[0]
    out = aiter.mtp_verify_attn_fwd_asm(q, k2, v2, bt, seq_lens, cu_q, kd, vd, 1.0 / math.sqrt(HD))
    ref = torch_ref(k2, v2, bt, q, seq_lens, kd, vd, num_q_heads, num_kv_heads)
    tag = f"ctx={ctx} seqs={num_seqs} q_heads={num_q_heads} kv_heads={num_kv_heads} kd={k_descale} vd={v_descale}"
    # Q and P are fp8-quantized inside the kernel: error is ~0.13 of the output std.
    checkAllclose(ref.float(), out.float(), atol=4e-3, rtol=0.2, msg=tag)
    return (out.float() - ref.float()).abs().max().item() / ref.float().std().item()


def test_perf(ctx, num_seqs, num_q_heads, num_kv_heads, replicas=8):
    caches, bt, q, cu_q, seq_lens, kd, vd, kvlen = make_inputs(
        ctx, num_seqs, num_q_heads, num_kv_heads, replicas=replicas)
    scale = 1.0 / math.sqrt(HD)
    num_segments = aiter.mtp_verify_attn_num_segments(num_seqs, num_kv_heads)
    state = {"i": 0}

    def run():
        k2, v2 = caches[state["i"] % replicas]  # rotate replicas so the KV read is cold
        state["i"] += 1
        return aiter.mtp_verify_attn_fwd_asm(q, k2, v2, bt, seq_lens, cu_q, kd, vd, scale,
                                             num_segments=num_segments)

    _, us = run_perftest(run, num_iters=64, num_warmup=4)
    kv_bytes = num_seqs * kvlen * num_kv_heads * HD * 2
    return us, kv_bytes / us / 1e6, num_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true", help="also run the bandwidth benchmark")
    args = parser.parse_args()
    if get_gfx() not in SUPPORTED_GFX:
        print(f"skip: mtp_verify_attn asm kernel is only shipped for {SUPPORTED_GFX}, got {get_gfx()}")
        return
    rows = []
    for ctx, seqs, hq, hkv, kd, vd in [
        (1000, 2, 16, 1, 1.0, 1.0),   # TP2 shape, GQA 16
        (1000, 2, 16, 1, 0.5, 2.0),   # per-tensor KV descales
        (1000, 2, 8, 1, 1.0, 1.0),    # TP4 shape, GQA 8
        (777, 2, 32, 2, 1.0, 1.0),    # 2 KV heads, ragged page count
        (8000, 3, 16, 1, 1.0, 1.0),   # empty tail segments
    ]:
        err = test_correctness(ctx, seqs, hq, hkv, kd, vd)
        rows.append(dict(ctx=ctx, seqs=seqs, q_heads=hq, kv_heads=hkv, k_descale=kd, v_descale=vd,
                         max_err_over_std=round(err, 3)))
    print(pd.DataFrame(rows).to_markdown(index=False))
    if args.perf:
        rows = []
        for ctx, seqs, hq, hkv in [(35000, 16, 16, 1), (70000, 16, 16, 1), (35000, 16, 8, 1), (35000, 4, 16, 1)]:
            us, tbps, segs = test_perf(ctx, seqs, hq, hkv)
            rows.append(dict(ctx=ctx, seqs=seqs, q_heads=hq, kv_heads=hkv, segments=segs,
                             us=round(us, 1), kv_TBps=round(tbps, 2)))
        print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
