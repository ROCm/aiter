# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness test for the multi-group FP8 GQA paged-attention decode path
# (mtp==3 / ql=3, block_size 16 and 64, fp8 OR bf16 query), dispatched from
# `PagedAttention.forward_decode` -> `pa_fp8_decode_v2_mg`.
#
# The kernel applies a per-token MTP causal mask: query token j (0..mtp-1)
# attends only to the first ctx-(mtp-1-j) KV tokens.  bf16 query is quantized
# to fp8 in-kernel (per (token,head) max-abs), matching the v2 path; the
# reference emulates that quant so the residual is fp8-quantization noise.
#
# Usage:
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_fp8_gqa_mg.py
#   HIP_VISIBLE_DEVICES=0 python op_tests/test_pa_fp8_gqa_mg.py --qdtype fp8 bf16 --block 16 64
import argparse
import math
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.paged_attn import PagedAttention
from aiter.ops.attention import _pa_fp8_gqa_eligible

NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_SIZE = 128
PARTITION_SIZE = 256
FP8 = dtypes.fp8


def _make_inputs(bs, ctx, mtp, block_size, q_dtype, seed=0):
    torch.manual_seed(seed + bs * 31 + ctx * 17 + mtp + block_size)
    bps = (ctx + block_size - 1) // block_size
    nb = bs * bps
    fp8_max = torch.finfo(FP8).max

    q_bf16 = torch.randn(bs * mtp, NUM_Q_HEADS, HEAD_SIZE, dtype=dtypes.bf16, device="cuda")
    k_bf16 = torch.randn(nb, NUM_KV_HEADS, HEAD_SIZE, block_size, dtype=dtypes.bf16, device="cuda")
    v_bf16 = torch.randn_like(k_bf16)

    k_scale = (k_bf16.float().abs().amax(dim=2) / fp8_max).clamp_min(1e-6)    # [nb, 1, block]
    v_scale = (
        v_bf16.float().abs().permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1).amax(dim=1)
        / fp8_max
    ).clamp_min(1e-6)                                                         # [1]

    if q_dtype == "fp8":
        q_scale = (q_bf16.float().abs().amax(dim=-1) / fp8_max).clamp_min(1e-6)  # [bs*mtp, 8]
        query = (q_bf16.float() / q_scale[:, :, None]).clamp(-fp8_max, fp8_max).to(FP8)
        # q_deq the kernel effectively sees.
        q_deq = query.float() * q_scale[:, :, None]
        q_scale_arg = q_scale
    else:  # bf16 query: kernel quantizes in-kernel (per token-head max-abs).
        query = q_bf16
        s = (q_bf16.float().abs().amax(dim=-1, keepdim=True) / fp8_max).clamp_min(1e-12)
        q_deq = (q_bf16.float() / s).clamp(-fp8_max, fp8_max).to(FP8).float() * s
        q_scale_arg = None

    k_cache = (
        (k_bf16.float() / k_scale[:, :, None, :]).clamp(-fp8_max, fp8_max).to(FP8)
        .view(nb, NUM_KV_HEADS, HEAD_SIZE // 16, 16, block_size)
        .permute(0, 1, 2, 4, 3).contiguous()
    )
    value_cache = (
        v_bf16.float() / v_scale[None, :, None, None]
    ).clamp(-fp8_max, fp8_max).to(FP8)  # 4D [nb, nkv, hd, block]

    block_tables = torch.arange(nb, dtype=torch.int32, device="cuda").view(bs, bps)
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device="cuda")
    return dict(query=query, k_cache=k_cache, value_cache=value_cache,
                block_tables=block_tables, seq_lens=seq_lens, q_scale_arg=q_scale_arg,
                k_scale=k_scale, v_scale=v_scale, q_deq=q_deq)


def _reference(d, *, mtp, block_size, scale, causal=True):
    seq_lens = d["seq_lens"]; k_cache = d["k_cache"]; value_cache = d["value_cache"]
    k_scale = d["k_scale"]; v_scale = d["v_scale"]; q_deq = d["q_deq"]
    bs = seq_lens.numel()
    fp8_max = torch.finfo(FP8).max
    out = torch.empty(q_deq.shape[0], NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float32,
                      device=q_deq.device)
    for s in range(bs):
        L = int(seq_lens[s].item())
        bps = (L + block_size - 1) // block_size
        kt = torch.empty(L, HEAD_SIZE, dtype=torch.float32, device=q_deq.device)
        vt = torch.empty_like(kt)
        for pos in range(L):
            blk = s * bps + pos // block_size
            slot = pos % block_size
            kt[pos] = k_cache[blk, 0, :, slot, :].reshape(HEAD_SIZE).float() * k_scale[blk, 0, slot]
            vt[pos] = value_cache[blk, 0, :, slot].float() * v_scale[0]
        for mi in range(mtp):
            qi = s * mtp + mi
            vu = (L - (mtp - 1 - mi)) if causal else L
            for h in range(NUM_Q_HEADS):
                logits = (kt[:vu] @ q_deq[qi, h]) * scale
                probs = torch.exp(logits - logits.max())
                denom = probs.sum()
                probs = probs.clamp(-fp8_max, fp8_max).to(FP8).float()  # match kernel P pack
                out[qi, h] = (probs @ vt[:vu]) / denom
    return out


def _run_kernel(d, ctx, mtp, block_size):
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    out = torch.empty_like(d["query"], dtype=dtypes.bf16)
    assert _pa_fp8_gqa_eligible(
        d["query"], d["k_cache"], d["value_cache"], out, NUM_KV_HEADS, block_size,
        PARTITION_SIZE, mtp, None, None, d["q_scale_arg"], d["k_scale"], d["v_scale"],
        None, None,
    ), f"not eligible: bs={d['seq_lens'].numel()} ctx={ctx} mtp={mtp} block={block_size}"
    return PagedAttention.forward_decode(
        d["query"], d["k_cache"], d["value_cache"], d["block_tables"], d["seq_lens"], ctx,
        kv_cache_dtype="auto", num_kv_heads=NUM_KV_HEADS, scale=scale, alibi_slopes=None,
        k_scale=d["k_scale"], v_scale=d["v_scale"], q_scale=d["q_scale_arg"], mtp=mtp,
        p_scale=None, p_scale_inv=None,
    ).float()


def _one_case(bs, ctx, mtp, block_size, q_dtype, atol=6e-2):
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    d = _make_inputs(bs, ctx, mtp, block_size, q_dtype)
    out = _run_kernel(d, ctx, mtp, block_size)
    ref_c = _reference(d, mtp=mtp, block_size=block_size, scale=scale, causal=True)
    ref_u = _reference(d, mtp=mtp, block_size=block_size, scale=scale, causal=False)
    dc = (out - ref_c).abs().max().item()
    du = (out - ref_u).abs().max().item()
    # PASS = kernel matches the causal reference within tolerance.
    acc_ok = dc <= atol
    # The causal-vs-uniform direction check is only meaningful when the two
    # references differ enough to distinguish (short ctx: masking 1-2 of few
    # KV tokens moves the output a lot).  At long ctx they converge (masking
    # 1-2 of thousands is negligible), so dc≈du and the sign is pure noise —
    # do NOT require causal>uniform there.
    causal_distinguishable = du > max(3.0 * atol, 5.0 * dc)
    is_causal = dc < du
    if causal_distinguishable:
        tag = "CAUSAL" if is_causal else "UNIFORM"
        passed = acc_ok and is_causal
    else:
        tag = "n/a(long-ctx)"
        passed = acc_ok
    return {
        "bs": bs, "ctx": ctx, "mtp": mtp, "block": block_size, "q": q_dtype,
        "max_abs_causal": round(dc, 5), "max_abs_uniform": round(du, 5),
        "kernel": tag,
        "acc_pass": bool(passed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 8])
    ap.add_argument("--ctx", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--mtp", type=int, nargs="+", default=[3])
    ap.add_argument("--block", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--qdtype", type=str, nargs="+", default=["fp8", "bf16"],
                    choices=["fp8", "bf16"])
    ap.add_argument("--atol", type=float, default=6e-2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        aiter.logger.warning("CUDA/HIP device unavailable, skip test_pa_fp8_gqa_mg.")
        sys.exit(0)

    rows = []
    for q_dtype in args.qdtype:
        for block_size in args.block:
            for mtp in args.mtp:
                for bs in args.bs:
                    for ctx in args.ctx:
                        rows.append(_one_case(bs, ctx, mtp, block_size, q_dtype, atol=args.atol))
    df = pd.DataFrame(rows)
    aiter.logger.info("FP8 GQA multi-group (mtp>=3) summary:\n%s", df.to_markdown(index=False))
    n_pass = int(df["acc_pass"].sum())
    n_total = len(df)
    aiter.logger.info("Accuracy: %d/%d cells pass", n_pass, n_total)
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
