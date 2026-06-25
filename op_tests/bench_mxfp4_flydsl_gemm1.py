# SPDX-License-Identifier: MIT
"""Microbench for the FlyDSL mxfp4 gemm1 (a4w4) port — v1(raw) vs v2(layout API).

Builds the KIMI BM32 inputs exactly like test_flydsl_gemm1_parametrized_shape_numeric
and times flydsl_mxfp4_gemm1 over many launches. Select backend via env
AITER_MXFP4_GEMM1_V2={0,1} (the kernels dispatch reads it). Usage:

  /opt/venv/bin/python op_tests/bench_mxfp4_flydsl_gemm1.py --M 256 --iters 200
"""
import argparse
import torch

from op_tests.test_mxfp4_flydsl_gemm1 import (
    _torch_a_scale_sorted_shuffled,
    _torch_threestage_sort,
)


def _setup(M, NE=385, H=7168, INTER=512, TOPK=9, BM=32, interleave=True, seed=2):
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    device = torch.device("cuda")
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    torch.manual_seed(seed)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w1u8 = shuffle_weight_a16w4(w1q, 16, interleave).view(torch.uint8)
    w1_scale_u8 = shuffle_scale_a16w4(w1s, NE, interleave).view(torch.uint8)

    torch.manual_seed(seed + 1)
    hidden = torch.randn((M, H), dtype=dtypes.bf16, device=device) / 10
    g = torch.Generator(device=device).manual_seed(seed + 1)
    bias = torch.randn(NE - 1, generator=g, device=device) * 0.5
    scores = torch.randn(M, NE - 1, generator=g, device=device) + bias
    rw, rid = torch.topk(scores.softmax(-1), TOPK - 1, dim=-1)
    sid = torch.full((M, 1), NE - 1, device=device, dtype=rid.dtype)
    sw = torch.ones((M, 1), device=device, dtype=rw.dtype)
    topk_ids = torch.cat([sid, rid], 1).to(torch.int32)
    topk_weight = torch.cat([sw, rw], 1).to(torch.float32)

    active = min(NE, M * TOPK)
    max_sorted = ((M * TOPK + active * (BM - 1) + BM - 1) // BM) * BM
    sti, sei, cumsum, mind = _torch_threestage_sort(
        topk_ids, topk_weight, M, NE, TOPK, BM, max_sorted
    )
    aq, asc = tq(hidden, quant_dtype=dtypes.fp4x2)
    aq = aq.view(torch.uint8).view(M, H // 2).contiguous()
    asc = asc.view(torch.uint8).view(M, H // 32).contiguous()
    assh = _torch_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=BM)

    isq = torch.zeros((max_sorted, INTER // 2), device=device, dtype=torch.uint8)
    isc = INTER // 32
    N_OUT = 2 * INTER
    isr = (((max_sorted * (N_OUT // 64) * 4) + isc - 1) // isc + 31) // 32 * 32
    iss = torch.zeros((isr, isc), device=device, dtype=torch.uint8)
    return dict(
        a_quant=aq, a_scale_sorted_shuffled=assh, w1_u8=w1u8, w1_scale_u8=w1_scale_u8,
        sorted_expert_ids=sei, cumsum_tensor=cumsum, m_indices=mind,
        inter_sorted_quant=isq, inter_sorted_shuffled_scale=iss, hidden_states=hidden,
        n_tokens=M, BM=BM, use_nt=True, inline_quant=False, NE=NE, D_HIDDEN=H,
        D_INTER=INTER, topk=TOPK, interleave=interleave,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

    kw = _setup(args.M)
    for _ in range(args.warmup):
        flydsl_mxfp4_gemm1(**kw)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        flydsl_mxfp4_gemm1(**kw)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) / args.iters * 1000.0
    print(f"M={args.M} iters={args.iters}  per-launch = {us:.2f} us")


if __name__ == "__main__":
    main()
