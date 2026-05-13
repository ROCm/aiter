#!/usr/bin/env python3
"""Attention backward test — @flyc.kernel API.

Kernel implementation lives in `flydsl/kernels/attn_bwd_mxfp8_gfx950.py`.
This file is the perf and correctness harness.
"""

import logging
import torch
from aiter.ops.flydsl.kernels.attn_bwd_mxfp8_gfx950 import compile_attn_bwd_mxfp8_gfx950
from utils import run_perftest
from op_tests.flydsl_tests.test_attn_bwd_mxfp8_gfx950 import run_torch, mx_quant, check_result
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)
ARCH = str(get_rocm_arch())
DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3

def bench_attn_bwd_flyc(
    batch, num_heads_q, num_heads_kv, seqlen, head_dim,
    tile_m, tile_n,
    causal,
    test_graph,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
    waves_per_eu: int = 0,
    check_correctness: bool = False
):
    """Attention bwd using the @flyc.kernel / @flyc.jit API."""
    tile_head = head_dim
    print("=" * 80)
    print(
        f"[flyc]  Attention Backward Test (Tile: {tile_m}x{tile_n}x{tile_head})"
    )
    print("=" * 80)
    
    sm_scale = 0.5
    _wpe = int(waves_per_eu) if waves_per_eu else 0
    launch_fn = compile_attn_bwd_mxfp8_gfx950(
        num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, seqlen=seqlen, head_dim=head_dim,
        tile_m=tile_m, tile_n=tile_n, tile_head=tile_head,
        sm_scale=sm_scale,
        causal=causal,
        waves_per_eu=_wpe,
    )
    print(f"✓ Kernel prepared")

    device = torch.device("cuda")
    gqa_size = num_heads_q // num_heads_kv
    q_fp32 = torch.randn(batch, num_heads_q, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    k_fp32 = torch.randn(batch, num_heads_kv, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    v_fp32 = torch.randn(batch, num_heads_kv, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    o_fp32 = torch.randn(batch, num_heads_q, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    do_fp32 = torch.randn(batch, num_heads_q, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5

    q_fp32_head, q_quant_head, q_scale_head = mx_quant(q_fp32, -1)
    q_fp32_m, q_quant_m, q_scale_m = mx_quant(q_fp32, -2)
    k_fp32_head, k_quant_head, k_scale_head = mx_quant(k_fp32, -1)
    k_fp32_n, k_quant_n, k_scale_n = mx_quant(k_fp32, -2)
    v_fp32, v_quant, v_scale = mx_quant(v_fp32)
    do_fp32_head, do_quant_head, do_scale_head = mx_quant(do_fp32, -1)
    do_fp32_m, do_quant_m, do_scale_m = mx_quant(do_fp32, -2)

    k_fp32 = k_fp32.repeat_interleave(gqa_size, dim=1)
    k_fp32_head = k_fp32_head.repeat_interleave(gqa_size, dim=1)
    k_fp32_n = k_fp32_n.repeat_interleave(gqa_size, dim=1)
    v_fp32 = v_fp32.repeat_interleave(gqa_size, dim=1)

    qk = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
    qk = qk * sm_scale
    m = qk.max(dim=-1)[0]
    p = (qk - m[:, :, :, None]).exp()
    l = p.sum(dim=-1)
    m = m + torch.log(l)
    D = (o_fp32 * do_fp32).sum(dim=-1)

    if check_correctness:
        dq_ref, dk_ref, dv_ref = run_torch(q_fp32_head, q_fp32_m, k_fp32_head, k_fp32_n, v_fp32, do_fp32_head, do_fp32_m, m, D, sm_scale, causal, gqa_size)
    dq_fly = torch.zeros((batch, num_heads_q, seqlen, head_dim), dtype=torch.float32, device=device)
    dk_fly = torch.zeros((batch, num_heads_kv, seqlen, head_dim), dtype=torch.float32, device=device)
    dv_fly = torch.zeros((batch, num_heads_kv, seqlen, head_dim), dtype=torch.float32, device=device)

    def launch_kernel(dq, dk, dv, q_quant_head, q_scale_head, q_quant_m, q_scale_m, k_quant_head, k_scale_head, k_quant_n, k_scale_n, v, v_scale, do_quant_head, do_scale_head, do_quant_m, do_scale_m, m, D, batch):
        launch_fn(
            dq.contiguous().view(-1),
            dk.contiguous().view(-1),
            dv.contiguous().view(-1),
            q_quant_head.contiguous().view(-1),
            q_scale_head.contiguous().view(-1),
            q_quant_m.contiguous().view(-1),
            q_scale_m.contiguous().view(-1),
            k_quant_head.contiguous().view(-1),
            k_scale_head.contiguous().view(-1),
            k_quant_n.contiguous().view(-1),
            k_scale_n.contiguous().view(-1),
            v.contiguous().view(-1),
            v_scale.contiguous().view(-1),
            do_quant_head.contiguous().view(-1),
            do_scale_head.contiguous().view(-1),
            do_quant_m.contiguous().view(-1),
            do_scale_m.contiguous().view(-1),
            m.contiguous().view(-1),
            D.contiguous().view(-1),
            batch,
            q_quant_head.stride(0),
            q_scale_head.stride(0),
            k_quant_head.stride(0),
            k_scale_head.stride(0),
            m.stride(0),
            q_quant_head.stride(1),
            q_scale_head.stride(1),
            m.stride(1),
            torch.cuda.current_stream(),
        )

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        dq_fly,
        dk_fly,
        dv_fly,
        q_quant_head,
        q_scale_head,
        q_quant_m,
        q_scale_m,
        k_quant_head,
        k_scale_head,
        k_quant_n,
        k_scale_n,
        v_quant,
        v_scale,
        do_quant_head,
        do_scale_head,
        do_quant_m,
        do_scale_m,
        m,
        D,
        batch,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
    )
    torch.cuda.synchronize()

    dq_fly.zero_()
    dk_fly.zero_()
    dv_fly.zero_()
    launch_kernel(
        dq_fly,
        dk_fly,
        dv_fly,
        q_quant_head,
        q_scale_head,
        q_quant_m,
        q_scale_m,
        k_quant_head,
        k_scale_head,
        k_quant_n,
        k_scale_n,
        v_quant,
        v_scale,
        do_quant_head,
        do_scale_head,
        do_quant_m,
        do_scale_m,
        m,
        D,
        batch
    )

    dq_fly_fp32 = dq_fly.to(torch.float32)
    dk_fly_fp32 = dk_fly.to(torch.float32)
    dv_fly_fp32 = dv_fly.to(torch.float32)

    if check_correctness:
        assert check_result(dq_fly_fp32, dq_ref, rtol=0.01, atol=0.01)
        assert check_result(dk_fly_fp32, dk_ref, rtol=0.01, atol=0.01)
        assert check_result(dv_fly_fp32, dv_ref, rtol=0.01, atol=0.01)

    bytes_moved = (4 + 4) * batch * num_heads_q * seqlen * head_dim + (3 + 2 * 4) * batch * num_heads_kv * seqlen * head_dim + 2 * 4 * batch * num_heads_q * seqlen
    flops = batch * num_heads_q * (5 * 2 * seqlen * seqlen * head_dim + 5 * seqlen * seqlen + 2 * 3 * seqlen * seqlen)
    if causal:
        flops /= 2
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preshuffle GEMM benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_heads_q", type=int, default=128)
    parser.add_argument("--num_heads_kv", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--tile_m", type=int, default=128)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--waves_per_eu", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--test_graph", action="store_true", default=False)
    parser.add_argument("--check_correctness", action="store_true", default=False)
    args = parser.parse_args()
    torch.set_default_device("cuda")

    bench_attn_bwd_flyc(
        batch=args.batch, num_heads_q=args.num_heads_q, num_heads_kv=args.num_heads_kv, seqlen=args.seqlen, head_dim=args.head,
        tile_m=args.tile_m, tile_n=args.tile_n,
        causal=args.causal,
        test_graph=bool(args.test_graph),
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
        waves_per_eu=int(args.waves_per_eu),
        check_correctness=args.check_correctness
    )
