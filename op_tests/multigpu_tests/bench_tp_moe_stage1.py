# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage breakdown for TPMoEStage1, to size the phase-2 fused all-gather.

The question this answers: how much time can a fused, overlapped all-gather
actually save? The ceiling is NOT the all-gather time -- overlap can only hide
as much communication as there is compute to hide it behind:

    max_saving = min(T_allgather, T_gemm1)

So both numbers are needed. A second discriminator is the BF16 entry versus the
prequantized entry: they do identical work except the gather moves 14336 vs
7392 bytes per row. If halving the bytes roughly halves the gather time, the
gather is bandwidth-bound and overlap will help. If the time barely moves, it
is latency/sync-bound and overlap cannot help either.

Run with:

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/bench_tp_moe_stage1.py
"""

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import TPMoEStage1
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
from aiter.utility.fp4_utils import moe_mxfp4_sort
from tp_moe_stage1_ref import build_mxfp4_w1

NETWORK = dict(model_dim=7168, experts=384, topk=6, swiglu_limit=10.0)
STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    return rank, world, device


def _random_routes(m, experts, topk, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(experts, generator=g)[:topk] for _ in range(m)]
    ).to(device=device, dtype=torch.int32)
    w = torch.rand((m, topk), generator=g).to(device=device, dtype=torch.float32)
    return ids, w / w.sum(dim=-1, keepdim=True)


def _stages_bf16(op, x, wts, ids, m_global):
    """Replay forward() with events between stages. Returns 4 ms values."""
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    ev[0].record()
    x_g, wts_g, ids_g = op._all_gather_inputs(x, wts, ids)
    ev[1].record()
    sids, sw, seids, nvalid = op._sort(ids_g, wts_g)
    ev[2].record()
    a_fp8, a_scale = fused_dynamic_mxfp8_quant_moe_sort(
        x_g,
        sorted_ids=sids,
        num_valid_ids=nvalid,
        token_num=m_global,
        topk=op.topk,
        block_size=op.sort_block_m,
        sorted_weights=None,
    )
    ev[3].record()
    op._run_gemm1(a_fp8, a_scale, sids, seids, nvalid)
    ev[4].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(4)]


def _stages_prequant(op, x_q, x_scale, wts, ids, m_global):
    """Replay forward_prequant() with events between stages. Returns 4 ms values."""
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    ev[0].record()
    x_g, wts_g, ids_g = op._all_gather_inputs(x_q, wts, ids)
    scale_g = op._all_gather_one(x_scale)
    ev[1].record()
    sids, sw, seids, nvalid = op._sort(ids_g, wts_g)
    ev[2].record()
    a_scale = moe_mxfp4_sort(
        scale_g.view(m_global, 1, -1), sids, nvalid, m_global, op.sort_block_m
    )
    ev[3].record()
    op._run_gemm1(x_g, a_scale, sids, seids, nvalid)
    ev[4].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(4)]


def _median_stages(fn, iters, warmup):
    """Barrier before each timed iteration so the reading excludes rank skew."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        samples.append(fn())
    return [statistics.median(s[i] for s in samples) for i in range(len(samples[0]))]


def _max_across_ranks(values, device):
    """The slowest rank sets the end-to-end cost, so reduce with MAX."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-local", default="1,8,64,128")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inter-dim", type=int, default=384)
    args = ap.parse_args()

    rank, world, device = _setup_dist()
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    inter_dim = args.inter_dim

    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=2026
    )
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )

    bf16_row = model_dim * 2
    fp8_row = model_dim + model_dim // 32

    if rank == 0:
        print(
            f"world={world} model_dim={model_dim} inter_dim={inter_dim} "
            f"experts={experts} topk={topk} iters={args.iters}"
        )
        print(f"gather row bytes: bf16={bf16_row}  fp8+scale={fp8_row}")
        print()

    for m_local in [int(v) for v in args.m_local.split(",")]:
        gx = torch.Generator(device="cpu").manual_seed(7000 + rank * 31 + m_local)
        x = torch.randn((m_local, model_dim), generator=gx).to(
            device=device, dtype=torch.bfloat16
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=99 + rank)
        x_q, x_scale = op.quantize(x)
        m_global = op.m_logical_for(m_local)

        b = _median_stages(
            lambda: _stages_bf16(op, x, wts, ids, m_global), args.iters, args.warmup
        )
        p = _median_stages(
            lambda: _stages_prequant(op, x_q, x_scale, wts, ids, m_global),
            args.iters,
            args.warmup,
        )
        merged = _max_across_ranks(b + p, device)
        b, p = merged[:4], merged[4:]

        if rank != 0:
            continue

        # Incoming traffic per rank: it receives (world-1) peers' shards.
        rx_bf16 = (world - 1) * m_local * bf16_row
        rx_fp8 = (world - 1) * m_local * fp8_row
        bw_bf16 = rx_bf16 / (b[0] * 1e-3) / 1e9
        bw_fp8 = rx_fp8 / (p[0] * 1e-3) / 1e9

        tot_b, tot_p = sum(b), sum(p)
        ceiling_b = min(b[0], b[3])
        ceiling_p = min(p[0], p[3])

        print(f"===== m_local={m_local}  m_global={m_global} =====")
        print(
            f"  {'stage':<14}{'bf16 entry':>12}{'fp8 entry':>12}   (ms, median, max over ranks)"
        )
        print(f"  {'all-gather':<14}{b[0]:>12.4f}{p[0]:>12.4f}")
        print(f"  {'moe_sorting':<14}{b[1]:>12.4f}{p[1]:>12.4f}")
        print(f"  {'quant/scale':<14}{b[2]:>12.4f}{p[2]:>12.4f}")
        print(f"  {'gemm1':<14}{b[3]:>12.4f}{p[3]:>12.4f}")
        print(f"  {'TOTAL':<14}{tot_b:>12.4f}{tot_p:>12.4f}")
        print(
            f"  gather rx/rank: bf16 {rx_bf16/1e6:.2f} MB @ {bw_bf16:.1f} GB/s | "
            f"fp8 {rx_fp8/1e6:.2f} MB @ {bw_fp8:.1f} GB/s"
        )
        print(
            f"  gather share of total: bf16 {100*b[0]/tot_b:.1f}%  "
            f"fp8 {100*p[0]/tot_p:.1f}%"
        )
        print(
            f"  overlap ceiling min(T_gather,T_gemm1): bf16 {ceiling_b:.4f} ms "
            f"({100*ceiling_b/tot_b:.1f}% of total)  |  fp8 {ceiling_p:.4f} ms "
            f"({100*ceiling_p/tot_p:.1f}% of total)"
        )
        print(
            f"  quantize-before-gather already saves: "
            f"{b[0]-p[0]:+.4f} ms on the gather, {tot_b-tot_p:+.4f} ms on the total"
        )
        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
