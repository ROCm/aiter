#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent v4_pro MegaMoEV2 accuracy and performance test."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import aiter
import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.nn.functional as F
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils


NETWORKS = {
    "v4_pro": dict(model_dim=7168, inter_dim=3072, experts=384, topk=6),
}


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _cleanup():
    try:
        ms.shmem_finalize()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _reduce_float(value, device, op):
    result = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(result, op=op)
    return float(result.item())


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _make_inputs(tokens, model_dim, experts, topk, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn((tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator)
    scores = torch.randn((tokens, experts), dtype=torch.float32, device=device, generator=generator)
    values, ids = torch.topk(scores, topk, dim=-1)
    return x.contiguous(), values.softmax(dim=-1).contiguous(), ids.to(torch.int32).contiguous()


def _quantize_weights(model_dim, inter_dim, local_experts, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)

    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_scale_ref = w1_scale.view(local_experts, 2 * inter_dim, model_dim // 32)
    del w1
    w1_kernel = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale_kernel = shuffle_scale_a16w4(w1_scale, local_experts, True).contiguous()

    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(inter_dim**-0.25)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    w2_scale_ref = w2_scale.view(local_experts, model_dim, inter_dim // 32)
    del w2
    w2_kernel = shuffle_weight_a16w4(w2_q, 16, False).contiguous()
    w2_scale_kernel = shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous()
    torch.cuda.empty_cache()
    return w1_kernel, w1_scale_kernel, w2_kernel, w2_scale_kernel, w1_q, w1_scale_ref, w2_q, w2_scale_ref


def _dequant_expert(weight, scale, rows, cols):
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


@torch.no_grad()
def _reference(x, route_weights, ids, ref_weights, rank, world, model_dim, inter_dim, experts):
    x_all, weights_all, ids_all = _all_gather(x), _all_gather(route_weights), _all_gather(ids)
    partial = torch.zeros((x_all.shape[0], model_dim), dtype=torch.float32, device=x.device)
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = experts // world
    expert_start = rank * local_experts
    expert_end = expert_start + local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_end)]
    for expert in active.tolist():
        positions = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = positions[:, 0], positions[:, 1]
        local_id = expert - expert_start
        w1 = _dequant_expert(w1_q[local_id], w1_scale[local_id], 2 * inter_dim, model_dim)
        w2 = _dequant_expert(w2_q[local_id], w2_scale[local_id], model_dim, inter_dim)
        inp = x_all[rows].float()
        hidden = F.silu(inp @ w1[:inter_dim].T) * (inp @ w1[inter_dim:].T)
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
        del w1, w2, inp, hidden, out
    dist.all_reduce(partial)
    start = rank * x.shape[0]
    return partial[start : start + x.shape[0]]


def _time_graph(fn, device, iters):
    _barrier()
    fn()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean_ms = _reduce_float(local_ms, device, dist.ReduceOp.SUM) / dist.get_world_size()
    max_ms = _reduce_float(local_ms, device, dist.ReduceOp.MAX)
    return mean_ms, max_ms


def _run_size(moe, x, weights, ids, ref_weights, args, rank, world, device):
    tokens = x.shape[0]
    output = moe(x, weights, ids)[:tokens]
    _barrier()
    rel_l2 = -1.0
    if tokens <= args.accuracy_max_bs:
        reference = _reference(
            x,
            weights,
            ids,
            ref_weights,
            rank,
            world,
            moe.model_dim,
            moe.inter_dim,
            moe.experts,
        )
        rel_l2 = float(torch.linalg.vector_norm(output.float() - reference) / torch.linalg.vector_norm(reference))
        rel_l2 = _reduce_float(rel_l2, device, dist.ReduceOp.MAX)
        if rel_l2 >= args.rtol:
            raise AssertionError(f"bs={tokens} relL2={rel_l2:.6f} exceeds {args.rtol}")

    x_q, scale = moe.quantize(x)
    state = {}

    def stage1():
        moe._run_fused_stage1(x_q, weights, scale, ids)

    def stage2():
        state["output"] = moe._run_stage2(tokens, None, True, moe._active_config)

    def end_to_end():
        state["output"] = moe(x, weights, ids)

    stage1_ms = _time_graph(stage1, device, args.iters)
    stage1()
    _barrier()
    stage2_ms = _time_graph(stage2, device, args.iters)
    e2e_ms = _time_graph(end_to_end, device, args.iters)
    sbm = int(moe._s1_active_tile_m)
    gemm2_bm = int(moe._g2_active_block_m)
    p2p_quant = moe._active_config.p2p_quant
    if rank == 0:
        print(
            f"[MEGA-V2] bs={tokens} relL2={rel_l2:.6f} "
            f"path={'fixed' if moe._s1_fixed_slot else 'compact'} "
            f"p2p_quant={p2p_quant} SBM={sbm} G2_BM={gemm2_bm} "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms "
            f"e2e={e2e_ms[0]:.4f}/{e2e_ms[1]:.4f}ms mean/max",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, default="v4_pro")
    parser.add_argument("--bs-list", default="128")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--accuracy-max-bs", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--max-tok-per-rank", type=int)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.bs_list.split(",")]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("--bs-list must contain positive integers")

    rank, world, device = _setup_dist()
    try:
        network = NETWORKS[args.network]
        if network["experts"] % world:
            raise ValueError(f"experts={network['experts']} must be divisible by world={world}")
        if args.max_tok_per_rank is not None and args.max_tok_per_rank < max(batch_sizes):
            raise ValueError("--max-tok-per-rank must cover the largest batch size")
        local_experts = network["experts"] // world
        packed = _quantize_weights(
            network["model_dim"],
            network["inter_dim"],
            local_experts,
            rank,
            args.seed,
            device,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        max_bs = max(batch_sizes)
        x, weights, ids = _make_inputs(
            max_bs,
            network["model_dim"],
            network["experts"],
            network["topk"],
            rank,
            args.seed,
            device,
        )
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
        for batch_size in batch_sizes:
            max_tok_per_rank = args.max_tok_per_rank or max(16, _next_power_of_two(batch_size))
            moe = MegaMoEV2(
                rank=rank,
                world_size=world,
                quant="a8w4",
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=max_tok_per_rank,
                **network,
            )
            _run_size(
                moe,
                x[:batch_size].contiguous(),
                weights[:batch_size].contiguous(),
                ids[:batch_size].contiguous(),
                ref_weights,
                args,
                rank,
                world,
                device,
            )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
