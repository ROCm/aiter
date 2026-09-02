# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent v4_pro MegaMoEV2 accuracy and performance test."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

NETWORKS = {
    "v4_pro": {
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "swiglu_limit": 10.0,
    },
    # Kimi-K3 routing/weight geometry.  This exercises topk16 and EP8/epr112;
    # the numerical reference intentionally keeps MegaMoEV2's current bounded
    # SwiGLU activation while the K3 activation integration remains separate.
    "kimi_k3_route": {
        "model_dim": 3584,
        "inter_dim": 512,
        "experts": 896,
        "topk": 16,
        "swiglu_limit": 10.0,
    },
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


def _make_inputs(
    tokens,
    model_dim,
    experts,
    topk,
    rank,
    seed,
    device,
    *,
    force_fanout_boundary,
    inject_invalid_route,
    force_padding_boundary,
):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    values, ids = torch.topk(scores, topk, dim=-1)
    if force_fanout_boundary or inject_invalid_route or force_padding_boundary:
        if topk != 16 or experts % dist.get_world_size():
            raise ValueError("fanout adversarial cases require topk=16 and EP")
        world = dist.get_world_size()
        local_experts = experts // world
        if local_experts < 112:
            raise ValueError("fanout adversarial cases require at least 112 epr")
        token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
        slot = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
        if force_padding_boundary:
            if tokens != 1:
                raise ValueError("--force-padding-boundary requires one token")
            local = (rank * topk + slot) % local_experts
            ids = local.expand(tokens, -1).clone()
            if rank == world - 1:
                ids[:, :-2] = torch.arange(
                    1, topk - 1, dtype=torch.int64, device=device
                )
                ids[:, -2] = 0
                ids[:, -1] = local_experts - 1
        else:
            destination = (token + rank) % world
            local = (token * 17 + slot * 7) % (local_experts - 2)
            if inject_invalid_route:
                local = local + 1
            ids = destination * local_experts + local
            if inject_invalid_route:
                ids[:, -2] = -1
                ids[:, -1] = destination[:, 0] * local_experts + local_experts - 1
            else:
                ids[:, -2] = destination[:, 0] * local_experts + local_experts - 2
                ids[:, -1] = destination[:, 0] * local_experts + local_experts - 1
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


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
    return (
        w1_kernel,
        w1_scale_kernel,
        w2_kernel,
        w2_scale_kernel,
        w1_q,
        w1_scale_ref,
        w2_q,
        w2_scale_ref,
    )


def _dequant_expert(weight, scale, rows, cols):
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather_variable(tensor, token_counts=None):
    world = dist.get_world_size()
    if token_counts is None:
        local_count = torch.tensor(
            tensor.shape[0], dtype=torch.int64, device=tensor.device
        )
        gathered_counts = [torch.empty_like(local_count) for _ in range(world)]
        dist.all_gather(gathered_counts, local_count)
        token_counts = [int(count.item()) for count in gathered_counts]
    max_count = max(token_counts)
    if tensor.shape[0] < max_count:
        padding = torch.empty(
            (max_count - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding))
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor)
    return (
        torch.cat([value[:count] for value, count in zip(gathered, token_counts)]),
        token_counts,
    )


@torch.no_grad()
def _reference(
    x,
    route_weights,
    ids,
    ref_weights,
    rank,
    world,
    model_dim,
    inter_dim,
    experts,
    swiglu_limit,
):
    x_all, token_counts = _all_gather_variable(x)
    weights_all, _ = _all_gather_variable(route_weights, token_counts)
    ids_all, _ = _all_gather_variable(ids, token_counts)
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
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
        w1 = _dequant_expert(
            w1_q[local_id], w1_scale[local_id], 2 * inter_dim, model_dim
        )
        w2 = _dequant_expert(w2_q[local_id], w2_scale[local_id], model_dim, inter_dim)
        inp = x_all[rows].float()
        gate = (inp @ w1[:inter_dim].T).clamp(max=swiglu_limit)
        up = (inp @ w1[inter_dim:].T).clamp(-swiglu_limit, swiglu_limit)
        hidden = F.silu(gate) * up
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
        del w1, w2, inp, hidden, out
    dist.all_reduce(partial)
    start = sum(token_counts[:rank])
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
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
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
    if moe._active_config.stage2.aligned_pair:
        aligned_output = output.clone()
        full_output = moe(x, weights, ids, slice_output=False)
        expected_shape = (moe.mtpr, moe.model_dim)
        if tuple(full_output.shape) != expected_shape:
            raise AssertionError(
                f"slice_output=False returned {tuple(full_output.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.equal(full_output[:tokens], aligned_output):
            raise AssertionError("slice_output changed aligned-pair valid rows")
        output = full_output[:tokens]
    _barrier()
    eager_output = output.clone()
    rel_l2 = -1.0
    # Every rank must make the same decision before entering the reference
    # collectives; the reference itself pads variable token counts for gather.
    token_max = torch.tensor(tokens, dtype=torch.int32, device=device)
    dist.all_reduce(token_max, op=dist.ReduceOp.MAX)
    if 0 < int(token_max.item()) <= args.accuracy_max_bs:
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
            moe.swiglu_limit,
        )
        error_sq = torch.sum((output.float() - reference) ** 2)
        reference_sq = torch.sum(reference**2)
        dist.all_reduce(error_sq)
        dist.all_reduce(reference_sq)
        rel_l2 = float(torch.sqrt(error_sq / reference_sq))
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
    graph_output = state["output"][:tokens]
    graph_replay_exact = torch.tensor(
        int(torch.equal(graph_output, eager_output)), dtype=torch.int32, device=device
    )
    dist.all_reduce(graph_replay_exact, op=dist.ReduceOp.MIN)
    if not int(graph_replay_exact.item()):
        raise AssertionError(f"bs={tokens} CUDA Graph replay changed the output")
    sbm = int(moe._s1_active_tile_m)
    gemm2_bm = int(moe._g2_active_block_m)
    p2p_quant = moe._active_config.p2p_quant
    if rank == 0:
        print(
            f"[MEGA-V2] bs={tokens} relL2={rel_l2:.6f} "
            f"path={'fixed' if moe._s1_fixed_slot else 'compact'} "
            f"graph_replay=PASS "
            f"p2p_quant={p2p_quant} SBM={sbm} G2_BM={gemm2_bm} "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms "
            f"e2e={e2e_ms[0]:.4f}/{e2e_ms[1]:.4f}ms mean/max",
            flush=True,
        )


def _run_burst(moe, x, weights, ids, depth, rank):
    moe(x, weights, ids)
    _barrier()
    for _ in range(depth):
        moe(x, weights, ids)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[BURST] completed={depth}/{depth}", flush=True)


def _install_config_policy(moe, config_tokens, unify_fields):
    if not config_tokens:
        return
    reference = moe._select_config(config_tokens)
    original_select = moe._select_config
    fields = [field for field in unify_fields.split(",") if field]

    if not fields:

        def select_config(_tokens):
            moe._active_config = reference
            return reference

    else:

        def select_config(tokens):
            local = original_select(tokens)
            stage1_updates = {}
            stage2_updates = {}
            p2p_quant = local.p2p_quant
            for field in fields:
                if field.startswith("stage1."):
                    name = field.removeprefix("stage1.")
                    stage1_updates[name] = getattr(reference.stage1, name)
                elif field.startswith("stage2."):
                    name = field.removeprefix("stage2.")
                    stage2_updates[name] = getattr(reference.stage2, name)
                elif field == "p2p_quant":
                    p2p_quant = reference.p2p_quant
                else:
                    raise ValueError(f"invalid config field {field!r}")
            coupled_stage1 = {"payload_tile_ready"}
            if coupled_stage1 & stage1_updates.keys():
                for name in coupled_stage1:
                    stage1_updates.setdefault(name, getattr(reference.stage1, name))
            config = replace(
                local,
                stage1=replace(local.stage1, **stage1_updates),
                stage2=replace(local.stage2, **stage2_updates),
                p2p_quant=p2p_quant,
            )
            moe._active_config = config
            return config

    moe._select_config = select_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, default="v4_pro")
    parser.add_argument("--bs-list", default="128")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--accuracy-max-bs", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--max-tok-per-rank", type=int)
    parser.add_argument("--rank-tokens", default="")
    parser.add_argument("--config-tokens", type=int, default=0)
    parser.add_argument("--unify-fields", default="")
    parser.add_argument("--burst-depth", type=int, default=0)
    parser.add_argument("--force-fanout-boundary", action="store_true")
    parser.add_argument("--inject-invalid-route", action="store_true")
    parser.add_argument("--force-padding-boundary", action="store_true")
    args = parser.parse_args()
    if (
        sum(
            (
                args.force_fanout_boundary,
                args.inject_invalid_route,
                args.force_padding_boundary,
            )
        )
        > 1
    ):
        raise ValueError("fanout adversarial flags are mutually exclusive")
    batch_sizes = [int(value) for value in args.bs_list.split(",")]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("--bs-list must contain positive integers")
    rank_tokens = [int(value) for value in args.rank_tokens.split(",") if value]

    rank, world, device = _setup_dist()
    try:
        network = NETWORKS[args.network]
        if network["experts"] % world:
            raise ValueError(
                f"experts={network['experts']} must be divisible by world={world}"
            )
        if args.max_tok_per_rank is not None and args.max_tok_per_rank < max(
            batch_sizes
        ):
            raise ValueError("--max-tok-per-rank must cover the largest batch size")
        if args.config_tokens and args.max_tok_per_rank is None:
            raise ValueError("--config-tokens requires --max-tok-per-rank")
        if args.config_tokens < 0 or (
            args.max_tok_per_rank is not None
            and args.config_tokens > args.max_tok_per_rank
        ):
            raise ValueError(
                "--config-tokens must be between zero and max-tok-per-rank"
            )
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
        if rank_tokens and len(rank_tokens) != world:
            raise ValueError("--rank-tokens must contain one value per rank")
        max_bs = max(batch_sizes + rank_tokens)
        x, weights, ids = _make_inputs(
            max_bs,
            network["model_dim"],
            network["experts"],
            network["topk"],
            rank,
            args.seed,
            device,
            force_fanout_boundary=args.force_fanout_boundary,
            inject_invalid_route=args.inject_invalid_route,
            force_padding_boundary=args.force_padding_boundary,
        )
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
        # MegaMoE's producer/consumer wire geometry is rank-invariant. A
        # deliberately skewed test therefore selects one configuration that
        # covers the largest participating rank instead of mixing protocols.
        config_tokens = args.config_tokens
        if rank_tokens and not config_tokens:
            config_tokens = max(1, max(rank_tokens))

        shared_moe = None
        for batch_size in batch_sizes:
            local_batch_size = rank_tokens[rank] if rank_tokens else batch_size
            max_tok_per_rank = args.max_tok_per_rank or max(
                16, _next_power_of_two(batch_size)
            )
            if shared_moe is None or args.max_tok_per_rank is None:
                fanout_masks = ()
                if args.force_fanout_boundary:
                    pair_mask = (1 << (local_experts - 2)) | (1 << (local_experts - 1))
                    fanout_masks = (pair_mask,) * world
                elif args.inject_invalid_route or args.force_padding_boundary:
                    pair_mask = 1 | (1 << (local_experts - 1))
                    fanout_masks = (
                        (pair_mask,) + (0,) * (world - 1)
                        if args.force_padding_boundary
                        else (pair_mask,) * world
                    )
                shared_moe = MegaMoEV2(
                    rank=rank,
                    world_size=world,
                    quant="a8w4",
                    w1=w1,
                    w1_scale=w1_scale,
                    w2=w2,
                    w2_scale=w2_scale,
                    max_tok_per_rank=max_tok_per_rank,
                    fanout_masks=fanout_masks,
                    **network,
                )
            moe = shared_moe
            _install_config_policy(moe, config_tokens, args.unify_fields)
            if rank_tokens:
                selected = moe._select_config(local_batch_size)
                configs = [None] * world
                dist.all_gather_object(
                    configs,
                    (
                        rank,
                        local_batch_size,
                        selected.stage1,
                        selected.stage2,
                        selected.p2p_quant,
                    ),
                )
                if rank == 0:
                    for config in configs:
                        print(f"[CONFIG] {config}", flush=True)
            local_x = x[:local_batch_size].contiguous()
            local_weights = weights[:local_batch_size].contiguous()
            local_ids = ids[:local_batch_size].contiguous()
            if args.burst_depth:
                _run_burst(
                    moe, local_x, local_weights, local_ids, args.burst_depth, rank
                )
            else:
                _run_size(
                    moe,
                    local_x,
                    local_weights,
                    local_ids,
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
