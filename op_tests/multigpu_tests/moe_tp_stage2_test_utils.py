# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared fixtures for the pure-TP comm-fused MoE benchmark."""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.fused_moe import moe_sorting
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


@dataclass
class Stage2Case:
    tokens: int
    hidden: int
    inter_dim: int
    experts: int
    topk: int
    block_m: int
    inter_states: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    a2_scale: torch.Tensor
    a2_scale_unsorted: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor
    num_valid_ids: torch.Tensor
    route_out: torch.Tensor
    partial_out: torch.Tensor


def setup_distributed(expected_world: int):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != expected_world:
        raise ValueError(
            f"torchrun world size is {world}, but --tp-size={expected_world}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    ensure_model_parallel_initialized(world, 1)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, dtype=torch.int32, device=device), group=group)
    torch.cuda.synchronize(device)
    return rank, world, local_rank, device, group


def barrier(group):
    torch.cuda.synchronize()
    dist.barrier(group=group)


def log_stage(rank: int, message: str):
    if rank == 0:
        print(f"[STAGE] {message}", flush=True)


def cleanup_distributed(rank: int):
    """Release graph-registered IPC buffers before process-group teardown."""
    log_stage(rank, "release distributed communicators")
    for get_group in (
        get_tp_group,
        get_pcp_group,
        get_pp_group,
        get_dp_group,
        get_ep_group,
        get_dcp_group,
    ):
        group = get_group()
        communicator = group.device_communicator
        if communicator is not None:
            communicator.destroy()
            group.device_communicator = None
        if group.mq_broadcaster is not None:
            group.mq_broadcaster = None
    # GroupCoordinator.destroy() currently destroys its ProcessGroups before
    # the coordinator's device communicator and hangs after graph-registered
    # custom RS. The communicators are already released above; destroy only
    # the default group here and let process exit release subgroup handles.
    dist.destroy_process_group()
    log_stage(rank, "distributed teardown complete")


def make_routes(
    tokens: int,
    experts: int,
    topk: int,
    device: torch.device,
    route: str = "uniform",
    *,
    global_token_ids: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
):
    """Build deterministic, rank-invariant route metadata."""
    if topk > experts:
        raise ValueError(f"topk={topk} must not exceed experts={experts}")
    if global_token_ids is None:
        token = torch.arange(tokens, dtype=torch.int64, device=device)
    else:
        if (
            global_token_ids.device != device
            or global_token_ids.dtype != torch.int64
            or tuple(global_token_ids.shape) != (tokens,)
        ):
            raise ValueError(
                "global_token_ids must be device int64 with shape "
                f"{(tokens,)}"
            )
        token = global_token_ids
    token = token[:, None]
    slot = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
    if route == "uniform":
        topk_ids = (token * topk + slot) % experts
    elif route == "skew":
        if experts == 1:
            topk_ids = torch.zeros_like(token + slot)
        else:
            topk_ids = 1 + ((token + slot - 1) % (experts - 1))
            hot_slot = (token % 8) != 7
            cold_slot = 1 + ((token + topk - 1) % (experts - 1))
            topk_ids[:, :1] = torch.where(hot_slot, 0, cold_slot)
    else:
        raise ValueError(f"unsupported route pattern: {route}")
    topk_ids = topk_ids.to(torch.int32).contiguous()
    raw_weights = (slot + 1).expand(tokens, -1).to(torch.float32)
    topk_weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()
    if valid_token_mask is not None:
        if (
            valid_token_mask.device != device
            or valid_token_mask.dtype != torch.bool
            or tuple(valid_token_mask.shape) != (tokens,)
        ):
            raise ValueError(
                "valid_token_mask must be device bool with shape "
                f"{(tokens,)}"
            )
        topk_weights.masked_fill_(~valid_token_mask[:, None], 0.0)
    return topk_ids, topk_weights


def make_stage2_case(
    args,
    rank: int,
    device: torch.device,
    *,
    global_token_ids: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    prequant_inter_states: torch.Tensor | None = None,
    prequant_a2_scale: torch.Tensor | None = None,
    seed_offset: int = 0,
    allocate_outputs: bool = True,
    accumulate: bool = True,
) -> Stage2Case:
    """Create rank-specific TP partial inputs with rank-invariant routes."""
    topk_ids, topk_weights = make_routes(
        args.tokens,
        args.experts,
        args.topk,
        device,
        args.route,
        global_token_ids=global_token_ids,
        valid_token_mask=valid_token_mask,
    )
    logical_tokens = int(getattr(args, "logical_tokens", args.tokens))
    if not 0 < logical_tokens <= args.tokens:
        raise ValueError(
            f"logical_tokens must be in [1, {args.tokens}], got {logical_tokens}"
        )
    if valid_token_mask is None and logical_tokens < args.tokens:
        topk_weights[logical_tokens:].zero_()
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorting_out,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        args.experts,
        args.hidden,
        torch.bfloat16,
        args.tile_m,
        accumulate=accumulate,
    )
    sorted_ids = sorted_ids.to(torch.int32).contiguous()
    sorted_weights = sorted_weights.to(torch.float32).contiguous()
    sorted_expert_ids = sorted_expert_ids.to(torch.int32).contiguous()
    num_valid_ids = num_valid_ids.to(torch.int32).contiguous()
    row_capacity = int(sorted_expert_ids.numel()) * args.tile_m
    if sorted_ids.numel() < row_capacity:
        pad_rows = row_capacity - int(sorted_ids.numel())
        sentinel = (args.topk << 24) | args.tokens
        sorted_ids = torch.nn.functional.pad(
            sorted_ids, (0, pad_rows), value=sentinel
        ).contiguous()
        sorted_weights = torch.nn.functional.pad(
            sorted_weights, (0, pad_rows), value=0.0
        ).contiguous()

    generator = torch.Generator(device=device).manual_seed(
        args.seed + rank + seed_offset
    )
    if (prequant_inter_states is None) != (prequant_a2_scale is None):
        raise ValueError(
            "prequant_inter_states and prequant_a2_scale must be provided together"
        )
    scale_shape = (args.tokens, args.topk, args.inter_dim // 32)
    if prequant_inter_states is None:
        a = torch.randn(
            (args.tokens, args.topk, args.inter_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(args.inter_dim**-0.25)
        inter_states, a_scale = per_1x32_f8_scale_f8_quant(
            a,
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
        )
        del a
        inter_states = inter_states.view(
            args.tokens, args.topk, args.inter_dim
        ).contiguous()
        a_scale_unsorted = a_scale.view(scale_shape).contiguous()
    else:
        expected_inter_shape = (args.tokens, args.topk, args.inter_dim)
        if (
            prequant_inter_states.device != device
            or tuple(prequant_inter_states.shape) != expected_inter_shape
            or prequant_inter_states.dtype != dtypes.fp8
        ):
            raise ValueError(
                "prequant_inter_states must be device FP8 with shape "
                f"{expected_inter_shape}"
            )
        if (
            prequant_a2_scale.device != device
            or prequant_a2_scale.numel() != math.prod(scale_shape)
            or prequant_a2_scale.element_size() != 1
        ):
            raise ValueError(
                "prequant_a2_scale must be a device one-byte tensor with "
                f"{math.prod(scale_shape)} elements"
            )
        inter_states = prequant_inter_states.contiguous()
        a_scale_unsorted = prequant_a2_scale.view(scale_shape).contiguous()
    a2_scale = mxfp4_moe_sort_fwd(
        a_scale_unsorted.view(args.tokens * args.topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=args.tokens,
        cols=args.inter_dim,
    ).contiguous()

    if (shared_w2 is None) != (shared_w2_scale is None):
        raise ValueError("shared_w2 and shared_w2_scale must be provided together")
    if shared_w2 is None:
        w2_bf16 = torch.randn(
            (args.experts, args.hidden, args.inter_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(args.inter_dim**-0.25)
        if getattr(args, "low_memory_w2_quant", False):
            # Quantize one expert at a time to avoid a second full-E FP32 copy.
            first_quant, first_scale = per_1x32_f4_quant(
                w2_bf16[0], quant_dtype=dtypes.fp4x2
            )
            w2_quant = torch.empty(
                (args.experts, *first_quant.shape),
                dtype=first_quant.dtype,
                device=device,
            )
            w2_scale = torch.empty(
                (args.experts, *first_scale.shape),
                dtype=first_scale.dtype,
                device=device,
            )
            w2_quant[0].copy_(first_quant)
            w2_scale[0].copy_(first_scale)
            for expert in range(1, args.experts):
                quant, scale = per_1x32_f4_quant(
                    w2_bf16[expert], quant_dtype=dtypes.fp4x2
                )
                w2_quant[expert].copy_(quant)
                w2_scale[expert].copy_(scale)
            w2_scale = w2_scale.view(-1, w2_scale.shape[-1])
        else:
            w2_quant, w2_scale = per_1x32_f4_quant(
                w2_bf16, quant_dtype=dtypes.fp4x2
            )
        del w2_bf16
        w2_quant = w2_quant.view(
            args.experts, args.hidden, args.inter_dim // 2
        )
        w2 = shuffle_weight_a16w4(w2_quant, 16, False).contiguous()
        w2_scale = shuffle_scale_a16w4(
            w2_scale, args.experts, False
        ).contiguous()
        del w2_quant
    else:
        w2 = shared_w2
        w2_scale = shared_w2_scale

    if allocate_outputs:
        route_out = torch.empty(
            (args.tokens, args.topk, args.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        partial_out = (
            sorting_out
            if sorting_out.numel()
            else torch.empty(
                (args.tokens, args.hidden), dtype=torch.bfloat16, device=device
            )
        )
    else:
        route_out = torch.empty(0, dtype=torch.bfloat16, device=device)
        partial_out = torch.empty(0, dtype=torch.bfloat16, device=device)
    del sorting_out
    torch.cuda.empty_cache()
    return Stage2Case(
        tokens=args.tokens,
        hidden=args.hidden,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        block_m=args.tile_m,
        inter_states=inter_states,
        w2=w2,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        a2_scale_unsorted=a_scale_unsorted,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        sorted_weights=sorted_weights,
        num_valid_ids=num_valid_ids,
        route_out=route_out,
        partial_out=partial_out,
    )


def error_stats(actual: torch.Tensor, expected: torch.Tensor, group):
    diff = actual.float() - expected.float()
    max_abs = diff.abs().max()
    rel_l2 = diff.norm() / expected.float().norm().clamp_min(1.0e-12)
    stats = torch.stack((max_abs, rel_l2))
    dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)
    return float(stats[0].item()), float(stats[1].item())


def capture_graph(body, group, warmup_replays: int):
    """Capture one body with the TP communicator's graph-registration context."""
    barrier(group)
    body()
    barrier(group)
    graph = torch.cuda.CUDAGraph()
    with get_tp_group().graph_capture() as capture_context:
        with torch.cuda.graph(graph, stream=capture_context.stream):
            body()
    for _ in range(warmup_replays):
        graph.replay()
    barrier(group)
    return graph


def time_graph(
    graph_or_body,
    iterations: int,
    group,
    world: int,
    marker: str = "",
    eager: bool = False,
):
    """Time graph replays or the identical eager body inside one marker."""
    barrier(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if marker:
        torch.cuda.nvtx.range_push(marker)
    try:
        start.record()
        for _ in range(iterations):
            if eager:
                graph_or_body()
            else:
                graph_or_body.replay()
        end.record()
        end.synchronize()
    finally:
        if marker:
            torch.cuda.nvtx.range_pop()
    local_us = start.elapsed_time(end) * 1000.0 / iterations

    local = torch.tensor([local_us], dtype=torch.float64, device="cuda")
    all_ranks = torch.empty(world, dtype=torch.float64, device="cuda")
    dist.all_gather_into_tensor(all_ranks, local, group=group)
    return [float(value) for value in all_ranks.cpu().tolist()]


def summarize(samples_by_metric):
    result = {}
    for name, rounds in samples_by_metric.items():
        rank_max = [max(values) for values in rounds]
        rank_mean = [statistics.fmean(values) for values in rounds]
        result[name] = {
            "rank_max_us_by_round": rank_max,
            "rank_mean_us_by_round": rank_mean,
            "median_rank_max_us": statistics.median(rank_max),
            "min_rank_max_us": min(rank_max),
            "max_rank_max_us": max(rank_max),
        }
    return result
