# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-TP Stage1 input-side AllGather/quantization decomposition.

The first candidate exploits the row-local per-1x32 quantization contract:

``V1.input.{rccl,custom}``
    BF16 hidden AllGather + route-metadata AllGather + global MXFP8 quant.
``C1.input.quant_rccl``
    Source-local MXFP8 quant + MXFP8/scale/route-metadata RCCL AllGather.

The optional ``--with-gemm1`` path also compares the existing global
sorting/GEMM1 tail with an eight-source prerequisite. It includes a
device-resident descriptor producer and a local generation/source-ready
protocol. The publisher and consumer still execute on one stream: this proves
CUDA Graph freshness and the release/acquire contract, not communication
overlap.

``--peer-payload`` adds the real source-major MXFP8/scales/routes direct-push
transport.  Peer metrics deliberately replay once per host/rank rendezvous
because the first protocol version has no done/backpressure slot reuse yet.
``--peer-no-wait`` co-runs that transport with descriptor+GEMM1 on disjoint
buffers to test hardware co-residency before adding a ready scheduler.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from datetime import datetime, timezone

os.environ.setdefault("AITER_CUSTOM_AR_MAX_SIZE", str(1024**3))

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_gather
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
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.moe_tp_ag_gemm1 import (
    launch_tp_stage1_peer_payload_exchange,
    launch_tp_stage1_descriptor_pack,
    prepare_tp_stage1_all_ready_metadata,
    prepare_tp_stage1_device_metadata,
    tp_stage1_peer_payload_workspace_layout,
)
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_moe_tp_gemm1_all_ready,
    flydsl_moe_tp_register_peer_workspace,
)
from aiter.ops.quant import mxfp4_moe_sort_fwd
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

TARGET_M = 32768
TARGET_H = 7168
TARGET_EXPERTS = 384
TARGET_TOPK = 6
TARGET_TP = 8


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


def cleanup_distributed(rank: int):
    if rank == 0:
        print("[STAGE] release distributed communicators", flush=True)
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
    dist.destroy_process_group()


def barrier(group):
    torch.cuda.synchronize()
    dist.barrier(group=group)


def log_stage(rank: int, message: str):
    if rank == 0:
        print(f"[STAGE] {message}", flush=True)


def make_local_inputs(args, rank: int, device: torch.device):
    local_tokens = args.tokens // args.tp_size
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank * 1009)
    hidden = torch.randn(
        (local_tokens, args.hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) / 4
    token = torch.arange(local_tokens, dtype=torch.int32, device=device)[:, None]
    slot = torch.arange(args.topk, dtype=torch.int32, device=device)[None, :]
    topk_ids = (token * 17 + slot * 29 + rank * 37) % args.experts
    raw_weights = torch.arange(
        1, args.topk + 1, dtype=torch.float32, device=device
    )[None, :].expand(local_tokens, -1)
    topk_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)
    return hidden.contiguous(), topk_ids.contiguous(), topk_weights.contiguous()


def all_gather(tensor: torch.Tensor, *, use_custom: bool):
    return tensor_model_parallel_all_gather(
        tensor, use_custom=use_custom, dim=0
    )


def quantize(hidden: torch.Tensor):
    return per_1x32_mx_quant(hidden, quant_mode="fp8")


def gather_quantized_payload(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    # The gfx950 custom AG dispatcher only accepts FP32/FP16/BF16.  RCCL can
    # move the quantized byte payload directly; use a uint8 view to avoid
    # relying on collective-library FP8 dtype registration.
    gathered_q_bytes = all_gather(hidden_q.view(torch.uint8), use_custom=False)
    gathered_scale = all_gather(hidden_scale, use_custom=False)
    gathered_ids = all_gather(topk_ids, use_custom=False)
    gathered_weights = all_gather(topk_weights, use_custom=False)
    gathered_q = gathered_q_bytes.view(torch.float8_e4m3fn)
    return gathered_q, gathered_scale, gathered_ids, gathered_weights


def make_w1(args, rank: int, device: torch.device):
    """Build deterministic valid MXFP4 W1 without a multi-GiB BF16 staging copy."""
    nibble = (rank % 6) + 1
    packed = nibble | (nibble << 4)
    w1_bytes = torch.full(
        (args.experts, 2 * args.inter_dim, args.hidden // 2),
        packed,
        dtype=torch.uint8,
        device=device,
    )
    w1_q = w1_bytes.view(dtypes.fp4x2)
    w1_scale = torch.full(
        (args.experts * 2 * args.inter_dim, args.hidden // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    w1 = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale = shuffle_scale_a16w4(w1_scale, args.experts, True).contiguous()
    del w1_bytes, w1_q
    return w1, w1_scale


def prepare_stage1_metadata(
    args,
    hidden_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    token_num = topk_ids.shape[0]
    if topk_weights.shape[0] != token_num or hidden_scale.shape[0] != token_num:
        raise ValueError(
            "Stage1 metadata tensors must have the same token dimension: "
            f"scale={hidden_scale.shape[0]}, ids={token_num}, "
            f"weights={topk_weights.shape[0]}"
        )
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
    )
    del sorting_out
    sorted_ids = sorted_ids.to(torch.int32).contiguous()
    sorted_weights = sorted_weights.to(torch.float32).contiguous()
    sorted_expert_ids = sorted_expert_ids.to(torch.int32).contiguous()
    num_valid_ids = num_valid_ids.to(torch.int32).contiguous()
    scale_sorted = mxfp4_moe_sort_fwd(
        hidden_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        cols=args.hidden,
    ).contiguous()
    return (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        scale_sorted,
    )


def split_source_segments(
    args,
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    """Return one contiguous view per source rank."""
    token_num = hidden_q.shape[0]
    if token_num != args.tokens or token_num % args.tp_size:
        raise ValueError("gathered Stage1 payload must split evenly by TP source")
    if any(
        tensor.shape[0] != token_num
        for tensor in (hidden_scale, topk_ids, topk_weights)
    ):
        raise ValueError("all gathered Stage1 tensors must share the token dimension")
    segment_tokens = token_num // args.tp_size
    return tuple(
        (
            hidden_q.narrow(0, source * segment_tokens, segment_tokens),
            hidden_scale.narrow(0, source * segment_tokens, segment_tokens),
            topk_ids.narrow(0, source * segment_tokens, segment_tokens),
            topk_weights.narrow(0, source * segment_tokens, segment_tokens),
        )
        for source in range(args.tp_size)
    )


def prepare_segment_stage1_metadata(args, source_segments):
    return tuple(
        prepare_stage1_metadata(args, hidden_scale, topk_ids, topk_weights)
        for _hidden_q, hidden_scale, topk_ids, topk_weights in source_segments
    )


def run_stage1_compute(
    args,
    hidden_q: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    metadata,
):
    (
        sorted_ids,
        _sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        scale_sorted,
    ) = metadata
    return flydsl_moe_stage1(
        a=hidden_q,
        w1=w1,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=args.topk,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="silu",
        gate_mode="separated",
        w1_scale=w1_scale,
        a1_scale=scale_sorted,
        persist_m=0,
        waves_per_eu=3,
    )


def run_tp_all_ready_stage1_compute(
    args,
    hidden_q: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    metadata,
):
    return flydsl_moe_tp_gemm1_all_ready(
        a=hidden_q,
        w1=w1,
        w1_scale=w1_scale,
        metadata=metadata,
        topk=args.topk,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        waves_per_eu=3,
    )


def source_ready_correctness(metadata, *, prefix: str):
    """Snapshot one completed device-resident source-ready generation."""
    epoch = int(metadata.source_current_epoch.item())
    entry = int(metadata.source_ready_entry.item())
    ready = metadata.source_ready
    payload = metadata.source_payload_epoch
    observed = metadata.source_observed_epoch
    return {
        f"{prefix}_epoch": epoch,
        f"{prefix}_entry_epoch_mismatch": int(epoch <= 0 or entry != epoch),
        f"{prefix}_ready_mismatches": int((ready != epoch).sum().item()),
        f"{prefix}_payload_epoch_mismatches": int(
            (payload != epoch).sum().item()
        ),
        f"{prefix}_observed_epoch_mismatches": int(
            (observed != epoch).sum().item()
        ),
        f"{prefix}_errors": int(metadata.source_ready_errors.item()),
    }


def peer_payload_views(
    workspace: torch.Tensor,
    layout,
    *,
    source_count: int,
    tokens_per_source: int,
    hidden: int,
    topk: int,
):
    """Return typed source-major views of one registered peer workspace."""
    records = source_count * layout.chunks_per_source

    def bytes_view(offset: int, count: int):
        return workspace.narrow(0, offset, count)

    def i64_view(offset: int, count: int):
        return bytes_view(offset, count * 8).view(torch.int64)

    values_count = source_count * tokens_per_source * hidden
    scales_count = source_count * tokens_per_source * (hidden // 32)
    routes_count = source_count * tokens_per_source * topk
    return {
        "values": bytes_view(layout.values_offset, values_count).view(
            source_count * tokens_per_source, hidden
        ),
        "scales": bytes_view(layout.scales_offset, scales_count).view(
            source_count * tokens_per_source, hidden // 32
        ),
        "topk_ids": bytes_view(
            layout.topk_ids_offset, routes_count * 4
        ).view(torch.int32).view(source_count * tokens_per_source, topk),
        "topk_weights": bytes_view(
            layout.topk_weights_offset, routes_count * 4
        ).view(torch.float32).view(source_count * tokens_per_source, topk),
        "entry": i64_view(layout.entry_offset, 1),
        "current_epoch": i64_view(layout.current_epoch_offset, 1),
        "payload_epoch": i64_view(layout.payload_epoch_offset, records),
        "ready": i64_view(layout.ready_offset, records),
        "observed_epoch": i64_view(layout.observed_epoch_offset, records),
        "errors": bytes_view(layout.errors_offset, 4).view(torch.int32),
    }


def peer_payload_correctness(
    views,
    *,
    expected_values: torch.Tensor,
    expected_scales: torch.Tensor,
    expected_ids: torch.Tensor,
    expected_weights: torch.Tensor,
    prefix: str,
):
    """Validate all four payload fields and one completed 64-bit generation."""
    epoch = int(views["current_epoch"].item())
    entry_epoch = int(views["entry"].item())
    return {
        f"{prefix}_epoch": epoch,
        f"{prefix}_entry_epoch": entry_epoch,
        f"{prefix}_entry_epoch_mismatch": int(
            epoch <= 0 or entry_epoch != epoch
        ),
        f"{prefix}_value_byte_mismatches": int(
            (
                views["values"]
                != expected_values.view(torch.uint8)
            ).sum().item()
        ),
        f"{prefix}_scale_byte_mismatches": int(
            (
                views["scales"]
                != expected_scales.view(torch.uint8)
            ).sum().item()
        ),
        f"{prefix}_route_id_mismatches": int(
            (views["topk_ids"] != expected_ids).sum().item()
        ),
        f"{prefix}_route_weight_bit_mismatches": int(
            (
                views["topk_weights"].view(torch.int32)
                != expected_weights.view(torch.int32)
            ).sum().item()
        ),
        f"{prefix}_payload_epoch_mismatches": int(
            (views["payload_epoch"] != epoch).sum().item()
        ),
        f"{prefix}_ready_mismatches": int(
            (views["ready"] != epoch).sum().item()
        ),
        f"{prefix}_observed_epoch_mismatches": int(
            (views["observed_epoch"] != epoch).sum().item()
        ),
        f"{prefix}_errors": int(views["errors"].item()),
    }


def peer_payload_failures(correctness: dict, *, prefix: str):
    return {
        name: value
        for name, value in correctness.items()
        if name.startswith(f"{prefix}_")
        and not name.endswith("_epoch")
        and value != 0
    }


def capture_graph(body, group, warmup_replays: int):
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


def time_graph(graph, iterations: int, group, world: int):
    barrier(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) * 1000.0 / iterations
    local = torch.tensor([local_us], dtype=torch.float64, device="cuda")
    all_ranks = torch.empty(world, dtype=torch.float64, device="cuda")
    dist.all_gather_into_tensor(all_ranks, local, group=group)
    return [float(value) for value in all_ranks.cpu().tolist()]


def summarize(samples_by_metric):
    result = {}
    for name, rounds in samples_by_metric.items():
        rank_max = [max(values) for values in rounds]
        result[name] = {
            "rank_max_us_by_round": rank_max,
            "median_rank_max_us": statistics.median(rank_max),
            "min_rank_max_us": min(rank_max),
            "max_rank_max_us": max(rank_max),
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure-TP Stage1 BF16-vs-quantized AllGather decomposition"
    )
    parser.add_argument("--tp-size", type=int, default=TARGET_TP)
    parser.add_argument("--tokens", type=int, default=TARGET_M)
    parser.add_argument("--hidden", type=int, default=TARGET_H)
    parser.add_argument("--inter-dim", type=int, default=384)
    parser.add_argument("--experts", type=int, default=TARGET_EXPERTS)
    parser.add_argument("--topk", type=int, default=TARGET_TOPK)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=256)
    parser.add_argument(
        "--with-gemm1",
        action="store_true",
        help="append identical global sorting + FlyDSL GEMM1 tails",
    )
    parser.add_argument(
        "--peer-payload",
        action="store_true",
        help="benchmark the real quantized source-major peer payload exchange",
    )
    parser.add_argument(
        "--peer-no-wait",
        action="store_true",
        help=(
            "co-run peer payload exchange with prepared descriptor+GEMM1; "
            "requires --peer-payload --with-gemm1"
        ),
    )
    parser.add_argument("--peer-tokens-per-chunk", type=int, default=512)
    parser.add_argument("--peer-blocks-per-destination", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-replays", type=int, default=2)
    parser.add_argument(
        "--metrics",
        default="",
        help="optional comma-separated metric whitelist for focused scans",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.tp_size <= 1 or args.tokens <= 0 or args.tokens % args.tp_size:
        parser.error("--tokens must be positive and divisible by --tp-size > 1")
    if args.hidden <= 0 or args.hidden % 32:
        parser.error("--hidden must be positive and divisible by 32")
    if args.inter_dim <= 0 or args.hidden % args.tile_k:
        parser.error("--inter-dim must be positive and --tile-k must divide hidden")
    if args.tile_m <= 0 or args.tile_n <= 0 or args.tile_k <= 0:
        parser.error("tile sizes must be positive")
    if args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
        parser.error("--experts/--topk must describe a valid top-k")
    if args.rounds <= 0 or args.iterations <= 0 or args.warmup_replays < 0:
        parser.error("rounds/iterations must be positive and warmup non-negative")
    if (
        args.peer_tokens_per_chunk <= 0
        or args.peer_blocks_per_destination <= 0
    ):
        parser.error("peer chunk and blocks-per-destination must be positive")
    if args.peer_no_wait and not (args.peer_payload and args.with_gemm1):
        parser.error("--peer-no-wait requires --peer-payload --with-gemm1")
    args.metrics = tuple(
        value.strip() for value in args.metrics.split(",") if value.strip()
    )
    if len(set(args.metrics)) != len(args.metrics):
        parser.error("--metrics entries must be unique")
    return args


def main():
    args = parse_args()
    rank, world, local_rank, device, group = setup_distributed(args.tp_size)
    graphs = {}
    holders = {}
    try:
        log_stage(rank, "build deterministic source-local inputs")
        hidden, topk_ids, topk_weights = make_local_inputs(args, rank, device)
        local_q, local_scale = quantize(hidden)

        log_stage(rank, "correctness: local quant + AG equals BF16 AG + global quant")
        global_hidden_rccl = all_gather(hidden, use_custom=False)
        global_hidden_custom = all_gather(hidden, use_custom=True)
        global_q_ref, global_scale_ref = quantize(global_hidden_rccl)
        (
            global_q_candidate,
            global_scale_candidate,
            global_ids_candidate,
            global_weights_candidate,
        ) = gather_quantized_payload(local_q, local_scale, topk_ids, topk_weights)
        global_ids_ref = all_gather(topk_ids, use_custom=False)
        global_weights_ref = all_gather(topk_weights, use_custom=False)
        torch.cuda.synchronize()

        correctness = {
            "bf16_custom_vs_rccl_max_abs": float(
                (global_hidden_custom.float() - global_hidden_rccl.float())
                .abs()
                .max()
                .item()
            ),
            "quant_payload_byte_mismatches": int(
                (global_q_candidate.view(torch.uint8) != global_q_ref.view(torch.uint8))
                .sum()
                .item()
            ),
            "scale_byte_mismatches": int(
                (global_scale_candidate != global_scale_ref).sum().item()
            ),
            "route_id_mismatches": int(
                (global_ids_candidate != global_ids_ref).sum().item()
            ),
            "route_weight_max_abs": float(
                (global_weights_candidate - global_weights_ref).abs().max().item()
            ),
        }
        if any(
            (
                correctness["bf16_custom_vs_rccl_max_abs"] != 0.0,
                correctness["quant_payload_byte_mismatches"] != 0,
                correctness["scale_byte_mismatches"] != 0,
                correctness["route_id_mismatches"] != 0,
                correctness["route_weight_max_abs"] != 0.0,
            )
        ):
            raise AssertionError(f"Stage1 input correctness failed: {correctness}")

        w1 = None
        w1_scale = None
        global_metadata = None
        source_segments = None
        segment_metadata = None
        tp_all_ready_metadata = None
        tp_device_metadata = None
        peer_payload_layout = None
        peer_payload_workspace = None
        peer_payload_rank_data = None
        peer_payload_typed_views = None
        peer_payload_stream = None
        if args.with_gemm1:
            log_stage(
                rank,
                "build deterministic MXFP4 W1 and validate expert-major GEMM1 tails",
            )
            w1, w1_scale = make_w1(args, rank, device)
            global_metadata = prepare_stage1_metadata(
                args, global_scale_ref, global_ids_ref, global_weights_ref
            )
            candidate_metadata = prepare_stage1_metadata(
                args,
                global_scale_candidate,
                global_ids_candidate,
                global_weights_candidate,
            )
            stage1_ref = run_stage1_compute(
                args, global_q_ref, w1, w1_scale, global_metadata
            )
            stage1_candidate = run_stage1_compute(
                args, global_q_candidate, w1, w1_scale, candidate_metadata
            )
            torch.cuda.synchronize()
            correctness["stage1_output_max_abs"] = float(
                (stage1_candidate.float() - stage1_ref.float()).abs().max().item()
            )
            correctness["stage1_output_byte_mismatches"] = int(
                (stage1_candidate.view(torch.uint8) != stage1_ref.view(torch.uint8))
                .sum()
                .item()
            )
            if correctness["stage1_output_byte_mismatches"] != 0:
                raise AssertionError(
                    f"Stage1 common tail mismatch: {correctness}"
                )
            source_segments = split_source_segments(
                args,
                global_q_ref,
                global_scale_ref,
                global_ids_ref,
                global_weights_ref,
            )
            segment_metadata = prepare_segment_stage1_metadata(
                args, source_segments
            )
            tp_all_ready_metadata = prepare_tp_stage1_all_ready_metadata(
                segment_metadata,
                tokens_per_source=args.tokens // args.tp_size,
                experts=args.experts,
                topk=args.topk,
                tile_m=args.tile_m,
            )
            tp_device_metadata = prepare_tp_stage1_device_metadata(
                segment_metadata,
                tokens_per_source=args.tokens // args.tp_size,
                experts=args.experts,
                topk=args.topk,
                tile_m=args.tile_m,
            )
            torch.cuda.synchronize()
            expected_work_blocks = tp_all_ready_metadata.work_descriptors.numel()
            correctness["tp_device_descriptor_row_count_mismatch"] = int(
                tp_device_metadata.num_valid_ids[0].item()
                != expected_work_blocks * args.tile_m
            )
            correctness["tp_device_descriptor_byte_mismatches"] = int(
                (
                    tp_device_metadata.work_descriptors[:expected_work_blocks]
                    != tp_all_ready_metadata.work_descriptors
                )
                .sum()
                .item()
            )
            if any(
                (
                    correctness["tp_device_descriptor_row_count_mismatch"] != 0,
                    correctness["tp_device_descriptor_byte_mismatches"] != 0,
                )
            ):
                raise AssertionError(
                    f"Stage1 device descriptor mismatch: {correctness}"
                )
            launch_tp_stage1_descriptor_pack(
                tp_device_metadata, wait_source_ready=True
            )
            torch.cuda.synchronize()
            correctness["tp_source_ready_descriptor_row_count_mismatch"] = int(
                tp_device_metadata.num_valid_ids[0].item()
                != expected_work_blocks * args.tile_m
            )
            correctness["tp_source_ready_descriptor_byte_mismatches"] = int(
                (
                    tp_device_metadata.work_descriptors[:expected_work_blocks]
                    != tp_all_ready_metadata.work_descriptors
                )
                .sum()
                .item()
            )
            correctness.update(
                source_ready_correctness(
                    tp_device_metadata, prefix="tp_source_ready_eager"
                )
            )
            source_ready_eager_failures = (
                correctness["tp_source_ready_descriptor_row_count_mismatch"],
                correctness["tp_source_ready_descriptor_byte_mismatches"],
                correctness["tp_source_ready_eager_entry_epoch_mismatch"],
                correctness["tp_source_ready_eager_ready_mismatches"],
                correctness["tp_source_ready_eager_payload_epoch_mismatches"],
                correctness["tp_source_ready_eager_observed_epoch_mismatches"],
                correctness["tp_source_ready_eager_errors"],
            )
            if any(source_ready_eager_failures):
                raise AssertionError(
                    f"Stage1 source-ready eager mismatch: {correctness}"
                )
            tp_all_ready_output = run_tp_all_ready_stage1_compute(
                args,
                global_q_ref,
                w1,
                w1_scale,
                tp_all_ready_metadata,
            )
            tp_device_output = run_tp_all_ready_stage1_compute(
                args,
                global_q_ref,
                w1,
                w1_scale,
                tp_device_metadata,
            )
            launch_tp_stage1_descriptor_pack(
                tp_device_metadata, wait_source_ready=True
            )
            tp_source_ready_output = run_tp_all_ready_stage1_compute(
                args,
                global_q_ref,
                w1,
                w1_scale,
                tp_device_metadata,
            )
            torch.cuda.synchronize()
            correctness["tp_all_ready_output_eager_max_abs"] = float(
                (tp_all_ready_output.float() - stage1_ref.float())
                .abs()
                .max()
                .item()
            )
            correctness["tp_all_ready_output_eager_byte_mismatches"] = int(
                (
                    tp_all_ready_output.view(torch.uint8)
                    != stage1_ref.view(torch.uint8)
                )
                .sum()
                .item()
            )
            if correctness["tp_all_ready_output_eager_byte_mismatches"] != 0:
                raise AssertionError(
                    f"Stage1 TP all-ready eager tail mismatch: {correctness}"
                )
            correctness["tp_device_output_eager_max_abs"] = float(
                (tp_device_output.float() - stage1_ref.float()).abs().max().item()
            )
            correctness["tp_device_output_eager_byte_mismatches"] = int(
                (
                    tp_device_output.view(torch.uint8)
                    != stage1_ref.view(torch.uint8)
                )
                .sum()
                .item()
            )
            if correctness["tp_device_output_eager_byte_mismatches"] != 0:
                raise AssertionError(
                    f"Stage1 TP device-descriptor eager tail mismatch: {correctness}"
                )
            correctness["tp_source_ready_output_eager_max_abs"] = float(
                (tp_source_ready_output.float() - stage1_ref.float())
                .abs()
                .max()
                .item()
            )
            correctness["tp_source_ready_output_eager_byte_mismatches"] = int(
                (
                    tp_source_ready_output.view(torch.uint8)
                    != stage1_ref.view(torch.uint8)
                )
                .sum()
                .item()
            )
            if correctness["tp_source_ready_output_eager_byte_mismatches"] != 0:
                raise AssertionError(
                    f"Stage1 TP source-ready eager tail mismatch: {correctness}"
                )

        if args.peer_payload:
            log_stage(rank, "allocate and register real Stage1 peer payload workspace")
            local_tokens = args.tokens // world
            peer_payload_layout = tp_stage1_peer_payload_workspace_layout(
                source_count=world,
                tokens_per_source=local_tokens,
                model_dim=args.hidden,
                topk=args.topk,
                tokens_per_chunk=args.peer_tokens_per_chunk,
            )
            peer_payload_workspace = torch.zeros(
                peer_payload_layout.total_bytes,
                dtype=torch.uint8,
                device=device,
            )
            peer_payload_rank_data = flydsl_moe_tp_register_peer_workspace(
                peer_payload_workspace
            )
            peer_payload_typed_views = peer_payload_views(
                peer_payload_workspace,
                peer_payload_layout,
                source_count=world,
                tokens_per_source=local_tokens,
                hidden=args.hidden,
                topk=args.topk,
            )
            if args.peer_no_wait:
                peer_payload_stream = torch.cuda.Stream(device=device)

            log_stage(rank, "correctness: real Stage1 peer payload eager exchange")
            launch_tp_stage1_peer_payload_exchange(
                workspace=peer_payload_workspace,
                peer_rank_data=peer_payload_rank_data,
                local_values=local_q,
                local_scales=local_scale,
                local_topk_ids=topk_ids,
                local_topk_weights=topk_weights,
                rank=rank,
                source_count=world,
                tokens_per_chunk=args.peer_tokens_per_chunk,
                blocks_per_destination=args.peer_blocks_per_destination,
            )
            barrier(group)
            correctness.update(
                peer_payload_correctness(
                    peer_payload_typed_views,
                    expected_values=global_q_candidate,
                    expected_scales=global_scale_candidate,
                    expected_ids=global_ids_candidate,
                    expected_weights=global_weights_candidate,
                    prefix="peer_payload_eager",
                )
            )
            eager_peer_failures = peer_payload_failures(
                correctness, prefix="peer_payload_eager"
            )
            if eager_peer_failures:
                raise AssertionError(
                    f"Stage1 peer payload eager mismatch: {eager_peer_failures}; "
                    f"entry={correctness['peer_payload_eager_entry_epoch']} "
                    f"current={correctness['peer_payload_eager_epoch']}"
                )

        def run_quant_local():
            holders["quant_local"] = quantize(hidden)

        def run_quant_global():
            holders["quant_global"] = quantize(global_hidden_rccl)

        def run_bf16_ag(use_custom: bool, key: str):
            holders[key] = all_gather(hidden, use_custom=use_custom)

        def run_quant_payload_ag():
            holders["quant_payload_ag"] = gather_quantized_payload(
                local_q, local_scale, topk_ids, topk_weights
            )

        def run_peer_payload_ag():
            launch_tp_stage1_peer_payload_exchange(
                workspace=peer_payload_workspace,
                peer_rank_data=peer_payload_rank_data,
                local_values=local_q,
                local_scales=local_scale,
                local_topk_ids=topk_ids,
                local_topk_weights=topk_weights,
                rank=rank,
                source_count=world,
                tokens_per_chunk=args.peer_tokens_per_chunk,
                blocks_per_destination=args.peer_blocks_per_destination,
            )

        def run_v1_input(use_custom: bool, key: str):
            gathered_hidden = all_gather(hidden, use_custom=use_custom)
            gathered_ids = all_gather(topk_ids, use_custom=use_custom)
            gathered_weights = all_gather(topk_weights, use_custom=use_custom)
            gathered_q, gathered_scale = quantize(gathered_hidden)
            holders[key] = (
                gathered_q,
                gathered_scale,
                gathered_ids,
                gathered_weights,
            )

        def run_c1_input_quant_rccl():
            source_q, source_scale = quantize(hidden)
            holders["c1_input_quant_rccl"] = gather_quantized_payload(
                source_q, source_scale, topk_ids, topk_weights
            )

        def run_route_prep():
            holders["route_prep"] = prepare_stage1_metadata(
                args, global_scale_ref, global_ids_ref, global_weights_ref
            )

        def run_gemm1_compute():
            holders["gemm1_compute"] = run_stage1_compute(
                args, global_q_ref, w1, w1_scale, global_metadata
            )

        def run_route_gemm1():
            metadata = prepare_stage1_metadata(
                args, global_scale_ref, global_ids_ref, global_weights_ref
            )
            holders["route_gemm1"] = run_stage1_compute(
                args, global_q_ref, w1, w1_scale, metadata
            )

        def run_tp_all_ready_gemm1_compute():
            holders["tp_all_ready_gemm1_compute"] = (
                run_tp_all_ready_stage1_compute(
                    args,
                    global_q_ref,
                    w1,
                    w1_scale,
                    tp_all_ready_metadata,
                )
            )

        def run_tp_device_descriptor_prep():
            launch_tp_stage1_descriptor_pack(tp_device_metadata)

        def run_tp_device_descriptor_gemm1():
            launch_tp_stage1_descriptor_pack(tp_device_metadata)
            holders["tp_device_descriptor_gemm1"] = (
                run_tp_all_ready_stage1_compute(
                    args,
                    global_q_ref,
                    w1,
                    w1_scale,
                    tp_device_metadata,
                )
            )

        def run_peer_payload_nowait():
            """Co-run independent peer transport and descriptor+GEMM1."""
            current_stream = torch.cuda.current_stream(device)
            peer_payload_stream.wait_stream(current_stream)
            launch_tp_stage1_peer_payload_exchange(
                workspace=peer_payload_workspace,
                peer_rank_data=peer_payload_rank_data,
                local_values=local_q,
                local_scales=local_scale,
                local_topk_ids=topk_ids,
                local_topk_weights=topk_weights,
                rank=rank,
                source_count=world,
                tokens_per_chunk=args.peer_tokens_per_chunk,
                blocks_per_destination=args.peer_blocks_per_destination,
                stream=peer_payload_stream,
            )
            launch_tp_stage1_descriptor_pack(tp_device_metadata)
            holders["peer_payload_nowait"] = run_tp_all_ready_stage1_compute(
                args,
                global_q_ref,
                w1,
                w1_scale,
                tp_device_metadata,
            )
            current_stream.wait_stream(peer_payload_stream)

        def run_tp_source_ready_descriptor_prep():
            launch_tp_stage1_descriptor_pack(
                tp_device_metadata, wait_source_ready=True
            )

        def run_tp_source_ready_descriptor_gemm1():
            launch_tp_stage1_descriptor_pack(
                tp_device_metadata, wait_source_ready=True
            )
            holders["tp_source_ready_descriptor_gemm1"] = (
                run_tp_all_ready_stage1_compute(
                    args,
                    global_q_ref,
                    w1,
                    w1_scale,
                    tp_device_metadata,
                )
            )

        def run_v1_full(use_custom: bool, key: str):
            gathered_hidden = all_gather(hidden, use_custom=use_custom)
            gathered_ids = all_gather(topk_ids, use_custom=use_custom)
            gathered_weights = all_gather(topk_weights, use_custom=use_custom)
            gathered_q, gathered_scale = quantize(gathered_hidden)
            metadata = prepare_stage1_metadata(
                args, gathered_scale, gathered_ids, gathered_weights
            )
            holders[key] = run_stage1_compute(
                args, gathered_q, w1, w1_scale, metadata
            )

        def run_c1_full():
            source_q, source_scale = quantize(hidden)
            gathered_q, gathered_scale, gathered_ids, gathered_weights = (
                gather_quantized_payload(
                    source_q, source_scale, topk_ids, topk_weights
                )
            )
            metadata = prepare_stage1_metadata(
                args, gathered_scale, gathered_ids, gathered_weights
            )
            holders["c1_quant_rccl"] = run_stage1_compute(
                args, gathered_q, w1, w1_scale, metadata
            )

        bodies = {
            "quant_local": run_quant_local,
            "quant_global": run_quant_global,
            "ag_bf16_rccl": lambda: run_bf16_ag(False, "ag_bf16_rccl"),
            "ag_bf16_custom": lambda: run_bf16_ag(True, "ag_bf16_custom"),
            "ag_quant_payload_rccl": run_quant_payload_ag,
            "v1_input_rccl": lambda: run_v1_input(False, "v1_input_rccl"),
            "v1_input_custom": lambda: run_v1_input(True, "v1_input_custom"),
            "c1_input_quant_rccl": run_c1_input_quant_rccl,
        }
        if args.peer_payload:
            bodies["ag_quant_payload_peer"] = run_peer_payload_ag
        if args.with_gemm1:
            bodies.update(
                route_prep=run_route_prep,
                gemm1_compute=run_gemm1_compute,
                route_gemm1=run_route_gemm1,
                tp_all_ready_gemm1_compute=run_tp_all_ready_gemm1_compute,
                tp_device_descriptor_prep=run_tp_device_descriptor_prep,
                tp_device_descriptor_gemm1=run_tp_device_descriptor_gemm1,
                tp_source_ready_descriptor_prep=(
                    run_tp_source_ready_descriptor_prep
                ),
                tp_source_ready_descriptor_gemm1=(
                    run_tp_source_ready_descriptor_gemm1
                ),
                v1_0_custom=lambda: run_v1_full(True, "v1_0_custom"),
                v1_0_rccl=lambda: run_v1_full(False, "v1_0_rccl"),
                c1_quant_rccl=run_c1_full,
            )
            if args.peer_no_wait:
                bodies["peer_payload_nowait"] = run_peer_payload_nowait
        if args.metrics:
            missing_metrics = sorted(set(args.metrics) - set(bodies))
            if missing_metrics:
                raise ValueError(
                    f"unknown --metrics entries {missing_metrics}; "
                    f"available metrics are {sorted(bodies)}"
                )
            bodies = {name: bodies[name] for name in args.metrics}
        peer_metrics = {"ag_quant_payload_peer", "peer_payload_nowait"}
        for name, body in bodies.items():
            log_stage(rank, f"capture {name}")
            warmup_replays = 0 if name in peer_metrics else args.warmup_replays
            graphs[name] = capture_graph(body, group, warmup_replays)

        if args.peer_payload and peer_metrics.intersection(graphs):
            log_stage(rank, "validate real peer payload after graph capture")
            if "peer_payload_nowait" in graphs:
                # Graph capture records the GEMM output allocation but does not
                # execute it.  Peer graphs intentionally have no warmup replay
                # because this protocol has no slot backpressure yet, so issue
                # exactly one rank-synchronized replay before reading either
                # the output or its generation signals.
                barrier(group)
                graphs["peer_payload_nowait"].replay()
                torch.cuda.synchronize()
            barrier(group)
            correctness.update(
                peer_payload_correctness(
                    peer_payload_typed_views,
                    expected_values=global_q_candidate,
                    expected_scales=global_scale_candidate,
                    expected_ids=global_ids_candidate,
                    expected_weights=global_weights_candidate,
                    prefix="peer_payload_graph",
                )
            )
            graph_peer_failures = peer_payload_failures(
                correctness, prefix="peer_payload_graph"
            )
            if graph_peer_failures:
                raise AssertionError(
                    f"Stage1 peer payload graph mismatch: {graph_peer_failures}"
                )
            if "peer_payload_nowait" in graphs:
                nowait_output = holders["peer_payload_nowait"]
                correctness["peer_payload_nowait_output_byte_mismatches"] = int(
                    (
                        nowait_output.view(torch.uint8)
                        != stage1_ref.view(torch.uint8)
                    ).sum().item()
                )
                correctness["peer_payload_nowait_output_max_abs"] = float(
                    (nowait_output.float() - stage1_ref.float()).abs().max().item()
                )
                if correctness["peer_payload_nowait_output_byte_mismatches"]:
                    raise AssertionError(
                        "Stage1 peer no-wait GEMM1 graph output mismatch: "
                        f"{correctness}"
                    )

        if args.with_gemm1 and not args.metrics:
            log_stage(rank, "validate TP all-ready GEMM1 graph replay")
            graphs["tp_all_ready_gemm1_compute"].replay()
            graphs["tp_device_descriptor_gemm1"].replay()
            for _ in range(3):
                graphs["tp_source_ready_descriptor_gemm1"].replay()
            torch.cuda.synchronize()
            tp_graph_output = holders["tp_all_ready_gemm1_compute"]
            correctness["tp_all_ready_output_graph_max_abs"] = float(
                (tp_graph_output.float() - stage1_ref.float()).abs().max().item()
            )
            correctness["tp_all_ready_output_graph_byte_mismatches"] = int(
                (
                    tp_graph_output.view(torch.uint8)
                    != stage1_ref.view(torch.uint8)
                )
                .sum()
                .item()
            )
            if correctness["tp_all_ready_output_graph_byte_mismatches"] != 0:
                raise AssertionError(
                    f"Stage1 TP all-ready graph tail mismatch: {correctness}"
                )
            tp_device_graph_output = holders["tp_device_descriptor_gemm1"]
            correctness["tp_device_output_graph_max_abs"] = float(
                (tp_device_graph_output.float() - stage1_ref.float())
                .abs()
                .max()
                .item()
            )
            correctness["tp_device_output_graph_byte_mismatches"] = int(
                (
                    tp_device_graph_output.view(torch.uint8)
                    != stage1_ref.view(torch.uint8)
                )
                .sum()
                .item()
            )
            if correctness["tp_device_output_graph_byte_mismatches"] != 0:
                raise AssertionError(
                    f"Stage1 TP device-descriptor graph tail mismatch: {correctness}"
                )
            tp_source_ready_graph_output = holders[
                "tp_source_ready_descriptor_gemm1"
            ]
            correctness["tp_source_ready_output_graph_max_abs"] = float(
                (tp_source_ready_graph_output.float() - stage1_ref.float())
                .abs()
                .max()
                .item()
            )
            correctness["tp_source_ready_output_graph_byte_mismatches"] = int(
                (
                    tp_source_ready_graph_output.view(torch.uint8)
                    != stage1_ref.view(torch.uint8)
                )
                .sum()
                .item()
            )
            correctness.update(
                source_ready_correctness(
                    tp_device_metadata, prefix="tp_source_ready_graph"
                )
            )
            correctness["tp_source_ready_descriptor_graph_row_count_mismatch"] = (
                int(
                    tp_device_metadata.num_valid_ids[0].item()
                    != expected_work_blocks * args.tile_m
                )
            )
            correctness["tp_source_ready_descriptor_graph_byte_mismatches"] = int(
                (
                    tp_device_metadata.work_descriptors[:expected_work_blocks]
                    != tp_all_ready_metadata.work_descriptors
                )
                .sum()
                .item()
            )
            source_ready_graph_failures = (
                correctness["tp_source_ready_output_graph_byte_mismatches"],
                correctness[
                    "tp_source_ready_descriptor_graph_row_count_mismatch"
                ],
                correctness[
                    "tp_source_ready_descriptor_graph_byte_mismatches"
                ],
                correctness["tp_source_ready_graph_entry_epoch_mismatch"],
                correctness["tp_source_ready_graph_ready_mismatches"],
                correctness["tp_source_ready_graph_payload_epoch_mismatches"],
                correctness["tp_source_ready_graph_observed_epoch_mismatches"],
                correctness["tp_source_ready_graph_errors"],
            )
            if any(source_ready_graph_failures):
                raise AssertionError(
                    f"Stage1 TP source-ready graph mismatch: {correctness}"
                )
            barrier(group)

        samples_by_metric = {name: [] for name in graphs}
        metric_order = list(graphs)
        for round_index in range(args.rounds):
            log_stage(rank, f"timing round {round_index + 1}/{args.rounds}")
            order = (
                metric_order
                if round_index % 2 == 0
                else list(reversed(metric_order))
            )
            for name in order:
                # The first peer ABI has generation freshness but no consumer
                # done/backpressure.  One replay followed by the rank-wide
                # timing gather prevents a fast rank from overwriting an epoch
                # that a slow rank still expects.
                metric_iterations = 1 if name in peer_metrics else args.iterations
                samples_by_metric[name].append(
                    time_graph(graphs[name], metric_iterations, group, world)
                )

        summary = summarize(samples_by_metric)
        peer_nowait_analysis = None
        peer_nowait_metrics = {
            "ag_quant_payload_peer",
            "tp_device_descriptor_gemm1",
            "peer_payload_nowait",
        }
        if peer_nowait_metrics.issubset(summary):
            transfer_us = summary["ag_quant_payload_peer"]["median_rank_max_us"]
            compute_us = summary["tp_device_descriptor_gemm1"][
                "median_rank_max_us"
            ]
            union_us = summary["peer_payload_nowait"]["median_rank_max_us"]
            overlap_us = compute_us + transfer_us - union_us
            peer_nowait_analysis = {
                "transfer_us": transfer_us,
                "descriptor_gemm1_us": compute_us,
                "union_us": union_us,
                "serial_sum_us": compute_us + transfer_us,
                "ideal_overlap_floor_us": max(compute_us, transfer_us),
                "semantic_overlap_us": overlap_us,
                "recovered_transfer_ratio": overlap_us / transfer_us,
                "union_over_ideal_us": union_us - max(compute_us, transfer_us),
            }
        local_tokens = args.tokens // world
        bf16_bytes = local_tokens * args.hidden * 2
        quant_bytes = (
            local_tokens * args.hidden
            + local_tokens * (args.hidden // 32)
            + local_tokens * args.topk * 4
            + local_tokens * args.topk * 4
        )
        record = {
            "schema": "flydsl_moe_tp_gemm1_ag_input_v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "local_rank": local_rank,
            "config": {
                "device": torch.cuda.get_device_name(device),
                "gfx": torch.cuda.get_device_properties(device).gcnArchName,
                "tp": world,
                "tokens": args.tokens,
                "tokens_per_rank": local_tokens,
                "hidden": args.hidden,
                "inter_dim_rank": args.inter_dim,
                "experts": args.experts,
                "topk": args.topk,
                "with_gemm1": args.with_gemm1,
                "metrics": list(args.metrics),
                "iterations": args.iterations,
                "peer_metric_iterations": 1 if args.peer_payload else 0,
                "peer_payload": args.peer_payload,
                "peer_no_wait": args.peer_no_wait,
                "peer_tokens_per_chunk": (
                    args.peer_tokens_per_chunk if args.peer_payload else 0
                ),
                "peer_chunks_per_source": (
                    peer_payload_layout.chunks_per_source
                    if args.peer_payload
                    else 0
                ),
                "peer_blocks_per_destination": (
                    args.peer_blocks_per_destination if args.peer_payload else 0
                ),
                "peer_workspace_bytes": (
                    peer_payload_layout.total_bytes if args.peer_payload else 0
                ),
                "peer_logical_write_bytes_per_rank": (
                    quant_bytes * world if args.peer_payload else 0
                ),
                "peer_remote_write_bytes_per_rank": (
                    quant_bytes * (world - 1) if args.peer_payload else 0
                ),
                "stage1_source_segments": world if args.with_gemm1 else 0,
                "stage1_tokens_per_segment": local_tokens if args.with_gemm1 else 0,
                "tp_all_ready_work_blocks": (
                    tp_all_ready_metadata.work_descriptors.numel()
                    if args.with_gemm1
                    else 0
                ),
                "tp_device_descriptor_capacity_blocks": (
                    tp_device_metadata.work_descriptors.numel()
                    if args.with_gemm1
                    else 0
                ),
                "stage1_segment_output_layout": (
                    "[source][tokens_per_rank,topk,inter_dim]"
                    if args.with_gemm1
                    else None
                ),
                "tile": [args.tile_m, args.tile_n, args.tile_k],
                "bf16_source_bytes": bf16_bytes,
                "quantized_source_bytes_including_routes": quant_bytes,
                "payload_ratio": quant_bytes / bf16_bytes,
            },
            "correctness": correctness,
            "local_us_by_round": {
                name: [round_values[rank] for round_values in rounds]
                for name, rounds in samples_by_metric.items()
            },
        }
        for owner in range(world):
            if rank == owner:
                print("[RANK_JSON] " + json.dumps(record, sort_keys=True), flush=True)
            dist.barrier(group=group)

        if rank == 0:
            report = {
                "schema": "flydsl_moe_tp_gemm1_ag_input_summary_v1",
                "config": record["config"],
                "correctness": correctness,
                "summary": summary,
            }
            if peer_nowait_analysis is not None:
                report["peer_nowait_analysis"] = peer_nowait_analysis
            print("[SUMMARY_JSON] " + json.dumps(report, sort_keys=True), flush=True)
            if args.metrics:
                values = " ".join(
                    f"{name}={data['median_rank_max_us']:.3f}us"
                    for name, data in summary.items()
                )
                print(f"[RESULT_PROFILE] {values}", flush=True)
                if peer_nowait_analysis is not None:
                    print(
                        "[RESULT_PEER_OVERLAP] "
                        f"overlap={peer_nowait_analysis['semantic_overlap_us']:.3f}us "
                        "recovered_transfer="
                        f"{peer_nowait_analysis['recovered_transfer_ratio']:.4f} "
                        "union_over_ideal="
                        f"{peer_nowait_analysis['union_over_ideal_us']:.3f}us",
                        flush=True,
                    )
            else:
                print(
                    "[RESULT] "
                    f"payload_ratio={quant_bytes / bf16_bytes:.4f} "
                    f"quant.local={summary['quant_local']['median_rank_max_us']:.3f}us "
                    f"quant.global={summary['quant_global']['median_rank_max_us']:.3f}us "
                    f"ag.bf16.rccl={summary['ag_bf16_rccl']['median_rank_max_us']:.3f}us "
                    f"ag.bf16.custom={summary['ag_bf16_custom']['median_rank_max_us']:.3f}us "
                    "ag.quant_payload.rccl="
                    f"{summary['ag_quant_payload_rccl']['median_rank_max_us']:.3f}us "
                    f"v1.input.rccl={summary['v1_input_rccl']['median_rank_max_us']:.3f}us "
                    f"v1.input.custom={summary['v1_input_custom']['median_rank_max_us']:.3f}us "
                    f"c1.input.quant_rccl={summary['c1_input_quant_rccl']['median_rank_max_us']:.3f}us",
                    flush=True,
                )
                if args.peer_payload:
                    print(
                        "[RESULT_PEER] ag.quant_payload.peer="
                        f"{summary['ag_quant_payload_peer']['median_rank_max_us']:.3f}us",
                        flush=True,
                    )
                if args.with_gemm1:
                    print(
                        "[RESULT_GEMM1] "
                        f"route={summary['route_prep']['median_rank_max_us']:.3f}us "
                        f"gemm1={summary['gemm1_compute']['median_rank_max_us']:.3f}us "
                        f"route_gemm1={summary['route_gemm1']['median_rank_max_us']:.3f}us "
                        "tp_all_ready_gemm1="
                        f"{summary['tp_all_ready_gemm1_compute']['median_rank_max_us']:.3f}us "
                        "tp_device_desc="
                        f"{summary['tp_device_descriptor_prep']['median_rank_max_us']:.3f}us "
                        "tp_device_desc_gemm1="
                        f"{summary['tp_device_descriptor_gemm1']['median_rank_max_us']:.3f}us "
                        "tp_source_ready_desc="
                        f"{summary['tp_source_ready_descriptor_prep']['median_rank_max_us']:.3f}us "
                        "tp_source_ready_desc_gemm1="
                        f"{summary['tp_source_ready_descriptor_gemm1']['median_rank_max_us']:.3f}us "
                        f"v1.0.custom={summary['v1_0_custom']['median_rank_max_us']:.3f}us "
                        f"v1.0.rccl={summary['v1_0_rccl']['median_rank_max_us']:.3f}us "
                        f"c1.quant_rccl={summary['c1_quant_rccl']['median_rank_max_us']:.3f}us",
                        flush=True,
                    )
                    if args.peer_no_wait:
                        print(
                            "[RESULT_PEER_NOWAIT] "
                            f"peer_payload_nowait={summary['peer_payload_nowait']['median_rank_max_us']:.3f}us",
                            flush=True,
                        )
        barrier(group)
        graphs.clear()
        holders.clear()
        bodies.clear()
        del hidden, topk_ids, topk_weights, local_q, local_scale
        del global_hidden_rccl, global_hidden_custom, global_q_ref, global_scale_ref
        del global_q_candidate, global_scale_candidate
        del global_ids_candidate, global_weights_candidate
        del global_ids_ref, global_weights_ref
        if args.with_gemm1:
            del w1, w1_scale, global_metadata, candidate_metadata
            del stage1_ref, stage1_candidate
            del source_segments, segment_metadata
            del tp_all_ready_metadata, tp_all_ready_output
            del tp_device_metadata, tp_device_output
            del tp_source_ready_output
            if not args.metrics:
                del tp_graph_output, tp_device_graph_output
                del tp_source_ready_graph_output
        if args.peer_payload:
            del peer_payload_typed_views, peer_payload_workspace
            del peer_payload_layout, peer_payload_rank_data
            if args.peer_no_wait:
                if not args.metrics or "peer_payload_nowait" in args.metrics:
                    del nowait_output
                del peer_payload_stream
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier(group=group)
    finally:
        cleanup_distributed(rank)


if __name__ == "__main__":
    main()
