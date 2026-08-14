# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP2 correctness/freshness gate for the real quantized Stage1 AG payload.

Run with::

    torchrun --standalone --nproc_per_node=2 \
      op_tests/multigpu_tests/test_flydsl_moe_tp_gemm1_payload.py

The registered workspace is a nonzero-offset view.  Each source publishes
MXFP8 values, E8M0 scales, top-k ids, and top-k weights in four token chunks;
the last chunk is deliberately short.  Eager and alternating graph replays
must observe one matching full 64-bit generation for every source/chunk.
"""

from __future__ import annotations

import argparse
import gc
import json

import torch

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.moe_tp_ag_gemm1 import (
    launch_tp_stage1_peer_payload_exchange,
    tp_stage1_peer_payload_workspace_layout,
)
from aiter.ops.flydsl.moe_kernels import flydsl_moe_tp_register_peer_workspace
from op_tests.multigpu_tests.bench_flydsl_moe_tp_gemm1_ag import (
    barrier,
    capture_graph,
    cleanup_distributed,
    setup_distributed,
)


def make_payload(
    source: int,
    variant: int,
    *,
    tokens: int,
    hidden: int,
    experts: int,
    topk: int,
    device: torch.device,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(91_003 + source * 1009 + variant * 17_011)
    hidden_bf16 = torch.randn(
        (tokens, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    values, scales = per_1x32_mx_quant(hidden_bf16, quant_mode="fp8")
    token = torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
    slot = torch.arange(topk, dtype=torch.int32, device=device)[None, :]
    topk_ids = (
        token * 17 + slot * 29 + source * 37 + variant * 43
    ) % experts
    raw_weights = (
        slot.to(torch.float32)
        + 1.0
        + float(source) * 0.125
        + float(variant) * 0.0625
    ).expand(tokens, -1)
    topk_weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()
    return (
        values.contiguous(),
        scales.contiguous(),
        topk_ids.contiguous(),
        topk_weights,
    )


def workspace_views(workspace, layout, source_count, tokens, hidden, topk):
    records = source_count * layout.chunks_per_source

    def bytes_view(offset: int, count: int):
        return workspace[offset : offset + count]

    def i64_view(offset: int, count: int):
        return bytes_view(offset, count * 8).view(torch.int64)

    values_count = source_count * tokens * hidden
    scales_count = source_count * tokens * (hidden // 32)
    routes_count = source_count * tokens * topk
    return {
        "values": bytes_view(layout.values_offset, values_count).view(
            source_count, tokens, hidden
        ),
        "scales": bytes_view(layout.scales_offset, scales_count).view(
            source_count, tokens, hidden // 32
        ),
        "topk_ids": bytes_view(
            layout.topk_ids_offset, routes_count * 4
        ).view(torch.int32).view(source_count, tokens, topk),
        "topk_weights": bytes_view(
            layout.topk_weights_offset, routes_count * 4
        ).view(torch.float32).view(source_count, tokens, topk),
        "entry": i64_view(layout.entry_offset, 1),
        "current_epoch": i64_view(layout.current_epoch_offset, 1),
        "payload_epoch": i64_view(
            layout.payload_epoch_offset, records
        ).view(source_count, layout.chunks_per_source),
        "ready": i64_view(layout.ready_offset, records).view(
            source_count, layout.chunks_per_source
        ),
        "observed_epoch": i64_view(
            layout.observed_epoch_offset, records
        ).view(source_count, layout.chunks_per_source),
        "skew_scratch": i64_view(layout.skew_scratch_offset, 1),
        "errors": bytes_view(layout.errors_offset, 4).view(torch.int32),
    }


def validate_payload(
    *,
    rank: int,
    views,
    expected,
    previous_epoch: int,
    skew_iterations: int,
):
    epoch = int(views["current_epoch"].item())
    expected_values, expected_scales, expected_ids, expected_weights = expected
    failures = {
        "epoch_step": int(epoch != previous_epoch + 1),
        "entry_epoch": int(int(views["entry"].item()) != epoch),
        "values": int((views["values"] != expected_values).sum().item()),
        "scales": int((views["scales"] != expected_scales).sum().item()),
        "topk_ids": int((views["topk_ids"] != expected_ids).sum().item()),
        "topk_weights": int(
            (views["topk_weights"].view(torch.int32)
             != expected_weights.view(torch.int32)).sum().item()
        ),
        "payload_epoch": int((views["payload_epoch"] != epoch).sum().item()),
        "ready": int((views["ready"] != epoch).sum().item()),
        "observed_epoch": int((views["observed_epoch"] != epoch).sum().item()),
        "errors": int(views["errors"].item()),
    }
    expected_skew = epoch * skew_iterations if rank == 1 else 0
    failures["skew_scratch"] = int(
        int(views["skew_scratch"].item()) != expected_skew
    )
    if any(failures.values()):
        raise AssertionError(
            f"rank {rank} real Stage1 payload epoch {epoch} failed: {failures}"
        )
    return epoch, failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="TP2 real quantized Stage1 peer payload freshness gate"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tokens-per-source", type=int, default=65)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens-per-chunk", type=int, default=17)
    parser.add_argument("--freshness-replays", type=int, default=16)
    parser.add_argument("--skew-iterations", type=int, default=20_000)
    parser.add_argument("--workspace-view-offset", type=int, default=4096)
    parser.add_argument("--blocks-per-destination", type=int, default=4)
    args = parser.parse_args()
    if args.tp_size != 2:
        parser.error("this freshness gate currently requires --tp-size=2")
    if args.tokens_per_source <= 0 or args.hidden <= 0 or args.hidden % 32:
        parser.error("tokens must be positive and hidden divisible by 32")
    if args.topk <= 0 or args.topk > args.experts:
        parser.error("topk must be in [1, experts]")
    if args.tokens_per_chunk <= 0:
        parser.error("tokens-per-chunk must be positive")
    if args.tokens_per_source % args.tokens_per_chunk == 0:
        parser.error("this gate requires a nonempty short tail chunk")
    if args.freshness_replays < 2 or args.skew_iterations <= 0:
        parser.error("freshness replays >=2 and positive skew are required")
    if args.workspace_view_offset <= 0 or args.workspace_view_offset % 16:
        parser.error("workspace offset must be positive and 16-byte aligned")
    if args.blocks_per_destination <= 0:
        parser.error("blocks-per-destination must be positive")
    return args


def main():
    args = parse_args()
    rank, world, _, device, group = setup_distributed(args.tp_size)
    graph = None
    try:
        layout = tp_stage1_peer_payload_workspace_layout(
            source_count=world,
            tokens_per_source=args.tokens_per_source,
            model_dim=args.hidden,
            topk=args.topk,
            tokens_per_chunk=args.tokens_per_chunk,
        )
        backing = torch.zeros(
            layout.total_bytes + args.workspace_view_offset,
            dtype=torch.uint8,
            device=device,
        )
        workspace = backing.narrow(
            0, args.workspace_view_offset, layout.total_bytes
        )
        if workspace.data_ptr() - backing.data_ptr() != args.workspace_view_offset:
            raise AssertionError("registered workspace view offset mismatch")
        peer_rank_data = flydsl_moe_tp_register_peer_workspace(workspace)

        variants = tuple(
            tuple(
                make_payload(
                    source,
                    variant,
                    tokens=args.tokens_per_source,
                    hidden=args.hidden,
                    experts=args.experts,
                    topk=args.topk,
                    device=device,
                )
                for source in range(world)
            )
            for variant in range(2)
        )
        expected = tuple(
            (
                torch.stack(
                    tuple(payload[0].view(torch.uint8) for payload in per_source)
                ),
                torch.stack(
                    tuple(payload[1].view(torch.uint8) for payload in per_source)
                ),
                torch.stack(tuple(payload[2] for payload in per_source)),
                torch.stack(tuple(payload[3] for payload in per_source)),
            )
            for per_source in variants
        )
        local_values = torch.empty_like(variants[0][rank][0])
        local_scales = torch.empty_like(variants[0][rank][1])
        local_topk_ids = torch.empty_like(variants[0][rank][2])
        local_topk_weights = torch.empty_like(variants[0][rank][3])
        views = workspace_views(
            workspace,
            layout,
            world,
            args.tokens_per_source,
            args.hidden,
            args.topk,
        )

        def set_variant(variant_index: int):
            payload = variants[variant_index][rank]
            local_values.copy_(payload[0])
            local_scales.copy_(payload[1])
            local_topk_ids.copy_(payload[2])
            local_topk_weights.copy_(payload[3])

        def exchange_body():
            launch_tp_stage1_peer_payload_exchange(
                workspace=workspace,
                peer_rank_data=peer_rank_data,
                local_values=local_values,
                local_scales=local_scales,
                local_topk_ids=local_topk_ids,
                local_topk_weights=local_topk_weights,
                rank=rank,
                source_count=world,
                tokens_per_chunk=args.tokens_per_chunk,
                skew_iterations=args.skew_iterations,
                blocks_per_destination=args.blocks_per_destination,
            )

        set_variant(0)
        exchange_body()
        barrier(group)
        previous_epoch, eager_failures = validate_payload(
            rank=rank,
            views=views,
            expected=expected[0],
            previous_epoch=0,
            skew_iterations=args.skew_iterations,
        )

        set_variant(1)
        graph = capture_graph(exchange_body, group, warmup_replays=0)
        barrier(group)
        previous_epoch, capture_failures = validate_payload(
            rank=rank,
            views=views,
            expected=expected[1],
            previous_epoch=previous_epoch,
            skew_iterations=args.skew_iterations,
        )

        replay_epochs = []
        for replay in range(args.freshness_replays):
            variant_index = replay & 1
            set_variant(variant_index)
            graph.replay()
            barrier(group)
            previous_epoch, _ = validate_payload(
                rank=rank,
                views=views,
                expected=expected[variant_index],
                previous_epoch=previous_epoch,
                skew_iterations=args.skew_iterations,
            )
            replay_epochs.append(previous_epoch)

        result = {
            "rank": rank,
            "workspace_view_offset": args.workspace_view_offset,
            "workspace_bytes": layout.total_bytes,
            "tokens_per_source": args.tokens_per_source,
            "tokens_per_chunk": args.tokens_per_chunk,
            "chunks_per_source": layout.chunks_per_source,
            "hidden": args.hidden,
            "topk": args.topk,
            "final_epoch": previous_epoch,
            "replay_epochs": replay_epochs,
            "eager_failures": eager_failures,
            "capture_failures": capture_failures,
        }
        if rank == 0:
            print(
                "[PASS] TP2 real Stage1 MXFP8/scales/routes peer payload: "
                f"short tail, source skew, and {args.freshness_replays} graph replays",
                flush=True,
            )
            print("[RESULT_JSON] " + json.dumps(result, sort_keys=True), flush=True)

        graph = None
        del backing, workspace, variants, expected, views
        gc.collect()
        torch.cuda.synchronize()
        barrier(group)
    finally:
        graph = None
        cleanup_distributed(rank)


if __name__ == "__main__":
    main()
