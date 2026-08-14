# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP2 peer source/chunk publication regression for the Stage1 AG path.

Run on two gfx950 GPUs with::

    torchrun --standalone --nproc_per_node=2 \
      op_tests/multigpu_tests/test_flydsl_moe_tp_gemm1_ag.py

The registered workspace is deliberately a nonzero-offset tensor view. Each
rank publishes two real payload chunks to every peer, while the last source
executes a device-side delay before publication. Eager and alternating CUDA
Graph replays must observe the matching full 64-bit generation without host
epoch updates or signal memset.
"""

from __future__ import annotations

import argparse
import gc
import json

import torch

from aiter.ops.flydsl.kernels.moe_tp_ag_gemm1 import (
    launch_tp_stage1_peer_source_exchange,
    tp_stage1_peer_source_workspace_layout,
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
    chunks_per_source: int,
    payload_elements: int,
    device: torch.device,
) -> torch.Tensor:
    element = torch.arange(
        chunks_per_source * payload_elements,
        dtype=torch.int32,
        device=device,
    ).view(chunks_per_source, payload_elements)
    multiplier = 17 + variant * 12
    return (
        element * multiplier
        + source * 100_003
        + variant * 1_000_003
        + 41
    ).contiguous()


def workspace_views(workspace, layout, source_count, chunks_per_source, elements):
    records = source_count * chunks_per_source

    def i64_view(offset: int, count: int):
        return workspace[offset : offset + count * 8].view(torch.int64)

    payload = workspace[
        layout.payload_offset : layout.entry_offset
    ].view(torch.int32)[: records * elements]
    return {
        "payload": payload.view(source_count, chunks_per_source, elements),
        "entry": i64_view(layout.entry_offset, 1),
        "current_epoch": i64_view(layout.current_epoch_offset, 1),
        "payload_epoch": i64_view(layout.payload_epoch_offset, records).view(
            source_count, chunks_per_source
        ),
        "ready": i64_view(layout.ready_offset, records).view(
            source_count, chunks_per_source
        ),
        "observed_epoch": i64_view(layout.observed_epoch_offset, records).view(
            source_count, chunks_per_source
        ),
        "skew_scratch": i64_view(layout.skew_scratch_offset, 1),
        "errors": workspace[
            layout.errors_offset : layout.errors_offset + 4
        ].view(torch.int32),
    }


def validate_generation(
    *,
    rank: int,
    workspace_views_by_name,
    observed: torch.Tensor,
    expected: torch.Tensor,
    previous_epoch: int,
    expected_increment: int,
    skew_iterations: int,
):
    views = workspace_views_by_name
    epoch = int(views["current_epoch"].item())
    entry = int(views["entry"].item())
    failures = {
        "epoch_step": int(epoch != previous_epoch + expected_increment),
        "entry_epoch": int(entry != epoch),
        "payload": int((views["payload"] != expected).sum().item()),
        "observed": int((observed != expected).sum().item()),
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
            f"rank {rank} peer source epoch {epoch} failed: {failures}"
        )
    return epoch, failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="TP2 Stage1 peer source/chunk publication freshness gate"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--chunks-per-source", type=int, default=2)
    parser.add_argument("--payload-elements", type=int, default=64)
    parser.add_argument("--freshness-replays", type=int, default=16)
    parser.add_argument("--skew-iterations", type=int, default=20_000)
    parser.add_argument("--workspace-view-offset", type=int, default=2048)
    args = parser.parse_args()
    if args.tp_size != 2:
        parser.error("this peer freshness gate currently requires --tp-size=2")
    if args.chunks_per_source <= 0 or args.payload_elements <= 0:
        parser.error("chunk count and payload elements must be positive")
    if args.freshness_replays < 2 or args.skew_iterations <= 0:
        parser.error("freshness replays must be >=2 and skew iterations positive")
    if args.workspace_view_offset <= 0 or args.workspace_view_offset % 16:
        parser.error("workspace view offset must be positive and 16-byte aligned")
    return args


def main():
    args = parse_args()
    rank, world, _, device, group = setup_distributed(args.tp_size)
    graph = None
    try:
        layout = tp_stage1_peer_source_workspace_layout(
            source_count=world,
            chunks_per_source=args.chunks_per_source,
            payload_elements=args.payload_elements,
        )
        backing = torch.zeros(
            layout.total_bytes + args.workspace_view_offset,
            dtype=torch.uint8,
            device=device,
        )
        workspace = backing.narrow(
            0, args.workspace_view_offset, layout.total_bytes
        )
        if not workspace.is_contiguous():
            raise AssertionError("nonzero-offset workspace view is not contiguous")
        actual_view_offset = workspace.data_ptr() - backing.data_ptr()
        if actual_view_offset != args.workspace_view_offset:
            raise AssertionError(
                f"workspace pointer offset {actual_view_offset} does not match "
                f"requested {args.workspace_view_offset}"
            )

        peer_rank_data = flydsl_moe_tp_register_peer_workspace(workspace)
        observed = torch.empty(
            (
                world,
                args.chunks_per_source,
                args.payload_elements,
            ),
            dtype=torch.int32,
            device=device,
        )
        local_payload = torch.empty(
            (args.chunks_per_source, args.payload_elements),
            dtype=torch.int32,
            device=device,
        )
        variants = tuple(
            tuple(
                make_payload(
                    source,
                    variant,
                    chunks_per_source=args.chunks_per_source,
                    payload_elements=args.payload_elements,
                    device=device,
                )
                for source in range(world)
            )
            for variant in range(2)
        )
        expected = tuple(torch.stack(per_source) for per_source in variants)
        if int((expected[0] != expected[1]).sum().item()) == 0:
            raise AssertionError("freshness payload variants are identical")

        views = workspace_views(
            workspace,
            layout,
            world,
            args.chunks_per_source,
            args.payload_elements,
        )

        def exchange_body():
            launch_tp_stage1_peer_source_exchange(
                workspace=workspace,
                peer_rank_data=peer_rank_data,
                local_payload=local_payload,
                observed_payload=observed,
                rank=rank,
                source_count=world,
                chunks_per_source=args.chunks_per_source,
                payload_elements=args.payload_elements,
                skew_iterations=args.skew_iterations,
            )

        local_payload.copy_(variants[0][rank])
        previous_epoch = 0
        exchange_body()
        barrier(group)
        previous_epoch, eager_failures = validate_generation(
            rank=rank,
            workspace_views_by_name=views,
            observed=observed,
            expected=expected[0],
            previous_epoch=previous_epoch,
            expected_increment=1,
            skew_iterations=args.skew_iterations,
        )

        local_payload.copy_(variants[1][rank])
        graph = capture_graph(exchange_body, group, warmup_replays=0)
        barrier(group)
        capture_epoch, capture_failures = validate_generation(
            rank=rank,
            workspace_views_by_name=views,
            observed=observed,
            expected=expected[1],
            previous_epoch=previous_epoch,
            expected_increment=1,
            skew_iterations=args.skew_iterations,
        )
        previous_epoch = capture_epoch

        replay_epochs = []
        for replay in range(args.freshness_replays):
            variant_index = replay & 1
            local_payload.copy_(variants[variant_index][rank])
            graph.replay()
            barrier(group)
            previous_epoch, _ = validate_generation(
                rank=rank,
                workspace_views_by_name=views,
                observed=observed,
                expected=expected[variant_index],
                previous_epoch=previous_epoch,
                expected_increment=1,
                skew_iterations=args.skew_iterations,
            )
            replay_epochs.append(previous_epoch)

        result = {
            "rank": rank,
            "workspace_view_offset": actual_view_offset,
            "workspace_bytes": layout.total_bytes,
            "chunks_per_source": args.chunks_per_source,
            "payload_elements": args.payload_elements,
            "skewed_source": world - 1,
            "skew_iterations": args.skew_iterations,
            "eager_epoch": 1,
            "capture_epoch": capture_epoch,
            "final_epoch": previous_epoch,
            "replay_epochs": replay_epochs,
            "eager_failures": eager_failures,
            "capture_failures": capture_failures,
        }
        if rank == 0:
            print(
                "[PASS] TP2 Stage1 peer source/chunk publication: nonzero IPC "
                f"offset, source skew, and {args.freshness_replays} graph replays",
                flush=True,
            )
            print("[RESULT_JSON] " + json.dumps(result, sort_keys=True), flush=True)

        graph = None
        del backing, workspace, observed, local_payload, variants, expected, views
        gc.collect()
        torch.cuda.synchronize()
        barrier(group)
    finally:
        graph = None
        cleanup_distributed(rank)


if __name__ == "__main__":
    main()
