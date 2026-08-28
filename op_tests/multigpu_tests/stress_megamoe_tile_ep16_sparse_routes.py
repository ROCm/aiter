# SPDX-License-Identifier: MIT
"""Correctness stress for sparse EP16 route counts, parity reuse, and capacity."""

from __future__ import annotations

import argparse
import json
import struct

import torch
import torch.distributed as dist

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_comm_only import (
    _lightweight_shared_inputs,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    _permuted_arbitrary_destination_oracle,
    _permuted_arbitrary_topk_cpu,
    _setup_dist,
)
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    MegaMoETileA4W4CommProbe,
)


def _prefix_masks(count: int) -> list[int]:
    masks = []
    for qp in range(4):
        bits = max(0, (int(count) + 3 - qp) // 4)
        masks.append((1 << bits) - 1)
    return masks


def _arbitrary_shared_inputs(
    shape: BenchmarkShape,
    rank: int,
    world: int,
    device: torch.device,
):
    shared = _lightweight_shared_inputs(
        shape,
        rank,
        world,
        device,
        quantize_for_mori=False,
        route_pattern="rank-balanced-hot",
    )
    topk_ids, route_weights, rank_slot_masks = _permuted_arbitrary_topk_cpu(
        shape, rank
    )
    shared.topk_ids = topk_ids.to(device)
    shared.route_weights = route_weights.to(device)
    return shared, rank_slot_masks


def _check_arbitrary_packed_records(
    operator,
    *,
    generation: int,
    shape: BenchmarkShape,
    topk_ids_cpu: torch.Tensor,
    route_weights_cpu: torch.Tensor,
    rank_slot_masks: list[list[int]],
) -> None:
    """Compare the fused producer's packed metadata with the CPU oracle."""

    from aiter.ops.flydsl.kernels.megamoe_tile.cco import read_window_bytes

    wire = operator.stage1_layout.wire
    parity = int(generation) & 1
    address = int(operator._runtime.window.local_ptr) + int(
        operator.stage1_layout.offset("dispatch_staging", parity=parity)
    )
    payload = read_window_bytes(address, shape.tokens * wire.record_bytes)
    expected_weight_bits = (
        route_weights_cpu.contiguous().view(torch.int32).tolist()
    )
    expected_ids = topk_ids_cpu.tolist()
    for token in range(shape.tokens):
        record = token * wire.record_bytes
        actual_ids = list(
            struct.unpack_from(
                f"<{shape.topk}i", payload, record + wire.ids_offset
            )
        )
        actual_weight_bits = list(
            struct.unpack_from(
                f"<{shape.topk}I", payload, record + wire.weights_offset
            )
        )
        actual_rank_masks = list(
            struct.unpack_from(
                f"<{shape.ep_size}H",
                payload,
                record + wire.rank_slot_masks_offset,
            )
        )
        actual_route_mask = struct.unpack_from(
            "<Q", payload, record + wire.route_mask_offset
        )[0]
        actual_source = struct.unpack_from(
            "<I", payload, record + wire.source_offset
        )[0]
        if actual_ids != [int(value) for value in expected_ids[token]]:
            raise AssertionError(f"token {token}: packed Top-K IDs mismatch")
        if actual_weight_bits != [
            int(value) & 0xFFFFFFFF for value in expected_weight_bits[token]
        ]:
            raise AssertionError(f"token {token}: packed Top-K weights mismatch")
        if actual_rank_masks != rank_slot_masks[token]:
            raise AssertionError(f"token {token}: packed rank-slot masks mismatch")
        if actual_route_mask != (1 << shape.topk) - 1:
            raise AssertionError(f"token {token}: packed route mask mismatch")
        if actual_source != operator.rank * shape.tokens + token:
            raise AssertionError(f"token {token}: packed source ID mismatch")


def _check_common(
    state: dict,
    *,
    generation: int,
    send_tokens: int,
    receive_tokens: int,
    receive_masks: list[int],
    routes: int,
    tiles: int,
    compute_jobs: int,
    expert_count: list[int],
    split_flags: int = 32,
    tile_pipeline: bool = False,
    require_overlap: bool = False,
) -> None:
    expected = {
        "dispatch_staging_ready_count": 128,
        "sparse_remote_send_count": int(send_tokens),
        "sparse_remote_token_ready_count": int(receive_tokens),
        "sparse_remote_qp_ready_count": 4,
        "sparse_remote_request_nonzero_count": 4,
        "sparse_remote_batch_ready": int(generation),
        "sparse_remote_credit": int(generation),
        "sparse_remote_consumed": 8,
        "expert_count_sum": int(routes),
        "tile_arrived_sum": int(routes),
        "tile_alloc": int(tiles),
        "num_valid": int(tiles * 32),
        "h1_queue_tail": int(tiles * 24),
        "compute_done": int(compute_jobs),
        "stage1_done": int(generation),
        "stage1_error_count": 0,
    }
    for field, value in expected.items():
        if state[field] != value:
            raise AssertionError(
                f"{field}={state[field]!r}, expected={value!r}"
            )
    if state["expert_count"] != expert_count:
        raise AssertionError(
            f"expert_count={state['expert_count']!r}, expected={expert_count!r}"
        )
    if state["sparse_remote_qp_token_masks"] != receive_masks:
        raise AssertionError(
            "QP masks mismatch: "
            f"{state['sparse_remote_qp_token_masks']} != {receive_masks}"
        )
    if state["comm_role_eos"] != [generation] * 8:
        raise AssertionError("communication EOS coverage is incomplete")
    if state["split_flags_ready_per_dest"] != [split_flags] * 8:
        raise AssertionError("split fanout completion coverage is incomplete")
    if tile_pipeline and state.get("queue_permutation_mismatch") != 0:
        raise AssertionError("tile-pipeline queue is not a complete permutation")
    if require_overlap and routes > 0:
        early_tiles = state.get("early_full_tiles", 0)
        started = state.get("gmm_jobs_started_before_all_comm_eos", 0)
        completed = state.get("gmm_jobs_completed_before_all_comm_eos", 0)
        if not (
            0 < early_tiles <= tiles
            and 0 < completed <= started <= compute_jobs
        ):
            raise AssertionError("tile-pipeline did not overlap GMM1 before EOS")
    elif require_overlap:
        if any(
            state.get(field, 0) != 0
            for field in (
                "early_full_tiles",
                "gmm_jobs_started_before_all_comm_eos",
                "gmm_jobs_completed_before_all_comm_eos",
            )
        ):
            raise AssertionError("zero-work rank reported tile-pipeline activity")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        default="1,2,3,4,31,32,33,63,64,127,128,128,128,0,0,128,128,0,0",
    )
    parser.add_argument("--hot-ranks", default="0,8")
    parser.add_argument("--single-expert-rank", type=int, default=0)
    parser.add_argument("--rejoin", action="store_true")
    parser.add_argument("--tile-pipeline", action="store_true")
    parser.add_argument("--tile-pipeline-instrument", action="store_true")
    parser.add_argument(
        "--tile-pipeline-fanout-shards",
        type=int,
        choices=(8, 12, 16),
        default=16,
    )
    parser.add_argument("--stage2-expected-sweep", action="store_true")
    parser.add_argument("--poison-stage2", action="store_true")
    deferred_stage2 = parser.add_mutually_exclusive_group()
    deferred_stage2.add_argument("--route-store-stage2", action="store_true")
    deferred_stage2.add_argument("--rank-local-stage2", action="store_true")
    parser.add_argument(
        "--stage2-node-reduce-vec-bytes",
        type=int,
        choices=(4, 8, 16),
        default=8,
    )
    parser.add_argument(
        "--stage2-node-reduce-load-schedule",
        choices=("interleaved", "load_first"),
        default="interleaved",
    )
    parser.add_argument(
        "--stage2-node-reduce-work-schedule",
        choices=("static_strided", "dynamic_head"),
        default="static_strided",
    )
    parser.add_argument(
        "--stage2-node-reduce-rejoin-blocks",
        type=int,
        choices=(0, 8, 16, 32),
        default=0,
    )
    parser.add_argument(
        "--stage2-rank-epilogue-lds-addressing",
        choices=("expanded", "dynamic_base"),
        default="expanded",
    )
    args = parser.parse_args()
    counts = [int(value) for value in args.counts.split(",") if value]
    hot_ranks = [int(value) for value in args.hot_ranks.split(",") if value]
    if any(value < 0 or value > 128 for value in counts):
        raise ValueError("every remote count must be in [0,128]")
    if any(value < 0 or value >= 16 for value in hot_ranks):
        raise ValueError("every hot rank must be in [0,16)")
    if not -1 <= args.single_expert_rank < 16:
        raise ValueError("single-expert-rank must be -1 or in [0,16)")
    if args.rejoin and args.tile_pipeline:
        raise ValueError("choose either post-EOS rejoin or tile pipeline")
    if args.tile_pipeline_instrument and not args.tile_pipeline:
        raise ValueError("tile-pipeline instrumentation requires --tile-pipeline")
    if args.stage2_node_reduce_vec_bytes == 16 and not args.rank_local_stage2:
        parser.error(
            "16-byte Stage2 node reduction requires --rank-local-stage2"
        )
    if args.stage2_node_reduce_rejoin_blocks > 0 and not (
        args.rank_local_stage2
        and args.stage2_node_reduce_work_schedule == "dynamic_head"
    ):
        parser.error(
            "Stage2 reducer rejoin requires --rank-local-stage2 and "
            "--stage2-node-reduce-work-schedule=dynamic_head"
        )
    if args.stage2_rank_epilogue_lds_addressing == "dynamic_base" and not (
        args.rank_local_stage2
        and args.stage2_node_reduce_vec_bytes == 8
        and args.stage2_node_reduce_load_schedule == "load_first"
        and args.stage2_node_reduce_work_schedule == "static_strided"
        and args.stage2_node_reduce_rejoin_blocks == 0
    ):
        parser.error(
            "dynamic_base LDS addressing requires rank-local vec8/load_first/"
            "static_strided with rejoin_blocks=0"
        )
    if (
        args.route_store_stage2
        or args.rank_local_stage2
        or args.poison_stage2
    ) and not (
        args.stage2_expected_sweep
    ):
        parser.error(
            "deferred Stage2 and --poison-stage2 require "
            "--stage2-expected-sweep"
        )
    has_gmm1 = args.rejoin or args.tile_pipeline

    shape = BenchmarkShape()
    shape.validate()
    rank, world, _local_rank, device = _setup_dist(needs_mori=False)
    if world != 16:
        raise ValueError(f"sparse route stress requires EP16, got {world}")

    seed = _lightweight_shared_inputs(
        shape,
        rank,
        world,
        device,
        quantize_for_mori=False,
        prepare_stage1_weights=has_gmm1,
        route_pattern="paired-rank-remote-prefix",
        remote_token_count=0,
    )
    weights = seed.prepared_weights
    operator = MegaMoETileA4W4CommProbe(
        rank=rank,
        world_size=world,
        model_dim=shape.hidden,
        inter_dim=shape.inter,
        experts=shape.experts,
        topk=shape.topk,
        quant="a4w4",
        w1=weights.w1,
        w1_scale=weights.w1_scale,
        w2=weights.w2,
        w2_scale=weights.w2_scale,
        max_tok_per_rank=shape.tokens,
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
        probe_stage=("both" if args.stage2_expected_sweep else "stage1"),
        stage1_mode=(
            "internodev1_tilepipe"
            if args.tile_pipeline
            else (
                "internodev1_split128x2_rejoin"
                if args.rejoin
                else "internodev1_split128x2"
            )
        ),
        stage1_cco_geometry="sparse_wqe",
        stage1_phase="full",
        stage1_tile_pipeline_instrument=args.tile_pipeline_instrument,
        stage1_tile_pipeline_fanout_shards=(
            args.tile_pipeline_fanout_shards
        ),
        stage2_worker_blocks=(
            176
            if args.route_store_stage2 or args.rank_local_stage2
            else 160
        ),
        stage2_node_accumulation_mode=(
            "rank_local"
            if args.rank_local_stage2
            else ("route_store" if args.route_store_stage2 else "direct_atomic")
        ),
        stage2_node_reduce_blocks=16,
        stage2_node_reduce_vec_bytes=args.stage2_node_reduce_vec_bytes,
        stage2_node_reduce_schedule="token",
        stage2_node_reduce_load_schedule=(
            args.stage2_node_reduce_load_schedule
            if args.rank_local_stage2
            else "interleaved"
        ),
        stage2_node_reduce_work_schedule=(
            args.stage2_node_reduce_work_schedule
            if args.rank_local_stage2
            else "static_strided"
        ),
        stage2_node_reduce_rejoin_blocks=(
            args.stage2_node_reduce_rejoin_blocks
            if args.rank_local_stage2
            else 0
        ),
        stage2_rank_epilogue_lds_addressing=(
            args.stage2_rank_epilogue_lds_addressing
            if args.rank_local_stage2
            else "expanded"
        ),
        stage2_return_chunk_tokens=(
            16 if args.rank_local_stage2 else 8
        ),
        stage2_rail_return_schedule=(
            "compact" if args.rank_local_stage2 else "lockstep"
        ),
    )

    completed = []
    try:
        for count in counts:
            shared = _lightweight_shared_inputs(
                shape,
                rank,
                world,
                device,
                quantize_for_mori=False,
                route_pattern="paired-rank-remote-prefix",
                remote_token_count=count,
            )
            dist.barrier()
            operator._generation += 1
            generation = operator._generation
            operator._launch_stage1(
                shared.x,
                shared.route_weights,
                shared.topk_ids,
                shape.tokens,
                generation,
                operator._flydsl_stream(None),
            )
            torch.cuda.synchronize(device)
            state = operator.debug_stage1_comm_snapshot()
            masks = _prefix_masks(count)
            _check_common(
                state,
                generation=generation,
                send_tokens=count,
                receive_tokens=count,
                receive_masks=masks,
                routes=2048,
                tiles=88,
                compute_jobs=88 * 24 if has_gmm1 else 0,
                expert_count=[40] * 32 + [32] * 24,
                split_flags=(
                    2 * args.tile_pipeline_fanout_shards
                    if args.tile_pipeline
                    else 32
                ),
                tile_pipeline=args.tile_pipeline,
                require_overlap=args.tile_pipeline_instrument,
            )
            completed.append(
                {
                    "generation": generation,
                    "kind": "remote_prefix",
                    "remote_tokens": count,
                    "qp_masks": masks,
                }
            )
            dist.barrier()

        # One deterministic arbitrary-Top-K generation deliberately breaks the
        # adjacent-pair structure used above.  It mixes local and remote ranks,
        # maps non-adjacent slots to the same rank, repeats exact expert IDs,
        # and assigns a distinct FP32 weight to every slot.  Validate both the
        # producer's packed record and the destination's expanded route set.
        shared, rank_slot_masks = _arbitrary_shared_inputs(
            shape, rank, world, device
        )
        topk_ids_cpu = shared.topk_ids.cpu()
        route_weights_cpu = shared.route_weights.cpu()
        oracle = _permuted_arbitrary_destination_oracle(shape, rank)
        if oracle["routes"] != 2048 or oracle["unique_sources"] != 768:
            raise AssertionError(
                "arbitrary fixture must produce 2048 routes over 768 source rows"
            )
        dist.barrier()
        operator._generation += 1
        generation = operator._generation
        operator._launch_stage1(
            shared.x,
            shared.route_weights,
            shared.topk_ids,
            shape.tokens,
            generation,
            operator._flydsl_stream(None),
        )
        torch.cuda.synchronize(device)
        state = operator.debug_stage1_comm_snapshot()
        _check_common(
            state,
            generation=generation,
            send_tokens=shape.tokens,
            receive_tokens=shape.tokens,
            receive_masks=_prefix_masks(shape.tokens),
            routes=int(oracle["routes"]),
            tiles=int(oracle["tiles"]),
            compute_jobs=(
                int(oracle["tiles"]) * 24 if has_gmm1 else 0
            ),
            expert_count=list(oracle["expert_count"]),
            split_flags=(
                2 * args.tile_pipeline_fanout_shards
                if args.tile_pipeline
                else 32
            ),
            tile_pipeline=args.tile_pipeline,
            require_overlap=args.tile_pipeline_instrument,
        )
        _check_arbitrary_packed_records(
            operator,
            generation=generation,
            shape=shape,
            topk_ids_cpu=topk_ids_cpu,
            route_weights_cpu=route_weights_cpu,
            rank_slot_masks=rank_slot_masks,
        )
        from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import MegaMoETileA4W4

        canonical = MegaMoETileA4W4.debug_direct_tile_snapshot(operator)[
            "canonical_h1"
        ]
        expected_canonical = {
            "valid_rows": int(oracle["routes"]),
            "unique_input_rows": int(oracle["unique_sources"]),
            "shared_input_route_rows": (
                int(oracle["routes"]) - int(oracle["unique_sources"])
            ),
            "invalid_input_rows": 0,
            "duplicate_packed_keys": 0,
            "metadata_sha256": str(oracle["metadata_sha256"]),
        }
        canonical_mismatch = {
            field: (canonical.get(field), expected)
            for field, expected in expected_canonical.items()
            if canonical.get(field) != expected
        }
        if canonical_mismatch:
            raise AssertionError(
                f"arbitrary fixture canonical mismatch: {canonical_mismatch}"
            )
        completed.append(
            {
                "generation": generation,
                "kind": "permuted_arbitrary_topk",
                "routes": int(oracle["routes"]),
                "unique_sources": int(oracle["unique_sources"]),
                "tiles": int(oracle["tiles"]),
                "metadata_sha256": str(oracle["metadata_sha256"]),
            }
        )
        dist.barrier()

        if args.stage2_expected_sweep:
            for local_routes in range(17):
                shared = _lightweight_shared_inputs(
                    shape,
                    rank,
                    world,
                    device,
                    quantize_for_mori=False,
                    route_pattern="node-route-count",
                    local_route_count=local_routes,
                )
                dist.barrier()
                operator._generation += 1
                generation = operator._generation
                stream = operator._flydsl_stream(None)
                operator._launch_stage1(
                    shared.x,
                    shared.route_weights,
                    shared.topk_ids,
                    shape.tokens,
                    generation,
                    stream,
                )
                torch.cuda.synchronize(device)
                stage1_state = operator.debug_stage1_comm_snapshot()
                local_expert = rank % shape.gpus_per_node
                expert_count = [0] * shape.local_experts
                expert_count[local_expert] = 1024
                expert_count[local_expert + 8] = 1024
                remote_tokens = 0 if local_routes == 16 else 128
                _check_common(
                    stage1_state,
                    generation=generation,
                    send_tokens=remote_tokens,
                    receive_tokens=remote_tokens,
                    receive_masks=_prefix_masks(remote_tokens),
                    routes=2048,
                    tiles=64,
                    compute_jobs=64 * 24 if has_gmm1 else 0,
                    expert_count=expert_count,
                    split_flags=(
                        2 * args.tile_pipeline_fanout_shards
                        if args.tile_pipeline
                        else 32
                    ),
                    tile_pipeline=args.tile_pipeline,
                    require_overlap=args.tile_pipeline_instrument,
                )
                dist.barrier()
                if args.poison_stage2:
                    operator.poison_stage2_buffers()
                    dist.barrier()
                operator._launch_stage2(
                    shape.tokens, generation, stream
                )
                torch.cuda.synchronize(device)
                stage2_state = operator.debug_stage2_scoreboard_snapshot()
                if stage2_state["node_expected"] != [local_routes] * 128:
                    raise AssertionError(
                        "Stage2 local expected route count mismatch"
                    )
                if stage2_state["node_done"] != [local_routes] * 128:
                    raise AssertionError("Stage2 local done count mismatch")
                if stage2_state["node_ready"] != [1] * 128:
                    raise AssertionError("Stage2 local ready coverage mismatch")
                expected_accumulation_mode = (
                    "rank_local"
                    if args.rank_local_stage2
                    else (
                        "route_store"
                        if args.route_store_stage2
                        else "direct_atomic"
                    )
                )
                if (
                    stage2_state["node_accumulation_mode"]
                    != expected_accumulation_mode
                ):
                    raise AssertionError("Stage2 accumulation mode mismatch")
                final_expected = (
                    128 if args.rank_local_stage2 else 128 * 28
                )
                expected_stage2 = {
                    "node_expected_done_mismatch": 0,
                    "node_not_ready": 0,
                    "final_done": final_expected,
                    "final_expected": final_expected,
                    "return_groups_ready": (
                        shape.tokens
                        // int(operator._stage2.return_chunk_tokens)
                    ),
                    "return_consumed": generation,
                    "stage2_error_count": 0,
                    "node_token_done_mismatch": 0,
                    "node_partial_done_mismatch": 0,
                }
                if args.rank_local_stage2:
                    local_rank = rank % shape.gpus_per_node
                    active_source_nodes = int(
                        local_rank < local_routes
                        or local_rank + shape.gpus_per_node < local_routes
                    ) + int(
                        local_rank >= local_routes
                        or local_rank + shape.gpus_per_node >= local_routes
                    )
                    expected_stage2.update(
                        {
                            "rank_local_active_tokens": (
                                active_source_nodes
                                * shape.gpus_per_node
                                * shape.tokens
                            ),
                            "rank_local_pending_nonzero": 0,
                            "rank_local_pending_nonzero_all": 0,
                            "rank_local_ready_missing": 0,
                            "rank_local_ready_unexpected": 0,
                            "rank_reduce_queue_expected": (
                                int(local_routes > 0)
                                + int(local_routes < shape.topk)
                            )
                            * shape.tokens,
                            "rank_reduce_queue_count": (
                                int(local_routes > 0)
                                + int(local_routes < shape.topk)
                            )
                            * shape.tokens,
                            "rank_reduce_queue_tail": (
                                int(local_routes > 0)
                                + int(local_routes < shape.topk)
                            )
                            * shape.tokens,
                            "rank_reduce_queue_head": (
                                (
                                    int(local_routes > 0)
                                    + int(local_routes < shape.topk)
                                )
                                * shape.tokens
                                + 16
                                + args.stage2_node_reduce_rejoin_blocks
                                if args.stage2_node_reduce_work_schedule
                                == "dynamic_head"
                                else 0
                            ),
                            "rank_reduce_queue_permutation_mismatch": 0,
                            "return_groups_ready": (
                                0
                                if local_routes == shape.topk
                                else shape.tokens
                                // int(operator._stage2.return_chunk_tokens)
                            ),
                        }
                    )
                for field, value in expected_stage2.items():
                    if stage2_state[field] != value:
                        raise AssertionError(
                            f"Stage2 {field}={stage2_state[field]}, expected={value}"
                        )
                if args.poison_stage2:
                    zero_state = operator.debug_stage2_zero_payload_snapshot()
                    nonzero_fields = {
                        field: value
                        for field, value in zero_state.items()
                        if field != "generation" and value != 0
                    }
                    if nonzero_fields:
                        raise AssertionError(
                            f"Stage2 zero payload retained poison: {nonzero_fields}"
                        )
                completed.append(
                    {
                        "generation": generation,
                        "kind": "stage2_expected",
                        "local_routes": local_routes,
                    }
                )
                dist.barrier()

        for hot_rank in hot_ranks:
            shared = _lightweight_shared_inputs(
                shape,
                rank,
                world,
                device,
                quantize_for_mori=False,
                route_pattern="single-rank-max",
                hot_rank=hot_rank,
            )
            dist.barrier()
            operator._generation += 1
            generation = operator._generation
            operator._launch_stage1(
                shared.x,
                shared.route_weights,
                shared.topk_ids,
                shape.tokens,
                generation,
                operator._flydsl_stream(None),
            )
            torch.cuda.synchronize(device)
            state = operator.debug_stage1_comm_snapshot()
            hot_node = hot_rank // shape.gpus_per_node
            local_node = rank // shape.gpus_per_node
            send_tokens = 0 if local_node == hot_node else 128
            receive_tokens = 128 if local_node == hot_node else 0
            _check_common(
                state,
                generation=generation,
                send_tokens=send_tokens,
                receive_tokens=receive_tokens,
                receive_masks=_prefix_masks(receive_tokens),
                routes=32768 if rank == hot_rank else 0,
                tiles=1024 if rank == hot_rank else 0,
                compute_jobs=(
                    1024 * 24 if has_gmm1 and rank == hot_rank else 0
                ),
                expert_count=(
                    [2048] * 16 + [0] * 40
                    if rank == hot_rank
                    else [0] * 56
                ),
                split_flags=(
                    2 * args.tile_pipeline_fanout_shards
                    if args.tile_pipeline
                    else 32
                ),
                tile_pipeline=args.tile_pipeline,
                require_overlap=args.tile_pipeline_instrument,
            )
            completed.append(
                {
                    "generation": generation,
                    "kind": "single_rank_max",
                    "hot_rank": hot_rank,
                    "send_tokens": send_tokens,
                    "receive_tokens": receive_tokens,
                }
            )
            dist.barrier()

        if args.single_expert_rank >= 0:
            hot_rank = args.single_expert_rank
            shared = _lightweight_shared_inputs(
                shape,
                rank,
                world,
                device,
                quantize_for_mori=False,
                route_pattern="single-expert-max",
                hot_rank=hot_rank,
            )
            dist.barrier()
            operator._generation += 1
            generation = operator._generation
            operator._launch_stage1(
                shared.x,
                shared.route_weights,
                shared.topk_ids,
                shape.tokens,
                generation,
                operator._flydsl_stream(None),
            )
            torch.cuda.synchronize(device)
            state = operator.debug_stage1_comm_snapshot()
            hot_node = hot_rank // shape.gpus_per_node
            local_node = rank // shape.gpus_per_node
            send_tokens = 0 if local_node == hot_node else 128
            receive_tokens = 128 if local_node == hot_node else 0
            _check_common(
                state,
                generation=generation,
                send_tokens=send_tokens,
                receive_tokens=receive_tokens,
                receive_masks=_prefix_masks(receive_tokens),
                routes=32768 if rank == hot_rank else 0,
                tiles=1024 if rank == hot_rank else 0,
                compute_jobs=(
                    1024 * 24 if has_gmm1 and rank == hot_rank else 0
                ),
                expert_count=(
                    [32768] + [0] * 55
                    if rank == hot_rank
                    else [0] * 56
                ),
                split_flags=(
                    2 * args.tile_pipeline_fanout_shards
                    if args.tile_pipeline
                    else 32
                ),
                tile_pipeline=args.tile_pipeline,
                require_overlap=args.tile_pipeline_instrument,
            )
            completed.append(
                {
                    "generation": generation,
                    "kind": "single_expert_max",
                    "hot_rank": hot_rank,
                    "send_tokens": send_tokens,
                    "receive_tokens": receive_tokens,
                }
            )
            dist.barrier()

        print(
            "MEGAMOE_SPARSE_ROUTE_STRESS_RESULT "
            + json.dumps(
                {
                    "rank": rank,
                    "status": "pass",
                    "completed": completed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        operator.close()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
