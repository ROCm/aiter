#!/usr/bin/env python3
"""Compile-only smoke for the strict EP16 Stage-2.

The launcher uses a zero-sized grid so FlyDSL lowers and links the complete
kernel without entering its CCO protocol.  This is intentionally not a
functional test; the EP16 driver supplies real registered windows/DevComm.
"""

import argparse

import torch

from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import Stage2ArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2 import (
    compile_megamoe_tile_ep16_stage2_a4w4,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--worker-blocks",
        type=int,
        default=177,
        help="resident Stage-2 grid used for cache-key-compatible warmup",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=(
            "full",
            "init_only",
            "atomic_only",
            "gmm2_only",
            "gmm2_atomic_only",
            "route_store_only",
            "return_only",
        ),
        default="full",
    )
    parser.add_argument("--accumulator-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--final-combine-blocks", type=int, default=14)
    parser.add_argument("--gmm-schedule", default="persistent_queue")
    parser.add_argument("--return-chunk-tokens", type=int, default=8)
    parser.add_argument("--bf16-atomic-kind", default="buffer")
    parser.add_argument(
        "--node-accumulation-mode",
        choices=("direct_atomic", "route_store", "rank_local"),
        default="direct_atomic",
    )
    parser.add_argument(
        "--rank-accumulation-mode",
        choices=("atomic", "staged_reduce"),
        default="atomic",
    )
    parser.add_argument("--node-reduce-blocks", type=int, default=32)
    parser.add_argument(
        "--node-reduce-vec-bytes", type=int, choices=(4, 8, 16), default=4
    )
    parser.add_argument(
        "--node-reduce-schedule",
        choices=("token", "group", "tile"),
        default="token",
    )
    parser.add_argument(
        "--node-reduce-load-schedule",
        choices=("interleaved", "load_first"),
        default="interleaved",
    )
    parser.add_argument(
        "--node-reduce-work-schedule",
        choices=("static_strided", "dynamic_head"),
        default="static_strided",
    )
    parser.add_argument(
        "--node-reduce-rejoin-blocks",
        type=int,
        choices=(0, 8, 16, 32),
        default=0,
    )
    parser.add_argument(
        "--rank-epilogue-lds-addressing",
        choices=("expanded", "dynamic_base"),
        default="expanded",
    )
    parser.add_argument(
        "--rail-return-schedule",
        choices=("lockstep", "qp_independent", "qp_prepost", "compact"),
        default="lockstep",
    )
    parser.add_argument("--epilogue-schedule", default="lane32_meta")
    parser.add_argument("--n-tile-group", type=int, default=2)
    parser.add_argument("--group-pipeline-schedule", default="a_double_buffer")
    parser.add_argument("--scoreboard-schedule", default="wave0")
    parser.add_argument("--atomic-issue-schedule", default="interleaved")
    parser.add_argument("--timeline-instrument", action="store_true")
    parser.add_argument("--kernel-name-override")
    args = parser.parse_args()
    if (
        args.node_reduce_vec_bytes == 16
        and args.node_accumulation_mode != "rank_local"
    ):
        parser.error(
            "16-byte node reduction requires "
            "--node-accumulation-mode=rank_local"
        )
    if args.rank_epilogue_lds_addressing == "dynamic_base" and not (
        args.node_accumulation_mode == "rank_local"
        and args.node_reduce_vec_bytes == 8
        and args.node_reduce_load_schedule == "load_first"
        and args.node_reduce_work_schedule == "static_strided"
        and args.node_reduce_rejoin_blocks == 0
    ):
        parser.error(
            "dynamic_base LDS addressing requires rank_local, vec8, "
            "load_first, static_strided reduction, and rejoin_blocks=0"
        )
    s1 = Stage1ArenaLayout.create()
    s2 = Stage2ArenaLayout.create(
        include_route_slots=args.node_accumulation_mode == "route_store",
        include_rank_partials=args.node_accumulation_mode == "rank_local",
        include_staged_reduce=(
            args.node_accumulation_mode == "rank_local"
            and args.rank_accumulation_mode == "staged_reduce"
        ),
        include_staged_ring=False,
    )
    layout = TwoKernelArenaLayout.compose(s1, s2)
    launch = compile_megamoe_tile_ep16_stage2_a4w4(
        layout,
        rank=args.rank,
        diagnostic_mode=args.diagnostic_mode,
        accumulator_dtype=args.accumulator_dtype,
        final_combine_blocks=args.final_combine_blocks,
        gmm_schedule=args.gmm_schedule,
        return_chunk_tokens=args.return_chunk_tokens,
        bf16_atomic_kind=args.bf16_atomic_kind,
        node_accumulation_mode=args.node_accumulation_mode,
        rank_accumulation_mode=args.rank_accumulation_mode,
        node_reduce_blocks=args.node_reduce_blocks,
        node_reduce_vec_bytes=args.node_reduce_vec_bytes,
        node_reduce_schedule=args.node_reduce_schedule,
        node_reduce_load_schedule=args.node_reduce_load_schedule,
        node_reduce_work_schedule=args.node_reduce_work_schedule,
        node_reduce_rejoin_blocks=args.node_reduce_rejoin_blocks,
        rank_epilogue_lds_addressing=args.rank_epilogue_lds_addressing,
        rail_return_schedule=args.rail_return_schedule,
        epilogue_schedule=args.epilogue_schedule,
        n_tile_group=args.n_tile_group,
        group_pipeline_schedule=args.group_pipeline_schedule,
        scoreboard_schedule=args.scoreboard_schedule,
        atomic_issue_schedule=args.atomic_issue_schedule,
        timeline_instrument=args.timeline_instrument,
        kernel_name_override=args.kernel_name_override,
    )
    dummy = torch.zeros(1, dtype=torch.uint8, device="cuda")
    output = torch.empty((128, 7168), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()
    launch(
        args.worker_blocks,
        0,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        1,
        128,
        0,
        output.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize()
    print(launch.kernel_name)
    print("lds_bytes", launch.lds_bytes)


if __name__ == "__main__":
    main()
