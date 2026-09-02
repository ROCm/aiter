#!/usr/bin/env python3
"""Lower/link both strict EP16 kernels without executing device code."""

import argparse

import flydsl.compiler as flyc
import flydsl.compiler.jit_function as jfmod
import flydsl.expr as fx

from aiter.ops.flydsl.kernels.megamoe_tile.stage1 import (
    compile_megamoe_tile_ep16_stage1,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage2 import (
    compile_megamoe_tile_ep16_stage2_a4w4,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import Stage2ArenaLayout


class _NoLaunch:
    def __call__(self, _args):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument(
        "--max-routes-per-token-per-rank", type=int, default=None
    )
    parser.add_argument("--worker-blocks", type=int, default=160)
    parser.add_argument(
        "--staged-ring",
        action="store_true",
        help="build Stage-1 against the staged_ring Stage-2 arena layout",
    )
    args = parser.parse_args()

    # Preserve full MLIR/LLVM/HSACO generation and replace only the final HIP
    # call, whose real pointers/DevComm exist only in the EP16 launcher.
    jfmod._build_call_state = lambda *unused_args, **unused_kwargs: _NoLaunch()

    s1 = Stage1ArenaLayout.create(
        max_tokens=args.tokens,
        max_routes_per_token_per_rank=args.max_routes_per_token_per_rank,
    )
    s2 = Stage2ArenaLayout.create(
        max_tokens=args.tokens,
        include_rank_partials=True,
        include_staged_ring=args.staged_ring,
    )
    combo = TwoKernelArenaLayout.compose(s1, s2)
    stage1 = compile_megamoe_tile_ep16_stage1(
        s1,
        s2,
        rank=args.rank,
        stage2_window_offset=combo.stage2_offset,
        worker_blocks=args.worker_blocks,
        enable_cco=True,
    )
    stage2 = compile_megamoe_tile_ep16_stage2_a4w4(
        combo,
        rank=args.rank,
        final_combine_blocks=4,
        return_chunk_tokens=8,
        rail_return_schedule="compact",
        node_accumulation_mode="rank_local",
        rank_accumulation_mode="staged_ring" if args.staged_ring else "atomic",
        node_reduce_blocks=16,
        node_reduce_vec_bytes=8,
        node_reduce_load_schedule="load_first",
        node_reduce_work_schedule="dynamic_head",
        group_pipeline_schedule="a_double_buffer",
    )
    s1_args = (
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0),
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int32(args.tokens), fx.Int64(1),
        fx.Stream(None),
    )
    s2_args = (
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0),
        fx.Int64(1), fx.Int32(args.tokens), fx.Int32(args.worker_blocks), fx.Int64(0),
        fx.Stream(None),
    )
    flyc.compile(stage1, *s1_args)
    flyc.compile(stage2, *s2_args)
    print("STAGE1_COMPILE_OK", stage1.kernel_name, "LDS", stage1.lds_bytes)
    print("STAGE2_COMPILE_OK", stage2.kernel_name, "LDS", stage2.lds_bytes)
    print(
        "ARENA_BYTES",
        "stage1", s1.total_bytes,
        "stage2", s2.total_bytes,
        "stage2_offset", combo.stage2_offset,
        "total", combo.total_bytes,
    )


if __name__ == "__main__":
    main()
