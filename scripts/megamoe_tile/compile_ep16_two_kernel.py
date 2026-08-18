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
    parser.add_argument("--worker-blocks", type=int, default=160)
    args = parser.parse_args()

    # Preserve full MLIR/LLVM/HSACO generation and replace only the final HIP
    # call, whose real pointers/DevComm exist only in the EP16 launcher.
    jfmod._build_call_state = lambda *unused_args, **unused_kwargs: _NoLaunch()

    s1 = Stage1ArenaLayout.create()
    s2 = Stage2ArenaLayout.create()
    combo = TwoKernelArenaLayout.compose(s1, s2)
    stage1 = compile_megamoe_tile_ep16_stage1(
        s1,
        s2,
        rank=args.rank,
        stage2_window_offset=combo.stage2_offset,
        worker_blocks=args.worker_blocks,
        enable_cco=True,
    )
    stage2 = compile_megamoe_tile_ep16_stage2_a4w4(combo, rank=args.rank)
    s1_args = (
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0),
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int32(128), fx.Int64(1),
        fx.Stream(None),
    )
    s2_args = (
        fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0), fx.Int64(0),
        fx.Int64(1), fx.Int32(128), fx.Int32(args.worker_blocks), fx.Int64(0),
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
