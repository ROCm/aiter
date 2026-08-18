#!/usr/bin/env python3
"""Compile-only smoke for the strict EP16 Stage-2.

The launcher uses a zero-sized grid so FlyDSL lowers and links the complete
kernel without entering its CCO protocol.  This is intentionally not a
functional test; the EP16 driver supplies real registered windows/DevComm.
"""

import torch

from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import Stage2ArenaLayout
from aiter.ops.flydsl.kernels.megamoe_tile.stage2 import (
    compile_megamoe_tile_ep16_stage2_a4w4,
)


def main():
    s1 = Stage1ArenaLayout.create()
    s2 = Stage2ArenaLayout.create()
    layout = TwoKernelArenaLayout.compose(s1, s2)
    launch = compile_megamoe_tile_ep16_stage2_a4w4(layout, rank=0)
    dummy = torch.zeros(1, dtype=torch.uint8, device="cuda")
    output = torch.empty((128, 7168), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()
    launch(
        0,
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
