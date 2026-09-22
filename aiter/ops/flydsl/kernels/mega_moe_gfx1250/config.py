# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250-only launch geometry for Stage2-fused MegaMoE."""

_WAVE_SIZE = 32
_LANE_MASK = _WAVE_SIZE - 1
_LOG2_WAVE_SIZE = 5

_DISPATCH_EP4 = (
    (256, 128, 16),
    (512, 192, 32),
    (4096, 192, 32),
    (None, 192, 32),
)

_DISPATCH_EP4_TOPK6 = (
    (256, 128, 16),
    (512, 192, 32),
    (1024, 192, 32),
    (None, 256, 32),
)

_DISPATCH_EP8 = (
    (256, 128, 16),
    (1024, 128, 32),
    (None, 128, 32),
)

_DISPATCH_SCHEDULES = {
    (4, 7168, 8): _DISPATCH_EP4,
    (4, 7168, 6): _DISPATCH_EP4_TOPK6,
    (8, 7168, 8): _DISPATCH_EP8,
    (8, 7168, 6): _DISPATCH_EP8,
}

# The TDM dispatch is a different kernel with different optima and a much
# tighter LDS budget (one hidden-dim payload tile per warp), so it must not
# inherit the vector table's 192-256 x 32 grids -- 32 warps of a 7168-wide bf16
# tile want 448 KB against a 320 KB budget and fail to build outright. From a
# gfx1250 EP4 hidden-7168 geometry sweep; topk does not move the dispatch half.
#   ct      64x8  128x8  64x16 128x16     (dispatch us, graph)
#   64      50.0   49.8   57.6   56.6
#   512     53.3   53.2   63.5   62.5
#   1024    73.2   65.5   72.5   72.1
#   2048   108.8   91.8  113.3   99.6
#   4096   187.3  157.6  155.6  161.2
_DISPATCH_EP4_TDM = (
    (512, 128, 8),
    (2048, 128, 8),
    (4096, 64, 16),
    (None, 128, 16),
)

# EP8 TDM: no dedicated sweep yet, so the EP4 columns carry over with the 4096
# bucket folded into the tail.
_DISPATCH_EP8_TDM = (
    (512, 128, 8),
    (2048, 128, 8),
    (None, 128, 16),
)

_DISPATCH_TDM_SCHEDULES = {
    (4, 7168, 8): _DISPATCH_EP4_TDM,
    (4, 7168, 6): _DISPATCH_EP4_TDM,
    (8, 7168, 8): _DISPATCH_EP8_TDM,
    (8, 7168, 6): _DISPATCH_EP8_TDM,
}


def _select_dispatch_config(
    world_size: int, hidden_dim: int, topk: int, tdm: bool = False
) -> dict[str, object]:
    if tdm:
        table, fallback_ep8, fallback = (
            _DISPATCH_TDM_SCHEDULES,
            _DISPATCH_EP8_TDM,
            _DISPATCH_EP4_TDM,
        )
    else:
        table, fallback_ep8, fallback = (
            _DISPATCH_SCHEDULES,
            _DISPATCH_EP8,
            _DISPATCH_EP4,
        )
    schedule = table.get((world_size, hidden_dim, topk))
    if schedule is None:
        schedule = fallback_ep8 if world_size == 8 else fallback
    _, block, warp = schedule[-1]
    return {
        "dispatch_block_num": block,
        "dispatch_warp_num_per_block": warp,
        "schedule": schedule,
    }
