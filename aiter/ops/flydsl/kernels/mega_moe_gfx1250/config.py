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


def _select_dispatch_config(
    world_size: int, hidden_dim: int, topk: int
) -> dict[str, object]:
    schedule = _DISPATCH_SCHEDULES.get((world_size, hidden_dim, topk))
    if schedule is None:
        schedule = _DISPATCH_EP8 if world_size == 8 else _DISPATCH_EP4
    _, block, warp = schedule[-1]
    return {
        "dispatch_block_num": block,
        "dispatch_warp_num_per_block": warp,
        "schedule": schedule,
    }
