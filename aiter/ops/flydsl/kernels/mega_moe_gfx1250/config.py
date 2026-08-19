# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250-only launch geometry for Stage2-fused MegaMoE."""

_WAVE_SIZE = 32
_LANE_MASK = _WAVE_SIZE - 1
_LOG2_WAVE_SIZE = 5

# Vector transport (legacy buffer_load/store dispatch).
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

# TDM transport schedules (from mori PR #578 / tuning_configs TDM sweep).
_DISPATCH_TDM_EP4 = (
    (512, 128, 8),
    (2048, 128, 8),
    (4096, 64, 16),
    (None, 128, 16),
)

_DISPATCH_TDM_EP4_TOPK6 = (
    (512, 128, 8),
    (2048, 128, 8),
    (4096, 64, 16),
    (None, 128, 16),
)

_DISPATCH_TDM_EP8 = (
    (512, 128, 8),
    (2048, 128, 8),
    (None, 128, 16),
)

_DISPATCH_SCHEDULES = {
    (4, 7168, 8): _DISPATCH_EP4,
    (4, 7168, 6): _DISPATCH_EP4_TOPK6,
    (8, 7168, 8): _DISPATCH_EP8,
    (8, 7168, 6): _DISPATCH_EP8,
}

_DISPATCH_TDM_SCHEDULES = {
    (4, 7168, 8): _DISPATCH_TDM_EP4,
    (4, 7168, 6): _DISPATCH_TDM_EP4_TOPK6,
    (8, 7168, 8): _DISPATCH_TDM_EP8,
    (8, 7168, 6): _DISPATCH_TDM_EP8,
}


def _select_dispatch_config(
    world_size: int,
    hidden_dim: int,
    topk: int,
    *,
    dispatch_transport: str = "vector",
) -> dict[str, object]:
    table = (
        _DISPATCH_TDM_SCHEDULES
        if dispatch_transport == "tdm"
        else _DISPATCH_SCHEDULES
    )
    schedule = table.get((world_size, hidden_dim, topk))
    if schedule is None:
        fallback = _DISPATCH_TDM_EP8 if dispatch_transport == "tdm" else _DISPATCH_EP8
        schedule = fallback if world_size == 8 else (
            _DISPATCH_TDM_EP4 if dispatch_transport == "tdm" else _DISPATCH_EP4
        )
    _, block, warp = schedule[-1]
    return {
        "dispatch_block_num": block,
        "dispatch_warp_num_per_block": warp,
        "schedule": schedule,
    }
