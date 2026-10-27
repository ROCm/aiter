# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250-only launch geometry for Stage2-fused MegaMoE."""

_WAVE_SIZE = 32
_LANE_MASK = _WAVE_SIZE - 1
_LOG2_WAVE_SIZE = 5

# The FlyDSL dispatch uses TDM and has a tight LDS budget (one hidden-dim payload
# tile per warp). From a gfx1250 EP4 hidden-7168 geometry sweep; topk does not
# move the dispatch half.
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

# Compact (stage1_fused) dispatch: (bound, block, warp, route_parallel). One
# warp's remote TDM stores retire serially, so decode walks routes and wants
# about one warp per route (tokens * topk); prefill keeps the token-major walk,
# where one load feeds all topk stores. EP4 h7168 topk6 fp4 wire, dispatch us:
#   tpr   token-major 64x8   route 64x8   96x16   192x8   192x16   384x8
#   1           27.3            10.7
#   64          29.6            11.8
#   128                                           13.2
#   256         31.0            21.2      16.3    14.4
#   512         32.0            34.2                       18.6     18.8
#   1536                                                        384x16 route
_DISPATCH_COMPACT = (
    (64, 64, 8, True),
    (256, 192, 8, True),
    (512, 192, 16, True),
    # EP4, h7168, topk8, fp4 wire: 1536-token decode is 1.9% faster
    # end-to-end in the MegaMoE microbenchmark with this route-parallel
    # geometry than the generic 128x16 token-major fallback below.
    (1536, 384, 16, True),
    # 4096-token prefill: route-parallel 256x16 lowers dispatch from 159 us
    # (128x16 token-major) to 130 us on EP4/h7168/topk8/fp4.
    (4096, 256, 16, True),
    (None, 128, 16, False),
)

# EP4 h7168 topk9 (topk8 + the fused shared expert) fp4 wire. Route-parallel
# wins at every size here, prefill included; the topk8 table's 384-block decode
# geometry loses to 1024x8. Dispatch us, MegaMoE microbenchmark (eager):
#   tpr    old table            this table
#   64     22.7 (64x8)          15.9 (256x8)
#   256    24.0 (192x8)         17.8 (256x16)
#   1024   37.9 (384x16)        31.9 (1024x8)
#   1536   47.8 (384x16)        38.5 (1024x8)
#   8192   171.7 (128x16 tok)   100.3 (512x16)
#   16384  314 (128x16 tok)     182 (512x16)
_DISPATCH_COMPACT_EP4_TOPK9 = (
    (64, 256, 8, True),
    (256, 256, 16, True),
    (2048, 1024, 8, True),
    (4096, 256, 16, True),
    (None, 512, 16, True),
)


def _compact_dispatch_schedule(world_size: int, topk: int):
    if (int(world_size), int(topk)) == (4, 9):
        return _DISPATCH_COMPACT_EP4_TOPK9
    return _DISPATCH_COMPACT

_DISPATCH_TDM_SCHEDULES = {
    (4, 7168, 8): _DISPATCH_EP4_TDM,
    (4, 7168, 6): _DISPATCH_EP4_TDM,
    (8, 7168, 8): _DISPATCH_EP8_TDM,
    (8, 7168, 6): _DISPATCH_EP8_TDM,
}


def _select_dispatch_config(
    world_size: int, hidden_dim: int, topk: int
) -> dict[str, object]:
    schedule = _DISPATCH_TDM_SCHEDULES.get((world_size, hidden_dim, topk))
    if schedule is None:
        schedule = _DISPATCH_EP8_TDM if world_size == 8 else _DISPATCH_EP4_TDM
    _, block, warp = schedule[-1]
    return {
        "dispatch_block_num": block,
        "dispatch_warp_num_per_block": warp,
        "schedule": schedule,
    }
