# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of fused split-K: per-stream arrival counters and the partials.

The kernels use _tile_on_xcd and _sum_splits_on_xcd from
_triton_kernels/common/splitk_reduce.py.
"""

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.utility.graph_alloc import (
    ROUTES_INSIDE_CAPTURE,
    capture_scoped_cache,
    persistent_alloc,
)

# One arrival counter per output tile of a fused split-K launch.
SPLIT_COUNTERS = 1 << 16


@capture_scoped_cache
def _split_counters_keyed(device: torch.device, stream_id: int) -> torch.Tensor:
    with persistent_alloc(device):
        return torch.zeros(SPLIT_COUNTERS, dtype=torch.int32, device=device)


def split_counters(device: torch.device) -> torch.Tensor | None:
    """Zeroed arrival counters per (device, stream), as for the a16w16 split-K
    semaphores; kernels re-zero what they use. None when a stream's first use
    falls inside a capture that cannot allocate outside the graph pool."""
    stream_id = torch.cuda.current_stream(device).cuda_stream
    if not ROUTES_INSIDE_CAPTURE and torch.cuda.is_current_stream_capturing():
        return _split_counters_keyed.eager_cached(device, stream_id)
    return _split_counters_keyed(device, stream_id)


def fused_splitk_workspace(y, tiles, splits, block_m, block_n):
    """(grid, partials, counters) for fused split-K, or None to run unfused.

    The grid puts all splits of a tile on one XCD, which gfx950 dispatch
    guarantees. On other archs it returns None.
    """
    assert tiles <= SPLIT_COUNTERS, "Too many output tiles for fused split-K"
    if get_arch() != "gfx950":
        return None
    counters = split_counters(y.device)
    if counters is None:
        return None
    partials = torch.empty(
        tiles * splits * block_m * block_n, dtype=torch.float32, device=y.device
    )
    return (-(-tiles // 8) * 8 * splits,), partials, counters
