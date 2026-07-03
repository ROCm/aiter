# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight, opt-in timing for distributed initialization.

Enable with ``AITER_INIT_TIMING=1``. When disabled, ``timed()`` is a no-op
context manager with effectively zero overhead, so it is safe to leave the
instrumentation in production code paths.

Usage:

    from aiter.dist.init_timing import timed, report_init_timing

    with timed("initialize_model_parallel"):
        ...
        with timed("new_group:tp"):   # nesting is supported
            ...

    report_init_timing()  # rank-0 prints a hierarchical breakdown

The report shows, for each named region, its wall time and how it nests inside
parent regions, so you can see exactly where init time goes (e.g. which
``new_group`` / ``in_the_same_node`` / device-communicator step dominates).
"""

import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

ENABLED = os.environ.get("AITER_INIT_TIMING", "0") not in ("0", "", "false", "False")

# Each event: (name, depth, duration_seconds). Recorded in finish order.
_events: List[Tuple[str, int, float]] = []
_depth = 0


@contextmanager
def timed(name: str):
    """Time a named (optionally nested) region. No-op unless AITER_INIT_TIMING is set."""
    if not ENABLED:
        yield
        return
    global _depth
    depth = _depth
    _depth += 1
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        _depth = depth
        _events.append((name, depth, dur))


def reset_init_timing() -> None:
    """Clear recorded events (e.g. before a fresh init in the same process)."""
    global _depth
    _events.clear()
    _depth = 0


def _local_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def report_init_timing(only_rank: Optional[int] = 0) -> None:
    """Print a hierarchical timing report. By default only rank 0 prints.

    Pass ``only_rank=None`` to make every rank print its own report.
    """
    if not ENABLED or not _events:
        return
    rank = _local_rank()
    if only_rank is not None and rank != only_rank:
        return

    # Events were appended in finish order; a parent finishes after its
    # children, so reconstruct source order by stable-sorting is not trivial.
    # Instead we re-emit in recorded order, which (because we record on exit)
    # lists children before their parent. Reverse within equal-depth runs to
    # restore a readable top-down view by walking and grouping.
    lines = ["", f"[aiter] ===== init timing (rank {rank}) ====="]
    # Build a readable top-down tree by replaying the depth structure.
    # Since children are recorded before parents, walk the list and attach.
    for name, depth, dur in _reorder_topdown(_events):
        indent = "  " * depth
        lines.append(f"[aiter]   {indent}{name:<40s} {dur * 1e3:9.2f} ms")
    lines.append("[aiter] =======================================")

    from aiter import logger

    logger.info("\n".join(lines))


def _reorder_topdown(
    events: List[Tuple[str, int, float]],
) -> List[Tuple[str, int, float]]:
    """Reorder finish-order events into top-down (parent-before-children) order.

    A region's children are all consecutive entries that precede it with a
    strictly greater depth. We recursively place each parent ahead of the
    block of children it contains.
    """

    def emit(lo: int, hi: int, out: List[Tuple[str, int, float]]):
        # Collect the top-level regions in [lo, hi) (highest depth = children,
        # which finish first) in finish order, then reverse so siblings read
        # top-down in source order.
        regions = []  # (parent_index, child_lo)
        i = hi - 1
        while i >= lo:
            depth = events[i][1]
            j = i - 1
            while j >= lo and events[j][1] > depth:
                j -= 1
            regions.append((i, j + 1))
            i = j
        for parent_idx, child_lo in reversed(regions):
            name, depth, dur = events[parent_idx]
            out.append((name, depth, dur))
            emit(child_lo, parent_idx, out)

    out: List[Tuple[str, int, float]] = []
    emit(0, len(events), out)
    return out
