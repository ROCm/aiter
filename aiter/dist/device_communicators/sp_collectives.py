# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""INT4 sequence-parallel reduce-scatter and all-gather, chosen by size.

A sequence-parallel layer reduce-scatters a bf16 ``[T, H]`` activation over a
tensor-parallel group and all-gathers it back, and the caller has an exact
collective for both (an IPC kernel, RCCL). ``SequenceParallelCollectives``
runs the INT4 single-phase kernels of ``FlyQuickAllReduce`` instead where the
payload is inside the window of ``allreduce_policy.SP_INT4_WINDOWS`` for this
platform, and returns ``None`` everywhere else, which means "not served here:
continue with the exact collective". The window counts the bytes of the full
tensor (the reduce-scatter input, the all-gather output), not the padding. A
padded tensor of 4 GiB or more, the limit of the kernels, is not served either.
The router owns what the kernels need of the shape: each rank's block is padded
with zero rows to a whole number of tiles, and the result is returned unpadded.

``create_sp_collectives`` is a collective: call it on every rank, outside CUDA
graph capture, with the same ``FlyQuickAllReduce``. It returns ``None`` when
FlyDSL is unavailable, a rank has no window for its platform, the windows
differ, or the INT4 kernels cannot run, and every rank then gets ``None``. After that, whether a
call is served depends only on its shape, its dtype, the window and the
platform, never on a rank's data or addresses, so the ranks of a group always
agree: a rank that took the exact path while a peer ran INT4 would leave the
peer polling for it forever. The call rules of ``FlyQuickAllReduce`` apply to
every served call: one serialized device order for all launches on the
communicator, and the same sequence of calls and sizes on every rank.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist

logger = logging.getLogger("aiter")

# flydsl is an optional dependency whose version is checked at import time.
try:
    from aiter.ops.flydsl import allreduce_policy as fly_policy
    from aiter.ops.flydsl.allreduce_policy import (
        MAX_PAYLOAD_BYTES,
        PAYLOAD_ALIGN_BYTES,
    )
    from aiter.ops.flydsl.kernels.comm.allreduce_shared import TILE_BYTES
except Exception:  # noqa: BLE001 -- flydsl is absent or too old: nothing is served
    fly_policy = None

ELEMENT_BYTES = torch.bfloat16.itemsize


def block_row_unit(world_size: int, hidden: int) -> int:
    """Rows a rank's block must be a multiple of for the full tensor to be a
    whole number of tiles: the block's bytes must be a multiple of
    ``TILE_BYTES / world_size``."""
    block_bytes = TILE_BYTES // world_size
    return block_bytes // math.gcd(block_bytes, ELEMENT_BYTES * hidden)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """*t* itself when the kernel can read it, else a contiguous aligned copy."""
    if t.is_contiguous() and t.data_ptr() % PAYLOAD_ALIGN_BYTES == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


class SequenceParallelCollectives:
    """INT4 reduce-scatter / all-gather over a ``FlyQuickAllReduce``, served
    inside a payload window and ``None`` outside it. Build it with
    ``create_sp_collectives``."""

    def __init__(self, quick, window):
        self._quick = quick
        self._window = window
        self.world_size = quick.world_size

    def _padded_rows(self, rows: int, hidden: int) -> int | None:
        """The rows of a rank's block after padding, or None when the call is not
        served: the bytes of the full tensor are outside the window (padding does
        not count, or a block of a few rows would be served because its padding
        fills whole tiles), or the padded tensor is 4 GiB or more, the limit of
        the kernels."""
        unit = block_row_unit(self.world_size, hidden)
        padded = -(-rows // unit) * unit
        sent = self.world_size * rows * hidden * ELEMENT_BYTES
        moved = self.world_size * padded * hidden * ELEMENT_BYTES
        if self._window.contains(sent) and moved <= MAX_PAYLOAD_BYTES:
            return padded
        return None

    def reduce_scatter(self, inp: torch.Tensor) -> torch.Tensor | None:
        """The block of the element-wise sum over ranks that this rank owns,
        ``[T / world_size, H]``, from this rank's bf16 ``[T, H]`` partial; the
        rows ``[rank * T/W, (rank + 1) * T/W)`` of the sum. ``None`` when not
        served: another dtype or rank, ``T`` not divisible by the group size,
        a payload outside the window, or a padded tensor of 4 GiB or more."""
        w = self.world_size
        if inp.dtype != torch.bfloat16 or inp.dim() != 2 or inp.shape[0] % w:
            return None
        rows, hidden = inp.shape[0] // w, inp.shape[1]
        padded = self._padded_rows(rows, hidden)
        if padded is None:
            return None
        if padded == rows:
            src = _aligned(inp)
        else:
            src = inp.new_zeros((w, padded, hidden))
            src[:, :rows] = inp.reshape(w, rows, hidden)
        out = inp.new_empty((padded, hidden))
        self._quick.reduce_scatter(src.view(w * padded, hidden), out)
        return out[:rows]

    def all_gather(self, inp: torch.Tensor) -> torch.Tensor | None:
        """The ``[world_size * rows, H]`` concatenation, in rank order, of every
        rank's bf16 ``[rows, H]`` block. ``None`` when not served: another
        dtype or rank, a payload outside the window, or a padded tensor of
        4 GiB or more."""
        w = self.world_size
        if inp.dtype != torch.bfloat16 or inp.dim() != 2:
            return None
        rows, hidden = inp.shape
        padded = self._padded_rows(rows, hidden)
        if padded is None:
            return None
        if padded == rows:
            src = _aligned(inp)
        else:
            src = inp.new_zeros((padded, hidden))
            src[:rows] = inp
        full = inp.new_empty((w * padded, hidden))
        self._quick.all_gather(src, full)
        if padded == rows:
            return full
        return full.view(w, padded, hidden)[:, :rows].reshape(w * rows, hidden)


def create_sp_collectives(quick) -> SequenceParallelCollectives | None:
    """The INT4 sequence-parallel collectives of *quick*, or None when FlyDSL is
    unavailable, a rank has no window for its platform, or the kernels cannot
    run here.

    A collective: every rank calls it, outside CUDA graph capture. The ranks
    first compare their windows, then compile and load the single-phase kernels
    and agree that they can run (``FlyQuickAllReduce.compile_sp``), so the first
    served call launches at once, and a failure on any rank gives every rank
    ``None``.
    """
    window = None
    if fly_policy is not None:
        window = fly_policy.sp_int4_window(
            arch=fly_policy.detect_arch(), link=quick.link, world_size=quick.world_size
        )
    # Every rank votes before any rank returns: one that returned alone would
    # leave its peers waiting in compile_sp's own vote.
    windows = [None] * quick.world_size
    dist.all_gather_object(windows, window, group=quick.group)
    if windows[0] is None or any(w != windows[0] for w in windows):
        return None
    elements = TILE_BYTES // ELEMENT_BYTES
    full = torch.empty(elements, dtype=torch.bfloat16, device=quick.device)
    shard = torch.empty(
        elements // quick.world_size, dtype=torch.bfloat16, device=quick.device
    )
    # compile_sp raises on every rank when any rank cannot run the kernels.
    try:
        quick.compile_sp(full, shard)
    except Exception as exc:  # noqa: BLE001
        logger.warning("INT4 sequence-parallel collectives are unavailable: %s", exc)
        return None
    return SequenceParallelCollectives(quick, window)
