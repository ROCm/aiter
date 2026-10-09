# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Expert-weight pool for MoonEP: one layer's weights ("parts", e.g. w1, its
scale, w2 and its scale) for every rank in one virtual range.

Each part has a ``[epn + B]`` window per rank
(``aiter.ops.flydsl.moonep_vmm_pool.MoonEPVmmPool``)::

    local[i][:epn]   home experts of part i
    local[i][epn:]   its prefetch slots

``local[i]`` is the weight tensor of a ``moonep_slots=B`` MegaMoE instance and
``home[i]`` the owner-only one.  Other ranks' segments are *their* memory,
mapped here over XGMI, so one prefetch launch pulls every part of the selected
experts by offset arithmetic.  ATOM's weights are ordinary local tensors and are
not reachable by peers, so they must be staged into ``home`` once after loading.

The pool is dtype-transparent: staging and prefetch are byte copies, so it
holds the weights in whatever layout and dtype the experts kernel already
expects -- shuffled fp4 slabs and their block scales -- without unshuffling or
dequantising anything.
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch

from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)
from aiter.ops.flydsl.moonep_vmm_pool import MoonEPVmmPool


class MoonEPWeightPool:
    """One layer's ``[epn | B]`` part windows on every rank + prefetch launcher."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        experts_per_rank: int,
        prefetch_slots: int,
        parts: list[tuple[tuple[int, ...], torch.dtype]],
        block_num: int = 1024,
        block_threads: int = 256,
        group=None,
    ) -> None:
        """``parts`` lists each tensor's per-expert ``(shape, dtype)``;
        ``rank``/``world_size`` are positions in ``group``, the EP process group
        whose ranks map each other's segments (None: the default group)."""
        row_bytes = tuple(
            math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
            for shape, dtype in parts
        )
        if any(b % 16 for b in row_bytes):
            raise ValueError(f"expert rows must be 16-byte aligned, got {row_bytes}")

        self.rank = rank
        self.world_size = world_size
        self.experts_per_rank = experts_per_rank
        self.prefetch_slots = prefetch_slots
        self.parts = [(tuple(shape), dtype) for shape, dtype in parts]
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._staged = False
        self._closed = False

        self._vmm = MoonEPVmmPool(
            part_row_bytes=row_bytes,
            experts_per_rank=experts_per_rank,
            prefetch_slots=prefetch_slots,
            rank=rank,
            world_size=world_size,
            device=self.device,
            group=group,
        )
        self.local = [
            self._vmm.part(i, dtype, shape) for i, (shape, dtype) in enumerate(parts)
        ]
        self.home = [window[:experts_per_rank] for window in self.local]
        self.prefetched = [window[experts_per_rank:] for window in self.local]
        # Zero only our own segment -- the rest of the range is peer memory and
        # clearing it would wipe their weights. Through bytes: the narrow dtypes
        # this pool carries (fp4x2, e8m0) have no fill_ kernel in torch.
        self._vmm.local_segment.zero_()
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()

        self._jit = make_moonep_weight_prefetch_fast_jit(
            experts_per_rank=experts_per_rank,
            prefetch_slots=prefetch_slots,
            part_row_bytes=row_bytes,
            part_offsets=self._vmm.part_offsets,
            segment_bytes=self._vmm.segment_bytes,
            block_num=block_num,
            block_threads=block_threads,
            track_resident=True,
        )
        # All -1: no slot is known to hold anything, so every live slot copies.
        self._no_resident = torch.full(
            (prefetch_slots,), -1, dtype=torch.int32, device=self.device
        )
        self._compiled = None

    def stage_home(self, weights: list[torch.Tensor]) -> None:
        """Copy this rank's expert weights, one tensor per part, into its home rows."""
        if self._closed:
            raise RuntimeError("weight pool is closed")
        if len(weights) != len(self.parts):
            raise ValueError(f"expected {len(self.parts)} parts, got {len(weights)}")
        for i, (weight, (shape, dtype)) in enumerate(zip(weights, self.parts)):
            expected = (self.experts_per_rank, *shape)
            if tuple(weight.shape) != expected or weight.dtype != dtype:
                raise ValueError(
                    f"part {i}: expected {expected} {dtype}, got "
                    f"{tuple(weight.shape)} {weight.dtype}; the pool is a byte "
                    "copy and must match the experts kernel's layout exactly"
                )
            # Copy as bytes: fp4x2/e8m0 lack complete elementwise coverage in
            # torch, and the pool only ever needs the bytes anyway.
            self.home[i].view(torch.uint8).copy_(weight.contiguous().view(torch.uint8))
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        self._staged = True

    def prefetch(
        self,
        experts_to_copy_row: torch.Tensor,
        resident: torch.Tensor | None = None,
    ) -> None:
        """Pull every part of the selected remote experts into the prefetch slots.

        ``resident[slot]`` names the expert a slot already holds; such slots
        are skipped.  The caller owns updating it after the copy.
        """
        if self._closed:
            raise RuntimeError("weight pool is closed")
        if not self._staged:
            raise RuntimeError(
                "stage_home() must run on every rank before prefetch: peers "
                "read the home rows directly, so an unstaged rank serves "
                "zeros without any error"
            )
        if experts_to_copy_row.dtype != torch.int32:
            raise ValueError("experts_to_copy must be int32")
        stream = torch.cuda.current_stream(self.device)
        sel = experts_to_copy_row.contiguous()
        if resident is None:
            resident = self._no_resident
        elif resident.dtype != torch.int32 or resident.numel() != self.prefetch_slots:
            raise ValueError("resident must be int32 with one entry per slot")
        raw = (
            sel.data_ptr(),
            self._vmm.base,
            self._vmm.local_segment.data_ptr(),
            resident.data_ptr(),
            stream,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(
                self._jit, *(fx.Int64(a) for a in raw[:4]), stream
            )
        self._compiled(*raw)

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        # Drop the views before the mapping they borrow from.
        self.home = self.prefetched = self.local = None
        self._vmm.close()
        self._closed = True


__all__ = ["MoonEPWeightPool"]
