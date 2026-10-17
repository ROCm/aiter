# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Symmetric expert-weight storage for MoonEP.

Every rank collectively allocates one byte buffer containing each weight part
as a contiguous ``[experts_per_rank + prefetch_slots]`` window::

    local[i][:experts_per_rank]   home experts of part i
    local[i][experts_per_rank:]   local prefetch slots

Mori supplies the P2P address of the same allocation on every peer. The
prefetch kernel reads a selected expert from its owner's home rows and copies
it into this rank's slot rows. Slots are not read remotely, but keeping them
next to the home rows preserves MegaMoE's contiguous expert-window ABI.

The pool is dtype-transparent: shuffle output and prefetch both write bytes, so
it holds exactly the fp4/e8m0 layout the experts kernel consumes.
"""

from __future__ import annotations

import math

import flydsl.expr as fx
import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


class MoonEPWeightPool:
    """One layer's symmetric ``[home | prefetch]`` expert windows."""

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
    ) -> None:
        """Create one collective Mori allocation.

        ``parts`` lists each tensor's per-expert ``(shape, dtype)``. ``rank``
        and ``world_size`` must describe the already-initialized Mori shmem
        group.
        """
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
        if experts_per_rank <= 0 or prefetch_slots < 0:
            raise ValueError(
                "experts_per_rank must be positive and prefetch_slots non-negative"
            )

        row_bytes = tuple(
            math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
            for shape, dtype in parts
        )
        if not row_bytes or any(b <= 0 or b % 16 for b in row_bytes):
            raise ValueError(
                f"expert rows must be positive 16-byte multiples, got {row_bytes}"
            )
        window = experts_per_rank + prefetch_slots
        if any(window * b > 1 << 32 for b in row_bytes):
            raise ValueError(
                f"a {window}-expert window of {max(row_bytes)}-byte rows exceeds "
                "4 GiB; use more ranks or fewer prefetch slots"
            )

        offsets, used = [], 0
        for row_bytes_i in row_bytes:
            offsets.append(used)
            used += window * row_bytes_i

        self.rank = rank
        self.world_size = world_size
        self.experts_per_rank = experts_per_rank
        self.prefetch_slots = prefetch_slots
        self.parts = [(tuple(shape), dtype) for shape, dtype in parts]
        self.part_row_bytes = row_bytes
        self.part_offsets = tuple(offsets)
        self.storage_bytes = used
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._staged = False
        self._closed = False

        # All ranks must create symmetric tensors in the same order and with
        # the same size. Only this rank's tensor is a torch object; peer_bases
        # supplies the corresponding address in every other rank.
        self._storage = mori_shmem_create_tensor((used,), torch.uint8)
        self.local = [
            self._part(i, dtype, shape) for i, (shape, dtype) in enumerate(parts)
        ]
        self.home = [part[:experts_per_rank] for part in self.local]
        self.prefetched = [part[experts_per_rank:] for part in self.local]
        self._storage.zero_()
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()

        self._peer_bases = torch.empty(
            (world_size,), dtype=torch.int64, device=self.device
        )
        for pe in range(world_size):
            self._peer_bases[pe] = ms.shmem_ptr_p2p(self._storage.data_ptr(), rank, pe)
        torch.cuda.synchronize(self.device)

        self._jit = make_moonep_weight_prefetch_fast_jit(
            experts_per_rank=experts_per_rank,
            prefetch_slots=prefetch_slots,
            part_row_bytes=row_bytes,
            part_offsets=self.part_offsets,
            block_num=block_num,
            block_threads=block_threads,
            track_resident=True,
        )
        # All -1: no slot is known to hold anything, so every live slot copies.
        self._no_resident = torch.full(
            (prefetch_slots,), -1, dtype=torch.int32, device=self.device
        )

    def _part(
        self, index: int, dtype: torch.dtype, row_shape: tuple[int, ...]
    ) -> torch.Tensor:
        row_bytes = self.part_row_bytes[index]
        want = math.prod(row_shape) * torch.empty(0, dtype=dtype).element_size()
        if want != row_bytes:
            raise ValueError(
                f"row_shape {row_shape} of {dtype} spans {want} B but part "
                f"{index} rows hold {row_bytes} B"
            )
        window = self.experts_per_rank + self.prefetch_slots
        start = self.part_offsets[index]
        raw = self._storage[start : start + window * row_bytes]
        return raw.view(dtype).view(window, *row_shape)

    def finish_home(self) -> None:
        """Publish home rows written directly by caller-owned shuffle outputs."""
        if self._closed:
            raise RuntimeError("weight pool is closed")
        if self._staged:
            raise RuntimeError("weight pool home rows were already published")
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        self._staged = True

    def stage_home(self, weights: list[torch.Tensor]) -> None:
        """Compatibility path: copy ordinary tensors into this rank's home rows."""
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
            self.home[i].view(torch.uint8).copy_(weight.contiguous().view(torch.uint8))
        self.finish_home()

    def prefetch(
        self,
        experts_to_copy_row: torch.Tensor,
        resident: torch.Tensor | None = None,
    ) -> None:
        """Pull every part of selected remote experts into the prefetch slots."""
        if self._closed:
            raise RuntimeError("weight pool is closed")
        if not self._staged:
            raise RuntimeError(
                "finish_home() must run on every rank before prefetch: peers "
                "read the home rows directly"
            )
        if experts_to_copy_row.dtype != torch.int32:
            raise ValueError("experts_to_copy must be int32")
        stream = torch.cuda.current_stream(self.device)
        selected = experts_to_copy_row.contiguous()
        if resident is None:
            resident = self._no_resident
        elif resident.dtype != torch.int32 or resident.numel() != self.prefetch_slots:
            raise ValueError("resident must be int32 with one entry per slot")
        _run_compiled(
            self._jit,
            fx.Int64(selected.data_ptr()),
            fx.Int64(self._peer_bases.data_ptr()),
            fx.Int64(self._storage.data_ptr()),
            fx.Int64(resident.data_ptr()),
            fx.Stream(stream.cuda_stream),
        )

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        self.home = self.prefetched = self.local = None
        self._peer_bases = None
        self._storage = None
        self._closed = True


__all__ = ["MoonEPWeightPool"]
