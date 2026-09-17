# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node symmetric memory: one arena per rank, peer-addressable by all.

Every rank allocates an arena of identical size and exports it over HIP IPC, so
rank ``r`` ends up holding a device pointer to *every* rank's copy.  A kernel
can then read or write a peer's slice directly instead of going through a
collective, which is what makes fusing the collective into a GEMM epilogue
possible.

This is the piece ``torch.distributed._symmetric_memory`` would normally
provide.  Its ROCm rendezvous fails on this stack (``HIP error: invalid
argument`` from ``_SymmetricMemory.rendezvous``), and mori is not installed, so
the mapping is built directly on the HIP runtime's own ``hipIpc*`` calls, via
the ctypes wrapper in :mod:`.hip_ipc`.

Usage::

    arena = SymmetricArena(group=tp_group, device=dev)
    xq = arena.reserve("xq", (M, H // 2), torch.uint8)
    arena.commit()                       # one IPC exchange for the whole arena
    xq.local[...]                        # this rank's view, an ordinary tensor
    xq.peer_ptrs[p]                      # device address of rank p's copy

``reserve`` must be called in the same order on every rank -- offsets are
assigned deterministically from the reservation order, and peers address each
other's slices by offset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .hip_ipc import (
    ipc_close_mem_handle,
    ipc_get_mem_handle,
    ipc_open_mem_handle,
    mem_allocation_base,
)

logger = logging.getLogger("aiter")

__all__ = ["SymmetricArena", "SymmetricSlice"]

_DEFAULT_ALIGN = 256

# hipIpcOpenMemHandle refuses a handle that is already mapped in this process,
# and two arenas can easily be sub-allocations of one caching-allocator segment
# (which is what hipIpcGetMemHandle actually exports). Map each distinct
# allocation once and hand out offsets from it.
_OPEN_HANDLES: dict[bytes, int] = {}


def _open_once(handle: bytes) -> int:
    cached = _OPEN_HANDLES.get(handle)
    if cached is not None:
        return cached
    peer = ipc_open_mem_handle(handle)
    _OPEN_HANDLES[handle] = peer
    return peer


@dataclass
class SymmetricSlice:
    """One named region of the arena, with every rank's address for it."""

    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    local: torch.Tensor = field(default=None, repr=False)
    peer_ptrs: tuple[int, ...] = ()
    #: int64 device tensor of ``peer_ptrs``, for kernels that index by rank.
    peer_ptr_table: torch.Tensor = field(default=None, repr=False)

    def peer_ptr(self, rank: int, byte_offset: int = 0) -> int:
        return self.peer_ptrs[rank] + byte_offset


class SymmetricArena:
    """A same-size-on-every-rank device arena, mapped into all peers.

    The arena is one allocation, so the whole layout costs a single IPC
    exchange no matter how many slices it holds.
    """

    def __init__(
        self,
        *,
        group=None,
        device: torch.device | None = None,
        align: int = _DEFAULT_ALIGN,
    ):
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.align = int(align)
        self._slices: dict[str, SymmetricSlice] = {}
        self._cursor = 0
        self._storage: torch.Tensor | None = None
        self._base_ptrs: tuple[int, ...] = ()
        self._committed = False

    # -- layout -------------------------------------------------------------
    def reserve(
        self, name: str, shape, dtype: torch.dtype, *, align: int | None = None
    ) -> SymmetricSlice:
        """Carve out a named region. Must run in the same order on every rank."""
        if self._committed:
            raise RuntimeError("cannot reserve after commit()")
        if name in self._slices:
            raise KeyError(f"symmetric slice {name!r} already reserved")
        shape = tuple(int(s) for s in shape)
        step = align or self.align
        offset = (self._cursor + step - 1) // step * step
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        self._cursor = offset + nbytes
        entry = SymmetricSlice(
            name=name, offset=offset, nbytes=nbytes, shape=shape, dtype=dtype
        )
        self._slices[name] = entry
        return entry

    @property
    def nbytes(self) -> int:
        return self._cursor

    def commit(self) -> "SymmetricArena":
        """Allocate, zero, export and map the arena on every rank."""
        if self._committed:
            return self
        if self._cursor == 0:
            raise RuntimeError("nothing reserved")
        total = (self._cursor + self.align - 1) // self.align * self.align
        # Zeroed so flag and counter slices start in a known state; the payload
        # regions are overwritten before they are read.
        self._storage = torch.zeros(total, dtype=torch.uint8, device=self.device)
        base_ptr = int(self._storage.data_ptr())
        # hipIpc* act on the current device, which a caller need not have set to
        # this arena's, so every HIP call here is made under an explicit guard.
        with torch.cuda.device(self.device):
            alloc_base = mem_allocation_base(base_ptr)
            handle = ipc_get_mem_handle(base_ptr)

        torch.cuda.synchronize(self.device)
        payload = (handle, base_ptr - alloc_base, total)
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, payload, group=self.group)

        base_ptrs = []
        with torch.cuda.device(self.device):
            for peer_rank, entry in enumerate(gathered):
                peer_handle, peer_offset, peer_total = entry
                if peer_total != total:
                    raise RuntimeError(
                        f"symmetric arena size disagrees: rank {self.rank} has "
                        f"{total} bytes, rank {peer_rank} has {peer_total}. Every "
                        "rank must reserve the same layout."
                    )
                if peer_rank == self.rank:
                    base_ptrs.append(base_ptr)
                else:
                    base_ptrs.append(_open_once(peer_handle) + peer_offset)
        self._base_ptrs = tuple(base_ptrs)

        for entry in self._slices.values():
            end = entry.offset + entry.nbytes
            entry.local = (
                self._storage[entry.offset : end].view(entry.dtype).view(entry.shape)
            )
            entry.peer_ptrs = tuple(p + entry.offset for p in self._base_ptrs)
            entry.peer_ptr_table = torch.tensor(
                entry.peer_ptrs, dtype=torch.int64, device=self.device
            )
        self._committed = True
        # Nobody may touch a peer slice until every rank has finished mapping.
        dist.barrier(group=self.group)
        return self

    # -- access -------------------------------------------------------------
    def __getitem__(self, name: str) -> SymmetricSlice:
        if not self._committed:
            raise RuntimeError("commit() the arena before using its slices")
        return self._slices[name]

    def __contains__(self, name: str) -> bool:
        return name in self._slices

    @property
    def base_ptrs(self) -> tuple[int, ...]:
        return self._base_ptrs

    def close(self) -> None:
        """Unmap the peer arenas. The local storage is released with the arena."""
        self._slices.clear()
        self._storage = None
        self._base_ptrs = ()
        self._committed = False


def _close_all_open_handles() -> None:
    """Release every peer mapping this process holds. Test/teardown helper."""
    for peer in _OPEN_HANDLES.values():
        try:
            ipc_close_mem_handle(peer)
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            logger.warning("[symmetric_arena] ipc_close_mem_handle failed: %s", exc)
    _OPEN_HANDLES.clear()
