# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node symmetric memory: one arena per rank, peer-addressable by all."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .hip_ipc import (
    ipc_get_mem_handle,
    ipc_open_mem_handle,
    mem_allocation_base,
)

logger = logging.getLogger("aiter")

__all__ = ["PeerArenaGroup", "SymmetricArena", "SymmetricSlice"]

_DEFAULT_ALIGN = 256

_OPEN_HANDLES: dict[bytes, int] = {}


def _open_once(handle: bytes) -> int:
    cached = _OPEN_HANDLES.get(handle)
    if cached is not None:
        return cached
    peer = ipc_open_mem_handle(handle)
    _OPEN_HANDLES[handle] = peer
    return peer


class PeerArenaGroup:
    """Every rank of a TP group driven by this one process (no IPC).

    Each rank's arena is ordinary device memory on its own GPU; peers reach it
    through peer access, which the constructor enables between every pair.
    Pass the group as ``group=`` together with the rank."""

    def __init__(self, devices):
        import ctypes

        self.devices = [torch.device(d) for d in devices]
        self.world_size = len(self.devices)
        self._base: dict[int, int] = {}
        hip = ctypes.CDLL("libamdhip64.so")
        prev = torch.cuda.current_device()
        for d in self.devices:
            torch.cuda.set_device(d)
            for peer in self.devices:
                if peer != d:
                    err = hip.hipDeviceEnablePeerAccess(peer.index, 0)
                    if err not in (0, 704):  # 704: already enabled
                        raise RuntimeError(f"hipDeviceEnablePeerAccess({d} -> {peer}) = {err}")
                    if err:
                        hip.hipGetLastError()
        torch.cuda.set_device(prev)

    def rank_of(self, device: torch.device) -> int:
        return self.devices.index(torch.device(device))

    def register(self, rank: int, base_ptr: int) -> None:
        self._base[rank] = base_ptr

    def base_ptrs(self) -> tuple[int, ...]:
        if len(self._base) != self.world_size:
            raise RuntimeError(
                f"{len(self._base)} of {self.world_size} ranks committed their arena"
            )
        return tuple(self._base[r] for r in range(self.world_size))


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
    peer_ptr_table: torch.Tensor = field(default=None, repr=False)

    def peer_ptr(self, rank: int, byte_offset: int = 0) -> int:
        return self.peer_ptrs[rank] + byte_offset


class SymmetricArena:
    """A same-size-on-every-rank device arena, mapped into all peers."""

    def __init__(
        self,
        *,
        group=None,
        device: torch.device | None = None,
        align: int = _DEFAULT_ALIGN,
    ):
        self.group = group
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        if isinstance(group, PeerArenaGroup):
            self.rank = group.rank_of(self.device)
            self.world_size = group.world_size
        else:
            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)
        self.align = int(align)
        self._slices: dict[str, SymmetricSlice] = {}
        self._cursor = 0
        self._storage: torch.Tensor | None = None
        self._base_ptrs: tuple[int, ...] = ()
        self._committed = False

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

    def commit(self) -> SymmetricArena:
        """Allocate, zero, export and map the arena on every rank."""
        if self._committed:
            return self
        if self._cursor == 0:
            raise RuntimeError("nothing reserved")
        total = (self._cursor + self.align - 1) // self.align * self.align
        self._storage = torch.zeros(total, dtype=torch.uint8, device=self.device)
        base_ptr = int(self._storage.data_ptr())
        if isinstance(self.group, PeerArenaGroup):
            return self._commit_peer_group(base_ptr)
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
        dist.barrier(group=self.group)
        return self

    def _commit_peer_group(self, base_ptr: int) -> SymmetricArena:
        self.group.register(self.rank, base_ptr)
        for entry in self._slices.values():
            end = entry.offset + entry.nbytes
            entry.local = (
                self._storage[entry.offset : end].view(entry.dtype).view(entry.shape)
            )
        self._committed = True
        return self

    @property
    def base_ptrs(self) -> tuple[int, ...]:
        if isinstance(self.group, PeerArenaGroup):
            return self.group.base_ptrs()
        return self._base_ptrs

    def close(self) -> None:
        """Unmap the peer arenas. The local storage is released with the arena."""
        self._slices.clear()
        self._storage = None
        self._base_ptrs = ()
        self._committed = False
