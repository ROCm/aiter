# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounce buffers for the xGMI relay variants of the mesh quick-allreduce.

- **TP2 (`K = 1..3` relays per direction):** each receiving rank allocates its
  ``K`` incoming bounces on ``relay_ins`` and exchanges their IPC handles with
  its peer, which writes them as ``out_ptrs``.
- **TP4 (1 sender-owned relay per rank):** each sending rank allocates its
  single outgoing bounce on ``relay_devices[rank]`` and exports the handle to
  its three peers, keeping HIP context overhead to one lending GPU per rank.

Nothing runs on any relay GPU itself. By the time a rank's last kernel
finishes, its peers have already retired every write into its bounce, so
teardown needs no extra barrier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .ipc_heap import UncachedIpcHeap


@dataclass
class RelayBounce:
    """One rank's view of its relay bounce buffer(s)."""

    out_ptrs: tuple[int, ...]  # device addresses this rank writes
    in_ptrs: tuple[int, ...]  # device addresses this rank reads
    nbytes: int
    close: Callable[[], None]

    @property
    def out_ptr(self) -> int:
        return self.out_ptrs[0]

    @property
    def in_ptr(self) -> int:
        return self.in_ptrs[0]


def _release(heap, owns, opened) -> None:
    try:
        for ptr in opened:
            if ptr is not None:
                try:
                    heap.close_mem_handle(ptr)
                except RuntimeError:
                    pass
    finally:
        for ptr in owns:
            if ptr is not None:
                try:
                    heap.free_device_mem(ptr)
                except RuntimeError:
                    pass


def open_relay_bounce(
    *,
    group,
    rank: int,
    device: int,
    relay_outs: tuple[int, ...] | int,
    relay_ins: tuple[int, ...] | int,
    nbytes: int,
) -> RelayBounce:
    """Allocate this rank's ``K`` incoming bounces on ``relay_ins`` and
    exchange handles over *group* (TP2). Collective: raises on every rank or on
    none."""
    relay_outs = (
        (int(relay_outs),)
        if isinstance(relay_outs, int)
        else tuple(int(d) for d in relay_outs)
    )
    relay_ins = (
        (int(relay_ins),)
        if isinstance(relay_ins, int)
        else tuple(int(d) for d in relay_ins)
    )
    if not relay_outs or len(relay_outs) != len(relay_ins):
        raise ValueError(
            "relay_outs and relay_ins must be non-empty and equal length, got "
            f"{relay_outs} vs {relay_ins}"
        )
    k = len(relay_outs)

    heap = UncachedIpcHeap
    err = None
    owns: list[int] = []
    handles: list[bytes] = []
    try:
        for r_in, r_out in zip(relay_ins, relay_outs):
            heap.enable_peer_access(device, r_in)
            heap.enable_peer_access(device, r_out)
            with torch.cuda.device(r_in):
                ptr = heap.alloc_uncached(nbytes)
            owns.append(ptr)
            handles.append(heap.get_mem_handle_bytes(ptr))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    # Both ranks reach this exchange before either raises. A peer-access or
    # allocation failure on one rank would otherwise leave the other waiting.
    metas = heap.gather_object_list_via_broadcast(group, (err, handles, [0] * k))
    failures = heap.rank_failures([item[0] for item in metas])
    if failures:
        _release(heap, owns, ())
        raise RuntimeError(failures)

    opened_bases: list[int] = []
    offs: list[int] = []
    err = None
    try:
        peer_handles, offs = metas[1 - rank][1], metas[1 - rank][2]
        with torch.cuda.device(device):
            for ph in peer_handles:
                opened_bases.append(int(heap.open_mem_handle(bytes(ph))))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    open_failures = heap.rank_failures(
        heap.gather_object_list_via_broadcast(group, err)
    )
    if open_failures:
        _release(heap, owns, opened_bases)
        raise RuntimeError(open_failures)

    out_ptrs = tuple(base + int(off) for base, off in zip(opened_bases, offs))
    in_ptrs = tuple(int(p) for p in owns)
    return RelayBounce(
        out_ptrs=out_ptrs,
        in_ptrs=in_ptrs,
        nbytes=int(nbytes),
        close=lambda: _release(heap, owns, opened_bases),
    )


def open_sender_relay_bounces(
    *,
    group,
    rank: int,
    world_size: int,
    device: int,
    relay_devices: tuple[int, ...],
    nbytes: int,
) -> RelayBounce:
    """Allocate this rank's outgoing bounce on ``relay_devices[rank]`` and
    exchange handles over *group* (TP4 sender-owned).

    Each rank creates a HIP context on only one relay GPU (its own sender-owned
    relay), exports its IPC handle, and opens the ``world_size - 1`` peer
    handles on its own ``device``. Collective: raises on every rank or on none.
    """
    heap = UncachedIpcHeap
    relay_out = int(relay_devices[rank])
    err = None
    own = None
    handle = None
    try:
        for r_dev in relay_devices:
            heap.enable_peer_access(device, int(r_dev))
        with torch.cuda.device(relay_out):
            own = heap.alloc_uncached(nbytes)
        handle = heap.get_mem_handle_bytes(own)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    metas = heap.gather_object_list_via_broadcast(group, (err, handle, 0))
    failures = heap.rank_failures([item[0] for item in metas])
    if failures:
        _release(heap, (own,), ())
        raise RuntimeError(failures)

    opened_bases: list[int] = []
    in_ptrs: list[int] = []
    err = None
    try:
        with torch.cuda.device(device):
            for src in range(world_size):
                if src == rank:
                    continue
                peer_handle, off = metas[src][1], metas[src][2]
                base = int(heap.open_mem_handle(bytes(peer_handle)))
                opened_bases.append(base)
                in_ptrs.append(base + int(off))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    open_failures = heap.rank_failures(
        heap.gather_object_list_via_broadcast(group, err)
    )
    if open_failures:
        _release(heap, (own,), opened_bases)
        raise RuntimeError(open_failures)

    return RelayBounce(
        out_ptrs=(int(own),),
        in_ptrs=tuple(in_ptrs),
        nbytes=int(nbytes),
        close=lambda: _release(heap, (own,), opened_bases),
    )
