# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounce buffers for the TP2 relay variant of the mesh quick-allreduce.

A relay GPU holds one bounce buffer per direction: the sending rank writes it,
the receiving rank allocates, owns and reads it, and nothing runs on the relay
GPU itself. By the time a rank's last kernel finishes, its peer has already
retired every write into its bounce, so teardown needs no extra barrier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .ipc_heap import UncachedIpcHeap


@dataclass
class RelayBounce:
    """One rank's view of a bounce pair."""

    out_ptr: int  # device address this rank writes, on the relay it sends through
    in_ptr: int  # device address this rank reads, on the relay it receives through
    nbytes: int
    close: Callable[[], None]


def _release(heap, own, opened) -> None:
    try:
        if opened is not None:
            heap.close_mem_handle(opened)
    finally:
        if own is not None:
            heap.free_device_mem(own)


def open_relay_bounce(
    *,
    group,
    rank: int,
    device: int,
    relay_out: int,
    relay_in: int,
    nbytes: int,
) -> RelayBounce:
    """Allocate this rank's incoming bounce on ``relay_in`` and exchange handles
    over *group*. Collective: raises on every rank or on none."""
    heap = UncachedIpcHeap
    err = None
    own = None
    handle = None
    try:
        heap.enable_peer_access(device, relay_in)
        heap.enable_peer_access(device, relay_out)
        with torch.cuda.device(relay_in):
            own = heap.alloc_uncached(nbytes)
        handle = heap.get_mem_handle_bytes(own)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    # Both ranks reach this exchange before either raises. A peer-access or
    # allocation failure on one rank would otherwise leave the other waiting.
    metas = heap.gather_object_list_via_broadcast(group, (err, handle, 0))
    failures = heap.rank_failures([item[0] for item in metas])
    if failures:
        _release(heap, own, None)
        raise RuntimeError(failures)

    opened = None
    err = None
    try:
        peer_handle, off = metas[1 - rank][1], metas[1 - rank][2]
        with torch.cuda.device(device):
            opened = int(heap.open_mem_handle(bytes(peer_handle)))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    open_failures = heap.rank_failures(
        heap.gather_object_list_via_broadcast(group, err)
    )
    if open_failures:
        _release(heap, own, opened)
        raise RuntimeError(open_failures)

    return RelayBounce(
        out_ptr=opened + off,
        in_ptr=own,
        nbytes=int(nbytes),
        close=lambda: _release(heap, own, opened),
    )
