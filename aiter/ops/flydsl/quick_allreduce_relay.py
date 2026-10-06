# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounce buffers for the TP2 relay variant of the mesh quick-allreduce.

A relay GPU holds one bounce buffer per direction. The rank that sends writes
it, the rank that receives reads it, and the relay GPU's own kernels never touch
it. ``RelayProvider`` is where the buffers come from: the in-group provider
allocates them in the TP processes themselves, and the seam stays open for one
that opens buffers a third process owns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from .quick_allreduce_ipc import UncachedIpcHeap

# ``(first GPU, second GPU)`` of each xGMI pair on an 8-GPU node.
_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))


def tp2_relays(pair_index: int) -> tuple[int, int]:
    """``(relay for rank0→rank1, relay for rank1→rank0)`` of an ``_PAIRS`` pair.

    Each direction bounces through the same slot of the next pair, so no relay
    is a member of the pair and no GPU relays for two pairs in one direction.
    """
    if not 0 <= pair_index < len(_PAIRS):
        raise ValueError(f"pair_index must be in [0, {len(_PAIRS)}), got {pair_index}")
    first, second = _PAIRS[(pair_index + 1) % len(_PAIRS)]
    return first, second


@dataclass
class RelayBounce:
    """One rank's view of a bounce pair."""

    out_ptr: int  # device address this rank writes, on the relay it sends through
    in_ptr: int  # device address this rank reads, on the relay it receives through
    nbytes: int
    close: Callable[[], None]


class RelayProvider(Protocol):
    def open(
        self,
        *,
        group,
        rank: int,
        device: int,
        relay_out: int,
        relay_in: int,
        nbytes: int,
    ) -> RelayBounce:
        """Collective over *group*; every rank calls it in the same order."""


class InGroupRelayProvider:
    """Each rank allocates its incoming bounce on ``relay_in`` and shares it.

    The receiver owns the memory, as it owns its inbox: by the time a rank's
    last kernel has finished, its peer has already retired every write into it,
    so teardown needs no extra barrier.
    """

    def open(
        self,
        *,
        group,
        rank: int,
        device: int,
        relay_out: int,
        relay_in: int,
        nbytes: int,
    ) -> RelayBounce:
        heap = UncachedIpcHeap
        heap.enable_peer_access(device, relay_in)
        heap.enable_peer_access(device, relay_out)
        with torch.cuda.device(relay_in):
            own = heap.alloc_uncached(nbytes)
        opened = None
        try:
            handle = heap.get_mem_handle_bytes(own)
            metas = heap.gather_object_list_via_broadcast(group, (handle, 0))
            peer_handle, off = metas[1 - rank]
            with torch.cuda.device(device):
                opened = int(heap.open_mem_handle(bytes(peer_handle)))
        except Exception:
            if opened is not None:
                heap.close_mem_handle(opened)
            heap.free_device_mem(own)
            raise

        def close():
            heap.close_mem_handle(opened)
            heap.free_device_mem(own)

        return RelayBounce(
            out_ptr=opened + off, in_ptr=own, nbytes=int(nbytes), close=close
        )
