# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/peer_memory.py
# atom/mono/runtime/consensus.py
"""Uncached symmetric peer memory for DCP mailbox kernels."""

import torch
import torch.distributed as dist

from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """Map one uncached buffer from each rank and expose their addresses."""

    def __init__(
        self,
        nbytes: int,
        group,
        rank: int,
        npes: int,
        device: torch.device,
    ):
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.rank = rank
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = UncachedIpcHeap.open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:  # every rank must still reach the agreement
                failure = err
            if not _all_ok(failure is None, group):
                self.close()
                raise RuntimeError(
                    "peer memory handshake failed on "
                    f"{'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Close peer mappings and free the local buffer."""
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0


def _all_ok(ok: bool, group) -> bool:
    """Return true when every rank mapped all peer buffers."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())
