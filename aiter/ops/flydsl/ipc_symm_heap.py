# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Symmetric device memory for the FlyDSL intra-node communication ops.

``IpcSymmetricAllocator`` needs nothing beyond HIP and ``torch.distributed``:
every ``create_tensor`` call is collective over ``group`` (the ranks of one
node); each rank allocates the same number of bytes, exports a HIP IPC handle,
all-gathers the handles and opens every peer's allocation. ``p2p_table(t)``
returns an i64 device tensor holding every peer's mapping of ``t`` (the own
rank's entry is the local pointer), which is what the kernels index by PE.

Memory is uncached (fine-grained), the same mode as mori's shmem heap: the
kernels' cross-GPU stores and flag polls rely on it.

``MoriShmemAllocator`` keeps the previous behaviour (mori's symmetric heap)
behind the same interface for callers that share buffers with mori kernels.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .quick_allreduce_int4_ipc import UncachedIpcHeap

_ALIGN = 4096


class _CudaArray:
    """``__cuda_array_interface__`` view of a raw device pointer for torch."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "version": 3,
            "strides": None,
        }


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


class IpcSymmetricAllocator:
    """Collective, zero-filled, uncached allocations shared over HIP IPC."""

    def __init__(self, group, device: torch.device):
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._allocs: list[tuple[int, list[int], int]] = []

    def create_tensor(self, shape, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        numel = _numel(shape)
        elem = torch.tensor([], dtype=dtype).element_size()
        nbytes = max(1, numel * elem)
        nbytes = (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
        with torch.cuda.device(self.device):
            ptr = UncachedIpcHeap.alloc_uncached(nbytes)
            handle = UncachedIpcHeap.get_mem_handle_bytes(ptr)
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, (nbytes, handle), group=self.group)
            peers = []
            for r, (peer_nbytes, peer_handle) in enumerate(gathered):
                if peer_nbytes != nbytes:
                    raise RuntimeError(
                        f"symmetric allocation size mismatch: rank {r} asked for "
                        f"{peer_nbytes} bytes, rank {self.rank} for {nbytes}"
                    )
                peers.append(
                    ptr
                    if r == self.rank
                    else UncachedIpcHeap.open_mem_handle(peer_handle)
                )
            raw = torch.as_tensor(_CudaArray(ptr, nbytes), device=self.device)
        self._allocs.append((ptr, peers, nbytes))
        return raw[: numel * elem].view(dtype).view(*shape)

    def p2p_table(self, t: torch.Tensor) -> torch.Tensor:
        base = t.data_ptr()
        for ptr, peers, nbytes in self._allocs:
            if ptr <= base < ptr + nbytes:
                off = base - ptr
                return torch.tensor(
                    [p + off for p in peers], dtype=torch.int64, device=self.device
                )
        raise ValueError("tensor was not allocated by this IpcSymmetricAllocator")

    def barrier(self):
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def close(self):
        """Collective: unmap the peers' allocations, then free the own ones."""
        for _, peers, _ in self._allocs:
            for r, p in enumerate(peers):
                if r != self.rank:
                    UncachedIpcHeap.close_mem_handle(p)
        self.barrier()
        for ptr, _, _ in self._allocs:
            UncachedIpcHeap.free_device_mem(ptr)
        self._allocs.clear()


class MoriShmemAllocator:
    """mori's symmetric heap behind the ``IpcSymmetricAllocator`` interface."""

    def __init__(self, rank: int, world_size: int, device: torch.device):
        import mori.shmem as ms

        self._ms = ms
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)

    def create_tensor(self, shape, dtype: torch.dtype) -> torch.Tensor:
        t = self._ms.mori_shmem_create_tensor(shape, dtype)
        t.zero_()
        return t

    def p2p_table(self, t: torch.Tensor) -> torch.Tensor:
        from .communication_ops_host_utils import build_p2p_table

        return build_p2p_table(t, self.rank, self.world_size, self.device)

    def barrier(self):
        self._ms.shmem_barrier_all()

    def close(self):
        pass
