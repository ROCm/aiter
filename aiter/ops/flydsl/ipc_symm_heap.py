# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Symmetric device memory for the FlyDSL intra-node communication ops.

``IpcSymmetricAllocator`` needs nothing beyond HIP and ``torch.distributed``:
every ``create_tensor`` call is collective over ``group`` (the ranks of one
node) and maps one :class:`~aiter.ops.flydsl.dcp_mailbox.peer.PeerBuffer`:
each rank allocates the same number of bytes, the IPC handles are exchanged,
every peer's allocation is opened, and the ranks agree on the outcome (a rank
that fails to open a peer makes every rank raise instead of hanging in the
next collective). ``p2p_table(t)`` returns an i64 device tensor holding every
peer's mapping of ``t`` (the own rank's entry is the local pointer), which is
what the kernels index by PE.

Memory is uncached (``hipDeviceMallocUncached``), the mode mori's shmem heap
uses: the kernels' cross-GPU stores and flag polls rely on it. Every buffer
starts zero-filled.

``MoriShmemAllocator`` keeps the previous behaviour (mori's symmetric heap)
behind the same interface for callers that share buffers with mori kernels.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .dcp_mailbox.peer import PeerBuffer

_ALIGN = 4096


def _group_alive(group) -> bool:
    """Whether *group* is still a live process group of this process."""
    try:
        dist.get_rank(group=group)
    except (RuntimeError, ValueError):  # destroyed / unregistered group
        return False
    return dist.is_initialized()


class IpcSymmetricAllocator:
    """Collective, zero-filled, uncached allocations shared over HIP IPC."""

    def __init__(self, group, device: torch.device):
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._bufs: list[PeerBuffer] = []

    def _check_same_size(self, nbytes: int) -> None:
        """Collective: raise on every rank unless all ranks asked for *nbytes*."""
        span = torch.tensor([nbytes, -nbytes], dtype=torch.int64)
        dist.all_reduce(span, op=dist.ReduceOp.MAX, group=self.group)
        if int(span[0]) != -int(span[1]):
            raise RuntimeError(
                "symmetric allocation size mismatch across ranks: "
                f"{-int(span[1])}..{int(span[0])} bytes (this rank {nbytes})"
            )

    def create_tensor(self, shape, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        nbytes_used = (
            math.prod(int(s) for s in shape)
            * torch.empty((), dtype=dtype).element_size()
        )
        nbytes = (max(1, nbytes_used) + _ALIGN - 1) // _ALIGN * _ALIGN
        self._check_same_size(nbytes)
        buf = PeerBuffer(nbytes, self.group, self.rank, self.world_size, self.device)
        self._bufs.append(buf)
        return buf.bytes[:nbytes_used].view(dtype).view(*shape)

    def p2p_table(self, t: torch.Tensor) -> torch.Tensor:
        base = t.data_ptr()
        for buf in self._bufs:
            if buf.local <= base < buf.local + buf.bytes.numel():
                return buf.addresses + (base - buf.local)
        raise ValueError("tensor was not allocated by this IpcSymmetricAllocator")

    def barrier(self):
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def close(self):
        """Collective: once every rank's kernels are done, unmap the peers'
        allocations; once every rank has unmapped, free the own ones.

        If ``group`` is already destroyed (``GroupCoordinator.destroy`` tears
        down its process groups before its device communicator), the ranks can
        no longer agree that the peers are done, so the memory is left to
        process exit."""
        if not self._bufs:
            return
        if not _group_alive(self.group):
            self._bufs.clear()
            return
        self.barrier()
        for buf in self._bufs:
            buf.close_peers()
        dist.barrier(group=self.group)
        for buf in self._bufs:
            buf.close()
        self._bufs.clear()


class MoriShmemAllocator:
    """mori's symmetric heap behind the ``IpcSymmetricAllocator`` interface."""

    def __init__(self, rank: int, world_size: int, device: torch.device):
        import mori.shmem as ms

        from aiter.dist.device_communicators.all2all import ensure_mori_shmem

        ensure_mori_shmem()

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
