# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host helpers shared by FlyDSL communication operators."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .comm_collectives_host_shared import _resolve_inbox_flags
from .ipc_heap import UncachedIpcHeap


def build_p2p_table(tensor, rank, world_size, device) -> torch.Tensor:
    """i64 table of intra-node P2P pointers to ``tensor`` on every peer."""
    import mori.shmem as ms

    table = torch.zeros(world_size, dtype=torch.int64, device=device)
    for peer in range(world_size):
        ptr = ms.shmem_ptr_p2p(tensor.data_ptr(), rank, peer)
        if not ptr:
            raise ValueError(
                f"shmem_ptr_p2p returned a null pointer for peer {peer} "
                "(no intranode P2P path, or tensor is not in the symmetric heap)"
            )
        table[peer] = ptr
    return table


# Symmetric heaps
# ===============
#
# A symmetric heap hands out device tensors that every rank allocates in the
# same order, and the table of each tensor's address on every peer. The calls
# are collective: every rank makes the same ones in the same order.
#
#   create_tensor(shape, dtype)  A new tensor, not necessarily zeroed. No peer
#                                can address it before ``publish``.
#   publish()                    Make every tensor created so far addressable
#                                by the peers. Also a barrier.
#   p2p_table(tensor)            i64 device table of the tensor on every rank.
#   barrier()                    Host barrier over the participants.


class MoriSymmetricHeap:
    """MoRI shmem symmetric heap. MoRI must be initialized by the caller.

    The memory type is MoRI's, set by ``MORI_SHMEM_HEAP_TYPE``: uncached unless
    it is ``normal``.
    """

    memory = "uncached"

    def __init__(self, *, rank, world_size, device):
        import mori.shmem as ms

        if rank != ms.shmem_mype() or world_size != ms.shmem_npes():
            raise ValueError(
                f"rank/world_size ({rank}/{world_size}) != MoRI "
                f"PE/npes ({ms.shmem_mype()}/{ms.shmem_npes()})"
            )
        self._ms = ms
        self.rank, self.world_size, self.device = rank, world_size, device

    def create_tensor(self, shape, dtype):
        from mori.shmem import mori_shmem_create_tensor

        return mori_shmem_create_tensor(shape, dtype)

    def publish(self):
        self._ms.shmem_barrier_all()

    def barrier(self):
        self._ms.shmem_barrier_all()

    def p2p_table(self, tensor):
        return build_p2p_table(tensor, self.rank, self.world_size, self.device)

    def close(self):
        """The tensors stay in MoRI's heap until MoRI is finalized."""


class _DeviceBuffer:
    """A raw device allocation as a ``__cuda_array_interface__`` byte array."""

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


class IpcSymmetricHeap:
    """HIP IPC buffers in a chosen memory type, exchanged over *group*.

    One allocation per tensor, each exported by its own IPC handle. ``publish``
    gathers every rank's handles and opens the peers'. Works on any
    ``torch.distributed`` backend: the handles travel as Python objects.
    """

    def __init__(self, *, rank, world_size, device, group, memory):
        self._flags, self.memory = _resolve_inbox_flags(memory, world_size)
        self.rank, self.world_size, self.device = rank, world_size, device
        self.group = group
        self._local = []
        # Per allocation, the base address on every rank; ``None`` for the
        # allocations not yet published.
        self._peer_ptrs = []
        self._opened = []

    def create_tensor(self, shape, dtype):
        nbytes = torch.Size(shape).numel() * dtype.itemsize
        with torch.cuda.device(self.device):
            ptr = UncachedIpcHeap.alloc(max(nbytes, 1), self._flags)
        self._local.append(ptr)
        self._peer_ptrs.append(None)
        tensor = torch.as_tensor(_DeviceBuffer(ptr, nbytes), device=self.device)
        return tensor.view(dtype).view(*shape)

    def publish(self):
        start = next(
            (i for i, p in enumerate(self._peer_ptrs) if p is None),
            len(self._peer_ptrs),
        )
        error = None
        try:
            handles = [
                UncachedIpcHeap.get_mem_handle_bytes(p) for p in self._local[start:]
            ]
        except Exception as exc:  # noqa: BLE001
            error, handles = exc, None
        # Every rank joins both collectives, and a failure on one rank raises on
        # all of them after the vote: raising at once would leave the peers
        # waiting in a collective this rank never joins.
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, handles, group=self.group)
        if error is None:
            try:
                if any(h is None or len(h) != len(handles) for h in gathered):
                    raise RuntimeError("a peer rank failed to export its buffers")
                for i, ptr in enumerate(self._local[start:]):
                    ptrs = []
                    for peer, peer_handles in enumerate(gathered):
                        if peer == self.rank:
                            ptrs.append(ptr)
                        else:
                            base = UncachedIpcHeap.open_mem_handle(peer_handles[i])
                            self._opened.append(base)
                            ptrs.append(base)
                    self._peer_ptrs[start + i] = ptrs
            except Exception as exc:  # noqa: BLE001
                error = exc
        oks = [None] * self.world_size
        dist.all_gather_object(oks, error is None, group=self.group)
        if error is not None or not all(oks):
            raise error or RuntimeError("a peer rank failed to open the IPC buffers")

    def barrier(self):
        dist.barrier(group=self.group)

    def p2p_table(self, tensor):
        ptr = tensor.data_ptr()
        index = self._local.index(ptr)
        ptrs = self._peer_ptrs[index]
        if ptrs is None:
            raise RuntimeError("p2p_table before publish")
        return torch.tensor(ptrs, dtype=torch.int64, device=self.device)

    def close(self):
        """Unmap the peers' buffers and free ours, once this device is idle.

        Local: when the device is idle no store of ours is in flight, and a
        peer that still maps our buffers keeps them alive until it unmaps.
        """
        if not self._local:
            return
        torch.cuda.synchronize(self.device)
        for base in self._opened:
            try:
                UncachedIpcHeap.close_mem_handle(base)
            except RuntimeError:
                pass
        for ptr in self._local:
            try:
                UncachedIpcHeap.free_device_mem(ptr)
            except RuntimeError:
                pass
        self._opened, self._local, self._peer_ptrs = [], [], []


def make_symmetric_heap(*, memory, rank, world_size, device, group):
    """The heap for a resolved *memory* type: MoRI for ``uncached``, HIP IPC
    for ``finegrained``. MoRI's heap has no fine-grained mode."""
    if memory == "uncached":
        return MoriSymmetricHeap(rank=rank, world_size=world_size, device=device)
    return IpcSymmetricHeap(
        rank=rank, world_size=world_size, device=device, group=group, memory=memory
    )
