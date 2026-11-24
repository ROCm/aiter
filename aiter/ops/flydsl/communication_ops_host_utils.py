# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host helpers shared by FlyDSL communication operators."""

from __future__ import annotations

import logging
import weakref

import torch
import torch.distributed as dist

from .comm_collectives_host_shared import resolve_inbox_flags
from .ipc_heap import IpcHeap

logger = logging.getLogger("aiter")


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
#   publish(error=None)          Make every tensor created so far addressable
#                                by the peers. Also a barrier. A rank that
#                                failed since the heap was created passes its
#                                error: every rank still joins, and every rank
#                                raises, so none is left waiting.
#   p2p_table(tensor)            i64 device table of the tensor on every rank.
#   barrier()                    Host barrier over the participants.
#   close()                      Release the heap's memory. Collective.
#   needs_close                  True while memory awaits a collective close.


def _vote(group, error):
    """Raise on every rank in *group* if any of them has an *error*.

    The rank that failed raises its own error, the others a RuntimeError.
    """
    oks = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(oks, error is None, group=group)
    if error is not None:
        raise error
    if not all(oks):
        failed = [rank for rank, ok in enumerate(oks) if not ok]
        raise RuntimeError(f"rank(s) {failed} failed to set up the symmetric heap")


class MoriSymmetricHeap:
    """MoRI shmem symmetric heap. MoRI must be initialized by the caller.

    The memory type is MoRI's, set by ``MORI_SHMEM_HEAP_TYPE``: uncached unless
    it is ``normal``.
    """

    memory = "uncached"
    # MoRI owns the memory and releases it when it is finalized.
    needs_close = False

    def __init__(self, *, rank, world_size, device, group):
        import mori.shmem as ms

        if rank != ms.shmem_mype() or world_size != ms.shmem_npes():
            raise ValueError(
                f"rank/world_size ({rank}/{world_size}) != MoRI "
                f"PE/npes ({ms.shmem_mype()}/{ms.shmem_npes()})"
            )
        self._ms = ms
        self.rank, self.world_size, self.device = rank, world_size, device
        self.group = group

    def create_tensor(self, shape, dtype):
        from mori.shmem import mori_shmem_create_tensor

        return mori_shmem_create_tensor(shape, dtype)

    def publish(self, error=None):
        # The vote comes first: a rank that failed must not be awaited in the
        # MoRI barrier.
        _vote(self.group, error)
        self._ms.shmem_barrier_all()

    def barrier(self):
        self._ms.shmem_barrier_all()

    def p2p_table(self, tensor):
        return build_p2p_table(tensor, self.rank, self.world_size, self.device)

    def close(self):
        """The tensors stay in MoRI's heap until MoRI is finalized."""


def _free_allocation(ptr, device, exported):
    # Freeing memory that a peer still maps is undefined behavior in HIP, so
    # an allocation that may still be mapped is leaked rather than freed.
    if exported[0]:
        return
    try:
        with torch.cuda.device(device):
            IpcHeap.free_device_mem(ptr)
    except RuntimeError as exc:
        logger.warning("IPC heap: freeing %#x failed: %s", ptr, exc)


class _IpcAllocation:
    """One HIP IPC allocation, as a ``__cuda_array_interface__`` byte array.

    The tensors made from the IPC allocation hold it through their storage.
    Hence, the memory stays valid while any of them, or any view of them, is alive.
    The IPC memory is freed only after the last tensor referencing it is out of scope,
    but only while no peer maps it. Collective ``close`` confirms that every peer
    has unmapped the IPC memory, otherwise the IPC memory is leaked.
    """

    def __init__(self, ptr, nbytes, device):
        self.ptr = int(ptr)
        self.__cuda_array_interface__ = {
            "data": (self.ptr, False),
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }
        # A list, shared with the finalizer, which must not reference self.
        self._exported = [False]
        finalizer = weakref.finalize(
            self, _free_allocation, self.ptr, device, self._exported
        )
        # At exit the process releases the memory, and HIP may be torn down.
        finalizer.atexit = False

    @property
    def exported(self):
        return self._exported[0]

    @exported.setter
    def exported(self, value):
        self._exported[0] = bool(value)


class IpcSymmetricHeap:
    """HIP IPC buffers in a chosen memory type, exchanged over *group*.

    One allocation per tensor, each exported by its own IPC handle. ``publish``
    gathers every rank's handles and opens the peers'. Works on any
    ``torch.distributed`` backend: the handles travel as Python objects.

    The tensors own their memory (see ``_IpcAllocation``). The heap only
    releases it: ``close`` is collective, and without it the memory exported to
    the peers is never freed.
    """

    def __init__(self, *, rank, world_size, device, group, memory):
        if memory not in ("uncached", "finegrained"):
            raise ValueError(
                f"IpcSymmetricHeap needs a resolved memory type, got {memory!r}"
            )
        self._flags, self.memory = resolve_inbox_flags(memory, world_size)
        self.rank, self.world_size, self.device = rank, world_size, device
        self.group = group
        self._local = []
        # Per allocation, the base address on every rank; ``None`` for the
        # allocations not yet published.
        self._peer_ptrs = []
        self._index = {}
        self._opened = []
        self._closed = False

    @property
    def needs_close(self):
        return not self._closed and bool(self._local)

    def create_tensor(self, shape, dtype):
        if self._closed:
            raise RuntimeError("IpcSymmetricHeap is closed")
        nbytes = torch.Size(shape).numel() * dtype.itemsize
        with torch.cuda.device(self.device):
            ptr = IpcHeap.alloc(max(nbytes, 1), self._flags)
        allocation = _IpcAllocation(ptr, nbytes, self.device)
        self._index[allocation.ptr] = len(self._local)
        self._local.append(allocation)
        self._peer_ptrs.append(None)
        tensor = torch.as_tensor(allocation, device=self.device)
        return tensor.view(dtype).view(*shape)

    def publish(self, error=None):
        start = next(
            (i for i, p in enumerate(self._peer_ptrs) if p is None),
            len(self._peer_ptrs),
        )
        pending = self._local[start:]
        handles = None
        if error is None:
            try:
                handles = [IpcHeap.get_mem_handle_bytes(a.ptr) for a in pending]
            except Exception as exc:  # noqa: BLE001
                error = exc
        # From the gather on, a peer may map them.
        for allocation in pending:
            allocation.exported = True
        # Every rank joins every collective, and a failure on one rank raises on
        # all of them after the vote: raising at once would leave the peers
        # waiting in a collective this rank never joins.
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, handles, group=self.group)
        if error is None:
            try:
                if any(h is None or len(h) != len(handles) for h in gathered):
                    raise RuntimeError("a peer rank failed to export its buffers")
                with torch.cuda.device(self.device):
                    for i, allocation in enumerate(pending):
                        ptrs = []
                        for peer, peer_handles in enumerate(gathered):
                            if peer == self.rank:
                                ptrs.append(allocation.ptr)
                            else:
                                base = IpcHeap.open_mem_handle(peer_handles[i])
                                self._opened.append(base)
                                ptrs.append(base)
                        self._peer_ptrs[start + i] = ptrs
            except Exception as exc:  # noqa: BLE001
                error = exc
        try:
            _vote(self.group, error)
        except Exception:
            # Every rank is here, so the teardown can be collective too.
            self.close()
            raise

    def barrier(self):
        dist.barrier(group=self.group)

    def p2p_table(self, tensor):
        index = self._index.get(tensor.data_ptr())
        if index is None:
            raise ValueError("p2p_table: tensor was not created by this heap")
        ptrs = self._peer_ptrs[index]
        if ptrs is None:
            raise RuntimeError("p2p_table before publish")
        return torch.tensor(ptrs, dtype=torch.int64, device=self.device)

    def close(self):
        """Collective: unmap the peers' buffers and release ours.

        Every rank must call it, after its last use of the heap. IPC buffers
        are freed once every peer has unmapped them and no tensor on them is left.
        """
        if self._closed:
            return
        self._closed = True
        error = None
        try:
            with torch.cuda.device(self.device):
                # No store of ours into a peer's buffer is in flight after this.
                torch.cuda.synchronize(self.device)
                for base in self._opened:
                    IpcHeap.close_mem_handle(base)
        except Exception as exc:  # noqa: BLE001
            error = exc
        local, self._opened = self._local, []
        self._local, self._peer_ptrs, self._index = [], [], {}
        # The acknowledgement: past it, no peer maps our buffers, unless one of
        # them failed to unmap, and then every rank leaks rather than frees.
        oks = [None] * self.world_size
        dist.all_gather_object(oks, error is None, group=self.group)
        if all(oks):
            for allocation in local:
                allocation.exported = False
        else:
            logger.warning(
                "IPC heap: a rank failed to unmap its peers' buffers (%s); "
                "leaking %d buffers",
                error,
                len(local),
            )


def make_symmetric_heap(*, memory, rank, world_size, device, group):
    """The heap for a resolved *memory* type: MoRI for ``uncached``, HIP IPC
    for ``finegrained``. MoRI's heap has no fine-grained mode."""
    if memory == "uncached":
        return MoriSymmetricHeap(
            rank=rank, world_size=world_size, device=device, group=group
        )
    return IpcSymmetricHeap(
        rank=rank, world_size=world_size, device=device, group=group, memory=memory
    )
