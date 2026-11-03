# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host helpers shared by FlyDSL communication operators."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .ipc_heap import UncachedIpcHeap

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


# Interconnect
# ============

LINKS = ("pcie", "xgmi")
# Read only when the KFD topology cannot be; the same variable as the FlyDSL
# all-reduce, so one setting covers every FlyDSL collective on a host.
LINK_ENV = "AITER_FLY_AR_LINK"
INBOX_MEMORY_MODES = ("auto", "uncached", "finegrained")

# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def probe_xgmi_peer_links() -> bool | None:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe, from
    the KFD topology. None when the topology cannot be read.

    The arch cannot tell: an MI350X (xGMI) and an MI350P (PCIe-only) both
    report ``gfx950``, and the right inbox memory type is opposite on the two.

    Both ``io_links`` and ``p2p_links`` are scanned. KFD fills ``p2p_links``
    only for peers reached through a host bridge, so on a directly connected
    xGMI mesh the peers appear solely under ``io_links``.

    The answer is per host: one xGMI link classifies the whole node as xGMI,
    which holds for the uniform single-node systems this targets.
    """
    try:
        found = False
        for subdir in ("io_links", "p2p_links"):
            for props in _KFD_NODES.glob(f"*/{subdir}/*/properties"):
                found = True
                for line in props.read_text().splitlines():
                    field, _, value = line.partition(" ")
                    if field == "type" and int(value) == _HSA_IOLINK_TYPE_XGMI:
                        return True
        return False if found else None
    except (OSError, ValueError):
        logger.debug("FlyDSL comm: cannot read KFD topology", exc_info=True)
        return None


@functools.lru_cache(maxsize=1)
def _probe_link() -> str | None:
    xgmi = probe_xgmi_peer_links()
    return None if xgmi is None else "xgmi" if xgmi else "pcie"


def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host: the KFD topology, or
    ``AITER_FLY_AR_LINK`` when the topology cannot be read."""
    link = _probe_link()
    if link is not None:
        return link
    raw = os.environ.get(LINK_ENV, "").strip().lower()
    if raw not in LINKS:
        raise ValueError(
            "cannot tell if this host has xGMI or PCIe links: the KFD topology "
            f"is not readable. Set {LINK_ENV} to one of {LINKS}, got {raw!r}."
        )
    return raw


def resolve_inbox_memory(mode: str, world_size: int, link: str | None = None) -> str:
    """The memory type an ``inbox_memory`` mode allocates peer-written buffers in.

    ``"auto"`` is ``uncached`` on xGMI and ``finegrained`` on PCIe, except at
    TP2, where it is ``uncached`` on PCIe too. Over PCIe, uncached peer writes
    serialize per destination as soon as a GPU writes to more than one peer;
    fine-grained memory is cacheable, so its L2 combines the writes into full
    PCIe packets, and a release fence publishes them.
    """
    if mode not in INBOX_MEMORY_MODES:
        raise ValueError(
            f"inbox_memory must be one of {INBOX_MEMORY_MODES}, got {mode!r}"
        )
    if mode != "auto":
        return mode
    if int(world_size) == 2 or (link or detect_link()) == "xgmi":
        return "uncached"
    return "finegrained"


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


# hipExtMallocWithFlags mode of each IPC heap memory type.
_IPC_FLAGS = {
    "uncached": UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
    "finegrained": UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED,
}


class IpcSymmetricHeap:
    """HIP IPC buffers in a chosen memory type, exchanged over *group*.

    One allocation per tensor, each exported by its own IPC handle. ``publish``
    gathers every rank's handles and opens the peers'. Works on any
    ``torch.distributed`` backend: the handles travel as Python objects.
    """

    def __init__(self, *, rank, world_size, device, group, memory):
        if memory not in _IPC_FLAGS:
            raise ValueError(
                f"memory must be one of {tuple(_IPC_FLAGS)}, got {memory!r}"
            )
        self.memory = memory
        self.rank, self.world_size, self.device = rank, world_size, device
        self.group = group
        self._flags = _IPC_FLAGS[memory]
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
