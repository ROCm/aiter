# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node copy-engine (SDMA) transfers of the big-M MegaMoE TP kernel. MORI CCO provides
the SDMA queues and their signal slots (its devComm); the kernel posts remote-signalled puts
(each lands on the peer, then bumps the peer's signal slot for (sender, queue)) on raw
addresses: the data regions live in a same-layout hipIpc arena on every rank (SymmetricArena,
torch caching-allocator memory), not in a MORI window. A MORI window is a VMM allocation
(hipMemCreate), which ROCm can refuse after a pageable host copy pinned a low heap address
(KFD's SVM-range overlap check, reported as out of memory); hipMalloc'd memory is not affected.

Needs MORI built with ``BUILD_CCO_SDMA=ON`` whose CCO device wrapper (libmori_cco_device.bc)
provides ``cco_sdma_put_ptr_rsig__thread`` / ``cco_sdma_signal_slot``, and ``MORI_ENABLE_SDMA=1``.
"""

from __future__ import annotations

import contextlib
import os
import sys

import torch
import torch.distributed as dist

__all__ = ["SdmaComm", "sdma_bitcode"]


@contextlib.contextmanager
def _mori_env():
    """For MORI's import / communicator setup only (the process environment is restored):
    AITER_MORI_PATH (MORI's install directory, if not importable) on sys.path, and
    MORI_ENABLE_SDMA=1 unless set (MORI builds the SDMA queues at communicator creation).
    """
    path = os.environ.get("AITER_MORI_PATH")
    added = bool(path) and path not in sys.path
    if added:
        sys.path.insert(0, path)
    unset = "MORI_ENABLE_SDMA" not in os.environ
    if unset:
        os.environ["MORI_ENABLE_SDMA"] = "1"
    try:
        yield
    finally:
        if added and path in sys.path:
            sys.path.remove(path)
        if unset:
            os.environ.pop("MORI_ENABLE_SDMA", None)


def _mori():
    with _mori_env():
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

    return Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE


def sdma_bitcode(cov: int = 6) -> str:
    """MORI's CCO device bitcode (code object version cov), which the big-M kernel links."""
    with _mori_env():
        from mori.cco.device.bitcode import find_cco_bitcode

        return find_cco_bitcode(cov)


class SdmaComm:
    """MORI SDMA queues (``num_queues`` per peer) of ``group`` plus a same-layout data arena on
    every rank. ``reserve`` carves named regions (same order on every rank), ``commit``
    (collective) creates the MORI communicator / devComm and allocates and maps the arena.
    ``base`` is this rank's arena address, ``bases`` [world] int64 every rank's arena address
    as mapped here (the kernel's put destinations)."""

    def __init__(self, *, group=None, num_queues: int = 8, device=None):
        from .mega_moe_tp import SymmetricArena

        Communicator, Reqs, gda_none = _mori()
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.nq = num_queues
        self._comm_cls, self._reqs_cls, self._gda_none = Communicator, Reqs, gda_none
        self.arena = SymmetricArena(group=group, device=self.device)
        self._slices: dict = {}
        self._committed = False

    def reserve(self, name: str, nbytes: int) -> None:
        if self._committed or name in self._slices:
            raise RuntimeError(f"SdmaComm: cannot reserve {name!r}")
        self._slices[name] = self.arena.reserve(
            name, (max(int(nbytes), 1),), torch.uint8
        )

    def offset(self, name: str) -> int:
        return self._slices[name].offset

    def commit(self) -> SdmaComm:
        uid = [self._comm_cls.get_unique_id() if self.rank == 0 else None]
        src0 = dist.get_global_rank(self.group, 0) if self.group else 0
        dist.broadcast_object_list(uid, src=src0, group=self.group)
        err = ""
        try:
            with _mori_env():
                # (per_rank_vmm: only a virtual-address reservation, rounded up to 4 GiB by
                # MORI; no MORI window is used, the data lives in the hipIpc arena)
                self.comm = self._comm_cls.init(
                    self.world, self.rank, uid[0], per_rank_vmm=1 << 30
                )
                reqs = self._reqs_cls()
                reqs.gda_connection_type = self._gda_none
                reqs.gda_signal_count = 0
                reqs.gda_counter_count = 0
                reqs.sdma_queue_count = self.nq
                self.dev_comm = self.comm.create_dev_comm(reqs)
        except Exception as exc:  # noqa: BLE001 (reported on every rank below)
            err = f"{type(exc).__name__}: {exc}"
        errs: list = [None] * self.world
        dist.all_gather_object(errs, err, group=self.group)
        bad = {r: e for r, e in enumerate(errs) if e}
        if bad:
            raise RuntimeError(f"SdmaComm: MORI SDMA setup failed on ranks {bad}")
        self.arena.commit()  # (raises on every rank if any rank fails)
        self.storage = self.arena.storage
        self.base = int(self.storage.data_ptr())
        self.bases = torch.tensor(
            self.arena._base_ptrs, dtype=torch.int64, device=self.device
        )
        # [2][world * nq] int32 per-(peer, queue) signal bases of the big-M kernel: each launch
        # reads one buffer and writes the next launch's into the other
        self.consumed = torch.zeros(
            (2, self.world * self.nq), dtype=torch.int32, device=self.device
        )
        self._committed = True
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return self

    def close(self) -> None:
        """Collective: free the MORI device communicator / communicator (its SDMA queues)
        and the arena. The instance is unusable afterwards."""
        if not self._committed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.dev_comm.close()
        self.comm.destroy()
        self.arena.close()
        self._committed = False

    def view(
        self, name: str, shape, dtype: torch.dtype, offset: int = 0
    ) -> torch.Tensor:
        """A torch view of region ``name`` (plus a byte offset into it)."""
        sl = self._slices[name]
        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        assert offset + n <= sl.nbytes, (name, offset, n, sl.nbytes)
        return (
            self.storage[sl.offset + offset : sl.offset + offset + n]
            .view(dtype)
            .view(shape)
        )
