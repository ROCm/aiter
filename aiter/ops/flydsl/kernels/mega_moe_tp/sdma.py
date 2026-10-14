# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node copy-engine (SDMA) transfers of the big-M MegaMoE TP kernel. MORI CCO provides
the SDMA queues and their signal slots (its devComm); the kernel posts remote-signalled puts
(each lands on the peer, then bumps the peer's signal slot for (sender, queue)) on raw
addresses: the data regions live in a same-layout hipIpc arena on every rank (SymmetricArena,
torch caching-allocator memory), not in a MORI window. A MORI window is a VMM allocation
(hipMemCreate), which ROCm can refuse after a pageable host copy pinned a low heap address
(KFD's SVM-range overlap check, reported as out of memory); hipMalloc'd memory is not affected.

Needs MORI built with ``BUILD_CCO_SDMA=ON`` (unmodified: the device functions come from
cco_sdma_rsig.cpp next to this file, compiled against MORI's installed CCO headers by
``sdma_bitcode``) and ``MORI_ENABLE_SDMA=1``.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

__all__ = ["SdmaComm", "sdma_bitcode"]


def _mori():
    path = os.environ.get("AITER_MORI_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    from mori.cco import CCODevCommRequirements, Communicator, GDA_CONNECTION_NONE

    return Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE, sdma_bitcode


_SDMA_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cco_sdma_rsig.cpp")
_sdma_bc: dict = {}


def sdma_bitcode(cov: int = 6) -> str:
    """cco_sdma_rsig.cpp (aiter_cco_sdma_put_rsig__thread / aiter_cco_sdma_signal_slot) as LLVM
    bitcode for code object version cov: compiled by MORI's JIT toolchain (the hipcc flags and
    include dirs of MORI's own device wrapper) against the installed MORI headers, cached in
    MORI's JIT cache (keyed by this source and those headers)."""
    got = _sdma_bc.get(cov)
    if got is not None:
        return got
    import tempfile
    from pathlib import Path

    _mori()  # (AITER_MORI_PATH on sys.path)
    from mori.jit.cache import get_cache_dir
    from mori.jit.config import detect_build_config, detect_nic_type, get_mori_source_root
    from mori.jit.core import FileBaton, _collect_include_dirs, _hipcc_device_bc, _strip_lifetime_intrinsics

    root = get_mori_source_root()
    if root is None:
        raise RuntimeError("MegaMoE TP big-M: MORI's device sources (mori/_jit-sources) not found")
    src = Path(_SDMA_SRC)
    cfg = detect_build_config()
    deps = [src, root / "include" / "mori" / "cco", root / "include" / "mori" / "core"]
    cache = get_cache_dir(cfg.arch, deps, detect_nic_type(), cov=cov) / "aiter_sdma"
    cache.mkdir(parents=True, exist_ok=True)
    bc = cache / "aiter_cco_sdma.bc"
    if not bc.is_file():
        with FileBaton(cache / ".aiter_cco_sdma.bc.lock", wait_for=str(bc)) as baton:
            if not (baton.skipped or bc.is_file()):
                with tempfile.TemporaryDirectory() as td:
                    raw = Path(td) / "raw.bc"
                    _hipcc_device_bc(
                        cfg, src, _collect_include_dirs(root), raw, cov=cov, extra_defines=["-DBUILD_CCO_SDMA=1"]
                    )
                    _strip_lifetime_intrinsics(cfg, raw, bc)
    _sdma_bc[cov] = str(bc)
    return str(bc)


class SdmaComm:
    """MORI SDMA queues (``num_queues`` per peer) of ``group`` plus a same-layout data arena on
    every rank. ``reserve`` carves named regions (same order on every rank), ``commit``
    (collective) creates the MORI communicator / devComm and allocates and maps the arena.
    ``base`` is this rank's arena address, ``bases`` [world] int64 every rank's arena address
    as mapped here (the kernel's put destinations)."""

    def __init__(self, *, group=None, num_queues: int = 8, device=None):
        from .mega_moe_tp import SymmetricArena

        Communicator, Reqs, gda_none, _ = _mori()
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
        self._slices[name] = self.arena.reserve(name, (max(int(nbytes), 1),), torch.uint8)

    def offset(self, name: str) -> int:
        return self._slices[name].offset

    def commit(self) -> "SdmaComm":
        uid = [self._comm_cls.get_unique_id() if self.rank == 0 else None]
        src0 = dist.get_global_rank(self.group, 0) if self.group else 0
        dist.broadcast_object_list(uid, src=src0, group=self.group)
        err = ""
        try:
            self.comm = self._comm_cls.init(self.world, self.rank, uid[0], per_rank_vmm=1 << 30)
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
        self.bases = torch.tensor(self.arena._base_ptrs, dtype=torch.int64, device=self.device)
        self.consumed = torch.zeros(self.world * self.nq, dtype=torch.int32, device=self.device)
        self._committed = True
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return self

    def view(self, name: str, shape, dtype: torch.dtype, offset: int = 0) -> torch.Tensor:
        """A torch view of region ``name`` (plus a byte offset into it)."""
        sl = self._slices[name]
        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        assert offset + n <= sl.nbytes, (name, offset, n, sl.nbytes)
        return self.storage[sl.offset + offset : sl.offset + offset + n].view(dtype).view(shape)
