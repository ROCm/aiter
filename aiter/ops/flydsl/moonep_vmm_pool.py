# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Expert-weight pool over HIP VMM: every rank's segment in one virtual range.

A pool holds several tensors ("parts", e.g. w1, its scale, w2 and its scale)
with one ``[epn + B]`` window each.  Every rank owns one segment with its
parts back to back, each part's home experts followed by its prefetch slots::

    segment pe = base + pe * segment_bytes
        part i at  segment + part_offsets[i]
            rows [0, epn)        home experts
            rows [epn, epn + B)  prefetch slots

so each part's window is an ordinary contiguous ``[epn + B, ...]`` tensor and
its ``[epn]`` prefix the owner-only one.  ``segment_bytes`` is the parts' bytes
rounded up to the VMM granularity; the padding sits at the segment's end.

Other ranks' segments are *their* physical memory, mapped here over XGMI, so a
prefetch kernel reaches any expert by offset arithmetic on one base pointer.
One pool per layer means one allocation, one fd exchange and one prefetch
launch for all of its parts.

Modelled on mori's CCO (``src/cco/cco_init.cpp``: one ``hipMemAddressReserve``,
then ``mapPeer`` maps each imported peer handle at ``flatBase + pe * stride``).
CCO itself is unusable here because it rounds its per-rank stride up to 4 GiB so
the device side can pack it into ``stride >> 32``, which leaves 4 GiB holes
between peers; our stride is exactly one segment.

Two constraints from mori's own experience with these APIs:

* **Build pools single-threaded, at init.** ROCr races on concurrent
  ``hsa_amd_vmem_map`` / ``hipMemSetAccess``; mori serialises its own VMM calls
  behind a process mutex, but that mutex lives in an anonymous namespace in
  ``cco_init.cpp`` and does not cover callers outside mori.
* **ROCm >= 7.1.4 for production.** Earlier runtimes leak a DRM fd per
  ``hipMemUnmap`` and can crash in ``hsa_amd_vmem_map`` (mori ``91d1710e``
  carried a rocr patch for both; ``e65e64af`` dropped it on the 7.1.4 image).
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import math
import os
import shutil
import socket
import struct
import tempfile

import torch
import torch.distributed as dist

_hip = ctypes.CDLL("libamdhip64.so")

_ALLOC_TYPE_PINNED = 1
_HANDLE_TYPE_POSIX_FD = 1
_LOCATION_TYPE_DEVICE = 1
_ACCESS_PROT_READWRITE = 3
_GRANULARITY_RECOMMENDED = 1
_ERR_PEER_ACCESS_ALREADY_ENABLED = 704

_P, _SZ, _I, _ULL = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_ulonglong


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", _I), ("id", _I)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", _I),
        ("requestedHandleType", _I),
        ("location", _MemLocation),
        ("win32HandleMetaData", _P),
        ("allocFlags", _AllocFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", _I)]


# Without explicit argtypes ctypes passes bare python ints as 32-bit C int,
# which silently truncates every size_t here -- fatal past a 4 GiB reservation.
_hip.hipMemGetAllocationGranularity.argtypes = [_P, _P, _I]
_hip.hipMemAddressReserve.argtypes = [_P, _SZ, _SZ, _P, _ULL]
_hip.hipMemAddressFree.argtypes = [_P, _SZ]
_hip.hipMemCreate.argtypes = [_P, _SZ, _P, _ULL]
_hip.hipMemMap.argtypes = [_P, _SZ, _SZ, _P, _ULL]
_hip.hipMemUnmap.argtypes = [_P, _SZ]
_hip.hipMemSetAccess.argtypes = [_P, _SZ, _P, _SZ]
_hip.hipMemRelease.argtypes = [_P]
_hip.hipMemExportToShareableHandle.argtypes = [_P, _P, _I, _ULL]
_hip.hipMemImportFromShareableHandle.argtypes = [_P, _P, _I]
_hip.hipDeviceEnablePeerAccess.argtypes = [_I, ctypes.c_uint]
_hip.hipGetErrorString.argtypes = [_I]
_hip.hipGetErrorString.restype = ctypes.c_char_p

_instance_counter = itertools.count()


def _check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(
            f"{what} failed: hipError {err} ({_hip.hipGetErrorString(err).decode()})"
        )


def _prop(dev: int) -> _MemAllocationProp:
    p = _MemAllocationProp()
    p.type = _ALLOC_TYPE_PINNED
    p.requestedHandleType = _HANDLE_TYPE_POSIX_FD
    p.location.type = _LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


def vmm_granularity(dev: int) -> int:
    g = _SZ(0)
    p = _prop(dev)
    _check(
        _hip.hipMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(p), _GRANULARITY_RECOMMENDED
        ),
        "hipMemGetAllocationGranularity",
    )
    return g.value


class _RawBuf:
    """Adopt a raw device pointer into torch, the way mori wraps shmem tensors."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "<u1",
            "strides": None,
            "version": 3,
        }


class MoonEPVmmPool:
    """Per-rank segments of ``[epn | B]`` part windows in one virtual range."""

    def __init__(
        self,
        *,
        part_row_bytes: tuple[int, ...],
        experts_per_rank: int,
        prefetch_slots: int,
        rank: int,
        world_size: int,
        device: torch.device | None = None,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        if not part_row_bytes or any(b <= 0 or b % 16 for b in part_row_bytes):
            raise ValueError(
                f"every part row must be a positive multiple of 16 bytes, "
                f"got {part_row_bytes}"
            )
        if experts_per_rank <= 0 or prefetch_slots < 0:
            raise ValueError(
                "experts_per_rank must be positive and prefetch_slots non-negative"
            )
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        dev = self.device.index
        self.rank, self.world_size = rank, world_size
        self.experts_per_rank = experts_per_rank
        self.prefetch_slots = prefetch_slots
        self.part_row_bytes = tuple(part_row_bytes)
        self._group = group
        self._closed = False

        window = experts_per_rank + prefetch_slots
        offsets, used = [], 0
        for row_bytes in self.part_row_bytes:
            offsets.append(used)
            used += window * row_bytes
        self.part_offsets = tuple(offsets)
        gran = vmm_granularity(dev)
        self.granularity = gran
        # The segment is the unit we map, so it must be granularity-aligned;
        # the parts need no alignment beyond their 16-byte rows.
        self.segment_bytes = -(-used // gran) * gran
        self.total_bytes = world_size * self.segment_bytes

        self._handles: list[ctypes.c_void_p] = []
        self._base = _P(0)
        _check(
            _hip.hipMemAddressReserve(
                ctypes.byref(self._base), self.total_bytes, gran, None, 0
            ),
            "hipMemAddressReserve",
        )
        try:
            self._map_local(dev)
            self._map_peers(dev)
            desc = _MemAccessDesc()
            desc.location.type = _LOCATION_TYPE_DEVICE
            desc.location.id = dev
            desc.flags = _ACCESS_PROT_READWRITE
            _check(
                _hip.hipMemSetAccess(
                    self._base, self.total_bytes, ctypes.byref(desc), 1
                ),
                "hipMemSetAccess",
            )
        except Exception:
            self.close()
            raise

    # ---------------------------------------------------------------- mapping

    def _map_local(self, dev: int) -> None:
        prop = _prop(dev)
        h = _P(0)
        _check(
            _hip.hipMemCreate(
                ctypes.byref(h), self.segment_bytes, ctypes.byref(prop), 0
            ),
            "hipMemCreate",
        )
        self._handles.append(h)
        _check(
            _hip.hipMemMap(
                _P(self._base.value + self.rank * self.segment_bytes),
                self.segment_bytes,
                0,
                h,
                0,
            ),
            "hipMemMap (local)",
        )

    def _map_peers(self, dev: int) -> None:
        if self.world_size <= 1:
            return
        fd = _I(-1)
        _check(
            _hip.hipMemExportToShareableHandle(
                ctypes.byref(fd), self._handles[0], _HANDLE_TYPE_POSIX_FD, 0
            ),
            "hipMemExportToShareableHandle",
        )
        # The fds only carry the allocations across processes: once a peer's is
        # imported the HIP handle owns it, and every peer holds its own copy of
        # ours after the exchange.  Close them all now; a deep model keeps
        # several pools per layer alive for its whole lifetime.
        try:
            peer_fds, devs = _exchange_fds(
                self.rank, self.world_size, fd.value, dev, self._group
            )
        finally:
            os.close(fd.value)
        try:
            for pe in range(self.world_size):
                if pe == self.rank:
                    continue
                _enable_peer(devs[pe], dev)
                h = _P(0)
                err = _hip.hipMemImportFromShareableHandle(
                    ctypes.byref(h), _P(peer_fds[pe]), _HANDLE_TYPE_POSIX_FD
                )
                if err != 0:
                    # ROCm 7.0.x wants &fd where 7.1.0+ wants (void*)(uintptr_t)fd;
                    # mori carries the same shim in utils/hip_compat.hpp.
                    cfd = _I(peer_fds[pe])
                    err = _hip.hipMemImportFromShareableHandle(
                        ctypes.byref(h), ctypes.byref(cfd), _HANDLE_TYPE_POSIX_FD
                    )
                _check(err, f"hipMemImportFromShareableHandle (peer {pe})")
                os.close(peer_fds[pe])
                peer_fds[pe] = -1
                self._handles.append(h)
                _check(
                    _hip.hipMemMap(
                        _P(self._base.value + pe * self.segment_bytes),
                        self.segment_bytes,
                        0,
                        h,
                        0,
                    ),
                    f"hipMemMap (peer {pe})",
                )
        finally:
            for pe, peer_fd in enumerate(peer_fds):
                if pe != self.rank and peer_fd >= 0:
                    os.close(peer_fd)

    # ---------------------------------------------------------------- tensors

    @property
    def base(self) -> int:
        """Address of rank 0's segment; rank ``pe``'s is ``pe * segment_bytes`` on."""
        return self._base.value

    @property
    def local_segment(self) -> torch.Tensor:
        """This rank's whole segment as bytes."""
        start = self._base.value + self.rank * self.segment_bytes
        return torch.as_tensor(_RawBuf(start, self.segment_bytes), device=self.device)

    def part(self, index: int, dtype: torch.dtype, row_shape: tuple[int, ...]):
        """This rank's ``[epn + B, *row_shape]`` window of part ``index``.

        The result does **not** inherit ``is_shuffled``: a freshly adopted
        pointer has no attributes, so callers holding pre-shuffled weights set it.
        """
        row_bytes = self.part_row_bytes[index]
        want = math.prod(row_shape) * torch.empty(0, dtype=dtype).element_size()
        if want != row_bytes:
            raise ValueError(
                f"row_shape {row_shape} of {dtype} spans {want} B but part "
                f"{index} rows hold {row_bytes} B"
            )
        window = self.experts_per_rank + self.prefetch_slots
        start = self.part_offsets[index]
        raw = self.local_segment[start : start + window * row_bytes]
        return raw.view(dtype).view(window, *row_shape)

    # ---------------------------------------------------------------- teardown

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._base.value:
            _hip.hipMemUnmap(self._base, self.total_bytes)
        for h in self._handles:
            _hip.hipMemRelease(h)
        self._handles.clear()
        if self._base.value:
            _hip.hipMemAddressFree(self._base, self.total_bytes)
            self._base = _P(0)

    def __del__(self):  # best effort; explicit close() is the contract
        with contextlib.suppress(Exception):
            self.close()


# ------------------------------------------------------------------ internals


def _enable_peer(peer_dev: int, my_dev: int) -> None:
    if peer_dev == my_dev:
        return
    err = _hip.hipDeviceEnablePeerAccess(peer_dev, 0)
    # Consume the sticky last-error: this call leaves it set even on the benign
    # AlreadyEnabled path, and the next torch op picks it up via
    # hipGetLastError() and reports "operation not supported" (seen on gfx950).
    # mori does the same in cco_init.cpp mapPeer.
    _hip.hipGetLastError()
    if err not in (0, _ERR_PEER_ACCESS_ALREADY_ENABLED):
        raise RuntimeError(f"hipDeviceEnablePeerAccess({peer_dev}) -> hipError {err}")


def _exchange_fds(rank, world, fd, dev, group) -> tuple[list[int], list[int]]:
    """All-gather each rank's shareable fd and device over unix sockets + SCM_RIGHTS.

    fds cannot travel through torch.distributed; mori solves the same problem
    with LocalBootstrapNetwork ("Use cases: VMM shareable handle (file
    descriptor) exchange", application/bootstrap/local_bootstrap.hpp).

    Ordering matters: connect() to a listening socket completes out of the
    backlog without the peer accepting, so everyone must finish connecting
    before anyone blocks in accept().  accept-then-connect deadlocks.
    """

    # ``rank``/``world`` are positions inside ``group``.  The leader's global
    # rank sources the broadcast and names the socket directory, so several
    # EP groups in one world bootstrap side by side.
    leader = 0 if group is None else dist.get_global_rank(group, 0)
    token = next(_instance_counter)
    base = os.environ.get("MOONEP_VMM_SOCKDIR") or tempfile.gettempdir()
    seed = [None]
    if rank == 0:
        seed[0] = os.path.join(base, f"moonep_vmm_g{leader}_{os.getpid()}_{token}")
    dist.broadcast_object_list(seed, src=leader, group=group)
    sockdir = seed[0]
    os.makedirs(sockdir, exist_ok=True)

    path = os.path.join(sockdir, f"r{rank}.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(path)
        srv.listen(world)
        dist.barrier(group=group)  # every socket bound and listening

        clients = {}
        for step in range(1, world):
            peer = (rank + step) % world
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(60.0)
            c.connect(os.path.join(sockdir, f"r{peer}.sock"))
            clients[peer] = c

        srv.settimeout(60.0)
        for _ in range(world - 1):
            conn, _addr = srv.accept()
            try:
                socket.send_fds(conn, [struct.pack("<i", dev)], [fd])
            finally:
                conn.close()

        out = [-1] * world
        devs = [-1] * world
        out[rank], devs[rank] = fd, dev
        for peer, c in clients.items():
            try:
                msg, fds, _flags, _addr = socket.recv_fds(c, 4, 1)
                if len(fds) != 1 or len(msg) != 4:
                    raise RuntimeError(f"rank {rank}: no fd/device from peer {peer}")
                out[peer] = fds[0]
                (devs[peer],) = struct.unpack("<i", msg)
            finally:
                c.close()
        return out, devs
    finally:
        srv.close()
        dist.barrier(group=group)
        if rank == 0:
            shutil.rmtree(sockdir, ignore_errors=True)


__all__ = ["MoonEPVmmPool", "vmm_granularity"]
