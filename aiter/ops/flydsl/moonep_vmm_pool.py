# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Row-contiguous ``[R * stride, ...]`` expert-weight pool over HIP VMM.

Every rank owns one segment of ``stride`` rows holding its resident experts
followed by its own prefetch slots, and the segments of all ranks are stitched
into one contiguous virtual range::

    segment pe      rows [pe * stride, pe * stride + epn)          home experts
                    rows [pe * stride + epn, pe * stride + epn + B) prefetch slots

``stride`` is ``epn + B`` rounded up to the VMM granularity, which on gfx950
means ``stride == epn + B``.  The segment layout is the MoRI/EPLB virtual-id
layout ``physical_id = dest * (epn + B) + slot``, so this rank's
``[epn + B]`` window is exactly the weight tensor a single ``fused_moe`` call
needs under a virtual-id ``expert_mask``, and its ``[epn]`` prefix is the
ordinary owner-only weight tensor.

Other ranks' segments are *their* physical memory, mapped here over XGMI, so a
prefetch kernel reaches any expert by row arithmetic on one base pointer.  Only
``stride`` rows per rank are actually resident.

Modelled on mori's CCO (``src/cco/cco_init.cpp``: one ``hipMemAddressReserve``,
then ``mapPeer`` maps each imported peer handle at ``flatBase + pe * stride``).
CCO itself is unusable here because it rounds its per-rank stride up to 4 GiB so
the device side can pack it into ``stride >> 32``, which leaves 4 GiB holes
between peers; our stride is exactly one segment, so rows stay contiguous
across the seam.

Verified on gfx950 (8x MI355X): granularity is 4 KiB, so every real weight and
scale row is already aligned; a stitched pool feeds mxfp4 a8w4 2-stage
``fused_moe`` bit-identically to a fully local reference.

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

import ctypes
import itertools
import math
import os
import shutil
import socket
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


def pad_experts(rows: int, row_bytes: int, granularity: int) -> int:
    """Smallest ``padded >= rows`` whose segment is a whole granularity multiple.

    The segment is the unit we map, so it must be granularity-aligned, and it
    must hold a whole number of rows or the next peer's segment starts mid-row
    and ``row = pe * stride + k`` stops being true.  Rows *inside* a segment
    need no alignment of their own, which is why this is far weaker than it
    looks: at the measured 4 KiB granularity every real weight/scale row
    divides it exactly and no padding is added.
    """

    step = granularity // math.gcd(row_bytes, granularity)
    return ((rows + step - 1) // step) * step


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
    """One row-contiguous ``[R * stride]`` range of per-rank ``[epn | B]`` segments."""

    def __init__(
        self,
        *,
        row_bytes: int,
        experts_per_rank: int,
        prefetch_slots: int,
        rank: int,
        world_size: int,
        device: torch.device | None = None,
        group: "dist.ProcessGroup | None" = None,
        max_pad_ratio: int = 4,
    ) -> None:
        if row_bytes <= 0 or row_bytes % 16:
            raise ValueError(f"row_bytes must be a positive multiple of 16, got {row_bytes}")
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        dev = self.device.index
        self.rank, self.world_size = rank, world_size
        self.experts_per_rank = experts_per_rank
        self.prefetch_slots = prefetch_slots
        self.row_bytes = row_bytes
        self._group = group
        self._closed = False

        if experts_per_rank <= 0 or prefetch_slots < 0:
            raise ValueError("experts_per_rank must be positive and prefetch_slots non-negative")

        gran = vmm_granularity(dev)
        self.granularity = gran
        segment_rows = experts_per_rank + prefetch_slots
        self.stride = pad_experts(segment_rows, row_bytes, gran)
        if self.stride > max_pad_ratio * segment_rows:
            raise ValueError(
                f"row_bytes={row_bytes} is nearly coprime with the VMM "
                f"granularity {gran} (gcd={math.gcd(row_bytes, gran)}), so a "
                f"whole-row segment would need {self.stride} rows for "
                f"{segment_rows} rows of experts and prefetch slots. Pad each "
                f"row up to a divisor of the granularity before pooling it."
            )

        self.segment_bytes = self.stride * row_bytes
        self.rows = world_size * self.stride
        self.total_bytes = world_size * self.segment_bytes
        self.tensor_bytes = self.total_bytes

        self._handles: list[ctypes.c_void_p] = []
        self._peer_fds: list[int] = []
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
                self.segment_bytes, 0, h, 0,
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
        self._peer_fds = _exchange_fds(
            self.rank, self.world_size, fd.value, self._group
        )
        devs = [0] * self.world_size
        dist.all_gather_object(devs, dev, group=self._group)
        for pe in range(self.world_size):
            if pe == self.rank:
                continue
            _enable_peer(devs[pe], dev)
            h = _P(0)
            err = _hip.hipMemImportFromShareableHandle(
                ctypes.byref(h), _P(self._peer_fds[pe]), _HANDLE_TYPE_POSIX_FD
            )
            if err != 0:
                # ROCm 7.0.x wants &fd where 7.1.0+ wants (void*)(uintptr_t)fd;
                # mori carries the same shim in utils/hip_compat.hpp.
                cfd = _I(self._peer_fds[pe])
                err = _hip.hipMemImportFromShareableHandle(
                    ctypes.byref(h), ctypes.byref(cfd), _HANDLE_TYPE_POSIX_FD
                )
            _check(err, f"hipMemImportFromShareableHandle (peer {pe})")
            self._handles.append(h)
            _check(
                _hip.hipMemMap(
                    _P(self._base.value + pe * self.segment_bytes),
                    self.segment_bytes, 0, h, 0,
                ),
                f"hipMemMap (peer {pe})",
            )

    # ------------------------------------------------------------ row indices

    def global_row(self, expert: int) -> int:
        """Pool row holding global expert ``expert``, on every rank alike."""
        epn = self.experts_per_rank
        return (expert // epn) * self.stride + (expert % epn)

    def prefetch_row(self, slot: int) -> int:
        """Pool row of this rank's prefetch slot ``slot``, right after its home rows."""
        if not 0 <= slot < self.prefetch_slots:
            raise IndexError(f"prefetch slot {slot} out of range")
        return self.rank * self.stride + self.experts_per_rank + slot

    @property
    def home_row_begin(self) -> int:
        return self.rank * self.stride

    # ---------------------------------------------------------------- tensors

    def tensor(self, dtype: torch.dtype, row_shape: tuple[int, ...]) -> torch.Tensor:
        """View the whole range as ``[rows, *row_shape]``.

        The result does **not** inherit ``is_shuffled``: ``fused_moe`` reads that
        off the tensor object (``aiter/fused_moe.py:758``) and a freshly adopted
        pointer has no attributes.  Callers holding pre-shuffled weights must set
        it, or the kernel raises.  See ``mark_shuffled``.
        """

        want = self.rows * math.prod(row_shape) * torch.empty(0, dtype=dtype).element_size()
        if want != self.tensor_bytes:
            raise ValueError(
                f"row_shape {row_shape} of {dtype} spans {want} B but the pool "
                f"holds {self.tensor_bytes} B ({self.rows} rows x {self.row_bytes} B)"
            )
        flat = torch.as_tensor(_RawBuf(self._base.value, self.tensor_bytes), device=self.device)
        return flat.view(dtype).view(self.rows, *row_shape)

    @staticmethod
    def mark_shuffled(*tensors: torch.Tensor) -> None:
        """Restore the flag ``shuffle_weight`` sets and pointer adoption drops."""
        for t in tensors:
            t.is_shuffled = True

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
        for i, fd in enumerate(self._peer_fds):
            if fd >= 0 and i != self.rank:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._peer_fds = []
        if self._base.value:
            _hip.hipMemAddressFree(self._base, self.total_bytes)
            self._base = _P(0)

    def __del__(self):  # best effort; explicit close() is the contract
        try:
            self.close()
        except Exception:
            pass


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


def _exchange_fds(rank, world, fd, group) -> list[int]:
    """All-gather one shareable fd per rank over unix sockets + SCM_RIGHTS.

    fds cannot travel through torch.distributed; mori solves the same problem
    with LocalBootstrapNetwork ("Use cases: VMM shareable handle (file
    descriptor) exchange", application/bootstrap/local_bootstrap.hpp).

    Ordering matters: connect() to a listening socket completes out of the
    backlog without the peer accepting, so everyone must finish connecting
    before anyone blocks in accept().  accept-then-connect deadlocks.
    """

    token = next(_instance_counter)
    base = os.environ.get("MOONEP_VMM_SOCKDIR") or tempfile.gettempdir()
    seed = [None]
    if rank == 0:
        seed[0] = os.path.join(base, f"moonep_vmm_{os.getpid()}_{token}")
    dist.broadcast_object_list(seed, src=0, group=group)
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
                socket.send_fds(conn, [b"x"], [fd])
            finally:
                conn.close()

        out = [-1] * world
        out[rank] = fd
        for peer, c in clients.items():
            try:
                _msg, fds, _flags, _addr = socket.recv_fds(c, 1, 1)
                if len(fds) != 1:
                    raise RuntimeError(f"rank {rank}: no fd from peer {peer}")
                out[peer] = fds[0]
            finally:
                c.close()
        return out
    finally:
        srv.close()
        dist.barrier(group=group)
        if rank == 0:
            shutil.rmtree(sockdir, ignore_errors=True)


__all__ = ["MoonEPVmmPool", "pad_experts", "vmm_granularity"]
