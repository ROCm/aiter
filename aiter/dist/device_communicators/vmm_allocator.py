# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
HIP Virtual Memory Management (VMM) allocator for gfx1250.

hipIpcGetMemHandle / hipIpcOpenMemHandle are not available on gfx1250.
This module provides an alternative using the HIP VMM API
(hipMemCreate / hipMemExportToShareableHandle / hipMemImportFromShareableHandle)
which exports a POSIX fd that can be passed across processes via Unix sockets.
"""

import ctypes
import os
import socket
import struct
import array
import threading
from typing import List

from aiter import logger


def load_hip_runtime() -> ctypes.CDLL:
    """Load the HIP runtime shared library robustly.

    Split rocm-sdk pip layouts (e.g. `_rocm_sdk_core/lib`) ship only the
    versioned soname `libamdhip64.so.N` on the runtime path; the unversioned
    `libamdhip64.so` symlink lives in the -devel package which need not be on
    LD_LIBRARY_PATH at runtime. Try the plain name first, then versioned
    sonames so it works regardless of which package is present.
    """
    candidates = ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]
    last_err = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
    raise OSError(
        f"Could not load the HIP runtime; tried {candidates}. "
        f"Last error: {last_err}"
    )


_hip = load_hip_runtime()

# ---- HIP VMM ctypes structs ----

# hipMemAllocationType
_PINNED = 1
# hipMemAllocationHandleType
_POSIX_FD = 1
# hipMemLocationType
_DEVICE = 1
# hipMemAccessFlags (hipMemAccessFlagsProt*): None=0, Read=1, ReadWrite=3
_READ_WRITE = 3
# hipMemAllocationGranularity_flags
_MINIMUM = 0
_RECOMMENDED = 1


class _hipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("win32_handle", ctypes.c_void_p),
        ("win32_name", ctypes.c_void_p),
        ("allocFlags_compressionType", ctypes.c_ubyte),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_ubyte),
        ("allocFlags_usage", ctypes.c_ushort),
        ("_pad", ctypes.c_byte * 4),
    ]


class _hipMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("flags", ctypes.c_int),
    ]


def _check(ret, msg=""):
    if ret != 0:
        raise RuntimeError(f"HIP VMM error {ret}: {msg}")


def _make_prop(device_id: int) -> _hipMemAllocationProp:
    prop = _hipMemAllocationProp()
    prop.type = _PINNED
    prop.requestedHandleType = _POSIX_FD
    prop.location_type = _DEVICE
    prop.location_id = device_id
    return prop


def _get_granularity(device_id: int) -> int:
    prop = _make_prop(device_id)
    gran = ctypes.c_size_t()
    _check(
        _hip.hipMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(prop), _RECOMMENDED
        ),
        "hipMemGetAllocationGranularity",
    )
    return gran.value


def _round_up(size: int, gran: int) -> int:
    return ((size + gran - 1) // gran) * gran


# ---- VMMBuffer ----


class VMMBuffer:
    """GPU buffer allocated via HIP VMM, shareable across processes via fd."""

    def __init__(self, size: int, device_id: int):
        self._device_id = device_id
        self._gran = _get_granularity(device_id)
        self._alloc_size = _round_up(size, self._gran)
        self._is_imported = False

        prop = _make_prop(device_id)
        self._phys_handle = ctypes.c_void_p()
        _check(
            _hip.hipMemCreate(
                ctypes.byref(self._phys_handle),
                self._alloc_size,
                ctypes.byref(prop),
                0,
            ),
            "hipMemCreate",
        )

        self._va = ctypes.c_void_p()
        _check(
            _hip.hipMemAddressReserve(
                ctypes.byref(self._va),
                self._alloc_size,
                self._gran,
                ctypes.c_void_p(0),
                0,
            ),
            "hipMemAddressReserve",
        )

        _check(
            _hip.hipMemMap(self._va, self._alloc_size, 0, self._phys_handle, 0),
            "hipMemMap",
        )

        self._set_access([device_id])

    def _set_access(self, device_ids: List[int]):
        for dev in device_ids:
            desc = _hipMemAccessDesc(_DEVICE, dev, _READ_WRITE)
            _check(
                _hip.hipMemSetAccess(self._va, self._alloc_size, ctypes.byref(desc), 1),
                f"hipMemSetAccess dev={dev}",
            )

    @property
    def data_ptr(self) -> int:
        return self._va.value

    @property
    def alloc_size(self) -> int:
        return self._alloc_size

    def export_fd(self) -> int:
        fd = ctypes.c_int()
        _check(
            _hip.hipMemExportToShareableHandle(
                ctypes.byref(fd), self._phys_handle, _POSIX_FD, 0
            ),
            "hipMemExportToShareableHandle",
        )
        return fd.value

    @classmethod
    def import_from_fd(
        cls,
        fd: int,
        alloc_size: int,
        local_device_id: int,
        access_device_ids: List[int],
    ) -> "VMMBuffer":
        obj = object.__new__(cls)
        obj._device_id = local_device_id
        obj._gran = _get_granularity(local_device_id)
        obj._alloc_size = alloc_size
        obj._is_imported = True

        obj._phys_handle = ctypes.c_void_p()
        _check(
            _hip.hipMemImportFromShareableHandle(
                ctypes.byref(obj._phys_handle), ctypes.c_void_p(fd), _POSIX_FD
            ),
            "hipMemImportFromShareableHandle",
        )

        obj._va = ctypes.c_void_p()
        _check(
            _hip.hipMemAddressReserve(
                ctypes.byref(obj._va),
                alloc_size,
                obj._gran,
                ctypes.c_void_p(0),
                0,
            ),
            "hipMemAddressReserve (import)",
        )

        _check(
            _hip.hipMemMap(obj._va, alloc_size, 0, obj._phys_handle, 0),
            "hipMemMap (import)",
        )

        obj._set_access(access_device_ids)
        return obj

    def close(self):
        if hasattr(self, "_va") and self._va.value:
            _hip.hipMemUnmap(self._va, self._alloc_size)
            _hip.hipMemAddressFree(self._va, self._alloc_size)
            self._va = ctypes.c_void_p(0)
        if hasattr(self, "_phys_handle") and self._phys_handle.value:
            _hip.hipMemRelease(self._phys_handle)
            self._phys_handle = ctypes.c_void_p(0)

    def __del__(self):
        self.close()


# ---- Unix socket fd passing ----


def _send_fd(sock_path: str, fd: int, timeout: float = 120.0):
    """Connect to a Unix socket and send a file descriptor via SCM_RIGHTS."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(sock_path)
                fds = array.array("i", [fd])
                s.sendmsg(
                    [b"\x00"],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)],
                )
                return
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def _recv_fd(sock_path: str, timeout: float = 120.0) -> int:
    """Listen on a Unix socket and receive one file descriptor via SCM_RIGHTS."""
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(sock_path)
        server.listen(1)
        server.settimeout(timeout)
        conn, _ = server.accept()
        with conn:
            msg, ancdata, _, _ = conn.recvmsg(
                1, socket.CMSG_LEN(ctypes.sizeof(ctypes.c_int))
            )
            for cmsg_level, cmsg_type, cmsg_data in ancdata:
                if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                    fds = array.array("i")
                    fds.frombytes(cmsg_data[: ctypes.sizeof(ctypes.c_int)])
                    return fds[0]
        raise RuntimeError("No fd received")


def vmm_exchange(
    rank: int,
    world_size: int,
    key: str,
    local_buf: VMMBuffer,
    store,
    ranks_tag: str,
    all_device_ids: List[int],
) -> List[int]:
    """Exchange a VMMBuffer across all ranks and return device pointers.

    Each rank exports its buffer as an fd, sends it to every other rank via
    Unix sockets, imports the received fds, and returns a list of local VA
    pointers (one per rank) that can be used to access all ranks' buffers.
    """
    local_device = local_buf._device_id
    alloc_size = local_buf.alloc_size
    prefix = f"aiter_vmm/{ranks_tag}/{key}"

    # Publish socket paths for receiving fds
    sock_dir = f"/tmp/aiter_vmm_{os.getpid()}"
    os.makedirs(sock_dir, exist_ok=True)

    recv_paths = []
    for r in range(world_size):
        if r == rank:
            continue
        path = f"{sock_dir}/{key}_from_r{r}"
        recv_paths.append((r, path))
        store.set(f"{prefix}/sock/r{rank}_from_r{r}", path.encode())

    # Export fd
    export_fd = local_buf.export_fd()

    # Start receiver threads
    received_fds = {}
    recv_errors = {}

    def _do_recv(src_rank, path):
        try:
            received_fds[src_rank] = _recv_fd(path)
        except Exception as e:
            recv_errors[src_rank] = e

    threads = []
    for r, path in recv_paths:
        t = threading.Thread(target=_do_recv, args=(r, path), daemon=True)
        t.start()
        threads.append(t)

    # Store alloc_size so importers know the size
    store.set(f"{prefix}/size/r{rank}", struct.pack("Q", alloc_size))

    # Send our fd to each other rank
    for r in range(world_size):
        if r == rank:
            continue
        target_path = store.get(f"{prefix}/sock/r{r}_from_r{rank}").decode()
        dup_fd = os.dup(export_fd)
        os.set_inheritable(dup_fd, True)
        _send_fd(target_path, dup_fd)
        os.close(dup_fd)

    os.close(export_fd)

    # Wait for all receives
    for t in threads:
        t.join(timeout=120)

    if recv_errors:
        raise RuntimeError(f"VMM fd exchange failed: {recv_errors}")

    # Import received fds and build pointer list
    ptrs = [0] * world_size
    ptrs[rank] = local_buf.data_ptr
    imported_bufs = []

    for r in range(world_size):
        if r == rank:
            continue
        fd = received_fds[r]
        raw_size = store.get(f"{prefix}/size/r{r}")
        remote_size = struct.unpack("Q", raw_size)[0]
        buf = VMMBuffer.import_from_fd(fd, remote_size, local_device, all_device_ids)
        os.close(fd)
        imported_bufs.append(buf)
        ptrs[r] = buf.data_ptr

    # Cleanup socket files
    for r, path in recv_paths:
        try:
            os.unlink(path)
        except OSError:
            pass

    return ptrs, imported_bufs
