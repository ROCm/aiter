# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The four HIP IPC calls :mod:`.symmetric_arena` needs, via ctypes."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IPC_HANDLE_BYTES",
    "ipc_close_mem_handle",
    "ipc_get_mem_handle",
    "ipc_open_mem_handle",
    "mem_allocation_base",
]

IPC_HANDLE_BYTES = 64

_LAZY_ENABLE_PEER_ACCESS = 1


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * IPC_HANDLE_BYTES)]


@dataclass
class _Function:
    name: str
    restype: Any
    argtypes: list[Any]


_EXPORTED = [
    _Function("hipGetErrorString", ctypes.c_char_p, [ctypes.c_int]),
    _Function(
        "hipIpcGetMemHandle",
        ctypes.c_int,
        [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p],
    ),
    _Function(
        "hipIpcOpenMemHandle",
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint],
    ),
    _Function("hipIpcCloseMemHandle", ctypes.c_int, [ctypes.c_void_p]),
    _Function(
        "hipMemGetAddressRange",
        ctypes.c_int,
        [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ],
    ),
]

_FUNCS: dict[str, Any] | None = None


def _loaded_library_path(name: str) -> str | None:
    """Path of an already-mapped shared library, from ``/proc/self/maps``."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            if name in line and "/" in line:
                return line[line.index("/") :].strip()
    return None


def _funcs() -> dict[str, Any]:
    """Resolve the runtime lazily -- importing this module must not touch HIP."""
    global _FUNCS
    if _FUNCS is not None:
        return _FUNCS
    path = _loaded_library_path("libamdhip64") or "libamdhip64.so"
    lib = ctypes.CDLL(path)
    resolved = {}
    for entry in _EXPORTED:
        func = getattr(lib, entry.name)
        func.restype = entry.restype
        func.argtypes = entry.argtypes
        resolved[entry.name] = func
    _FUNCS = resolved
    return _FUNCS


def _check(funcs, status: int, what: str) -> None:
    if status != 0:
        message = funcs["hipGetErrorString"](status)
        detail = message.decode("utf-8") if message else f"hipError_t {status}"
        raise RuntimeError(f"{what} failed: {detail}")


def ipc_get_mem_handle(ptr: int) -> bytes:
    """Export the allocation containing ``ptr`` for other processes to map."""
    funcs = _funcs()
    handle = hipIpcMemHandle_t()
    _check(
        funcs,
        funcs["hipIpcGetMemHandle"](ctypes.byref(handle), ctypes.c_void_p(ptr)),
        "hipIpcGetMemHandle",
    )
    return bytes(bytearray(handle.reserved))


def ipc_open_mem_handle(handle: bytes) -> int:
    """Map a peer's exported allocation and return its device address here."""
    if len(handle) != IPC_HANDLE_BYTES:
        raise ValueError(
            f"IPC handle must be {IPC_HANDLE_BYTES} bytes, got {len(handle)}"
        )
    funcs = _funcs()
    raw = hipIpcMemHandle_t()
    ctypes.memmove(ctypes.byref(raw), handle, IPC_HANDLE_BYTES)
    peer = ctypes.c_void_p()
    _check(
        funcs,
        funcs["hipIpcOpenMemHandle"](
            ctypes.byref(peer), raw, ctypes.c_uint(_LAZY_ENABLE_PEER_ACCESS)
        ),
        "hipIpcOpenMemHandle",
    )
    return int(peer.value or 0)


def ipc_close_mem_handle(peer_ptr: int) -> None:
    """Unmap a pointer returned by :func:`ipc_open_mem_handle`."""
    if not peer_ptr:
        return
    funcs = _funcs()
    _check(
        funcs,
        funcs["hipIpcCloseMemHandle"](ctypes.c_void_p(peer_ptr)),
        "hipIpcCloseMemHandle",
    )


def mem_allocation_base(ptr: int) -> int:
    """Base address of the allocation that contains ``ptr``."""
    funcs = _funcs()
    base = ctypes.c_void_p()
    size = ctypes.c_size_t()
    _check(
        funcs,
        funcs["hipMemGetAddressRange"](
            ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
        ),
        "hipMemGetAddressRange",
    )
    return int(base.value or 0)
