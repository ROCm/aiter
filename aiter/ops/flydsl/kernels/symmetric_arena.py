# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node symmetric memory: one arena per rank, peer-addressable by all.

Each rank allocates the same layout and maps every peer's arena with hipIpc. The
exported handle covers the caching-allocator segment holding the arena (peers address
the arena by its offset in it). Destruction is not collective: synchronize every rank
(e.g. a barrier after the last kernel using the arena) before dropping an instance, so
no peer still writes into the memory it returns to the allocator. In containers on the
host network ROCm 7.1 needs HSA_ENABLE_IPC_MODE_LEGACY=1 for the handle exchange.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

__all__ = ["SymmetricArena"]

_ALIGN = 256
_IPC_HANDLE_BYTES = 64


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _IPC_HANDLE_BYTES)]


@functools.cache
def _hip():
    # the HIP runtime instance torch allocated with (already mapped into the process),
    # not whichever libamdhip64 a plain dlopen would find: IPC handles are per runtime
    path = "libamdhip64.so"
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64" in line and "/" in line:
                path = line[line.index("/") :].strip()
                break
    lib = ctypes.CDLL(path)
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    return lib


def _check(status: int, what: str) -> None:
    if status != 0:
        msg = _hip().hipGetErrorString(status)
        raise RuntimeError(f"{what} failed: {msg.decode() if msg else status}")


def _allocation_base(ptr: int) -> int:
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    _check(
        _hip().hipMemGetAddressRange(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
        ),
        "hipMemGetAddressRange",
    )
    return int(base.value or 0)


def _ipc_handle(ptr: int) -> bytes:
    h = _IpcHandle()
    _check(
        _hip().hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)),
        "hipIpcGetMemHandle",
    )
    return bytes(bytearray(h.reserved))


# a peer allocation maps once per process (arenas can share one): handle -> [ptr, refs]
_OPEN: dict[bytes, list] = {}


def _ipc_open(handle: bytes) -> int:
    ent = _OPEN.get(handle)
    if ent is None:
        h = _IpcHandle()
        ctypes.memmove(ctypes.byref(h), handle, _IPC_HANDLE_BYTES)
        peer = ctypes.c_void_p()
        _check(
            _hip().hipIpcOpenMemHandle(ctypes.byref(peer), h, ctypes.c_uint(1)),
            "hipIpcOpenMemHandle",
        )
        ent = _OPEN[handle] = [int(peer.value or 0), 0]
    ent[1] += 1
    return ent[0]


def _ipc_close(handle: bytes) -> None:
    ent = _OPEN[handle]
    ent[1] -= 1
    if ent[1] == 0:
        del _OPEN[handle]
        _check(
            _hip().hipIpcCloseMemHandle(ctypes.c_void_p(ent[0])), "hipIpcCloseMemHandle"
        )


@contextlib.contextmanager
def _no_expandable_segments():
    # hipIpcGetMemHandle cannot export expandable-segment (VMM) memory
    conf = os.environ.get("PYTORCH_HIP_ALLOC_CONF") or os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    )
    on = "expandable_segments:true" in conf.replace(" ", "").lower()
    if on:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    try:
        yield
    finally:
        if on:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")


@dataclass
class SymmetricSlice:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    local: torch.Tensor = field(default=None, repr=False)


class SymmetricArena:
    """A same-layout device arena on every rank of ``group``, mapped into all peers."""

    def __init__(self, *, group=None, device: torch.device | None = None):
        self.group = group
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._slices: dict[str, SymmetricSlice] = {}
        self._cursor = 0
        self._storage: torch.Tensor | None = None
        self._base_ptrs: tuple[int, ...] = ()
        self._opened: list[bytes] = []

    def reserve(self, name: str, shape, dtype: torch.dtype) -> SymmetricSlice:
        """Carve out a named region. Must run in the same order on every rank."""
        if self._storage is not None or name in self._slices:
            raise RuntimeError(f"cannot reserve {name!r}")
        shape = tuple(int(s) for s in shape)
        offset = (self._cursor + _ALIGN - 1) // _ALIGN * _ALIGN
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        self._cursor = offset + nbytes
        self._slices[name] = SymmetricSlice(offset, nbytes, shape, dtype)
        return self._slices[name]

    def commit(self) -> SymmetricArena:
        """Collective: allocate and map every rank's arena (raises on every rank if any
        rank fails)."""
        total = (self._cursor + _ALIGN - 1) // _ALIGN * _ALIGN
        with _no_expandable_segments():
            self._storage = torch.zeros(total, dtype=torch.uint8, device=self.device)
        base_ptr = int(self._storage.data_ptr())
        for s in self._slices.values():
            s.local = (
                self._storage[s.offset : s.offset + s.nbytes]
                .view(s.dtype)
                .view(s.shape)
            )
        try:
            with torch.cuda.device(self.device):
                payload = (
                    _ipc_handle(base_ptr),
                    base_ptr - _allocation_base(base_ptr),
                    total,
                )
            err = ""
        except RuntimeError as exc:
            payload, err = None, str(exc)
        torch.cuda.synchronize(self.device)
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, (payload, err), group=self.group)
        bad = {r: e for r, (_, e) in enumerate(gathered) if e}
        if not bad and any(p[2] != total for p, _ in gathered):
            bad = {r: f"{p[2]} B, not {total} B" for r, (p, _) in enumerate(gathered)}
        ptrs, err = [], ""
        if not bad:
            try:
                with torch.cuda.device(self.device):
                    for r, ((handle, off, _), _) in enumerate(gathered):
                        if r == self.rank:
                            ptrs.append(base_ptr)
                        else:
                            ptrs.append(_ipc_open(handle) + off)
                            self._opened.append(handle)
            except RuntimeError as exc:
                err = str(exc)
            opened: list = [None] * self.world_size
            dist.all_gather_object(opened, err, group=self.group)
            bad = {r: e for r, e in enumerate(opened) if e}
        if bad:
            self.close()
            raise RuntimeError(f"SymmetricArena: mapping failed on ranks {bad}")
        self._base_ptrs = tuple(ptrs)
        dist.barrier(group=self.group)
        return self

    def close(self) -> None:
        """Unmap the peers' arenas (no kernel may use them any more)."""
        if self._opened:
            torch.cuda.synchronize(self.device)
            for handle in self._opened:
                _ipc_close(handle)
            self._opened = []
            self._base_ptrs = ()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 (interpreter teardown)
            pass

    def barrier(self) -> None:
        """Collective; also waits for this device, so a barrier implemented as a GPU
        kernel (NCCL) never runs next to the next (persistent) launch."""
        dist.barrier(group=self.group)
        torch.cuda.synchronize(self.device)

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def base_ptrs(self) -> tuple[int, ...]:
        return self._base_ptrs
