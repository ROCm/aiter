# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node symmetric memory: one arena per rank, peer-addressable by all."""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import warnings
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


_MORI_RANK_STRIDE = 1 << 32
_MORI_COMMS: dict = {}


class _DeviceBytes:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _clear_hip_error() -> None:
    _hip().hipGetLastError()


def _mori_comm(group, rank: int, world_size: int):
    comm = _MORI_COMMS.get(id(group))
    if comm is None:
        os.environ.setdefault("MORI_CCO_FABRIC_DISABLE", "1")
        os.environ.setdefault("CCO_GDR_CAPABLE", "0")
        from mori.cco import Communicator

        uid = [Communicator.get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(
            uid, src=dist.get_global_rank(group, 0) if group else 0, group=group
        )
        comm = _MORI_COMMS[id(group)] = Communicator.init(
            world_size, rank, uid[0], per_rank_vmm=_MORI_RANK_STRIDE
        )
        _clear_hip_error()
    return comm


def _backend() -> str:
    name = os.environ.get("AITER_SYMM_BACKEND", "ipc")
    if name not in ("ipc", "mori"):
        raise ValueError(f"AITER_SYMM_BACKEND must be ipc or mori, not {name!r}")
    return name


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
    """A same-layout device arena on every rank, mapped into all peers."""

    def __init__(self, *, group=None, device: torch.device | None = None):
        self.group = group
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        if not self.ipc:
            self.rank, self.world_size = group.rank_of(self.device), group.world_size
        else:
            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)
        self._slices: dict[str, SymmetricSlice] = {}
        self._cursor = 0
        self._storage: torch.Tensor | None = None
        self._base_ptrs: tuple[int, ...] = ()
        self._opened: list[bytes] = []
        self.backend = _backend() if self.ipc else "p2p"
        self._mori: tuple = ()

    def reserve(self, name: str, shape, dtype: torch.dtype) -> SymmetricSlice:
        if self._storage is not None or name in self._slices:
            raise RuntimeError(f"cannot reserve {name!r}")
        shape = tuple(int(s) for s in shape)
        offset = (self._cursor + _ALIGN - 1) // _ALIGN * _ALIGN
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        self._cursor = offset + nbytes
        self._slices[name] = SymmetricSlice(offset, nbytes, shape, dtype)
        return self._slices[name]

    def commit(self) -> SymmetricArena:
        total = (self._cursor + _ALIGN - 1) // _ALIGN * _ALIGN
        if self.backend == "mori":
            if self._commit_mori(total):
                return self
            self.backend = "ipc"
        with _no_expandable_segments() if self.ipc else contextlib.nullcontext():
            self._storage = torch.zeros(total, dtype=torch.uint8, device=self.device)
        base_ptr = int(self._storage.data_ptr())
        self._bind_slices()
        if not self.ipc:
            self.group.register(self.rank, base_ptr)
            return self
        with torch.cuda.device(self.device):
            payload = (
                _ipc_handle(base_ptr),
                base_ptr - _allocation_base(base_ptr),
                total,
            )
        torch.cuda.synchronize(self.device)
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, payload, group=self.group)
        ptrs = []
        with torch.cuda.device(self.device):
            for r, (handle, off, size) in enumerate(gathered):
                if size != total:
                    raise RuntimeError(
                        f"arena size disagrees: rank {self.rank} {total} B, rank {r} {size} B"
                    )
                if r == self.rank:
                    ptrs.append(base_ptr)
                else:
                    ptrs.append(_ipc_open(handle) + off)
                    self._opened.append(handle)
        self._base_ptrs = tuple(ptrs)
        dist.barrier(group=self.group)
        return self

    def _bind_slices(self) -> None:
        for s in self._slices.values():
            s.local = (
                self._storage[s.offset : s.offset + s.nbytes]
                .view(s.dtype)
                .view(s.shape)
            )

    def _commit_mori(self, total: int) -> bool:
        with torch.cuda.device(self.device):
            comm = _mori_comm(self.group, self.rank, self.world_size)
            mem, err = None, ""
            try:
                mem = comm.alloc_mem(total)
            except RuntimeError as exc:
                err = str(exc)
            _clear_hip_error()
            errs = [None] * self.world_size
            dist.all_gather_object(errs, err, group=self.group)
            if any(errs):
                if mem is not None:
                    mem.close()
                bad = [r for r, e in enumerate(errs) if e]
                warnings.warn(
                    f"mori window of {total} B failed on ranks {bad} ({errs[bad[0]]}); "
                    "using hipIpc for this arena"
                )
                return False
            win = comm.register_window(mem.ptr, total)
            _clear_hip_error()
            self._mori = (win, mem)
            self._storage = torch.as_tensor(
                _DeviceBytes(mem.ptr, total), device=self.device
            )
            self._storage.zero_()
        self._bind_slices()
        flat = win.local_ptr - self.rank * _MORI_RANK_STRIDE
        self._base_ptrs = tuple(
            flat + r * _MORI_RANK_STRIDE for r in range(self.world_size)
        )
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return True

    def close(self) -> None:
        if self._mori:
            torch.cuda.synchronize(self.device)
            self._storage = None
            for res in self._mori:
                res.close()
            self._mori = ()
            self._base_ptrs = ()
        if self._opened:
            torch.cuda.synchronize(self.device)
            for handle in self._opened:
                _ipc_close(handle)
            self._opened = []
            self._base_ptrs = ()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def barrier(self) -> None:
        if not self.ipc:
            self.group.barrier()
        else:
            dist.barrier(group=self.group)

    @property
    def ipc(self) -> bool:
        return not hasattr(self.group, "register")

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def base_ptrs(self) -> tuple[int, ...]:
        if not self.ipc:
            return self.group.base_ptrs()
        return self._base_ptrs
