# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Uncached HIP IPC helpers for the FlyDSL INT4 QuickReduce inbox."""

from __future__ import annotations

import ctypes
import logging
import re
import sys

logger = logging.getLogger(__name__)

_HIP_SONAME_RE = re.compile(r"/libamdhip64\.so(?:\.\d+)*$")


def _mapped_hip_paths() -> list[str]:
    """Paths of every ``libamdhip64`` already mapped into this process.
    """
    paths: list[str] = []
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.rsplit(" ", 1)[-1].strip()
                if _HIP_SONAME_RE.search(path) and path not in paths:
                    paths.append(path)
    except OSError:
        logger.debug("cannot read /proc/self/maps", exc_info=True)
    return paths


def _torch_current_device() -> int | None:
    """torch's bound device, or ``None`` if torch has not initialised HIP.

    Read from ``sys.modules`` rather than imported: importing this module must
    not drag torch in, and loading a HIP runtime through ctypes *before* torch
    aborts the process outright ("Option 'spirv-expand-step' registered more
    than once").
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None


def _device_of_handle(lib) -> int | None:
    """``hipGetDevice`` on an already-open handle, without disturbing it."""
    try:
        fn = lib.hipGetDevice
    except AttributeError:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    dev = ctypes.c_int(-1)
    if int(fn(ctypes.byref(dev))) != 0:
        return None
    return int(dev.value)


class UncachedIpcHeap:
    """Allocate uncached device memory and share it across ranks via HIP IPC."""

    _HIP_IPC_HANDLE_BYTES = 64
    _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
    # hipExtMallocWithFlags modes. Uncached is right on xGMI, where the peer
    # aperture has no penalty and skipping the caches keeps the handshake
    # simple. It is catastrophic on PCIe: peer writes into uncached memory
    # serialize per destination, so bandwidth collapses as the number of peers
    # written grows (measured on MI350P: 55 GB/s to 1 peer, 4.45 to 2, 1.44 to
    # 3, against 33.5 GB/s fine-grained). See docs/qr_int4_mi350p.md.
    _HIP_DEVICE_MALLOC_DEFAULT = 0x0
    _HIP_DEVICE_MALLOC_FINEGRAINED = 0x1
    _HIP_DEVICE_MALLOC_UNCACHED = 0x3
    _HIP_MEMCPY_HOST_TO_DEVICE = 1
    _hip = None
    _hipIpcMemHandle_t = None
    _hipPointerAttribute_t = None

    @classmethod
    def _load_hip(cls):
        if cls._hip is not None:
            return cls._hip
        # Several byte-identical copies of the runtime are typically already
        # mapped (see ``_mapped_hip_paths``), and they do not share device
        # state, so "already mapped" does not identify the right one. Pick the
        # copy that reports the device torch is bound to; re-opening a mapped
        # path is free and adds no new copy.
        mapped = _mapped_hip_paths()
        want = _torch_current_device()
        if want is not None:
            for path in mapped:
                try:
                    lib = ctypes.CDLL(path)
                except OSError:
                    continue
                if _device_of_handle(lib) == want:
                    cls._hip = lib
                    break
            else:
                logger.debug(
                    "no mapped libamdhip64 reports torch's device %d; "
                    "falling back to soname resolution",
                    want,
                )
        if cls._hip is None:
            for name in (
                *mapped,
                "libamdhip64.so",
                "libamdhip64.so.7",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
            ):
                try:
                    cls._hip = ctypes.CDLL(name)
                    break
                except OSError:
                    continue
        if cls._hip is None:
            raise RuntimeError("Failed to load HIP runtime library")

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * cls._HIP_IPC_HANDLE_BYTES)]

        class hipPointerAttribute_t(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),
                ("device", ctypes.c_int),
                ("devicePointer", ctypes.c_void_p),
                ("hostPointer", ctypes.c_void_p),
                ("isManaged", ctypes.c_int),
                ("allocationFlags", ctypes.c_uint),
            ]

        cls._hipIpcMemHandle_t = hipIpcMemHandle_t
        cls._hipPointerAttribute_t = hipPointerAttribute_t

        cls._hip.hipGetDevice.restype = ctypes.c_int
        cls._hip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cls._hip.hipPointerGetAttributes.restype = ctypes.c_int
        cls._hip.hipPointerGetAttributes.argtypes = [
            ctypes.POINTER(hipPointerAttribute_t),
            ctypes.c_void_p,
        ]

        cls._hip.hipIpcGetMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(hipIpcMemHandle_t),
            ctypes.c_void_p,
        ]
        cls._hip.hipIpcOpenMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            hipIpcMemHandle_t,
            ctypes.c_uint,
        ]
        cls._hip.hipIpcCloseMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        cls._hip.hipGetErrorString.restype = ctypes.c_char_p
        cls._hip.hipGetErrorString.argtypes = [ctypes.c_int]
        cls._hip.hipExtMallocWithFlags.restype = ctypes.c_int
        cls._hip.hipExtMallocWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cls._hip.hipFree.restype = ctypes.c_int
        cls._hip.hipFree.argtypes = [ctypes.c_void_p]
        cls._hip.hipMemset.restype = ctypes.c_int
        cls._hip.hipMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        cls._hip.hipMemcpy.restype = ctypes.c_int
        cls._hip.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        return cls._hip

    @classmethod
    def _hip_check(cls, err: int, *, what: str):
        if int(err) == 0:
            return
        hip = cls._load_hip()
        try:
            s = hip.hipGetErrorString(int(err))
            msg = s.decode("utf-8", errors="replace") if s else f"hipError({err})"
        except Exception:  # noqa: BLE001
            msg = f"hipError({err})"
        raise RuntimeError(f"{what} failed: {msg}")

    @classmethod
    def get_mem_handle_bytes(cls, base_ptr: int) -> bytes:
        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()
        err = hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(h), cls._HIP_IPC_HANDLE_BYTES))

    @classmethod
    def open_mem_handle(cls, handle_bytes: bytes) -> int:
        if len(handle_bytes) != cls._HIP_IPC_HANDLE_BYTES:
            raise ValueError(f"Expected {cls._HIP_IPC_HANDLE_BYTES}B handle")
        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(h), bytes(handle_bytes), cls._HIP_IPC_HANDLE_BYTES)
        out_ptr = ctypes.c_void_p()
        err = hip.hipIpcOpenMemHandle(
            ctypes.byref(out_ptr),
            h,
            ctypes.c_uint(int(cls._HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)),
        )
        cls._hip_check(err, what="hipIpcOpenMemHandle")
        return int(out_ptr.value)

    @classmethod
    def close_mem_handle(cls, base_ptr: int) -> None:
        hip = cls._load_hip()
        err = hip.hipIpcCloseMemHandle(ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcCloseMemHandle")

    @classmethod
    def current_device(cls) -> int | None:
        """HIP's current device on this thread, or ``None`` if unreadable."""
        hip = cls._load_hip()
        dev = ctypes.c_int(-1)
        if int(hip.hipGetDevice(ctypes.byref(dev))) != 0:
            return None
        return int(dev.value)

    @classmethod
    def device_of(cls, ptr: int) -> int | None:
        """Device index owning *ptr*, or ``None`` if HIP cannot report it."""
        hip = cls._load_hip()
        attr = cls._hipPointerAttribute_t()
        err = hip.hipPointerGetAttributes(
            ctypes.byref(attr), ctypes.c_void_p(int(ptr))
        )
        if int(err) != 0:
            return None
        return int(attr.device)

    @classmethod
    def _check_device(cls, ptr: int, expected: int, *, size: int) -> None:
        """Fail loudly if an allocation did not land on *expected*.

        Checked against the allocation itself rather than ``hipGetDevice``: if
        a second HIP runtime got loaded (see ``_mapped_hip_paths``) the current
        device reads 0 while the allocation is still correct, so the pointer's
        own attributes are the only trustworthy signal. A wrong-device inbox is
        functionally correct and only shows up as lost bandwidth, so it has to
        be an error rather than a warning.
        """
        got = cls.device_of(ptr)
        if got is None:
            logger.debug("hipPointerGetAttributes failed; skipping device check")
            return
        if got == expected:
            return
        cls.free_device_mem(ptr)
        raise RuntimeError(
            f"IPC allocation of {size} B landed on device {got}, expected "
            f"{expected} (HIP current device reports {cls.current_device()}). "
            "Every peer write would cross to the wrong GPU. Most likely a "
            "second libamdhip64 copy is loaded, or the device was never bound "
            "before this allocation."
        )

    @classmethod
    def alloc(
        cls, size: int, flags: int | None = None, *, expected_device: int | None = None
    ) -> int:
        """Zeroed device allocation, IPC-shareable, in the given memory mode.

        *flags* is a ``hipExtMallocWithFlags`` mode; ``None`` means uncached.
        Only uncached and fine-grained are used in practice -- coarse-grained
        (``_HIP_DEVICE_MALLOC_DEFAULT``) additionally requires the kernel's
        payload loads to bypass L2 (``sc0 sc1`` rather than the current ``nt``,
        which is only a hint) and buys nothing over fine-grained on PCIe.

        *expected_device*, when given, is the device index the allocation has
        to land on; a mismatch raises rather than silently costing bandwidth.
        """
        hip = cls._load_hip()
        if flags is None:
            flags = cls._HIP_DEVICE_MALLOC_UNCACHED
        buf = ctypes.c_void_p()
        err = hip.hipExtMallocWithFlags(
            ctypes.byref(buf),
            ctypes.c_size_t(size),
            ctypes.c_uint(int(flags)),
        )
        cls._hip_check(err, what=f"hipExtMallocWithFlags(flags={int(flags):#x})")
        if expected_device is not None:
            cls._check_device(int(buf.value), int(expected_device), size=size)
        err = hip.hipMemset(buf, 0, ctypes.c_size_t(size))
        cls._hip_check(err, what="hipMemset")
        return int(buf.value)

    @classmethod
    def alloc_uncached(cls, size: int, *, expected_device: int | None = None) -> int:
        return cls.alloc(
            size, cls._HIP_DEVICE_MALLOC_UNCACHED, expected_device=expected_device
        )

    @classmethod
    def copy_host_to_device(cls, dst_ptr: int, src, nbytes: int) -> None:
        hip = cls._load_hip()
        err = hip.hipMemcpy(
            ctypes.c_void_p(int(dst_ptr)),
            src,
            ctypes.c_size_t(nbytes),
            ctypes.c_int(cls._HIP_MEMCPY_HOST_TO_DEVICE),
        )
        cls._hip_check(err, what="hipMemcpy")

    @classmethod
    def free_device_mem(cls, ptr: int) -> None:
        hip = cls._load_hip()
        err = hip.hipFree(ctypes.c_void_p(ptr))
        cls._hip_check(err, what="hipFree")

    @staticmethod
    def gather_object_list_via_broadcast(group, shard_data):
        """All-gather Python objects over ``group``.

        Only torch.distributed surface. Index by group-local rank; ``src`` is
        the matching entry in unsorted ``get_process_group_ranks``.
        """
        import torch.distributed as dist

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        all_data = [[None] for _ in range(world_size)]
        all_data[rank][0] = shard_data
        ranks = dist.get_process_group_ranks(group=group)
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=r, group=group, device="cpu")
        return [all_data[i][0] for i in range(world_size)]
