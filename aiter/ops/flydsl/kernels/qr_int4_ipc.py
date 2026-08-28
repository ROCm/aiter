# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Uncached HIP IPC helpers for the FlyDSL INT4 QuickReduce inbox."""

from __future__ import annotations

import ctypes


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

    @classmethod
    def _load_hip(cls):
        if cls._hip is not None:
            return cls._hip
        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                cls._hip = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if cls._hip is None:
            raise RuntimeError("Failed to load HIP runtime library")

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * cls._HIP_IPC_HANDLE_BYTES)]

        cls._hipIpcMemHandle_t = hipIpcMemHandle_t

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
    def alloc(cls, size: int, flags: int | None = None) -> int:
        """Zeroed device allocation, IPC-shareable, in the given memory mode.

        *flags* is a ``hipExtMallocWithFlags`` mode; ``None`` means uncached.
        Only uncached and fine-grained are used in practice -- coarse-grained
        (``_HIP_DEVICE_MALLOC_DEFAULT``) additionally requires the kernel's
        payload loads to bypass L2 (``sc0 sc1`` rather than the current ``nt``,
        which is only a hint) and buys nothing over fine-grained on PCIe.
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
        err = hip.hipMemset(buf, 0, ctypes.c_size_t(size))
        cls._hip_check(err, what="hipMemset")
        return int(buf.value)

    @classmethod
    def alloc_uncached(cls, size: int) -> int:
        return cls.alloc(size, cls._HIP_DEVICE_MALLOC_UNCACHED)

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
