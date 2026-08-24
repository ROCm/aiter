# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942 TP∈{2,4,8} INT4 two-shot all-reduce.

Public type ``QRInt4``. Super-tile ST∈{1,8}; ST=1 when ``num_tiles ≤ GRID``.
INT4 nibble + group-16 E4M3. Payload HBM is bf16.
"""

from __future__ import annotations

import ctypes

import flydsl.compiler as flyc
import torch
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .qr_int4_ipc import UncachedIpcHeap
from .qr_int4_kernel import (
    GRID,
    SUPER_TILES,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    make_qr_int4_kernel,
)

_SUPPORTED_ARCH = "gfx942"


def _cuda_index(device) -> int:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"QRInt4 requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(self, *, spec, group, rank: int, world_size: int):
        self.spec = spec
        self.launch = spec["launch"]
        self.compiled = None
        self.super_tile = spec["super_tile"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self._peer_bases = [None] * world_size
        self._buf_ptr = UncachedIpcHeap.alloc_uncached(self.buf_bytes)
        self._meta_ptr = None
        my_handle = UncachedIpcHeap.get_mem_handle_bytes(self._buf_ptr)
        all_meta = UncachedIpcHeap.gather_object_list_via_broadcast(
            group, (my_handle, 0)
        )

        peer_ptrs = [0] * world_size
        for r in range(world_size):
            handle, off = all_meta[r]
            if r == rank:
                peer_ptrs[r] = self._buf_ptr + off
            else:
                base = int(UncachedIpcHeap.open_mem_handle(bytes(handle)))
                self._peer_bases[r] = base
                peer_ptrs[r] = base + off

        peer_bytes = world_size * 8
        color_bytes = GRID * 4
        self._meta_ptr = UncachedIpcHeap.alloc_uncached(peer_bytes + color_bytes)
        self._gpu_peer_ptrs = self._meta_ptr
        self._colors = self._meta_ptr + peer_bytes
        UncachedIpcHeap.copy_host_to_device(
            self._gpu_peer_ptrs,
            (ctypes.c_int64 * world_size)(*peer_ptrs),
            peer_bytes,
        )
        UncachedIpcHeap.copy_host_to_device(
            self._colors,
            (ctypes.c_int32 * GRID)(*([1] * GRID)),
            color_bytes,
        )

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:  # noqa: S110
                    pass
        self._peer_bases = []
        if self._meta_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._meta_ptr)
            except RuntimeError:  # noqa: S110
                pass
            self._meta_ptr = None
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:  # noqa: S110
                pass
            self._buf_ptr = None


class QRInt4:
    """IPC inbox + flag buffer and launch wrapper for ``qr_int4``."""

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        super_tile: int = 8,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if super_tile not in SUPER_TILES:
            raise ValueError(
                f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}"
            )
        arch = get_gfx_runtime()
        if arch != _SUPPORTED_ARCH:
            raise RuntimeError(f"QRInt4 is {_SUPPORTED_ARCH}-only, got {arch}")
        torch.cuda.set_device(device)
        self.group = group
        self.device = device
        self._device_index = _cuda_index(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.super_tile = int(super_tile)

        sts = [1]
        if self.super_tile != 1:
            sts.append(self.super_tile)
        self._by_st = {}
        for st in sts:
            spec = make_qr_int4_kernel(
                world_size=self.world_size,
                super_tile=st,
            )
            self._by_st[st] = _StEngine(
                spec=spec,
                group=self.group,
                rank=self.rank,
                world_size=self.world_size,
            )

        primary = self._by_st[self.super_tile]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    def _pick_st(self, num_tiles: int) -> int:
        if self.super_tile == 1 or num_tiles > GRID:
            return self.super_tile
        return 1

    def _check_payload(self, inp, out) -> int:
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise TypeError("QRInt4 requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("QRInt4 supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("QRInt4 requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("QRInt4 requires contiguous input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        return live_bytes

    def _launch_eng(self, eng: _StEngine, inp, out, stream) -> None:
        live_bytes = int(inp.numel()) * int(inp.element_size())
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        grid_x = min(num_tiles, GRID)
        if stream is None:
            stream = Stream(None)
        args = (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(grid_x),
            stream,
        )
        if eng.compiled is None:
            eng.compiled = flyc.compile(eng.launch, *args)
        else:
            eng.compiled(*args)

    def compile(self, inp, out, stream=None) -> None:
        """Eager-JIT every ST binary. Optional: first ``allreduce`` JIT-compiles the picked ST.

        Default ST=8 also builds an ST=1 engine for ``num_tiles ≤ GRID``.
        Skipping this method is correct for a single size class: that
        ``allreduce`` calls ``flyc.compile`` for the chosen ST only, and a
        later size that picks the other ST JIT-compiles then.

        ``flyc.compile`` also launches, so this is a real collective: every
        rank must call it with the same ``inp``/``out`` shape. The warmup
        tensor may be small; we still launch every engine so a later
        prefill-sized ``allreduce`` does not JIT mid-collective. ``out`` is
        overwritten.
        """
        self._check_payload(inp, out)
        for eng in self._by_st.values():
            self._launch_eng(eng, inp, out, stream)

    def close(self):
        for eng in self._by_st.values():
            eng.close()
        self._by_st = {}

    def allreduce(self, inp, out, stream=None):
        live_bytes = self._check_payload(inp, out)
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles)
        self._launch_eng(self._by_st[st], inp, out, stream)
