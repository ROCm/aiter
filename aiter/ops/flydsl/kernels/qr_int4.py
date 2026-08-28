# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce."""

from __future__ import annotations

import ctypes

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.dist.parallel_state import in_the_same_node_as
from aiter.jit.utils.chip_info import get_gfx_runtime

from .qr_int4_ipc import UncachedIpcHeap
from .qr_int4_kernel import (
    DEFAULT_GRID_CAP,
    SUPER_TILES,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    clamp_grid_cap,
    make_qr_int4_kernel,
)
from .tensor_shim import _run_compiled

_SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _cuda_index(device) -> int:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"QRInt4 requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _validate_ipc_process_group(group, *, rank: int) -> None:
    """Reject groups that cannot exchange HIP IPC handles or CPU-side metadata."""
    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"QRInt4 does not support NCCL process groups (got {backend!r} on "
            f"group rank {rank}): IPC handle exchange requires CPU-side "
            "broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            "QRInt4 does not support multi-node process groups: HIP IPC "
            f"handles are node-local (ranks not on rank 0's node: {off_node})."
        )


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(self, *, spec, group, rank: int, world_size: int):
        self.spec = spec
        self.launch = spec["launch"]
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self._peer_bases = [None] * world_size
        self._buf_ptr = None
        self._meta_ptr = None
        try:
            self._buf_ptr = UncachedIpcHeap.alloc_uncached(self.buf_bytes)
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
            color_bytes = self.grid * 4
            self._meta_ptr = UncachedIpcHeap.alloc_uncached(peer_bytes + color_bytes)
            UncachedIpcHeap.copy_host_to_device(
                self._meta_ptr,
                (ctypes.c_int64 * world_size)(*peer_ptrs),
                peer_bytes,
            )
            UncachedIpcHeap.copy_host_to_device(
                self._meta_ptr + peer_bytes,
                (ctypes.c_int32 * self.grid)(*([1] * self.grid)),
                color_bytes,
            )
        except Exception:
            self.close()
            raise

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:
                    pass
        self._peer_bases = []
        if self._meta_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._meta_ptr)
            except RuntimeError:
                pass
            self._meta_ptr = None
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


class QRInt4:
    """IPC inbox + flag buffer and launch wrapper for ``qr_int4``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        super_tile: int = 8,
        grid_cap: int | None = None,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if super_tile not in SUPER_TILES:
            raise ValueError(
                f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}"
            )
        group_world = dist.get_world_size(group=group)
        group_rank = dist.get_rank(group=group)
        if group_world != int(world_size):
            raise ValueError(
                f"world_size={world_size} does not match group size {group_world}"
            )
        if group_rank != int(rank):
            raise ValueError(f"rank={rank} does not match group rank {group_rank}")
        _validate_ipc_process_group(group, rank=int(rank))
        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(
                f"QRInt4 supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")
        torch.cuda.set_device(device)
        self.group = group
        self.device = device
        self._device_index = _cuda_index(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.super_tile = int(super_tile)
        cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )

        sts = [1]
        if self.super_tile != 1:
            sts.append(self.super_tile)
        self._by_st = {}
        try:
            for st in sts:
                grid = clamp_grid_cap(
                    cap,
                    arch=arch,
                    world_size=self.world_size,
                    super_tile=st,
                    cu_count=cu_count,
                )
                shared_grid = torch.tensor(grid, dtype=torch.int64)
                dist.all_reduce(shared_grid, op=dist.ReduceOp.MIN, group=group)
                spec = make_qr_int4_kernel(
                    world_size=self.world_size,
                    super_tile=st,
                    grid=int(shared_grid.item()),
                )
                self._by_st[st] = _StEngine(
                    spec=spec,
                    group=self.group,
                    rank=self.rank,
                    world_size=self.world_size,
                )
        except Exception:
            self.close()
            raise

        primary = self._by_st[self.super_tile]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    def _pick_st(self, num_tiles: int) -> int:
        if self.super_tile == 1:
            return 1
        st1 = self._by_st.get(1)
        if st1 is not None and num_tiles <= st1.grid:
            return 1
        return self.super_tile

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

    def _launch_args(
        self,
        eng: _StEngine,
        inp,
        out,
        stream,
        *,
        live_bytes: int,
        num_tiles: int,
        grid_x: int,
    ):
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int64(int(eng._meta_ptr)),
            Int64(int(eng._meta_ptr + self.world_size * 8)),
            Int32(grid_x),
            stream,
        )

    def _launch_eng(
        self,
        eng: _StEngine,
        inp,
        out,
        stream,
        *,
        live_bytes: int,
    ) -> None:
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        args = self._launch_args(
            eng,
            inp,
            out,
            stream,
            live_bytes=live_bytes,
            num_tiles=num_tiles,
            grid_x=min(num_tiles, eng.grid),
        )
        _run_compiled(eng.launch, *args)

    def compile(self, inp, out, stream=None) -> None:
        """Eager-JIT every ST binary.

        Default ST=8 also builds an ST=1 engine for small payloads.
        Every rank must call this with the same ``inp``/``out`` shape.
        ``out`` is overwritten.
        """
        live_bytes = self._check_payload(inp, out)
        for eng in self._by_st.values():
            self._launch_eng(eng, inp, out, stream, live_bytes=live_bytes)

    def close(self):
        for eng in self._by_st.values():
            eng.close()
        self._by_st = {}

    def allreduce(self, inp, out, stream=None):
        """Two-shot INT4 all-reduce into ``out``.

        ``stream=None`` uses the current PyTorch stream on this device.
        """
        live_bytes = self._check_payload(inp, out)
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles)
        eng = self._by_st[st]
        self._launch_eng(eng, inp, out, stream, live_bytes=live_bytes)
