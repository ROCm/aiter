# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for lockstep quad-fanout INT4 two-shot all-reduce."""

from __future__ import annotations

import torch

from .qr_int4_ipc import UncachedIpcHeap
from .qr_int4_quad_fanout_kernel import (
    GRID,
    SUPER_TILES,
    TILE_BYTES,
    WORLD,
    make_qr_int4_quad_fanout_kernel,
)


def padded_nbytes(nbytes: int) -> int:
    """Round user payload up to the CodecQ4 tile quantum (32 KiB / 16384 fp16)."""
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    if nbytes == 0:
        return TILE_BYTES
    return ((nbytes + TILE_BYTES - 1) // TILE_BYTES) * TILE_BYTES


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(self, *, spec, group, device, rank: int, world_size: int):
        self.spec = spec
        self.launch = spec["launch"]
        self.super_tile = spec["super_tile"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self._peer_bases = [None] * world_size
        self._buf_ptr = UncachedIpcHeap.alloc_uncached(self.buf_bytes)
        my_handle = UncachedIpcHeap.get_mem_handle_bytes(self._buf_ptr)
        all_meta = UncachedIpcHeap.gather_object_list_via_broadcast(
            group, (my_handle, 0)
        )

        peer_ptrs = [0] * WORLD
        for r in range(world_size):
            handle, off = all_meta[r]
            if r == rank:
                peer_ptrs[r] = self._buf_ptr + off
            else:
                base = int(UncachedIpcHeap.open_mem_handle(bytes(handle)))
                self._peer_bases[r] = base
                peer_ptrs[r] = base + off
        for r in range(world_size, WORLD):
            peer_ptrs[r] = peer_ptrs[0]

        self._gpu_peer_ptrs = torch.tensor(peer_ptrs, dtype=torch.int64, device=device)
        self._colors = torch.ones(GRID, dtype=torch.int32, device=device)

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:  # noqa: S110
                    pass
        self._peer_bases = []
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:  # noqa: S110
                pass
            self._buf_ptr = None


class QRInt4QuadFanout:
    """IPC inbox + flag buffer and launch wrapper for ``qr_int4_quad_fanout``.

    Compile-time ``super_tile`` ST∈{1,2,4,8}. When ST>1 a ST=1 engine is also
    built so ``num_tiles ≤ GRID`` (decode) does not 2×/4×-flag the inbox.
    Pass ``force_super=True`` to glance the fat stride on small collectives.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        quant_dtype: str = "fp16",
        codec: str = "c16q4",
        super_tile: int = 1,
        force_super: bool = False,
    ):
        if world_size != WORLD:
            raise ValueError(
                f"only world_size={WORLD} is implemented, got {world_size}"
            )
        if super_tile not in SUPER_TILES:
            raise ValueError(
                f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}"
            )
        if quant_dtype not in ("bf16", "fp16"):
            raise ValueError(
                f"quant_dtype must be 'bf16' or 'fp16', got {quant_dtype!r}"
            )
        self.group = group
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.quant_dtype = quant_dtype
        self.codec = codec
        self.super_tile = int(super_tile)
        self.force_super = bool(force_super)
        self._compiled = False

        sts = [1]
        if self.super_tile != 1:
            sts.append(self.super_tile)
        self._by_st = {}
        for st in sts:
            spec = make_qr_int4_quad_fanout_kernel(
                world_size=self.world_size,
                quant_dtype=quant_dtype,
                codec=codec,
                super_tile=st,
            )
            self._by_st[st] = _StEngine(
                spec=spec,
                group=self.group,
                device=self.device,
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
        self.quant_dtype = primary.spec["quant_dtype"]
        self.codec = primary.spec["codec"]

    def _pick_st(self, num_tiles: int) -> int:
        if self.force_super or self.super_tile == 1 or num_tiles > GRID:
            return self.super_tile
        return 1

    def _launch_eng(
        self, eng: _StEngine, inp: torch.Tensor, out: torch.Tensor, stream
    ) -> None:
        from flydsl.expr.typing import Int32, Int64

        live_bytes = int(inp.numel()) * int(inp.element_size())
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        grid_x = min(num_tiles, GRID)
        if stream is None:
            stream = torch.cuda.current_stream()
        eng.launch(
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int64(int(eng._gpu_peer_ptrs.data_ptr())),
            Int64(int(eng._colors.data_ptr())),
            Int32(grid_x),
            stream=stream,
        )

    def compile(self, inp: torch.Tensor, out: torch.Tensor, stream=None) -> None:
        """Launch every ST engine so JIT finishes before the first wait.

        All ranks must participate. Compile-only / cache warming belongs in
        developer scripts, not this host or pytest.
        """
        for eng in self._by_st.values():
            self._launch_eng(eng, inp, out, stream)
        self._compiled = True

    def close(self):
        for eng in self._by_st.values():
            eng.close()
        self._by_st = {}

    def allreduce(self, inp: torch.Tensor, out: torch.Tensor, stream=None):
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("QRInt4QuadFanout supports bf16 input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles)
        self._launch_eng(self._by_st[st], inp, out, stream)
