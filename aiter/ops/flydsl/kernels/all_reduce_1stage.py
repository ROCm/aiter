# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the exact one-shot (1-stage) all-reduce.

Public type ``OneShotAllReduce``. Decode-only by policy: one round and no
grid-wide barrier, paid for with ``(N-1)*S`` of wire volume against the
two-shot's ``2(N-1)/N*S``. That is a 4x increase at TP8, irrelevant at 14 KiB
and decisive by ~1 MiB, so ``MAX_PAYLOAD_BYTES`` gates it rather than leaving a
caller to discover the crossover.

Everything around the kernel -- the IPC inbox, the peer-pointer table, the
per-block colours -- is ``qr_int4``'s ``_StEngine``, reused as-is. See
docs/all_reduce_1stage.md.
"""

from __future__ import annotations

import logging

import flydsl.compiler as flyc
import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .all_reduce_1stage_kernel import (
    DEFAULT_ATOMS,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
    SUPPORTED_ATOMS,
    make_all_reduce_1stage_kernel,
)
from .qr_int4 import (
    _SUPPORTED_ARCHS,
    _cuda_index,
    _resolve_inbox_flags,
    _validate_ipc_process_group,
    _StEngine,
)
from .qr_int4_kernel import SUPPORTED_WORLDS

logger = logging.getLogger("aiter")

# Largest payload this kernel should be asked to move.
#
# A speed policy, not an accuracy one -- the kernel is bit-comparable with
# ``cross_device_reduce`` at every size. One-shot pushes the whole payload to
# every peer, so wire volume is ``(N-1)*S`` against the two-shot's
# ``2(N-1)/N*S``: 7S vs 1.75S at TP8, 3S vs 1.5S at TP4. Below ~100 KiB the
# round-trip saving dominates and the extra bytes are free; well above it they
# are not.
#
# 256 KiB is a provisional boundary chosen to cover the decode range of interest
# (M<=16 at hidden 7168 is 224 KiB) and stop short of where the extra volume can
# plausibly matter. It has NOT been measured -- the crossover sweep is a tuning
# task, and until it is run this number should be treated as a guard rail rather
# than a tuned threshold.
MAX_PAYLOAD_BYTES = 256 << 10


class OneShotAllReduce:
    """IPC inbox + launch wrapper for ``all_reduce_1stage``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange,
    the same constraint ``QRInt4`` has and for the same reason.

    ``atoms`` and ``grid_cap`` are the tuning surface. Both change the block
    count for a given payload, which is the knob the TP8 data says is *not*
    obviously important -- ``cross_device_reduce`` runs 7-8x more blocks than
    the naive path at no measurable cost -- so they are exposed to be swept
    rather than pinned to a guess. See docs/all_reduce_1stage.md §6.3.

    ``inbox_memory`` follows ``QRInt4``: ``"auto"`` picks ``uncached`` on xGMI
    hosts and ``finegrained`` on PCIe ones from the KFD topology, because
    MI350X and MI350P both report ``gfx950`` and want opposite answers.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        atoms: int = DEFAULT_ATOMS,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        fanout: str = DEFAULT_FANOUT,
        max_bytes: int | None = None,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if atoms not in SUPPORTED_ATOMS:
            raise ValueError(f"atoms must be one of {SUPPORTED_ATOMS}, got {atoms!r}")
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
                f"OneShotAllReduce supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        torch.cuda.set_device(device)
        self.group = group
        self.device = device
        self._device_index = _cuda_index(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.atoms = int(atoms)
        self.inbox_memory = resolved_inbox
        self.fanout = fanout
        self.max_bytes = MAX_PAYLOAD_BYTES if max_bytes is None else int(max_bytes)

        spec = make_all_reduce_1stage_kernel(
            world_size=self.world_size,
            atoms=self.atoms,
            grid=cap,
            inbox_memory=resolved_inbox,
            fanout=fanout,
        )
        self._eng = _StEngine(
            spec=spec,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            inbox_flags=inbox_flags,
        )
        self.tile_bytes = spec["tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.buf_bytes = self._eng.buf_bytes
        self.grid_cap = cap

    def _num_tiles(self, live_bytes: int) -> int:
        return max(1, (live_bytes + self.tile_bytes - 1) // self.tile_bytes)

    def _grid_x(self, num_tiles: int) -> int:
        return max(1, min(num_tiles, self.grid_cap))

    def _check_payload(self, inp, out) -> int:
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise TypeError("OneShotAllReduce requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("OneShotAllReduce supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("OneShotAllReduce requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("OneShotAllReduce requires contiguous input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        return live_bytes

    def _launch(self, inp, out, stream) -> None:
        live_bytes = int(inp.numel()) * int(inp.element_size())
        num_tiles = self._num_tiles(live_bytes)
        grid_x = self._grid_x(num_tiles)
        if stream is None:
            stream = Stream(None)
        args = (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int64(int(self._eng._gpu_peer_ptrs)),
            Int64(int(self._eng._colors)),
            Int32(grid_x),
            stream,
        )
        if self._eng.compiled is None:
            # flyc.compile also launches, so this path is a real collective.
            self._eng.compiled = flyc.compile(self._eng.launch, *args)
        else:
            self._eng.compiled(*args)

    def compile(self, inp, out, stream=None) -> None:
        """Eager-JIT the binary. A real collective -- every rank must call it
        with the same shape, and ``out`` is overwritten."""
        self._check_payload(inp, out)
        self._launch(inp, out, stream)

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def allreduce(self, inp, out, stream=None):
        live_bytes = self._check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduce got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling: this kernel pushes the whole "
                "payload to every peer, so its wire volume is (N-1)x the "
                "message where a two-shot moves 2(N-1)/N. Route large messages "
                "to QRInt4 or cross_device_reduce, or pass max_bytes to "
                "override."
            )
        self._launch(inp, out, stream)

    def close(self):
        if self._eng is not None:
            self._eng.close()
            self._eng = None
