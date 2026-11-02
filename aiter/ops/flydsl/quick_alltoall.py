# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the gfx942/gfx950 TP∈{2,4,8} FlyDSL all-to-all.

Public type ``FlyQuickAllToAll``: an equal-split all-to-all, the semantics of
``torch.distributed.all_to_all_single`` without split sizes. The input is cut
into ``N`` equal chunks; chunk ``j`` goes to rank ``j`` and output chunk ``i``
comes from rank ``i``. Two schedules:

* ``"mesh"`` (default) -- every tile goes to all ``N-1`` peers at once.
* ``"ring"`` -- shifted pairwise: ``N-1`` steps, each to a single peer, so
  every store has one destination.

Which one is faster is a function of the fabric, the world size and the
payload; ``alltoall_policy`` owns that choice for the dispatcher.

Wire formats: ``None`` (register-direct, byte-exact, any dtype), ``"int4"``
and ``"int6"`` (group-16 E4M3-scaled, bf16 only). ``"fp16"`` is the lossless
LDS-staged path, for testing the transport.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .allreduce_host_shared import (
    _cuda_index,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    check_world_size,
    kernel_symbol,
    payload_probes,
    preload_launcher,
    require_arch,
)
from .allreduce_policy import resolve_link, select_rung
from .alltoall_policy import (
    A2A_CODECS,
    A2A_DIRECT_ORDERS,
    A2A_MESH_SUPER_TILES,
    A2A_QUANT_CODECS,
    A2A_RING_SUPER_TILES,
    DEFAULT_GRID_CAP,
    MIN_BATCH_BLOCKS,
    SUPPORTED_BLOCKS,
    TwoShotRung,
    mesh_st_ladder,
    ring_st_ladder,
)
from .kernels.comm.quick_allreduce_mesh import clamp_grid_cap
from .kernels.comm.quick_alltoall_mesh import make_quick_alltoall_mesh_kernel
from .kernels.comm.quick_alltoall_ring import make_quick_alltoall_ring_kernel
from .kernels.comm.transport import ATOMS, BLOCK, has_release_fence
from .kernels.tensor_shim import _run_compiled

logger = logging.getLogger("aiter")

# Dtypes the quantizing codecs read and write; the codec math is packed fp16.
_QUANT_DTYPES = (torch.bfloat16,)


@dataclass(frozen=True)
class _A2AAlgorithm:
    name: str
    build: Callable[..., dict]
    super_tiles: tuple[int, ...]
    default_super_tile: int
    st_ladder: Callable[[int, str], tuple[TwoShotRung, ...]]


ALGORITHMS = {
    "mesh": _A2AAlgorithm(
        name="mesh",
        build=make_quick_alltoall_mesh_kernel,
        super_tiles=A2A_MESH_SUPER_TILES,
        default_super_tile=1,
        st_ladder=mesh_st_ladder,
    ),
    "ring": _A2AAlgorithm(
        name="ring",
        build=make_quick_alltoall_ring_kernel,
        super_tiles=A2A_RING_SUPER_TILES,
        default_super_tile=8,
        st_ladder=ring_st_ladder,
    ),
}
CODEC_CHOICES = (None, *A2A_CODECS)


def resolve_codec(codec: str | None) -> str:
    """``None`` and ``"none"`` both name the register-direct, unquantized wire."""
    codec = "none" if codec is None else str(codec).lower()
    if codec not in A2A_CODECS:
        raise ValueError(f"codec must be one of {CODEC_CHOICES}, got {codec!r}")
    return codec


class FlyQuickAllToAll:
    """IPC inbox and launch wrapper for the FlyDSL equal-split all-to-all.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``algorithm`` -- ``"mesh"`` (default) or ``"ring"``. ``codec`` -- ``None`` (lossless, register-direct),
    ``"int4"`` or ``"int6"``; ``"fp16"`` is the lossless LDS-staged transport.

    ``inbox_memory`` and ``link`` mean what they mean for
    ``FlyQuickAllReduce``. ``super_tile``, ``grid_cap`` and ``block`` each
    override that field on every rung of the tuning ladder
    (``alltoall_policy``); ``ladder`` replaces the ladder outright, for tests
    and tuning runs. ``order``
    picks the register-direct store order of the mesh (``"peer"``: one
    destination's atoms back to back; ``"atom"``: interleaved).

    ``min_bytes`` is the payload below which :meth:`all_to_all` refuses to run;
    0 by default -- unlike the all-reduce, a lossless all-to-all has no
    accuracy reason to refuse small payloads, only a speed one, which is the
    dispatcher's call.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        algorithm: str = "mesh",
        codec: str | None = None,
        super_tile: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        batch_publishes: bool | None = None,
        link: str | None = None,
        block: int | None = None,
        order: str = "peer",
        min_bytes: int = 0,
        ladder: tuple[TwoShotRung, ...] | None = None,
    ):
        check_world_size(world_size)
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {tuple(ALGORITHMS)}, got {algorithm!r}"
            )
        algo = ALGORITHMS[algorithm]
        codec = resolve_codec(codec)
        link = resolve_link(link)
        if block is not None and block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        if order not in A2A_DIRECT_ORDERS:
            raise ValueError(f"order must be one of {A2A_DIRECT_ORDERS}, got {order!r}")
        if super_tile is not None and super_tile not in algo.super_tiles:
            raise ValueError(
                f"super_tile must be one of {algo.super_tiles} for "
                f"algorithm={algorithm!r}, got {super_tile!r}"
            )
        if grid_cap is not None and int(grid_cap) < 1:
            raise ValueError(f"grid_cap must be positive, got {grid_cap}")
        if int(min_bytes) < 0:
            raise ValueError(f"min_bytes must be non-negative, got {min_bytes}")
        _validate_ipc_process_group(
            group, rank=int(rank), world_size=world_size, owner="FlyQuickAllToAll"
        )
        arch = require_arch(owner="FlyQuickAllToAll")

        if ladder is None:
            ladder = algo.st_ladder(int(world_size), link) or (
                TwoShotRung(0, algo.default_super_tile, DEFAULT_GRID_CAP, BLOCK),
            )
        if not ladder:
            raise ValueError("ladder must have at least one rung")
        # A pinned field overrides that field on every rung, and a pinned
        # super-tile collapses the ladder to its first rung: the rungs differ
        # mainly in super-tile, so keeping them would only build duplicates.
        if super_tile is not None:
            ladder = ladder[:1]
        ladder = tuple(
            TwoShotRung(
                r.min_bytes,
                r.super_tile if super_tile is None else int(super_tile),
                r.grid_cap if grid_cap is None else int(grid_cap),
                r.block if block is None else int(block),
            )
            for r in ladder
        )
        inbox_flags, resolved_inbox = _resolve_inbox_flags(
            inbox_memory, world_size, link
        )

        self._device_index = _cuda_index(device, owner="FlyQuickAllToAll")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.link = link
        self.algorithm = algorithm
        self.codec = codec
        self.order = order
        self.inbox_memory = resolved_inbox
        self.min_bytes = int(min_bytes)
        self._has_launched = False
        self._batch_publishes = (
            has_release_fence(resolved_inbox) or algorithm == "ring"
            if batch_publishes is None
            else bool(batch_publishes)
        )
        cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )
        lds_capacity = get_lds_capacity_bytes(arch)

        # One engine per distinct ``(super_tile, block)``, plus each block's
        # ST=1 fallback, built in a fixed order because each engine's IPC
        # handle exchange is a collective. See ``FlyQuickAllReduce`` for why
        # the fallback takes the smallest cap among its block's rungs.
        caps = {}
        for r in ladder:
            caps.setdefault((r.super_tile, r.block), r.grid_cap)
        for b in {b for _st, b in list(caps)}:
            caps.setdefault((1, b), min(c for (_st, cb), c in caps.items() if cb == b))
        self._ladder = ladder
        self._by_cfg = {}
        try:
            with torch.cuda.device(self._device_index):
                for key in sorted(caps):
                    st, b = key
                    # A persistent kernel deadlocks if it launches more
                    # workgroups than fit, and the ranks have to agree on the
                    # number: take the minimum across the group.
                    grid = clamp_grid_cap(
                        caps[key],
                        arch=arch,
                        world_size=self.world_size,
                        super_tile=st,
                        cu_count=cu_count,
                        block=b,
                    )
                    shared_grid = torch.tensor(grid, dtype=torch.int64)
                    dist.all_reduce(shared_grid, op=dist.ReduceOp.MIN, group=group)
                    spec = algo.build(
                        world_size=self.world_size,
                        rank=self.rank,
                        super_tile=st,
                        grid=int(shared_grid.item()),
                        inbox_memory=resolved_inbox,
                        codec=codec,
                        block=b,
                        order=order,
                    )
                    if spec["lds_bytes"] > lds_capacity:
                        raise ValueError(
                            f"all-to-all {algorithm} {codec} at block={b} needs "
                            f"{spec['lds_bytes']} B of LDS, over the "
                            f"{lds_capacity} B {arch} has"
                        )
                    self._by_cfg[key] = _StEngine(
                        spec=spec,
                        group=self.group,
                        rank=self.rank,
                        world_size=self.world_size,
                        inbox_flags=inbox_flags,
                    )
        except Exception:
            self.close()
            raise

        # The first rung's shape, reported as this object's own.
        self.super_tile, self.block = ladder[0].super_tile, ladder[0].block
        primary = self._by_cfg[(self.super_tile, self.block)]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on *this* rank, across every rung."""
        return sum(eng.buf_bytes for eng in self._by_cfg.values())

    @staticmethod
    def _num_tiles(live_bytes: int, block: int) -> int:
        """Tiles a *live_bytes* payload spans: a tile is ``ATOMS / N`` atoms of
        each of the ``N`` chunks, so ``ATOMS`` atoms of the whole payload, as
        in the all-reduce."""
        tile_bytes = int(block) * ATOMS * 16
        return max(1, (live_bytes + tile_bytes - 1) // tile_bytes)

    def _pick_cfg(self, live_bytes: int) -> tuple[tuple[int, int], int]:
        """Engine key for a *live_bytes* payload, and its tile count.

        The ladder chooses the rung, which fixes the block and so the tile
        count; the tile count then only has to confirm there is a whole
        super-tile to take, falling back to the same block's ST=1 engine when
        there is not. See ``FlyQuickAllReduce._pick_cfg``.
        """
        rung = select_rung(self._ladder, live_bytes)
        want, b = rung.super_tile, rung.block
        num_tiles = self._num_tiles(live_bytes, b)
        if want == 1:
            st = 1
        elif self._batch_publishes:
            st = want if num_tiles >= want else 1
        else:
            st = want if num_tiles > self._by_cfg[(want, b)].grid else 1
        return (st, b), num_tiles

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int) -> int:
        """Blocks to launch for *num_tiles* tiles under *super_tile*, at most
        the engine's compile-time *grid*. See ``FlyQuickAllReduce._grid_x``."""
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), MIN_BATCH_BLOCKS)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, grid))

    def _launch_args(self, eng: _StEngine, inp_ptr, out_ptr, stream, *, live_bytes):
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        # The kernel sees one chunk.
        chunk_bytes = int(live_bytes) // self.world_size
        num_tiles = max(1, (chunk_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        return (
            Int32(self.rank),
            Int64(chunk_bytes),
            Int32(num_tiles),
            Int64(inp_ptr),
            Int64(out_ptr),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(self._grid_x(num_tiles, eng.super_tile, eng.grid)),
            stream,
        )

    def _launch_eng(self, eng: _StEngine, inp, out, stream, *, live_bytes: int):
        args = self._launch_args(
            eng,
            int(inp.data_ptr()),
            int(out.data_ptr()),
            stream,
            live_bytes=live_bytes,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch, *args)

    def cfgs_for(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """``_by_cfg`` keys a payload of ``lo..hi`` bytes (inclusive) can
        select, in build order."""
        floors = [rung.min_bytes for rung in self._ladder]
        picked = {self._pick_cfg(n)[0] for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def preload(self, *, payload_range=None) -> None:
        """JIT-compile engine binaries and load them onto the device, without
        launching any of them.

        ``payload_range=(lo, hi)`` takes only the binaries ``all_to_all`` would
        run for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes every
        one. Local to this rank, not a collective. ``QuickAllToAll`` calls it
        once at init, via ``preload_fly_engines``, to keep JIT compiles and
        module loads out of HIP graph capture.
        """
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        with torch.cuda.device(self._device_index):
            for key in keys:
                eng = self._by_cfg[key]
                args = self._launch_args(eng, 0, 0, None, live_bytes=0)
                preload_launcher(eng.launch, *args)

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng in engines.values():
                eng.close()
            engines.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would actually run."""
        cfg, num_tiles = self._pick_cfg(int(nbytes))
        eng = self._by_cfg[cfg]
        grid_x = self._grid_x(num_tiles, eng.super_tile, eng.grid)
        return f"{kernel_symbol(eng.launch)}/grid_x{grid_x}"

    def check_payload(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> int:
        """Validate *inp* (and *out*, when given); return the payload's bytes.

        Raises on anything the kernel cannot serve. :meth:`supports` is the
        non-raising form for dispatchers, which ask before allocating *out*.
        """
        tensors = (inp,) if out is None else (inp, out)
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("FlyQuickAllToAll requires torch.Tensor input/output")
        if self.codec in A2A_QUANT_CODECS and any(
            t.dtype not in _QUANT_DTYPES for t in tensors
        ):
            raise ValueError(
                f"codec={self.codec!r} needs {_QUANT_DTYPES} input/output, got "
                f"{[t.dtype for t in tensors]}"
            )
        for t in tensors:
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(
                    f"FlyQuickAllToAll requires tensors on cuda:{self._device_index}, "
                    f"got {t.device}"
                )
            if not t.is_contiguous():
                raise ValueError("FlyQuickAllToAll requires contiguous input/output")
            if int(t.data_ptr()) % 16:
                raise ValueError(
                    "FlyQuickAllToAll requires 16-byte-aligned input/output"
                )
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % (16 * self.world_size):
            raise ValueError(
                f"byte size {live_bytes} must split into {self.world_size} chunks "
                "of a whole number of 16 B"
            )
        if live_bytes // self.world_size > 0xFFFFFFFF:
            raise ValueError("FlyQuickAllToAll chunks must not exceed 4 GiB")
        if out is not None:
            if int(out.numel()) * int(out.element_size()) != live_bytes:
                raise ValueError("inp/out byte size mismatch")
            inp_ptr, out_ptr = int(inp.data_ptr()), int(out.data_ptr())
            if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
                raise ValueError(
                    "FlyQuickAllToAll requires non-overlapping input/output"
                )
        return live_bytes

    def supports(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> bool:
        """Whether :meth:`all_to_all` would accept *inp* (and *out*). Never raises."""
        try:
            nbytes = self.check_payload(inp, out)
        except (TypeError, ValueError):
            return False
        return self.is_beneficial(nbytes)

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) >= self.min_bytes

    def all_to_all(self, inp: torch.Tensor, out: torch.Tensor, stream=None) -> None:
        """Equal-split all-to-all from *inp* into *out*.

        ``stream=None`` uses the current PyTorch stream on this device.
        """
        live_bytes = self.check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"FlyQuickAllToAll.all_to_all got a {live_bytes} B payload, below "
                f"the {self.min_bytes} B floor"
            )
        cfg, _num_tiles = self._pick_cfg(live_bytes)
        self._launch_eng(self._by_cfg[cfg], inp, out, stream, live_bytes=live_bytes)
