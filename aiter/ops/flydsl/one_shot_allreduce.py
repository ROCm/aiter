# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the exact one-shot (1-stage) all-reduce.
"""

from __future__ import annotations

import logging

import torch
from flydsl.expr.typing import Int32, Int64, Stream

from .allreduce_policy import family_policy, oneshot_ladder, resolve_link
from .allreduce_shared import (
    _cuda_index,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    check_payload,
    check_world_size,
    kernel_symbol,
    payload_probes,
    require_arch,
)
from .kernels.comm.allreduce_shared import SUPPORTED_BLOCKS, OneShotRung, select_rung
from .kernels.comm.one_shot_allreduce import (
    DEFAULT_ATOMS,
    DEFAULT_BLOCK,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
    SUPPORTED_ATOMS,
    make_one_shot_allreduce_kernel,
)
from .kernels.tensor_shim import _preload_compiled, _run_compiled

logger = logging.getLogger("aiter")


def max_payload_bytes(
    world_size: int, link: str = "pcie", arch: str | None = None
) -> int:
    """Largest payload this kernel should be asked to move"""
    return family_policy(str(link), int(world_size), arch).oneshot_max


class OneShotAllReduce:
    """IPC inbox + launch wrapper for ``one_shot_allreduce``.

    ``atoms``, ``grid_cap``, ``fanout`` and ``block`` are the tuning surface.
    ``atoms`` and ``block`` both set the tile width, ``grid_cap`` bounds it
    from above.

    ``max_bytes`` is the payload above which this ``allreduce`` refuses to run.

    ``link`` is the interconnect type and it is detected from the KFD topology 
    when not explicitly given.

    ``inbox_memory`` type: ``"auto"`` picks ``uncached`` on xGMI
    hosts and ``finegrained`` on PCIe ones. One exception: at TP2 it picks 
    ``uncached`` on PCIe too, since a single remote destination cannot collapse.

    A rank never round-trips through its own inbox: its own contribution stays
    in registers. That specialises the kernel to this rank, so the JIT symbol
    carries an ``_r<n>_`` field and the binary is not shared across ranks.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        atoms: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        fanout: str | None = None,
        block: int | None = None,
        max_bytes: int | None = None,
        link: str | None = None,
    ):
        check_world_size(world_size)
        link = resolve_link(link)
        self.link = link
        pinned = (
            atoms is not None
            or grid_cap is not None
            or fanout is not None
            or block is not None
        )
        if atoms is None:
            atoms = DEFAULT_ATOMS
        if fanout is None:
            fanout = DEFAULT_FANOUT
        if block is None:
            block = DEFAULT_BLOCK
        if atoms not in SUPPORTED_ATOMS:
            raise ValueError(f"atoms must be one of {SUPPORTED_ATOMS}, got {atoms!r}")
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        _validate_ipc_process_group(
            group, rank=int(rank), world_size=world_size, owner="OneShotAllReduce"
        )
        require_arch(owner="OneShotAllReduce")
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory, world_size)
        self._device_index = _cuda_index(device, owner="OneShotAllReduce")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self._has_launched = False
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.inbox_memory = resolved_inbox
        self.max_bytes = (
            max_payload_bytes(world_size, link) if max_bytes is None else int(max_bytes)
        )

        if pinned:
            self._ladder = (
                OneShotRung(
                    min_bytes=0,
                    atoms=int(atoms),
                    grid_cap=cap,
                    fanout=fanout,
                    block=int(block),
                ),
            )
        else:
            self._ladder = oneshot_ladder(world_size, link)

        # One engine per distinct rung config, built in a fixed sorted order:
        # each does its own IPC handle exchange, which is a collective, so ranks
        # disagreeing on the order would deadlock.
        #
        # No ``clamp_grid_cap`` here, unlike the two-shot schedules: no rung
        # caps the grid above 128, which is more than an order of magnitude
        # under the residency of any supported device, so the clamp could only
        # ever be a no-op bought with an extra collective per engine.
        self._by_cfg = {}
        try:
            with torch.cuda.device(self._device_index):
                for rung in self._ladder:
                    key = rung.cfg
                    if key in self._by_cfg:
                        continue
                    spec = make_one_shot_allreduce_kernel(
                        world_size=self.world_size,
                        atoms=key[0],
                        grid=key[1],
                        inbox_memory=resolved_inbox,
                        fanout=key[2],
                        block=key[3],
                        rank=self.rank,
                    )
                    self._by_cfg[key] = (
                        _StEngine(
                            spec=spec,
                            group=group,
                            rank=self.rank,
                            world_size=self.world_size,
                            inbox_flags=inbox_flags,
                        ),
                        spec,
                    )
        except Exception:
            self.close()
            raise

        # Lowest rung's shape, reported as this object's own. With a pinned
        # config that is the only rung and these are exact; with a ladder they
        # describe the smallest payloads, which is what a caller inspecting
        # ``tile_bytes`` is almost always asking about.
        first = self._ladder[0].cfg
        eng, spec = self._by_cfg[first]
        self.atoms = first[0]
        self.grid_cap = first[1]
        self.fanout = first[2]
        self.block = first[3]
        self.tile_bytes = spec["tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.buf_bytes = eng.buf_bytes

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every rung."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    def _pick_cfg(self, live_bytes: int) -> tuple:
        """``(atoms, grid_cap, fanout, block)`` the ladder assigns to
        *live_bytes*."""
        return select_rung(self._ladder, live_bytes).cfg

    def _num_tiles(self, live_bytes: int, tile_bytes: int | None = None) -> int:
        tb = self.tile_bytes if tile_bytes is None else tile_bytes
        return max(1, (live_bytes + tb - 1) // tb)

    def _grid_x(self, num_tiles: int, grid_cap: int | None = None) -> int:
        return max(1, min(num_tiles, self.grid_cap if grid_cap is None else grid_cap))

    def _launch_args(self, eng, spec, inp_ptr, out_ptr, stream, *, live_bytes):
        num_tiles = self._num_tiles(live_bytes, spec["tile_bytes"])
        grid_x = self._grid_x(num_tiles, spec["grid"])
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(inp_ptr),
            Int64(out_ptr),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(grid_x),
            stream,
        )

    def _launch_eng(self, eng, spec, inp, out, stream, *, live_bytes: int) -> None:
        args = self._launch_args(
            eng,
            spec,
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

    def _launch(self, inp, out, stream, *, live_bytes: int) -> None:
        eng, spec = self._by_cfg[self._pick_cfg(live_bytes)]
        self._launch_eng(eng, spec, inp, out, stream, live_bytes=live_bytes)

    def cfgs_for(self, lo: int, hi: int) -> list[tuple]:
        """``_by_cfg`` keys a payload of ``lo..hi`` bytes (inclusive) can
        select, in build order."""
        floors = [rung.min_bytes for rung in self._ladder]
        picked = {self._pick_cfg(n) for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def preload(self, *, payload_range=None) -> None:
        """JIT-compile rung binaries without launching any of them.

        ``payload_range=(lo, hi)`` takes only the rungs ``allreduce`` would run
        for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes every one.
        The ladder builds engines for the whole size range, while a dispatcher
        routes only its own window here, so the rest never run.

        Local to this rank, not a collective. ``CustomAllreduce`` calls it once
        at init, via ``preload_fly_engines``, to keep each rung's JIT compile off
        the first real all-reduce, which may be inside a CUDA graph capture.
        """
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        for key in keys:
            eng, spec = self._by_cfg[key]
            args = self._launch_args(eng, spec, 0, 0, None, live_bytes=0)
            _preload_compiled(eng.launch, *args)

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would run."""
        cfg = self._pick_cfg(int(nbytes))
        eng, spec = self._by_cfg[cfg]
        grid_x = self._grid_x(self._num_tiles(int(nbytes), spec["tile_bytes"]), cfg[1])
        return f"{kernel_symbol(eng.launch)}/grid_x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def allreduce(self, inp, out, stream=None):
        live_bytes = check_payload(
            inp, out, device_index=self._device_index, owner="OneShotAllReduce"
        )
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduce got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling: this kernel pushes the whole "
                "payload to every peer, so its wire volume is (N-1)x the "
                "message where a mesh schedule moves 2(N-1)/N. Route large messages "
                "to QuickAllReduceInt4 or cross_device_reduce, or pass max_bytes to "
                "override."
            )
        self._launch(inp, out, stream, live_bytes=live_bytes)

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng, _ in engines.values():
                eng.close()
            engines.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return
