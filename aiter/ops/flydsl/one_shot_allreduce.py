# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the exact one-shot (1-stage) all-reduce.

Public type ``OneShotAllReduce``. Decode-only by policy: one round and no
grid-wide barrier, paid for with ``(N-1)*S`` of wire volume against the
mesh's ``2(N-1)/N*S``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .allreduce_policy import FAMILY_POLICY
from .kernels.one_shot_allreduce import (
    DEFAULT_ATOMS,
    DEFAULT_BLOCK,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
    DEFAULT_SKIP_SELF,
    SUPPORTED_ATOMS,
    SUPPORTED_BLOCKS,
    make_one_shot_allreduce_kernel,
    oneshot_ladder,
)
from .kernels.quick_allreduce_shared import SUPPORTED_WORLDS
from .kernels.tensor_shim import _run_compiled
from .quick_allreduce_int4 import (
    _SUPPORTED_ARCHS,
    _cuda_index,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    has_xgmi_peer_links,
    kernel_symbol,
    payload_probes,
)

logger = logging.getLogger("aiter")

# Largest payload this kernel should be asked to move, per (link, world size).
MAX_PAYLOAD_BYTES = {
    (link, ws): FAMILY_POLICY[(link, ws)].oneshot_max_exact
    for link in ("pcie", "xgmi")
    for ws in SUPPORTED_WORLDS
}


def max_payload_bytes(world_size: int, link: str = "pcie") -> int:
    return MAX_PAYLOAD_BYTES[(str(link), int(world_size))]


class OneShotAllReduce:
    """IPC inbox + launch wrapper for ``one_shot_allreduce``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange,
    the same constraint ``QuickAllReduceInt4`` has and for the same reason.

    ``atoms``, ``grid_cap``, ``fanout`` and ``block`` are the tuning surface.
    ``atoms`` and ``block`` both set the tile width, ``grid_cap`` bounds it
    from above.

    ``max_bytes`` is the payload above which ``allreduce`` refuses to run,
    defaulting to this ``(link, world_size)``'s entry in ``MAX_PAYLOAD_BYTES``.
    That ceiling is the point of the class: wire volume is ``(N-1)*S`` against a
    two-shot's ``2(N-1)/N*S``, so where it stops paying is a function of ``N``.

    ``link`` selects both the tuning ladder and that default ceiling, and is
    detected from the KFD topology when not given.

    ``inbox_memory`` follows ``QuickAllReduceInt4``: ``"auto"`` picks ``uncached`` on xGMI
    hosts and ``finegrained`` on PCIe ones from the KFD topology, because
    MI350X and MI350P both report ``gfx950`` and want opposite answers.
    One exception: at TP2 it picks ``uncached`` on PCIe too, since a single
    remote destination cannot collapse.

    ``skip_self`` drops the round trip this rank does through its own inbox.
    It specialises the kernel to this rank, so the JIT symbol carries an ``_r<n>_``
    field and the binary is not shared across ranks -- one extra compile per process,
    not per world.
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
        skip_self: bool | None = None,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if link is None:
            link = "xgmi" if has_xgmi_peer_links() else "pcie"
        if link not in ("pcie", "xgmi"):
            raise ValueError(f"link must be 'pcie' or 'xgmi', got {link!r}")
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

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory, world_size)
        self._device_index = _cuda_index(device)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self._has_launched = False
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.inbox_memory = resolved_inbox
        self.max_bytes = (
            max_payload_bytes(world_size, link) if max_bytes is None else int(max_bytes)
        )

        # ``skip_self``: None means "whatever the rung says".
        ss = None if skip_self is None else bool(skip_self)
        if pinned:
            self._ladder = (
                (
                    0,
                    int(atoms),
                    cap,
                    fanout,
                    int(block),
                    DEFAULT_SKIP_SELF if ss is None else ss,
                ),
            )
        else:
            ceiling = cap if grid_cap is not None else None
            self._ladder = tuple(
                (
                    floor,
                    a,
                    rung_cap if ceiling is None else min(rung_cap, ceiling),
                    f,
                    b,
                    s if ss is None else ss,
                )
                for floor, a, rung_cap, f, b, s in oneshot_ladder(world_size, link)
            )

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
                    key = self._cfg_of(rung)
                    if key in self._by_cfg:
                        continue
                    spec = make_one_shot_allreduce_kernel(
                        world_size=self.world_size,
                        atoms=key[0],
                        grid=key[1],
                        inbox_memory=resolved_inbox,
                        fanout=key[2],
                        block=key[3],
                        skip_self=key[4],
                        rank=self.rank,
                    )
                    self._by_cfg[key] = (
                        _StEngine(
                            spec=spec,
                            group=group,
                            rank=self.rank,
                            world_size=self.world_size,
                            inbox_flags=inbox_flags,
                            device_index=self._device_index,
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
        first = self._cfg_of(self._ladder[0])
        eng, spec = self._by_cfg[first]
        self.atoms = first[0]
        self.grid_cap = first[1]
        self.fanout = first[2]
        self.block = first[3]
        self.skip_self = first[4]
        self.tile_bytes = spec["tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.buf_bytes = eng.buf_bytes

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every rung."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    @staticmethod
    def _cfg_of(rung) -> tuple:
        """A ladder rung's engine key: everything but its ``min_bytes``."""
        _floor, atoms, cap, fanout, block, skip_self = rung
        return (int(atoms), int(cap), fanout, int(block), bool(skip_self))

    def _pick_cfg(self, live_bytes: int) -> tuple:
        """``(atoms, grid_cap, fanout, block, skip_self)`` the ladder assigns to
        *live_bytes*."""
        chosen = self._ladder[0]
        for rung in self._ladder:
            if live_bytes >= rung[0]:
                chosen = rung
        return self._cfg_of(chosen)

    def _num_tiles(self, live_bytes: int, tile_bytes: int | None = None) -> int:
        tb = self.tile_bytes if tile_bytes is None else tile_bytes
        return max(1, (live_bytes + tb - 1) // tb)

    def _grid_x(self, num_tiles: int, grid_cap: int | None = None) -> int:
        return max(1, min(num_tiles, self.grid_cap if grid_cap is None else grid_cap))

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
        inp_ptr = int(inp.data_ptr())
        out_ptr = int(out.data_ptr())
        if inp_ptr % 16 != 0 or out_ptr % 16 != 0:
            raise ValueError("OneShotAllReduce requires 16-byte-aligned input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes > 0xFFFFFFFF:
            raise ValueError(
                "OneShotAllReduce payload must not exceed the 4 GiB buffer window"
            )
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
            raise ValueError("OneShotAllReduce requires non-overlapping input/output")
        return live_bytes

    def _launch_eng(self, eng, spec, inp, out, stream, *, live_bytes: int) -> None:
        num_tiles = self._num_tiles(live_bytes, spec["tile_bytes"])
        grid_x = self._grid_x(num_tiles, spec["grid"])
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
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
        floors = [rung[0] for rung in self._ladder]
        picked = {self._pick_cfg(n) for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def compile_and_launch(
        self, inp, out=None, stream=None, *, payload_range=None
    ) -> None:
        """Eager-JIT rung binaries and launch each of them once, for real,
        against *inp*/*out*.

        ``payload_range=(lo, hi)`` takes only the rungs ``allreduce`` would run
        for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes every one.
        The ladder builds engines for the whole size range, while a dispatcher
        routes only its own window here, so the rest never run.

        ``out`` ends up holding whichever rung ran last, and this is a real
        collective: every rank must call it with the same shape and range. Used
        by ``bench_comm_allreduce.py`` and the flydsl op tests to force a real
        warm launch (and, for the tests, to exercise the launch path directly)
        before timing or correctness checks begin.

        ``CustomAllreduce`` also calls it once at init, via ``warm_fly_engines``.
        Not for timing: it keeps each rung's JIT compile and module load off the
        first real all-reduce, which may be inside a CUDA graph capture.
        """
        if out is None:
            out = torch.empty_like(inp)
        live_bytes = self._check_payload(inp, out)
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        for key in keys:
            eng, spec = self._by_cfg[key]
            self._launch_eng(eng, spec, inp, out, stream, live_bytes=live_bytes)

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would run."""
        cfg = self._pick_cfg(int(nbytes))
        eng, spec = self._by_cfg[cfg]
        grid_x = self._grid_x(self._num_tiles(int(nbytes), spec["tile_bytes"]), cfg[1])
        return f"{kernel_symbol(eng.launch)}/grid_x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def allreduce(self, inp, out, stream=None):
        live_bytes = self._check_payload(inp, out)
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
