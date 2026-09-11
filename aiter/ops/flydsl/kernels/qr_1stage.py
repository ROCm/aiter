# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the exact one-shot (1-stage) all-reduce.

Public type ``OneShotAllReduce``. Decode-only by policy: one round and no
grid-wide barrier, paid for with ``(N-1)*S`` of wire volume against the
mesh's ``2(N-1)/N*S``.
"""

from __future__ import annotations

import logging

import flydsl.compiler as flyc
import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .qr_1stage_kernel import (
    DEFAULT_ATOMS,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
    DEFAULT_SPIN_SLEEP,
    SUPPORTED_ATOMS,
    make_qr_1stage_kernel,
    oneshot_ladder,
)
from .qr_ar_policy import FAMILY_POLICY
from .qr_int4 import (
    _SUPPORTED_ARCHS,
    _cuda_index,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    kernel_symbol,
)
from .qr_int_shared import SUPPORTED_WORLDS

logger = logging.getLogger("aiter")

# Largest payload this kernel should be asked to move, per world size.
MAX_PAYLOAD_BYTES_BY_WORLD = {
    ws: FAMILY_POLICY[("pcie", ws)].oneshot_max_exact for ws in SUPPORTED_WORLDS
}


def max_payload_bytes(world_size: int) -> int:
    return MAX_PAYLOAD_BYTES_BY_WORLD[int(world_size)]


class OneShotAllReduce:
    """IPC inbox + launch wrapper for ``qr_1stage``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange,
    the same constraint ``QRInt4`` has and for the same reason.

    ``atoms``, ``grid_cap`` and ``fanout`` are the tuning surface.

    ``max_bytes`` is the payload above which ``allreduce`` refuses to run,
    defaulting to this world size's entry in ``MAX_PAYLOAD_BYTES_BY_WORLD``.
    That ceiling is the point of the class: wire volume is ``(N-1)*S`` against a
    two-shot's ``2(N-1)/N*S``, so where it stops paying is a function of ``N``.

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
        atoms: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        fanout: str | None = None,
        max_bytes: int | None = None,
        probe: str = "full",
        spin_sleep: int = DEFAULT_SPIN_SLEEP,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        pinned = atoms is not None or grid_cap is not None or fanout is not None
        if atoms is None:
            atoms = DEFAULT_ATOMS
        if fanout is None:
            fanout = DEFAULT_FANOUT
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
        self.inbox_memory = resolved_inbox
        self.max_bytes = (
            max_payload_bytes(world_size) if max_bytes is None else int(max_bytes)
        )
        self.probe = probe
        self.spin_sleep = int(spin_sleep)

        if pinned:
            self._ladder = ((0, int(atoms), cap, fanout),)
        else:
            ceiling = cap if grid_cap is not None else None
            self._ladder = tuple(
                (floor, a, rung_cap if ceiling is None else min(rung_cap, ceiling), f)
                for floor, a, rung_cap, f in oneshot_ladder(world_size)
            )

        # One engine per distinct rung config, built in a fixed sorted order:
        # each does its own IPC handle exchange, which is a collective, so ranks
        # disagreeing on the order would deadlock.
        self._by_cfg = {}
        for _floor, a, c, f in self._ladder:
            key = (int(a), int(c), f)
            if key in self._by_cfg:
                continue
            spec = make_qr_1stage_kernel(
                world_size=self.world_size,
                atoms=key[0],
                grid=key[1],
                inbox_memory=resolved_inbox,
                fanout=key[2],
                probe=probe,
                spin_sleep=int(spin_sleep),
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

        # Lowest rung's shape, reported as this object's own. With a pinned
        # config that is the only rung and these are exact; with a ladder they
        # describe the smallest payloads, which is what a caller inspecting
        # ``tile_bytes`` is almost always asking about.
        first = (int(self._ladder[0][1]), int(self._ladder[0][2]), self._ladder[0][3])
        eng, spec = self._by_cfg[first]
        self.atoms = first[0]
        self.grid_cap = first[1]
        self.fanout = first[2]
        self.tile_bytes = spec["tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.buf_bytes = eng.buf_bytes

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every rung."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    def _pick_cfg(self, live_bytes: int):
        """``(atoms, grid_cap, fanout)`` the ladder assigns to *live_bytes*."""
        chosen = self._ladder[0]
        for rung in self._ladder:
            if live_bytes >= rung[0]:
                chosen = rung
        return (int(chosen[1]), int(chosen[2]), chosen[3])

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
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        return live_bytes

    def _launch_eng(self, eng, spec, inp, out, stream) -> None:
        live_bytes = int(inp.numel()) * int(inp.element_size())
        num_tiles = self._num_tiles(live_bytes, spec["tile_bytes"])
        grid_x = self._grid_x(num_tiles, spec["grid"])
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
            # flyc.compile also launches, so this path is a real collective.
            eng.compiled = flyc.compile(eng.launch, *args)
        else:
            eng.compiled(*args)

    def _launch(self, inp, out, stream) -> None:
        live_bytes = int(inp.numel()) * int(inp.element_size())
        eng, spec = self._by_cfg[self._pick_cfg(live_bytes)]
        self._launch_eng(eng, spec, inp, out, stream)

    def compile_and_launch(self, inp, out=None, stream=None) -> None:
        """Eager-JIT every rung's binary and launch each of them once, for
        real, against *inp*/*out*.

        This runs every rung on the GPU -- ``out`` ends up holding whichever
        rung ran last, and it is a real collective: every rank must call it
        with the same shape. Used by ``bench_comm_allreduce.py`` and the
        flydsl op tests to force a real warm launch (and, for the tests, to
        exercise the launch path directly) before timing or correctness
        checks begin. Production never calls this: it tolerates the first
        real call paying a JIT-compile cost instead.
        """
        if out is None:
            out = torch.empty_like(inp)
        self._check_payload(inp, out)
        for eng, spec in self._by_cfg.values():
            self._launch_eng(eng, spec, inp, out, stream)

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would run.

        ``<jit symbol>/g<grid_cap>/x<grid_x>``, matching ``QRInt4.variant``.
        Resolves the rung through the same ``_pick_cfg`` the launch path uses,
        so for a ladder-driven engine this is the only way to see which rung a
        given size takes.
        """
        cfg = self._pick_cfg(int(nbytes))
        eng, spec = self._by_cfg[cfg]
        grid_x = self._grid_x(self._num_tiles(int(nbytes), spec["tile_bytes"]), cfg[1])
        return f"{kernel_symbol(eng.launch)}/g{cfg[1]}/x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def allreduce(self, inp, out, stream=None):
        if self.probe != "full":
            raise RuntimeError(
                f"OneShotAllReduce was built with probe={self.probe!r}, a "
                "measurement-only variant that does not move the payload and "
                "computes a wrong answer. Use compile_and_launch()/_launch() "
                "to time it."
            )
        live_bytes = self._check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduce got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling: this kernel pushes the whole "
                "payload to every peer, so its wire volume is (N-1)x the "
                "message where a mesh schedule moves 2(N-1)/N. Route large messages "
                "to QRInt4 or cross_device_reduce, or pass max_bytes to "
                "override."
            )
        self._launch(inp, out, stream)

    def close(self):
        for eng, _ in self._by_cfg.values():
            eng.close()
        self._by_cfg = {}
