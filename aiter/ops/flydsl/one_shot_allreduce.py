# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the exact one-shot (1-stage) all-reduce.

Public types ``OneShotAllReduce`` (plain) and ``OneShotAllReduceRMSNorm``
(fused with residual-add + RMSNorm). Decode-only by policy: one round and no
grid-wide barrier, paid for with ``(N-1)*S`` of wire volume against the
mesh's ``2(N-1)/N*S``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .allreduce_policy import FAMILY_POLICY
from .kernels.one_shot_allreduce import (
    DEFAULT_ATOMS,
    DEFAULT_BLOCK,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
    DEFAULT_SKIP_SELF,
    DEFAULT_SPIN_SLEEP,
    SUPPORTED_ATOMS,
    SUPPORTED_BLOCKS,
    fused_hidden_supported,
    fused_oneshot_ladder,
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
        probe: str = "full",
        spin_sleep: int = DEFAULT_SPIN_SLEEP,
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

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        # set_device rejects torch.device("cuda") with no index; resolve first.
        self._device_index = _cuda_index(device)
        torch.cuda.set_device(self._device_index)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self._has_launched = False
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.inbox_memory = resolved_inbox
        self.max_bytes = (
            max_payload_bytes(world_size, link)
            if max_bytes is None
            else int(max_bytes)
        )
        self.probe = probe
        self.spin_sleep = int(spin_sleep)

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
                    probe=probe,
                    spin_sleep=int(spin_sleep),
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
            # The fused epilogue's operands. One kernel signature serves both
            # modes (see the factory), and a plain build's ``const_expr(fused)``
            # arms never read these.
            Int64(0),
            Int64(0),
            Int64(0),
            Float32(0.0),
            stream,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        _run_compiled(eng.launch, *args)

    def _launch(self, inp, out, stream, *, live_bytes: int) -> None:
        eng, spec = self._by_cfg[self._pick_cfg(live_bytes)]
        self._launch_eng(eng, spec, inp, out, stream, live_bytes=live_bytes)

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
        live_bytes = self._check_payload(inp, out)
        for eng, spec in self._by_cfg.values():
            self._launch_eng(eng, spec, inp, out, stream, live_bytes=live_bytes)

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would run.

        ``<jit symbol>/g<grid_cap>/x<grid_x>``, matching
        ``QuickAllReduceInt4.variant``. Resolves the rung through the same ``_pick_cfg`` the launch path uses,
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
                "to QuickAllReduceInt4 or cross_device_reduce, or pass max_bytes to "
                "override."
            )
        self._launch(inp, out, stream, live_bytes=live_bytes)

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
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


class OneShotAllReduceRMSNorm:
    """One-shot all-reduce fused with residual-add and RMSNorm.

    Per token row:

        acc = sum_r input_r            (fp32, rank order)
        acc = float(bf16(acc))         (deliberate: matches the unfused path)
        acc += residual_in
        residual_out = bf16(acc)
        out = bf16(acc * rsqrt(sum(acc^2)/hidden + eps) * weight)

    Unlike ``OneShotAllReduce`` the wire layout depends on ``hidden``, because
    the tile is pinned to one token row so the norm's reduction fits in one
    workgroup. A new hidden therefore needs a new engine, and building one is a
    collective (IPC handle exchange). Engines are built lazily on first use,
    which is safe because TP ranks enter a collective with the same shape in
    lockstep; pass ``hiddens=(...)`` to build them up front instead.
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
        spin_sleep: int = DEFAULT_SPIN_SLEEP,
        hiddens: tuple[int, ...] = (),
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        pinned = atoms is not None or grid_cap is not None or fanout is not None
        if atoms is not None and atoms not in SUPPORTED_ATOMS:
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
                f"OneShotAllReduceRMSNorm supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        if grid_cap is not None and int(grid_cap) < 1:
            raise ValueError(f"grid_cap must be positive, got {grid_cap}")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        self._device_index = _cuda_index(device)
        torch.cuda.set_device(self._device_index)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self._has_launched = False
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.inbox_memory = resolved_inbox
        self._inbox_flags = inbox_flags
        self.max_bytes = (
            max_payload_bytes(world_size) if max_bytes is None else int(max_bytes)
        )
        self.spin_sleep = int(spin_sleep)

        # ``FUSED_ONESHOT_LADDER``: ``atoms`` sets tile
        # width in the plain schedule and block width here.
        if pinned:
            self._ladder = (
                (
                    0,
                    DEFAULT_ATOMS if atoms is None else int(atoms),
                    DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap),
                    DEFAULT_FANOUT if fanout is None else fanout,
                ),
            )
        else:
            ceiling = None if grid_cap is None else int(grid_cap)
            self._ladder = tuple(
                (floor, a, rung_cap if ceiling is None else min(rung_cap, ceiling), f)
                for floor, a, rung_cap, f in fused_oneshot_ladder(world_size)
            )

        # (hidden, atoms, grid_cap, fanout) -> (engine, spec)
        self._by_cfg: dict[tuple, tuple] = {}
        try:
            for h in sorted({int(x) for x in hiddens}):
                self._build_hidden(h)
        except Exception:
            self.close()
            raise

    # -- engine construction -------------------------------------------------

    def _build_hidden(self, hidden: int) -> None:
        """Build every rung for hidden dim. Collective: all ranks must call it in
        the same order,."""
        for _floor, a, cap, f in self._ladder:
            key = (int(hidden), int(a), int(cap), f)
            if key in self._by_cfg:
                continue
            spec = make_one_shot_allreduce_kernel(
                world_size=self.world_size,
                atoms=key[1],
                grid=key[2],
                inbox_memory=self.inbox_memory,
                fanout=key[3],
                spin_sleep=self.spin_sleep,
                fusion="rmsnorm",
                hidden=key[0],
            )
            self._by_cfg[key] = (
                _StEngine(
                    spec=spec,
                    group=self.group,
                    rank=self.rank,
                    world_size=self.world_size,
                    inbox_flags=self._inbox_flags,
                    device_index=self._device_index,
                ),
                spec,
            )

    def _pick_cfg(self, hidden: int, live_bytes: int) -> tuple:
        chosen = self._ladder[0]
        for rung in self._ladder:
            if live_bytes >= rung[0]:
                chosen = rung
        key = (int(hidden), int(chosen[1]), int(chosen[2]), chosen[3])
        if key not in self._by_cfg:
            self._build_hidden(int(hidden))
        return key

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every engine."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    @property
    def hiddens(self) -> tuple[int, ...]:
        return tuple(sorted({k[0] for k in self._by_cfg}))

    def supports_hidden(self, hidden: int) -> bool:
        """Whether a build exists for *hidden* at every rung of this ladder."""
        return all(fused_hidden_supported(int(hidden), r[1]) for r in self._ladder)

    # -- launch --------------------------------------------------------------

    def _check(self, inp, residual_in, weight, out, residual_out):
        tensors = {
            "inp": inp,
            "residual_in": residual_in,
            "weight": weight,
            "out": out,
            "residual_out": residual_out,
        }
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"OneShotAllReduceRMSNorm requires a Tensor for {name}")
            if t.dtype != torch.bfloat16:
                raise ValueError(
                    f"OneShotAllReduceRMSNorm is bf16-only, {name} is {t.dtype}"
                )
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(
                    f"{name} must be on cuda:{self._device_index}, got {t.device}"
                )
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if int(t.data_ptr()) % 16 != 0:
                raise ValueError(f"{name} must be 16-byte aligned")

        hidden = int(inp.shape[-1])
        if weight.numel() != hidden:
            raise ValueError(
                f"weight width {weight.numel()} does not match input width {hidden}"
            )
        if tuple(residual_in.shape) != tuple(inp.shape):
            raise ValueError(
                f"residual_in shape {tuple(residual_in.shape)} != inp {tuple(inp.shape)}"
            )
        if tuple(out.shape) != tuple(inp.shape) or tuple(residual_out.shape) != tuple(
            inp.shape
        ):
            raise ValueError("out/residual_out must have the input's shape")
        if not self.supports_hidden(hidden):
            raise ValueError(
                f"no fused build for hidden={hidden} at atoms={[r[1] for r in self._ladder]}: "
                "one block covers one row, so hidden/(8*atoms) must be a multiple of 64 "
                "and at most 1024"
            )
        live_bytes = int(inp.numel()) * 2
        if live_bytes > 0xFFFFFFFF:
            raise ValueError("payload must not exceed the 4 GiB buffer window")
        # Distinct destinations: the kernel writes out and residual_out from the
        # same thread and reads residual_in after, so aliasing any pair is a race.
        spans = [
            (int(t.data_ptr()), int(t.data_ptr()) + live_bytes)
            for t in (inp, residual_in, out, residual_out)
        ]
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if max(spans[i][0], spans[j][0]) < min(spans[i][1], spans[j][1]):
                    raise ValueError(
                        "inp/residual_in/out/residual_out must not overlap"
                    )
        return hidden, live_bytes

    def _launch(
        self,
        inp,
        residual_in,
        weight,
        eps,
        out,
        residual_out,
        stream,
        *,
        hidden,
        live_bytes,
    ):
        eng, spec = self._by_cfg[self._pick_cfg(hidden, live_bytes)]
        num_tiles = int(inp.numel()) // hidden
        grid_x = max(1, min(num_tiles, spec["grid"]))
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
            Int64(int(residual_in.data_ptr())),
            Int64(int(residual_out.data_ptr())),
            Int64(int(weight.data_ptr())),
            Float32(float(eps)),
            stream,
        )
        self._has_launched = True
        _run_compiled(eng.launch, *args)

    def allreduce_rmsnorm(
        self,
        inp,
        residual_in,
        weight,
        eps,
        *,
        out=None,
        residual_out=None,
        stream=None,
    ):
        """``(out, residual_out)``; both are allocated when not supplied."""
        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(residual_in)
        hidden, live_bytes = self._check(inp, residual_in, weight, out, residual_out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduceRMSNorm got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling: this schedule pushes the whole payload "
                "to every peer, so its wire volume is (N-1)x the message. Pass "
                "max_bytes to override."
            )
        self._launch(
            inp,
            residual_in,
            weight,
            eps,
            out,
            residual_out,
            stream,
            hidden=hidden,
            live_bytes=live_bytes,
        )
        return out, residual_out

    def compile_and_launch(self, inp, residual_in, weight, eps=1e-6, stream=None):
        """Eager-JIT every rung for this shape and run each once.

        A real collective -- every rank must call it with the same shape. Used
        by the bench and the op tests to force a warm launch before timing or
        before a graph capture, where a first-call JIT would be fatal.
        """
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual_in)
        hidden, live_bytes = self._check(inp, residual_in, weight, out, residual_out)
        self._pick_cfg(hidden, live_bytes)  # force the engines for this hidden
        for key, (eng, spec) in self._by_cfg.items():
            if key[0] != hidden:
                continue
            num_tiles = int(inp.numel()) // hidden
            grid_x = max(1, min(num_tiles, spec["grid"]))
            st = (
                Stream(torch.cuda.current_stream(self._device_index))
                if stream is None
                else (stream if isinstance(stream, Stream) else Stream(stream))
            )
            self._has_launched = True
            _run_compiled(
                eng.launch,
                Int32(self.rank),
                Int64(live_bytes),
                Int32(num_tiles),
                Int64(int(inp.data_ptr())),
                Int64(int(out.data_ptr())),
                Int64(int(eng._gpu_peer_ptrs)),
                Int64(int(eng._colors)),
                Int32(grid_x),
                Int64(int(residual_in.data_ptr())),
                Int64(int(residual_out.data_ptr())),
                Int64(int(weight.data_ptr())),
                Float32(float(eps)),
                st,
            )
        return out, residual_out

    def variant(self, hidden: int, nbytes: int) -> str:
        """``<jit symbol>/g<grid_cap>/x<grid_x>`` for this (hidden, payload)."""
        key = self._pick_cfg(int(hidden), int(nbytes))
        eng, _spec = self._by_cfg[key]
        num_tiles = max(1, int(nbytes) // (int(hidden) * 2))
        grid_x = max(1, min(num_tiles, key[2]))
        return f"{kernel_symbol(eng.launch)}/g{key[2]}/x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
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
