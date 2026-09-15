# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT6 all-reduce.

Public type ``QuickAllReduceInt6``. Mesh schedule: fanout to all N-1 peers,
twice (reduce-scatter then all-gather). Super-tile ST∈{1,8} from
``MESH_ST_LADDER``: ST=1 until the payload fills the occupancy-clamped
grid, then ST=8. INT6 bit-plane pair with group-16 E4M3 scales. Payload
HBM is bf16.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.quick_allreduce_codec import CODECS
from .kernels.quick_allreduce_int6 import (
    MESH_CODECS,
    MESH_ST_LADDER,
    SUPER_TILES,
    clamp_grid_cap,
    make_quick_allreduce_int6_kernel,
)
from .kernels.quick_allreduce_shared import (
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
)
from .kernels.tensor_shim import _run_compiled
from .quick_allreduce_int4_ipc import UncachedIpcHeap

logger = logging.getLogger("aiter")

_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# How the IPC inbox is allocated. The wire protocol is identical in every
# mode; only the memory type changes.
INBOX_MEMORY_MODES = ("auto", "uncached", "finegrained", "default")

# Process-wide codec override, applied to *both* laps of whichever schedule is
# selected. Unset means the per-world-size defaults in ``_resolve_codecs``.
_CODEC_ENV_VAR = "AITER_ALL_REDUCE_CODEC"


def _parse_codec_env() -> str | None:
    """The codec named by ``AITER_ALL_REDUCE_CODEC``, or None.

    Parsed once at import so an unrecognized value warns once rather than per
    ``QuickAllReduceInt6``. An unrecognized value is ignored rather than fatal.
    """
    raw = os.environ.get(_CODEC_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    name = raw.strip().lower()
    if name not in CODECS:
        logger.warning(
            "QuickAllReduceInt6: ignoring %s=%r, expected one of %s",
            _CODEC_ENV_VAR,
            raw,
            tuple(n.upper() for n in CODECS),
        )
        return None
    return name


AITER_ALL_REDUCE_CODEC = _parse_codec_env()

# Smallest payload sent through this kernel, in bytes.
MIN_PAYLOAD_BYTES = 128 << 10

# Floor kept on the schedule record for the host dataclass; the mesh launch
# grid is ``min(num_tiles, occupancy-clamped cap)``.
_MIN_BATCH_BLOCKS = 32


@dataclass(frozen=True)
class _Algorithm:
    """Mesh schedule record: kernel factory, ST values, codecs, size floor."""

    name: str
    build: Callable[..., dict]
    super_tiles: tuple[int, ...]
    rs_codecs: tuple[str, ...]
    ag_codecs: tuple[str, ...]
    min_bytes: int
    min_batch_blocks: int
    default_super_tile: int
    # Per-world-size override of ``min_bytes``. Empty: the mesh floor does not
    # move with N.
    min_bytes_by_world: tuple[tuple[int, int], ...] = ()
    # ``world_size -> ((min_payload_bytes, super_tile, grid_cap), ...)``.
    st_ladder: dict[int, tuple[tuple[int, int, int], ...]] | None = None

    def floor_bytes(self, world_size: int) -> int:
        return dict(self.min_bytes_by_world).get(int(world_size), self.min_bytes)

    def ladder_for(self, world_size: int) -> tuple[tuple[int, int, int], ...]:
        if not self.st_ladder:
            return ()
        return self.st_ladder.get(int(world_size), ())


def _build_mesh(
    *, world_size, rank, super_tile, grid, inbox_memory, rs_codec, ag_codec
):
    del rank  # a runtime kernel argument, not a mesh build knob
    if rs_codec != ag_codec:
        raise ValueError(
            f"mesh algorithm has one wire format for both laps, got "
            f"rs_codec={rs_codec!r} != ag_codec={ag_codec!r}"
        )
    return make_quick_allreduce_int6_kernel(
        world_size=world_size,
        super_tile=super_tile,
        grid=grid,
        inbox_memory=inbox_memory,
        codec=rs_codec,
    )


ALGORITHMS = {
    "mesh": _Algorithm(
        name="mesh",
        build=_build_mesh,
        super_tiles=SUPER_TILES,
        rs_codecs=MESH_CODECS,
        ag_codecs=MESH_CODECS,
        min_bytes=MIN_PAYLOAD_BYTES,
        min_batch_blocks=_MIN_BATCH_BLOCKS,
        default_super_tile=8,
        st_ladder=MESH_ST_LADDER,
    ),
}
DEFAULT_ALGORITHM = "mesh"

_warned_codecs: set[tuple[str, str, str]] = set()


def _warn_codec_unavailable(algo_name, label, requested, used):
    """Say it once per (schedule, lap, request), not once per engine.

    ``QuickAllReduceInt6`` builds one engine per super-tile rung, so a
    per-construction warning would fire several times for one object and again
    for every object -- for a condition that is a property of the schedule and
    cannot change within a process.
    """
    key = (algo_name, label, requested)
    if key in _warned_codecs:
        return
    _warned_codecs.add(key)
    logger.warning(
        "QuickAllReduceInt6: %s=%s does not apply to %s on algorithm=%r; using %r",
        _CODEC_ENV_VAR,
        requested.upper(),
        label,
        algo_name,
        used,
    )


def _resolve_codecs(algo, world_size, rs_codec, ag_codec):
    """Codecs for one engine: explicit argument > env var > default.

    ``None`` means "not specified", which is why the constructor cannot simply
    default these to ``"int6"``: an explicit ``rs_codec="int6"`` has to outrank
    ``AITER_ALL_REDUCE_CODEC=INT4``, and it cannot if the two are
    indistinguishable by the time they get here.

    A codec the selected schedule cannot build falls back with a warning rather
    than raising. An explicit argument still raises.
    """
    rs_default = "int6"
    ag_default = "int6"

    def _pick(requested, default, supported, label):
        if requested is not None:
            if requested not in supported:
                raise ValueError(
                    f"{label} must be one of {supported} for "
                    f"algorithm={algo.name!r}, got {requested!r}"
                )
            return requested
        if default not in supported:
            default = supported[0]
        env = AITER_ALL_REDUCE_CODEC
        if env is not None and env != default:
            if env in supported:
                return env
            _warn_codec_unavailable(algo.name, label, env, default)
        return default

    resolved_rs = _pick(rs_codec, rs_default, algo.rs_codecs, "rs_codec")
    resolved_ag = _pick(ag_codec, ag_default, algo.ag_codecs, "ag_codec")
    return resolved_rs, resolved_ag


# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def has_xgmi_peer_links() -> bool:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe.

    Arch is not enough: gfx950 is both an xGMI SKU and a PCIe-only SKU, and
    they want opposite inbox types. KFD exposes the real link type per peer.

    Scan both ``io_links`` and ``p2p_links``. KFD only fills ``p2p_links``
    for peers reachable through a host bridge, so a directly-connected mesh
    lists xGMI peers under ``io_links`` only.

    Returns True when the topology cannot be read, which keeps uncached
    allocation -- guessing PCIe on an xGMI box is the worse failure.
    """
    try:
        for subdir in ("io_links", "p2p_links"):
            for props in _KFD_NODES.glob(f"*/{subdir}/*/properties"):
                for line in props.read_text().splitlines():
                    field, _, value = line.partition(" ")
                    if field == "type" and int(value) == _HSA_IOLINK_TYPE_XGMI:
                        return True
        return False
    except (OSError, ValueError):
        logger.debug(
            "QuickAllReduceInt6: cannot read KFD topology; assuming xGMI",
            exc_info=True,
        )
        return True


def _resolve_inbox_flags(mode: str) -> tuple[int, str]:
    """(hipExtMallocWithFlags mode, resolved name) for an ``inbox_memory``."""
    if mode not in INBOX_MEMORY_MODES:
        raise ValueError(
            f"inbox_memory must be one of {INBOX_MEMORY_MODES}, got {mode!r}"
        )
    if mode == "auto":
        mode = "uncached" if has_xgmi_peer_links() else "finegrained"
    flags = {
        "uncached": UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
        "finegrained": UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED,
        "default": UncachedIpcHeap._HIP_DEVICE_MALLOC_DEFAULT,
    }[mode]
    return flags, mode


def _cuda_index(device) -> int:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"QuickAllReduceInt6 requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _validate_ipc_process_group(group, *, rank: int) -> None:
    """Reject groups that cannot exchange HIP IPC handles or CPU-side metadata."""
    # Keep parallel_state lazy: this module is imported while aiter's AOT setup
    # is still initializing the top-level package.
    from aiter.dist.parallel_state import in_the_same_node_as

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"QuickAllReduceInt6 does not support NCCL process groups (got "
            f"{backend!r} on group rank {rank}): IPC handle exchange requires "
            "CPU-side broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            "QuickAllReduceInt6 does not support multi-node process groups: HIP "
            f"IPC handles are node-local (ranks not on rank 0's node: {off_node})."
        )


def kernel_symbol(launch) -> str:
    """The JIT symbol a kernel factory stamped on its launch wrapper.

    Every factory names its wrapper ``launch_<kernel>_<tag>``, where the tag
    carries every compile-time knob that changes the emitted code -- world
    size, super-tile, inbox memory and wire format. That string is the only
    place the *actual* variant that ran is written down, so a benchmark
    reporting a candidate alias cannot say which binary it timed, and an
    "auto" row that walks a size ladder cannot say anything at all.

    Falls back to ``"?"`` rather than raising: this is reporting metadata, and
    a flydsl build that stops exposing ``.func`` should not take a sweep down.
    """
    name = getattr(getattr(launch, "func", None), "__name__", None)
    if not name:
        return "?"
    return name.removeprefix("launch_")


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(
        self,
        *,
        spec,
        group,
        rank: int,
        world_size: int,
        inbox_flags: int,
        device_index: int,
    ):
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
        self._gpu_peer_ptrs = None
        self._colors = None
        try:
            # The inbox is the only allocation peers write into, so it is the
            # only one whose memory type matters for fabric throughput.
            self._buf_ptr = UncachedIpcHeap.alloc(
                self.buf_bytes, inbox_flags, expected_device=device_index
            )
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
            # Peer-pointer table and per-block colours: written by the host once
            # and by this rank's own kernel, never by a peer. Stays uncached in
            # every mode -- no cross-GPU visibility question, and it is a few
            # KiB.
            self._meta_ptr = UncachedIpcHeap.alloc_uncached(
                peer_bytes + color_bytes, expected_device=device_index
            )
            self._gpu_peer_ptrs = self._meta_ptr
            self._colors = self._meta_ptr + peer_bytes
            UncachedIpcHeap.copy_host_to_device(
                self._gpu_peer_ptrs,
                (ctypes.c_int64 * world_size)(*peer_ptrs),
                peer_bytes,
            )
            UncachedIpcHeap.copy_host_to_device(
                self._colors,
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
            self._gpu_peer_ptrs = None
            self._colors = None
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


class QuickAllReduceInt6:
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_int6``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    Mesh schedule: each rank pushes to every one of the ``N-1`` peers, twice
    (reduce-scatter then all-gather). Super-tile ST∈{1,8} is chosen from
    ``MESH_ST_LADDER``: ST=1 until the payload fills the occupancy-clamped
    grid, then ST=8. Passing ``super_tile`` pins that value and skips the
    ladder.

    ``rs_codec`` and ``ag_codec`` are the wire formats of the two laps. Both
    default to ``"int6"``. The mesh carries one format across both laps, so
    they have to agree.

    Leave both ``None`` to get those defaults. ``AITER_ALL_REDUCE_CODEC=INT4``
    or ``INT6`` overrides them process-wide, for both laps at once; an explicit
    argument here outranks the environment.

    ``inbox_memory`` selects how the IPC inbox is allocated:

    * ``"auto"`` (default) -- ``uncached`` on xGMI, ``finegrained`` on PCIe.
      From the KFD topology, not the arch string (gfx950 is both).
    * ``"uncached"`` -- right on xGMI; peer writes serialize on PCIe.
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so
      peer stores use ``sc0 sc1`` to write through; the wire protocol is
      unchanged.

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to ``MIN_PAYLOAD_BYTES``. ``compile_and_launch`` is deliberately
    not gated: its warmup tensor is allowed to be small.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        super_tile: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        min_bytes: int | None = None,
        algorithm: str = DEFAULT_ALGORITHM,
        rs_codec: str | None = None,
        ag_codec: str | None = None,
    ):
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {tuple(ALGORITHMS)}, got {algorithm!r}"
            )
        algo = ALGORITHMS[algorithm]
        # ``None`` walks MESH_ST_LADDER. Passing a value pins one super-tile
        # for every size.
        pinned_st = super_tile is not None
        if super_tile is None:
            super_tile = algo.default_super_tile
        if super_tile not in algo.super_tiles:
            raise ValueError(
                f"super_tile must be one of {algo.super_tiles} for "
                f"algorithm={algorithm!r}, got {super_tile!r}"
            )
        rs_codec, ag_codec = _resolve_codecs(algo, int(world_size), rs_codec, ag_codec)
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
                f"QuickAllReduceInt6 supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")
        world_ladder = algo.ladder_for(world_size)
        if world_ladder and not pinned_st:
            rungs = [(st, min(rung_cap, cap)) for _, st, rung_cap in world_ladder]
            ladder = world_ladder
            super_tile = ladder[0][1]
        else:
            rungs = [(super_tile, cap)]
            ladder = ()
        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        # set_device rejects torch.device("cuda") with no index; resolve first.
        self._device_index = _cuda_index(device)
        torch.cuda.set_device(self._device_index)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.super_tile = int(super_tile)
        self._grid = cap
        self.inbox_memory = resolved_inbox
        self.algorithm = algorithm
        self.rs_codec = rs_codec
        self.ag_codec = ag_codec
        self._algo = algo
        self._has_launched = False
        cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )

        self.min_bytes = (
            algo.floor_bytes(self.world_size) if min_bytes is None else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        # ST=1 is always built so decode-sized payloads can take it. Rungs from
        # the ladder keep their own cap; a pin builds that ST plus ST=1, both
        # occupancy-clamped. Order is a collective: each engine does its own
        # IPC handle exchange.
        by_cap = {}
        for st, rung_cap in rungs:
            by_cap.setdefault(st, rung_cap)
        by_cap.setdefault(1, min(by_cap.values()) if by_cap else cap)
        self._ladder = ladder
        self._by_st = {}
        try:
            for st in sorted(by_cap):
                # A persistent kernel deadlocks if it launches more workgroups
                # than fit, and the ranks have to agree on the number: take the
                # minimum across the group so a heterogeneous node converges.
                grid = clamp_grid_cap(
                    by_cap[st],
                    arch=arch,
                    world_size=self.world_size,
                    super_tile=st,
                    cu_count=cu_count,
                )
                shared_grid = torch.tensor(grid, dtype=torch.int64)
                dist.all_reduce(shared_grid, op=dist.ReduceOp.MIN, group=group)
                spec = algo.build(
                    world_size=self.world_size,
                    rank=self.rank,
                    super_tile=st,
                    grid=int(shared_grid.item()),
                    inbox_memory=resolved_inbox,
                    rs_codec=rs_codec,
                    ag_codec=ag_codec,
                )
                self._by_st[st] = _StEngine(
                    spec=spec,
                    group=self.group,
                    rank=self.rank,
                    world_size=self.world_size,
                    inbox_flags=inbox_flags,
                    device_index=self._device_index,
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

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on *this* rank, across every ST."""
        return sum(eng.buf_bytes for eng in self._by_st.values())

    def _ladder_st(self, live_bytes: int) -> int:
        """Super-tile ``MESH_ST_LADDER`` assigns to a *live_bytes* payload."""
        st = self._by_st and min(self._by_st)
        for floor, rung_st, _cap in self._ladder:
            if live_bytes >= floor:
                st = rung_st
        return st

    def _pick_st(self, num_tiles: int, live_bytes: int | None = None) -> int:
        """Super-tile for a payload of *num_tiles* tiles.

        With a ladder, *live_bytes* chooses the rung and *num_tiles* only has
        to confirm there is a whole super-tile to take; without one the
        configured super-tile is the only candidate.
        """
        want = self.super_tile
        if self._ladder and live_bytes is not None:
            want = self._ladder_st(live_bytes)
        if want == 1:
            return 1
        return want if num_tiles >= want else 1

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int | None = None) -> int:
        """Blocks to launch for *num_tiles* tiles.

        Persistent grid is occupancy-clamped at compile time; launch at most
        that many workgroups, and no more than one per tile.
        """
        del super_tile
        return max(1, min(num_tiles, self._grid if grid is None else grid))

    def _check_payload(self, inp, out) -> int:
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise TypeError("QuickAllReduceInt6 requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("QuickAllReduceInt6 supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("QuickAllReduceInt6 requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("QuickAllReduceInt6 requires contiguous input/output")
        inp_ptr = int(inp.data_ptr())
        out_ptr = int(out.data_ptr())
        if inp_ptr % 16 != 0 or out_ptr % 16 != 0:
            raise ValueError("QuickAllReduceInt6 requires 16-byte-aligned input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes > 0xFFFFFFFF:
            raise ValueError(
                "QuickAllReduceInt6 payload must not exceed the 4 GiB buffer window"
            )
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
            raise ValueError("QuickAllReduceInt6 requires non-overlapping input/output")
        return live_bytes

    def _launch_args(self, eng: _StEngine, inp, out, stream, *, live_bytes, num_tiles):
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
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(self._grid_x(num_tiles, eng.super_tile, eng.grid)),
            stream,
        )

    def _launch_eng(self, eng: _StEngine, inp, out, stream, *, live_bytes: int) -> None:
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        args = self._launch_args(
            eng, inp, out, stream, live_bytes=live_bytes, num_tiles=num_tiles
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        _run_compiled(eng.launch, *args)

    def compile_and_launch(self, inp, out=None, stream=None) -> None:
        """Eager-JIT every ST binary and launch each of them once, for real,
        against *inp*/*out*.

        This runs every ST on the GPU -- ``out`` ends up holding whichever ST
        ran last, and it is a real collective: every rank must call it with
        the same shape. Used by ``bench_comm_allreduce.py`` and the op tests to
        force a real warm launch before timing or correctness checks begin.
        Production never calls this: it tolerates the first real call paying a
        JIT-compile cost instead.
        """
        if out is None:
            out = torch.empty_like(inp)
        live_bytes = self._check_payload(inp, out)
        for eng in self._by_st.values():
            self._launch_eng(eng, inp, out, stream, live_bytes=live_bytes)

    def close(self):
        engines = getattr(self, "_by_st", None)
        if not engines:
            return
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
        """Identity of the binary an *nbytes* payload would actually run.

        ``<jit symbol>/g<grid_cap>/x<grid_x>``. Uses the same ``_pick_st`` as
        launch, so this is the ST the payload actually gets. Pure: builds
        nothing and launches nothing.
        """
        live_bytes = int(nbytes)
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        eng = self._by_st[self._pick_st(num_tiles, live_bytes)]
        grid_x = self._grid_x(num_tiles, eng.super_tile, eng.grid)
        return f"{kernel_symbol(eng.launch)}/g{eng.grid}/x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        """Whether *nbytes* is large enough for this kernel to be worth using.

        Callers with a fallback should route anything smaller to it; see
        ``MIN_PAYLOAD_BYTES``. ``allreduce`` refuses payloads below the
        threshold rather than silently running them slowly.
        """
        return int(nbytes) >= self.min_bytes

    def allreduce(self, inp, out, stream=None):
        """Two-shot INT6 all-reduce into ``out``.

        ``stream=None`` uses the current PyTorch stream on this device.
        """
        live_bytes = self._check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"QuickAllReduceInt6.allreduce got a {live_bytes} B payload, "
                f"below the {self.min_bytes} B floor: route small messages "
                "to an exact all-reduce, or pass min_bytes=0 to override."
            )
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles, live_bytes)
        self._launch_eng(self._by_st[st], inp, out, stream, live_bytes=live_bytes)
