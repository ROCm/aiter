# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4/INT6 all-reduce.

Public type ``QRInt4``, with two interchangeable schedules selected by
``algorithm``. Both are two-shot -- reduce-scatter then all-gather -- so they
are named for the topology of each lap instead: 
- ``"mesh"`` the default: fanout to all N-1 peers, twice.
- ``"ring"`` 2(N-1) single-destination hops.
Super-tile ST∈{1,8}. INT4 nibble or INT6 bit-plane pair, both with group-16
E4M3 scales. Payload HBM is bf16.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.dist.parallel_state import in_the_same_node_as
from aiter.jit.utils.chip_info import get_gfx_runtime

from .qr_int4_ipc import UncachedIpcHeap
from .qr_int4_kernel import SUPER_TILES, make_qr_int4_kernel
from .qr_int4_ring_kernel import (
    AG_CODECS,
    RING_ST_LADDER,
    RING_SUPER_TILES,
    RS_CODECS,
    make_qr_int4_ring_kernel,
)
from .qr_int_codec import CODECS
from .qr_int_shared import (
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    has_release_fence,
)

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
    ``QRInt4``. An unrecognized value is ignored rather than fatal.
    """
    raw = os.environ.get(_CODEC_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    name = raw.strip().lower()
    if name not in CODECS:
        logger.warning(
            "QRInt4: ignoring %s=%r, expected one of %s",
            _CODEC_ENV_VAR,
            raw,
            tuple(n.upper() for n in CODECS),
        )
        return None
    return name


AITER_ALL_REDUCE_CODEC = _parse_codec_env()

# Smallest payload sent through this kernel, in bytes.
MIN_PAYLOAD_BYTES = 128 << 10

# Floor on the block count when batching publishes into super-tiles; see
# ``QRInt4._grid_x``. Shrinking the grid trades parallelism for fewer release
# fences, which is only a good trade once there are enough fences to matter.
_MIN_BATCH_BLOCKS = 32

# Size floor for the ring algorithm.
# TODO: Measure the eaxct crossover between mesh and ring.
_RING_INT4_MIN_PAYLOAD_BYTES = 12 << 20
_RING_INT6_MIN_PAYLOAD_BYTES = 16 << 20
_RING_MIN_PAYLOAD_BYTES_BY_RS_CODEC = {
    "int4": _RING_INT4_MIN_PAYLOAD_BYTES,
    "int6": _RING_INT6_MIN_PAYLOAD_BYTES,
}


@dataclass(frozen=True)
class _Algorithm:
    """One all-reduce schedule, plus the host-side policy that tunes it.

    Everything ``QRInt4`` does *around* the kernel -- IPC setup, payload
    validation, one engine per super-tile, launch -- is identical across
    schedules and stays on the class. What differs is which kernel factory to
    call, which super-tile values and wire formats that factory accepts, and
    where the size floor sits. That is this record.

    ``build`` is keyword-only and always receives ``rank``, ``rs_codec`` and
    ``ag_codec``, whether or not a given schedule uses them. The ring bakes
    ``rank`` in at compile time -- the chunk a step operates on is
    ``(rank - step) % N``, which has to be a Python constant to index a
    register-resident atom list -- while the mesh takes it as a runtime kernel
    argument and ignores it here.
    """

    name: str
    build: Callable[..., dict]
    super_tiles: tuple[int, ...]
    rs_codecs: tuple[str, ...]
    ag_codecs: tuple[str, ...]
    min_bytes: int
    min_batch_blocks: int
    default_super_tile: int
    # Per-reduce-scatter-codec override of ``min_bytes``. Empty means the codec
    # does not move the floor, which is true of any schedule with one wire
    # format. See ``_RING_MIN_PAYLOAD_BYTES_BY_RS_CODEC``.
    min_bytes_by_rs_codec: tuple[tuple[str, int], ...] = ()

    def floor_bytes(self, rs_codec: str) -> int:
        return dict(self.min_bytes_by_rs_codec).get(rs_codec, self.min_bytes)
    # ``(min_payload_bytes, super_tile, grid_cap)`` rungs, ascending. Empty means
    # "one super-tile for every size", which is what the mesh does. When it is
    # non-empty and the caller did not pin ``super_tile``, QRInt4 builds an
    # engine per rung and selects by payload size at launch.
    st_ladder: tuple[tuple[int, int, int], ...] = ()


def _build_mesh(
    *, world_size, rank, super_tile, grid, inbox_memory, rs_codec, ag_codec
):
    del rank, rs_codec, ag_codec  # a runtime kernel argument; and not mesh knobs
    return make_qr_int4_kernel(
        world_size=world_size,
        super_tile=super_tile,
        grid=grid,
        inbox_memory=inbox_memory,
    )


ALGORITHMS = {
    "mesh": _Algorithm(
        name="mesh",
        build=_build_mesh,
        super_tiles=SUPER_TILES,
        rs_codecs=("int4",),
        ag_codecs=("int4",),
        min_bytes=MIN_PAYLOAD_BYTES,
        min_batch_blocks=_MIN_BATCH_BLOCKS,
        default_super_tile=8,
    ),
    "ring": _Algorithm(
        name="ring",
        build=make_qr_int4_ring_kernel,
        super_tiles=RING_SUPER_TILES,
        rs_codecs=RS_CODECS,
        ag_codecs=AG_CODECS,
        min_bytes=_RING_INT4_MIN_PAYLOAD_BYTES,
        min_batch_blocks=_MIN_BATCH_BLOCKS,
        default_super_tile=8,
        st_ladder=RING_ST_LADDER,
        min_bytes_by_rs_codec=tuple(_RING_MIN_PAYLOAD_BYTES_BY_RS_CODEC.items()),
    ),
}
DEFAULT_ALGORITHM = "mesh"

# World size at which the ring's reduce-scatter lap needs INT6 to clear the
# 18 dB SQNR floor the schedules are held to.
#
# The ring's error grows with N -- it requantizes the running partial at every
# hop, and the partial's extremum grows with the contributions folded in -- so
# unlike the mesh it does not have one SQNR for every world size.
_RS_INT6_MIN_WORLD = 8


_warned_codecs: set[tuple[str, str, str]] = set()


def _warn_codec_unavailable(algo_name, label, requested, used):
    """Say it once per (schedule, lap, request), not once per engine.

    ``QRInt4`` builds one engine per super-tile rung, so a per-construction
    warning would fire several times for one object and again for every object
    -- for a condition that is a property of the schedule and cannot change
    within a process.
    """
    key = (algo_name, label, requested)
    if key in _warned_codecs:
        return
    _warned_codecs.add(key)
    logger.warning(
        "QRInt4: %s=%s does not apply to %s on algorithm=%r; using %r",
        _CODEC_ENV_VAR,
        requested.upper(),
        label,
        algo_name,
        used,
    )


def _resolve_codecs(algo, world_size, rs_codec, ag_codec):
    """Codecs for one engine: explicit argument > env var > per-N default.

    ``None`` means "not specified", which is why the constructor cannot simply
    default these to ``"int4"``: an explicit ``rs_codec="int4"`` has to outrank
    ``AITER_ALL_REDUCE_CODEC=INT6``, and it cannot if the two are indistinguishable
    by the time they get here.

    A codec the selected schedule cannot build falls back with a warning rather
    than raising . An explicit argument still raises.
    """
    rs_default = "int6" if world_size >= _RS_INT6_MIN_WORLD else "int4"

    # The all-gather lap forwards bytes verbatim and so contributes exactly one
    # quantization. It is the dominant error term if the RS lap is INT6.
    ag_default = "int4"

    def _pick(requested, default, supported, label):
        if requested is not None:
            if requested not in supported:
                raise ValueError(
                    f"{label} must be one of {supported} for "
                    f"algorithm={algo.name!r}, got {requested!r}"
                )
            return requested
        # The default is a property of the world size, not of the schedule.
        if default not in supported:
            default = supported[0]
        env = AITER_ALL_REDUCE_CODEC
        if env is not None and env != default:
            if env in supported:
                return env
            _warn_codec_unavailable(algo.name, label, env, default)
        return default

    return (
        _pick(rs_codec, rs_default, algo.rs_codecs, "rs_codec"),
        _pick(ag_codec, ag_default, algo.ag_codecs, "ag_codec"),
    )

# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def has_xgmi_peer_links() -> bool:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe.

    Arch is not enough to make this call: an MI350X (xGMI) and an MI350P
    (PCIe-only) both report ``gfx950``, and the right inbox memory type is
    opposite on the two. KFD exposes the real link type per peer pair, so read
    that instead of guessing from the SKU.

    Both ``io_links`` and ``p2p_links`` have to be scanned. KFD only populates
    ``p2p_links`` for peers reachable indirectly (through a host bridge), so on
    a directly-connected mesh it holds nothing but the PCIe links to the CPU
    nodes and the xGMI peers appear solely under ``io_links``. Reading
    ``p2p_links`` alone reports "PCIe" on an 8-GPU all-xGMI MI350X, which flips
    ``inbox_memory="auto"`` to the fine-grained heap and silently costs the
    uncached fanout the kernel was designed around.

    Returns True when the topology cannot be read, which keeps the historical
    uncached allocation on any host we cannot classify -- the failure mode of
    guessing "PCIe" on an xGMI box is a silent perf regression on hardware
    where the current design is already optimal.
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
        logger.debug("QRInt4: cannot read KFD topology; assuming xGMI", exc_info=True)
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
        self.compiled = None
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self._peer_bases = [None] * world_size
        # The inbox is the only allocation peers write into, so it is the only
        # one whose memory type matters for fabric throughput.
        self._buf_ptr = UncachedIpcHeap.alloc(
            self.buf_bytes, inbox_flags, expected_device=device_index
        )
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
        color_bytes = self.grid * 4
        # Peer-pointer table and per-block colours: written by the host once and
        # by this rank's own kernel, never by a peer. Stays uncached in every
        # mode -- no cross-GPU visibility question, and it is a few KiB.
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

    ``algorithm`` selects the schedule. Both are two-shot -- reduce-scatter then
    all-gather -- so they are named for the topology of each lap:

    * ``"mesh"`` (default) -- each rank pushes to every one of the ``N-1``
      peers, twice. Two hops. Optimal on a meshed xGMI node.
    * ``"ring"`` -- ``2(N-1)`` hops, each a single contiguous run into exactly
      one peer's inbox. Same wire volume (``2(N-1)/N`` of the payload), traded
      for per-destination locality. Structurally worse at decode sizes and on
      xGMI - opt in deliberately.

    ``rs_codec`` and ``ag_codec`` are the wire formats of the ring's two laps.
    The reduce-scatter lap is the only place the ring loses accuracy the mesh
    does not -- it requantizes ``N-1`` times where the mesh requantizes once --
    so it defaults to ``"int6"`` at TP8 where we would otherwise lose too much acccuracy. 
    The all-gather lap forwards bytes verbatim and contributes a single quantization, 
    so it defaults to ``"int4"`` everywhere and widens only by request.

    Leave both ``None`` to get those defaults. ``AITER_ALL_REDUCE_CODEC=INT4``
    or ``INT6`` overrides them process-wide, for both laps at once; an explicit
    argument here outranks the environment.

    ``inbox_memory`` selects how the IPC inbox is allocated:

    * ``"auto"`` (default) -- ``uncached`` on hosts with xGMI peer links,
      ``finegrained`` on PCIe-attached hosts. Decided from the KFD topology,
      not from the arch string: MI350X and MI350P both report ``gfx950`` and
      want opposite answers.
    * ``"uncached"`` -- the historical behaviour; correct everywhere, but peer
      writes collapse on PCIe (1.44 GB/s to 3 peers on MI350P, against
      33.5 GB/s fine-grained).
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so the
      peer stores are emitted ``sc0 sc1`` to write through rather than parking
      in the writer's L2; the wire protocol is unchanged.

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to ``MIN_PAYLOAD_BYTES``. ``compile`` is deliberately not gated:
    its warmup tensor is allowed to be small.
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
        batch_publishes: bool | None = None,
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
        # ``None`` means "use the schedule's own policy": for the mesh that is a
        # single super-tile, for the ring the payload-size ladder. Passing a
        # value pins one super-tile for every size, which is what the benchmark
        # variants and the tuning sweeps do.
        pinned_st = super_tile is not None
        if super_tile is None:
            super_tile = algo.default_super_tile
        if super_tile not in algo.super_tiles:
            raise ValueError(
                f"super_tile must be one of {algo.super_tiles} for "
                f"algorithm={algorithm!r}, got {super_tile!r}"
            )
        rs_codec, ag_codec = _resolve_codecs(
            algo, int(world_size), rs_codec, ag_codec
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
        # Rungs to build, each ``(super_tile, grid_cap)``. Pinning
        # ``super_tile`` collapses the ladder to that one rung -- a caller who
        # named a super-tile gets exactly it, at every size.
        #
        # ``grid_cap`` is a *ceiling*, not a pin: it bounds every rung rather
        # than disabling size-dependent selection. Raising it above a rung's own
        # cap is a no-op (the rung cap is already sized so ``_grid_x`` never
        # binds over that rung's payload range), while lowering it constrains
        # the wire buffer, which is what a caller passing it usually wants.
        if algo.st_ladder and not pinned_st:
            rungs = [(st, min(rung_cap, cap)) for _, st, rung_cap in algo.st_ladder]
            ladder = algo.st_ladder
        else:
            rungs = [(super_tile, cap)]
            ladder = ()
        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        torch.cuda.set_device(device)
        self.group = group
        self.device = device
        self._device_index = _cuda_index(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.super_tile = int(super_tile)
        self._grid = cap
        self.inbox_memory = resolved_inbox
        self.algorithm = algorithm
        self.rs_codec = rs_codec
        self.ag_codec = ag_codec
        self._algo = algo

        #self._batch_publishes = has_release_fence(resolved_inbox)
        self._batch_publishes = (
            has_release_fence(resolved_inbox)
            if batch_publishes is None
            else bool(batch_publishes)
        )
        
        self.min_bytes = (
            algo.floor_bytes(rs_codec) if min_bytes is None else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        # ST=1 is always built: _pick_st falls back to it when a payload has
        # fewer tiles than the chosen super-tile. Engines are built in a fixed
        # order because each does its own IPC handle exchange, which is a
        # collective -- ranks disagreeing on the order would deadlock.
        by_cap = {1: cap}
        for st, rung_cap in rungs:
            by_cap.setdefault(st, rung_cap)
        self._ladder = ladder
        self._by_st = {}
        for st in sorted(by_cap):
            spec = algo.build(
                world_size=self.world_size,
                rank=self.rank,
                super_tile=st,
                grid=by_cap[st],
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

        primary = self._by_st[self.super_tile]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    def _ladder_st(self, live_bytes: int) -> int:
        """Super-tile the ladder assigns to a *live_bytes* payload.

        Publishes per rank are ``num_tiles / ST * 2(N-1)`` and cost a full L2
        writeback each, so a bigger payload wants a bigger ST -- but ST also
        divides the block count, so it cannot simply be maximised. The rungs and
        the measurements behind them are in ``RING_ST_LADDER``.
        """
        st = self._by_st and min(self._by_st)
        for floor, rung_st, _cap in self._ladder:
            if live_bytes >= floor:
                st = rung_st
        return st

    def _pick_st(self, num_tiles: int, live_bytes: int | None = None) -> int:
        """Super-tile for a payload of *num_tiles* tiles.

        With a ladder, *live_bytes* chooses the rung and *num_tiles* only has to
        confirm there is a whole super-tile to take; without one the single
        configured super-tile is the only candidate.

        Without a release fence a publish is nearly free, so the only reason to
        batch tiles is when there are more of them than blocks -- prefer ST=1
        and the parallelism it buys.

        With one, that trade inverts: every publish costs a full L2 writeback,
        and ST=1 pays one per tile per phase. Take a super-tile as soon as
        there is a whole one to take. Measured on MI350P at 1024x7168, TP4:
        577.71 us at ST=1 against 269.01 at ST=8.
        """
        want = self.super_tile
        if self._ladder and live_bytes is not None:
            want = self._ladder_st(live_bytes)
        if want == 1:
            return 1
        if self._batch_publishes:
            return want if num_tiles >= want else 1
        return want if num_tiles > self._by_st[want].grid else 1

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int | None = None) -> int:
        """Blocks to launch for *num_tiles* tiles under *super_tile*.

        *grid* is the compile-time cap of the engine that will run, which is
        per-super-tile once a ladder is in play -- the wire buffer scales with
        ``ST * grid``, so a high rung pairs a large ST with a small cap.

        Batching publishes only pays if a block actually owns a super-tile's
        worth of work: ST=8 across 448 blocks holding one tile each still
        publishes per tile. Hand each block a full super-tile instead, which
        cuts publishes to ``num_tiles / ST`` per phase.

        Bounded below by ``_MIN_BATCH_BLOCKS``, because that trade inverts at
        small sizes: 14 tiles over 2 blocks saves a handful of fences and gives
        up the whole machine to do it. Measured on MI350P at 32x7168, TP4,
        61.95 us unbounded against 23.98 with the grid left alone.
        """
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), self._algo.min_batch_blocks)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, self._grid if grid is None else grid))

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
        grid_x = self._grid_x(num_tiles, eng.super_tile, eng.grid)
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

        Default ST=8 also builds an ST=1 engine for ``num_tiles ≤ grid_cap``.
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

    def is_beneficial(self, nbytes: int) -> bool:
        """Whether *nbytes* is large enough for this kernel to be worth using.

        Callers with a fallback should route anything smaller to it; see
        ``MIN_PAYLOAD_BYTES``. ``allreduce`` refuses payloads below the
        threshold rather than silently running them slowly.
        """
        return int(nbytes) >= self.min_bytes

    def allreduce(self, inp, out, stream=None):
        live_bytes = self._check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"QRInt4.allreduce got a {live_bytes} B payload, below the "
                f"{self.min_bytes} B floor: at decode sizes this kernel saves a "
                "few microseconds on a collective that is not the bottleneck, "
                "and charges ~36 dB of SQNR for them. Route small messages to "
                "an exact all-reduce, or pass min_bytes=0 to override."
            )
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles, live_bytes)

        self._launch_eng(self._by_st[st], inp, out, stream)
