# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4/INT6 all-reduce.

Public type ``QuickAllReduceInt4``, with two interchangeable schedules selected
by ``algorithm``. Both are two-shot -- reduce-scatter then all-gather -- so
they are named for the topology of each lap instead:

* ``"mesh"`` the default: fanout to all N-1 peers, twice.
* ``"ring"`` 2(N-1) single-destination hops.

Super-tile ST∈{1,8} on the mesh, ST∈{1,8,16,32} on the ring. INT4 nibble or
INT6 bit-plane pair, both with group-16 E4M3 scales. Payload HBM is bf16.
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
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes

from .kernels.quick_allreduce_codec import CODECS
from .kernels.quick_allreduce_fusions import (
    quick_reduce_hidden_supported,
    quick_reduce_row_block,
)
from .kernels.quick_allreduce_int4 import (
    MESH_CODECS,
    MESH_ST_LADDER,
    SUPER_TILES,
    clamp_grid_cap,
    make_quick_allreduce_int4_kernel,
)
from .kernels.quick_allreduce_int4_ring import (
    AG_CODECS,
    RING_ST_LADDER,
    RING_SUPER_TILES,
    RS_CODECS,
    make_quick_allreduce_int4_ring_kernel,
)
from .kernels.quick_allreduce_shared import (
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WAVE,
    WORLD,
    has_release_fence,
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
    ``QuickAllReduceInt4``. An unrecognized value is ignored rather than fatal.
    """
    raw = os.environ.get(_CODEC_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    name = raw.strip().lower()
    if name not in CODECS:
        logger.warning(
            "QuickAllReduceInt4: ignoring %s=%r, expected one of %s",
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
# ``QuickAllReduceInt4._grid_x``. Shrinking the grid trades parallelism for
# fewer release fences, which is only a good trade once there are enough fences
# to matter.
_MIN_BATCH_BLOCKS = 32

# Size floor for the ring schedule, i.e. where the mesh stops winning.
#
# Keyed on world size rather than on the reduce-scatter codec: measurement says
# the codec is not the variable that moves this boundary, N is. See
# ``allreduce_policy.FAMILY_POLICY``, whose ``mesh_max`` these mirror; the
# numbers come from the same fit.
#
# This is only the *standalone* guard rail -- what ``QuickAllReduceInt4``
# refuses below when someone constructs one directly with ``algorithm="ring"``.
# Production dispatch does not consult it; ``FlyDSLAllReduce`` owns the real
# boundary, which is additionally keyed on link type.
_RING_MIN_PAYLOAD_BYTES_BY_WORLD = {
    2: 4 << 20,
    4: 12 << 20,
    8: 12 << 20,
}
_RING_DEFAULT_MIN_PAYLOAD_BYTES = 12 << 20


@dataclass(frozen=True)
class _Algorithm:
    """One all-reduce schedule, plus the host-side policy that tunes it.

    Everything ``QuickAllReduceInt4`` does *around* the kernel -- IPC setup,
    payload validation, one engine per super-tile, launch -- is identical
    across schedules and stays on the class. What differs is which kernel
    factory to call, which super-tile values and wire formats that factory
    accepts, and where the size floor sits. That is this record.

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
    # Per-world-size override of ``min_bytes``. Empty means the world does not
    # move this schedule's floor, which is true of the mesh -- it is gated from
    # below by accuracy, which does not depend on N. The ring's floor is where
    # the mesh stops winning, which very much does. See
    # ``_RING_MIN_PAYLOAD_BYTES_BY_WORLD``.
    min_bytes_by_world: tuple[tuple[int, int], ...] = ()

    def floor_bytes(self, world_size: int) -> int:
        return dict(self.min_bytes_by_world).get(int(world_size), self.min_bytes)

    # ``world_size -> ((min_payload_bytes, super_tile, grid_cap), ...)``,
    # ascending. When the caller did not pin ``super_tile``,
    # ``QuickAllReduceInt4`` builds an engine per rung and selects by payload
    # size at launch.
    #
    # Keyed on world size because the rungs genuinely move with it: publishes
    # per rank are ``num_tiles / ST * 2(N-1)``, so the batching crossover
    # arrives sooner the wider the world. Empty means "one super-tile for every
    # size"; no schedule uses that any more, but the code path stays because
    # pinning ``super_tile`` still collapses to it.
    st_ladder: dict[int, tuple[tuple[int, int, int], ...]] | None = None

    def ladder_for(self, world_size: int) -> tuple[tuple[int, int, int], ...]:
        """Rungs for *world_size*; ``()`` when this schedule has no ladder."""
        if not self.st_ladder:
            return ()
        return self.st_ladder.get(int(world_size), ())


def _build_mesh(
    *,
    world_size,
    rank,
    super_tile,
    grid,
    inbox_memory,
    rs_codec,
    ag_codec,
    fusion="none",
    hidden=None,
):
    del rank  # a runtime kernel argument, not a mesh build knob
    if rs_codec != ag_codec:
        raise ValueError(
            f"mesh algorithm has one wire format for both laps, got "
            f"rs_codec={rs_codec!r} != ag_codec={ag_codec!r}"
        )
    return make_quick_allreduce_int4_kernel(
        world_size=world_size,
        super_tile=super_tile,
        grid=grid,
        inbox_memory=inbox_memory,
        codec=rs_codec,
        fusion=fusion,
        hidden=hidden,
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
    "ring": _Algorithm(
        name="ring",
        build=make_quick_allreduce_int4_ring_kernel,
        super_tiles=RING_SUPER_TILES,
        rs_codecs=RS_CODECS,
        ag_codecs=AG_CODECS,
        min_bytes=_RING_DEFAULT_MIN_PAYLOAD_BYTES,
        min_batch_blocks=_MIN_BATCH_BLOCKS,
        default_super_tile=8,
        st_ladder=RING_ST_LADDER,
        min_bytes_by_world=tuple(_RING_MIN_PAYLOAD_BYTES_BY_WORLD.items()),
    ),
}
DEFAULT_ALGORITHM = "mesh"

# World size at which a schedule's reduce-scatter lap needs INT6 to clear the
# 18 dB SQNR floor the schedules are held to.
#
# The ring's error grows with N -- it requantizes the running partial at every
# hop, and the partial's extremum grows with the contributions folded in -- so
# unlike the mesh it does not have one SQNR for every world size.
_RS_INT6_MIN_WORLD = 8


_warned_codecs: set[tuple[str, str, str]] = set()


def _warn_codec_unavailable(algo_name, label, requested, used):
    """Say it once per (schedule, lap, request), not once per engine.

    ``QuickAllReduceInt4`` builds one engine per super-tile rung, so a
    per-construction warning would fire several times for one object and again
    for every object -- for a condition that is a property of the schedule and
    cannot change within a process.
    """
    key = (algo_name, label, requested)
    if key in _warned_codecs:
        return
    _warned_codecs.add(key)
    logger.warning(
        "QuickAllReduceInt4: %s=%s does not apply to %s on algorithm=%r; using %r",
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
    ``AITER_ALL_REDUCE_CODEC=INT6``, and it cannot if the two are
    indistinguishable by the time they get here.

    A codec the selected schedule cannot build falls back with a warning rather
    than raising. An explicit argument still raises.
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

    resolved_rs = _pick(rs_codec, rs_default, algo.rs_codecs, "rs_codec")
    resolved_ag = _pick(ag_codec, ag_default, algo.ag_codecs, "ag_codec")
    return resolved_rs, resolved_ag


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
        logger.debug(
            "QuickAllReduceInt4: cannot read KFD topology; assuming xGMI",
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
            raise ValueError(f"QuickAllReduceInt4 requires a CUDA device, got {device}")
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
            f"QuickAllReduceInt4 does not support NCCL process groups (got "
            f"{backend!r} on group rank {rank}): IPC handle exchange requires "
            "CPU-side broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            "QuickAllReduceInt4 does not support multi-node process groups: HIP "
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


class QuickAllReduceInt4:
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_int4``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``algorithm`` selects the schedule. Both are two-shot -- reduce-scatter
    then all-gather -- so they are named for the topology of each lap:

    * ``"mesh"`` (default) -- each rank pushes to every one of the ``N-1``
      peers, twice. Two hops. Optimal on a meshed xGMI node.
    * ``"ring"`` -- ``2(N-1)`` hops, each a single contiguous run into exactly
      one peer's inbox. Same wire volume (``2(N-1)/N`` of the payload), traded
      for per-destination locality. Structurally worse at decode sizes and on
      xGMI -- opt in deliberately.

    ``rs_codec`` and ``ag_codec`` are the wire formats of the ring's two laps.
    The reduce-scatter lap is the only place the ring loses accuracy the mesh
    does not -- it requantizes ``N-1`` times where the mesh requantizes once --
    so it defaults to ``"int6"`` at TP8, where INT4 would cost too much
    accuracy. The all-gather lap forwards bytes verbatim and contributes a
    single quantization, so it defaults to ``"int4"`` everywhere and widens
    only by request.

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
        # ``None`` means "use the schedule's own policy", which for both is
        # the payload-size ladder. Passing a value pins one super-tile for
        # every size, which is what the benchmark variants and the tuning
        # sweeps do.
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
                f"QuickAllReduceInt4 supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
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
        world_ladder = algo.ladder_for(world_size)
        if world_ladder and not pinned_st:
            rungs = [(st, min(rung_cap, cap)) for _, st, rung_cap in world_ladder]
            ladder = world_ladder
            # The schedule's ``default_super_tile`` describes the *unladdered*
            # case and is not necessarily a rung: the TP8 ring runs ST=16 and
            # ST=32 and never 8. Take the bottom rung instead, so
            # ``super_tile`` names an engine that exists --
            # ``_by_st[self.super_tile]`` below indexes it directly, and
            # ``_pick_st`` falls back to it whenever the ladder does not apply.
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

        self._batch_publishes = (
            has_release_fence(resolved_inbox)
            if batch_publishes is None
            else bool(batch_publishes)
        )

        self.min_bytes = (
            algo.floor_bytes(self.world_size) if min_bytes is None else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        # ST=1 is always built: _pick_st falls back to it when a payload has
        # fewer tiles than the chosen super-tile. Engines are built in a fixed
        # order because each does its own IPC handle exchange, which is a
        # collective -- ranks disagreeing on the order would deadlock.
        #
        # The rungs go in first so an ST=1 that the ladder *sites* keeps its own
        # cap. Only then is the fallback filled in, and at the smallest cap on
        # the ladder rather than at the global default: the fallback fires only
        # when a payload has fewer tiles than the super-tile it would otherwise
        # take, so ``_grid_x`` there is bounded by that super-tile (<= 32) with
        # a release fence, and by the chosen rung's own cap without one --
        # under 128 either way. Seeding it with the 1216 default instead built a
        # 194 MiB inbox to launch at most 32 blocks into, and did it on every
        # object ever constructed.
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
        """IPC inbox bytes this object holds on *this* rank, across every rung.

        ``buf_bytes`` is the primary engine's alone, which understates a
        ladder-driven object by however many rungs it built. The total is what
        actually has to fit: the wire buffer is
        ``2(N-1) * grid * (ST * rank_atoms * tile + 64)``, so a high rung is
        large on its own and a sweep holding several tuning variants live at
        once is the realistic way to exhaust a device.
        """
        return sum(eng.buf_bytes for eng in self._by_st.values())

    def _ladder_st(self, live_bytes: int) -> int:
        """Super-tile the ladder assigns to a *live_bytes* payload.

        Publishes per rank are ``num_tiles / ST * 2(N-1)`` and cost a full L2
        writeback each, so a bigger payload wants a bigger ST -- but ST also
        divides the block count, so it cannot simply be maximised. The rungs
        and the measurements behind them are in ``MESH_ST_LADDER`` and
        ``RING_ST_LADDER``.
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
            raise TypeError("QuickAllReduceInt4 requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("QuickAllReduceInt4 supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("QuickAllReduceInt4 requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("QuickAllReduceInt4 requires contiguous input/output")
        inp_ptr = int(inp.data_ptr())
        out_ptr = int(out.data_ptr())
        if inp_ptr % 16 != 0 or out_ptr % 16 != 0:
            raise ValueError("QuickAllReduceInt4 requires 16-byte-aligned input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes > 0xFFFFFFFF:
            raise ValueError(
                "QuickAllReduceInt4 payload must not exceed the 4 GiB buffer window"
            )
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
            raise ValueError("QuickAllReduceInt4 requires non-overlapping input/output")
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

        ``<jit symbol>/g<grid_cap>/x<grid_x>``. Resolves the super-tile through
        the same ``_pick_st`` the launch path uses, so for a ladder-driven
        engine this is the only way to see which rung a given size takes --
        ``super_tile`` on the object is the *nominal* value, not the one a
        particular payload gets. Pure: builds nothing and launches nothing.
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
        """Two-shot INT4 all-reduce into ``out``.

        ``stream=None`` uses the current PyTorch stream on this device.
        """
        live_bytes = self._check_payload(inp, out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"QuickAllReduceInt4.allreduce got a {live_bytes} B payload, "
                f"below the {self.min_bytes} B floor: at decode sizes this "
                "kernel saves a few microseconds on a collective that is not "
                "the bottleneck, and charges ~36 dB of SQNR for them. Route "
                "small messages to an exact all-reduce, or pass min_bytes=0 "
                "to override."
            )
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles, live_bytes)
        self._launch_eng(self._by_st[st], inp, out, stream, live_bytes=live_bytes)


class QuickAllReduceInt4RMSNorm:
    """Quantized all-reduce fused with residual-add and RMSNorm.

    Per token row::

        acc = allreduce(input)         (quantized wire, dequantized to fp16)
        acc = float(bf16(acc))         (deliberate: matches the unfused path)
        acc += residual_in
        residual_out = bf16(acc)
        out = bf16(acc * rsqrt(sum(acc^2)/hidden + eps) * weight)

    Defaults to ``algorithm="ring"``, the schedule whose structure the
    row-sized block was chosen for -- but **measure before choosing it**. On
    MI350P at TP8, fusing pays on the mesh and costs on the ring:

        1024x7168   fused mesh 323 us   separate mesh 383 us   (1.18x)
                    fused ring 509 us   separate ring 383 us   (0.75x)
        4096x8192   fused mesh 1398 us  separate mesh 1693 us  (1.21x)
                    fused ring 1281 us  separate ring 1110 us  (0.87x)

    The asymmetry is structural. The mesh runs its epilogue once per tile, as a
    tail after the all-gather, on values already in registers -- so it collects
    the two saved HBM passes and nothing else changes. The ring runs it inside
    the ``2(N-1)``-op pipeline, once per chunk, putting barriers and norm
    arithmetic between one receive and the next; that pipeline is what the
    ring's throughput is, and disturbing it costs more than the passes save.

    Note the mesh is also ~2.4 dB noisier in that comparison, for an unrelated
    reason: its codec defaults to INT4 on both laps where the ring widens its
    reduce-scatter lap to INT6 at TP8.

    Unlike ``QuickAllReduceInt4`` the *geometry* depends on ``hidden``, not just
    the epilogue -- the block is sized so one 16 B atom is one token row (see
    ``quick_allreduce_fusions.quick_reduce_row_block``). A new hidden therefore
    needs a new engine, and building one is a collective (IPC handle exchange).
    Engines are built lazily on first use, which is safe because TP ranks enter
    a collective with the same shape in lockstep; pass ``hiddens=(...)`` to
    build them up front instead.

    Supported widths are multiples of 1024 up to 8192 -- 7168 and 5120
    included, which the tile-aligned formulation this replaces could not do.
    """

    #: Fused launches are capped at one workgroup per CU. That is not a
    #: performance choice: a block spins on flags written by the *same block id*
    #: on every peer, so every launched block must be resident on every rank at
    #: once or the ranks deadlock. At one workgroup per CU that needs no
    #: residency model -- either a single workgroup fits, or HIP fails the
    #: launch out of resources, which is loud and immediate. It costs nothing
    #: here: both ST ladders already cap the grid at 128, and a 14-wave fused
    #: workgroup puts more waves on a CU than the 4-wave plain one does at the
    #: same block count.
    WGS_PER_CU = 1

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        algorithm: str = "ring",
        super_tile: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        batch_publishes: bool | None = None,
        max_bytes: int | None = None,
        rs_codec: str | None = None,
        ag_codec: str | None = None,
        hiddens: tuple[int, ...] = (),
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
        pinned_st = super_tile is not None
        if super_tile is not None and super_tile not in algo.super_tiles:
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
                f"QuickAllReduceInt4RMSNorm supports {', '.join(_SUPPORTED_ARCHS)}, "
                f"got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        self._device_index = _cuda_index(device)
        torch.cuda.set_device(self._device_index)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.algorithm = algorithm
        self.rs_codec = rs_codec
        self.ag_codec = ag_codec
        self.inbox_memory = resolved_inbox
        self.arch = arch
        self._algo = algo
        self._inbox_flags = inbox_flags
        self._has_launched = False
        self._lds_capacity = get_lds_capacity_bytes(arch)
        self._cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )
        self._batch_publishes = (
            has_release_fence(resolved_inbox)
            if batch_publishes is None
            else bool(batch_publishes)
        )
        # The plain class's floor is an accuracy gate on the codec, which the
        # epilogue does not change; the ceiling is None because nothing here
        # pushes the whole payload to every peer.
        self.min_bytes = algo.floor_bytes(self.world_size)
        self.max_bytes = max_bytes

        # Rungs to build, as ``(super_tile, grid_cap)``. Same ladder the plain
        # class walks; pinning ``super_tile`` collapses it to one rung.
        world_ladder = algo.ladder_for(self.world_size)
        if world_ladder and not pinned_st:
            self._ladder = world_ladder
            self._rungs = [(st, min(rung_cap, cap)) for _, st, rung_cap in world_ladder]
        else:
            st = algo.default_super_tile if super_tile is None else int(super_tile)
            self._ladder = ()
            self._rungs = [(st, cap)]
        # ST=1 is the fallback when a payload has fewer tiles than the chosen
        # super-tile, exactly as in QuickAllReduceInt4.
        by_cap = {}
        for st, rung_cap in self._rungs:
            by_cap.setdefault(st, rung_cap)
        by_cap.setdefault(1, min(by_cap.values()))
        self._by_cap = dict(sorted(by_cap.items()))

        # (hidden, super_tile) -> (engine, spec)
        self._by_cfg: dict[tuple, tuple] = {}
        try:
            for h in sorted({int(x) for x in hiddens}):
                self._build_hidden(h)
        except Exception:
            self.close()
            raise

    # -- engine construction -------------------------------------------------

    def supports_hidden(self, hidden: int) -> bool:
        """Whether a fused build exists for *hidden* at this world size."""
        return quick_reduce_hidden_supported(int(hidden), self.world_size)

    def _grid_for(self, super_tile: int, rung_cap: int, block: int) -> int:
        """Persistent grid for one rung: the minimum of every bound we have.

        One workgroup per CU is the bound that does not rest on a model; see
        ``WGS_PER_CU``. ``clamp_grid_cap`` is kept above it as a second opinion,
        and the group-wide MIN makes the ranks agree -- which is the other half
        of the co-residency invariant, since a block waits on its own id at
        every peer.
        """
        grid = clamp_grid_cap(
            rung_cap,
            arch=self.arch,
            world_size=self.world_size,
            super_tile=super_tile,
            cu_count=self._cu_count,
            block=block,
        )
        grid = min(grid, self.WGS_PER_CU * self._cu_count)
        shared = torch.tensor(grid, dtype=torch.int64)
        dist.all_reduce(shared, op=dist.ReduceOp.MIN, group=self.group)
        return max(1, int(shared.item()))

    def _build_hidden(self, hidden: int) -> None:
        """Build every rung for *hidden*.

        A collective: each rung does its own IPC handle exchange and a grid
        MIN-reduce, so every rank must reach this with the same hidden and walk
        the rungs in the same order.
        """
        hidden = int(hidden)
        if not self.supports_hidden(hidden):
            # Resolve again for the message: it names the constraint that failed.
            quick_reduce_row_block(hidden, self.world_size)
        block, _atoms_per_row = quick_reduce_row_block(hidden, self.world_size)
        for st in self._by_cap:
            key = (hidden, st)
            if key in self._by_cfg:
                continue
            spec = self._algo.build(
                world_size=self.world_size,
                rank=self.rank,
                super_tile=st,
                grid=self._grid_for(st, self._by_cap[st], block),
                inbox_memory=self.inbox_memory,
                rs_codec=self.rs_codec,
                ag_codec=self.ag_codec,
                fusion="rmsnorm",
                hidden=hidden,
            )
            if spec["lds_bytes"] > self._lds_capacity:
                raise ValueError(
                    f"fused {self._algo.name} at hidden={hidden} needs "
                    f"{spec['lds_bytes']} B of LDS, over {self.arch}'s "
                    f"{self._lds_capacity} B per workgroup: use a narrower "
                    "hidden, a narrower codec, or the ring schedule"
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

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every engine."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    @property
    def hiddens(self) -> tuple[int, ...]:
        return tuple(sorted({k[0] for k in self._by_cfg}))

    # -- selection -----------------------------------------------------------

    def _tile_bytes(self, hidden: int) -> int:
        """Tile size for *hidden*, building its engines if they do not exist.

        Build-derived rather than a constant: a fused tile is ``ATOMS`` token
        rows of this width, not the plain schedule's 32 KiB, so the tile count
        a launch derives has to come from the engine.
        """
        key = (int(hidden), 1)  # ST=1 is always built; see the constructor
        if key not in self._by_cfg:
            self._build_hidden(int(hidden))
        return self._by_cfg[key][1]["tile_bytes"]

    def _ladder_st(self, live_bytes: int) -> int:
        st = min(self._by_cap)
        for floor, rung_st, _cap in self._ladder:
            if live_bytes >= floor:
                st = rung_st
        return st

    def _pick_cfg(self, hidden: int, live_bytes: int, num_tiles: int) -> tuple:
        want = self._ladder_st(live_bytes) if self._ladder else self._rungs[0][0]
        if want != 1:
            if self._batch_publishes:
                want = want if num_tiles >= want else 1
            else:
                key = (int(hidden), want)
                built = self._by_cfg.get(key)
                grid = built[0].grid if built else self._by_cap[want]
                want = want if num_tiles > grid else 1
        key = (int(hidden), want)
        if key not in self._by_cfg:
            self._build_hidden(int(hidden))
        return key

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int) -> int:
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), self._algo.min_batch_blocks)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, grid))

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
                raise TypeError(
                    f"QuickAllReduceInt4RMSNorm requires a Tensor for {name}"
                )
            if t.dtype != torch.bfloat16:
                raise ValueError(
                    f"QuickAllReduceInt4RMSNorm is bf16-only, {name} is {t.dtype}"
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
            quick_reduce_row_block(hidden, self.world_size)  # raises, naming why
        live_bytes = int(inp.numel()) * 2
        if live_bytes > 0xFFFFFFFF:
            raise ValueError("payload must not exceed the 4 GiB buffer window")

        # Aliasing. ``residual_out is residual_in`` is allowed: a thread writes
        # only the bytes it read, and vLLM's fused_add_rmsnorm is in-place on
        # the residual. Anything else that overlaps is a race -- out and
        # residual_out are written from the same thread, and the input is still
        # being read by peers when the epilogue runs.
        spans = {
            "inp": inp,
            "out": out,
            "residual_in": residual_in,
        }
        if residual_out.data_ptr() != residual_in.data_ptr():
            spans["residual_out"] = residual_out
        elif tuple(residual_out.shape) != tuple(residual_in.shape):
            raise ValueError("an in-place residual must have residual_in's shape")
        items = [
            (n, int(t.data_ptr()), int(t.data_ptr()) + live_bytes)
            for n, t in spans.items()
        ]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if max(items[i][1], items[j][1]) < min(items[i][2], items[j][2]):
                    raise ValueError(
                        f"{items[i][0]} and {items[j][0]} overlap; only "
                        "residual_out aliasing residual_in exactly is allowed"
                    )
        return hidden, live_bytes

    def _launch_eng(self, eng, spec, args) -> None:
        """Run *args*, compiling under a waves-per-EU floor on the first call.

        The hint is what makes "one workgroup fits a CU" the compiler's problem
        rather than an assumption: a fused workgroup is
        ``ceil(block/WAVE/4)`` waves deep on its busiest SIMD, and asking for
        that many waves per EU caps the register allocation to match. It only
        has an effect on the compile, so the steady-state path skips the
        context entirely.
        """
        self._has_launched = True
        if getattr(eng.launch, "_cf", None) is not None:
            _run_compiled(eng.launch, *args)
            return
        waves_per_eu = -(-(spec["block"] // WAVE) // 4)
        with CompilationContext.compile_hints({"waves_per_eu": waves_per_eu}):
            _run_compiled(eng.launch, *args)

    def _launch(
        self, inp, residual_in, weight, eps, out, residual_out, stream, *, hidden
    ):
        live_bytes = int(inp.numel()) * 2
        num_tiles = max(1, -(-live_bytes // self._tile_bytes(hidden)))
        eng, spec = self._by_cfg[self._pick_cfg(hidden, live_bytes, num_tiles)]
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
            Int32(self._grid_x(num_tiles, spec["super_tile"], eng.grid)),
            Int64(int(residual_in.data_ptr())),
            Int64(int(residual_out.data_ptr())),
            Int64(int(weight.data_ptr())),
            Float32(float(eps)),
            stream,
        )
        self._launch_eng(eng, spec, args)

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
                f"QuickAllReduceInt4RMSNorm got a {live_bytes} B payload, below "
                f"the {self.min_bytes} B floor: at decode sizes this kernel "
                "charges real SQNR for microseconds that are not the "
                "bottleneck. Route small messages to OneShotAllReduceRMSNorm."
            )
        self._launch(
            inp, residual_in, weight, eps, out, residual_out, stream, hidden=hidden
        )
        return out, residual_out

    def compile_and_launch(self, inp, residual_in, weight, eps=1e-6, stream=None):
        """Eager-JIT every rung for this shape and run each once.

        A real collective -- every rank must call it with the same shape. Used
        by the bench and the op tests to force a warm launch before timing or a
        graph capture, where a first-call JIT would be fatal. Not gated by
        ``min_bytes``: a warmup tensor is allowed to be small.
        """
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual_in)
        hidden, live_bytes = self._check(inp, residual_in, weight, out, residual_out)
        self._build_hidden(hidden)
        for (h, _st), (eng, spec) in self._by_cfg.items():
            if h != hidden:
                continue
            tile_bytes = spec["tile_bytes"]
            num_tiles = max(1, (live_bytes + tile_bytes - 1) // tile_bytes)
            st = (
                Stream(torch.cuda.current_stream(self._device_index))
                if stream is None
                else (stream if isinstance(stream, Stream) else Stream(stream))
            )
            self._launch_eng(
                eng,
                spec,
                (
                    Int32(self.rank),
                    Int64(live_bytes),
                    Int32(num_tiles),
                    Int64(int(inp.data_ptr())),
                    Int64(int(out.data_ptr())),
                    Int64(int(eng._gpu_peer_ptrs)),
                    Int64(int(eng._colors)),
                    Int32(self._grid_x(num_tiles, spec["super_tile"], eng.grid)),
                    Int64(int(residual_in.data_ptr())),
                    Int64(int(residual_out.data_ptr())),
                    Int64(int(weight.data_ptr())),
                    Float32(float(eps)),
                    st,
                ),
            )
        return out, residual_out

    def variant(self, hidden: int, nbytes: int) -> str:
        """``<jit symbol>/g<grid>/x<grid_x>`` for this (hidden, payload)."""
        hidden = int(hidden)
        num_tiles = max(1, -(-int(nbytes) // self._tile_bytes(hidden)))
        eng, spec = self._by_cfg[self._pick_cfg(hidden, int(nbytes), num_tiles)]
        grid_x = self._grid_x(num_tiles, spec["super_tile"], eng.grid)
        return f"{kernel_symbol(eng.launch)}/g{eng.grid}/x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        if self.max_bytes is not None and int(nbytes) > self.max_bytes:
            return False
        return int(nbytes) >= self.min_bytes

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
