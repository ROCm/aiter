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

Two more tuning knobs ride every ladder rung: ``block`` (threads per workgroup,
which sets the tile) and, on the mesh only, ``skip_self`` (no round trip through
this rank's own inbox). Ladders are keyed on ``(link, world_size)``.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes

from .kernels.quick_allreduce_codec import SUPPORTED_BLOCKS
from .kernels.quick_allreduce_int4 import (
    MESH_CODECS,
    SUPER_TILES,
    clamp_grid_cap,
    make_quick_allreduce_int4_kernel,
    mesh_st_ladder,
)
from .kernels.quick_allreduce_int4_ring import (
    AG_CODECS,
    RING_SUPER_TILES,
    RS_CODECS,
    make_quick_allreduce_int4_ring_kernel,
    ring_st_ladder,
)
from .kernels.quick_allreduce_shared import (
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
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
# Production dispatch does not consult it; the FlyDSL backend inside
# ``QuickAllReduce`` owns the real boundary via ``allreduce_policy.resolve_quant``,
# which is additionally keyed on link type.
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

    ``build`` is keyword-only and always receives ``rank``, ``rs_codec``,
    ``ag_codec``, ``block`` and ``skip_self``, whether or not a given schedule
    uses them. The ring bakes ``rank`` in at compile time -- the chunk a step
    operates on is ``(rank - step) % N``, which has to be a Python constant to
    index a register-resident atom list -- while the mesh takes it as a runtime
    kernel argument and bakes it in only under ``skip_self``.
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

    # ``(world_size, link) -> ((min_payload_bytes, super_tile, grid_cap,
    # block, skip_self), ...)``, ascending. When the caller did not pin
    # ``super_tile``, ``QuickAllReduceInt4`` builds an engine per rung and
    # selects by payload size at launch.
    #
    # Keyed on world size because the rungs genuinely move with it: publishes
    # per rank are ``num_tiles / ST * 2(N-1)``, so the batching crossover
    # arrives sooner the wider the world. Keyed on link because PCIe and xGMI
    # are tuned separately. An empty ladder means "one super-tile for every
    # size"; no schedule uses that any more, but the code path stays because
    # pinning ``super_tile`` still collapses to it.
    st_ladder: Callable[[int, str], tuple] | None = None
    # Whether the schedule has a round trip through its own inbox to skip.
    supports_skip_self: bool = False

    def ladder_for(self, world_size: int, link: str = "pcie") -> tuple:
        """Rungs for *(link, world_size)*; ``()`` when there is no ladder."""
        if self.st_ladder is None:
            return ()
        return tuple(self.st_ladder(int(world_size), str(link)))


def _build_mesh(
    *,
    world_size,
    rank,
    super_tile,
    grid,
    inbox_memory,
    rs_codec,
    ag_codec,
    block,
    skip_self,
):
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
        block=block,
        skip_self=skip_self,
        rank=rank if skip_self else None,
    )


def _build_ring(*, skip_self, **kw):
    if skip_self:
        raise ValueError(
            "skip_self does not apply to the ring: it never writes its own "
            "inbox, so there is no round trip to skip"
        )
    return make_quick_allreduce_int4_ring_kernel(**kw)


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
        st_ladder=mesh_st_ladder,
        supports_skip_self=True,
    ),
    "ring": _Algorithm(
        name="ring",
        build=_build_ring,
        super_tiles=RING_SUPER_TILES,
        rs_codecs=RS_CODECS,
        ag_codecs=AG_CODECS,
        min_bytes=_RING_DEFAULT_MIN_PAYLOAD_BYTES,
        min_batch_blocks=_MIN_BATCH_BLOCKS,
        default_super_tile=8,
        st_ladder=ring_st_ladder,
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


def _resolve_codecs(algo, world_size, rs_codec, ag_codec):
    """Codecs for one engine: explicit argument > per-N default.

    ``None`` means "not specified", and the caller is expected to pass it
    whenever it wants the schedule's own default rather than a pinned wire
    format. That distinction matters at TP8, where the ring's reduce-scatter
    lap defaults to INT6: an explicit ``rs_codec="int4"`` there is a real
    downgrade, not a restatement of the default.

    A codec the selected schedule cannot build raises. The per-N *default*
    silently narrows to what the schedule supports -- the mesh has no INT6
    path, so TP8's INT6 reduce-scatter default does not leak into it.
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

    The answer is per host, not per process group: one xGMI link anywhere
    classifies every group on the node as xGMI. That assumes a node is uniformly
    xGMI or uniformly PCIe, which holds for the single-node systems this targets.
    On a mixed host, a group of PCIe-only peers would get the xGMI policy and an
    uncached inbox; fixing that means classifying only the links between the
    group's own devices.
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
    if isinstance(device, str):
        device = torch.device(device)
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
    """The JIT symbol a kernel factory stamped on its launch wrapper."""
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
        self.block = spec["block"]
        self.skip_self = spec.get("skip_self", False)
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

    Leave both ``None`` to get those defaults.

    ``inbox_memory`` selects how the IPC inbox is allocated:

    * ``"auto"`` (default) -- ``uncached`` on hosts with xGMI peer links,
      ``finegrained`` on PCIe-attached hosts. Decided from the KFD topology.
    * ``"uncached"`` -- correct everywhere, but peer writes collapse on PCIe.
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so each
      publish writes the payload back from the writer's L2 with a release fence
      before the flag goes out write-through (``sc0 sc1``).

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to ``MIN_PAYLOAD_BYTES``. ``compile_and_launch`` is deliberately
    not gated: its warmup tensor is allowed to be small.

    ``link`` selects the tuning ladder and is detected from the KFD topology
    when not given. ``block`` and ``skip_self`` override those knobs on every
    rung, ``None`` leaving each rung's own value; ``skip_self`` is mesh-only.
    Under ``skip_self`` the mesh specialises its binary to this rank, so the
    JIT symbol carries an ``_r<n>_`` field.
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
        link: str | None = None,
        block: int | None = None,
        skip_self: bool | None = None,
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
        if link is None:
            link = "xgmi" if has_xgmi_peer_links() else "pcie"
        if link not in ("pcie", "xgmi"):
            raise ValueError(f"link must be 'pcie' or 'xgmi', got {link!r}")
        if block is not None and block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        if skip_self and not algo.supports_skip_self:
            raise ValueError(
                f"skip_self does not apply to algorithm={algorithm!r}: it never "
                "writes its own inbox, so there is no round trip to skip"
            )
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

        def _knobs(rung_block, rung_skip):
            """A rung's ``(block, skip_self)`` with the caller's overrides."""
            b = int(rung_block if block is None else block)
            ss = bool(rung_skip if skip_self is None else skip_self)
            return b, ss

        # Rungs to build, each ``(min_bytes, super_tile, grid_cap, block,
        # skip_self)``. Pinning ``super_tile`` collapses the ladder to that one
        # rung -- a caller who named a super-tile gets exactly it, at every
        # size.
        #
        # ``grid_cap`` is a *ceiling*, not a pin: it bounds every rung rather
        # than disabling size-dependent selection. Raising it above a rung's own
        # cap is a no-op (the rung cap is already sized so ``_grid_x`` never
        # binds over that rung's payload range), while lowering it constrains
        # the wire buffer, which is what a caller passing it usually wants.
        world_ladder = algo.ladder_for(world_size, link)
        if world_ladder and not pinned_st:
            ladder = tuple(
                (floor, st, min(rung_cap, cap), *_knobs(b, ss))
                for floor, st, rung_cap, b, ss in world_ladder
            )
        else:
            ladder = ((0, int(super_tile), cap, *_knobs(BLOCK, False)),)
        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory)
        self._device_index = _cuda_index(device)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.link = link
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
        lds_capacity = get_lds_capacity_bytes(arch)

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

        # One engine per distinct ``(super_tile, block, skip_self)``, keyed
        # that way in ``_by_cfg``. Each ``(block, skip_self)`` also gets an ST=1
        # engine: _pick_cfg falls back to it when a payload has fewer tiles than
        # the chosen super-tile, and the fallback has to share the rung's block
        # -- a different tile size would change the tile count it was picked
        # for. Engines are built in a fixed order because each does its own IPC
        # handle exchange, which is a collective -- ranks disagreeing on the
        # order would deadlock.
        #
        # The rungs go in first so an ST=1 that the ladder *sites* keeps its own
        # cap. Only then is the fallback filled in, and at the smallest cap
        # among its rungs rather than at the global default: the fallback fires
        # only when a payload has fewer tiles than the super-tile it would
        # otherwise take, so ``_grid_x`` there is bounded by that super-tile
        # (<= 32) with a release fence, and by the chosen rung's own cap without
        # one -- under 128 either way. Seeding it with the 1216 default instead
        # built a 194 MiB inbox to launch at most 32 blocks into, and did it on
        # every object ever constructed.
        caps = {}
        for _floor, st, rung_cap, b, ss in ladder:
            caps.setdefault((st, b, ss), rung_cap)
        for b, ss in {(b, ss) for _st, b, ss in list(caps)}:
            caps.setdefault(
                (1, b, ss),
                min(c for (_st, cb, css), c in caps.items() if (cb, css) == (b, ss)),
            )
        self._ladder = ladder if (world_ladder and not pinned_st) else ()
        self._primary = ladder[0][1:2] + ladder[0][3:]
        self._by_cfg = {}
        try:
            with torch.cuda.device(self._device_index):
                for key in sorted(caps):
                    st, b, ss = key
                    # A persistent kernel deadlocks if it launches more workgroups
                    # than fit, and the ranks have to agree on the number: take the
                    # minimum across the group so a heterogeneous node converges.
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
                        rs_codec=rs_codec,
                        ag_codec=ag_codec,
                        block=b,
                        skip_self=ss,
                    )
                    if spec["lds_bytes"] > lds_capacity:
                        raise ValueError(
                            f"{algorithm} {rs_codec} at block={b} needs "
                            f"{spec['lds_bytes']} B of LDS, over the "
                            f"{lds_capacity} B {arch} has"
                        )
                    self._by_cfg[key] = _StEngine(
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

        primary = self._by_cfg[self._primary]
        self.super_tile, self.block, self.skip_self = self._primary
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
        return sum(eng.buf_bytes for eng in self._by_cfg.values())

    def _ladder_cfg(self, live_bytes: int) -> tuple[int, int, bool]:
        """``(super_tile, block, skip_self)`` the ladder assigns to *live_bytes*.

        Publishes per rank are ``num_tiles / ST * 2(N-1)`` and cost a full L2
        writeback each, so a bigger payload wants a bigger ST -- but ST also
        divides the block count, so it cannot simply be maximised. The rungs
        and the measurements behind them are in ``MESH_ST_LADDER`` and
        ``RING_ST_LADDER``.
        """
        cfg = self._primary
        for floor, st, _cap, b, ss in self._ladder:
            if live_bytes >= floor:
                cfg = (st, b, ss)
        return cfg

    @staticmethod
    def _num_tiles(live_bytes: int, block: int) -> int:
        tile_bytes = int(block) * ATOMS * 16
        return max(1, (live_bytes + tile_bytes - 1) // tile_bytes)

    def _pick_cfg(self, live_bytes: int) -> tuple[tuple[int, int, bool], int]:
        """Engine key for a *live_bytes* payload, and its tile count.

        The ladder chooses the rung, which fixes the block and so the tile
        count; the tile count then only has to confirm there is a whole
        super-tile to take, falling back to the same block's ST=1 engine when
        there is not.

        Without a release fence a publish is nearly free, so the only reason to
        batch tiles is when there are more of them than blocks -- prefer ST=1
        and the parallelism it buys.

        With one, that trade inverts: every publish costs a full L2 writeback,
        and ST=1 pays one per tile per phase. Take a super-tile as soon as
        there is a whole one to take. Measured on MI350P at 1024x7168, TP4:
        577.71 us at ST=1 against 269.01 at ST=8.
        """
        want, b, ss = self._ladder_cfg(live_bytes)
        num_tiles = self._num_tiles(live_bytes, b)
        if want == 1:
            st = 1
        elif self._batch_publishes:
            st = want if num_tiles >= want else 1
        else:
            st = want if num_tiles > self._by_cfg[(want, b, ss)].grid else 1
        return (st, b, ss), num_tiles

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
        num_tiles = max(1, (live_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        args = self._launch_args(
            eng, inp, out, stream, live_bytes=live_bytes, num_tiles=num_tiles
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch, *args)

    def compile_and_launch(self, inp, out=None, stream=None) -> None:
        """Eager-JIT every engine's binary and launch each of them once, 
        against *inp*/*out*.

        This runs every engine on the GPU -- ``out`` ends up holding whichever
        one ran last, and it is a real collective: every rank must call it with
        the same shape. Used by ``bench_comm_allreduce.py`` and the op tests to
        force a real warm launch before timing or correctness checks begin.
        Production never calls this: it tolerates the first real call paying a
        JIT-compile cost instead.
        """
        if out is None:
            out = torch.empty_like(inp)
        live_bytes = self._check_payload(inp, out)
        for eng in self._by_cfg.values():
            self._launch_eng(eng, inp, out, stream, live_bytes=live_bytes)

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
        cfg, _num_tiles = self._pick_cfg(live_bytes)
        self._launch_eng(self._by_cfg[cfg], inp, out, stream, live_bytes=live_bytes)
