# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4/INT6 all-reduce.

Public type ``FlyQuickAllReduce``, with two interchangeable schedules selected
by ``algorithm``. Both are two-shot -- reduce-scatter then all-gather -- so
they are named for the topology of each lap instead:

* ``"mesh"`` the default: fanout to all N-1 peers, twice.
* ``"ring"`` 2(N-1) single-destination hops.

Super-tile ST∈{1,8} on the mesh, ST∈{1,8,16,32} on the ring. INT4 nibble or
INT6 bit-plane pair, both with group-16 E4M3 scales. Payload HBM is bf16.

One more tuning knob rides every ladder rung: ``block`` (threads per workgroup,
which sets the tile). Ladders are keyed on ``(link, world_size)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes

from .allreduce_shared import (
    _SUPPORTED_ARCHS,
    _cuda_index,
    _LadderedIpcOp,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    has_xgmi_peer_links,
    kernel_symbol,
    payload_probes,
)
from .kernels.collectives_shared import (
    BLOCK,
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    WAVE,
    WORLD,
    clamp_grid_cap,
    has_release_fence,
)
from .kernels.quick_allreduce_codec import CODECS, SUPPORTED_BLOCKS
from .kernels.quick_allreduce_fusions import (
    PAD_MASK_MAX_BYTES as _PAD_MASK_MAX_BYTES,
)
from .kernels.quick_allreduce_fusions import (
    quick_reduce_padded_row_block_options,
    quick_reduce_row_block_at,
    quick_reduce_row_block_for,
    quick_reduce_row_block_options,
)
from .kernels.quick_allreduce_mesh import (
    MESH_CODECS,
    SUPER_TILES,
    make_quick_allreduce_mesh_kernel,
    mesh_fanout_fits,
    mesh_st_ladder,
)
from .kernels.quick_allreduce_ring import (
    AG_CODECS,
    RING_SUPER_TILES,
    RS_CODECS,
    make_quick_allreduce_ring_kernel,
    ring_st_ladder,
)
from .kernels.tensor_shim import _preload_compiled, _run_compiled

logger = logging.getLogger("aiter")

# Smallest payload sent through this kernel, in bytes.
MIN_PAYLOAD_BYTES = 128 << 10

# Floor on the block count when batching publishes into super-tiles; see
# ``FlyQuickAllReduce._grid_x``. Shrinking the grid trades parallelism for
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
# This is only the *standalone* guard rail -- what ``FlyQuickAllReduce``
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

    Everything ``FlyQuickAllReduce`` does *around* the kernel -- IPC setup,
    payload validation, one engine per super-tile, launch -- is identical
    across schedules and stays on the class. What differs is which kernel
    factory to call, which super-tile values and wire formats that factory
    accepts, and where the size floor sits. That is this record.

    ``build`` is keyword-only and always receives ``rank``, ``rs_codec``,
    ``ag_codec`` and ``block``, whether or not a given schedule uses them. Both
    bake ``rank`` in at compile time. The ring needs it because the chunk a
    step operates on is ``(rank - step) % N``, which has to be a Python
    constant to index a register-resident atom list. The mesh needs it to keep
    its own share of every tile in registers instead of round-tripping it
    through its own inbox.
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
    fused_block_ok: Callable[[int, int, str], bool] | None = None

    def floor_bytes(self, world_size: int) -> int:
        return dict(self.min_bytes_by_world).get(int(world_size), self.min_bytes)

    # ``(world_size, link) -> ((min_payload_bytes, super_tile, grid_cap,
    # block), ...)``, ascending. When the caller did not pin ``super_tile``,
    # ``FlyQuickAllReduce`` builds an engine per rung and selects by payload
    # size at launch.
    #
    # Keyed on world size because the rungs genuinely move with it: publishes
    # per rank are ``num_tiles / ST * 2(N-1)``, so the batching crossover
    # arrives sooner the wider the world. Keyed on link because PCIe and xGMI
    # are tuned separately. An empty ladder means "one super-tile for every
    # size"; no schedule uses that any more, but the code path stays because
    # pinning ``super_tile`` still collapses to it.
    st_ladder: Callable[[int, str], tuple] | None = None

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
    block=None,
    fusion="none",
    hidden=None,
    h_pad=None,
):
    if rs_codec != ag_codec:
        raise ValueError(
            f"mesh algorithm has one wire format for both laps, got "
            f"rs_codec={rs_codec!r} != ag_codec={ag_codec!r}"
        )
    return make_quick_allreduce_mesh_kernel(
        world_size=world_size,
        super_tile=super_tile,
        grid=grid,
        inbox_memory=inbox_memory,
        codec=rs_codec,
        block=block,
        rank=rank,
        fusion=fusion,
        hidden=hidden,
        h_pad=h_pad,
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
        st_ladder=mesh_st_ladder,
        fused_block_ok=mesh_fanout_fits,
    ),
    "ring": _Algorithm(
        name="ring",
        build=make_quick_allreduce_ring_kernel,
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


def batches_publishes(inbox_memory: str, algorithm: str, link: str) -> bool:
    """Whether ``FlyQuickAllReduce`` batches publishes into super-tiles.

    Always with a release fence, where every publish is an L2 writeback. The
    PCIe ring batches without one too: each of its ``2(N-1)`` hops ends in a
    handshake that is a PCIe round trip whether or not a writeback precedes it.
    """
    return has_release_fence(inbox_memory) or (algorithm == "ring" and link == "pcie")


class FlyQuickAllReduce(_LadderedIpcOp):
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_mesh``.

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
      ``finegrained`` on PCIe-attached hosts, decided from the KFD topology.
      TP2 is ``uncached`` on PCIe too: one remote peer cannot collapse.
    * ``"uncached"`` -- correct everywhere, but peer writes collapse on PCIe.
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so each
      publish writes the payload back from the writer's L2 with a release fence
      before the flag goes out write-through (``sc0 sc1``).

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to ``MIN_PAYLOAD_BYTES``.

    ``link`` selects the tuning ladder and is detected from the KFD topology
    when not given. ``block`` overrides that knob on every rung, ``None``
    leaving each rung's own value. Both schedules specialise their binary to
    this rank, so the JIT symbol carries an ``_r<n>_`` field.
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
        link: str | None = None,
        block: int | None = None,
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
                f"FlyQuickAllReduce supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")

        def _block(rung_block):
            """A rung's ``block`` with the caller's override."""
            return int(rung_block if block is None else block)

        # Rungs to build, each ``(min_bytes, super_tile, grid_cap, block)``.
        # Pinning ``super_tile`` collapses the ladder to that one rung -- a
        # caller who named a super-tile gets exactly it, at every size.
        #
        # ``grid_cap`` is a *ceiling*, not a pin: it bounds every rung rather
        # than disabling size-dependent selection. Raising it above a rung's own
        # cap is a no-op (the rung cap is already sized so ``_grid_x`` never
        # binds over that rung's payload range), while lowering it constrains
        # the wire buffer, which is what a caller passing it usually wants.
        world_ladder = algo.ladder_for(world_size, link)
        if world_ladder and not pinned_st:
            ladder = tuple(
                (floor, st, min(rung_cap, cap), _block(b))
                for floor, st, rung_cap, b in world_ladder
            )
        else:
            ladder = ((0, int(super_tile), cap, _block(BLOCK)),)
        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory, world_size)
        self.device = torch.device("cuda", _cuda_index(device))
        self.link = link
        self.inbox_memory = resolved_inbox
        self.algorithm = algorithm
        self.rs_codec = rs_codec
        self.ag_codec = ag_codec
        self._algo = algo

        self.min_bytes = (
            algo.floor_bytes(int(world_size)) if min_bytes is None else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        def _build(*, super_tile, grid, block):
            return algo.build(
                world_size=int(world_size),
                rank=int(rank),
                super_tile=super_tile,
                grid=grid,
                inbox_memory=resolved_inbox,
                rs_codec=rs_codec,
                ag_codec=ag_codec,
                block=block,
            )

        self._init_engines(
            group=group,
            device_index=self.device.index,
            rank=rank,
            world_size=world_size,
            ladder=ladder,
            laddered=bool(world_ladder) and not pinned_st,
            build=_build,
            inbox_flags=inbox_flags,
            arch=arch,
            batch_publishes=batches_publishes(resolved_inbox, algorithm, link),
            min_batch_blocks=algo.min_batch_blocks,
            grid_cap=cap,
            label=f"{algorithm} {rs_codec}",
        )

    def _check_payload(self, inp, out) -> int:
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise TypeError("FlyQuickAllReduce requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("FlyQuickAllReduce supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("FlyQuickAllReduce requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("FlyQuickAllReduce requires contiguous input/output")
        inp_ptr = int(inp.data_ptr())
        out_ptr = int(out.data_ptr())
        if inp_ptr % 16 != 0 or out_ptr % 16 != 0:
            raise ValueError("FlyQuickAllReduce requires 16-byte-aligned input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes > 0xFFFFFFFF:
            raise ValueError(
                "FlyQuickAllReduce payload must not exceed the 4 GiB buffer window"
            )
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
            raise ValueError("FlyQuickAllReduce requires non-overlapping input/output")
        return live_bytes

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
                f"FlyQuickAllReduce.allreduce got a {live_bytes} B payload, "
                f"below the {self.min_bytes} B floor: at decode sizes this "
                "kernel saves a few microseconds on a collective that is not "
                "the bottleneck, and charges ~36 dB of SQNR for them. Route "
                "small messages to an exact all-reduce, or pass min_bytes=0 "
                "to override."
            )
        cfg, _num_tiles = self._pick_cfg(live_bytes)
        self._launch_eng(self._by_cfg[cfg], inp, out, stream, live_bytes=live_bytes)


class FlyQuickAllReduceRMSNorm:
    """Quantized all-reduce fused with residual-add and RMSNorm.

    Per token row::

        acc = allreduce(input)         (quantized wire, dequantized to fp16)
        acc = float(bf16(acc))         (deliberate: matches the unfused path)
        acc += residual_in
        residual_out = bf16(acc)
        out = bf16(acc * rsqrt(sum(acc^2)/hidden + eps) * weight)

    Defaults to ``algorithm="ring"``, the schedule whose structure the
    row-sized block was chosen for.

    The mesh runs its epilogue once per tile, as a
    tail after the all-gather, on values already in registers -- so it collects
    the two saved HBM passes and nothing else changes. The ring runs it inside
    the ``2(N-1)``-op pipeline, once per chunk, putting barriers and norm
    arithmetic between one receive and the next.

    Unlike ``FlyQuickAllReduce`` the *geometry* depends on ``hidden``, not just
    the epilogue -- the block is sized so one 16 B atom is one token row (see
    ``quick_allreduce_fusions.quick_reduce_row_block``). A new hidden therefore
    needs a new engine, and building one is a collective (IPC handle exchange).
    Engines are built lazily on first use, which is safe because TP ranks enter
    a collective with the same shape in lockstep; pass ``hiddens=(...)`` to
    build them up front instead.

    Supported widths are multiples of 1024 up to 8192 -- 7168 and 5120
    included.
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
        block: int | None = None,
        atoms_per_row: int | None = None,
        hiddens: tuple[int, ...] = (),
        pad: bool = True,
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
                f"FlyQuickAllReduceRMSNorm supports {', '.join(_SUPPORTED_ARCHS)}, "
                f"got {arch}"
            )
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(inbox_memory, world_size)
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
        # Threads per block, or None for the widest this width admits. It is not
        # a free knob: one workgroup covers one token row.
        if block is not None and atoms_per_row is not None:
            raise ValueError("pin block or atoms_per_row, not both")
        self.block = None if block is None else int(block)
        self.atoms_per_row = None if atoms_per_row is None else int(atoms_per_row)
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
        self.pad = bool(pad)

        # Rungs to build, as ``(super_tile, grid_cap)``. Same ladder the plain
        # class walks; pinning ``super_tile`` collapses it to one rung.
        world_ladder = algo.ladder_for(self.world_size)
        if world_ladder and not pinned_st:
            self._ladder = world_ladder
            self._rungs = [
                (st, min(rung_cap, cap)) for _, st, rung_cap, _b in world_ladder
            ]
        else:
            st = algo.default_super_tile if super_tile is None else int(super_tile)
            self._ladder = ()
            self._rungs = [(st, cap)]
        # ST=1 is the fallback when a payload has fewer tiles than the chosen
        # super-tile, exactly as in FlyQuickAllReduce.
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
        """Whether a fused build exists for hidden dim, padded ones included."""
        return bool(self._native_opts(hidden) or self._padded_opts(hidden))

    def _block_ok(self, block: int) -> bool:
        """Whether this schedule can actually build at *block*."""
        ok = self._algo.fused_block_ok
        return ok is None or ok(int(block), self.world_size, self.rs_codec)

    def _native_opts(self, hidden: int) -> tuple[tuple[int, int], ...]:
        """``(block, atoms_per_row)`` this width admits with no padding."""
        opts = quick_reduce_row_block_options(int(hidden), self.world_size)
        opts = tuple(o for o in opts if self._block_ok(o[0]))
        if self.block is not None:
            opts = tuple(o for o in opts if o[0] == self.block)
        return opts

    def _padded_opts(self, hidden: int) -> tuple[tuple[int, int, int], ...]:
        """``(block, atoms_per_row, h_pad)`` this width admits with padding.

        Ascending ``h_pad``, so ``[0]`` is the least wire volume that every
        constraint accepts.
        """
        if not self.pad:
            return ()
        opts = quick_reduce_padded_row_block_options(int(hidden), self.world_size)
        opts = tuple(o for o in opts if self._block_ok(o[0]))
        if self.block is not None:
            opts = tuple(o for o in opts if o[0] == self.block)
        return opts

    def _geom_for(self, hidden: int) -> tuple[int, int, int]:
        """``(block, atoms_per_row, h_pad)`` for hidden dim.

        A width with a native geometry resolves exactly. A width without one
        falls to the padded set, least wire volume first, with a pinned ``atoms_per_row``
        breaking ties among equally-padded candidates.
        """
        hidden = int(hidden)
        if self._native_opts(hidden):
            block, apr = quick_reduce_row_block_for(
                hidden,
                self.world_size,
                block=self.block,
                atoms_per_row=self.atoms_per_row,
            )
            return block, apr, hidden
        opts = self._padded_opts(hidden)
        if not opts:
            # No geometry at all: re-resolve so the build raises with the
            # message that names the constraint that failed.
            block, apr = quick_reduce_row_block_for(
                hidden,
                self.world_size,
                block=self.block,
                atoms_per_row=self.atoms_per_row,
            )
            return block, apr, hidden
        tied = [o for o in opts if o[2] == opts[0][2]]
        if self.atoms_per_row is not None:
            want = int(self.atoms_per_row)
            return min(tied, key=lambda o: (abs(o[1] - want), o[1]))
        return tied[0]

    def pads_hidden(self, hidden: int) -> int:
        """``h_pad`` this width runs at, or hidden dim when it needs no padding."""
        return self._geom_for(int(hidden))[2]

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
            quick_reduce_row_block_at(hidden, self.world_size, self.block)
        block, _atoms_per_row, h_pad = self._geom_for(hidden)
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
                block=block,
                h_pad=h_pad,
            )
            if spec["lds_bytes"] > self._lds_capacity:
                raise ValueError(
                    f"fused {self._algo.name} at hidden={hidden} needs "
                    f"{spec['lds_bytes']} B of LDS, over {self.arch}'s "
                    f"{self._lds_capacity} B per workgroup: use a narrower "
                    "hidden, a narrower codec, or the ring schedule"
                )
            with torch.cuda.device(self._device_index):
                self._by_cfg[key] = (
                    _StEngine(
                        spec=spec,
                        group=self.group,
                        rank=self.rank,
                        world_size=self.world_size,
                        inbox_flags=self._inbox_flags,
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
        """HBM bytes one tile spans at hidden dim, building its engines if needed.

        A fused tile is ``rows_per_tile`` token rows of this width.

        ``hbm_tile_bytes``, not ``tile_bytes``: on a padded build the wire tile
        is rows of ``h_pad`` while the footprint is rows of ``hidden``, and it
        is the footprint a tile *count* has to divide. They are equal whenever
        the build is not padded.
        """
        key = (int(hidden), 1)  # ST=1 is always built; see the constructor
        if key not in self._by_cfg:
            self._build_hidden(int(hidden))
        return self._by_cfg[key][1]["hbm_tile_bytes"]

    def _ladder_st(self, live_bytes: int) -> int:
        st = min(self._by_cap)
        for floor, rung_st, _cap, _b in self._ladder:
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
                    f"FlyQuickAllReduceRMSNorm requires a Tensor for {name}"
                )
            if t.dtype != torch.bfloat16:
                raise ValueError(
                    f"FlyQuickAllReduceRMSNorm is bf16-only, {name} is {t.dtype}"
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
            quick_reduce_row_block_at(hidden, self.world_size, self.block)
        live_bytes = int(inp.numel()) * 2
        if live_bytes > 0xFFFFFFFF:
            raise ValueError("payload must not exceed the 4 GiB buffer window")
        if live_bytes > _PAD_MASK_MAX_BYTES and self.pads_hidden(hidden) != hidden:
            raise ValueError(
                f"a padded fused build (hidden={hidden} runs at "
                f"h_pad={self.pads_hidden(hidden)}) masks its pad lanes with an "
                f"offset of {_PAD_MASK_MAX_BYTES} B, so the payload must not "
                f"exceed that; got {live_bytes} B"
            )

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

    @staticmethod
    def _compile_hints(spec):
        """The waves-per-EU floor every fused binary is compiled under.

        The hint is what makes "one workgroup fits a CU" the compiler's problem
        rather than an assumption: a fused workgroup is
        ``ceil(block/WAVE/4)`` waves deep on its busiest SIMD, and asking for
        that many waves per EU caps the register allocation to match.

        FlyDSL keys its JIT cache on the hints, so ``preload`` must compile
        under exactly this context too, or the first launch compiles again.
        """
        waves_per_eu = -(-(spec["block"] // WAVE) // 4)
        return CompilationContext.compile_hints({"waves_per_eu": waves_per_eu})

    def _launch_eng(self, eng, spec, args) -> None:
        """Run *args*, compiling under ``_compile_hints`` on the first call.

        The hint only has an effect on the compile, so the steady-state path
        skips the context entirely.
        """
        self._has_launched = True
        if getattr(eng.launch, "_cf", None) is not None:
            _run_compiled(eng.launch, *args)
            return
        with self._compile_hints(spec):
            _run_compiled(eng.launch, *args)

    def _launch_args(self, eng, spec, ptrs, eps, stream, *, live_bytes, num_tiles):
        """Kernel arguments. *ptrs* is ``(inp, out, residual_in, residual_out,
        weight)`` as raw addresses."""
        inp_ptr, out_ptr, res_in_ptr, res_out_ptr, w_ptr = ptrs
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
            Int32(self._grid_x(num_tiles, spec["super_tile"], eng.grid)),
            Int64(res_in_ptr),
            Int64(res_out_ptr),
            Int64(w_ptr),
            Float32(float(eps)),
            stream,
        )

    def _launch(
        self, inp, residual_in, weight, eps, out, residual_out, stream, *, hidden
    ):
        live_bytes = int(inp.numel()) * 2
        num_tiles = max(1, -(-live_bytes // self._tile_bytes(hidden)))
        eng, spec = self._by_cfg[self._pick_cfg(hidden, live_bytes, num_tiles)]
        ptrs = tuple(
            int(t.data_ptr()) for t in (inp, out, residual_in, residual_out, weight)
        )
        args = self._launch_args(
            eng, spec, ptrs, eps, stream, live_bytes=live_bytes, num_tiles=num_tiles
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
                f"FlyQuickAllReduceRMSNorm got a {live_bytes} B payload, below "
                f"the {self.min_bytes} B floor: at decode sizes this kernel "
                "charges real SQNR for microseconds that are not the "
                "bottleneck. Route small messages to OneShotAllReduceRMSNorm."
            )
        self._launch(
            inp, residual_in, weight, eps, out, residual_out, stream, hidden=hidden
        )
        return out, residual_out

    def cfgs_for(self, hidden: int, lo: int, hi: int) -> list[tuple]:
        """``_by_cfg`` keys at hidden dim a payload of ``lo..hi`` bytes
        (inclusive) can select, in build order. Builds the width if needed."""
        hidden = int(hidden)
        tile_bytes = self._tile_bytes(hidden)
        floors = [rung[0] for rung in self._ladder]
        picked = {
            self._pick_cfg(hidden, n, max(1, -(-n // tile_bytes)))
            for n in payload_probes(floors, lo, hi)
        }
        return [key for key in self._by_cfg if key in picked]

    def preload(self, hidden: int, *, payload_range=None) -> None:
        """Build hidden dim and JIT-compile its engine binaries without
        launching any of them.

        ``payload_range=(lo, hi)`` takes only the binaries ``allreduce_rmsnorm``
        would run for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes
        every one. Not gated by ``min_bytes``. The build still covers every
        super-tile: a width's engines are built together.

        A collective only when hidden dim is not built yet, since building
        exchanges IPC handles and MIN-reduces the grid, so every rank must call
        it with the same widths in the same order. The compile itself is local.
        ``FlyDSLAllReduceRMSNorm`` calls it via ``preload_fly_engines`` to keep
        the JIT off the first real call, which may be inside a HIP graph
        capture. The HIP module load still happens on each binary's first launch.
        """
        hidden = int(hidden)
        self._build_hidden(hidden)
        if payload_range is None:
            keys = [key for key in self._by_cfg if key[0] == hidden]
        else:
            keys = self.cfgs_for(hidden, *payload_range)
        for key in keys:
            eng, spec = self._by_cfg[key]
            args = self._launch_args(
                eng, spec, (0,) * 5, 0.0, None, live_bytes=0, num_tiles=0
            )
            with self._compile_hints(spec):
                _preload_compiled(eng.launch, *args)

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
