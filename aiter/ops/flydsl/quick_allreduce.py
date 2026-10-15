# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4/INT6 all-reduce.

Public types ``FlyQuickAllReduce`` (plain) and ``FlyQuickAllReduceRMSNorm``
(fused with residual-add + RMSNorm), with two interchangeable schedules selected
by ``algorithm``. Both are two-shot -- reduce-scatter then all-gather -- so
they are named for the topology of each lap instead:

* ``"mesh"`` the default: fanout to all N-1 peers, twice.
* ``"ring"`` 2(N-1) single-destination hops.

Super-tile ST∈{1,8} on the mesh, ST∈{1,8,16,32} on the ring. INT4 nibble or
INT6 bit-plane pair, both with group-16 E4M3 scales. Payload HBM is bf16.

The kernel configuration comes from a tuning ladder of ``TwoShotRung``, keyed
on ``(arch, link, world_size)`` in ``allreduce_policy``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .allreduce_host_shared import (
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
from .allreduce_policy import (
    AG_CODECS,
    MESH_CODECS,
    MIN_BATCH_BLOCKS,
    PAD_MASK_MAX_BYTES,
    RING_SUPER_TILES,
    RS_CODECS,
    SUPER_TILES,
    FusedTwoShotRung,
    TwoShotRung,
    default_codecs,
    fused_mesh_st_ladder,
    fused_ring_st_ladder,
    mesh_st_ladder,
    quantized_floor,
    resolve_link,
    ring_st_ladder,
    select_rung,
)
from .kernels.comm.quick_allreduce_fusions import (
    quick_reduce_padded_row_block_options,
    quick_reduce_row_block_at,
    quick_reduce_row_block_for,
    quick_reduce_row_block_options,
)
from .kernels.comm.quick_allreduce_mesh import (
    clamp_grid_cap,
    make_quick_allreduce_mesh_kernel,
    mesh_fanout_fits,
)
from .kernels.comm.quick_allreduce_ring import (
    make_quick_allreduce_ring_kernel,
)
from .kernels.comm.transport import (
    ATOMS,
    WAVE,
    has_release_fence,
)
from .kernels.tensor_shim import _preload_compiled, _run_compiled

logger = logging.getLogger("aiter")


@dataclass(frozen=True)
class _Algorithm:
    """One all-reduce schedule, plus the host-side policy that tunes it."""

    name: str
    build: Callable[..., dict]
    super_tiles: tuple[int, ...]
    rs_codecs: tuple[str, ...]
    ag_codecs: tuple[str, ...]
    # ``(*, world_size, link, arch) -> (TwoShotRung, ...)``: the tuning ladder.
    st_ladder: Callable[..., tuple[TwoShotRung, ...]]
    # The same for the fused (all-reduce + RMSNorm) build.
    fused_st_ladder: Callable[..., tuple[FusedTwoShotRung, ...]]
    # ``(block, world_size, codec) -> bool``: whether a fused row-sized block
    # can build; None when every row geometry can.
    fused_block_ok: Callable[[int, int, str], bool] | None = None


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
        st_ladder=mesh_st_ladder,
        fused_st_ladder=fused_mesh_st_ladder,
        fused_block_ok=mesh_fanout_fits,
    ),
    "ring": _Algorithm(
        name="ring",
        build=make_quick_allreduce_ring_kernel,
        super_tiles=RING_SUPER_TILES,
        rs_codecs=RS_CODECS,
        ag_codecs=AG_CODECS,
        st_ladder=ring_st_ladder,
        fused_st_ladder=fused_ring_st_ladder,
    ),
}


def _resolve_codecs(algo, world_size, rs_codec, ag_codec):
    """Codecs for one engine: explicit argument > ``default_codecs``.

    ``None`` means "not specified", and the caller is expected to pass it
    whenever it wants the schedule's own default rather than a pinned wire
    format. That distinction matters at TP8, where the ring's reduce-scatter
    lap defaults to INT6: an explicit ``rs_codec="int4"`` there is a real
    downgrade, not a restatement of the default.

    A codec the selected schedule cannot build raises.

    The mesh has one wire format for both laps, so there ``rs_codec`` and
    ``ag_codec`` follow each other: give either one and the other takes the
    same value. Giving two different values raises when the kernel is built.
    """
    if algo.name == "mesh" and (rs_codec is None) != (ag_codec is None):
        rs_codec = ag_codec = rs_codec if rs_codec is not None else ag_codec

    rs_default, ag_default = default_codecs(algo.name, world_size)

    def _pick(requested, default, supported, label):
        if requested is None:
            return default
        if requested not in supported:
            raise ValueError(
                f"{label} must be one of {supported} for "
                f"algorithm={algo.name!r}, got {requested!r}"
            )
        return requested

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


class FlyQuickAllReduce:
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_mesh``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``algorithm`` selects the schedule. Both are two-shot -- reduce-scatter
    then all-gather -- so they are named for the topology of each lap:

    * ``"mesh"``: each rank pushes to every one of the ``N-1``
      peers, twice. Two hops. Optimal on a meshed xGMI node.
    * ``"ring"``: ``2(N-1)`` hops, each a single contiguous run into exactly
      one peer's inbox. Same wire volume (``2(N-1)/N`` of the payload), traded
      for per-destination locality. Optimal for large messages on PCIe.

    ``rs_codec`` and ``ag_codec`` are the wire formats of the ring's two laps.
    The reduce-scatter lap requantizes ``N-1`` times and it defaults to ``"int6"``
    at TP8, where INT4 would cost too much accuracy. The all-gather lap forwards
    bytes verbatim and contributes a single quantization, so it defaults to ``"int4"``
    everywhere.

    ``inbox_memory`` selects how the IPC inbox is allocated:

    * ``"auto"`` (default) -- ``uncached`` on hosts with xGMI peer links,
      ``finegrained`` on PCIe-attached hosts, decided from the KFD topology.
      TP2 is ``uncached`` on PCIe too: one remote peer cannot collapse.
    * ``"uncached"`` -- correct everywhere, but peer writes collapse on PCIe.
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so each
      publish writes the payload back from the writer's L2 with a release fence
      before the flag goes out write-through (``sc0 sc1``).

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to where the policy's mesh and ring windows start
    (``quantized_floor``).

    ``link`` selects the tuning ladder and is detected from the KFD topology
    when not given. ``ladder`` replaces the tuning ladder of the policy, for
    tests that run other configurations. Both schedules specialise their
    binary to this rank, so the JIT symbol carries an ``_r<n>_`` field.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        inbox_memory: str = "auto",
        min_bytes: int | None = None,
        algorithm: str,
        rs_codec: str | None = None,
        ag_codec: str | None = None,
        link: str | None = None,
        ladder: tuple[TwoShotRung, ...] | None = None,
    ):
        check_world_size(world_size)
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {tuple(ALGORITHMS)}, got {algorithm!r}"
            )
        algo = ALGORITHMS[algorithm]
        link = resolve_link(link)
        rs_codec, ag_codec = _resolve_codecs(algo, int(world_size), rs_codec, ag_codec)
        _validate_ipc_process_group(
            group, rank=int(rank), world_size=world_size, owner="FlyQuickAllReduce"
        )
        arch = require_arch(owner="FlyQuickAllReduce")
        if ladder is None:
            ladder = algo.st_ladder(world_size=world_size, link=link, arch=arch)
        if not ladder:
            raise ValueError("ladder must have at least one rung")
        inbox_flags, resolved_inbox = _resolve_inbox_flags(
            inbox_memory, world_size, link
        )
        self._device_index = _cuda_index(device, owner="FlyQuickAllReduce")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.link = link
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

        self._batch_publishes = batches_publishes(resolved_inbox, algorithm, link)

        self.min_bytes = (
            quantized_floor(world_size=world_size, link=link, arch=arch)
            if min_bytes is None
            else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        # One engine per distinct ``(super_tile, block)``, keyed that way in
        # ``_by_cfg``. Each ``block`` also gets an ST=1 engine: _pick_cfg falls
        # back to it when a payload has fewer tiles than the chosen super-tile,
        # and the fallback has to share the rung's block -- a different tile
        # size would change the tile count it was picked for. Engines are built
        # in a fixed order because each does its own IPC handle exchange, which
        # is a collective -- ranks disagreeing on the order would deadlock.
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
                    )
                    if spec["lds_bytes"] > lds_capacity:
                        raise ValueError(
                            f"{algorithm} (rs_codec={rs_codec}, ag_codec={ag_codec}) "
                            f"at block={b} needs "
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

    def _ladder_cfg(self, live_bytes: int) -> tuple[int, int]:
        """``(super_tile, block)`` the ladder assigns to *live_bytes*.

        Publishes per rank are ``num_tiles / ST * 2(N-1)`` and cost a full L2
        writeback each, so a bigger payload wants a bigger ST -- but ST also
        divides the block count, so it cannot simply be maximised. The rungs
        and the measurements behind them are in ``MESH_ST_LADDER`` and
        ``RING_ST_LADDER``.
        """
        rung = select_rung(self._ladder, live_bytes)
        return (rung.super_tile, rung.block)

    @staticmethod
    def _num_tiles(live_bytes: int, block: int) -> int:
        tile_bytes = int(block) * ATOMS * 16
        return max(1, (live_bytes + tile_bytes - 1) // tile_bytes)

    def _pick_cfg(self, live_bytes: int) -> tuple[tuple[int, int], int]:
        """Engine key for a *live_bytes* payload, and its tile count.

        The ladder chooses the rung, which fixes the block and so the tile
        count; the tile count then only has to confirm there is a whole
        super-tile to take, falling back to the same block's ST=1 engine when
        there is not.
        """

        want, b = self._ladder_cfg(live_bytes)
        num_tiles = self._num_tiles(live_bytes, b)
        if want == 1:
            st = 1
        elif self._batch_publishes:
            st = want if num_tiles >= want else 1
        else:
            st = want if num_tiles > self._by_cfg[(want, b)].grid else 1
        return (st, b), num_tiles

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int) -> int:
        """Blocks to launch for *num_tiles* tiles under *super_tile*.

        *grid* is the compile-time cap of the engine that will run, which is
        per-super-tile once a ladder is in play -- the wire buffer scales with
        ``ST * grid``, so a high rung pairs a large ST with a small cap.

        Batching publishes only pays if a block actually owns a super-tile's
        worth of work: ST=8 across 448 blocks holding one tile each still
        publishes per tile. Hand each block a full super-tile instead, which
        cuts publishes to ``num_tiles / ST`` per phase.
        """
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), MIN_BATCH_BLOCKS)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, grid))

    def _launch_args(
        self, eng: _StEngine, inp_ptr, out_ptr, stream, *, live_bytes, num_tiles
    ):
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
            Int32(self._grid_x(num_tiles, eng.super_tile, eng.grid)),
            stream,
        )

    def _launch_eng(self, eng: _StEngine, inp, out, stream, *, live_bytes: int) -> None:
        num_tiles = max(1, (live_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        args = self._launch_args(
            eng,
            int(inp.data_ptr()),
            int(out.data_ptr()),
            stream,
            live_bytes=live_bytes,
            num_tiles=num_tiles,
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
        """JIT-compile engine binaries without launching any of them.

        ``payload_range=(lo, hi)`` takes only the binaries ``allreduce`` would
        run for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes every
        one. The ladder builds engines for the whole size range, while a
        dispatcher routes only its own window here, so the rest never run.

        Local to this rank, not a collective. ``QuickAllReduce`` calls it once
        at init, via ``preload_fly_engines``, to keep JIT compiles out of CUDA
        graph capture; the op tests call it before timing or correctness
        checks begin. The HIP module load still happens on each binary's first
        launch.
        """
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        for key in keys:
            eng = self._by_cfg[key]
            args = self._launch_args(eng, 0, 0, None, live_bytes=0, num_tiles=0)
            _preload_compiled(eng.launch, *args)

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
        ``quantized_floor``. ``allreduce`` refuses payloads below the
        threshold rather than silently running them slowly.
        """
        return int(nbytes) >= self.min_bytes

    def allreduce(self, inp, out, stream=None):
        """Two-shot INT4 all-reduce into ``out``.

        ``stream=None`` uses the current PyTorch stream on this device.
        """
        live_bytes = check_payload(
            inp, out, device_index=self._device_index, owner="FlyQuickAllReduce"
        )
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

    ``link`` selects the super-tile ladder (``FUSED_MESH_ST_LADDER``,
    ``FUSED_RING_ST_LADDER``), as in ``FlyQuickAllReduce``, and is detected
    from the KFD topology when not given. ``min_bytes`` is the payload below
    which ``allreduce_rmsnorm`` refuses to run, defaulting to where the fused
    policy's mesh and ring windows start (``quantized_floor``).

    ``super_tile`` and ``grid_cap`` each override that field on every rung of
    the ladder; the knobs left unset keep the ladder's values.
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
        world_size: int,
        algorithm: str,
        super_tile: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        batch_publishes: bool | None = None,
        min_bytes: int | None = None,
        max_bytes: int | None = None,
        rs_codec: str | None = None,
        ag_codec: str | None = None,
        block: int | None = None,
        atoms_per_row: int | None = None,
        hiddens: tuple[int, ...] = (),
        link: str | None = None,
    ):
        check_world_size(world_size)
        link = resolve_link(link)
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {tuple(ALGORITHMS)}, got {algorithm!r}"
            )
        algo = ALGORITHMS[algorithm]
        if super_tile is not None and super_tile not in algo.super_tiles:
            raise ValueError(
                f"super_tile must be one of {algo.super_tiles} for "
                f"algorithm={algorithm!r}, got {super_tile!r}"
            )
        rs_codec, ag_codec = _resolve_codecs(algo, int(world_size), rs_codec, ag_codec)
        _validate_ipc_process_group(
            group,
            rank=int(rank),
            world_size=world_size,
            owner="FlyQuickAllReduceRMSNorm",
        )
        arch = require_arch(owner="FlyQuickAllReduceRMSNorm")
        if grid_cap is not None and int(grid_cap) < 1:
            raise ValueError(f"grid_cap must be positive, got {grid_cap}")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(
            inbox_memory, world_size, link
        )
        self._device_index = _cuda_index(device, owner="FlyQuickAllReduceRMSNorm")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.algorithm = algorithm
        self.link = link
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
        # The ceiling is None because nothing here pushes the whole payload to
        # every peer.
        self.min_bytes = (
            quantized_floor(world_size=world_size, link=link, arch=arch, fused=True)
            if min_bytes is None
            else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")
        self.max_bytes = max_bytes

        # Rungs to build, as ``(super_tile, grid_cap)``, from the fused ladder.
        # A pinned ``super_tile`` or ``grid_cap`` overrides that field on every
        # rung.
        pins = {
            name: int(value)
            for name, value in (("super_tile", super_tile), ("grid_cap", grid_cap))
            if value is not None
        }
        self._ladder = tuple(
            dataclasses.replace(r, **pins)
            for r in algo.fused_st_ladder(
                world_size=self.world_size, link=link, arch=arch
            )
        )
        self._rungs = [(r.super_tile, r.grid_cap) for r in self._ladder]
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

    def _pick_cfg(self, hidden: int, live_bytes: int, num_tiles: int) -> tuple:
        want = select_rung(self._ladder, live_bytes).super_tile
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
            batched = max(-(-num_tiles // super_tile), MIN_BATCH_BLOCKS)
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
        if live_bytes > PAD_MASK_MAX_BYTES and self.pads_hidden(hidden) != hidden:
            raise ValueError(
                f"a padded fused build (hidden={hidden} runs at "
                f"h_pad={self.pads_hidden(hidden)}) masks its pad lanes with an "
                f"offset of {PAD_MASK_MAX_BYTES} B, so the payload must not "
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
        with torch.cuda.device(self._device_index):
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
        floors = [rung.min_bytes for rung in self._ladder]
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
