# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4/INT5/INT6 all-reduce.

Public type ``FlyQuickAllReduce``, with two interchangeable schedules selected
by ``algorithm``. Both are two-shot -- reduce-scatter then all-gather -- so
they are named for the topology of each lap instead:

* ``"mesh"`` the default: fanout to all N-1 peers, twice.
* ``"ring"`` 2(N-1) single-destination hops.

Super-tile ST∈{1,8} on the mesh, ST∈{1,8,16,32} on the ring. INT4 nibble,
INT5 nibble+1-bit plane, or INT6 bit-plane pair, all with group-16 E4M3
scales. INT5 is mesh-only. Payload HBM is bf16.

The kernel configuration comes from a tuning ladder of ``TwoShotRung``, keyed
on ``(arch, link, world_size)`` in ``allreduce_policy``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

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
    MIN_BATCH_BLOCKS,
    QUANT_MIN_PAYLOAD_BYTES,
    RS_INT6_MIN_WORLD,
    mesh_st_ladder,
    resolve_link,
    ring_st_ladder,
)
from .kernels.comm.allreduce_shared import (
    ATOMS,
    BLOCK,
    PAYLOAD_ALIGN_BYTES,
    SUPPORTED_BLOCKS,
    WORLD,
    TwoShotRung,
    has_release_fence,
    select_rung,
)
from .kernels.comm.quick_allreduce_mesh import (
    MESH_CODECS,
    SUPER_TILES,
    clamp_grid_cap,
    latency_max_tiles,
    make_quick_allreduce_mesh_kernel,
    mxfp4_scale_shape,
    rmsnorm_hidden_supported,
)
from .kernels.comm.quick_allreduce_ring import (
    AG_CODECS,
    RING_SUPER_TILES,
    RS_CODECS,
    make_quick_allreduce_ring_kernel,
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
    # Trailing ``Int64`` kernel arguments before the grid size: the fused
    # RMSNorm's residual, residual_out, weight, MXFP4 out and scale pointers.
    # Plain launches pass zeros.
    norm_ptr_args: int = 0
    # Mesh carries one format on both laps. The ring may differ.
    single_codec: bool = False


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
    **norm,
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
        **norm,
    )


NORM_PTR_ARGS = 5

ALGORITHMS = {
    "mesh": _Algorithm(
        name="mesh",
        build=_build_mesh,
        super_tiles=SUPER_TILES,
        rs_codecs=MESH_CODECS,
        ag_codecs=MESH_CODECS,
        st_ladder=mesh_st_ladder,
        norm_ptr_args=NORM_PTR_ARGS,
        single_codec=True,
    ),
    "ring": _Algorithm(
        name="ring",
        build=make_quick_allreduce_ring_kernel,
        super_tiles=RING_SUPER_TILES,
        rs_codecs=RS_CODECS,
        ag_codecs=AG_CODECS,
        st_ladder=ring_st_ladder,
    ),
}
DEFAULT_ALGORITHM = "mesh"


def _resolve_codecs(algo, world_size, rs_codec, ag_codec):
    """Codecs for one engine: explicit argument > per-N default.

    ``None`` means "not specified", and the caller is expected to pass it
    whenever it wants the schedule's own default rather than a pinned wire
    format. That distinction matters at TP8, where the ring's reduce-scatter
    lap defaults to INT6: an explicit ``rs_codec="int4"`` there is a real
    downgrade, not a restatement of the default.

    A codec the selected schedule cannot build raises. The per-N *default*
    silently narrows to what the schedule supports.

    The mesh is single-codec: ``rs_codec`` and ``ag_codec`` follow each other,
    so give either one and the other takes the same value, and giving two
    different values raises. It stays on INT4 unless a caller names another
    format, so TP8's INT6 reduce-scatter default does not leak into it.
    """
    if algo.single_codec:
        if rs_codec is not None and ag_codec is not None and rs_codec != ag_codec:
            raise ValueError(
                f"{algo.name} algorithm carries one wire format, got "
                f"rs_codec={rs_codec!r} != ag_codec={ag_codec!r}"
            )
        chosen = rs_codec if rs_codec is not None else ag_codec
        if chosen is None:
            chosen = "int4"
        if chosen not in algo.rs_codecs:
            raise ValueError(
                f"rs_codec must be one of {algo.rs_codecs} for "
                f"algorithm={algo.name!r}, got {chosen!r}"
            )
        return chosen, chosen

    rs_default = "int6" if world_size >= RS_INT6_MIN_WORLD else "int4"

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


class FlyQuickAllReduce:
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_mesh``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``algorithm`` selects the schedule. Both are two-shot -- reduce-scatter
    then all-gather -- so they are named for the topology of each lap:

    * ``"mesh"`` (default): each rank pushes to every one of the ``N-1``
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
    defaulting to ``QUANT_MIN_PAYLOAD_BYTES``.

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
        world_size: int = WORLD,
        inbox_memory: str = "auto",
        min_bytes: int | None = None,
        algorithm: str = DEFAULT_ALGORITHM,
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
        self._latency_max_tiles = latency_max_tiles(cu_count)
        # v_cvt_scalef32_pk_fp4_bf16 is gfx950-only.
        self.supports_mxfp4 = arch == "gfx950"
        lds_capacity = get_lds_capacity_bytes(arch)

        self._batch_publishes = batches_publishes(resolved_inbox, algorithm, link)

        self.min_bytes = (
            QUANT_MIN_PAYLOAD_BYTES if min_bytes is None else int(min_bytes)
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
        # The fused RMSNorm cannot run at a block whose atom is wider than a
        # row (hidden=2048 needs BLOCK), and a ladder may use wider blocks only.
        # An ST=1 engine at BLOCK is a ~2 MiB inbox, so the mesh always has one.
        if algo.name == "mesh" and all(b != BLOCK for _st, b in caps):
            caps[(1, BLOCK)] = min(caps.values())
        self._ladder = ladder
        self._by_cfg = {}
        # Per engine, the arguments that rebuild its kernel, for the fused
        # RMSNorm builds that ``norm_launch`` adds on first use.
        self._build_kw = {}
        self._norm_launch = {}
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
                    build_kw = {
                        "world_size": self.world_size,
                        "rank": self.rank,
                        "super_tile": st,
                        "grid": int(shared_grid.item()),
                        "inbox_memory": resolved_inbox,
                        "rs_codec": rs_codec,
                        "ag_codec": ag_codec,
                        "block": b,
                    }
                    spec = algo.build(**build_kw)
                    self._build_kw[key] = build_kw
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
        self,
        eng: _StEngine,
        inp_ptr,
        out_ptr,
        stream,
        *,
        live_bytes,
        num_tiles,
        norm_ptrs=None,
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
            *(Int64(p) for p in self._norm_ptrs(norm_ptrs)),
            Int32(self._grid_x(num_tiles, eng.super_tile, eng.grid)),
            stream,
        )

    def _norm_ptrs(self, norm_ptrs) -> tuple[int, ...]:
        """*norm_ptrs* padded with zeros to what this algorithm's kernel takes."""
        n = self._algo.norm_ptr_args
        norm_ptrs = tuple(norm_ptrs or ())
        return norm_ptrs + (0,) * (n - len(norm_ptrs))

    def _launch_eng(
        self,
        eng: _StEngine,
        inp,
        out,
        stream,
        *,
        live_bytes: int,
        launch=None,
        norm_ptrs=None,
    ) -> None:
        num_tiles = max(1, (live_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        args = self._launch_args(
            eng,
            int(inp.data_ptr()),
            int(out.data_ptr()),
            stream,
            live_bytes=live_bytes,
            num_tiles=num_tiles,
            norm_ptrs=norm_ptrs,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch if launch is None else launch, *args)

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

    def norm_launch(self, cfg, hidden, eps, gemma_norm, latency=False, mxfp4=False):
        """Fused add+RMSNorm variant of engine *cfg*; shares its inbox and colors."""
        key = (cfg, int(hidden), float(eps), bool(gemma_norm), bool(latency), mxfp4)
        launch = self._norm_launch.get(key)
        if launch is None:
            launch = self._algo.build(
                **self._build_kw[cfg],
                norm_hidden=key[1],
                norm_eps=key[2],
                gemma_norm=key[3],
                norm_latency=key[4],
                norm_mxfp4=key[5],
            )["launch"]
            self._norm_launch[key] = launch
        return launch

    @staticmethod
    def supports_rmsnorm(hidden: int, nbytes: int) -> bool:
        """Whether ``allreduce_rmsnorm`` can take this row width and payload."""
        # Atom byte offsets travel in the signed 32-bit soffset.
        return (
            any(rmsnorm_hidden_supported(hidden, b) for b in SUPPORTED_BLOCKS)
            and nbytes < 1 << 31
        )

    @staticmethod
    def mxfp4_scale_shape(rows: int, hidden: int) -> tuple[int, int]:
        """Shape of the e8m0 ``mxfp4_scale`` buffer for ``rows x hidden``."""
        return mxfp4_scale_shape(rows, hidden)

    def _norm_cfg(self, live_bytes: int, hidden: int) -> tuple[tuple[int, int], int]:
        """Engine key and tile count for a fused-norm payload.

        The ladder's pick, unless its block cannot hold a *hidden*-wide row, in
        which case the ST=1 engine of the narrowest block that can.
        """
        if self.algorithm != "mesh" or (self.rs_codec, self.ag_codec) != (
            "int4",
            "int4",
        ):
            raise ValueError(
                "fused RMSNorm needs algorithm='mesh' with INT4 codecs, got "
                f"{self.algorithm!r} with rs_codec={self.rs_codec!r}, "
                f"ag_codec={self.ag_codec!r}"
            )
        if not self.supports_rmsnorm(hidden, live_bytes):
            raise ValueError(
                "fused RMSNorm needs a row that fits whole atoms in a tile "
                f"and a payload under 2 GiB, got hidden={hidden}, "
                f"{live_bytes} bytes"
            )
        cfg, num_tiles = self._pick_cfg(live_bytes)
        if rmsnorm_hidden_supported(hidden, cfg[1]):
            return cfg, num_tiles
        blocks = sorted(
            b for (_st, b) in self._by_cfg if rmsnorm_hidden_supported(hidden, b)
        )
        if not blocks:
            raise ValueError(
                f"no engine of this object has a block that supports hidden={hidden}"
            )
        return (1, blocks[0]), self._num_tiles(live_bytes, blocks[0])

    def compile_rmsnorm(
        self,
        hidden: int,
        eps: float,
        *,
        gemma_norm: bool = False,
        mxfp4: bool = False,
        stream=None,
    ) -> None:
        """Eager-JIT every fused build ``allreduce_rmsnorm`` can pick for ``hidden``.

        Without this the first ``allreduce_rmsnorm`` call of each build pays its
        FlyDSL compile: the low-latency and the occupancy-pinned build of an
        ST=1 engine, and the pinned build of an ST>1 engine, are separate
        binaries. Each is launched once on a zero one-row payload, so every rank
        must call this with the same arguments.

        Args:
            hidden: row width; see ``supports_rmsnorm``.
            eps: RMSNorm epsilon of the calls to warm.
            gemma_norm: warm the ``1 + weight`` builds.
            mxfp4: warm the ``allreduce_rmsnorm_mxfp4`` builds (gfx950).
            stream: ``None`` uses the current PyTorch stream on this device.
        """
        live_bytes = int(hidden) * 2
        self._norm_cfg(live_bytes, hidden)  # validates
        if mxfp4 and not self.supports_mxfp4:
            raise RuntimeError("fused MXFP4 quant needs gfx950")
        dev = torch.device("cuda", self._device_index)
        inp, out, residual, residual_out, weight = (
            torch.zeros(hidden, dtype=torch.bfloat16, device=dev) for _ in range(5)
        )
        mxfp4_bufs = ()
        if mxfp4:
            mxfp4_bufs = (
                torch.empty(hidden // 2, dtype=torch.uint8, device=dev),
                torch.empty(
                    mxfp4_scale_shape(1, hidden), dtype=torch.uint8, device=dev
                ),
            )
        norm_ptrs = (
            int(residual.data_ptr()),
            int(residual_out.data_ptr()),
            int(weight.data_ptr()),
            *(int(t.data_ptr()) for t in mxfp4_bufs),
        )
        for cfg, eng in self._by_cfg.items():
            if not rmsnorm_hidden_supported(hidden, cfg[1]):
                continue
            st, b = cfg
            for latency in (True, False) if (st, b) == (1, BLOCK) else (False,):
                self._launch_eng(
                    eng,
                    inp,
                    out,
                    stream,
                    live_bytes=live_bytes,
                    launch=self.norm_launch(
                        cfg, hidden, eps, gemma_norm, latency, mxfp4
                    ),
                    norm_ptrs=norm_ptrs,
                )
        # The dummy tensors go out of scope here; the launches may be on a
        # stream other than the one that allocated them.
        torch.cuda.synchronize(self._device_index)

    def allreduce_rmsnorm(
        self,
        inp,
        residual,
        weight,
        eps: float,
        out,
        residual_out,
        *,
        gemma_norm: bool = False,
        stream=None,
    ):
        """Fused INT4 all-reduce + residual add + RMSNorm (mesh algorithm).

        ``residual_out = allreduce(inp) + residual`` and
        ``out = rmsnorm(residual_out, eps) * w`` with ``w = weight`` or
        ``1 + weight`` when ``gemma_norm``.

        Args:
            inp: this rank's bf16 partial sum, ``rows x hidden`` elements.
            residual: bf16 residual, same number of elements as ``inp``.
            weight: bf16 RMSNorm weight of ``hidden`` elements; hidden must
                be 2048, 4096, 8192 or 16384.
            eps: RMSNorm epsilon.
            out: bf16 output for the normed rows, same size as ``inp``.
            residual_out: bf16 output for ``allreduce(inp) + residual``; may
                be ``residual`` (in place), must not overlap ``inp`` or
                ``out``.
            gemma_norm: scale by ``1 + weight`` instead of ``weight``.
            stream: ``None`` uses the current PyTorch stream on this device.
        """
        self._allreduce_rmsnorm(
            inp, residual, weight, eps, out, residual_out, gemma_norm, stream
        )

    def allreduce_rmsnorm_mxfp4(
        self,
        inp,
        residual,
        weight,
        eps: float,
        out,
        residual_out,
        mxfp4_out,
        mxfp4_scale,
        *,
        gemma_norm: bool = False,
        stream=None,
    ):
        """``allreduce_rmsnorm`` that also MXFP4-quantizes ``out`` (gfx950).

        Args:
            inp, residual, weight, eps, out, residual_out, gemma_norm,
                stream: as in ``allreduce_rmsnorm``.
            mxfp4_out: 1-byte tensor of ``rows x hidden // 2`` bytes that
                receives the packed FP4 values.
            mxfp4_scale: 1-byte tensor of ``mxfp4_scale_shape(rows, hidden)``
                that receives the shuffled e8m0 scales.

        ``mxfp4_out`` and ``mxfp4_scale`` hold what
        ``per_1x32_f4_quant_hip(out, shuffle=True)`` returns, bit for bit.
        """
        self._allreduce_rmsnorm(
            inp,
            residual,
            weight,
            eps,
            out,
            residual_out,
            gemma_norm,
            stream,
            mxfp4=(mxfp4_out, mxfp4_scale),
        )

    def _allreduce_rmsnorm(
        self,
        inp,
        residual,
        weight,
        eps,
        out,
        residual_out,
        gemma_norm,
        stream,
        mxfp4=None,
    ):
        live_bytes = check_payload(
            inp, out, device_index=self._device_index, owner="FlyQuickAllReduce"
        )
        hidden = int(weight.numel())
        cfg, num_tiles = self._norm_cfg(live_bytes, hidden)
        if int(inp.numel()) % hidden != 0:
            raise ValueError("numel must be a multiple of hidden")
        for name, t in (
            ("residual", residual),
            ("residual_out", residual_out),
            ("weight", weight),
        ):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16:
                raise ValueError(f"{name} must be a bf16 tensor")
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(f"{name} must be on cuda:{self._device_index}")
            if not t.is_contiguous() or int(t.data_ptr()) % PAYLOAD_ALIGN_BYTES:
                raise ValueError(
                    f"{name} must be contiguous and "
                    f"{PAYLOAD_ALIGN_BYTES}-byte aligned"
                )
        if residual.numel() != inp.numel() or residual_out.numel() != inp.numel():
            raise ValueError("residual/residual_out must match inp numel")

        def _overlap(a, b):
            pa, pb = int(a.data_ptr()), int(b.data_ptr())
            return max(pa, pb) < min(pa, pb) + live_bytes

        if _overlap(residual_out, inp) or _overlap(residual_out, out):
            raise ValueError("residual_out must not overlap inp or out")
        if _overlap(residual, out) or (
            _overlap(residual, residual_out)
            and residual.data_ptr() != residual_out.data_ptr()
        ):
            raise ValueError(
                "residual must be residual_out or not overlap it, and must "
                "not overlap out"
            )
        mxfp4_ptrs = ()
        if mxfp4 is not None:
            if not self.supports_mxfp4:
                raise RuntimeError("fused MXFP4 quant needs gfx950")
            rows = int(inp.numel()) // hidden
            for name, t, nbytes in (
                ("mxfp4_out", mxfp4[0], live_bytes // 4),
                ("mxfp4_scale", mxfp4[1], math.prod(mxfp4_scale_shape(rows, hidden))),
            ):
                if not isinstance(t, torch.Tensor) or t.element_size() != 1:
                    raise ValueError(f"{name} must be a 1-byte tensor")
                if not t.is_cuda or t.device.index != self._device_index:
                    raise ValueError(f"{name} must be on cuda:{self._device_index}")
                if not t.is_contiguous() or t.numel() < nbytes:
                    raise ValueError(f"{name} must be contiguous with {nbytes} bytes")
            mxfp4_ptrs = (int(mxfp4[0].data_ptr()), int(mxfp4[1].data_ptr()))
        eng = self._by_cfg[cfg]
        # One tile per workgroup at <= 2 workgroups per CU: occupancy is
        # free, so take the low-latency build.
        latency = cfg == (1, BLOCK) and num_tiles <= min(
            eng.grid, self._latency_max_tiles
        )
        self._launch_eng(
            eng,
            inp,
            out,
            stream,
            live_bytes=live_bytes,
            launch=self.norm_launch(
                cfg, hidden, eps, gemma_norm, latency, mxfp4 is not None
            ),
            norm_ptrs=(
                int(residual.data_ptr()),
                int(residual_out.data_ptr()),
                int(weight.data_ptr()),
                *mxfp4_ptrs,
            ),
        )

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
        ``QUANT_MIN_PAYLOAD_BYTES``. ``allreduce`` refuses payloads below the
        threshold rather than silently running them slowly.
        """
        return int(nbytes) >= self.min_bytes

    def allreduce(self, inp, out, stream=None):
        """Two-shot all-reduce into ``out``, in this engine's wire codecs.

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
