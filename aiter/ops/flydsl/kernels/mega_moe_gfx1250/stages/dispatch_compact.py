# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FlyDSL compact plan and its TDM dispatch as a dispatch stage."""

import os
from dataclasses import dataclass

import flydsl.expr as fx
import torch

from aiter import dtypes

from ...tensor_shim import _preload_compiled, _run_compiled
from ..arena import _align_up
from ..config import _DISPATCH_COMPACT, _WAVE_SIZE
from ..dispatch_tdm import _make_dispatch_tdm, tdm_max_warps, tdm_stage_capacity
from ..recipe import MegaMoEConfig
from ..types import _from_gpu_ptr
from .base import (
    BatchLayout,
    ExpertGrouping,
    _check_recv_token_bound,
    _Dispatch,
    _DispatchBatch,
    _EpResources,
    _pick,
)


def _compact_gemm_align_m(
    config: "MegaMoEConfig", recv_bound: int | None = None
) -> int:
    """Tile alignment for compact dest rows; must match GEMM ``max(tile_m, tile_m2)``.

    Arena size is fixed at MegaMoE init. Honour ``AITER_TDM_TILE_M{,2}``, otherwise
    use the same grouped-GEMM CSV row fused_moe will pick for this recv-token
    bucket. No floor at 64: the CSV tile can be 16 or 32, and padding every
    expert to 64 rows would run a 3-4x GEMM grid at small batches.

    ``recv_bound`` is this step's recv-token bound; it defaults to the arena's
    full capacity, which is the bucket to size the arena against but not the one
    a decode step runs at. Alignment is per-expert padding, so holding it at the
    capacity tile costs ``experts_per_rank * (tile - used)`` dead rows that the
    GEMM still computes.
    """

    def _env(name: str) -> int | None:
        v = os.environ.get(name)
        if v is None or v == "":
            return None
        return int(v)

    ov_m = _env("AITER_TDM_TILE_M")
    ov_m2 = _env("AITER_TDM_TILE_M2")
    if ov_m is not None or ov_m2 is not None:
        tile_m = 64 if ov_m is None else ov_m
        tile_m2 = tile_m if ov_m2 is None else ov_m2
        return max(int(tile_m), int(tile_m2))

    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        _as_int,
        _find_grouped_config,
        _get_padded_m,
    )

    if config.dispatch_wire == "fp4":
        q_dtype_a = dtypes.fp4x2
    elif config.dispatch_wire == "fp8":
        q_dtype_a = dtypes.fp8
    else:
        return 64

    row = _find_grouped_config(
        token_num=_get_padded_m(
            config.max_recv if recv_bound is None else int(recv_bound)
        ),
        model_dim=config.hidden_dim,
        inter_dim=config.geometry.inter_dim,
        experts=config.experts_per_rank,
        topk=-1,
        activation=config.recipe.compute.activation,
        dtype=dtypes.bf16,
        q_dtype_a=q_dtype_a,
        q_dtype_w=dtypes.fp4x2,
        quant_type=config.recipe.compute.quant_type,
    )
    tile_m = 64
    tile_m2 = 64
    if row is not None:
        tile_m = _as_int(row.get("tile_m"), tile_m) or tile_m
        tile_m2 = _as_int(row.get("tile_m2"), tile_m) or tile_m
    return max(int(tile_m), int(tile_m2))


# The widest compact tile a grouped-GEMM CSV row names; bounds the compact
# rows when the tile cannot be looked up (inter_dim unknown).
_MAX_COMPACT_TILE_M = 256
# The compact plan's limits: its LDS histogram's (rank, expert) segments, and
# the lanes of its one block, which runs a wave per peer.
_MAX_PLAN_SEGMENTS = 1024
_MAX_PLAN_THREADS = 1024


@dataclass(frozen=True)
class _CompactGeometry:
    """The compact dispatch's row geometry, sized for the arena's capacity bucket."""

    tile_m: int  # the capacity bucket's GEMM tile; each step clamps to it
    cap: int  # recv rows the arena holds
    wire_row: int  # payload, then its e8m0 row, 128 B aligned
    hist_stride: int
    max_routes: int

    @classmethod
    def of(cls, config: MegaMoEConfig, tile_m: int | None = None):
        from ..compact_plan import compact_hist_stride, compact_row_capacity

        tile = _compact_gemm_align_m(config) if tile_m is None else tile_m
        max_routes = config.max_tokens_per_rank * config.topk
        return cls(
            tile_m=tile,
            cap=compact_row_capacity(
                max_recv=config.max_recv,
                topk=config.topk,
                experts_per_rank=config.experts_per_rank,
                tile_m=tile,
            ),
            wire_row=_align_up(
                config.dispatch_token_nbytes + config.dispatch_scale_nbytes, 128
            ),
            hist_stride=compact_hist_stride(
                npes=config.world_size,
                experts_per_rank=config.experts_per_rank,
                max_routes=max_routes,
            ),
            max_routes=max_routes,
        )


@dataclass(frozen=True)
class _CompactStep:
    """One step's compact geometry, fixed from the caller's recv bound. Python
    ints, so a captured graph keeps a static grid and each captured batch size
    gets the geometry tuned for its own bucket."""

    recv_bound: int  # the CSV token bucket
    align_m: int  # per-expert row alignment the plan writes
    rows: int  # row upper bound, bounding the GEMM grid


class _CompactDispatch(_Dispatch):
    """The FlyDSL compact plan, then its TDM dispatch: recv rows grouped per
    expert and padded to the GEMM tile, the layout the grouped GEMM reads in
    place.

    The plan needs only topk_ids, so it runs on a side stream ahead of the
    dispatch. Two hist/done slots and local tok_map/psum buffers let the next
    layer's plan overlap this layer's expert GEMM.

    Recv slots are reserved one atomic per (block, peer) and handed out
    block-local. The TDM staging arrays stand in for the HIP kernel's
    ``__device__`` BSS: FINALIZE gathers idx / weights / srcmap into a peer-major
    destTokId SoA so META can ship each block's reserved run as contiguous TDM
    copies.
    """

    wires = ("fp8", "fp4")
    output_layout = BatchLayout.EXPERT_GROUPED
    uses_mori = False
    # The plan reserves the recv slots itself.
    provides_aggregated_reservation = False
    _PLAN_SLOTS = 2

    def __init__(self, config: MegaMoEConfig, caps=None):
        # Decode walks routes, prefill tokens (see _DISPATCH_COMPACT).
        blocks_env = os.environ.get("AITER_TDM_COMPACT_BLOCKS")
        warps_env = os.environ.get("AITER_TDM_COMPACT_WARPS")
        if blocks_env is not None or warps_env is not None:
            schedule = ((None, int(blocks_env or "64"), int(warps_env or "8")),)
        else:
            schedule = _DISPATCH_COMPACT
        super().__init__(config, schedule)
        self.compact = _CompactGeometry.of(config)
        self.scale_stride = _align_up(config.dispatch_scale_nbytes, 128)
        self._step = _CompactStep(
            recv_bound=config.max_recv,
            align_m=self.compact.tile_m,
            rows=self.compact.cap,
        )
        self._slot = 0
        self._plan_pending = False
        self._pending_align = None  # the alignment the pending plan writes
        self._plan_launches = {}

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        from ..compact_plan import compact_plan_waves

        segs = config.world_size * config.experts_per_rank
        if segs > _MAX_PLAN_SEGMENTS:
            raise ValueError(
                f"the compact plan holds at most {_MAX_PLAN_SEGMENTS} (rank, "
                f"expert) segments, got world_size * experts_per_rank = {segs}"
            )
        waves = compact_plan_waves(
            npes=config.world_size,
            max_routes=config.max_tokens_per_rank * config.topk,
        )
        if waves * _WAVE_SIZE > _MAX_PLAN_THREADS:
            raise ValueError(
                "the compact plan runs a wave per peer in one block of at most "
                f"{_MAX_PLAN_THREADS // _WAVE_SIZE} waves, got {waves} for "
                f"world_size={config.world_size}"
            )

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps=None) -> dict:
        return cls.regions_for_tile(config, _compact_gemm_align_m(config))

    @staticmethod
    def regions_for_tile(config: MegaMoEConfig, tile_m: int) -> dict:
        """The arena regions at GEMM tile ``tile_m``, which pads every expert's
        rows and so sizes the recv rows and their map."""
        from ..compact_plan import compact_done_nbytes

        geo = _CompactGeometry.of(config, tile_m)
        max_recv, topk = config.max_recv, config.topk
        return {
            "tok_off": 4,
            # 8 bytes a source, as mori's dispatch signals.
            "recv_num": 2 * config.world_size * 4,
            "recv_to_src_token": max_recv * 4,
            "out_idx": max_recv * topk * 4,
            "out_wts": max_recv * topk * 4,
            "disp_out": geo.cap * geo.wire_row,
            "ep_rowmap": (geo.cap + 1) * 8,
            "compact_hist": 2 * geo.hist_stride * 4,
            "compact_done": compact_done_nbytes(),
        }

    def build(self, res: _EpResources) -> None:
        super().build(res)
        from ..compact_plan import PLAN_BLOCKS

        config, geo, arena = self.config, self.compact, res.arena
        device = res.total_recv.device
        epr = config.experts_per_rank
        segs = config.world_size * epr
        slots = range(self._PLAN_SLOTS)
        # Slot 0 writes the shared map, slot 1 its own.
        self._tok_maps = [res.token_destination_map] + [
            torch.full_like(res.token_destination_map, -1) for _ in slots[1:]
        ]
        self._masked_ms = [
            torch.zeros(epr, dtype=torch.int32, device=device) for _ in slots
        ]
        self._psums = [
            torch.zeros(epr, dtype=torch.int32, device=device) for _ in slots
        ]
        self._block_hists = [
            torch.empty(PLAN_BLOCKS * segs, dtype=torch.int32, device=device)
            for _ in slots
        ]
        self._send_bases = [
            torch.empty(segs, dtype=torch.int32, device=device) for _ in slots
        ]
        self._barriers = [
            torch.zeros(4, dtype=torch.int32, device=device) for _ in slots
        ]
        # The grouped GEMM reads the compact rows' experts off the plan, so it
        # is handed placeholders for the per-row routing.
        self._dummy_ids = torch.zeros(
            (1, config.topk), dtype=torch.int32, device=device
        )
        self._dummy_wts = torch.zeros(
            (1, config.topk), dtype=torch.float32, device=device
        )
        self._plan_stream = torch.cuda.Stream()
        self._plan_event = torch.cuda.Event()
        # Warm both hist/done slots and the capacity bucket: compiling inside
        # graph capture is not allowed.
        for slot in slots:
            self._plan_for(geo.tile_m, slot)
        self._kernels = self._build_kernels()
        # Width in whatever recv_dtype counts: bytes on both quantizing wires.
        width = (
            config.dispatch_token_nbytes
            // config.dispatch_wire_spec.recv_dtype.itemsize
        )
        storage = _from_gpu_ptr(
            arena.local_ptr("disp_out"),
            (geo.cap * geo.wire_row,),
            config.dispatch_wire_spec.recv_dtype,
        )
        self.recv_x = torch.as_strided(storage, (geo.cap, width), (geo.wire_row, 1))
        # Each row's e8m0 scales follow its payload inside the wire row.
        scale_row = config.dispatch_scale_nbytes
        storage = _from_gpu_ptr(
            arena.local_ptr("disp_out") + config.dispatch_token_nbytes,
            ((geo.cap - 1) * geo.wire_row + scale_row,),
            torch.uint8,
        )
        self.recv_scales = torch.as_strided(
            storage, (geo.cap, scale_row), (geo.wire_row, 1)
        )
        self._ep_rowmap = _from_gpu_ptr(
            arena.local_ptr("ep_rowmap"), (geo.cap + 1, 2), torch.int32
        )

    def _build_kernels(self) -> dict:
        """The TDM dispatch kernels, keyed by (block, warp, route_parallel) spec."""
        config, geo, arena = self.config, self.compact, self.res.arena
        device = self.res.total_recv.device
        payload_dim = config.dispatch_wire_elem_count
        elem_size = config.dispatch_wire_spec.recv_dtype.itemsize
        _, slots = tdm_stage_capacity(npes=config.world_size, max_recv=config.max_recv)
        self._stage = (
            torch.empty(slots * config.topk, dtype=torch.int32, device=device),
            torch.empty(slots * config.topk, dtype=torch.int32, device=device),
            torch.empty(slots, dtype=torch.int32, device=device),
            torch.empty(
                slots * self.scale_stride,
                dtype=torch.uint8,
                device=device,
            ),
        )
        for buf in self._stage:
            if buf.data_ptr() % 128:
                raise RuntimeError(
                    "TDM staging allocations must be 128-byte aligned, got "
                    f"0x{buf.data_ptr():x}"
                )
        self._stage_ptrs = tuple(buf.data_ptr() for buf in self._stage)
        # Compact compiles the metadata batch away, so its LDS tile is one wire
        # row.
        slab_bytes = geo.wire_row
        # Clamp warp count to the LDS tile budget but keep the tuned block
        # count, which is what paces the grid barrier, and keep the caller's
        # spec as the key so the runtime pick still resolves.
        max_warps = tdm_max_warps(
            hidden_dim=payload_dim,
            hidden_elem_size=elem_size,
            npes=config.world_size,
            slab_bytes=slab_bytes,
        )
        built = {}
        kernels = {}
        for spec in self.specs:
            route_parallel = len(spec) > 2 and bool(spec[2])
            geom = (spec[0], min(spec[1], max_warps), route_parallel)
            if geom not in built:
                built[geom] = _make_dispatch_tdm(
                    rank=config.rank,
                    npes=config.world_size,
                    experts_per_rank=config.experts_per_rank,
                    experts_per_token=config.topk,
                    hidden_dim=payload_dim,
                    hidden_elem_size=elem_size,
                    slab_bytes=slab_bytes,
                    max_tok_per_rank=config.max_tokens_per_rank,
                    max_recv=geo.cap,
                    compact_row_stride=geo.wire_row,
                    enable_signal=True,
                    off_tok_off=arena.offset("tok_off"),
                    off_recv_num=arena.offset("recv_num"),
                    off_tis=arena.offset("recv_to_src_token"),
                    off_out_idx=arena.offset("out_idx"),
                    off_out_wts=arena.offset("out_wts"),
                    off_out_tok=arena.offset("disp_out"),
                    off_out_scales=(
                        arena.offset("disp_out") + config.dispatch_token_nbytes
                    ),
                    scale_bytes=config.dispatch_scale_nbytes,
                    scale_stride=self.scale_stride,
                    block_num=geom[0],
                    warp_num_per_block=geom[1],
                    clear_route_counter=False,
                    compact_plan=True,
                    off_ep_rowmap=arena.offset("ep_rowmap"),
                    max_tok_slot_stride=config.max_tokens_per_rank * config.topk,
                    route_parallel=route_parallel,
                )
            kernels[spec] = built[geom]
        return kernels

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        recv_token_bound: int | None = None,
        next_topk_ids: torch.Tensor | None = None,
        next_recv_token_bound: int | None = None,
        begin_step: bool = True,
    ) -> _DispatchBatch:
        """Compact cannot slice recv_x -- the rows are grouped per expert, not
        token-major -- so ``recv_token_bound`` is spent on the step's geometry
        instead: it picks the plan's alignment, the GEMM's tile and the GEMM's
        grid. ``next_topk_ids`` starts the next layer's plan once this dispatch
        is issued."""
        config, res = self.config, self.res
        token_count = int(hidden_states.shape[0])
        _check_recv_token_bound(config, "recv_token_bound", recv_token_bound)
        _check_recv_token_bound(config, "next_recv_token_bound", next_recv_token_bound)
        if begin_step:
            self._step = self._step_for(self._recv_bound(recv_token_bound))
        if self._plan_pending and self._pending_align != self._step.align_m:
            raise ValueError(
                "the compact plan already started for this step aligns each "
                f"expert's rows to {self._pending_align}, but "
                f"recv_token_bound={recv_token_bound} needs {self._step.align_m}; "
                "pass the bound the plan was started with"
            )
        kernel = self._kernels[_pick(self.schedule, token_count)]
        if not self._plan_pending:
            self._launch_plan(topk_ids, token_count, self._slot, self._step.align_m)
        stream = fx.Stream(torch.cuda.current_stream())
        payload, scales_ptr = self._quantize(hidden_states, token_count)
        self._wait_plan()
        slot, step = self._slot, self._step
        _run_compiled(
            kernel,
            res.arena.handle,
            payload.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            self._tok_maps[slot].data_ptr(),
            res.destination_peer_counter.data_ptr(),
            res.dispatch_barrier.data_ptr(),
            res.total_recv.data_ptr(),
            *self._stage_ptrs,
            scales_ptr,
            res.route_counter.data_ptr(),
            config.rank,
            token_count,
            stream,
        )
        batch = _DispatchBatch(
            x=self.recv_x,
            topk_weights=self._dummy_wts,
            topk_ids=self._dummy_ids,
            scales=self.recv_scales,
            num_local_tokens=None,
            routing=self._routing(token_count),
            layout=BatchLayout.EXPERT_GROUPED,
            grouping=ExpertGrouping(
                masked_m=self._masked_ms[slot],
                psum=self._psums[slot],
                ep_rowmap=self._ep_rowmap,
                wire_row_stride=self.compact.wire_row,
                recv_bound=step.recv_bound,
                align_m=step.align_m,
                rows=step.rows,
            ),
        )
        if next_topk_ids is not None and int(next_topk_ids.shape[0]) > 0:
            # Dispatch of this layer is done, so tok_map[slot] is free. GEMM
            # still reads psum/masked_m of this slot; the next plan writes the
            # other slot and overlaps the expert GEMM (and later combine/RMS).
            nxt = 1 - slot
            count = int(next_topk_ids.shape[0])
            ahead = self._step_for(self._recv_bound(next_recv_token_bound))
            self._launch_plan(next_topk_ids, count, nxt, ahead.align_m)
            self._slot = nxt
        return batch

    def prefetch(self, topk_ids, recv_token_bound=None) -> None:
        """Launch the plan for ``topk_ids`` on the side stream, into this step."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)
        token_count = int(topk_ids.shape[0])
        self._step = self._step_for(self._recv_bound(recv_token_bound))
        self._launch_plan(topk_ids, token_count, self._slot, self._step.align_m)

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        """Compile the plan this bound's tile needs, in both slots.
        ``token_count`` is unused: the bound alone picks the tile."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)
        self._step = self._step_for(self._recv_bound(recv_token_bound))
        for slot in range(self._PLAN_SLOTS):
            self._plan_for(self._step.align_m, slot)

    def _recv_bound(self, recv_token_bound: int | None) -> int:
        """This step's recv bound: the caller's, already checked, else the
        arena's capacity. What a rank receives depends on its peers' token
        counts, so nothing smaller can be inferred from its own."""
        if recv_token_bound is None:
            return int(self.config.max_recv)
        return int(recv_token_bound)

    def _step_for(self, recv_bound: int) -> _CompactStep:
        from ..compact_plan import compact_row_capacity

        config, geo = self.config, self.compact
        bound = max(1, min(int(recv_bound), config.max_recv))
        align_m = _compact_gemm_align_m(config, recv_bound=bound)
        # The arena holds rows for the capacity tile, and the CSV tiles are not
        # monotonic in the token bucket, so a smaller bound can still name a
        # wider tile. Clamping keeps every step inside the allocated rows.
        align_m = min(int(align_m), geo.tile_m)
        rows = compact_row_capacity(
            max_recv=bound,
            topk=config.topk,
            experts_per_rank=config.experts_per_rank,
            tile_m=align_m,
        )
        return _CompactStep(recv_bound=bound, align_m=align_m, rows=min(rows, geo.cap))

    def _plan_for(self, align_m: int, slot: int):
        """The compact-plan launch that aligns each expert's rows to ``align_m``.

        Compiled per alignment and hist/done slot, because the plan and the
        expert GEMM must agree on the tile, and double-buffered plans must not
        share a done counter.
        """
        from ..compact_plan import compile_tdm_compact_plan

        key = (int(align_m), int(slot))
        launch = self._plan_launches.get(key)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"compact plan for align_m={key[0]} slot={key[1]} was not "
                    "compiled before graph capture; warm this token bucket "
                    "eagerly first"
                )
            config, geo, arena = self.config, self.compact, self.res.arena
            launch = compile_tdm_compact_plan(
                rank=config.rank,
                npes=config.world_size,
                experts_per_rank=config.experts_per_rank,
                topk=config.topk,
                tile_m=key[0],
                # The arena's row count, not this step's: the plan bounds what it
                # may write against the region it writes into.
                compact_cap=geo.cap,
                off_hist=arena.offset("compact_hist") + key[1] * geo.hist_stride * 4,
                off_done=arena.offset("compact_done") + key[1] * 4,
                hist_stride=geo.hist_stride,
                max_routes=geo.max_routes,
                hist_pingpong=False,
            )
            # Compiled now, without a launch, so a graph capture never has to.
            _preload_compiled(launch, *self._plan_args(0, 0, key[1]))
            self._plan_launches[key] = launch
        return launch

    def _plan_args(self, topk_ids_ptr: int, token_count: int, slot: int) -> tuple:
        return (
            self.res.arena.handle,
            topk_ids_ptr,
            self._tok_maps[slot].data_ptr(),
            self._block_hists[slot].data_ptr(),
            self._send_bases[slot].data_ptr(),
            self._masked_ms[slot].data_ptr(),
            self._psums[slot].data_ptr(),
            self._barriers[slot].data_ptr(),
            self.config.rank,
            token_count,
            fx.Stream(self._plan_stream),
        )

    def _launch_plan(
        self, topk_ids: torch.Tensor, token_count: int, slot: int, align_m: int
    ) -> None:
        plan_stream = self._plan_stream
        plan_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(plan_stream):
            _run_compiled(
                self._plan_for(align_m, slot),
                *self._plan_args(topk_ids.data_ptr(), token_count, slot),
            )
        self._plan_event.record(plan_stream)
        self._plan_pending = True
        self._pending_align = align_m

    def _wait_plan(self) -> None:
        if not self._plan_pending:
            return
        torch.cuda.current_stream().wait_event(self._plan_event)
        self._plan_pending = False

    def close(self) -> None:
        """Nothing to release: its kernels and buffers are not held by mori."""
