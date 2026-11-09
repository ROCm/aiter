# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FlyDSL compact plan and its TDM dispatch as a dispatch stage."""

import itertools
import os
from dataclasses import dataclass

import flydsl.expr as fx
import torch

from aiter import dtypes, logger

from ...tensor_shim import _preload_compiled, _run_compiled
from ..arena import _align_up
from ..config import (
    _COMPACT_ROUTE_SPEC,
    _COMPACT_ROUTE_UPTO,
    _DISPATCH_COMPACT,
    _WAVE_SIZE,
)
from ..dispatch_tdm import _make_dispatch_tdm, tdm_max_warps, tdm_stage_capacity
from ..recipe import MegaMoEConfig
from ..s1f_ctl import (
    _QUANT_FUSE_BLOCKS,
    _Stage1Fusion,
    allocates_flags,
    tile_flag_nbytes,
)
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

    This pipeline always fuses the EP scatter into gemm2, and ``fused_moe``
    looks that shape up with ``ep_fused=True``. The compact plan has to use
    the same mode: the default ``False`` skips rows tuned for the scatter
    path and falls back to a 64 tile, so the plan alignment and the GEMM
    tile disagree.
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
        align = max(int(tile_m), int(tile_m2))
        seen = getattr(_compact_gemm_align_m, "_override_logged", None)
        if seen is None:
            seen = set()
            _compact_gemm_align_m._override_logged = seen
        sig = (ov_m, ov_m2, align)
        if sig not in seen:
            seen.add(sig)
            logger.info(
                "compact align overridden by AITER_TDM_TILE_M=%s "
                "AITER_TDM_TILE_M2=%s -> %d (CSV lookup skipped)",
                ov_m,
                ov_m2,
                align,
            )
        return align

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
        ep_fused=True,
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

    The plan needs this layer's topk_ids, so it runs on a side stream and can
    only overlap what the main stream does before the dispatch (the activation
    quant).

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
    _PLAN_SLOTS = 1

    def __init__(self, config: MegaMoEConfig, caps=None):
        # Decode walks routes, prefill tokens (see _DISPATCH_COMPACT).
        blocks_env = os.environ.get("AITER_TDM_COMPACT_BLOCKS")
        warps_env = os.environ.get("AITER_TDM_COMPACT_WARPS")
        if blocks_env is not None or warps_env is not None:
            schedule = ((None, int(blocks_env or "64"), int(warps_env or "8")),)
        else:
            # Up to _COMPACT_ROUTE_UPTO tokens/rank the steps past 256 walk
            # routes with _COMPACT_ROUTE_SPEC blocks/warps (see config.py).
            upto = _COMPACT_ROUTE_UPTO
            rb, rw = _COMPACT_ROUTE_SPEC
            head = tuple(
                e
                for e in _DISPATCH_COMPACT
                if e[0] is not None and e[0] < min(upto, 512)
            )
            schedule = head + ((upto, rb, rw, True), _DISPATCH_COMPACT[-1])
        super().__init__(config, schedule)
        self.compact = _CompactGeometry.of(config)
        self.scale_stride = _align_up(config.dispatch_scale_nbytes, 128)
        self._step = _CompactStep(
            recv_bound=config.max_recv,
            align_m=self.compact.tile_m,
            rows=self.compact.cap,
        )
        self._slot = 0
        self._plan_launches = {}
        self.fusion = _Stage1Fusion(self)

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
    def regions_for_tile(
        config: MegaMoEConfig, tile_m: int, *, s1f: bool | None = None
    ) -> dict:
        """The arena regions at GEMM tile ``tile_m``, which pads every expert's
        rows and so sizes the recv rows and their map.

        ``s1f=True`` adds the stage-1 flag region even when this config would
        not build the kernel, which is how the keyword VMM bound stays an
        upper bound. ``None`` follows ``AITER_MEGA_S1F`` and the shape gate,
        the same decision the instance allocates with.
        """
        from ..compact_plan import compact_done_nbytes

        geo = _CompactGeometry.of(config, tile_m)
        max_recv, topk = config.max_recv, config.topk
        regions = {
            "tok_off": 4,
            # 8 bytes a source, as mori's dispatch signals.
            "recv_num": 2 * config.world_size * 4,
            "recv_to_src_token": max_recv * 4,
            "out_idx": max_recv * topk * 4,
            "out_wts": max_recv * topk * 4,
            "disp_out": geo.cap * geo.wire_row,
            "ep_rowmap": (geo.cap + 1) * 8,
            "compact_hist": geo.hist_stride * 4,
            "compact_done": compact_done_nbytes(),
        }
        if s1f is None:
            s1f = allocates_flags(config)
        if s1f:
            from ..s1f_stage1 import FLAG_STRIDE

            # The keyword VMM bound forces the region on for every compact
            # wire. A world the flag word cannot hold never builds the kernel,
            # so it does not take the bytes either.
            if config.world_size * 4 <= FLAG_STRIDE:
                regions["tile_flag"] = tile_flag_nbytes(
                    compact_rows=geo.cap, world_size=config.world_size
                )
        return regions

    def build(self, res: _EpResources) -> None:
        super().build(res)
        from ..compact_plan import compact_comb_tab_dwords, compact_plan_blocks

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
        # The per-rank scatter's run table rides behind psum, so gemm2 finds it
        # through the m-tile map it already takes.
        psum_tail = (
            compact_comb_tab_dwords(npes=config.world_size, experts_per_rank=epr)
            if config.per_rank_scatter
            else 0
        )
        self._psums = [
            torch.zeros(epr + psum_tail, dtype=torch.int32, device=device)[:epr]
            for _ in slots
        ]
        self._comb_slots = [
            (
                torch.full_like(res.token_destination_map, -1)
                if config.per_rank_scatter
                else None
            )
            for _ in slots
        ]
        self._block_hists = [
            torch.empty(
                compact_plan_blocks(geo.max_routes) * segs,
                dtype=torch.int32,
                device=device,
            )
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
        # Buffers the stage-1 plan variants capture, before any of them compile.
        self.fusion.arm()
        # Warm the capacity bucket: compiling inside graph capture is not
        # allowed.
        self._warm_plans(geo.tile_m, None)
        self._precompile_s1f_plans()
        self._precompile_per_rank_plans()
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

    def prepare_step(self, recv_token_bound: int | None) -> None:
        """Fix this step's geometry before the stage-1 verdict, which is per tile."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)
        self._step = self._step_for(self._recv_bound(recv_token_bound))

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        recv_token_bound: int | None = None,
        begin_step: bool = True,
        s1f: bool = False,
        combine_quant_bits: int = 0,
    ) -> _DispatchBatch:
        """Compact cannot slice recv_x -- the rows are grouped per expert, not
        token-major -- so ``recv_token_bound`` is spent on the step's geometry
        instead: it picks the plan's alignment, the GEMM's tile and the GEMM's
        grid.

        ``s1f`` skips the dispatch kernel: the stage-1 kernel, launched in place
        of GEMM1, ships the rows off the plan's run table.

        ``combine_quant_bits`` is the wire this step combines on. The plan lays
        comb_inp out for it: dense for the per-rank scatter on a quantized
        wire, ``tok*topk + k`` otherwise. Every rank of a step must pass the
        same one, since gemm2 writes into its peers' comb_inp.
        """
        config, res = self.config, self.res
        token_count = int(hidden_states.shape[0])
        _check_recv_token_bound(config, "recv_token_bound", recv_token_bound)
        dense_comb = config.per_rank_scatter_for(combine_quant_bits)
        if begin_step:
            self.prepare_step(recv_token_bound)
        kernel = self._kernels[_pick(self.schedule, token_count)]
        # Launch the plan before the quant so the two overlap, unless the plan
        # itself is quantizing (then there is no second kernel to overlap).
        fused_quant = self.fusion.wants_fused_quant(s1f)
        self._launch_plan(
            topk_ids,
            token_count,
            self._slot,
            self._step.align_m,
            tok_map=not s1f,
            quant_src=hidden_states if fused_quant else None,
            dense_comb=dense_comb,
        )
        if fused_quant:
            payload = self._quant_payload[:token_count]
            scales_ptr = (
                self.fusion.wire[:token_count].data_ptr() + config.dispatch_token_nbytes
            )
        else:
            payload, scales_ptr, _ = self.fusion.quantize(
                hidden_states, token_count, s1f
            )
        torch.cuda.current_stream().wait_event(self._plan_event)
        slot, step = self._slot, self._step
        if s1f:
            self.fusion.stash(
                slot=slot,
                token_count=token_count,
                payload=payload,
                scales_ptr=scales_ptr,
                wts_ptr=topk_weights.data_ptr(),
            )
        else:
            stream = fx.Stream(torch.cuda.current_stream())
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
            routing=self._routing(
                token_count, self._comb_slots[slot] if dense_comb else None
            ),
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
        return batch

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        """Compile the plan this bound's tile needs.

        ``token_count`` selects the launch width. A captured batch whose width
        never ran eagerly would otherwise reach the JIT inside the capture.
        """
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)
        self._step = self._step_for(self._recv_bound(recv_token_bound))
        tok = (
            int(self.config.max_tokens_per_rank)
            if token_count is None
            else int(token_count)
        )
        self._warm_plans(self._step.align_m, tok)

    def _dense_modes(self) -> tuple:
        """The comb_inp layouts a step's plan may write: the dense one only
        when the per-rank scatter is built, for the quantized steps."""
        return (False, True) if self.config.per_rank_scatter else (False,)

    def _warm_plans(self, align_m: int, token_count: int | None) -> None:
        """Compile every plan variant a step at ``align_m`` may launch."""
        for dense in self._dense_modes():
            for slot in range(self._PLAN_SLOTS):
                self._plan_for(align_m, slot, token_count, dense_comb=dense)
                if self.fusion.plan_mode(align_m):
                    self._plan_for(
                        align_m, slot, token_count, tok_map=False, dense_comb=dense
                    )
                    if _QUANT_FUSE_BLOCKS:
                        self._plan_for(
                            align_m,
                            slot,
                            token_count,
                            tok_map=False,
                            quant_blocks=_QUANT_FUSE_BLOCKS,
                            dense_comb=dense,
                        )

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

    def _precompile_s1f_plans(self) -> None:
        """Compile every stage-1 plan variant before serving.

        With S1F on (the default) the plan is keyed by (align_m, slot, waves,
        blocks, tok_map, fused quant), and waves/blocks follow the step's token
        count, so eager steps (prefill, or any width the warmup never ran)
        reached the JIT inside the serving loop (the first ~100 s of a c512
        run had 250-470 ms prefill steps). This walks every recv bound a
        fused step can have (<= the S1F max recv, 4096), takes the
        alignments they map to, and one token count per distinct
        (waves, blocks), and compiles all variants (compile only, no launch).
        """
        if not self.fusion.enabled:
            return
        from ..compact_plan import compact_plan_blocks
        from ..s1f_ctl import _QUANT_FUSE_BLOCKS as _qfb
        from ..s1f_ctl import _S1F_MAX_RECV as _lim

        config = self.config
        limit = min(int(config.max_recv), int(_lim))
        saved = self._step
        bounds = sorted(set(range(16, limit + 1, 16)) | {1, limit})
        steps = {}
        for b in bounds:
            st = self._step_for(b)
            steps.setdefault(int(st.align_m), st)
        max_tok = min(int(config.max_tokens_per_rank), limit)
        widths = {}
        for tok in range(1, max_tok + 1):
            routes = tok * int(config.topk)
            wb = (self._plan_waves_for(tok), compact_plan_blocks(routes))
            widths.setdefault(wb, tok)
        try:
            for align_m, st in sorted(steps.items()):
                self._step = st
                if not self.fusion.plan_mode(align_m):
                    continue
                for tok in sorted(widths.values()):
                    for dense, slot in itertools.product(
                        self._dense_modes(), range(self._PLAN_SLOTS)
                    ):
                        self._plan_for(align_m, slot, tok, dense_comb=dense)
                        self._plan_for(
                            align_m, slot, tok, tok_map=False, dense_comb=dense
                        )
                        if _qfb:
                            self._plan_for(
                                align_m,
                                slot,
                                tok,
                                tok_map=False,
                                quant_blocks=_qfb,
                                dense_comb=dense,
                            )
        finally:
            self._step = saved

    def _plan_waves_for(self, token_count: int | None) -> int:
        """Wave count for this step's route count.

        The histogram layout stays on the arena capacity so every rank agrees
        on the stride. Only the launch width follows the step.
        """
        from ..compact_plan import compact_plan_waves

        routes = self.compact.max_routes
        if token_count is not None:
            routes = min(routes, max(1, int(token_count) * int(self.config.topk)))
        waves = compact_plan_waves(npes=self.config.world_size, max_routes=routes)
        if self.fusion.enabled and not self.fusion.step_off():
            # The stage-1 plan runs alone between the quant and the kernel,
            # and its tables want the wide scan.
            waves = max(waves, 16)
        return waves

    def _plan_blocks_for(self, token_count: int | None) -> int:
        """Block count for this step's route count (the arena's when None)."""
        from ..compact_plan import compact_plan_blocks

        routes = self.compact.max_routes
        if token_count is not None:
            routes = int(token_count) * int(self.config.topk)
        return compact_plan_blocks(routes)

    def _precompile_per_rank_plans(self) -> None:
        """Compile the per-rank scatter's (dense) S1F=0 plan for every block
        count.

        Off the stage-1 path that plan is keyed by its block count, which
        follows the step's token count, so a prefill width the warmup never
        ran would otherwise reach the JIT inside the serving loop. This walks
        the recv bounds that run the S1F=0 plan, takes the alignments they
        map to, and compiles one plan per distinct block count.
        """
        config = self.config
        if not config.per_rank_scatter:
            return
        from ..s1f_ctl import _S1F_MAX_RECV

        max_recv = int(config.max_recv)
        lo = int(_S1F_MAX_RECV) + 1 if self.fusion.enabled else 1
        if lo > max_recv:
            return
        steps = {}
        for b in sorted(set(range(lo, max_recv + 1, 16)) | {max_recv}):
            st = self._step_for(b)
            steps.setdefault(int(st.align_m), st)
        widths = {}
        for tok in range(1, int(config.max_tokens_per_rank) + 1):
            widths.setdefault(self._plan_blocks_for(tok), tok)
        saved = self._step
        try:
            for align_m, st in sorted(steps.items()):
                self._step = st
                for tok in sorted(widths.values()):
                    for slot in range(self._PLAN_SLOTS):
                        self._plan_for(align_m, slot, tok, dense_comb=True)
        finally:
            self._step = saved

    def _slot_bufs(self, slot: int) -> dict:
        bufs = {
            "tok_map": self._tok_maps[slot],
            "block_hist": self._block_hists[slot],
            "send_base": self._send_bases[slot],
            "masked_m": self._masked_ms[slot],
            "psum": self._psums[slot],
            "barrier": self._barriers[slot],
        }
        bufs.update(self.fusion.extra_bufs(slot))
        return bufs

    def _plan_for(
        self,
        align_m: int,
        slot: int,
        token_count: int | None = None,
        tok_map: bool = True,
        quant_blocks: int = 0,
        dense_comb: bool = False,
    ):
        """The compact-plan launch that aligns each expert's rows to ``align_m``.

        Compiled per alignment, hist/done slot, launch width, comb_inp layout
        (``dense_comb`` for a per-rank scatter step) and, on the stage-1 path,
        whether it rewrites tok_map or quantizes in the same launch. The plan
        and the expert GEMM must agree on the tile, and double-buffered plans
        must not share a done counter.
        """
        from ..compact_plan import compact_plan_waves, compile_tdm_compact_plan

        align_m = int(align_m)
        slot = int(slot)
        # A step over the S1F max recv keeps the S1F=0 plan.
        fused = self.fusion.enabled and not self.fusion.step_off()
        plan_s1f = fused and self.fusion.plan_mode(align_m)
        write_tok_map = (not plan_s1f) or bool(tok_map)
        qb = int(quant_blocks) if plan_s1f else 0
        dense_comb = bool(dense_comb)
        if dense_comb and not self.config.per_rank_scatter:
            raise ValueError("a dense comb_inp plan needs the per-rank scatter built")
        # With the stage-1 kernel off, one launch per alignment and slot, as
        # before: the width is the plan's own default. On, the width follows
        # the step, and tok_map / the fused quant are their own kernels.
        if fused:
            waves = self._plan_waves_for(token_count)
            blocks = self._plan_blocks_for(token_count)
            key = (align_m, slot, waves, blocks, write_tok_map, plan_s1f, qb)
        else:
            # The per-rank plan writes comb_slot for every route, so past one
            # block's LDS route table it splits the routes across blocks like
            # the fused plan does.
            waves = None
            blocks = self._plan_blocks_for(token_count) if dense_comb else None
            key = (align_m, slot, blocks)
        key += (dense_comb,)
        launch = self._plan_launches.get(key)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"compact plan for align_m={align_m} slot={slot} "
                    f"waves={waves} blocks={blocks} dense_comb={dense_comb} "
                    "was not compiled before "
                    "graph capture; warm this token bucket eagerly first"
                )
            config, geo, arena = self.config, self.compact, self.res.arena
            extra = {}
            if fused:
                full_waves = compact_plan_waves(
                    npes=config.world_size, max_routes=geo.max_routes
                )
                extra = {
                    "plan_waves": None if waves == full_waves else waves,
                    "plan_blocks": blocks,
                    "quant_blocks": qb,
                    "quant_model_dim": config.hidden_dim if qb else 0,
                    "quant_row_stride": geo.wire_row if qb else 0,
                    "quant_payload_bytes": (config.dispatch_token_nbytes if qb else 0),
                }
                if plan_s1f:
                    extra.update(s1f=True, write_tok_map=write_tok_map, align_m=align_m)
            elif blocks is not None:
                extra = {"plan_blocks": blocks}
            # Stage-1: runs and tile_info per producer flag tile; experts are
            # still padded to the step's GEMM1 tile (align_m).
            tile_m = align_m
            if plan_s1f:
                from ..s1f_stage1 import FLAG_M

                tile_m = FLAG_M
            launch = compile_tdm_compact_plan(
                rank=config.rank,
                npes=config.world_size,
                experts_per_rank=config.experts_per_rank,
                topk=config.topk,
                tile_m=tile_m,
                # The arena's row count, not this step's: the plan bounds what
                # it may write against the region it writes into.
                compact_cap=geo.cap,
                off_hist=arena.offset("compact_hist") + slot * geo.hist_stride * 4,
                off_done=arena.offset("compact_done") + slot * 4,
                hist_stride=geo.hist_stride,
                max_routes=geo.max_routes,
                hist_pingpong=False,
                dense_comb=dense_comb,
                **extra,
            )
            # Compiled now, without a launch, so a graph capture never has to.
            _preload_compiled(
                launch,
                *self._plan_args(
                    0, 0, slot, plan_s1f=plan_s1f, quant_src_ptr=0, wire_ptr=0
                ),
            )
            self._plan_launches[key] = launch
        return launch

    def _plan_args(
        self,
        topk_ids_ptr: int,
        token_count: int,
        slot: int,
        *,
        plan_s1f: bool = False,
        quant_src_ptr: int = 0,
        wire_ptr: int = 0,
    ) -> tuple:
        bufs = self._slot_bufs(slot)
        comb_slot = self._comb_slots[slot]
        args = (
            self.res.arena.handle,
            topk_ids_ptr,
            bufs["tok_map"].data_ptr(),
            bufs["block_hist"].data_ptr(),
            bufs["send_base"].data_ptr(),
            bufs["masked_m"].data_ptr(),
            bufs["psum"].data_ptr(),
            bufs["barrier"].data_ptr(),
            comb_slot.data_ptr() if comb_slot is not None else 0,
            self.config.rank,
            token_count,
        )
        if plan_s1f:
            args += (
                bufs["send_list"].data_ptr(),
                bufs["runs"].data_ptr(),
                bufs["tile_src"].data_ptr(),
                int(quant_src_ptr),
                int(wire_ptr),
            )
        return args + (fx.Stream(self._plan_stream),)

    def _launch_plan(
        self,
        topk_ids: torch.Tensor,
        token_count: int,
        slot: int,
        align_m: int,
        *,
        tok_map: bool = True,
        quant_src: torch.Tensor | None = None,
        dense_comb: bool = False,
    ) -> None:
        plan_s1f = self.fusion.plan_mode(align_m) and not self.fusion.step_off()
        qb = _QUANT_FUSE_BLOCKS if plan_s1f and quant_src is not None else 0
        plan_stream = self._plan_stream
        plan_stream.wait_stream(torch.cuda.current_stream())
        quant_ptr = 0 if quant_src is None else quant_src.data_ptr()
        wire_ptr = 0 if self.fusion.wire is None else self.fusion.wire.data_ptr()
        with torch.cuda.stream(plan_stream):
            _run_compiled(
                self._plan_for(
                    align_m,
                    slot,
                    token_count,
                    tok_map=tok_map,
                    quant_blocks=qb,
                    dense_comb=dense_comb,
                ),
                *self._plan_args(
                    topk_ids.data_ptr(),
                    token_count,
                    slot,
                    plan_s1f=plan_s1f,
                    quant_src_ptr=quant_ptr,
                    wire_ptr=wire_ptr,
                ),
            )
        self._plan_event.record(plan_stream)

    def close(self) -> None:
        """Nothing to release: its kernels and buffers are not held by mori."""
