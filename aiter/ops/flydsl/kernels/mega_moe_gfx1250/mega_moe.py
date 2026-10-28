# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Stage2-fused MegaMoE host pipeline."""

import contextlib
import functools
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Protocol

import flydsl.expr as fx
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

from ..tensor_shim import _run_compiled
from .combine import (
    CHUNK_ELEMS as _COMBINE_CHUNK_ELEMS,
)
from .combine import (
    COMBINE_LDS_BUDGET,
    _make_combine_fused_reduce,
    _make_combine_fused_sync,
    combine_reduce_lds_bytes,
)
from .config import _DISPATCH_COMPACT, _WAVE_SIZE
from .dispatch_tdm import _make_dispatch_tdm, tdm_max_warps, tdm_stage_capacity
from .types import COMBINE_SCALE_BLOCK as _COMBINE_SCALE_BLOCK
from .types import Stage2ScatterContext, _from_gpu_ptr

__all__ = [
    "CombineKind",
    "DispatchKind",
    "ExpertComputeConfig",
    "MegaMoEGeometry",
    "MegaMoEGfx1250",
    "MegaMoERecipe",
]

_MAX_WORLD_SIZE = 72
_MAX_EXPERTS_PER_RANK = 512

# "mxfp8" is what the mega_moe v2 path calls "fp8_blockwise_1x32" (kernels/
# flydsl_dispatch_combine_intranode_op.py); that path has no fp4.
_COMBINE_QUANT_MODES = ("none", "mxfp8", "mxfp4")
_COMBINE_QUANT_BITS = {"none": 0, "mxfp8": 8, "mxfp4": 4}
# The combine wires mori offers, as its quant_type names.
# "fp4_blockwise" is MXFP4 -- e2m1 plus one e8m0 per 32 -- the same format the
# fused reduce calls "mxfp4". It has no MXFP8.
_MORI_COMBINE_QUANT = {0: "none", 4: "fp4_blockwise"}
# The fp4 push combine reduces 32 elements a lane over a 32-lane wave.
_MORI_FP4_COMBINE_HIDDEN_ALIGN = 1024
# Preferred lane tile for the TDM combine: T tokens x C chunks per block
# iteration, which the reduce turns into T*C*256/16 lanes -- so the quantized
# path runs a wider block than the bf16 one. A ceiling, not the answer:
# `_combine_tile` walks these down until the tile fits a given topk.
_COMBINE_TOKENS_PER_BLOCK = 16
_COMBINE_CHUNKS_PER_ITER = 1
# The MXFP8 wire wants a larger C: its scale plane puts only C*8 bytes in a TDM
# row, so a small C makes that load pull a whole cache line per row for a
# handful of bytes.
_COMBINE_QUANT_TOKENS_PER_BLOCK = 8
_COMBINE_QUANT_CHUNKS_PER_ITER = 4


def _combine_tile(*, topk, hidden_dim, quant_bits):
    """(tokens_per_block, chunks_per_iter) for the reduce, widest that fits.

    C has to divide the hidden chunk count, since the reduce has no tail path
    along hidden, and 2*topk, so the tile's T*topk rows spread evenly over the
    block's T*C/2 warps.

    T then takes the tile as high as the LDS budget allows, since trips per block
    is what keeps the prefetch pipeline fed. LDS grows as T*topk, so a large topk
    trades height away: topk=8 on the mxfp8 wire needs 164KB at the preferred
    T=8 and lands on T=7.
    """
    n_chunks = hidden_dim // _COMBINE_CHUNK_ELEMS
    max_chunks, max_toks = (
        (_COMBINE_QUANT_CHUNKS_PER_ITER, _COMBINE_QUANT_TOKENS_PER_BLOCK)
        if quant_bits
        else (_COMBINE_CHUNKS_PER_ITER, _COMBINE_TOKENS_PER_BLOCK)
    )
    for chunks in range(max_chunks, 0, -1):
        if n_chunks % chunks or (2 * topk) % chunks:
            continue
        for toks in range(max_toks, 0, -1):
            lanes = toks * chunks * _COMBINE_CHUNK_ELEMS // 16
            if lanes % _WAVE_SIZE:
                continue
            # The tile's copies pass the block's warp count to a TDM atom, which
            # takes only a power of two.
            warps = lanes // _WAVE_SIZE
            if warps & (warps - 1):
                continue
            if (
                combine_reduce_lds_bytes(
                    experts_per_token=topk,
                    quant_bits=quant_bits,
                    tokens_per_block=toks,
                    chunks_per_iter=chunks,
                )
                <= COMBINE_LDS_BUDGET
            ):
                return toks, chunks
    raise ValueError(
        f"no combine reduce tile fits the {COMBINE_LDS_BUDGET // 1024}KB LDS "
        f"budget for topk={topk}, hidden_dim={hidden_dim}, "
        f"combine_quant_bits={quant_bits}"
    )


# mori's C++ EpArgs offset stems -> this package's arena region names, shared by
# the dispatch and the combine plan. A dispatch plan binds all of them but never
# touches `outTok`; the combine stages its input there and pulls peers' rows
# from it.
_MORI_REGION_NAMES = {
    "tokOff": "tok_off",
    "recvNum": "recv_num",
    "recvToSrc": "recv_to_src_token",
    "outIdx": "out_idx",
    "outWts": "out_wts",
    "dispOut": "disp_out",
    "outTok": "comb_inp",
    "xdb": "cross_device_barrier",
    # Only laid out on a quantizing wire; plan_api binds a missing region to 0 and
    # the kernel's `if constexpr` keeps that 0 from being read.
    "outScales": "disp_out_scales",
}


def read_dispatch_wire_env() -> str:
    """$MEGA_DISPATCH_WIRE, defaulting to bf16."""
    return os.environ.get("MEGA_DISPATCH_WIRE") or "bf16"


@dataclass(frozen=True)
class _DispatchWire:
    """One DISPATCH wire. Per token at hidden 7168: bf16 14336 B, fp8 7168 + 256,
    fp4 3584 + 256.

    Dispatch-only on purpose: a quantized combine cannot reuse this. mori
    carries no scales on a combine, the quant would have to happen inside the
    gemm2 epilogue on an LDS tile rather than host-side on a whole tensor, and
    recv_dtype exists only to build a torch view combine has no equivalent of.
    Only payload_bytes would carry over.
    """

    payload_bytes: float  # PER FEATURE; fp4 packs two features into a byte
    mori_dtype: torch.dtype
    quant_dtype: torch.dtype | None  # None: nothing for the sender to quantize to
    # fp4 is viewed as raw bytes: the gather addresses a row in BYTES, and a
    # packed dtype would make shape[-1] read as a feature count.
    recv_dtype: torch.dtype
    # mori's tuning-table key for this wire, its op layer's dtype_str: an fp4
    # dispatch pairs with a bf16-dtype combine, so it is "fp4_disp_bf16_comb".
    mori_tuning_dtype: str


_DISPATCH_WIRE_SPECS = {
    "bf16": _DispatchWire(2, torch.bfloat16, None, torch.bfloat16, "bf16"),
    "fp8": _DispatchWire(1, dtypes.fp8, dtypes.fp8, dtypes.fp8, "fp8"),
    "fp4": _DispatchWire(
        0.5, dtypes.fp4x2, dtypes.fp4x2, torch.uint8, "fp4_disp_bf16_comb"
    ),
}
_DISPATCH_WIRES = tuple(_DISPATCH_WIRE_SPECS)
# What mori's describe() resolves a request against: MegaMoE runs only here, and
# naming it keeps the sizing off the GPU.
_MORI_ARCH = "gfx1250"


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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


@dataclass(frozen=True)
class _MoriCapabilities:
    """The installed mori, as MegaMoE drives it.

    MegaMoE drives mori's EpDispatchPlan / EpCombinePlan directly, past its op
    layer, so every mori name it uses is reached through here. It needs plans
    that describe() themselves (ROCm/mori#844): the arena regions, the staging
    buffer and the scale stride all come from that, so none of mori's sizing is
    copied here. Such a mori also has what the pipeline relies on beyond sizing:
    the aggregated dispatch reservation the fp4 push combine reads its landing
    offsets from, and a dispatch that overwrites total_recv, so nothing zeroes
    it ahead of one.
    """

    ep_plans: object
    hip_backend: object
    lookup: object

    def tuning(self, config, quant_type: str = "none") -> dict:
        return self.lookup(
            config.world_size,
            config.hidden_dim,
            config.topk,
            dtype=config.dispatch_wire_spec.mori_tuning_dtype,
            experts_per_rank=config.experts_per_rank,
            quant_type=quant_type,
        )

    def dispatch_schedule(self, config) -> tuple:
        """mori's own tuned buckets, as this package's (bound, block, warp) triples.

        Not `_select_dispatch_config`'s: that table asks for 32 warps above 256
        tokens, and mori's gfx1250 dispatch stages a hidden-dim tile per warp in
        dynamic LDS -- 32 * 7168 * 2 = 458 KB, past the 320 KB budget. EpCfgIsValid
        does not check LDS, so the plan would build and then fail at launch.
        """
        tuned = self.tuning(config)
        schedule = tuned["schedule"]
        if schedule:
            return tuple((bound, block, warp) for bound, block, warp, _, _ in schedule)
        return ((None, tuned["dispatch_block_num"], tuned["warp_num_per_block"]),)

    def combine_schedule(self, config, quant_bits: int = 0) -> tuple:
        """The combine half of mori's tuned buckets for the ``quant_bits`` wire, as
        (bound, block, warp)."""
        tuned = self.tuning(config, _MORI_COMBINE_QUANT[quant_bits])
        schedule = tuned["schedule"]
        if schedule:
            return tuple((bound, block, warp) for bound, _, _, block, warp in schedule)
        return (
            (None, tuned["combine_block_num"], tuned["combine_warp_num_per_block"]),
        )

    def describe_dispatch(self, requests) -> list[dict]:
        """What mori reports for each dispatch plan request -- its arena regions,
        staging bytes and scale stride -- without building or compiling one."""
        plan = self.ep_plans.EpDispatchPlan
        return [plan.describe(arch=_MORI_ARCH, **request) for request in requests]

    def describe_combine(self, requests) -> list[dict]:
        """What mori reports for each combine plan request."""
        plan = self.ep_plans.EpCombinePlan
        return [plan.describe(arch=_MORI_ARCH, **request) for request in requests]

    def region_sizes(self, infos) -> dict:
        """The arena regions ``infos`` ask for, under this package's names."""
        return dict(
            self.ep_plans.arena_regions(*infos, region_names=_MORI_REGION_NAMES)
        )

    def dispatch_plan(self, request: dict, arena):
        return self.ep_plans.EpDispatchPlan(
            **request, arena=arena, region_names=_MORI_REGION_NAMES
        )

    def combine_plan(self, request: dict, arena):
        return self.ep_plans.EpCombinePlan(
            **request, arena=arena, region_names=_MORI_REGION_NAMES
        )

    def tokoff_ext_wanted(self, config) -> bool:
        """Whether mori's slot allocator word moves out of the cco window into
        tokoff-ext memory, as mori's op layer decides it.

        Collective when $MORI_EP_TOKOFF_EXT is on: TokOffExt.wanted checks that
        every rank shares the host.
        """
        return bool(self.hip_backend.TokOffExt.wanted(config.world_size))

    def tokoff_ext(self, config, device):
        """mori's tokoff-ext allocation, reused rather than duplicated: its IPC
        handle protocol must match the kernel byte for byte. Collective."""
        return self.hip_backend.TokOffExt(config.rank, config.world_size, device)

    @property
    def xdb_flag_slots(self) -> int:
        """Per-block epoch slots of the combine's entry barrier."""
        return self.hip_backend._XDB_FLAG_SLOTS


@functools.cache
def _mori_capabilities() -> _MoriCapabilities:
    try:
        from mori.ops.dispatch_combine_v2 import ep_plans
    except ImportError as error:
        raise RuntimeError(
            "MegaMoE's mori dispatch and combine need a mori with "
            "ops/dispatch_combine_v2"
        ) from error
    except OSError as error:
        raise RuntimeError(
            "MegaMoE's mori dispatch and combine need mori's libmori_ops_v2.so; "
            "it is built by mori's CMake and is not shipped by every install"
        ) from error
    if not (
        hasattr(ep_plans.EpDispatchPlan, "describe")
        and hasattr(ep_plans, "arena_regions")
    ):
        raise RuntimeError(
            "MegaMoE sizes mori's arena regions with Plan.describe(), which needs "
            "ROCm/mori#844 or later; the installed mori has none"
        )
    from mori.ops.dispatch_combine_v2 import hip_backend
    from mori.ops.dispatch_combine_v2.hip_tuning_configs import lookup

    return _MoriCapabilities(ep_plans=ep_plans, hip_backend=hip_backend, lookup=lookup)


class SymmetricArena:
    """One cco symmetric allocation, registered as a window and cut into the
    named regions of an ``_ArenaLayout`` (or a [(name, nbytes)] list, laid down
    in its own order)."""

    _ALIGNMENT = 256

    def __init__(self, communicator, layout):
        if not isinstance(layout, _ArenaLayout):
            layout = _ArenaLayout.of(layout)
        self._communicator = communicator
        self.layout = layout
        self._offsets = dict(layout.offsets)
        self._sizes = dict(layout.regions)
        self._total_bytes = layout.total_bytes
        self._memory = communicator.alloc_mem(self._total_bytes)
        try:
            self._window = communicator.register_window(
                self._memory.ptr, self._total_bytes
            )
        except BaseException:
            self._memory.close()
            raise

    @property
    def handle(self) -> int:
        return self._window.handle

    def offset(self, name: str) -> int:
        return self._offsets[name]

    def local_ptr(self, name: str) -> int:
        return self._window.local_ptr + self._offsets[name]

    def size(self, name: str) -> int:
        """A region's bytes; mori's plans check theirs against it at bind."""
        return self._sizes[name]

    def zero(self, name: str | None = None):
        if name is None:
            pointer, size = self._window.local_ptr, self._total_bytes
        else:
            pointer = self.local_ptr(name)
            size = self._sizes[name]
        _from_gpu_ptr(pointer, (size,), torch.int8).zero_()

    def close(self):
        try:
            self._window.close()
        finally:
            self._memory.close()


class DispatchKind(str, Enum):
    """How tokens reach their experts' rank."""

    # mori's EpDispatchPlan: a recv row per received token.
    MORI = "mori"
    # The FlyDSL compact plan: recv rows grouped per expert, the layout the
    # grouped GEMM reads in place.
    COMPACT = "compact"


class CombineKind(str, Enum):
    """How expert results return to their tokens."""

    # mori's EpCombinePlan over fused_moe's per-recv-row output.
    MORI = "mori"
    # The gemm2 epilogue P2P-scatters into the peers' slots and a FlyDSL reduce
    # sums them.
    FUSED_SCATTER = "fused_scatter"


# The combine's wire names, as the legacy combine_quant spells them.
_COMBINE_WIRE_QUANT = {"bf16": "none", "fp8": "mxfp8", "fp4": "mxfp4"}
_COMBINE_QUANT_WIRE = {quant: wire for wire, quant in _COMBINE_WIRE_QUANT.items()}


@dataclass(frozen=True)
class ExpertComputeConfig:
    """The expert GEMM a MegaMoE runs between its dispatch and its combine."""

    activation: ActivationType = ActivationType.Silu
    gate_mode: GateMode = GateMode.INTERLEAVE
    quant_type: QuantType = QuantType.per_1x32
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float = 0.0

    def __post_init__(self):
        def coerce(name, value):
            object.__setattr__(self, name, value)

        coerce("activation", ActivationType(self.activation))
        coerce("gate_mode", GateMode(self.gate_mode))
        coerce("quant_type", QuantType(self.quant_type))
        coerce("hidden_pad", int(self.hidden_pad))
        coerce("intermediate_pad", int(self.intermediate_pad))
        coerce("swiglu_limit", float(self.swiglu_limit))
        if self.hidden_pad < 0 or self.intermediate_pad < 0:
            raise ValueError(
                "padding must be non-negative, got "
                f"{self.hidden_pad}, {self.intermediate_pad}"
            )
        if self.swiglu_limit < 0:
            raise ValueError(
                f"swiglu_limit must be non-negative, got {self.swiglu_limit}"
            )
        if self.gate_mode != GateMode.INTERLEAVE:
            raise ValueError("MegaMoE's expert GEMM requires gate_mode=INTERLEAVE")
        if self.quant_type != QuantType.per_1x32:
            raise ValueError("MegaMoE's expert GEMM requires quant_type=per_1x32")
        if self.activation not in (
            ActivationType.Silu,
            ActivationType.Swiglu,
            ActivationType.Situv2,
        ):
            raise ValueError(
                "MegaMoE's expert GEMM does not support "
                f"activation={self.activation}"
            )


@dataclass(frozen=True)
class MegaMoERecipe:
    """What a MegaMoE runs: its transport, and the expert GEMM in between.

    Policy only -- the model's shape is a ``MegaMoEGeometry``. A caller picks a
    recipe and hands it to ``MegaMoEGfx1250.from_recipe``; everything built from
    it is fixed there, so forward() never chooses a path. Hashable, so it can key
    a cache of instances.

    Which pairings run is the stages' to say: a dispatch declares the wires it
    carries and the row layout it delivers, a combine the layouts it accepts and
    the wires it offers.

    ``combine_wires`` are the formats a combine is BUILT for. bf16 is always one
    of them, and each step names the one it runs on (forward's ``combine_wire``);
    a quantized wire only pays off on steps with enough tokens.
    """

    dispatch: DispatchKind = DispatchKind.MORI
    combine: CombineKind = CombineKind.FUSED_SCATTER
    # Must MATCH the expert GEMM's A operand (a8w4 -> fp8, a4w4 -> fp4): the
    # receiver hands the payload to the GEMM as-is, so a mismatch is a width
    # error, not a slow path. bf16 has the GEMM quantize on arrival.
    dispatch_wire: str = "bf16"
    combine_wires: tuple = ("bf16",)
    compute: ExpertComputeConfig = field(default_factory=ExpertComputeConfig)

    def __post_init__(self):
        def coerce(name, value):
            object.__setattr__(self, name, value)

        coerce("dispatch", DispatchKind(self.dispatch))
        coerce("combine", CombineKind(self.combine))
        if not isinstance(self.compute, ExpertComputeConfig):
            raise TypeError(
                "compute must be an ExpertComputeConfig, got "
                f"{type(self.compute).__name__}"
            )
        wires = tuple(self.combine_wires)
        unknown = [w for w in wires if w not in _COMBINE_WIRE_QUANT]
        if unknown or "bf16" not in wires or len(set(wires) - {"bf16"}) > 1:
            raise ValueError(
                "combine_wires must be ('bf16',) or bf16 plus one of fp8/fp4, "
                f"got {self.combine_wires!r}"
            )
        # Canonical order, so equal recipes hash equal.
        coerce("combine_wires", tuple(w for w in _COMBINE_WIRE_QUANT if w in wires))
        if self.dispatch_wire not in _DISPATCH_WIRES:
            raise ValueError(
                f"dispatch_wire must be one of {_DISPATCH_WIRES}, "
                f"got {self.dispatch_wire!r}"
            )
        dispatch, combine = _stage_classes(self)
        if self.dispatch_wire not in dispatch.wires:
            raise ValueError(
                f"dispatch={self.dispatch.value} carries an "
                f"{' or '.join(dispatch.wires)} dispatch_wire, "
                f"got {self.dispatch_wire!r}"
            )
        if dispatch.output_layout not in combine.accepted_layouts:
            takers = [
                kind.value
                for kind, stage in _stage_registry()[1].items()
                if dispatch.output_layout in stage.accepted_layouts
            ]
            raise ValueError(
                f"combine={self.combine.value} does not accept the "
                f"{dispatch.output_layout.value} rows "
                f"dispatch={self.dispatch.value} delivers; "
                f"combine={' or '.join(takers)} does"
            )
        if not set(self.combine_wires) <= set(combine.wires):
            raise ValueError(
                f"combine={self.combine.value} offers the {combine.wires} wires, "
                f"got combine_wires={self.combine_wires!r}"
            )

    @classmethod
    def from_legacy(
        cls,
        *,
        stage1_fused: bool = False,
        stage2_fused: bool = True,
        dispatch_backend: str | None = None,
        dispatch_wire: str | None = None,
        combine_quant: str | None = None,
        activation: ActivationType = ActivationType.Silu,
        gate_mode: int = GateMode.INTERLEAVE.value,
        quant_type: QuantType = QuantType.per_1x32,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        swiglu_limit: float = 0.0,
    ) -> "MegaMoERecipe":
        """The recipe the old constructor flags describe.

        The only reader of stage1_fused, stage2_fused and dispatch_backend, and of
        the $MEGA_DISPATCH_WIRE / $MEGA_COMBINE_QUANT defaults.
        """
        # Accepted for callers that still name the dispatch; stage1_fused decides.
        if dispatch_backend is not None:
            if dispatch_backend not in ("flydsl", "mori"):
                raise ValueError(
                    "dispatch_backend must be 'flydsl' or 'mori', "
                    f"got {dispatch_backend!r}"
                )
            if (dispatch_backend == "flydsl") != bool(stage1_fused):
                raise ValueError(
                    f"dispatch_backend={dispatch_backend!r} does not match "
                    f"stage1_fused={bool(stage1_fused)}: the flydsl dispatch runs "
                    "only as the compact plan (stage1_fused=True), and "
                    "stage1_fused=False dispatches through mori"
                )
        if combine_quant is None:
            combine_quant = os.environ.get("MEGA_COMBINE_QUANT", "none")
        if combine_quant not in _COMBINE_QUANT_MODES:
            raise ValueError(
                f"combine_quant must be one of {_COMBINE_QUANT_MODES}, "
                f"got {combine_quant!r}"
            )
        wire = _COMBINE_QUANT_WIRE[combine_quant]
        return cls(
            dispatch=DispatchKind.COMPACT if stage1_fused else DispatchKind.MORI,
            combine=CombineKind.FUSED_SCATTER if stage2_fused else CombineKind.MORI,
            dispatch_wire=(
                dispatch_wire if dispatch_wire is not None else read_dispatch_wire_env()
            ),
            combine_wires=("bf16",) if wire == "bf16" else ("bf16", wire),
            compute=ExpertComputeConfig(
                activation=activation,
                gate_mode=gate_mode,
                quant_type=quant_type,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                swiglu_limit=swiglu_limit,
            ),
        )

    @property
    def is_quant_dispatch_wire(self) -> bool:
        """The wire carries an MX payload plus its e8m0 row, not bf16."""
        return self.dispatch_wire in ("fp8", "fp4")

    @property
    def combine_quant(self) -> str:
        """The quantized combine format built, in the legacy spelling."""
        return _COMBINE_WIRE_QUANT[self.combine_wires[-1]]


def _stage_registry() -> tuple:
    """(dispatch stage per DispatchKind, combine stage per CombineKind)."""
    return _DISPATCH_STAGES, _COMBINE_STAGES


def _stage_classes(recipe: MegaMoERecipe) -> tuple:
    """The (dispatch, combine) stage classes ``recipe`` names."""
    dispatch_stages, combine_stages = _stage_registry()
    return dispatch_stages[recipe.dispatch], combine_stages[recipe.combine]


@dataclass(frozen=True)
class MegaMoEGeometry:
    """The model and EP shape one MegaMoE serves; one per EP rank."""

    rank: int
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tokens_per_rank: int

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.experts <= 0:
            raise ValueError(f"experts must be positive, got {self.experts}")
        if self.experts % self.world_size:
            raise ValueError(
                f"experts={self.experts} must be divisible by "
                f"world_size={self.world_size}"
            )
        if self.max_tokens_per_rank <= 0:
            raise ValueError(
                "max_tokens_per_rank must be positive, got "
                f"{self.max_tokens_per_rank}"
            )
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.topk > self.experts:
            raise ValueError(f"topk={self.topk} cannot exceed experts={self.experts}")
        if self.model_dim <= 0 or self.inter_dim <= 0:
            raise ValueError(
                f"model_dim and inter_dim must be positive, got "
                f"{self.model_dim}, {self.inter_dim}"
            )

    @property
    def experts_per_rank(self) -> int:
        return self.experts // self.world_size


@dataclass(frozen=True)
class MegaMoEConfig:
    """A recipe resolved against a geometry: everything the pipeline sizes and
    builds from, checked once.

    ``stage1_fused``, ``stage2_fused`` and ``dispatch_backend`` are kept for
    callers that read them; nothing here consults them.
    """

    recipe: MegaMoERecipe
    geometry: MegaMoEGeometry

    def __post_init__(self):
        if self.is_quant_dispatch_wire and self.hidden_dim % 32:
            raise ValueError(
                "one e8m0 scale covers 32 features, so a quantizing dispatch "
                f"wire needs hidden_dim % 32 == 0, got {self.hidden_dim}"
            )
        for stage in _stage_classes(self.recipe):
            stage.check(self)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank={self.rank} must be in [0, {self.world_size})")
        if self.world_size > _MAX_WORLD_SIZE:
            raise ValueError(
                f"rack-scale dispatch requires world_size <= {_MAX_WORLD_SIZE}, "
                f"got {self.world_size}"
            )
        if not 0 < self.topk <= _WAVE_SIZE:
            raise ValueError(
                f"dispatch requires topk in [1, {_WAVE_SIZE}], got {self.topk}"
            )
        if not 0 < self.experts_per_rank <= _MAX_EXPERTS_PER_RANK:
            raise ValueError(
                f"fused EP psum requires experts_per_rank in "
                f"[1, {_MAX_EXPERTS_PER_RANK}], got {self.experts_per_rank}"
            )
        if self.hidden_dim * 2 % 16:
            raise ValueError(
                f"bf16 token bytes must be 16-byte aligned, "
                f"got hidden_dim={self.hidden_dim}"
            )

    # -- geometry --------------------------------------------------------------
    @property
    def rank(self) -> int:
        return self.geometry.rank

    @property
    def world_size(self) -> int:
        return self.geometry.world_size

    @property
    def hidden_dim(self) -> int:
        return self.geometry.model_dim

    @property
    def max_tokens_per_rank(self) -> int:
        return self.geometry.max_tokens_per_rank

    @property
    def experts_per_rank(self) -> int:
        return self.geometry.experts_per_rank

    @property
    def topk(self) -> int:
        return self.geometry.topk

    @property
    def max_recv(self) -> int:
        return self.world_size * self.max_tokens_per_rank

    # -- the recipe, under the names callers read --------------------------------
    @property
    def dispatch_wire(self) -> str:
        return self.recipe.dispatch_wire

    @property
    def combine_quant(self) -> str:
        return self.recipe.combine_quant

    @property
    def stage1_fused(self) -> bool:
        return self.recipe.dispatch is DispatchKind.COMPACT

    @property
    def stage2_fused(self) -> bool:
        return self.recipe.combine is CombineKind.FUSED_SCATTER

    @property
    def dispatch_backend(self) -> str:
        """The dispatch stage1_fused picks, under the name callers used to pass."""
        return "flydsl" if self.stage1_fused else "mori"

    # -- wires -----------------------------------------------------------------
    @property
    def is_quant_dispatch_wire(self) -> bool:
        """The wire carries an MX payload plus its e8m0 row, not bf16."""
        return self.recipe.is_quant_dispatch_wire

    @property
    def dispatch_wire_spec(self) -> "_DispatchWire":
        return _DISPATCH_WIRE_SPECS[self.dispatch_wire]

    @property
    def dispatch_token_nbytes(self) -> int:
        return int(self.hidden_dim * self.dispatch_wire_spec.payload_bytes)

    @property
    def dispatch_wire_elem_count(self) -> int:
        """What mori's Cfg calls hidden_dim: ELEMENTS, at its own element size.

        fp8 and fp4 both transport as one byte per element, so an fp4 dispatch
        wire has to halve the count itself -- mori sizes the token as
        hidden_dim * elem_size
        and would otherwise move two bytes per packed byte.
        """
        return (
            self.dispatch_token_nbytes
            if self.is_quant_dispatch_wire
            else self.hidden_dim
        )

    @property
    def combine_quant_bits(self) -> int:
        """Payload width of the quantized wire this instance OFFERS; 0 is none.

        What a step actually runs on is forward()'s own choice; this only says
        which format it may name.
        """
        return _COMBINE_QUANT_BITS[self.combine_quant]

    @property
    def combine_quant_bit_modes(self) -> tuple[int, ...]:
        """Payload widths a step may pick, i.e. what to build a combine for --
        a reduce on the fused path, a mori plan set on the other.

        bf16 is always in, and is what a step gets unless it asks otherwise: a
        quantized wire only pays off once there are enough tokens on it to
        outweigh the per-token quant/dequant pair. Asking for none here keeps
        bf16 the only one, so a model that never quantizes builds one combine.
        """
        bits = self.combine_quant_bits
        return (0,) if not bits else (0, bits)

    def combine_wire_nbytes(self, quant_bits: int) -> int:
        """What combine moves per token at ``quant_bits``: MX, or bf16 at 0."""
        if quant_bits:
            # A payload plane of MX bytes, then its e8m0 scale plane.
            return (
                self.hidden_dim * quant_bits // 8
                + self.hidden_dim // _COMBINE_SCALE_BLOCK
            )
        return self.hidden_dim * 2

    @property
    def dispatch_scale_nbytes(self) -> int:
        """Per-token e8m0 row as WE produce it: one byte per 32 features, packed.

        Handed to the dispatch as-is. The rows arrive at the dispatch stage's own,
        128 B-aligned stride (its ``scale_stride``), which keeps a TDM run's start
        aligned; that padding is the transport's business, and the quant op's
        output goes straight onto the dispatch wire without a repack.
        """
        return self.hidden_dim // 32 if self.is_quant_dispatch_wire else 0

    def combine_slot_stride_for(self, quant_bits: int) -> int:
        """Slot pitch on the ``quant_bits`` wire: the payload rounded up to a
        power of two, so one row index addresses both peer and slot.

        Tight per wire, not shared across them. The scatter writes and the
        reduce reads one payload per slot at this pitch, so padding a quantized
        slot out to the bf16 one would leave three quarters of every cache line
        and TDM row untouched -- handing back most of what the narrower payload
        just won. The arena is what absorbs the difference instead: it is cut to
        the widest pitch (below) and a quantized wire packs into its front.
        """
        stride = 1
        while stride < self.combine_wire_nbytes(quant_bits):
            stride <<= 1
        return stride

    @property
    def combine_slot_stride_bytes(self) -> int:
        """The bf16 wire's pitch, and so the one the arena is SIZED for.

        bf16 is the widest wire, so a staging region cut to this pitch holds a
        step on any of them and the arena stops depending on which format a
        step picks -- which is what lets one instance serve them all.
        """
        return self.combine_slot_stride_for(0)


# The order the regions are laid down in: fixed, so a region's offset does not
# depend on which stage declares it.
_ARENA_REGION_ORDER = (
    "tok_off",
    "recv_num",
    "recv_to_src_token",
    "out_idx",
    "out_wts",
    "disp_out",
    "cross_device_barrier",
    "disp_out_scales",
    "ep_rowmap",
    "compact_hist",
    "compact_done",
    "comb_inp",
)
# cco rounds each allocation up to its VMM granularity -- 2 MiB on the parts
# MegaMoE runs on -- and, past 2 GiB, to 2 MiB again for the MR registration.
_CCO_VMM_GRANULARITY = 2 << 20
_CCO_MR_PAD_THRESHOLD = 2 << 30
_CCO_MR_PAD_ALIGN = 2 << 20
# The widest compact tile a grouped-GEMM CSV row names; bounds the compact
# rows when the tile cannot be looked up (inter_dim unknown).
_MAX_COMPACT_TILE_M = 256


@dataclass(frozen=True)
class _ArenaLayout:
    """The arena's named regions, their offsets, and the bytes the whole takes."""

    regions: tuple  # ((name, nbytes), ...), in layout order
    offsets: dict
    total_bytes: int

    @classmethod
    def of(cls, regions) -> "_ArenaLayout":
        """Lay ``regions`` down in the order given, each 256 B aligned."""
        offsets, offset = {}, 0
        for name, size in regions:
            offset = _align_up(offset, SymmetricArena._ALIGNMENT)
            offsets[name] = offset
            offset += size
        total = max(
            _align_up(offset, SymmetricArena._ALIGNMENT), SymmetricArena._ALIGNMENT
        )
        return cls(tuple((name, int(size)) for name, size in regions), offsets, total)

    @classmethod
    def merge(cls, *requests) -> "_ArenaLayout":
        """The regions every request asks for, each at the largest size asked,
        in _ARENA_REGION_ORDER."""
        need = {}
        for request in requests:
            for name, size in request.items():
                need[name] = max(need.get(name, 0), int(size))
        unknown = sorted(set(need) - set(_ARENA_REGION_ORDER))
        if unknown:
            raise ValueError(f"unknown arena region(s) {unknown}")
        return cls.of([(n, need[n]) for n in _ARENA_REGION_ORDER if need.get(n)])


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
        from .compact_plan import compact_hist_stride, compact_row_capacity

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


def _cco_alloc_bytes(nbytes: int, granularity: int) -> int:
    """What ccoMemAlloc takes out of per_rank_vmm for an ``nbytes`` allocation."""
    aligned = _align_up(nbytes, granularity)
    if nbytes > _CCO_MR_PAD_THRESHOLD:
        aligned = _align_up(aligned, _CCO_MR_PAD_ALIGN)
    return aligned


# -- the transport ---------------------------------------------------------------
#
# A dispatch stage and a combine stage, picked by the recipe, over one arena;
# ``DispatchStage`` and ``CombineStage`` are what each provides. A stage's class
# declares what can be checked before anything exists -- the wires and row
# layouts it handles, its geometry checks, the arena regions it needs -- so a
# recipe is checked and an arena sized before any communicator exists. An
# instance plans in ``__init__`` with nothing allocated, then allocates in
# ``build``. They meet only through the shared ``_EpResources``, the
# ``_DispatchBatch`` the dispatch hands the expert GEMM, and the
# ``_ComputeTarget`` the combine has the GEMM write into.


@dataclass
class Routing:
    token_count: int
    reverse_source_view: torch.Tensor

    @property
    def source_token_map(self) -> torch.Tensor:
        # Live view, not a copy: only the next dispatch rewrites this region,
        # and no peer reaches one until every rank clears _combine_sync, which
        # is stream-ordered after the gemm2 that reads it.
        return self.reverse_source_view


class _EpResources:
    """What the dispatch and the combine share: the arena, the per-step routing
    state both ends read, and the collective allocations, which go last."""

    def __init__(self, arena: SymmetricArena):
        self.arena = arena
        # mori's slot allocator word, when it lives outside the window.
        self.tokoff_ext = None

    def allocate(self, config: MegaMoEConfig, device) -> None:
        self.arena.zero()
        self._hidden_dim = config.hidden_dim
        # Shared with grouped_moe_gfx1250. TDM dispatch clears this at its tail,
        # before the following route kernel starts issuing slot atomics.
        from aiter.ops.flydsl.grouped_moe_gfx1250 import route_counter_buffer

        self.route_counter = route_counter_buffer(config.experts_per_rank, device)
        self.token_destination_map = torch.full(
            (config.max_tokens_per_rank * config.topk,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Send counts, then per destination mori's reservation base and push
        # landing offset.
        self.destination_peer_counter = torch.zeros(
            2 * config.world_size, dtype=torch.int32, device=device
        )
        self.dispatch_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=device)
        self.combine_output = torch.zeros(
            config.max_tokens_per_rank * config.hidden_dim,
            dtype=torch.int16,
            device=device,
        )

    def output(self, token_count: int) -> torch.Tensor:
        """The combine's result for this step's ``token_count`` tokens."""
        return (
            self.combine_output[: token_count * self._hidden_dim]
            .view(torch.bfloat16)
            .view(token_count, self._hidden_dim)
        )

    def close(self):
        tokoff_ext, self.tokoff_ext = self.tokoff_ext, None
        _close_all([self.arena] + ([] if tokoff_ext is None else [tokoff_ext]))


class BatchLayout(str, Enum):
    """How a dispatch lays its recv rows out for the expert GEMM."""

    # A row per received token; its topk ids and weights come alongside.
    TOKEN_MAJOR = "token_major"
    # Rows grouped per local expert and padded to the GEMM tile; an
    # ``ExpertGrouping`` says where each expert's rows are.
    EXPERT_GROUPED = "expert_grouped"


@dataclass(frozen=True)
class ExpertGrouping:
    """Where an expert-grouped batch keeps each expert's rows, as the grouped
    GEMM and its gemm2 scatter read them.

    The ints are this step's shape, not the arena's, and python ints, so a
    captured graph keeps a static grid.
    """

    masked_m: torch.Tensor  # per local expert, its rows
    psum: torch.Tensor  # per local expert, where its rows end
    ep_rowmap: torch.Tensor  # per row, where the gemm2 scatter sends it
    wire_row_stride: int  # bytes from a row to the next, its scales included
    recv_bound: int  # the recv-token bound, i.e. the CSV token bucket
    align_m: int  # per-expert row alignment the plan wrote
    rows: int  # row upper bound, bounding the GEMM grid

    def scatter_fields(self) -> dict:
        """As ``Stage2ScatterContext``'s compact_* fields."""
        return {
            "compact_layout": True,
            "compact_masked_m": self.masked_m,
            "compact_psum": self.psum,
            "compact_ep_rowmap": self.ep_rowmap,
            "compact_wire_row_stride": self.wire_row_stride,
            "compact_recv_bound": self.recv_bound,
            "compact_align_m": self.align_m,
            "compact_rows": self.rows,
        }


@dataclass
class _DispatchBatch:
    """One step's dispatched rows as the expert GEMM reads them, and the routing
    the combine sends the results home by."""

    x: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    scales: torch.Tensor | None  # the wire's e8m0 rows; None on bf16
    num_local_tokens: torch.Tensor | None
    routing: Routing
    layout: BatchLayout
    grouping: ExpertGrouping | None = None  # set on EXPERT_GROUPED rows only


@dataclass
class _ComputeTarget:
    """Where the expert GEMM leaves its rows: scattered into the peers' slots by
    its gemm2 epilogue, or written to ``output``.

    ``dispatch_context`` is what the grouped GEMM reads an expert-grouped batch's
    grouping from; None on token-major rows.
    """

    scatter: Stage2ScatterContext | None = None
    output: torch.Tensor | None = None
    dispatch_context: Stage2ScatterContext | None = None


class DispatchStage(Protocol):
    """Tokens out to their experts' ranks, into the arena's recv rows.

    Built as ``Stage(config, caps)``, where ``caps`` is mori's capabilities if
    either stage of the recipe drives mori and None otherwise.
    """

    wires: ClassVar[tuple]  # the dispatch wires it carries
    output_layout: ClassVar[BatchLayout]  # how it lays its recv rows out
    uses_mori: ClassVar[bool]
    # Whether it leaves mori's aggregated slot reservation for the combine.
    provides_aggregated_reservation: bool
    scale_stride: int  # the stride an e8m0 row arrives at; 0 on bf16
    recv_scales: torch.Tensor | None  # the arrived e8m0 rows; None on bf16

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        """Raise ValueError for a geometry it cannot run."""

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps) -> dict:
        """The arena regions it needs, by name. Pure: no instance, no GPU."""

    def build(self, res: "_EpResources") -> None:
        """Allocate and bind, once the arena exists."""

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
    ) -> _DispatchBatch: ...

    def prefetch(self, topk_ids, recv_token_bound=None) -> None: ...

    def warmup(self, recv_token_bound=None, token_count=None) -> None: ...

    def recv_views(self) -> tuple:
        """(recv_x, recv_weights, recv_ids) over the whole arena."""

    def close(self) -> None: ...


class CombineStage(Protocol):
    """Expert results back to their tokens' ranks, summed per token.

    Built as ``Stage(config, caps)``, as a dispatch stage is.
    """

    wires: ClassVar[tuple]  # the combine wires it can be built for
    accepted_layouts: ClassVar[frozenset]  # the BatchLayouts it can combine
    uses_mori: ClassVar[bool]
    # Whether it reads its landing offsets off mori's aggregated reservation.
    needs_aggregated_reservation: bool

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None: ...

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps) -> dict: ...

    def build(self, res: "_EpResources") -> None: ...

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        """Where the expert GEMM leaves this batch's rows."""

    def finish(
        self, routing: Routing, quant_bits: int, expert_out=None
    ) -> torch.Tensor:
        """Combine the rows ``prepare`` placed, on the ``quant_bits`` wire."""

    def close(self) -> None: ...


def _schedule_specs(schedule) -> list:
    """The launch specs a (bound, *spec) schedule names, one each."""
    return sorted({tuple(entry[1:]) for entry in schedule})


def _pick(schedule, token_count: int) -> tuple:
    """The spec of the first (bound, *spec) entry whose bound covers
    ``token_count``; past every bound, the last entry's."""
    for upper_bound, *spec in schedule:
        if upper_bound is None or token_count <= upper_bound:
            return tuple(spec)
    return tuple(schedule[-1][1:])


def _check_recv_token_bound(config: MegaMoEConfig, name: str, bound) -> None:
    """A caller's recv-token bound has to be a slot count the arena holds."""
    if bound is not None and not 0 < int(bound) <= config.max_recv:
        raise ValueError(f"{name} must be in (0, {config.max_recv}], got {bound}")


def _close_all(closeables) -> None:
    """close() each of ``closeables`` in order, the rest even if one raises."""
    with contextlib.ExitStack() as stack:
        for closeable in reversed(list(closeables)):
            stack.callback(closeable.close)


class _Dispatch:
    """What both dispatches share: the recv views over the routing regions, and
    the sender's rows quantized onto the wire."""

    def __init__(self, config: MegaMoEConfig, schedule: tuple):
        self.config = config
        self.schedule = schedule
        # A spec is (block, warp), plus route_parallel on the compact path.
        self.specs = _schedule_specs(schedule)
        self.res = None
        self._quant_payload = None
        self._quant_scales = None

    def build(self, res: _EpResources) -> None:
        self.res = res
        config, arena = self.config, res.arena
        device = res.total_recv.device
        # Quantized dispatch reuses fixed buffers across layers and graph
        # replays. The public quant wrapper allocates both tensors on every call;
        # keeping them here removes allocator/capture nodes from the stage-1
        # critical path without changing the quant kernel or wire format.
        if config.is_quant_dispatch_wire:
            payload_cols = (
                config.dispatch_token_nbytes
                // config.dispatch_wire_spec.quant_dtype.itemsize
            )
            self._quant_payload = torch.empty(
                (config.max_tokens_per_rank, payload_cols),
                dtype=config.dispatch_wire_spec.quant_dtype,
                device=device,
            )
            self._quant_scales = torch.empty(
                (config.max_tokens_per_rank, config.dispatch_scale_nbytes),
                dtype=torch.uint8,
                device=device,
            ).view(dtypes.fp8_e8m0)
        self.recv_weights = _from_gpu_ptr(
            arena.local_ptr("out_wts"), (config.max_recv, config.topk), torch.float32
        )
        self.recv_ids = _from_gpu_ptr(
            arena.local_ptr("out_idx"), (config.max_recv, config.topk), torch.int32
        )
        self._reverse_source = _from_gpu_ptr(
            arena.local_ptr("recv_to_src_token"), (config.max_recv,), torch.int32
        )

    def recv_views(self) -> tuple:
        """(recv_x, recv_weights, recv_ids) over the whole arena."""
        return self.recv_x, self.recv_weights, self.recv_ids

    def _quantize(self, hidden_states: torch.Tensor, token_count: int) -> tuple:
        """What goes on the wire, as (payload, address of its e8m0 rows):
        hidden_states and 0 on bf16, else its MX rows."""
        if self._quant_payload is None:
            return hidden_states, 0
        # Quantize ONCE PER LOCAL TOKEN here, instead of once per received
        # copy on the far side. Destination-independent, so the bytes are the
        # same either way; the preshuffle cannot move with it, because its
        # destination is the grouped row the receiver assigns.
        from aiter.ops.quant import dynamic_per_group_scaled_quant

        payload = self._quant_payload[:token_count]
        scale_rows = self._quant_scales[:token_count]
        dynamic_per_group_scaled_quant(
            payload,
            hidden_states,
            scale_rows,
            32,
            shuffle_scale=False,
        )
        # Straight onto the wire; the dispatch restrides these packed rows while
        # it stages them, so there is no repack here.
        return payload, scale_rows.data_ptr()

    def _routing(self, token_count: int) -> Routing:
        return Routing(
            token_count=token_count, reverse_source_view=self._reverse_source
        )


class _MoriDispatch(_Dispatch):
    """mori's EpDispatchPlan: token-major recv rows.

    The routing regions are laid out as on the compact path (out_idx/out_wts at
    slot*topk+k, recv_to_src_token as src_pe*max_tok+src_tok), and mori never
    touches cross_device_barrier. Recv slots are reserved one atomic per block
    and handed out block-local.
    """

    wires = _DISPATCH_WIRES
    output_layout = BatchLayout.TOKEN_MAJOR
    uses_mori = True

    def __init__(self, config: MegaMoEConfig, caps: _MoriCapabilities):
        # mori's geometry is a compile-time Cfg field, so it brings its own
        # tuned buckets.
        super().__init__(config, caps.dispatch_schedule(config))
        self.caps = caps
        # Where mori's slot word lives decides how the plans reserve slots:
        # outside the window, there is no aggregated reservation.
        self.tokoff_ext = caps.tokoff_ext_wanted(config)
        self.provides_aggregated_reservation = not self.tokoff_ext
        infos = self.describe(config, caps, self.specs)
        # One staging buffer serves every plan, as only one launches at a time.
        self._staging_nbytes = max(info["stagingBytes"] for info in infos)
        self.scale_stride = infos[0]["scaleStride"]
        self._staging = None
        self._plans = {}

    @staticmethod
    def request(config: MegaMoEConfig, spec: tuple, tokoff_ext=False) -> dict:
        """The plan request for ``spec``: what describe() reads, and what the
        plan is built from once the arena is added."""
        return {
            "world_size": config.world_size,
            # see dispatch_wire_elem_count; mori's plan_api: "the caller halves
            # hiddenDim"
            "hidden_dim": config.dispatch_wire_elem_count,
            "max_tok_per_rank": config.max_tokens_per_rank,
            "num_expert_per_rank": config.experts_per_rank,
            "num_expert_per_token": config.topk,
            "max_recv": config.max_recv,
            "dtype": config.dispatch_wire_spec.mori_dtype,
            "use_weights": True,
            "scale_bytes": config.dispatch_scale_nbytes,
            "tok_off_ext": int(tokoff_ext),
            "block_num": spec[0],
            "warp_per_block": spec[1],
        }

    @staticmethod
    def describe(config: MegaMoEConfig, caps, specs=None) -> list[dict]:
        """mori's info for every plan this dispatch builds.

        Without tokoff-ext: where the slot word lives sizes nothing, and asking
        would be collective.
        """
        if specs is None:
            specs = _schedule_specs(caps.dispatch_schedule(config))
        return caps.describe_dispatch(
            [_MoriDispatch.request(config, spec) for spec in specs]
        )

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        """Nothing beyond what mori's describe() refuses."""

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps) -> dict:
        return caps.region_sizes(cls.describe(config, caps))

    def build(self, res: _EpResources) -> None:
        super().build(res)
        config, arena = self.config, res.arena
        device = res.total_recv.device
        if self.tokoff_ext:
            res.tokoff_ext = self.caps.tokoff_ext(config, device)
        self._staging = torch.zeros(
            self._staging_nbytes, dtype=torch.uint8, device=device
        )
        for spec in self.specs:
            plan = self.caps.dispatch_plan(
                self.request(config, spec, self.tokoff_ext), arena
            )
            # The kernels dereference the window, so the plans have to outlive
            # them; held from here so a failed bind still closes it.
            self._plans[spec] = plan
            plan.bind(rank=config.rank)
            # The tensor, not its address: mori checks it against the plan's
            # stagingBytes.
            plan.bind(staging_base=self._staging)
        self._tok_off_peers = None if res.tokoff_ext is None else res.tokoff_ext.peers
        # Width in whatever recv_dtype counts: features for bf16/fp8, bytes for
        # fp4 -- see _DispatchWire.recv_dtype.
        width = (
            config.dispatch_token_nbytes
            // config.dispatch_wire_spec.recv_dtype.itemsize
        )
        self.recv_x = _from_gpu_ptr(
            arena.local_ptr("disp_out"),
            (config.max_recv, width),
            config.dispatch_wire_spec.recv_dtype,
        )
        self.recv_scales = (
            _from_gpu_ptr(
                arena.local_ptr("disp_out_scales"),
                (config.max_recv, self.scale_stride),
                torch.uint8,
            )
            if config.is_quant_dispatch_wire
            else None
        )

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
        """``recv_token_bound`` cuts the recv rows the GEMM is shown. There is no
        plan to run ahead, so the next layer's ids go unused."""
        config, res = self.config, self.res
        token_count = hidden_states.shape[0]
        _check_recv_token_bound(config, "recv_token_bound", recv_token_bound)
        _check_recv_token_bound(config, "next_recv_token_bound", next_recv_token_bound)
        plan = self._plans[_pick(self.schedule, token_count)]
        payload, scales_ptr = self._quantize(hidden_states, token_count)
        # No total_recv.zero_(): the dispatch overwrites it (see
        # _MoriCapabilities).
        plan.launch(
            stream=torch.cuda.current_stream().cuda_stream,
            token_indices=topk_ids.data_ptr(),
            inp_token_buf=payload.data_ptr(),
            weights_buf=topk_weights.data_ptr(),
            disp_dest_tok_id_map=res.token_destination_map.data_ptr(),
            dest_pe_token_counter=res.destination_peer_counter.data_ptr(),
            total_recv_token_num=res.total_recv.data_ptr(),
            grid_barrier=res.dispatch_barrier.data_ptr(),
            tok_off_peers=self._tok_off_peers,
            num_tokens=token_count,
            scales_buf=scales_ptr,
        )
        x, weights, ids, scales = (
            self.recv_x,
            self.recv_weights,
            self.recv_ids,
            self.recv_scales,
        )
        if recv_token_bound is not None:
            bound = int(recv_token_bound)
            x, weights, ids = x[:bound], weights[:bound], ids[:bound]
            if scales is not None:
                scales = scales[:bound]
        return _DispatchBatch(
            x=x,
            topk_weights=weights,
            topk_ids=ids,
            scales=scales,
            num_local_tokens=res.total_recv,
            routing=self._routing(token_count),
            layout=BatchLayout.TOKEN_MAJOR,
        )

    def prefetch(self, topk_ids, recv_token_bound=None) -> None:
        """Nothing to run ahead: the plan is the dispatch."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        """Nothing to compile ahead: every plan was built with the arena."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)

    def close(self) -> None:
        plans, self._plans = self._plans, {}
        _close_all(plans.values())
        # After the plans, which hold its address.
        self._staging = None


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
        self._plan_launches = {}

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        """Nothing beyond MegaMoEConfig's own checks."""

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps=None) -> dict:
        return cls.regions_for_tile(config, _compact_gemm_align_m(config))

    @staticmethod
    def regions_for_tile(config: MegaMoEConfig, tile_m: int) -> dict:
        """The arena regions at GEMM tile ``tile_m``, which pads every expert's
        rows and so sizes the recv rows and their map."""
        from .compact_plan import compact_done_nbytes

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
        from .compact_plan import PLAN_BLOCKS

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
            self._step = self._step_for(self._recv_bound(token_count, recv_token_bound))
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
            ahead = self._step_for(self._recv_bound(count, next_recv_token_bound))
            self._launch_plan(next_topk_ids, count, nxt, ahead.align_m)
            self._slot = nxt
        return batch

    def prefetch(self, topk_ids, recv_token_bound=None) -> None:
        """Launch the plan for ``topk_ids`` on the side stream, into this step."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)
        token_count = int(topk_ids.shape[0])
        self._step = self._step_for(self._recv_bound(token_count, recv_token_bound))
        self._launch_plan(topk_ids, token_count, self._slot, self._step.align_m)

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        """Compile the plan this bound's tile needs, in both slots."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)
        tok = (
            int(self.config.max_tokens_per_rank)
            if token_count is None
            else int(token_count)
        )
        self._step = self._step_for(self._recv_bound(tok, recv_token_bound))
        for slot in range(self._PLAN_SLOTS):
            self._plan_for(self._step.align_m, slot)

    def _recv_bound(self, token_count: int, recv_token_bound: int | None) -> int:
        """Static safe recv bound for this step: the caller's, already checked.

        Without one: dispatch deduplicates a token's routes to the same peer, so
        one rank can receive at most ``world_size * token_count`` rows.
        """
        if recv_token_bound is not None:
            return int(recv_token_bound)
        inferred = int(self.config.world_size) * max(1, int(token_count))
        return min(inferred, int(self.config.max_recv))

    def _step_for(self, recv_bound: int) -> _CompactStep:
        from .compact_plan import compact_row_capacity

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
        from .compact_plan import compile_tdm_compact_plan

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
            self._plan_launches[key] = launch
        return launch

    def _launch_plan(
        self, topk_ids: torch.Tensor, token_count: int, slot: int, align_m: int
    ) -> None:
        plan_stream = self._plan_stream
        plan_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(plan_stream):
            _run_compiled(
                self._plan_for(align_m, slot),
                self.res.arena.handle,
                topk_ids.data_ptr(),
                self._tok_maps[slot].data_ptr(),
                self._block_hists[slot].data_ptr(),
                self._send_bases[slot].data_ptr(),
                self._masked_ms[slot].data_ptr(),
                self._psums[slot].data_ptr(),
                self._barriers[slot].data_ptr(),
                self.config.rank,
                token_count,
                fx.Stream(plan_stream),
            )
        self._plan_event.record(plan_stream)
        self._plan_pending = True

    def _wait_plan(self) -> None:
        if not self._plan_pending:
            return
        torch.cuda.current_stream().wait_event(self._plan_event)
        self._plan_pending = False

    def close(self) -> None:
        """Nothing to release: its kernels and buffers are not held by mori."""


class _FusedScatterCombine:
    """The gemm2 epilogue P2P-scatters each row into the slot of the peer its
    token came from, packed as the step's wire; after a cross-device barrier a
    FlyDSL reduce sums each token's slots."""

    wires = tuple(_COMBINE_WIRE_QUANT)
    accepted_layouts = frozenset(BatchLayout)
    uses_mori = False
    needs_aggregated_reservation = False

    def __init__(self, config: MegaMoEConfig, caps=None):
        self.config = config
        self.res = None
        self._tiles = {
            bits: _combine_tile(
                topk=config.topk, hidden_dim=config.hidden_dim, quant_bits=bits
            )
            for bits in config.combine_quant_bit_modes
        }

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        # Every wire, not just the quantized ones: the reduce tiles hidden in
        # whole chunks and has no tail path.
        if config.hidden_dim % _COMBINE_CHUNK_ELEMS:
            raise ValueError(
                f"the combine reduce tiles hidden in {_COMBINE_CHUNK_ELEMS}-"
                f"element chunks with no tail path, so it needs hidden_dim % "
                f"{_COMBINE_CHUNK_ELEMS} == 0, got {config.hidden_dim}"
            )

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps=None) -> dict:
        return {
            "cross_device_barrier": config.world_size * 8,
            # Cut to the bf16 pitch whatever wire runs: it is the widest, so this
            # holds a step on any of them and a quantized one just packs into the
            # front of it at its own pitch.
            "comb_inp": (
                config.max_tokens_per_rank
                * config.topk
                * config.combine_slot_stride_bytes
            ),
        }

    def build(self, res: _EpResources) -> None:
        self.res = res
        config = self.config
        self._cross_device_flag = torch.ones(
            1, dtype=torch.int64, device=res.total_recv.device
        )
        # Keep the cross-device barrier in its own 1-block kernel so the reduce
        # grid is unconstrained. Both wires stage through LDS, so the block is
        # sized to the lane tile (T*C*256/16 lanes) -- which differs per wire,
        # hence one build per width rather than one shared spec.
        self._reduces = {}
        for bits, (toks, chunks) in self._tiles.items():
            lanes = toks * chunks * _COMBINE_CHUNK_ELEMS // 16
            self._reduces[bits] = _make_combine_fused_reduce(
                experts_per_token=config.topk,
                hidden_dim=config.hidden_dim,
                block_num=512,
                warp_num_per_block=lanes // _WAVE_SIZE,
                slot_stride_nbytes=config.combine_slot_stride_for(bits),
                quant_bits=bits,
                tokens_per_block=toks,
                chunks_per_iter=chunks,
            )
        self._sync = _make_combine_fused_sync(
            rank=config.rank,
            npes=config.world_size,
            off_xdb_mem=res.arena.offset("cross_device_barrier"),
        )

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        config, arena = self.config, self.res.arena
        grouped = batch.layout is BatchLayout.EXPERT_GROUPED
        scatter = Stage2ScatterContext(
            arena_handle=arena.handle,
            combine_input_offset=arena.offset("comb_inp"),
            slot_stride_bytes=config.combine_slot_stride_for(quant_bits),
            max_tokens_per_rank=config.max_tokens_per_rank,
            world_size=config.world_size,
            source_token_map=batch.routing.source_token_map,
            **(batch.grouping.scatter_fields() if grouped else {}),
            combine_quant_bits=quant_bits,
        )
        return _ComputeTarget(
            scatter=scatter, dispatch_context=scatter if grouped else None
        )

    def finish(
        self, routing: Routing, quant_bits: int, expert_out=None
    ) -> torch.Tensor:
        """The gemm2 epilogue has already scattered the results into the peers'
        slots, packed as the ``quant_bits`` wire; reduce them."""
        res = self.res
        stream = fx.Stream(torch.cuda.current_stream())
        # 1-block cross-device barrier, then the barrier-free reduce on the
        # same stream; the kernel boundary gives the reduce its visibility.
        _run_compiled(
            self._sync,
            res.arena.handle,
            self._cross_device_flag.data_ptr(),
            self.config.rank,
            stream,
        )
        _run_compiled(
            self._reduces[quant_bits],
            res.arena.local_ptr("comb_inp"),
            res.combine_output.data_ptr(),
            routing.token_count,
            stream,
        )
        return res.output(routing.token_count)

    def close(self) -> None:
        """Nothing to release: its kernels and buffers are not held by mori."""


class _MoriCombine:
    """mori's EpCombinePlan over fused_moe's per-recv-row output.

    bf16 gathers: fused_moe writes its rows straight into this rank's comb_inp
    staging rows, the combine meets every peer on cross_device_barrier, then
    pulls each token's rows back from the peers the dispatch's tok_map names and
    sums them. MXFP4 pushes instead: each rank quantizes its rows into the
    landing rows of the peer each came from, and after the barrier dequantizes
    and sums what landed for its own tokens. Both zero total_recv on the way out.
    """

    wires = ("bf16", "fp4")
    accepted_layouts = frozenset({BatchLayout.TOKEN_MAJOR})
    uses_mori = True

    def __init__(self, config: MegaMoEConfig, caps: _MoriCapabilities):
        self.config = config
        self.caps = caps
        self.res = None
        self._schedules = {
            bits: caps.combine_schedule(config, bits)
            for bits in config.combine_quant_bit_modes
        }
        self._specs = {
            bits: _schedule_specs(schedule)
            for bits, schedule in self._schedules.items()
        }
        self._max_blocks = max(
            block for wire in self._specs.values() for block, _ in wire
        )
        slots = caps.xdb_flag_slots
        if self._max_blocks > slots:
            raise ValueError(
                f"mori combine block_num {self._max_blocks} exceeds the "
                f"{slots} per-block epoch slots its entry barrier owns"
            )
        # The fp4 push combine reads its landing offsets off the reservation.
        self.needs_aggregated_reservation = bool(config.combine_quant_bits)
        self._plans = {}

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        if config.combine_quant_bits and (
            config.hidden_dim % _MORI_FP4_COMBINE_HIDDEN_ALIGN
        ):
            raise ValueError(
                "mori's fp4 combine needs hidden_dim % "
                f"{_MORI_FP4_COMBINE_HIDDEN_ALIGN} == 0, got {config.hidden_dim}"
            )

    @staticmethod
    def request(config: MegaMoEConfig, quant_bits: int, spec: tuple) -> dict:
        """The plan request for ``spec`` on the ``quant_bits`` wire."""
        return {
            "world_size": config.world_size,
            "hidden_dim": config.hidden_dim,
            "max_tok_per_rank": config.max_tokens_per_rank,
            "num_expert_per_rank": config.experts_per_rank,
            "num_expert_per_token": config.topk,
            "max_recv": config.max_recv,
            "dtype": torch.bfloat16,
            "use_weights": True,
            "combine_fp4": bool(quant_bits),
            "block_num": spec[0],
            "warp_per_block": spec[1],
        }

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps) -> dict:
        requests = [
            cls.request(config, bits, spec)
            for bits in config.combine_quant_bit_modes
            for spec in _schedule_specs(caps.combine_schedule(config, bits))
        ]
        return caps.region_sizes(caps.describe_combine(requests))

    def build(self, res: _EpResources) -> None:
        self.res = res
        config = self.config
        device = res.total_recv.device
        for bits, wire_specs in self._specs.items():
            for spec in wire_specs:
                plan = self.caps.combine_plan(
                    self.request(config, bits, spec), res.arena
                )
                self._plans[bits, spec] = plan
                plan.bind(rank=config.rank)

        # One epoch per block: each block is the only writer of its own slot.
        # Starts at 1 so the zeroed peer slots cannot alias the first launch.
        self._cross_device_flag = torch.ones(
            self.caps.xdb_flag_slots, dtype=torch.int64, device=device
        )
        self._grid_barrier = torch.zeros(1, dtype=torch.uint32, device=device)
        # The entry barrier's intra-grid fan-out: 16 lines per block.
        self._barrier_fan = torch.zeros(
            self._max_blocks * 16, dtype=torch.int32, device=device
        )
        self._static_args = {
            "out_token_buf": res.combine_output.data_ptr(),
            "out_weights_buf": None,
            "disp_dest_tok_id_map": res.token_destination_map.data_ptr(),
            # The push combine's landing offsets, which the dispatch wrote.
            "dest_pe_token_counter": res.destination_peer_counter.data_ptr(),
            "total_recv_token_num": res.total_recv.data_ptr(),
            "grid_barrier": self._grid_barrier.data_ptr(),
            "xdb_flag": self._cross_device_flag.data_ptr(),
            "combine_barrier_fan": self._barrier_fan.data_ptr(),
        }
        # The bf16 staging rows at the front of comb_inp. fused_moe writes its
        # rows straight into them, so the bf16 combine finds its input already
        # staged and skips its copy, and the fp4 send, which reads ahead as far
        # as row max_recv-1, stays inside the arena.
        self._stage = _from_gpu_ptr(
            res.arena.local_ptr("comb_inp"),
            (config.max_recv, config.hidden_dim),
            torch.bfloat16,
        )

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        """fused_moe writes its rows straight into this rank's comb_inp staging.

        Overwriting the previous step's rows there is safe because of the
        dispatch, not the combine: peers read those rows during their combine,
        and this step's dispatch, which the GEMM waits for, does not finish on
        this rank until every peer has dispatched to it -- which each peer does
        only after its own previous combine, in stream order. A dispatch that
        let a rank run ahead of a peer's combine would break this.
        """
        return _ComputeTarget(output=self._stage[: batch.x.shape[0]])

    def finish(
        self, routing: Routing, quant_bits: int, expert_out: torch.Tensor
    ) -> torch.Tensor:
        """``expert_out`` is fused_moe's per-recv-slot output, already in the
        staging rows; mori moves it from there."""
        count = routing.token_count
        plan = self._plans[quant_bits, _pick(self._schedules[quant_bits], count)]
        plan.launch(
            stream=torch.cuda.current_stream().cuda_stream,
            inp_token_buf=expert_out,
            num_tokens=count,
            **self._static_args,
        )
        return self.res.output(count)

    def close(self) -> None:
        plans, self._plans = self._plans, {}
        _close_all(plans.values())


_DISPATCH_STAGES: dict = {
    DispatchKind.MORI: _MoriDispatch,
    DispatchKind.COMPACT: _CompactDispatch,
}
_COMBINE_STAGES: dict = {
    CombineKind.MORI: _MoriCombine,
    CombineKind.FUSED_SCATTER: _FusedScatterCombine,
}


def _capabilities_for(recipe: MegaMoERecipe) -> _MoriCapabilities | None:
    """mori's capabilities if a stage ``recipe`` names drives mori, else None."""
    if any(stage.uses_mori for stage in _stage_classes(recipe)):
        return _mori_capabilities()
    return None


def _arena_layout(config: MegaMoEConfig, caps) -> _ArenaLayout:
    """The one arena formula: what the runtime allocates and what
    ``required_vmm_bytes`` reports."""
    dispatch, combine = _stage_classes(config.recipe)
    return _ArenaLayout.merge(
        dispatch.regions(config, caps), combine.regions(config, caps)
    )


class _EpDispatcher:
    """The recipe's dispatch and combine over one shared arena.

    Everything that can fail, and the host collectives, run before the arena --
    itself collective -- is allocated; a failure after it frees what was built.
    """

    def __init__(self, config: MegaMoEConfig, communicator):
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        recipe = config.recipe
        dispatch_cls, combine_cls = _stage_classes(recipe)
        caps = _capabilities_for(recipe)
        self._dispatch: DispatchStage = dispatch_cls(config, caps)
        self._combine: CombineStage = combine_cls(config, caps)
        # mori's op layer refuses this pair the same way.
        if (
            self._combine.needs_aggregated_reservation
            and not self._dispatch.provides_aggregated_reservation
        ):
            raise ValueError(
                f"combine={recipe.combine.value} on the {recipe.combine_wires[-1]} "
                "wire reads its landing offsets off mori's aggregated dispatch "
                "reservation, which this dispatch does not make "
                "(MORI_EP_TOKOFF_EXT=1 turns it off)"
            )
        self.layout = _arena_layout(config, caps)
        self.res = _EpResources(SymmetricArena(communicator, self.layout))
        try:
            self.res.allocate(config, self.device)
            self._dispatch.build(self.res)
            self._combine.build(self.res)
        except BaseException:
            self.close()
            raise

    @property
    def arena(self) -> SymmetricArena:
        return self.res.arena

    @property
    def total_recv(self) -> torch.Tensor:
        return self.res.total_recv

    @property
    def recv_scales(self) -> torch.Tensor | None:
        """The forwarded e8m0 rows over the whole arena; None on bf16."""
        return self._dispatch.recv_scales

    def recv_views(self) -> tuple:
        """(recv_x, recv_weights, recv_ids) over the whole arena."""
        return self._dispatch.recv_views()

    def dispatch(self, hidden_states, topk_weights, topk_ids, **step) -> _DispatchBatch:
        """``step`` is recv_token_bound, next_topk_ids, next_recv_token_bound
        and begin_step, as the dispatch stage takes them."""
        return self._dispatch.dispatch(hidden_states, topk_weights, topk_ids, **step)

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        return self._combine.prepare(batch, quant_bits)

    def combine(self, routing: Routing, quant_bits: int, expert_out=None):
        return self._combine.finish(routing, quant_bits, expert_out)

    def prefetch(self, topk_ids, recv_token_bound=None) -> None:
        self._dispatch.prefetch(topk_ids, recv_token_bound)

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        self._dispatch.warmup(recv_token_bound, token_count)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            # A launch still in flight (a graph replay, say) reads the window:
            # let it finish before anything it reads is freed.
            torch.cuda.synchronize(self.device)
        finally:
            # The plans hold the window and their kernels dereference it, so
            # they go before the arena does.
            _close_all((self._combine, self._dispatch, self.res))


def _require_gfx1250():
    gfx = get_gfx()
    if gfx != "gfx1250":
        raise RuntimeError(f"MegaMoEGfx1250 requires gfx1250, got {gfx}")


def _alias(owner: str, name: str) -> property:
    """A read-only attribute that is ``self.<owner>.<name>``."""
    return property(lambda self: getattr(getattr(self, owner), name))


class MegaMoEGfx1250:
    """A8W4 EP MoE: dispatch, the grouped expert GEMM, combine.

    Built by ``from_recipe`` from a ``MegaMoERecipe`` (what runs) and a
    ``MegaMoEGeometry`` (the shape it runs at); forward() takes the rest. A
    model's MoE layers share their geometry, recipe and communication arena and
    differ only in their weights, so one instance serves every layer (weights
    are forward() arguments) and the model keeps a single cco symmetric arena
    instead of one per layer.

    The keyword constructor, which takes the older stage1_fused / stage2_fused
    flags, is deprecated. ``_config``, ``_arena``, ``_dispatch()``,
    ``_combine()`` and ``_recv_dispatch_scales()`` are kept for ATOM's Triton
    experts, which drive the dispatch and the combine themselves.
    """

    model_dim = _alias("geometry", "model_dim")
    inter_dim = _alias("geometry", "inter_dim")
    experts = _alias("geometry", "experts")
    experts_per_rank = _alias("geometry", "experts_per_rank")
    topk = _alias("geometry", "topk")
    max_tokens_per_rank = _alias("geometry", "max_tokens_per_rank")
    activation = _alias("compute", "activation")
    gate_mode = _alias("compute", "gate_mode")
    quant_type = _alias("compute", "quant_type")
    hidden_pad = _alias("compute", "hidden_pad")
    intermediate_pad = _alias("compute", "intermediate_pad")
    swiglu_limit = _alias("compute", "swiglu_limit")

    def __init__(
        self,
        *,
        communicator,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        max_tokens_per_rank: int,
        activation: ActivationType = ActivationType.Silu,
        gate_mode: int = GateMode.INTERLEAVE.value,
        quant_type: QuantType = QuantType.per_1x32,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        swiglu_limit: float = 0.0,
        situ_beta: torch.Tensor | None = None,
        situ_linear_beta: torch.Tensor | None = None,
        dispatch_wire: str | None = None,
        stage1_fused: bool = False,
        stage2_fused: bool = True,
        combine_quant: str | None = None,
        dispatch_backend: str | None = None,
    ):
        """Deprecated: build a ``MegaMoERecipe`` and a ``MegaMoEGeometry`` and use
        ``from_recipe``. This turns the flags into a recipe the same way
        (``MegaMoERecipe.from_legacy``)."""
        warnings.warn(
            "MegaMoEGfx1250(**kwargs) is deprecated; use MegaMoEGfx1250.from_recipe "
            "with a MegaMoERecipe and a MegaMoEGeometry",
            DeprecationWarning,
            stacklevel=2,
        )
        _require_gfx1250()
        geometry = MegaMoEGeometry(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        recipe = MegaMoERecipe.from_legacy(
            stage1_fused=stage1_fused,
            stage2_fused=stage2_fused,
            dispatch_backend=dispatch_backend,
            dispatch_wire=dispatch_wire,
            combine_quant=combine_quant,
            activation=activation,
            gate_mode=gate_mode,
            quant_type=quant_type,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            swiglu_limit=swiglu_limit,
        )
        self._build(recipe, geometry, communicator, situ_beta, situ_linear_beta)

    @classmethod
    def from_recipe(
        cls,
        recipe: MegaMoERecipe,
        geometry: MegaMoEGeometry,
        *,
        communicator,
        situ_beta: torch.Tensor | None = None,
        situ_linear_beta: torch.Tensor | None = None,
    ) -> "MegaMoEGfx1250":
        """Build the MegaMoE ``recipe`` describes at ``geometry``.

        ``situ_beta`` / ``situ_linear_beta`` are the Situv2 activation's tensors,
        model-wide and needed only for that activation.
        """
        _require_gfx1250()
        self = cls.__new__(cls)
        self._build(recipe, geometry, communicator, situ_beta, situ_linear_beta)
        return self

    @classmethod
    def required_vmm_bytes(
        cls,
        recipe: MegaMoERecipe | None = None,
        geometry: MegaMoEGeometry | None = None,
        *,
        granularity: int = _CCO_VMM_GRANULARITY,
        **legacy,
    ) -> int:
        """The per-rank cco VMM (``per_rank_vmm``) this MegaMoE takes.

        Its arena -- the one cco allocation it makes -- rounded up the way
        ccoMemAlloc rounds it. The constructor allocates this same layout, so the
        allocation never exceeds it. It needs no communicator, so it can size the
        one about to be created; but it is MegaMoE's share only, and a
        communicator that also carries other cco allocations -- mori's own EP
        ops, or a DevComm's resource window (one granule each) -- needs theirs
        added on top. ``granularity`` is the communicator's VMM granularity, when
        known.

        The keyword form ``required_vmm_bytes(world_size=, hidden_dim=,
        max_tokens_per_rank=, experts=, topk=, stage1_fused=)`` names neither the
        wires nor the combine nor inter_dim, so it returns the largest any recipe
        with that dispatch could take.
        """
        if recipe is None:
            return cls._legacy_vmm_bound(granularity, **legacy)
        config = MegaMoEConfig(recipe=recipe, geometry=geometry)
        layout = _arena_layout(config, _capabilities_for(recipe))
        return _cco_alloc_bytes(layout.total_bytes, granularity)

    @classmethod
    def _legacy_vmm_bound(
        cls,
        granularity,
        *,
        world_size,
        hidden_dim,
        max_tokens_per_rank,
        experts,
        topk,
        stage1_fused=False,
    ):
        """The largest arena any recipe with this dispatch takes. inter_dim is
        not named, so the compact rows are sized for the widest GEMM tile a
        CSV row names."""
        geometry = MegaMoEGeometry(
            rank=0,
            world_size=world_size,
            model_dim=hidden_dim,
            inter_dim=1,
            experts=experts,
            topk=topk,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        if stage1_fused:
            candidates = [
                MegaMoERecipe(dispatch=DispatchKind.COMPACT, dispatch_wire=wire)
                for wire in ("fp8", "fp4")
            ]
        else:
            candidates = [
                MegaMoERecipe(dispatch_wire=wire, combine=combine, combine_wires=wires)
                for wire in _DISPATCH_WIRES
                for combine, wires in (
                    (CombineKind.FUSED_SCATTER, ("bf16",)),
                    (CombineKind.MORI, ("bf16", "fp4")),
                )
            ]
        caps = None if stage1_fused else _mori_capabilities()
        sizes, refused = [], []
        for recipe in candidates:
            try:
                config = MegaMoEConfig(recipe=recipe, geometry=geometry)
            except ValueError as error:
                refused.append(error)  # e.g. mori's fp4 combine at this hidden
                continue
            dispatch, combine = _stage_classes(recipe)
            dispatch_regions = (
                _CompactDispatch.regions_for_tile(config, _MAX_COMPACT_TILE_M)
                if stage1_fused
                else dispatch.regions(config, caps)
            )
            layout = _ArenaLayout.merge(dispatch_regions, combine.regions(config, caps))
            sizes.append(layout.total_bytes)
        if not sizes:
            raise ValueError(
                f"no recipe with stage1_fused={bool(stage1_fused)} runs at "
                f"world_size={world_size}, hidden_dim={hidden_dim}, "
                f"max_tokens_per_rank={max_tokens_per_rank}, experts={experts}, "
                f"topk={topk}: {refused[-1]}"
            )
        return _cco_alloc_bytes(max(sizes), granularity)

    def _build(self, recipe, geometry, communicator, situ_beta, situ_linear_beta):
        config = MegaMoEConfig(recipe=recipe, geometry=geometry)
        self.recipe = recipe
        self.geometry = geometry
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

        device = torch.device("cuda", torch.cuda.current_device())
        self.expert_mask = torch.zeros(
            geometry.experts, dtype=torch.int32, device=device
        )
        first_expert = geometry.rank * geometry.experts_per_rank
        self.expert_mask[first_expert : first_expert + geometry.experts_per_rank] = 1
        self._config = config
        self._ep = _EpDispatcher(config, communicator)

    @property
    def compute(self) -> ExpertComputeConfig:
        return self.recipe.compute

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        recv_token_bound: int | None = None,
        next_topk_ids: torch.Tensor | None = None,
        next_recv_token_bound: int | None = None,
        combine_quant: str | None = None,
        combine_wire: str | None = None,
        recv_counter_is_zero: bool = False,
    ) -> torch.Tensor:
        """Run one MoE layer: dispatch, its expert GEMM, then combine.

        ``recv_token_bound`` caps how many recv slots the GEMM is shown. Dispatch
        always fills a world_size*max_tokens_per_rank arena, so a caller that
        knows a tighter static bound (e.g. graph_bs*topk*world_size for a uniform
        decode batch) can shrink the GEMM's grid with it. It must be a python int
        so the shape stays static under graph capture, and must not cut below the
        received count -- the kernels skip the tail past the device-side count on
        their own.

        ``next_topk_ids`` (and optional ``next_recv_token_bound``) start the next
        layer's compact plan on a side stream after this dispatch, so it overlaps
        the expert GEMM. The two plans use independent hist/done slots and local
        tok_map/psum buffers. Omit them to keep the plan sequential.

        ``combine_wire`` picks this step's combine wire out of the recipe's
        ``combine_wires``, so a caller can quantize the steps with enough tokens
        on the wire to pay for it and leave the rest on bf16. Omitting it means
        bf16 -- building a format only makes it available, each step still has to
        ask for it. ``combine_quant`` is the same choice in the older spelling
        (none / mxfp8 / mxfp4).

        ``recv_counter_is_zero`` is accepted and ignored: the dispatch
        overwrites the received count, so there is no reset to skip.
        """
        if combine_wire is not None:
            if combine_quant is not None:
                raise ValueError("pass combine_wire or combine_quant, not both")
            if combine_wire not in _COMBINE_WIRE_QUANT:
                raise ValueError(
                    f"combine_wire must be one of {tuple(_COMBINE_WIRE_QUANT)}, "
                    f"got {combine_wire!r}"
                )
            combine_quant = _COMBINE_WIRE_QUANT[combine_wire]
        quant_bits = self._resolve_combine_quant(combine_quant)
        self._check_inputs(
            hidden_states, topk_weights, topk_ids, w1, w2, w1_scale, w2_scale, a1_scale
        )
        ep = self._ep
        batch = ep.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            recv_token_bound=recv_token_bound,
            next_topk_ids=next_topk_ids,
            next_recv_token_bound=next_recv_token_bound,
        )
        target = ep.prepare(batch, quant_bits)
        expert_out = self._compute(
            batch,
            target,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1,
            bias2=bias2,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )
        return ep.combine(batch.routing, quant_bits, expert_out)

    __call__ = forward

    def _check_inputs(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w1,
        w2,
        w1_scale,
        w2_scale,
        a1_scale,
    ) -> None:
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous bfloat16")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1":
            raise RuntimeError(
                "MegaMoE fused stage2 scatter requires the grouped A8W4 kernel; "
                "AITER_DISABLE_GROUPED_A8W4=1 is not supported"
            )
        if w1_scale is None or w2_scale is None:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires both w1_scale and w2_scale"
            )
        supported_weight_dtypes = (torch.uint8, dtypes.fp4x2)
        if (
            w1.dtype not in supported_weight_dtypes
            or w2.dtype not in supported_weight_dtypes
        ):
            raise ValueError(
                "MegaMoE fused stage2 scatter requires MXFP4 w1/w2 weights "
                f"(uint8 or fp4x2), got {w1.dtype} and {w2.dtype}"
            )
        expected_w1_shape = (
            self.experts_per_rank,
            2 * self.inter_dim,
            self.model_dim // 2,
        )
        expected_w2_shape = (
            self.experts_per_rank,
            self.model_dim,
            self.inter_dim // 2,
        )
        if tuple(w1.shape) != expected_w1_shape:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires interleaved G1U1 w1 with "
                f"shape {expected_w1_shape}, got {tuple(w1.shape)}"
            )
        if tuple(w2.shape) != expected_w2_shape:
            raise ValueError(
                f"MegaMoE fused stage2 scatter requires w2 shape "
                f"{expected_w2_shape}, got {tuple(w2.shape)}"
            )
        token_count = int(hidden_states.shape[0])
        if token_count > self.max_tokens_per_rank:
            raise ValueError(
                f"tokens={token_count} exceeds max_tokens_per_rank="
                f"{self.max_tokens_per_rank}"
            )
        expected_shape = (token_count, self.topk)
        if tuple(topk_weights.shape) != expected_shape:
            raise ValueError(
                f"topk_weights must have shape {expected_shape}, "
                f"got {tuple(topk_weights.shape)}"
            )
        if tuple(topk_ids.shape) != expected_shape:
            raise ValueError(
                f"topk_ids must have shape {expected_shape}, "
                f"got {tuple(topk_ids.shape)}"
            )
        if self._config.is_quant_dispatch_wire and a1_scale is not None:
            raise ValueError(
                "a1_scale is produced by the quantizing dispatch wire itself; a "
                "caller-supplied one would be silently discarded"
            )

    def _compute(
        self,
        batch: _DispatchBatch,
        target: _ComputeTarget,
        *,
        w1,
        w2,
        w1_scale,
        w2_scale,
        bias1,
        bias2,
        a1_scale,
        a2_scale,
    ) -> torch.Tensor:
        """The expert GEMM over one dispatched batch, leaving its rows where
        ``target`` says. Knows nothing of the transport beyond those two."""
        from aiter.ops.flydsl.grouped_moe_gfx1250 import (
            set_flydsl_dispatch_context,
        )

        compute = self.recipe.compute
        extra = {}
        if compute.activation == ActivationType.Situv2:
            extra["beta"] = self.situ_beta
            extra["linear_beta"] = self.situ_linear_beta
        # The grouped GEMM finds an expert-grouped batch's grouping here, not in
        # its arguments.
        set_flydsl_dispatch_context(target.dispatch_context)
        try:
            return fused_moe(
                batch.x,
                w1,
                w2,
                batch.topk_weights,
                batch.topk_ids,
                expert_mask=self.expert_mask,
                activation=compute.activation,
                gate_mode=compute.gate_mode,
                quant_type=compute.quant_type,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale if batch.scales is None else batch.scales,
                a2_scale=a2_scale,
                bias1=bias1,
                bias2=bias2,
                hidden_pad=compute.hidden_pad,
                intermediate_pad=compute.intermediate_pad,
                dtype=dtypes.bf16,
                num_local_tokens=batch.num_local_tokens,
                swiglu_limit=compute.swiglu_limit,
                stage2_scatter=target.scatter,
                output=target.output,
                **extra,
            )
        finally:
            set_flydsl_dispatch_context(None)

    def _resolve_combine_quant(self, combine_quant: str | None) -> int:
        """This step's combine payload width, checked against what was built.

        Unnamed means bf16. The constructor's ``combine_quant`` says which
        formats exist, not which one runs: a quantized wire only pays off once
        there are enough tokens on it to outweigh the per-token quant/dequant
        pair, and that is a property of the step, so a step has to ask.
        """
        if combine_quant is None:
            return 0
        if combine_quant not in _COMBINE_QUANT_MODES:
            raise ValueError(
                f"combine_quant must be one of {_COMBINE_QUANT_MODES}, "
                f"got {combine_quant!r}"
            )
        bits = _COMBINE_QUANT_BITS[combine_quant]
        built = self._config.combine_quant_bit_modes
        if bits not in built:
            raise ValueError(
                f"combine_quant={combine_quant!r} was not built: this MegaMoE "
                f"was constructed with combine_quant="
                f"{self._config.combine_quant!r}, which builds a combine for "
                f"{sorted(built)} bit widths only. Name the format at "
                "construction to make it available per step."
            )
        return bits

    def warmup_compact_plan(
        self,
        recv_token_bound: int | None = None,
        token_count: int | None = None,
    ) -> None:
        """Compile the compact plan this bound needs, before any graph capture.

        The plan is compiled per tile alignment and the alignment follows the
        bound's token bucket, so a captured batch whose bucket never ran eagerly
        would reach the JIT inside the capture -- which the plan refuses. A
        caller that captures a ladder of batch sizes warms every rung through
        here first. A no-op off the compact path.
        """
        self._ep.warmup(recv_token_bound, token_count)

    def prefetch_compact_plan(
        self,
        topk_ids: torch.Tensor,
        recv_token_bound: int | None = None,
    ) -> None:
        """Launch ``tdm_compact_plan`` on a side stream into the current slot.

        The plan only needs ``topk_ids``, so the caller can hide the first
        layer's plan behind RMSNorm. Later layers should pass ``next_topk_ids``
        to ``forward`` so the plan overlaps the previous expert GEMM instead.
        A no-op off the compact path.
        """
        self._ep.prefetch(topk_ids, recv_token_bound)

    # -- for callers that drive the transport themselves (ATOM's Triton experts)
    @property
    def _arena(self) -> SymmetricArena:
        return self._ep.arena

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        recv_counter_is_zero: bool = False,
    ):
        """Dispatch alone: (recv_x, recv_weights, recv_indices, total_recv,
        routing), each over the arena's full recv rows."""
        ep = self._ep
        batch = ep.dispatch(hidden_states, topk_weights, topk_ids, begin_step=False)
        return (*ep.recv_views(), ep.total_recv, batch.routing)

    def _recv_dispatch_scales(self) -> torch.Tensor | None:
        """The forwarded e8m0 rows, or None on the bf16 wire."""
        return self._ep.recv_scales

    def _combine(
        self,
        routing: Routing,
        combine_quant_bits: int = 0,
        expert_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine alone, on the ``combine_quant_bits`` wire (bf16 by default).
        On the fused path the results are already in the peers' slots; on
        mori's, ``expert_out`` is fused_moe's output in its staging rows."""
        return self._ep.combine(routing, combine_quant_bits, expert_out)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._ep.close()
