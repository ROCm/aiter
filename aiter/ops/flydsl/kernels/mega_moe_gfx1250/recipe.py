# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a MegaMoE runs and the shape it runs at.

``MegaMoERecipe`` (with its ``ExpertComputeConfig``) is the policy and a
``MegaMoEGeometry`` the shape; ``MegaMoEConfig`` is the two resolved and
checked together, which everything else sizes and builds from.
"""

import os
from dataclasses import dataclass, field
from enum import Enum

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .config import _WAVE_SIZE
from .types import COMBINE_SCALE_BLOCK as _COMBINE_SCALE_BLOCK

_MAX_WORLD_SIZE = 72
_MAX_EXPERTS_PER_RANK = 512


# "mxfp8" is what the mega_moe v2 path calls "fp8_blockwise_1x32" (kernels/
# flydsl_dispatch_combine_intranode_op.py); that path has no fp4.
_COMBINE_QUANT_MODES = ("none", "mxfp8", "mxfp4")
_COMBINE_QUANT_BITS = {"none": 0, "mxfp8": 8, "mxfp4": 4}


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
    """(dispatch stage per DispatchKind, combine stage per CombineKind).

    Imported when asked for: the stages import this module.
    """
    from .stages import _COMBINE_STAGES, _DISPATCH_STAGES

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
