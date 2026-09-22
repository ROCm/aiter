# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Stage2-fused MegaMoE host pipeline."""

import os
from dataclasses import dataclass

import flydsl.expr as fx
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

from .combine import _make_combine_fused_reduce, _make_combine_fused_sync
from .config import _WAVE_SIZE, _select_dispatch_config
from .dispatch import _make_dispatch
from .dispatch_tdm import _make_dispatch_tdm, tdm_max_warps, tdm_stage_capacity
from .types import Stage2ScatterContext, _from_gpu_ptr

__all__ = ["MegaMoEGfx1250"]

# "flydsl": per-lane vec4 payload copies. "tdm": the same arena protocol with
# the payload and the metadata moved by the gfx1250 Tensor Data Mover. "mori":
# mori's HIP/JIT kernel through its EpDispatchPlan.
_DISPATCH_BACKENDS = ("flydsl", "tdm", "mori")
_MAX_WORLD_SIZE = 72
_MAX_EXPERTS_PER_RANK = 512

# mori's C++ EpArgs offset stems -> this package's arena region names. All eight
# are bound when a plan is built even though mori's dispatch dereferences only the
# first six, so `outTok` is aimed at a region the dispatch never touches.
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
    """$MEGA_DISPATCH_WIRE, and a loud death for the name it replaced.

    Not a fallback: an env var that is silently ignored sends a run that asked
    for fp4 down the bf16 path and reports nothing, which is the one failure
    mode a wire benchmark cannot survive.
    """
    stale, current = os.environ.get("MEGA_WIRE"), os.environ.get("MEGA_DISPATCH_WIRE")
    if stale is not None and current != stale:
        raise RuntimeError(
            "MEGA_WIRE was renamed to MEGA_DISPATCH_WIRE (combine gets its own "
            f"wire); found MEGA_WIRE={stale!r} with MEGA_DISPATCH_WIRE="
            f"{current!r}. Update the launch script rather than relying on the "
            "old name -- it is no longer read."
        )
    return current or "bf16"


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


_DISPATCH_WIRE_SPECS = {
    "bf16": _DispatchWire(2, torch.bfloat16, None, torch.bfloat16),
    "fp8": _DispatchWire(1, dtypes.fp8, dtypes.fp8, dtypes.fp8),
    "fp4": _DispatchWire(0.5, dtypes.fp4x2, dtypes.fp4x2, torch.uint8),
}
_DISPATCH_WIRES = tuple(_DISPATCH_WIRE_SPECS)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _compact_gemm_align_m(
    config: "MegaMoEConfig",
    *,
    activation: ActivationType,
    quant_type: QuantType,
    inter_dim: int,
    recv_bound: int | None = None,
) -> int:
    """Tile alignment for compact dest rows; must match GEMM ``max(tile_m, tile_m2)``.

    Arena size is fixed at MegaMoE init. Honour ``AITER_TDM_TILE_M{,2}``, otherwise
    use the same grouped-GEMM CSV row fused_moe will pick for this recv-token
    bucket. Flooring at 64 used to pad compact_cap to EPR*64 even when the CSV
    tile is 16/32, which launched a 3–4× GEMM grid at small tpr.

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
        inter_dim=int(inter_dim),
        experts=config.experts_per_rank,
        topk=-1,
        activation=activation,
        dtype=dtypes.bf16,
        q_dtype_a=q_dtype_a,
        q_dtype_w=dtypes.fp4x2,
        quant_type=quant_type,
    )
    tile_m = 64
    tile_m2 = 64
    if row is not None:
        tile_m = _as_int(row.get("tile_m"), tile_m) or tile_m
        tile_m2 = _as_int(row.get("tile_m2"), tile_m) or tile_m
    return max(int(tile_m), int(tile_m2))


def _mori_dispatch_schedule(config) -> tuple:
    """mori's own tuned buckets, as this package's (bound, block, warp) triples.

    Not `_select_dispatch_config`'s: that table asks for 32 warps above 256
    tokens, and mori's gfx1250 dispatch stages a hidden-dim tile per warp in
    dynamic LDS -- 32 * 7168 * 2 = 458 KB, past the 320 KB budget. EpCfgIsValid
    does not check LDS, so the plan would build and then fail at launch.
    """
    from mori.ops.dispatch_combine_v2.hip_tuning_configs import lookup

    tuned = lookup(
        config.world_size,
        config.hidden_dim,
        config.topk,
        dtype="bf16",
        experts_per_rank=config.experts_per_rank,
    )
    schedule = tuned["schedule"]
    if schedule:
        return tuple((bound, block, warp) for bound, block, warp, _, _ in schedule)
    return ((None, tuned["dispatch_block_num"], tuned["warp_num_per_block"]),)


class SymmetricArena:
    _ALIGNMENT = 256

    def __init__(self, communicator, regions):
        self._communicator = communicator
        self._offsets = {}
        self._sizes = {}
        offset = 0
        for name, size in regions:
            offset = _align_up(offset, self._ALIGNMENT)
            self._offsets[name] = offset
            self._sizes[name] = size
            offset += size
        self._total_bytes = max(_align_up(offset, self._ALIGNMENT), self._ALIGNMENT)
        self._memory = communicator.alloc_mem(self._total_bytes)
        self._window = communicator.register_window(self._memory.ptr, self._total_bytes)

    @property
    def handle(self) -> int:
        return self._window.handle

    def offset(self, name: str) -> int:
        return self._offsets[name]

    def local_ptr(self, name: str) -> int:
        return self._window.local_ptr + self._offsets[name]

    def zero(self, name: str | None = None):
        if name is None:
            pointer, size = self._window.local_ptr, self._total_bytes
        else:
            pointer = self.local_ptr(name)
            size = self._sizes[name]
        _from_gpu_ptr(pointer, (size,), torch.int8).zero_()

    def close(self):
        self._window.close()
        self._memory.close()


@dataclass
class MegaMoEConfig:
    """Op-level config: geometry, plus the dispatch knobs.

    Nothing here tunes stage2 -- that is the gemm2 epilogue fused into combine
    (Stage2ScatterContext), which takes no parameter from this side.
    """

    rank: int
    world_size: int
    hidden_dim: int
    max_tokens_per_rank: int
    experts_per_rank: int
    topk: int
    # Dispatch (stage1) knobs. They live here because this package has no
    # stage1 config yet -- dispatch and gemm1 are not fused. When that fusion
    # lands they move together into whatever config it brings.
    dispatch_block_num: int | None = None
    dispatch_warp_num_per_block: int | None = None
    schedule: tuple | None = None
    dispatch_backend: str = "flydsl"
    # What dispatch puts on the wire. fp8 halves the payload and fp4 quarters it,
    # each sending a per-token e8m0 row along; the receiver then skips its own
    # quant. Combine is unaffected -- it moves post-expert tokens, which are bf16
    # whatever the wire carried.
    #
    # The wire must MATCH what the expert GEMM wants for its A operand (a8w4 ->
    # fp8, a4w4 -> fp4). It is not a free choice: the receiver hands the payload
    # to the grouped GEMM as-is, so a mismatch is a width error, not a slow path.
    dispatch_wire: str = "bf16"
    # None reads $AITER_TDM_COMPACT_PLAN; an explicit value wins over it, for a
    # caller that drives the recv rows with its own expert GEMM and cannot read
    # the compact layout.
    compact_plan_override: bool | None = None

    def __post_init__(self):
        if self.dispatch_wire not in _DISPATCH_WIRES:
            raise ValueError(
                f"dispatch_wire must be one of {_DISPATCH_WIRES}, "
                f"got {self.dispatch_wire!r}"
            )
        if self.is_quant_dispatch_wire and self.dispatch_backend not in ("tdm", "mori"):
            raise ValueError(
                f"dispatch_wire={self.dispatch_wire!r} requires "
                "dispatch_backend='tdm' or 'mori' "
                f"(got {self.dispatch_backend!r})"
            )
        if self.is_quant_dispatch_wire and self.hidden_dim % 32:
            raise ValueError(
                "one e8m0 scale covers 32 features, so a quantizing dispatch "
                f"wire needs hidden_dim % 32 == 0, got {self.hidden_dim}"
            )
        if self.dispatch_backend not in _DISPATCH_BACKENDS:
            raise ValueError(
                f"dispatch_backend must be one of {_DISPATCH_BACKENDS}, "
                f"got {self.dispatch_backend!r}"
            )
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
        tuned = _select_dispatch_config(
            self.world_size,
            self.hidden_dim,
            self.topk,
            tdm=self.dispatch_backend == "tdm",
        )
        if self.dispatch_block_num is None:
            self.dispatch_block_num = tuned["dispatch_block_num"]
        if self.dispatch_warp_num_per_block is None:
            self.dispatch_warp_num_per_block = tuned["dispatch_warp_num_per_block"]
        if self.schedule is None:
            self.schedule = tuned["schedule"]

    @property
    def max_recv(self) -> int:
        return self.world_size * self.max_tokens_per_rank

    @property
    def compact_plan(self) -> bool:
        """Send-side compact dest rows; TDM default, disable with env=0."""
        if self.dispatch_backend != "tdm" or not self.is_quant_dispatch_wire:
            return False
        if self.compact_plan_override is not None:
            return self.compact_plan_override
        return os.environ.get("AITER_TDM_COMPACT_PLAN", "1") in (
            "1",
            "true",
            "True",
        )

    def compact_row_cap(self, tile_m: int = 64) -> int:
        from .compact_plan import compact_row_capacity

        return compact_row_capacity(
            max_recv=self.max_recv,
            topk=self.topk,
            experts_per_rank=self.experts_per_rank,
            tile_m=tile_m,
        )

    @property
    def is_quant_dispatch_wire(self) -> bool:
        """The wire carries an MX payload plus its e8m0 row, not bf16."""
        return self.dispatch_wire in ("fp8", "fp4")

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
    def combine_token_nbytes(self) -> int:
        """Combine moves bf16 post-expert tokens, whatever the wire carried."""
        return self.hidden_dim * 2

    @property
    def dispatch_scale_nbytes(self) -> int:
        """Per-token e8m0 row as WE produce it: one byte per 32 features, packed.

        Handed to mori as-is. mori lays it down at its own, 128 B-aligned stride
        (dispatch_scale_dst_nbytes) because that is what keeps a TDM run's start aligned;
        that padding is mori's business, and the quant op's output can go straight
        onto the dispatch wire without a repack.
        """
        return self.hidden_dim // 32 if self.is_quant_dispatch_wire else 0

    @property
    def dispatch_scale_dst_nbytes(self) -> int:
        """The stride the rows ARRIVE at, which the receiving gather addresses by.

        Asked of mori rather than recomputed: it is the transport's layout
        decision, and a local copy of the rule would drift the first time the
        alignment changes.
        """
        if not self.is_quant_dispatch_wire:
            return 0
        if self.dispatch_backend == "tdm":
            return _align_up(self.dispatch_scale_nbytes, 128)
        try:
            from mori.ops.dispatch_combine_v2.hip_backend import scale_stride_bytes
        except ImportError as e:
            # Imported here, not at module scope: a bf16 wire needs none of this,
            # so an older mori keeps working until someone asks for fp8/fp4.
            raise RuntimeError(
                f"dispatch_wire={self.dispatch_wire!r} needs a mori whose EP "
                "dispatch carries a per-token scale row (ROCm/mori#593 or later); "
                "the installed one has no scale_stride_bytes"
            ) from e

        return scale_stride_bytes(self.dispatch_scale_nbytes)

    @property
    def combine_slot_stride_bytes(self) -> int:
        stride = 1
        while stride < self.combine_token_nbytes:
            stride <<= 1
        return stride


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


class MegaMoEGfx1250:
    """A8W4 EP MoE with GEMM2 P2P scatter fused into combine."""

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
        dispatch_backend: str | None = None,
        dispatch_wire: str | None = None,
        compact_plan: bool | None = None,
    ):
        """Everything here is fixed for the whole model; forward() takes the rest.

        A model's MoE layers share their geometry, expert-GEMM recipe and
        communication arena and differ only in their weights, so one instance
        serves every layer (weights are forward() arguments) and the model keeps a
        single cco symmetric arena instead of one per layer.
        """
        gfx = get_gfx()
        if gfx != "gfx1250":
            raise RuntimeError(f"MegaMoEGfx1250 requires gfx1250, got {gfx}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if experts <= 0:
            raise ValueError(f"experts must be positive, got {experts}")
        if experts % world_size:
            raise ValueError(
                f"experts={experts} must be divisible by world_size={world_size}"
            )
        if max_tokens_per_rank <= 0:
            raise ValueError(
                f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}"
            )
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if topk > experts:
            raise ValueError(f"topk={topk} cannot exceed experts={experts}")
        if model_dim <= 0 or inter_dim <= 0:
            raise ValueError(
                f"model_dim and inter_dim must be positive, got "
                f"{model_dim}, {inter_dim}"
            )
        if swiglu_limit < 0:
            raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}")
        if hidden_pad < 0 or intermediate_pad < 0:
            raise ValueError(
                f"padding must be non-negative, got {hidden_pad}, {intermediate_pad}"
            )
        activation = ActivationType(activation)
        gate_mode = GateMode(gate_mode)
        quant_type = QuantType(quant_type)
        if gate_mode != GateMode.INTERLEAVE:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires gate_mode=INTERLEAVE"
            )
        if quant_type != QuantType.per_1x32:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires quant_type=per_1x32"
            )
        if activation not in (
            ActivationType.Silu,
            ActivationType.Swiglu,
            ActivationType.Situv2,
        ):
            raise ValueError(
                f"MegaMoE fused stage2 scatter does not support activation={activation}"
            )

        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.experts_per_rank = self.experts // world_size
        self.topk = int(topk)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.activation = activation
        self.gate_mode = gate_mode
        self.quant_type = quant_type
        self.hidden_pad = int(hidden_pad)
        self.intermediate_pad = int(intermediate_pad)
        self.swiglu_limit = float(swiglu_limit)
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

        device = torch.device("cuda", torch.cuda.current_device())
        self.expert_mask = torch.zeros(self.experts, dtype=torch.int32, device=device)
        first_expert = rank * self.experts_per_rank
        self.expert_mask[first_expert : first_expert + self.experts_per_rank] = 1

        self._initialize_pipeline(
            MegaMoEConfig(
                rank=int(rank),
                world_size=int(world_size),
                hidden_dim=self.model_dim,
                max_tokens_per_rank=self.max_tokens_per_rank,
                experts_per_rank=self.experts_per_rank,
                topk=self.topk,
                dispatch_backend=(
                    dispatch_backend
                    if dispatch_backend is not None
                    else os.environ.get("MEGA_DISPATCH", "flydsl")
                ),
                dispatch_wire=(
                    dispatch_wire
                    if dispatch_wire is not None
                    else read_dispatch_wire_env()
                ),
                compact_plan_override=compact_plan,
            ),
            communicator,
        )

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
    ) -> torch.Tensor:
        """Run one MoE layer: dispatch, its expert GEMM, then the fused combine.

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
        """
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
        if self._compact_plan:
            # Compact cannot slice recv_x -- the rows are grouped per expert, not
            # token-major -- so the bound is spent on the geometry instead: it
            # picks the plan's alignment, the GEMM's tile and the GEMM's grid.
            self._begin_compact_step(recv_token_bound)
        recv_x, recv_weights, recv_ids, total_recv, routing = self._dispatch(
            hidden_states, topk_weights, topk_ids
        )
        if recv_token_bound is not None and not self._compact_plan:
            bound = int(recv_token_bound)
            if bound <= 0 or bound > recv_x.shape[0]:
                raise ValueError(
                    f"recv_token_bound must be in (0, {recv_x.shape[0]}], "
                    f"got {recv_token_bound}"
                )
            recv_x = recv_x[:bound]
            recv_weights = recv_weights[:bound]
            recv_ids = recv_ids[:bound]
        if self._config.is_quant_dispatch_wire:
            assert a1_scale is None, (
                "a1_scale is produced by the quantizing dispatch wire itself; a "
                "caller-supplied one would be silently discarded"
            )
            a1_scale = self._recv_dispatch_scales()
            if recv_token_bound is not None and not self._compact_plan:
                a1_scale = a1_scale[: int(recv_token_bound)]
        extra = {}
        if self.activation == ActivationType.Situv2:
            extra["beta"] = self.situ_beta
            extra["linear_beta"] = self.situ_linear_beta
        scatter = self._scatter_context(routing)
        from aiter.ops.flydsl.grouped_moe_gfx1250 import set_tdm_compact_plan

        set_tdm_compact_plan(scatter if self._compact_plan else None)
        moe_ids = self._compact_dummy_ids if self._compact_plan else recv_ids
        moe_wts = self._compact_dummy_wts if self._compact_plan else recv_weights
        if (
            self._compact_plan
            and next_topk_ids is not None
            and int(next_topk_ids.shape[0]) > 0
        ):
            # Dispatch of this layer is done, so tok_map[slot] is free. GEMM
            # still reads psum/masked_m of this slot; the next plan writes the
            # other slot and overlaps the expert GEMM (and later combine/RMS).
            self._prefetch_next_compact_plan(next_topk_ids, next_recv_token_bound)
        try:
            fused_moe(
                recv_x,
                w1,
                w2,
                moe_wts,
                moe_ids,
                expert_mask=self.expert_mask,
                activation=self.activation,
                gate_mode=self.gate_mode,
                quant_type=self.quant_type,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                bias1=bias1,
                bias2=bias2,
                hidden_pad=self.hidden_pad,
                intermediate_pad=self.intermediate_pad,
                dtype=dtypes.bf16,
                num_local_tokens=None if self._compact_plan else total_recv,
                swiglu_limit=self.swiglu_limit,
                stage2_scatter=scatter,
                **extra,
            )
        finally:
            set_tdm_compact_plan(None)
        return self._combine(routing)

    __call__ = forward

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _compact_plan_for(self, align_m: int, slot: int | None = None):
        """The compact-plan launch that aligns each expert's rows to ``align_m``.

        Compiled per alignment and hist/done slot, because the plan and the
        expert GEMM must agree on the tile, and double-buffered plans must not
        share a done counter.
        """
        from .compact_plan import compile_tdm_compact_plan

        align_m = int(align_m)
        slot = int(self._compact_slot if slot is None else slot)
        key = (align_m, slot)
        launch = self._compact_plan_launches.get(key)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"compact plan for align_m={align_m} slot={slot} was not "
                    "compiled before graph capture; warm this token bucket "
                    "eagerly first"
                )
            hist0 = self._arena.offset("compact_hist")
            done0 = self._arena.offset("compact_done")
            launch = compile_tdm_compact_plan(
                rank=self._config.rank,
                npes=self._config.world_size,
                experts_per_rank=self._config.experts_per_rank,
                topk=self._config.topk,
                tile_m=align_m,
                # The arena's row count, not this step's: the plan bounds what it
                # may write against the region it writes into.
                compact_cap=self._compact_cap,
                off_hist=hist0 + slot * self._compact_hist_stride * 4,
                off_done=done0 + slot * 4,
                hist_stride=self._compact_hist_stride,
                max_routes=self._compact_max_routes,
                hist_pingpong=False,
            )
            self._compact_plan_launches[key] = launch
        return launch

    def warmup_compact_plan(self, recv_token_bound: int | None = None) -> None:
        """Compile the compact plan this bound needs, before any graph capture.

        The plan is compiled per tile alignment and the alignment follows the
        bound's token bucket, so a captured batch whose bucket never ran eagerly
        would reach the JIT inside the capture -- which _compact_plan_for
        refuses. A caller that captures a ladder of batch sizes warms every rung
        through here first. A no-op off the compact path.
        """
        if not self._compact_plan:
            return
        self._begin_compact_step(recv_token_bound)
        for slot in range(self._COMPACT_PLAN_SLOTS):
            self._compact_plan_for(self._compact_step_align_m, slot)

    def prefetch_compact_plan(
        self,
        topk_ids: torch.Tensor,
        recv_token_bound: int | None = None,
    ) -> None:
        """Launch ``tdm_compact_plan`` on a side stream into the current slot.

        The plan only needs ``topk_ids``, so the caller can hide the first
        layer's plan behind RMSNorm. Later layers should pass ``next_topk_ids``
        to ``forward`` so the plan overlaps the previous expert GEMM instead.
        """
        if not self._compact_plan:
            return
        self._begin_compact_step(recv_token_bound)
        self._launch_compact_plan_async(
            topk_ids, int(topk_ids.shape[0]), self._compact_slot
        )

    def _prefetch_next_compact_plan(
        self,
        topk_ids: torch.Tensor,
        recv_token_bound: int | None,
    ) -> None:
        nxt = 1 - self._compact_slot
        saved = (
            self._compact_step_align_m,
            self._compact_step_rows,
            self._compact_step_recv_bound,
        )
        self._begin_compact_step(recv_token_bound)
        self._launch_compact_plan_async(topk_ids, int(topk_ids.shape[0]), nxt)
        (
            self._compact_step_align_m,
            self._compact_step_rows,
            self._compact_step_recv_bound,
        ) = saved
        self._compact_slot = nxt

    def _launch_compact_plan_async(
        self, topk_ids: torch.Tensor, token_count: int, slot: int | None = None
    ) -> None:
        slot = int(self._compact_slot if slot is None else slot)
        bufs = self._compact_slot_bufs(slot)
        cur = torch.cuda.current_stream()
        plan_stream = self._compact_plan_stream
        plan_stream.wait_stream(cur)
        with torch.cuda.stream(plan_stream):
            self._compact_plan_for(self._compact_step_align_m, slot)(
                self._arena.handle,
                topk_ids.data_ptr(),
                bufs["tok_map"].data_ptr(),
                bufs["block_hist"].data_ptr(),
                bufs["send_base"].data_ptr(),
                bufs["masked_m"].data_ptr(),
                bufs["psum"].data_ptr(),
                bufs["barrier"].data_ptr(),
                self._config.rank,
                token_count,
                fx.Stream(plan_stream),
            )
        self._compact_plan_event.record(plan_stream)
        self._compact_plan_pending = True

    def _wait_compact_plan(self) -> None:
        if not self._compact_plan_pending:
            return
        torch.cuda.current_stream().wait_event(self._compact_plan_event)
        self._compact_plan_pending = False

    def _compact_slot_bufs(self, slot: int) -> dict:
        return {
            "tok_map": self._tok_maps[slot],
            "block_hist": self._block_hists[slot],
            "send_base": self._send_bases[slot],
            "masked_m": self._masked_ms[slot],
            "psum": self._psums[slot],
            "barrier": self._barriers[slot],
        }

    def _begin_compact_step(self, recv_token_bound: int | None) -> None:
        """Fix this step's compact geometry from the caller's recv bound.

        Everything here is a python int, so a captured graph keeps a static grid
        and each captured batch size gets the geometry tuned for its own bucket.
        """
        from .compact_plan import compact_row_capacity

        config = self._config
        bound = config.max_recv if recv_token_bound is None else int(recv_token_bound)
        bound = max(1, min(bound, config.max_recv))
        align_m = _compact_gemm_align_m(
            config,
            activation=self.activation,
            quant_type=self.quant_type,
            inter_dim=self.inter_dim,
            recv_bound=bound,
        )
        # The arena holds rows for _compact_tile_m, and the CSV tiles are not
        # monotonic in the token bucket, so a smaller bound can still name a
        # wider tile. Clamping keeps every step inside the allocated rows.
        self._compact_step_align_m = min(int(align_m), self._compact_tile_m)
        self._compact_step_recv_bound = bound
        self._compact_step_rows = min(
            compact_row_capacity(
                max_recv=bound,
                topk=config.topk,
                experts_per_rank=config.experts_per_rank,
                tile_m=self._compact_step_align_m,
            ),
            self._compact_cap,
        )

    def _initialize_pipeline(self, config: MegaMoEConfig, communicator):
        self._config = config
        self._closed = False
        device = torch.device("cuda", torch.cuda.current_device())
        max_recv = config.max_recv
        self._compact_plan = config.compact_plan
        self._compact_tile_m = _compact_gemm_align_m(
            config,
            activation=self.activation,
            quant_type=self.quant_type,
            inter_dim=self.inter_dim,
        )
        self._compact_cap = (
            config.compact_row_cap(self._compact_tile_m)
            if self._compact_plan
            else max_recv
        )
        recv_rows = self._compact_cap if self._compact_plan else max_recv
        self._compact_wire_row = (
            _align_up(
                config.dispatch_token_nbytes + config.dispatch_scale_nbytes,
                128,
            )
            if self._compact_plan
            else config.dispatch_token_nbytes
        )
        from .compact_plan import (
            PLAN_BLOCKS,
            compact_done_nbytes,
            compact_hist_stride,
        )

        segs = config.world_size * config.experts_per_rank
        max_routes = config.max_tokens_per_rank * config.topk
        hist_stride = compact_hist_stride(
            npes=config.world_size,
            experts_per_rank=config.experts_per_rank,
            max_routes=max_routes,
        )
        self._compact_hist_stride = hist_stride
        self._compact_max_routes = max_routes
        arena_regions = [
            ("tok_off", 4),
            ("recv_num", config.world_size * 4),
            ("recv_to_src_token", max_recv * 4),
            ("out_idx", max_recv * config.topk * 4),
            ("out_wts", max_recv * config.topk * 4),
            ("disp_out", recv_rows * self._compact_wire_row),
            ("cross_device_barrier", config.world_size * 8),
        ]
        if config.dispatch_scale_dst_nbytes and not self._compact_plan:
            arena_regions.append(
                ("disp_out_scales", recv_rows * config.dispatch_scale_dst_nbytes)
            )
        if self._compact_plan:
            arena_regions.extend(
                [
                    ("ep_rowmap", (recv_rows + 1) * 8),
                    ("compact_hist", 2 * hist_stride * 4),
                    ("compact_done", compact_done_nbytes()),
                ]
            )
        arena_regions.append(
            (
                "comb_inp",
                config.max_tokens_per_rank
                * config.topk
                * config.combine_slot_stride_bytes,
            )
        )
        self._arena = SymmetricArena(communicator, arena_regions)
        self._compact_scale_row = (
            config.dispatch_scale_nbytes
            if self._compact_plan
            else config.dispatch_scale_dst_nbytes
        )
        self._arena.zero()
        # Shared with grouped_moe_gfx1250. TDM dispatch clears this at its tail,
        # before the following route kernel starts issuing slot atomics.
        from aiter.ops.flydsl.grouped_moe_gfx1250 import route_counter_buffer

        self._route_counter = route_counter_buffer(config.experts_per_rank, device)

        self._COMPACT_PLAN_SLOTS = 2
        self._compact_slot = 0
        self._token_destination_map = torch.full(
            (config.max_tokens_per_rank * config.topk,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._compact_masked_m = None
        self._compact_psum = None
        self._compact_plan_launches = {}
        self._compact_step_align_m = self._compact_tile_m
        self._compact_step_rows = self._compact_cap
        self._compact_step_recv_bound = max_recv
        if self._compact_plan:
            n_tok = config.max_tokens_per_rank * config.topk
            self._tok_maps = [
                torch.full((n_tok,), -1, dtype=torch.int32, device=device)
                for _ in range(self._COMPACT_PLAN_SLOTS)
            ]
            self._token_destination_map = self._tok_maps[0]
            self._masked_ms = [
                torch.zeros(config.experts_per_rank, dtype=torch.int32, device=device)
                for _ in range(self._COMPACT_PLAN_SLOTS)
            ]
            self._psums = [
                torch.zeros(config.experts_per_rank, dtype=torch.int32, device=device)
                for _ in range(self._COMPACT_PLAN_SLOTS)
            ]
            self._block_hists = [
                torch.empty(PLAN_BLOCKS * segs, dtype=torch.int32, device=device)
                for _ in range(self._COMPACT_PLAN_SLOTS)
            ]
            self._send_bases = [
                torch.empty(segs, dtype=torch.int32, device=device)
                for _ in range(self._COMPACT_PLAN_SLOTS)
            ]
            self._barriers = [
                torch.zeros(4, dtype=torch.int32, device=device)
                for _ in range(self._COMPACT_PLAN_SLOTS)
            ]
            self._compact_masked_m = self._masked_ms[0]
            self._compact_psum = self._psums[0]
            self._compact_dummy_ids = torch.zeros(
                (1, config.topk), dtype=torch.int32, device=device
            )
            self._compact_dummy_wts = torch.zeros(
                (1, config.topk), dtype=torch.float32, device=device
            )
            self._compact_plan_stream = torch.cuda.Stream()
            self._compact_plan_event = torch.cuda.Event()
            self._compact_plan_pending = False
            # Warm both hist/done slots and the capacity bucket: compiling
            # inside graph capture is not allowed.
            for slot in range(self._COMPACT_PLAN_SLOTS):
                self._compact_plan_for(self._compact_tile_m, slot)
        self._destination_peer_counter = torch.zeros(
            config.world_size, dtype=torch.int32, device=device
        )
        self._dispatch_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        # Points at the quant op's own scale rows, set per dispatch on a quantizing
        # wire; 0 (and unread) on bf16.
        self._dispatch_sent_scales_ptr = 0
        self._total_recv = torch.zeros(1, dtype=torch.int32, device=device)
        self._cross_device_flag = torch.ones(1, dtype=torch.int64, device=device)
        self._combine_output = torch.zeros(
            config.max_tokens_per_rank * config.hidden_dim,
            dtype=torch.int16,
            device=device,
        )

        if config.dispatch_backend == "mori":
            # mori's geometry is a compile-time Cfg field, so it brings its own
            # tuned buckets.
            config.schedule = _mori_dispatch_schedule(config)
        elif self._compact_plan:
            # Compact dispatch has one warp load a token and fan it out to all
            # final expert rows. Prefill needs enough token-owner warps to cover
            # the input in one pass; the wider grid cuts dispatch roughly in
            # half at 1K+ tokens, while its launch/synchronization overhead loses
            # on decode and smaller prompt buckets.
            blocks_env = os.environ.get("AITER_TDM_COMPACT_BLOCKS")
            warps_env = os.environ.get("AITER_TDM_COMPACT_WARPS")
            if blocks_env is not None or warps_env is not None:
                compact_blocks = int(blocks_env or "64")
                compact_warps = int(warps_env or "8")
                config.schedule = ((None, compact_blocks, compact_warps),)
            else:
                config.schedule = (
                    (512, 64, 8),
                    (None, 128, 16),
                )

        if config.schedule:
            dispatch_specs = sorted(
                {(block, warp) for _, block, warp in config.schedule}
            )
        else:
            dispatch_specs = [
                (
                    config.dispatch_block_num,
                    config.dispatch_warp_num_per_block,
                )
            ]
        self._dispatch_specs = dispatch_specs
        if config.dispatch_backend == "mori":
            self._dispatch_variants = self._build_mori_dispatch(config)
        elif config.dispatch_backend == "tdm":
            self._dispatch_variants = self._build_tdm_dispatch(config, device)
        else:
            self._dispatch_variants = {
                spec: _make_dispatch(
                    rank=config.rank,
                    npes=config.world_size,
                    experts_per_rank=config.experts_per_rank,
                    experts_per_token=config.topk,
                    hidden_dim=config.hidden_dim,
                    max_tok_per_rank=config.max_tokens_per_rank,
                    max_recv=config.max_recv,
                    off_tok_off=self._arena.offset("tok_off"),
                    off_recv_num=self._arena.offset("recv_num"),
                    off_tis=self._arena.offset("recv_to_src_token"),
                    off_out_idx=self._arena.offset("out_idx"),
                    off_out_wts=self._arena.offset("out_wts"),
                    off_out_tok=self._arena.offset("disp_out"),
                    block_num=spec[0],
                    warp_num_per_block=spec[1],
                )
                for spec in dispatch_specs
            }

        # Keep the cross-device barrier in its own 1-block kernel so the reduce
        # grid is unconstrained. 512x16 measured best-or-tied at every token count.
        combine_specs = [(512, 16)]
        self._combine_specs = combine_specs
        self._combine_variants = {
            spec: _make_combine_fused_reduce(
                experts_per_token=config.topk,
                hidden_dim=config.hidden_dim,
                block_num=spec[0],
                warp_num_per_block=spec[1],
                slot_stride_nbytes=config.combine_slot_stride_bytes,
            )
            for spec in combine_specs
        }
        self._combine_sync = _make_combine_fused_sync(
            rank=config.rank,
            npes=config.world_size,
            off_xdb_mem=self._arena.offset("cross_device_barrier"),
        )

    def _build_tdm_dispatch(self, config: MegaMoEConfig, device) -> dict:
        """The TDM dispatch, wearing `_make_dispatch`'s calling convention.

        It leaves the same arena state the vector dispatch does (disp_out rows
        at slot*hidden, out_idx/out_wts at slot*topk+k, recv_to_src_token as
        src_pe*max_tok+src_tok), so gemm1, the gemm2 P2P scatter and the fused
        combine are untouched. Only the recv SLOT a token lands in changes:
        slots are reserved one atomic per (block, peer) and handed out
        block-local, so a test comparing arena contents slot-by-slot against
        the vector dispatch will differ.

        The three staging arrays are this package's stand-in for the ``__device__``
        BSS the HIP kernel keeps: FINALIZE gathers idx / weights / srcmap into a
        peer-major destTokId SoA there so META can ship each block's reserved run
        as a couple of contiguous TDM copies.
        """
        payload_dim = config.dispatch_wire_elem_count
        elem_size = config.dispatch_wire_spec.recv_dtype.itemsize
        stg_cap, slots = tdm_stage_capacity(
            npes=config.world_size, max_recv=config.max_recv
        )
        stage = [
            torch.empty(slots * config.topk, dtype=torch.int32, device=device),
            torch.empty(slots * config.topk, dtype=torch.int32, device=device),
            torch.empty(slots, dtype=torch.int32, device=device),
        ]
        if config.dispatch_scale_dst_nbytes:
            stage.append(
                torch.empty(
                    slots * config.dispatch_scale_dst_nbytes,
                    dtype=torch.uint8,
                    device=device,
                )
            )
        self._tdm_stage = tuple(stage)
        for buf in self._tdm_stage:
            if buf.data_ptr() % 128:
                raise RuntimeError(
                    "TDM staging allocations must be 128-byte aligned, got "
                    f"0x{buf.data_ptr():x}"
                )
        # On a quantized wire the metadata batch grew by a scale row while the
        # payload shrank, so the shared LDS tile is floored at the bf16 width.
        slab_bytes = config.hidden_dim * 2 if config.is_quant_dispatch_wire else 0
        # A vector-tuned spec can name more warps than the payload tiles fit;
        # clamp the width but keep the tuned block count, which is what paces
        # the grid barrier, and keep the caller's spec as the variant key so the
        # runtime pick still resolves.
        max_warps = tdm_max_warps(
            hidden_dim=payload_dim,
            hidden_elem_size=elem_size,
            npes=config.world_size,
            slab_bytes=slab_bytes,
        )
        stg_idx, stg_wt, stg_src = self._tdm_stage[:3]
        stg_scale = self._tdm_stage[3] if len(self._tdm_stage) > 3 else None

        def make_variant(kern):
            def launch(
                arena_handle,
                addr_inp_tok,
                addr_inp_idx,
                addr_inp_wts,
                addr_tok_map,
                addr_dest_ctr,
                addr_disp_bar,
                addr_total_recv,
                my_lsa_rank,
                inp_cur_tok,
                stream,
            ):
                kern(
                    arena_handle,
                    addr_inp_tok,
                    addr_inp_idx,
                    addr_inp_wts,
                    addr_tok_map,
                    addr_dest_ctr,
                    addr_disp_bar,
                    addr_total_recv,
                    stg_idx.data_ptr(),
                    stg_wt.data_ptr(),
                    stg_src.data_ptr(),
                    stg_scale.data_ptr() if stg_scale is not None else 0,
                    self._dispatch_sent_scales_ptr,
                    self._route_counter.data_ptr(),
                    my_lsa_rank,
                    inp_cur_tok,
                    stream,
                )

            return launch

        built = {}
        variants = {}
        for spec in self._dispatch_specs:
            geom = (spec[0], min(spec[1], max_warps))
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
                    max_recv=(
                        self._compact_cap if self._compact_plan else config.max_recv
                    ),
                    compact_row_stride=(
                        self._compact_wire_row if self._compact_plan else 0
                    ),
                    enable_signal=True,
                    off_tok_off=self._arena.offset("tok_off"),
                    off_recv_num=self._arena.offset("recv_num"),
                    off_tis=self._arena.offset("recv_to_src_token"),
                    off_out_idx=self._arena.offset("out_idx"),
                    off_out_wts=self._arena.offset("out_wts"),
                    off_out_tok=self._arena.offset("disp_out"),
                    off_out_scales=(
                        (self._arena.offset("disp_out") + config.dispatch_token_nbytes)
                        if self._compact_plan
                        else (
                            self._arena.offset("disp_out_scales")
                            if config.dispatch_scale_dst_nbytes
                            else 0
                        )
                    ),
                    scale_bytes=config.dispatch_scale_nbytes,
                    scale_stride=config.dispatch_scale_dst_nbytes,
                    block_num=geom[0],
                    warp_num_per_block=geom[1],
                    clear_route_counter=os.environ.get("AITER_TDM_DIRECT_EP_MASK", "1")
                    in ("1", "true", "True")
                    and not self._compact_plan,
                    compact_plan=self._compact_plan,
                    off_ep_rowmap=(
                        self._arena.offset("ep_rowmap") if self._compact_plan else 0
                    ),
                    max_tok_slot_stride=config.max_tokens_per_rank * config.topk,
                )
            variants[spec] = make_variant(built[geom])
        self._tdm_stage_capacity = stg_cap
        return variants

    def _build_mori_dispatch(self, config: MegaMoEConfig) -> dict:
        """mori's HIP/JIT dispatch, wearing `_make_dispatch`'s calling convention.

        Only the kernel changes: mori leaves the same arena state this package's
        own dispatch does (disp_out rows at slot*hidden, out_idx/out_wts at
        slot*topk+k, recv_to_src_token as src_pe*max_tok+src_tok) and never
        touches cross_device_barrier, so gemm1, the gemm2 P2P scatter and the
        fused combine are untouched.

        The recv SLOT a token lands in does change -- mori reserves a block's
        slots with one atomic and hands them out block-local. Nothing indexes by
        slot order, but a test comparing arena contents slot-by-slot against the
        FlyDSL dispatch will differ.
        """
        try:
            from mori.ops.dispatch_combine_v2.ep_plans import EpDispatchPlan
        except ImportError as error:
            raise RuntimeError(
                "dispatch_backend='mori' needs a mori with ops/dispatch_combine_v2 "
                "(JIT v2, PR #548 or later)"
            ) from error
        except OSError as error:
            raise RuntimeError(
                "dispatch_backend='mori' needs mori's libmori_ops_v2.so; it is "
                "built by mori's CMake and is not shipped by every install"
            ) from error

        # Passed only on a quantizing wire, matching dispatch_scale_dst_nbytes: mori grew
        # scale_bytes in #593 and rejects UNKNOWN kwargs outright, so sending the
        # bf16 wire's harmless 0 would make an older mori refuse the whole plan.
        scale_kw = (
            {"scale_bytes": config.dispatch_scale_nbytes}
            if config.is_quant_dispatch_wire
            else {}
        )
        plans = {}
        for spec in self._dispatch_specs:
            plan = EpDispatchPlan(
                world_size=config.world_size,
                # see dispatch_wire_elem_count; mori's plan_api: "the caller halves
                # hiddenDim"
                hidden_dim=config.dispatch_wire_elem_count,
                max_tok_per_rank=config.max_tokens_per_rank,
                num_expert_per_rank=config.experts_per_rank,
                num_expert_per_token=config.topk,
                max_recv=config.max_recv,
                dtype=config.dispatch_wire_spec.mori_dtype,
                use_weights=True,
                **scale_kw,
                block_num=spec[0],
                warp_per_block=spec[1],
                arena=self._arena,
                region_names=_MORI_REGION_NAMES,
            )
            plan.bind(rank=config.rank)
            plans[spec] = plan
        # The kernels dereference the window, so the plans have to outlive them.
        self._mori_plans = plans

        def make_variant(plan):
            def launch(
                arena_handle,
                addr_inp_tok,
                addr_inp_idx,
                addr_inp_wts,
                addr_tok_map,
                addr_dest_ctr,
                addr_disp_bar,
                addr_total_recv,
                my_lsa_rank,
                inp_cur_tok,
                stream,
            ):
                # mori's dispatch only accumulates into total_recv (this package's
                # zeroes it in Phase 2), so without this it grows every forward.
                self._total_recv.zero_()
                plan.launch(
                    stream=torch.cuda.current_stream().cuda_stream,
                    token_indices=addr_inp_idx,
                    inp_token_buf=addr_inp_tok,
                    weights_buf=addr_inp_wts,
                    disp_dest_tok_id_map=addr_tok_map,
                    dest_pe_token_counter=addr_dest_ctr,
                    total_recv_token_num=addr_total_recv,
                    grid_barrier=addr_disp_bar,
                    num_tokens=inp_cur_tok,
                    # Read off self rather than through the variant's argument
                    # list: the list is shared with the FlyDSL dispatch, whose
                    # launcher is a traced @flyc.jit signature, and widening it
                    # would put a dead kernarg on a path that can never carry
                    # scales (fp8 requires dispatch_backend='mori').
                    scales_buf=self._dispatch_sent_scales_ptr,
                )

            return launch

        return {spec: make_variant(plan) for spec, plan in plans.items()}

    def _select_dispatch(self, token_count: int) -> tuple[int, int]:
        if not self._config.schedule:
            return self._dispatch_specs[0]
        for upper_bound, block, warp in self._config.schedule:
            if upper_bound is None or token_count <= upper_bound:
                spec = (block, warp)
                return (
                    spec
                    if spec in self._dispatch_variants
                    else self._dispatch_specs[-1]
                )
        return self._dispatch_specs[-1]

    def _recv_tokens(self) -> torch.Tensor:
        config = self._config
        # Width in whatever recv_dtype counts: features for bf16/fp8, bytes for
        # fp4 -- see _DispatchWire.recv_dtype.
        width = (
            config.dispatch_token_nbytes
            // config.dispatch_wire_spec.recv_dtype.itemsize
        )
        rows = self._compact_cap if self._compact_plan else config.max_recv
        if self._compact_plan:
            storage = _from_gpu_ptr(
                self._arena.local_ptr("disp_out"),
                (rows * self._compact_wire_row,),
                config.dispatch_wire_spec.recv_dtype,
            )
            return torch.as_strided(
                storage,
                (rows, width),
                (self._compact_wire_row, 1),
            )
        return _from_gpu_ptr(
            self._arena.local_ptr("disp_out"),
            (rows, width),
            config.dispatch_wire_spec.recv_dtype,
        )

    def _recv_dispatch_scales(self) -> torch.Tensor | None:
        """The forwarded e8m0 rows, or None on the bf16 wire."""
        if not self._config.is_quant_dispatch_wire:
            return None
        rows = self._compact_cap if self._compact_plan else self._config.max_recv
        if self._compact_plan:
            span = (rows - 1) * self._compact_wire_row + self._compact_scale_row
            storage = _from_gpu_ptr(
                self._arena.local_ptr("disp_out") + self._config.dispatch_token_nbytes,
                (span,),
                torch.uint8,
            )
            return torch.as_strided(
                storage,
                (rows, self._compact_scale_row),
                (self._compact_wire_row, 1),
            )
        return _from_gpu_ptr(
            self._arena.local_ptr("disp_out_scales"),
            (rows, self._compact_scale_row),
            torch.uint8,
        )

    def _recv_weights(self) -> torch.Tensor:
        return _from_gpu_ptr(
            self._arena.local_ptr("out_wts"),
            (self._config.max_recv, self._config.topk),
            torch.float32,
        )

    def _recv_indices(self) -> torch.Tensor:
        return _from_gpu_ptr(
            self._arena.local_ptr("out_idx"),
            (self._config.max_recv, self._config.topk),
            torch.int32,
        )

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        token_count = hidden_states.shape[0]
        spec = self._select_dispatch(token_count)
        if self._compact_plan and not self._compact_plan_pending:
            self._launch_compact_plan_async(topk_ids, token_count)
        stream = fx.Stream(torch.cuda.current_stream())
        payload = hidden_states
        if self._config.is_quant_dispatch_wire:
            # Quantize ONCE PER LOCAL TOKEN here, instead of once per received
            # copy on the far side. Destination-independent, so the bytes are the
            # same either way; the preshuffle cannot move with it, because its
            # destination is the grouped row the receiver assigns.
            from aiter.ops.quant import per_1x32_mx_quant_hip

            payload, scale_rows = per_1x32_mx_quant_hip(
                hidden_states,
                quant_dtype=self._config.dispatch_wire_spec.quant_dtype,
                scale_type=dtypes.fp8_e8m0,
                shuffle=False,
            )
            # Straight onto the wire; mori restrides these packed rows while it
            # stages them, so there is no repack here.
            self._dispatch_sent_scales_ptr = scale_rows.data_ptr()
        if self._compact_plan:
            self._wait_compact_plan()
            tok_map = self._tok_maps[self._compact_slot]
            self._token_destination_map = tok_map
            self._compact_masked_m = self._masked_ms[self._compact_slot]
            self._compact_psum = self._psums[self._compact_slot]
        else:
            tok_map = self._token_destination_map
        self._dispatch_variants[spec](
            self._arena.handle,
            payload.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            tok_map.data_ptr(),
            self._destination_peer_counter.data_ptr(),
            self._dispatch_barrier.data_ptr(),
            self._total_recv.data_ptr(),
            self._config.rank,
            token_count,
            stream,
        )
        reverse_source_view = _from_gpu_ptr(
            self._arena.local_ptr("recv_to_src_token"),
            (self._config.max_recv,),
            torch.int32,
        )
        routing = Routing(
            token_count=token_count,
            reverse_source_view=reverse_source_view,
        )
        return (
            self._recv_tokens(),
            self._recv_weights(),
            self._recv_indices(),
            self._total_recv,
            routing,
        )

    def _scatter_context(self, routing: Routing) -> Stage2ScatterContext:
        return Stage2ScatterContext(
            arena_handle=self._arena.handle,
            combine_input_offset=self._arena.offset("comb_inp"),
            slot_stride_bytes=self._config.combine_slot_stride_bytes,
            max_tokens_per_rank=self._config.max_tokens_per_rank,
            world_size=self._config.world_size,
            source_token_map=routing.source_token_map,
            compact_layout=self._compact_plan,
            compact_masked_m=self._compact_masked_m,
            compact_psum=self._compact_psum,
            compact_ep_rowmap=(
                _from_gpu_ptr(
                    self._arena.local_ptr("ep_rowmap"),
                    (self._compact_cap + 1, 2),
                    torch.int32,
                )
                if self._compact_plan
                else None
            ),
            compact_wire_row_stride=(
                self._compact_wire_row if self._compact_plan else 0
            ),
            compact_recv_bound=(
                self._compact_step_recv_bound if self._compact_plan else 0
            ),
            compact_align_m=(self._compact_step_align_m if self._compact_plan else 0),
            compact_rows=(self._compact_step_rows if self._compact_plan else 0),
        )

    def _combine(self, routing: Routing) -> torch.Tensor:
        spec = self._combine_specs[0]
        stream = fx.Stream(torch.cuda.current_stream())
        # 1-block cross-device barrier, then the barrier-free reduce on the same
        # stream; the kernel boundary gives the reduce its visibility.
        self._combine_sync(
            self._arena.handle,
            self._cross_device_flag.data_ptr(),
            self._config.rank,
            stream,
        )
        self._combine_variants[spec](
            self._arena.local_ptr("comb_inp"),
            self._combine_output.data_ptr(),
            routing.token_count,
            stream,
        )
        count = routing.token_count
        return (
            self._combine_output[: count * self._config.hidden_dim]
            .view(torch.bfloat16)
            .view(count, self._config.hidden_dim)
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._arena.close()
