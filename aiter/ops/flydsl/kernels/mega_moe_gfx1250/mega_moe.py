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
from .types import Stage2ScatterContext, _from_gpu_ptr

__all__ = ["MegaMoEGfx1250"]

_DISPATCH_BACKENDS = ("flydsl", "mori")
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

    def __post_init__(self):
        if self.dispatch_wire not in _DISPATCH_WIRES:
            raise ValueError(
                f"dispatch_wire must be one of {_DISPATCH_WIRES}, "
                f"got {self.dispatch_wire!r}"
            )
        if self.is_quant_dispatch_wire and self.dispatch_backend != "mori":
            # Only mori's kernel carries the scale row; this package's own
            # dispatch has no channel for it.
            raise ValueError(
                f"dispatch_wire={self.dispatch_wire!r} requires "
                f"dispatch_backend='mori' (got {self.dispatch_backend!r})"
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
        try:
            from aiter.ops.flydsl.grouped_moe_gfx1250 import (
                grouped_gemm_gfx1250_a8w4,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MegaMoE fused stage2 scatter requires the grouped A8W4 kernel"
            ) from exc
        if not callable(grouped_gemm_gfx1250_a8w4):
            raise TypeError(
                "MegaMoE fused stage2 scatter requires the grouped A8W4 kernel"
            )
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
    ) -> torch.Tensor:
        """Run one MoE layer: dispatch, its expert GEMM, then the fused combine.

        ``recv_token_bound`` caps how many recv slots the GEMM is shown. Dispatch
        always fills a world_size*max_tokens_per_rank arena, so a caller that
        knows a tighter static bound (e.g. graph_bs*topk*world_size for a uniform
        decode batch) can shrink the GEMM's grid with it. It must be a python int
        so the shape stays static under graph capture, and must not cut below the
        received count -- the kernels skip the tail past the device-side count on
        their own.
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
        recv_x, recv_weights, recv_ids, total_recv, routing = self._dispatch(
            hidden_states, topk_weights, topk_ids
        )
        if recv_token_bound is not None:
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
            if recv_token_bound is not None:
                a1_scale = a1_scale[: int(recv_token_bound)]
        extra = {}
        if self.activation == ActivationType.Situv2:
            extra["beta"] = self.situ_beta
            extra["linear_beta"] = self.situ_linear_beta
        fused_moe(
            recv_x,
            w1,
            w2,
            recv_weights,
            recv_ids,
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
            num_local_tokens=total_recv,
            swiglu_limit=self.swiglu_limit,
            stage2_scatter=self._scatter_context(routing),
            **extra,
        )
        return self._combine(routing)

    __call__ = forward

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _initialize_pipeline(self, config: MegaMoEConfig, communicator):
        self._config = config
        self._closed = False
        device = torch.device("cuda", torch.cuda.current_device())
        max_recv = config.max_recv

        self._arena = SymmetricArena(
            communicator,
            [
                ("tok_off", 4),
                ("recv_num", config.world_size * 4),
                ("recv_to_src_token", max_recv * 4),
                ("out_idx", max_recv * config.topk * 4),
                ("out_wts", max_recv * config.topk * 4),
                ("disp_out", max_recv * config.dispatch_token_nbytes),
                ("cross_device_barrier", config.world_size * 8),
                *(
                    # Arrival stride, not the packed row: undersizing overruns
                    # the last slots.
                    [("disp_out_scales", max_recv * config.dispatch_scale_dst_nbytes)]
                    if config.dispatch_scale_dst_nbytes
                    else []
                ),
                (
                    "comb_inp",
                    config.max_tokens_per_rank
                    * config.topk
                    * config.combine_slot_stride_bytes,
                ),
            ],
        )
        self._arena.zero()

        self._token_destination_map = torch.full(
            (config.max_tokens_per_rank * config.topk,),
            -1,
            dtype=torch.int32,
            device=device,
        )
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
        return _from_gpu_ptr(
            self._arena.local_ptr("disp_out"),
            (config.max_recv, width),
            config.dispatch_wire_spec.recv_dtype,
        )

    def _recv_dispatch_scales(self) -> torch.Tensor | None:
        """The forwarded e8m0 rows, or None on the bf16 wire."""
        if not self._config.is_quant_dispatch_wire:
            return None
        # Full padded rows, not a trimmed view: this goes to the gather kernel as a
        # base pointer plus a build-constant pitch, and that pitch is the arrival
        # stride. The kernel reads only the meaningful bytes of each row.
        return _from_gpu_ptr(
            self._arena.local_ptr("disp_out_scales"),
            (self._config.max_recv, self._config.dispatch_scale_dst_nbytes),
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
        self._dispatch_variants[spec](
            self._arena.handle,
            payload.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            self._token_destination_map.data_ptr(),
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
