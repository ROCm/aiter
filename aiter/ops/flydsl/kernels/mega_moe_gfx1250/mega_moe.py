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

from .combine import _make_combine_fused_reduce, _XDB_FLAG_SLOTS
from .config import _FUSED_COMBINE_TIERS, _select_dispatch_config
from .dispatch import _make_dispatch
from .types import Stage2ScatterContext, _from_gpu_ptr

__all__ = ["MegaMoEGfx1250"]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _SymmetricArena:
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
class _Stage2Config:
    rank: int
    world_size: int
    hidden_dim: int
    max_tokens_per_rank: int
    experts_per_rank: int
    topk: int
    dispatch_block_num: int | None = None
    dispatch_warp_num_per_block: int | None = None
    schedule: tuple | None = None

    def __post_init__(self):
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank={self.rank} must be in [0, {self.world_size})")
        if self.world_size > 64:
            raise ValueError(
                f"intranode dispatch requires world_size <= 64, "
                f"got {self.world_size}"
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
    def token_nbytes(self) -> int:
        return self.hidden_dim * 2

    @property
    def combine_slot_stride_bytes(self) -> int:
        stride = 1
        while stride < self.token_nbytes:
            stride <<= 1
        return stride


@dataclass
class _Routing:
    token_count: int
    reverse_source_view: torch.Tensor

    @property
    def source_token_map(self) -> torch.Tensor:
        return self.reverse_source_view.clone()


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
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        max_tokens_per_rank: int,
        swiglu_limit: float = 0.0,
    ):
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

        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.experts_per_rank = self.experts // world_size
        self.topk = int(topk)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.swiglu_limit = float(swiglu_limit)
        self.w1 = w1.contiguous()
        self.w1_scale = w1_scale.contiguous()
        self.w2 = w2.contiguous()
        self.w2_scale = w2_scale.contiguous()

        if self.w1.shape[0] != self.experts_per_rank:
            raise ValueError(
                f"w1 has {self.w1.shape[0]} local experts, "
                f"expected {self.experts_per_rank}"
            )
        if self.w2.shape[0] != self.experts_per_rank:
            raise ValueError(
                f"w2 has {self.w2.shape[0]} local experts, "
                f"expected {self.experts_per_rank}"
            )

        device = self.w1.device
        self.expert_mask = torch.zeros(self.experts, dtype=torch.int32, device=device)
        first_expert = rank * self.experts_per_rank
        self.expert_mask[first_expert : first_expert + self.experts_per_rank] = 1

        self._initialize_pipeline(
            _Stage2Config(
                rank=int(rank),
                world_size=int(world_size),
                hidden_dim=self.model_dim,
                max_tokens_per_rank=self.max_tokens_per_rank,
                experts_per_rank=self.experts_per_rank,
                topk=self.topk,
            ),
            communicator,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous bfloat16")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
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
        fused_moe(
            recv_x,
            self.w1,
            self.w2,
            recv_weights,
            recv_ids,
            expert_mask=self.expert_mask,
            activation=ActivationType.Silu,
            gate_mode=GateMode.INTERLEAVE.value,
            quant_type=QuantType.per_1x32,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            dtype=dtypes.bf16,
            num_local_tokens=total_recv,
            swiglu_limit=self.swiglu_limit,
            stage2_scatter=self._scatter_context(routing),
        )
        return self._combine(routing)

    __call__ = forward

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _initialize_pipeline(self, config: _Stage2Config, communicator):
        self._config = config
        self._closed = False
        device = torch.device("cuda", torch.cuda.current_device())
        max_recv = config.max_recv

        self._arena = _SymmetricArena(
            communicator,
            [
                ("tok_off", 4),
                ("recv_num", config.world_size * 4),
                ("recv_to_src_token", max_recv * 4),
                ("out_idx", max_recv * config.topk * 4),
                ("out_wts", max_recv * config.topk * 4),
                ("disp_out", max_recv * config.token_nbytes),
                ("cross_device_barrier", config.world_size * 8),
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
        self._total_recv = torch.zeros(1, dtype=torch.int32, device=device)
        self._cross_device_flag = torch.ones(
            _XDB_FLAG_SLOTS, dtype=torch.int64, device=device
        )
        self._combine_output = torch.zeros(
            config.max_tokens_per_rank * config.hidden_dim,
            dtype=torch.int16,
            device=device,
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

        combine_override = os.environ.get("SCATTER_COMB_BW")
        if combine_override:
            combine_specs = [tuple(int(value) for value in combine_override.split(","))]
            self._combine_tiers = None
        else:
            self._combine_tiers = _FUSED_COMBINE_TIERS
            combine_specs = sorted({geometry for _, geometry in self._combine_tiers})
        self._combine_specs = combine_specs
        self._combine_variants = {
            spec: _make_combine_fused_reduce(
                rank=config.rank,
                npes=config.world_size,
                experts_per_token=config.topk,
                hidden_dim=config.hidden_dim,
                block_num=spec[0],
                warp_num_per_block=spec[1],
                off_comb_inp=self._arena.offset("comb_inp"),
                off_xdb_mem=self._arena.offset("cross_device_barrier"),
                slot_stride_nbytes=config.combine_slot_stride_bytes,
            )
            for spec in combine_specs
        }

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

    def _select_combine(self, token_count: int) -> tuple[int, int]:
        if self._combine_tiers is None:
            return self._combine_specs[0]
        for upper_bound, spec in self._combine_tiers:
            if upper_bound is None or token_count <= upper_bound:
                return spec
        return self._combine_specs[-1]

    def _recv_tokens(self) -> torch.Tensor:
        return _from_gpu_ptr(
            self._arena.local_ptr("disp_out"),
            (self._config.max_recv, self._config.hidden_dim),
            torch.bfloat16,
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
        self._dispatch_variants[spec](
            self._arena.handle,
            hidden_states.data_ptr(),
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
        routing = _Routing(
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

    def _scatter_context(self, routing: _Routing) -> Stage2ScatterContext:
        return Stage2ScatterContext(
            arena_handle=self._arena.handle,
            combine_input_offset=self._arena.offset("comb_inp"),
            slot_stride_bytes=self._config.combine_slot_stride_bytes,
            max_tokens_per_rank=self._config.max_tokens_per_rank,
            world_size=self._config.world_size,
            source_token_map=routing.source_token_map,
        )

    def _combine(self, routing: _Routing) -> torch.Tensor:
        spec = self._select_combine(routing.token_count)
        stream = fx.Stream(torch.cuda.current_stream())
        self._combine_variants[spec](
            self._arena.handle,
            self._arena.local_ptr("comb_inp"),
            self._cross_device_flag.data_ptr(),
            self._combine_output.data_ptr(),
            self._config.rank,
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
