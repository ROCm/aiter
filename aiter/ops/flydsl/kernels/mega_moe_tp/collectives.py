# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host runtime for the fused TP MoE collectives."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num

from ..tensor_shim import _run_compiled
from .allgather_push import (
    AG_DESC_DONE,
    AG_DESC_EPOCH,
    PUSH_MIN_BYTES,
    PUSH_UNROLL,
    ag_desc_size,
    ag_desc_slot,
    compile_allgather_push,
    push_units,
)
from .allgather_quant_push import (
    compile_allgather_quant_push,
    quant_push_supported,
    quant_push_units,
)
from .p2p import (
    DESC_ARRIVE,
    DESC_FLAGS,
    DESC_PEER_BASE,
    desc_region_offset,
    desc_region_src,
)
from .reduce_scatter import (
    RS_ARRIVE_SLOTS,
    RS_DESC_EPOCH,
    RS_DESC_FANIN,
    RS_DESC_OUTPUT,
    RS_DESC_PARTIAL,
    RS_PULL_UNROLL,
    RS_UNIT_ELEMS,
    compile_reduce_scatter_publish,
    compile_reduce_scatter_pull,
    rs_desc_size,
    rs_desc_slot,
)
from .symmetric_arena import SymmetricArena

_MAX_WAVES_PER_CU = 4

_AG_FUSED_MAX_BYTES = int(os.environ.get("AITER_TP_AG_FUSED_MAX_BYTES", 32 << 20))
_RS_FUSED_MAX_BYTES = int(os.environ.get("AITER_TP_RS_FUSED_MAX_BYTES", 24 << 20))
_RS_TAIL_MAX_BYTES = int(os.environ.get("AITER_TP_RS_TAIL_MAX_BYTES", 64 << 20))
_RS_CUSTOM = os.environ.get("AITER_TP_RS_CUSTOM", "0") == "1"
_AG_CUSTOM = os.environ.get("AITER_TP_AG_CUSTOM", "0") == "1"

COLLECTIVE_BACKENDS = ("auto", "fused", "nccl")

__all__ = [
    "GatheredActivations",
    "TpMoeCollectives",
    "tp_moe_collectives_supported",
]


@dataclass(frozen=True)
class GatheredActivations:
    """Views of the arena holding the full token set after the push."""

    payload: torch.Tensor
    scale: torch.Tensor | None
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    tokens: int


def _region_widths(model_dim: int, topk: int, *, fp4_wire: bool) -> tuple[int, ...]:
    if fp4_wire:
        return (model_dim // 2, model_dim // 32, topk * 4, topk * 4)
    return (model_dim * 2, topk * 4, topk * 4)


def tp_moe_collectives_supported(
    model_dim: int, topk: int, *, fp4_wire: bool
) -> bool:
    """Whether every region width divides into whole push/pull units."""
    widths = _region_widths(model_dim, topk, fp4_wire=fp4_wire)
    return all(width % PUSH_MIN_BYTES == 0 for width in widths) and (
        model_dim % RS_UNIT_ELEMS == 0
    )


class TpMoeCollectives:
    """One-launch AllGather and ReduceScatter over a shared symmetric arena."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        topk: int,
        max_local_tokens: int,
        device: torch.device | None = None,
        group=None,
        fp4_wire: bool = True,
        grid_blocks: int | None = None,
        backend: str = "auto",
    ):
        if backend not in COLLECTIVE_BACKENDS:
            raise ValueError(f"backend must be one of {COLLECTIVE_BACKENDS}")
        self.backend = backend
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.topk = int(topk)
        self.max_local_tokens = int(max_local_tokens)
        self.fp4_wire = bool(fp4_wire)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.group = group
        self.row_bytes = _region_widths(model_dim, topk, fp4_wire=self.fp4_wire)
        if not tp_moe_collectives_supported(
            model_dim, topk, fp4_wire=self.fp4_wire
        ):
            raise ValueError(
                f"model_dim={model_dim} topk={topk} fp4_wire={self.fp4_wire} gives "
                f"region widths {self.row_bytes}, which do not divide into "
                f"{PUSH_MIN_BYTES}-byte push units / {RS_UNIT_ELEMS}-element "
                "reduce units"
            )

        total = self.max_local_tokens * self.world_size
        h, k = self.model_dim, self.topk
        arena = SymmetricArena(group=group, device=self.device)
        if self.fp4_wire:
            payload = arena.reserve("payload", (total, h // 2), torch.uint8)
            scale = arena.reserve("scale", (total, h // 32), torch.uint8)
        else:
            payload = arena.reserve("payload", (total, h), dtypes.bf16)
            scale = None
        ids = arena.reserve("topk_ids", (total, k), torch.int32)
        weights = arena.reserve("topk_weights", (total, k), torch.float32)
        partial = arena.reserve("partial", (total, h), dtypes.bf16)
        ag_flags = arena.reserve("ag_flags", (self.world_size,), torch.int32)
        rs_flags = arena.reserve("rs_flags", (self.world_size,), torch.int32)
        ag_arrive = arena.reserve("ag_arrive", (1,), torch.int32)
        rs_arrive = arena.reserve("rs_arrive", (1,), torch.int32)
        ag_epoch = arena.reserve("ag_epoch", (1,), torch.int32)
        ag_done = arena.reserve("ag_done", (1,), torch.int32)
        rs_epoch = arena.reserve("rs_epoch", (1,), torch.int32)
        arena.commit()
        self.arena = arena
        self._payload = payload
        self._scale = scale
        self._ids = ids
        self._weights = weights
        self._partial = partial
        self._output = torch.empty(
            (self.max_local_tokens, h), dtype=dtypes.bf16, device=self.device
        )

        self._ag_regions = (
            (payload, scale, ids, weights) if self.fp4_wire else (payload, ids, weights)
        )
        self._cu_num = get_cu_num()
        self._grid_blocks = grid_blocks

        regions = len(self._ag_regions)
        self._ag_desc, self._ag_desc_host = self._new_desc(
            ag_desc_size(self.world_size, regions), ag_arrive, ag_flags
        )
        for index, region in enumerate(self._ag_regions):
            self._ag_desc_host[desc_region_offset(self.world_size, index)] = int(
                region.offset
            )
        self._ag_desc_host[
            ag_desc_slot(self.world_size, regions, AG_DESC_EPOCH)
        ] = int(ag_epoch.peer_ptrs[self.rank])
        self._ag_desc_host[
            ag_desc_slot(self.world_size, regions, AG_DESC_DONE)
        ] = int(ag_done.peer_ptrs[self.rank])
        self._ag_source_slots = [
            desc_region_src(self.world_size, index) for index in range(regions)
        ]
        self._ag_sources: tuple[int, ...] = ()
        self._ag_desc_pinned: "torch.Tensor | None" = None
        self._ag_launch = compile_allgather_push(self.world_size, self.row_bytes)
        self._agq_launch = None
        self._agq_route_launch = None
        self._agq_payload_launch = None
        if self.fp4_wire and quant_push_supported(self.model_dim, self.topk):
            self._agq_launch = compile_allgather_quant_push(
                self.world_size, self.model_dim, self.topk
            )
            self._agq_route_launch = compile_allgather_quant_push(
                self.world_size, self.model_dim, self.topk, regions="route"
            )
            self._agq_payload_launch = compile_allgather_quant_push(
                self.world_size, self.model_dim, self.topk, regions="payload"
            )

        self._rs_desc, rs_host = self._new_desc(
            rs_desc_size(self.world_size), rs_arrive, rs_flags
        )
        self._rs_fanin = torch.zeros(
            RS_ARRIVE_SLOTS, dtype=torch.int32, device=self.device
        )
        for which, value in (
            (RS_DESC_PARTIAL, partial.offset),
            (RS_DESC_OUTPUT, self._output.data_ptr()),
            (RS_DESC_EPOCH, rs_epoch.peer_ptrs[self.rank]),
            (RS_DESC_FANIN, self._rs_fanin.data_ptr()),
        ):
            rs_host[rs_desc_slot(self.world_size, which)] = int(value)
        self._rs_desc.copy_(
            torch.tensor(rs_host, dtype=torch.int64, device="cpu"), non_blocking=True
        )
        self._rs_publish = compile_reduce_scatter_publish(self.world_size)
        self._rs_pull = compile_reduce_scatter_pull(self.world_size, h)

    def _new_desc(self, size: int, arrive, flags):
        host = [0] * size
        host[DESC_ARRIVE] = int(arrive.peer_ptrs[self.rank])
        host[DESC_FLAGS] = int(flags.offset)
        for peer in range(self.world_size):
            host[DESC_PEER_BASE + peer] = int(self.arena.base_ptrs[peer])
        return (
            torch.zeros(size, dtype=torch.int64, device=self.device),
            host,
        )

    def _publish_ag_sources(self, sources: tuple[int, ...]) -> None:
        """Push the per-call source addresses into the device descriptor."""
        if sources == self._ag_sources:
            return
        staging = self._ag_desc_pinned
        if staging is None:
            staging = torch.tensor(
                self._ag_desc_host, dtype=torch.int64, device="cpu",
                pin_memory=True,
            )
            self._ag_desc_pinned = staging
        for slot, address in zip(self._ag_source_slots, sources):
            self._ag_desc_host[slot] = int(address)
            staging[slot] = int(address)
        self._ag_desc.copy_(staging, non_blocking=True)
        self._ag_sources = sources

    def _grid(self, units: int, block: int, unroll: int) -> int:
        """CTAs for one grid-stride pass, sized purely for bandwidth."""
        if self._grid_blocks:
            return int(self._grid_blocks)
        per_block = block * unroll
        return max(
            1,
            min(self._cu_num * _MAX_WAVES_PER_CU, (units + per_block - 1) // per_block),
        )

    def _use_fused(self, wire_bytes: int, budget: int) -> bool:
        """Whether the fused kernel is the faster backend for this transfer."""
        if self.backend != "auto":
            return self.backend == "fused"
        return budget < 0 or wire_bytes <= budget

    def ag_wire_bytes(self, rows: int) -> int:
        """Bytes this rank pushes over the fabric for one AllGather."""
        return rows * sum(self.row_bytes) * (self.world_size - 1)

    def rs_wire_bytes(self, rows: int) -> int:
        """Bytes this rank pulls over the fabric for one ReduceScatter."""
        return rows * self.model_dim * 2 * (self.world_size - 1)

    def all_gather(self, payload, scale, topk_ids, topk_weights):
        rows = int(topk_ids.shape[0])
        if rows > self.max_local_tokens:
            raise ValueError(
                f"local tokens {rows} exceeds max_local_tokens {self.max_local_tokens}"
            )
        tensors = (
            (payload, scale, topk_ids, topk_weights)
            if self.fp4_wire
            else (payload, topk_ids, topk_weights)
        )
        for tensor, width in zip(tensors, self.row_bytes):
            if tensor is None or not tensor.is_contiguous():
                raise ValueError("every push source must be a contiguous tensor")
            if tensor.numel() * tensor.element_size() != rows * width:
                raise ValueError(
                    f"push source has {tensor.numel() * tensor.element_size()} bytes, "
                    f"expected {rows * width} for {rows} rows"
                )
        total = rows * self.world_size
        if self._use_fused(self.ag_wire_bytes(rows), _AG_FUSED_MAX_BYTES):
            self._publish_ag_sources(tuple(int(t.data_ptr()) for t in tensors))
            units = push_units(rows, self.row_bytes)
            _run_compiled(
                self._ag_launch,
                int(self._ag_desc.data_ptr()),
                int(self.rank),
                int(rows),
                self._grid(units, self._ag_launch.block, PUSH_UNROLL),
                torch.cuda.current_stream(),
            )
        else:
            for tensor, region in zip(tensors, self._ag_regions):
                if _AG_CUSTOM:
                    from aiter.dist.communication_op import (
                        tensor_model_parallel_all_gather,
                    )

                    region.local[:total].copy_(
                        tensor_model_parallel_all_gather(
                            tensor, use_custom=True, dim=0
                        )
                    )
                else:
                    dist.all_gather_into_tensor(
                        region.local[:total], tensor, group=self.group
                    )
        return GatheredActivations(
            payload=self._payload.local[:total],
            scale=None if self._scale is None else self._scale.local[:total],
            topk_ids=self._ids.local[:total],
            topk_weights=self._weights.local[:total],
            tokens=total,
        )

    def partial_buffer(self, tokens: int) -> torch.Tensor:
        """The ``[tokens, H]`` arena slice GEMM2 accumulates into."""
        if tokens > self._partial.shape[0]:
            raise ValueError(
                f"{tokens} tokens exceeds the partial arena ({self._partial.shape[0]})"
            )
        return self._partial.local[:tokens]

    def payload_views(self, total_tokens: int):
        """The arena payload and scale slices, *without* pushing anything."""
        if total_tokens > self._payload.local.shape[0]:
            raise ValueError(
                f"{total_tokens} tokens exceeds the payload arena "
                f"({self._payload.local.shape[0]})"
            )
        return self._payload.local[:total_tokens], self._scale.local[:total_tokens]

    def route_views(self, total_tokens: int):
        """The arena route slices, *without* pushing anything."""
        if total_tokens > self._ids.local.shape[0]:
            raise ValueError(
                f"{total_tokens} tokens exceeds the route arena "
                f"({self._ids.local.shape[0]})"
            )
        return self._weights.local[:total_tokens], self._ids.local[:total_tokens]

    def publish_payload_source(self, x_local, topk_ids, topk_weights) -> None:
        """Point the descriptor at the operands a hosted push will read."""
        self._check_push_activation(x_local, int(x_local.shape[0]))
        self._publish_ag_sources(
            (
                int(x_local.data_ptr()),
                int(x_local.data_ptr()),
                int(topk_ids.data_ptr()),
                int(topk_weights.data_ptr()),
            )
        )

    def ag_descriptor(self) -> int:
        """Device address of the AllGather descriptor."""
        return int(self._ag_desc.data_ptr())

    def quant_push_available(self) -> bool:
        """Whether this shape can quantize inside the AllGather kernel."""
        return self._agq_launch is not None

    def all_gather_route(self, topk_ids, topk_weights):
        """AllGather the routing metadata alone, ahead of everything else."""
        if self._agq_route_launch is None:
            raise RuntimeError("this shape has no fused quantize-and-push kernel")
        rows = self._check_push_rows(topk_ids, topk_weights)
        self._publish_ag_sources(
            (
                int(topk_ids.data_ptr()),
                int(topk_ids.data_ptr()),
                int(topk_ids.data_ptr()),
                int(topk_weights.data_ptr()),
            )
        )
        self._run_push(self._agq_route_launch, rows, "route")
        total = rows * self.world_size
        return self._weights.local[:total], self._ids.local[:total]

    def all_gather_payload(self, x_local):
        """AllGather the activation, quantizing to MXFP4 on the way out."""
        if self._agq_payload_launch is None:
            raise RuntimeError("this shape has no fused quantize-and-push kernel")
        rows = int(x_local.shape[0])
        self._check_push_activation(x_local, rows)
        self._publish_ag_sources(
            (
                int(x_local.data_ptr()),
                int(x_local.data_ptr()),
                int(x_local.data_ptr()),
                int(x_local.data_ptr()),
            )
        )
        self._run_push(self._agq_payload_launch, rows, "payload")
        total = rows * self.world_size
        return self._payload.local[:total], self._scale.local[:total]

    def _check_push_rows(self, topk_ids, topk_weights) -> int:
        rows = int(topk_ids.shape[0])
        if rows > self.max_local_tokens:
            raise ValueError(
                f"local tokens {rows} exceeds max_local_tokens {self.max_local_tokens}"
            )
        for tensor in (topk_ids, topk_weights):
            if not tensor.is_contiguous():
                raise ValueError("every push source must be contiguous")
        return rows

    def _check_push_activation(self, x_local, rows: int) -> None:
        if rows > self.max_local_tokens:
            raise ValueError(
                f"local tokens {rows} exceeds max_local_tokens {self.max_local_tokens}"
            )
        if x_local.dtype != dtypes.bf16 or not x_local.is_contiguous():
            raise ValueError("the fused quant push needs a contiguous bf16 operand")
        if x_local.shape != (rows, self.model_dim):
            raise ValueError(
                f"x_local {tuple(x_local.shape)} != ({rows}, {self.model_dim})"
            )

    def _run_push(self, launch, rows: int, regions: str) -> None:
        units = quant_push_units(rows, self.model_dim, self.topk, regions)
        _run_compiled(
            launch,
            int(self._ag_desc.data_ptr()),
            int(self.rank),
            int(rows),
            self._grid(units, launch.block, 1),
            torch.cuda.current_stream(),
        )

    def all_gather_quant(self, x_local, topk_ids, topk_weights):
        """AllGather a BF16 activation, quantizing to MXFP4 on the way out."""
        if self._agq_launch is None:
            raise RuntimeError("this shape has no fused quantize-and-push kernel")
        rows = int(topk_ids.shape[0])
        if rows > self.max_local_tokens:
            raise ValueError(
                f"local tokens {rows} exceeds max_local_tokens {self.max_local_tokens}"
            )
        if x_local.dtype != dtypes.bf16 or not x_local.is_contiguous():
            raise ValueError("the fused quant push needs a contiguous bf16 operand")
        if x_local.shape != (rows, self.model_dim):
            raise ValueError(
                f"x_local {tuple(x_local.shape)} != ({rows}, {self.model_dim})"
            )
        for tensor in (topk_ids, topk_weights):
            if not tensor.is_contiguous():
                raise ValueError("every push source must be contiguous")
        self._publish_ag_sources(
            (
                int(x_local.data_ptr()),
                int(x_local.data_ptr()),
                int(topk_ids.data_ptr()),
                int(topk_weights.data_ptr()),
            )
        )
        units = quant_push_units(rows, self.model_dim, self.topk)
        _run_compiled(
            self._agq_launch,
            int(self._ag_desc.data_ptr()),
            int(self.rank),
            int(rows),
            self._grid(units, self._agq_launch.block, 1),
            torch.cuda.current_stream(),
        )
        total = rows * self.world_size
        return GatheredActivations(
            payload=self._payload.local[:total],
            scale=self._scale.local[:total],
            topk_ids=self._ids.local[:total],
            topk_weights=self._weights.local[:total],
            tokens=total,
        )

    def rs_fused_is_profitable(self, local_rows: int) -> bool:
        """Whether folding the ReduceScatter into a compute kernel still wins."""
        return self._use_fused(self.rs_wire_bytes(local_rows), _RS_TAIL_MAX_BYTES)

    def rs_descriptor(self) -> int:
        """Device address of the ReduceScatter descriptor."""
        return int(self._rs_desc.data_ptr())

    def output_buffer(self, local_tokens: int) -> torch.Tensor:
        """The ``[local_tokens, H]`` buffer the ReduceScatter writes."""
        if local_tokens > self.max_local_tokens:
            raise ValueError(
                f"local tokens {local_tokens} exceeds max_local_tokens "
                f"{self.max_local_tokens}"
            )
        return self._output[:local_tokens]

    def reduce_scatter(self, local_tokens: int) -> torch.Tensor:
        """Sum every peer's partial over this rank's token range."""
        if local_tokens > self.max_local_tokens:
            raise ValueError(
                f"local tokens {local_tokens} exceeds max_local_tokens "
                f"{self.max_local_tokens}"
            )
        if not self._use_fused(self.rs_wire_bytes(local_tokens), _RS_FUSED_MAX_BYTES):
            if _RS_CUSTOM:
                from aiter.dist.communication_op import (
                    tensor_model_parallel_reduce_scatter,
                )

                out = tensor_model_parallel_reduce_scatter(
                    self._partial.local[: local_tokens * self.world_size],
                    use_custom=True,
                )
                self._output[:local_tokens].copy_(out)
                return self._output[:local_tokens]
            dist.reduce_scatter_tensor(
                self._output[:local_tokens],
                self._partial.local[: local_tokens * self.world_size],
                group=self.group,
            )
            return self._output[:local_tokens]
        stream = torch.cuda.current_stream()
        units = local_tokens * self._rs_pull.units_per_row
        desc = int(self._rs_desc.data_ptr())
        _run_compiled(
            self._rs_publish,
            desc,
            int(self.rank),
            int(local_tokens),
            self._rs_publish.blocks,
            stream,
        )
        _run_compiled(
            self._rs_pull,
            desc,
            int(self.rank),
            int(local_tokens),
            self._grid(units, self._rs_pull.block, RS_PULL_UNROLL),
            stream,
        )
        return self._output[:local_tokens]
