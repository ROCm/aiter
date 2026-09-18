# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host runtime for the fused TP MoE collectives.

Both legs of the layer run as a single kernel launch over one symmetric arena:

``all_gather``
    Pushes the activation payload, its E8M0 scales, and the routing ids and
    weights straight into every peer's arena, replacing four NCCL
    ``all_gather_into_tensor`` calls plus the routing unpack copies.

``reduce_scatter``
    Pulls this rank's token range out of every peer's GEMM2 partial and sums it
    in FP32, replacing ``reduce_scatter_tensor``.

The destination buffers are the ones GEMM1, ``moe_sorting`` and GEMM2 already
read and write, so the compute in between is untouched: ``payload``/``scale``
are the pre-quantized activation operand, ``topk_ids``/``topk_weights`` are the
gathered route, and ``partial`` is the GEMM2 atomic-accumulation target.

Where each backend wins
-----------------------
The fused kernels trade NCCL's multi-channel pipelining for a single launch and
no staging, which is the right trade while the transfer is latency-bound and
the wrong one once it is purely bandwidth-bound.  Measured on TP8 gfx950 at
H=3584 / topk=16 (fused vs NCCL, higher is better)::

    global tokens      8    128    512   2048   8192  32768
    AllGather      3.06x  3.36x  3.17x  2.56x  1.66x  0.89x
    ReduceScatter  1.68x  1.30x  1.51x  1.18x  0.56x  0.60x

So ``backend="auto"`` runs the fused path up to a per-leg byte budget and hands
the rest to NCCL.  The budgets sit just past the measured crossovers and are
overridable: this is a machine-specific tuning point, not a property of the
algorithm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num

from ..tensor_shim import _run_compiled
from .allgather_push import (
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

#: Upper bound on grid width, expressed as resident CTAs per CU. Past this the
#: extra CTAs only add launch and tail overhead.
_MAX_WAVES_PER_CU = 4

#: Per-leg wire budget (bytes this rank moves across the fabric) below which the
#: fused kernel beats NCCL. Past it NCCL's pipelined multi-channel transfer
#: wins. Both are overridable; 0 disables the fused path, -1 always uses it.
_AG_FUSED_MAX_BYTES = int(os.environ.get("AITER_TP_AG_FUSED_MAX_BYTES", 32 << 20))
_RS_FUSED_MAX_BYTES = int(os.environ.get("AITER_TP_RS_FUSED_MAX_BYTES", 24 << 20))

COLLECTIVE_BACKENDS = ("auto", "fused", "nccl")

__all__ = [
    "GatheredActivations",
    "TpMoeCollectives",
    "tp_moe_collectives_supported",
]


@dataclass(frozen=True)
class GatheredActivations:
    """Views of the arena holding the full token set after the push."""

    payload: torch.Tensor  # [M, H/2] uint8 (MXFP4) or [M, H] bf16
    scale: torch.Tensor | None  # [M, H/32] e8m0, None on the BF16 wire
    topk_ids: torch.Tensor  # [M, topk] int32
    topk_weights: torch.Tensor  # [M, topk] float32
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
    """One-launch AllGather and ReduceScatter over a shared symmetric arena.

    ``fp4_wire`` selects the MXFP4 wire (payload + E8M0 scales, 3.77x less
    traffic than BF16); with ``fp4_wire=False`` the payload region carries raw
    BF16 for the GEMM1 variants that quantize inline.
    """

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
        # Device-side epoch and phase-release counters keep both kernels free of
        # host state, so a captured graph replays correctly.
        ag_epoch = arena.reserve("ag_epoch", (1,), torch.int32)
        rs_epoch = arena.reserve("rs_epoch", (1,), torch.int32)
        arena.commit()
        self.arena = arena
        self._payload = payload
        self._scale = scale
        self._ids = ids
        self._weights = weights
        self._partial = partial
        # Only this rank ever writes or reads its own output, so it stays out of
        # the arena.
        self._output = torch.empty(
            (self.max_local_tokens, h), dtype=dtypes.bf16, device=self.device
        )

        self._ag_regions = (
            (payload, scale, ids, weights) if self.fp4_wire else (payload, ids, weights)
        )
        self._cu_num = get_cu_num()
        self._grid_blocks = grid_blocks

        # -- AllGather descriptor ---------------------------------------------
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
        self._ag_source_slots = [
            desc_region_src(self.world_size, index) for index in range(regions)
        ]
        self._ag_sources: tuple[int, ...] = ()
        self._ag_launch = compile_allgather_push(self.world_size, self.row_bytes)
        # Quantize-on-the-wire variant: same arena, same descriptor, but
        # region 0's source is the BF16 activation and the E8M0 scales are
        # produced in the kernel instead of read from a staging buffer.
        self._agq_launch = None
        if self.fp4_wire and quant_push_supported(self.model_dim, self.topk):
            self._agq_launch = compile_allgather_quant_push(
                self.world_size, self.model_dim, self.topk
            )

        # -- ReduceScatter descriptor -----------------------------------------
        self._rs_desc, rs_host = self._new_desc(
            rs_desc_size(self.world_size), rs_arrive, rs_flags
        )
        # Local-only, deliberately outside the arena: see RS_ARRIVE_SLOTS.
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

    # -- descriptor helpers -------------------------------------------------
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
        """Push the per-call source addresses into the device descriptor.

        A model reuses the same activation buffers every step, so this normally
        detects "unchanged" and skips the H2D copy entirely.
        """
        if sources == self._ag_sources:
            return
        for slot, address in zip(self._ag_source_slots, sources):
            self._ag_desc_host[slot] = int(address)
        self._ag_desc.copy_(
            torch.tensor(self._ag_desc_host, dtype=torch.int64, device="cpu"),
            non_blocking=True,
        )
        self._ag_sources = sources

    def _grid(self, units: int, block: int, unroll: int) -> int:
        """CTAs for one grid-stride pass, sized purely for bandwidth.

        No kernel here makes a CTA wait on a sibling -- the arrival counters are
        fetch-and-add only, and the one CTA that does spin is the last to
        arrive, so everything else has already finished. That means the grid is
        free to exceed what the device holds at once; the cap below only keeps
        launch overhead sane at token counts that do not need the width.
        """
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

    # -- AllGather ----------------------------------------------------------
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
            # Same destinations, so everything downstream is unaffected by
            # which backend ran.
            for tensor, region in zip(tensors, self._ag_regions):
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

    # -- GEMM2 target -------------------------------------------------------
    def partial_buffer(self, tokens: int) -> torch.Tensor:
        """The ``[tokens, H]`` arena slice GEMM2 accumulates into."""
        if tokens > self._partial.shape[0]:
            raise ValueError(
                f"{tokens} tokens exceeds the partial arena ({self._partial.shape[0]})"
            )
        return self._partial.local[:tokens]

    def quant_push_available(self) -> bool:
        """Whether this shape can quantize inside the AllGather kernel."""
        return self._agq_launch is not None

    def all_gather_quant(self, x_local, topk_ids, topk_weights):
        """AllGather a BF16 activation, quantizing to MXFP4 on the way out.

        Same result as ``all_gather`` on a pre-quantized operand -- the packed
        payload and E8M0 scales land in the same arena regions -- but the quant
        kernel, its staging buffers, and the round trip through them are gone.
        """
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
        # Region 1 (scales) has no source -- the kernel produces them -- but the
        # descriptor slot still has to hold a mapped address.
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

    # -- ReduceScatter ------------------------------------------------------
    def rs_descriptor(self) -> int:
        """Device address of the ReduceScatter descriptor.

        A kernel that folds the ReduceScatter into its own tail needs the same
        descriptor :meth:`reduce_scatter` hands the standalone publish and pull
        kernels; handing out the pointer keeps the arena and its slot layout
        owned here.
        """
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
