# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What every dispatch and combine stage provides, and what they share."""

import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Protocol

import torch

from aiter import dtypes

from ..arena import SymmetricArena
from ..recipe import MegaMoEConfig
from ..types import Stage2ScatterContext, _from_gpu_ptr

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
