# SPDX-License-Identifier: MIT
"""Host-side arena and epoch contract for the hierarchical MoE pipeline.

The arena is deliberately transport-neutral.  A production caller may back it
with one CCO registered window, while unit tests can use a contiguous uint8
tensor.  Device kernels receive raw pointers to individual regions; they never
need to carry a communicator or window object through an MFMA loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (int(value) + alignment - 1) & -alignment


@dataclass(frozen=True)
class ArenaRegion:
    name: str
    offset: int
    nbytes: int
    alignment: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class HierCcoArenaLayout:
    """Byte layout shared by dispatch, H1, H2 and return sidecars.

    ``max_m_tiles`` is the capacity of the expert-major route stream on one
    destination rank. ``max_source_tokens`` is the flattened
    ``source_rank * max_tokens_per_rank + token`` capacity.
    """

    ring_depth: int
    num_qp: int
    chunk_bytes: int
    max_m_tiles: int
    max_source_tokens: int
    max_h1_n_blocks: int
    regions: tuple[ArenaRegion, ...]
    total_bytes: int

    @classmethod
    def create(
        cls,
        *,
        ring_depth: int = 8,
        num_qp: int = 4,
        chunk_bytes: int = 64 * 1024,
        max_m_tiles: int,
        max_source_tokens: int,
        max_h1_n_blocks: int = 4,
        max_fanout_records: int | None = None,
        fanout_record_bytes: int = 2048,
    ) -> "HierCcoArenaLayout":
        fanout_records = (
            int(max_source_tokens)
            if max_fanout_records is None
            else int(max_fanout_records)
        )
        for name, value in (
            ("ring_depth", ring_depth),
            ("num_qp", num_qp),
            ("chunk_bytes", chunk_bytes),
            ("max_m_tiles", max_m_tiles),
            ("max_source_tokens", max_source_tokens),
            ("max_h1_n_blocks", max_h1_n_blocks),
            ("max_fanout_records", fanout_records),
            ("fanout_record_bytes", fanout_record_bytes),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(num_qp) not in (1, 2, 4, 8):
            raise ValueError("num_qp must be one of 1,2,4,8")
        if int(max_source_tokens) >= 1 << 24:
            raise ValueError("max_source_tokens exceeds the packed source ABI")
        if int(fanout_record_bytes) % 16:
            raise ValueError("fanout_record_bytes must be 16-byte aligned")

        specs: list[tuple[str, tuple[int, ...], torch.dtype, int]] = [
            ("dispatch_tx", (ring_depth, chunk_bytes), torch.uint8, 256),
            ("dispatch_rx", (ring_depth, chunk_bytes), torch.uint8, 256),
            ("partial_tx", (ring_depth, chunk_bytes), torch.uint8, 256),
            ("partial_rx", (ring_depth, chunk_bytes), torch.uint8, 256),
            # One absolute generation per (ring slot, QP). These are ordinary
            # registered-window data, not CCO signal-pool identifiers.
            ("dispatch_ready", (ring_depth, num_qp), torch.int64, 64),
            ("dispatch_credit", (ring_depth, num_qp), torch.int64, 64),
            # Opaque flushAsync completion token retained by the sender until
            # the corresponding remote credit permits ring-slot reuse.
            ("dispatch_request", (ring_depth, num_qp), torch.int64, 64),
            ("partial_ready", (ring_depth, num_qp), torch.int64, 64),
            ("partial_credit", (ring_depth, num_qp), torch.int64, 64),
            ("partial_request", (ring_depth, num_qp), torch.int64, 64),
            # Preallocated node-local LSA fan-out targets. One plan entry owns
            # one fixed slot; no device allocator is involved in this version.
            (
                "fanout_inbox",
                (fanout_records, int(fanout_record_bytes)),
                torch.uint8,
                256,
            ),
            ("fanout_ready", (2, fanout_records), torch.int64, 64),
            ("fanout_count", (2,), torch.int32, 64),
            ("dispatch_eos", (2,), torch.int64, 64),
            ("partial_eos", (2,), torch.int64, 64),
            # Double-buffered epoch state. Expected/ready are route counts;
            # stage-ready arrays carry absolute generations.
            ("plan_ready", (2,), torch.int64, 64),
            ("h1_input_expected", (2, max_m_tiles), torch.int32, 64),
            ("h1_input_ready", (2, max_m_tiles), torch.int32, 64),
            # Compute-only queue. One sidecar publisher writes flat tile IDs,
            # then release-publishes tail. Header fields are
            # [epoch, total_work, tail, done_generation].
            ("h1_queue_header", (2, 4), torch.int64, 64),
            (
                "h1_ready_queue",
                (2, max_m_tiles * max_h1_n_blocks),
                torch.int32,
                64,
            ),
            ("h1_output_done", (2, max_m_tiles), torch.int32, 64),
            ("h1_output_ready", (2, max_m_tiles), torch.int64, 64),
            ("h2_output_done", (2, max_m_tiles), torch.int32, 64),
            ("h2_output_ready", (2, max_m_tiles), torch.int64, 64),
            ("rank_route_expected", (2, max_source_tokens), torch.int32, 64),
            ("rank_route_ready", (2, max_source_tokens), torch.int32, 64),
            ("node_partial_ready", (2, max_source_tokens), torch.int64, 64),
            ("final_output_ready", (2, max_source_tokens), torch.int64, 64),
        ]

        offset = 0
        regions: list[ArenaRegion] = []
        for name, shape, dtype, alignment in specs:
            offset = _align_up(offset, alignment)
            elem_size = torch.empty((), dtype=dtype).element_size()
            numel = 1
            for dim in shape:
                numel *= int(dim)
            nbytes = numel * elem_size
            regions.append(
                ArenaRegion(name, offset, nbytes, alignment, tuple(shape), dtype)
            )
            offset += nbytes
        total = _align_up(offset, 4096)
        return cls(
            int(ring_depth),
            int(num_qp),
            int(chunk_bytes),
            int(max_m_tiles),
            int(max_source_tokens),
            int(max_h1_n_blocks),
            tuple(regions),
            total,
        )

    def region(self, name: str) -> ArenaRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def offset(self, name: str, *, parity: int | None = None) -> int:
        region = self.region(name)
        offset = region.offset
        if parity is not None:
            if parity not in (0, 1) or not region.shape or region.shape[0] != 2:
                raise ValueError(f"{name} is not a parity-indexed region")
            offset += parity * (region.nbytes // 2)
        return offset

    def pointer(self, base_ptr: int, name: str, *, parity: int | None = None) -> int:
        return int(base_ptr) + self.offset(name, parity=parity)

    def allocate_local(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(self.total_bytes, dtype=torch.uint8, device=device)

    def view(
        self, arena: torch.Tensor, name: str, *, parity: int | None = None
    ) -> torch.Tensor:
        if (
            arena.dtype != torch.uint8
            or not arena.is_contiguous()
            or arena.numel() < self.total_bytes
        ):
            raise ValueError(
                "arena must be a contiguous uint8 tensor with "
                "layout.total_bytes capacity"
            )
        region = self.region(name)
        offset = self.offset(name, parity=parity)
        nbytes = region.nbytes if parity is None else region.nbytes // 2
        if offset % torch.empty((), dtype=region.dtype).element_size():
            raise AssertionError(f"misaligned region {name}")
        out = arena[offset : offset + nbytes].view(region.dtype)
        shape = region.shape if parity is None else region.shape[1:]
        return out.view(shape)

    def epoch_pointers(self, base_ptr: int, generation: int) -> "HierEpochPointers":
        parity = int(generation) & 1
        return HierEpochPointers(
            generation=int(generation),
            parity=parity,
            rank_partial_epoch_ready=self.pointer(
                base_ptr, "partial_eos", parity=parity
            ),
            plan_ready=self.pointer(base_ptr, "plan_ready") + parity * 8,
            h1_input_expected=self.pointer(
                base_ptr, "h1_input_expected", parity=parity
            ),
            h1_input_ready=self.pointer(
                base_ptr, "h1_input_ready", parity=parity
            ),
            h1_queue_header=self.pointer(
                base_ptr, "h1_queue_header", parity=parity
            ),
            h1_ready_queue=self.pointer(
                base_ptr, "h1_ready_queue", parity=parity
            ),
            h1_output_done=self.pointer(
                base_ptr, "h1_output_done", parity=parity
            ),
            h1_output_ready=self.pointer(
                base_ptr, "h1_output_ready", parity=parity
            ),
            h2_output_done=self.pointer(
                base_ptr, "h2_output_done", parity=parity
            ),
            h2_output_ready=self.pointer(
                base_ptr, "h2_output_ready", parity=parity
            ),
            rank_route_expected=self.pointer(
                base_ptr, "rank_route_expected", parity=parity
            ),
            rank_route_ready=self.pointer(
                base_ptr, "rank_route_ready", parity=parity
            ),
            node_partial_ready=self.pointer(
                base_ptr, "node_partial_ready", parity=parity
            ),
            fanout_ready=self.pointer(
                base_ptr, "fanout_ready", parity=parity
            ),
            fanout_count=self.pointer(
                base_ptr, "fanout_count", parity=parity
            ),
            final_output_ready=self.pointer(
                base_ptr, "final_output_ready", parity=parity
            ),
        )

    def ring_chunk_offset(self, direction: str, slot: int) -> int:
        if direction not in ("dispatch_tx", "dispatch_rx", "partial_tx", "partial_rx"):
            raise ValueError(f"unknown ring direction {direction!r}")
        if not 0 <= int(slot) < self.ring_depth:
            raise ValueError("ring slot is outside the arena")
        return self.region(direction).offset + int(slot) * self.chunk_bytes

    def ring_qp_offset(self, region: str, slot: int, qp: int) -> int:
        item = self.region(region)
        if item.shape != (self.ring_depth, self.num_qp) or item.dtype != torch.int64:
            raise ValueError(f"{region!r} is not a ring/QP int64 region")
        if not 0 <= int(slot) < self.ring_depth:
            raise ValueError("ring slot is outside the arena")
        if not 0 <= int(qp) < self.num_qp:
            raise ValueError("QP is outside the arena")
        return item.offset + (int(slot) * self.num_qp + int(qp)) * 8


@dataclass(frozen=True)
class HierEpochPointers:
    generation: int
    parity: int
    rank_partial_epoch_ready: int
    plan_ready: int
    h1_input_expected: int
    h1_input_ready: int
    h1_queue_header: int
    h1_ready_queue: int
    h1_output_done: int
    h1_output_ready: int
    h2_output_done: int
    h2_output_ready: int
    rank_route_expected: int
    rank_route_ready: int
    node_partial_ready: int
    fanout_ready: int
    fanout_count: int
    final_output_ready: int


class EpochPhase(IntEnum):
    IDLE = 0
    PLAN = 1
    DISPATCH = 2
    H1 = 3
    H2 = 4
    RETURN = 5
    COMPLETE = 6


@dataclass
class HierEpoch:
    """Checked host sequencing for one arena epoch.

    Device readiness remains authoritative; this class prevents a host launcher
    from accidentally reusing a parity/ring generation out of order.
    """

    layout: HierCcoArenaLayout
    generation: int = 0
    phase: EpochPhase = EpochPhase.IDLE
    last_completed: int = 0
    _ring_generation: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._ring_generation:
            self._ring_generation = [0] * self.layout.ring_depth

    @property
    def parity(self) -> int:
        return self.generation & 1

    @property
    def ring_slot(self) -> int:
        return self.generation % self.layout.ring_depth

    def begin(self, generation: int) -> None:
        generation = int(generation)
        if self.phase not in (EpochPhase.IDLE, EpochPhase.COMPLETE):
            raise RuntimeError("previous hierarchical epoch is still active")
        if generation <= self.last_completed or generation <= 0:
            raise ValueError("generation must be positive and strictly increasing")
        slot = generation % self.layout.ring_depth
        if self._ring_generation[slot] > self.last_completed:
            raise RuntimeError("ring slot has not been credited/retired")
        self.generation = generation
        self.phase = EpochPhase.PLAN

    def advance(self, expected: EpochPhase, new_phase: EpochPhase) -> None:
        if self.phase != expected or int(new_phase) != int(expected) + 1:
            raise RuntimeError(f"invalid epoch transition {self.phase.name}->{new_phase.name}")
        self.phase = new_phase

    def complete(self) -> None:
        if self.phase != EpochPhase.RETURN:
            raise RuntimeError("epoch cannot complete before return/combine")
        self.phase = EpochPhase.COMPLETE
        self.last_completed = self.generation
        self._ring_generation[self.ring_slot] = self.generation


@dataclass
class LayeredHierPipeline:
    """Minimal host contract joining sidecars and compute-only kernels.

    The runtime intentionally owns no CCO object. A communication adapter gets
    ring offsets and epoch pointers from here, while GMM launchers receive only
    the five readiness pointers they consume.
    """

    layout: HierCcoArenaLayout
    arena_base: int
    epoch: HierEpoch = field(init=False)
    pointers: HierEpochPointers | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.epoch = HierEpoch(self.layout)

    def begin(self, generation: int) -> HierEpochPointers:
        self.epoch.begin(generation)
        self.pointers = self.layout.epoch_pointers(self.arena_base, generation)
        return self.pointers

    def dispatch_submitted(self) -> None:
        self.epoch.advance(EpochPhase.PLAN, EpochPhase.DISPATCH)

    def h1_submitted(self) -> None:
        self.epoch.advance(EpochPhase.DISPATCH, EpochPhase.H1)

    def h2_submitted(self) -> None:
        self.epoch.advance(EpochPhase.H1, EpochPhase.H2)

    def return_submitted(self) -> None:
        self.epoch.advance(EpochPhase.H2, EpochPhase.RETURN)

    def finish(self) -> None:
        self.epoch.complete()
