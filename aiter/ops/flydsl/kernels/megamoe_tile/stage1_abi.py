# SPDX-License-Identifier: MIT
"""Internal ABI for the fused EP16 A4W4 MegaMoE Stage-1 kernel.

The public operator deliberately exposes none of this state.  One Stage-1
launch consumes local BF16 tokens plus routing metadata and leaves expert-tile
major A4 output for Stage-2 in this registered CCO window.

The dispatch inbox follows MORI InterNodeV1 ownership rules: a token is copied
at most once to a selected rank, and at most once across the network to the
aligned proxy of a selected node.  The destination rank expands its locally
owned expert routes only after receiving the rank-deduplicated record.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SPARSE_QP_TOKEN_BITS = 32
SPARSE_QP_GENERATION_SHIFT = 32


def _align_up(value: int, alignment: int) -> int:
    value, alignment = int(value), int(alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class Stage1DispatchWire:
    """One node-deduplicated token record.

    K3 uses exactly 4096 bytes: an MXFP4 activation row, raw per-1x32 E8M0
    scales, complete top-k IDs/weights, source-token identity, and a top-k slot
    mask plus one 16-bit top-k-slot bitmap for every EP rank.  Complete top-k
    metadata is intentional: the aligned proxy needs it to select destination
    ranks, and each destination rank expands every set slot in its bitmap into
    an independent expert route.  The bitmap popcount is the per-rank route
    multiplicity, so the record does not need a redundant count or expert list.
    """

    hidden: int = 7168
    topk: int = 16
    record_alignment: int = 256

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.hidden % 128:
            raise ValueError("hidden must be positive and divisible by 128")
        if not 0 < self.topk <= 16:
            raise ValueError("topk must be in [1, 16] for the u16 rank-slot ABI")
        if self.record_bytes != 4096:
            raise ValueError(
                "the first fused EP16 Stage-1 requires the K3 4096-byte record"
            )

    @property
    def payload_bytes(self) -> int:
        return self.hidden // 2

    @property
    def scale_bytes(self) -> int:
        return self.hidden // 32

    @property
    def ids_offset(self) -> int:
        return self.payload_bytes + self.scale_bytes

    @property
    def weights_offset(self) -> int:
        return self.ids_offset + self.topk * 4

    @property
    def source_offset(self) -> int:
        return self.weights_offset + self.topk * 4

    @property
    def route_mask_offset(self) -> int:
        return self.source_offset + 8

    @property
    def rank_slot_masks_offset(self) -> int:
        return self.route_mask_offset + 8

    @property
    def rank_slot_masks_bytes(self) -> int:
        # The fused path is specialized to EP16.  One u16 per global rank maps
        # every top-k slot to its owner while preserving duplicate-rank routes.
        return 16 * 2

    @property
    def raw_bytes(self) -> int:
        return self.rank_slot_masks_offset + self.rank_slot_masks_bytes

    @property
    def record_bytes(self) -> int:
        return _align_up(self.raw_bytes, self.record_alignment)

    @property
    def records_per_chunk(self) -> int:
        return (64 * 1024) // self.record_bytes


@dataclass(frozen=True)
class Stage1ArenaRegion:
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
class Stage1ArenaLayout:
    """Registered-window layout for one rank's fused Stage-1 launch.

    The first implementation specializes the production K3 deployment.  A
    fixed rank inbox slot ``source_rank * max_tokens + source_token`` removes a
    device allocator from the rank-deduplicated dispatch path.  Expert routes
    are compacted independently into dynamically allocated BM tiles; physical
    tiles may interleave experts, but every tile is single-expert and therefore
    directly consumable by the existing A4W4 GMM bodies.
    """

    hidden: int
    inter: int
    experts: int
    world_size: int
    gpus_per_node: int
    topk: int
    max_tokens: int
    block_m: int
    block_n: int
    parity_depth: int
    num_qp: int
    regions: tuple[Stage1ArenaRegion, ...]
    total_bytes: int

    @property
    def local_experts(self) -> int:
        return self.experts // self.world_size

    @property
    def source_capacity(self) -> int:
        return self.world_size * self.max_tokens

    @property
    def route_capacity(self) -> int:
        # Arbitrary Top-K may place all routes on one destination rank.
        return self.source_capacity * self.topk

    @property
    def max_tiles_per_expert(self) -> int:
        # Also tolerate repeated exact expert IDs.  Normal top-k output is
        # unique, but sizing the small map for the full route capacity avoids
        # turning that assumption into an unchecked memory-safety contract.
        return (self.route_capacity + self.block_m - 1) // self.block_m

    @property
    def max_route_tiles(self) -> int:
        # Splitting route_capacity rows over E_local experts introduces at most
        # E_local-1 additional partially occupied tiles.  Keep one extra tile
        # as the existing conservative bound does.
        return (
            (self.route_capacity + self.block_m - 1) // self.block_m
            + self.local_experts
        )

    @property
    def max_route_rows(self) -> int:
        return self.max_route_tiles * self.block_m

    @property
    def h1_n_blocks(self) -> int:
        return (2 * self.inter) // self.block_n

    @property
    def dispatch_chunks(self) -> int:
        return (self.max_tokens + self.wire.records_per_chunk - 1) // self.wire.records_per_chunk

    @property
    def wire(self) -> Stage1DispatchWire:
        return Stage1DispatchWire(self.hidden, self.topk)

    @classmethod
    def create(
        cls,
        *,
        hidden: int = 7168,
        inter: int = 3072,
        experts: int = 896,
        world_size: int = 16,
        gpus_per_node: int = 8,
        topk: int = 16,
        max_tokens: int = 128,
        block_m: int = 32,
        block_n: int = 256,
        parity_depth: int = 2,
        num_qp: int = 4,
    ) -> "Stage1ArenaLayout":
        expected = {
            "hidden": 7168,
            "inter": 3072,
            "experts": 896,
            "world_size": 16,
            "gpus_per_node": 8,
            "topk": 16,
            "max_tokens": 128,
            "block_m": 32,
            "block_n": 256,
            "parity_depth": 2,
            "num_qp": 4,
        }
        actual = {
            "hidden": hidden,
            "inter": inter,
            "experts": experts,
            "world_size": world_size,
            "gpus_per_node": gpus_per_node,
            "topk": topk,
            "max_tokens": max_tokens,
            "block_m": block_m,
            "block_n": block_n,
            "parity_depth": parity_depth,
            "num_qp": num_qp,
        }
        bad = {k: (int(actual[k]), v) for k, v in expected.items() if int(actual[k]) != v}
        if bad:
            details = ", ".join(f"{k}={got} (expected {want})" for k, (got, want) in bad.items())
            raise ValueError(f"fused Stage-1 currently supports only the K3 EP16 shape: {details}")

        hidden = int(hidden)
        inter = int(inter)
        experts = int(experts)
        world_size = int(world_size)
        gpus_per_node = int(gpus_per_node)
        topk = int(topk)
        max_tokens = int(max_tokens)
        block_m = int(block_m)
        block_n = int(block_n)
        parity_depth = int(parity_depth)
        num_qp = int(num_qp)
        local_experts = experts // world_size
        source_capacity = world_size * max_tokens
        route_capacity = source_capacity * topk
        max_route_tiles = (route_capacity + block_m - 1) // block_m + local_experts
        max_route_rows = max_route_tiles * block_m
        max_tiles_per_expert = (route_capacity + block_m - 1) // block_m
        n_blocks = (2 * inter) // block_n
        wire = Stage1DispatchWire(hidden, topk)
        chunks = (max_tokens + wire.records_per_chunk - 1) // wire.records_per_chunk
        input_scale_bytes = hidden // 32
        output_scale_bytes = inter // 32

        # All payload/state that is remotely addressed through CCO or LSA lives
        # in this single registered window.  Generation arrays are absolute and
        # parity buffered; payload clearing between hot forwards is unnecessary.
        specs: list[tuple[str, tuple[int, ...], torch.dtype, int]] = [
            ("dispatch_staging", (parity_depth, max_tokens, wire.record_bytes), torch.uint8, 256),
            ("dispatch_staging_ready", (parity_depth, max_tokens), torch.int64, 64),
            ("remote_dispatch_rx", (parity_depth, max_tokens, wire.record_bytes), torch.uint8, 256),
            ("remote_chunk_ready", (parity_depth, chunks, num_qp), torch.int64, 64),
            ("remote_chunk_credit", (parity_depth, chunks, num_qp), torch.int64, 64),
            ("remote_chunk_request", (parity_depth, chunks, num_qp), torch.int64, 64),
            (
                "remote_chunk_consumed",
                (parity_depth, chunks, num_qp),
                torch.int32,
                64,
            ),
            # Eight source-local communication roles write expert tiles on each
            # destination rank.  One EOS slot covers the aligned pair of source
            # ranks (local node + remote node), hence exactly eight EOS values.
            ("comm_eos", (parity_depth, gpus_per_node), torch.int64, 64),
            ("launch_ready", (parity_depth,), torch.int64, 64),
            ("expert_count", (parity_depth, local_experts), torch.int32, 64),
            (
                "expert_tile_map",
                (parity_depth, local_experts, max_tiles_per_expert),
                torch.int32,
                64,
            ),
            (
                "expert_tile_map_ready",
                (parity_depth, local_experts, max_tiles_per_expert),
                torch.int64,
                64,
            ),
            ("tile_alloc", (parity_depth,), torch.int32, 64),
            ("tile_row_done", (parity_depth, max_route_tiles), torch.int32, 64),
            ("tile_expert", (parity_depth, max_route_tiles), torch.int32, 64),
            ("tile_row_base", (parity_depth, max_route_tiles), torch.int32, 64),
            ("num_valid", (parity_depth,), torch.int32, 64),
            # Quantized A is stored once per global source token on each
            # selected destination rank.  tile_row_input maps every grouped
            # expert route back to that shared source row; scales remain in
            # grouped BM32 layout for the existing GMM1 scale loader.
            ("tile_row_input", (parity_depth, max_route_rows), torch.int32, 64),
            ("tile_row_source", (parity_depth, max_route_rows), torch.int32, 64),
            ("tile_row_weight", (parity_depth, max_route_rows), torch.float32, 64),
            (
                "grouped_input_q",
                (parity_depth, source_capacity, hidden // 2),
                torch.uint8,
                256,
            ),
            (
                "grouped_input_scale",
                (parity_depth, max_route_rows * input_scale_bytes),
                torch.uint8,
                256,
            ),
            (
                "h1_output_q",
                (parity_depth, max_route_rows, inter // 2),
                torch.uint8,
                256,
            ),
            (
                "h1_output_scale",
                (parity_depth, max_route_rows * output_scale_bytes),
                torch.uint8,
                256,
            ),
            (
                "h1_ready_queue",
                (parity_depth, max_route_tiles * n_blocks),
                torch.int32,
                64,
            ),
            (
                "h1_ready_queue_generation",
                (parity_depth, max_route_tiles * n_blocks),
                torch.int64,
                64,
            ),
            # In tile-pipeline mode each contiguous n_blocks batch uses its
            # first generation word as the release-published ready marker.
            # Eight independent consumer heads occupy separate cache lines.
            ("h1_queue_head", (parity_depth, 8, 16), torch.int32, 64),
            ("h1_queue_tail", (parity_depth,), torch.int32, 64),
            ("h1_queue_eos", (parity_depth,), torch.int64, 64),
            ("h1_compute_done", (parity_depth,), torch.int32, 64),
            ("h1_tile_done", (parity_depth, max_route_tiles), torch.int32, 64),
            # Persistent launch state is intentionally not parity indexed.
            ("entry_count", (1,), torch.int64, 64),
            ("epoch_gate", (1,), torch.int64, 64),
            ("error_count", (1,), torch.int32, 64),
            # Diagnostic split-fanout completion flags. Appended so every
            # existing production region keeps its byte offset unchanged.
            ("fanout_done", (parity_depth, 8, 32), torch.int64, 64),
            # Split diagnostic: second quant CTA publishes one generation flag
            # per token before the metadata-owning first CTA marks staging ready.
            ("quant_half_done", (parity_depth, max_tokens), torch.int64, 64),
            # Sparse multi-CTA transport. Producer CTAs post token data WQEs
            # without ringing and leave one local 0/1 decision per token in
            # sparse_remote_token_ready.  The fixed CCO CTA ballots 32 decisions
            # into each QP's terminal ready word and flushes that QP once.
            ("sparse_remote_token_ready", (parity_depth, max_tokens), torch.int64, 64),
            ("sparse_remote_qp_ready", (parity_depth, num_qp), torch.int64, 64),
            ("sparse_remote_request", (parity_depth, num_qp), torch.int64, 64),
            ("sparse_remote_batch_ready", (parity_depth,), torch.int64, 64),
            ("sparse_remote_credit", (parity_depth,), torch.int64, 64),
            ("sparse_remote_consumed", (parity_depth,), torch.int32, 64),
            ("sparse_remote_send_count", (parity_depth,), torch.int32, 64),
            # Optional correctness-only instrumentation for the sparse
            # tile-ready pipeline.  Appending these preserves every pre-v3
            # region offset. Production builds leave the counters at zero so
            # they add no per-tile/job atomic traffic to steady runs.
            ("h1_early_full_tiles", (parity_depth,), torch.int32, 64),
            ("h1_gmm_started_before_all_comm_eos", (parity_depth,), torch.int32, 64),
            ("h1_gmm_completed_before_all_comm_eos", (parity_depth,), torch.int32, 64),
        ]

        offset = 0
        regions: list[Stage1ArenaRegion] = []
        for name, shape, dtype, alignment in specs:
            offset = _align_up(offset, alignment)
            numel = 1
            for dim in shape:
                numel *= int(dim)
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            regions.append(
                Stage1ArenaRegion(
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    alignment=alignment,
                    shape=tuple(int(v) for v in shape),
                    dtype=dtype,
                )
            )
            offset += nbytes

        return cls(
            hidden=hidden,
            inter=inter,
            experts=experts,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            topk=topk,
            max_tokens=max_tokens,
            block_m=block_m,
            block_n=block_n,
            parity_depth=parity_depth,
            num_qp=num_qp,
            regions=tuple(regions),
            total_bytes=_align_up(offset, 4096),
        )

    def region(self, name: str) -> Stage1ArenaRegion:
        for item in self.regions:
            if item.name == name:
                return item
        raise KeyError(name)

    def offset(self, name: str, *, parity: int | None = None) -> int:
        item = self.region(name)
        offset = item.offset
        if parity is not None:
            if not 0 <= int(parity) < self.parity_depth:
                raise ValueError("parity is outside parity_depth")
            if not item.shape or item.shape[0] != self.parity_depth:
                raise ValueError(f"{name} is not parity indexed")
            offset += int(parity) * (item.nbytes // self.parity_depth)
        return offset

    def pointer(self, base: int, name: str, *, parity: int | None = None) -> int:
        return int(base) + self.offset(name, parity=parity)

    def allocate_local(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(self.total_bytes, dtype=torch.uint8, device=device)


@dataclass(frozen=True)
class TwoKernelArenaLayout:
    """Compose the logical Stage-1 and Stage-2 ABIs in one CCO window."""

    stage1: Stage1ArenaLayout
    stage2: object
    stage2_offset: int
    total_bytes: int

    @classmethod
    def compose(
        cls, stage1: Stage1ArenaLayout, stage2: object, *, alignment: int = 4096
    ) -> "TwoKernelArenaLayout":
        if not hasattr(stage2, "total_bytes"):
            raise TypeError("stage2 layout must expose total_bytes")
        stage2_offset = _align_up(stage1.total_bytes, alignment)
        total_bytes = _align_up(stage2_offset + int(stage2.total_bytes), alignment)
        return cls(stage1, stage2, stage2_offset, total_bytes)

    def allocate_local(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(self.total_bytes, dtype=torch.uint8, device=device)


def validate_public_stage1_contract(
    x_bf16: torch.Tensor,
    routing_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden: int = 7168,
    topk: int = 16,
    max_tokens: int = 128,
) -> int:
    """Validate the only public inputs consumed by fused Stage-1."""

    if x_bf16.ndim != 2 or x_bf16.shape[1] != int(hidden):
        raise ValueError("x_bf16 must be [local_tokens, hidden]")
    tokens = int(x_bf16.shape[0])
    if not 0 <= tokens <= int(max_tokens):
        raise ValueError("local token count exceeds max_tokens")
    if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
        raise ValueError("x_bf16 must be contiguous bfloat16")
    expected = (tokens, int(topk))
    if tuple(routing_weights.shape) != expected:
        raise ValueError("routing_weights must be [local_tokens, topk]")
    if routing_weights.dtype != torch.float32 or not routing_weights.is_contiguous():
        raise ValueError("routing_weights must be contiguous float32")
    if tuple(topk_ids.shape) != expected:
        raise ValueError("topk_ids shape must match routing_weights")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous int32")
    return tokens


__all__ = [
    "SPARSE_QP_GENERATION_SHIFT",
    "SPARSE_QP_TOKEN_BITS",
    "Stage1ArenaLayout",
    "Stage1ArenaRegion",
    "Stage1DispatchWire",
    "TwoKernelArenaLayout",
    "validate_public_stage1_contract",
]
