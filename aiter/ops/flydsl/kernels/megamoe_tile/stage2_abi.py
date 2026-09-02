# SPDX-License-Identifier: MIT
"""Internal ABI for the two-kernel EP16 MegaMoE Stage-2 pipeline.

This module intentionally contains no public operator arguments.  The public
contract remains identical to MegaMoE v2::

    x_bf16[local_tokens, hidden], routing_weights[local_tokens, topk],
    topk_ids[local_tokens, topk] -> out_bf16[local_tokens, hidden]

The regions below are an implementation detail shared by the fused Stage-1
and Stage-2 kernels. Stage-1 dispatches each token at most once to a
destination rank and writes the aligned proxy's ``node_expected`` scoreboard.
The Stage-2 weighted GMM2 epilogue directly LSA atomic-adds packed BF16 output
into that proxy's node accumulator. The arena retains FP32-sized capacity for
the diagnostic reference. The default layout has no rank-partial payload and
no eight-rank LSA scan; optional experimental layouts are append-only. The
last route for each hidden tile advances a padded
per-token tile counter; the 28th tile publishes one whole-token ready flag.
The kernel returns at most one BF16 node partial per source token through CCO
and writes the final output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .stage1_abi import MAX_FUSED_TOKENS_PER_RANK, MAX_PACKED_SOURCE_CAPACITY


STAGE2_TIMELINE_FIELDS = (
    "stage1_entry",
    "stage1_dispatch_flush_pre",
    "stage1_dispatch_flush_post",
    "stage1_done_publish",
    "stage2_entry",
    "stage2_stage1_gate_done",
    "stage2_init_gate_done",
    "stage2_qp0_tokens_ready",
    "stage2_qp1_tokens_ready",
    "stage2_qp2_tokens_ready",
    "stage2_qp3_tokens_ready",
    "stage2_first_batch_ready",
    "stage2_qp0_payload_posted",
    "stage2_qp1_payload_posted",
    "stage2_qp2_payload_posted",
    "stage2_qp3_payload_posted",
    "stage2_first_batch_payloads_posted",
    "stage2_return_terminal_posted",
    "stage2_return_flush_pre",
    "stage2_return_flush_post",
    "stage2_return_request_done",
)
STAGE2_TIMELINE_INDEX = {
    name: index for index, name in enumerate(STAGE2_TIMELINE_FIELDS)
}


def _align_up(value: int, alignment: int) -> int:
    value, alignment = int(value), int(alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class Stage2NodePartialWire:
    """Fixed-order CCO return format.

    The RAIL connection is rank aligned.  Consequently record ``token`` is
    already destined for the same local rank on the peer node; no source-rank
    or token header is needed.  Four K3 BF16 rows form one 56-KiB aggregate
    request batch.
    """

    hidden: int = 7168
    records_per_group: int = 4

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.hidden % 256:
            raise ValueError("hidden must be positive and divisible by 256")
        if self.records_per_group <= 0:
            raise ValueError("records_per_group must be positive")
        if self.group_bytes > 64 * 1024:
            raise ValueError("one aggregate group must fit in 64 KiB")

    @property
    def record_bytes(self) -> int:
        return self.hidden * 2

    @property
    def group_bytes(self) -> int:
        return self.records_per_group * self.record_bytes

    def group_count(self, token_count: int) -> int:
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        return (token_count + self.records_per_group - 1) // self.records_per_group


@dataclass(frozen=True)
class Stage2ArenaRegion:
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
class Stage2ArenaLayout:
    """Registered-window layout used by one EP rank.

    ``source_nodes`` is two for the MI355 EP16 deployment.  On local rank ``l``
    the two source planes represent global source ranks ``l`` and ``8+l``.

    Weighted GMM2 output never crosses a rank as a route. For a source rank
    ``s``, every expert rank on the current node atomically adds directly to
    local LSA rank ``s % 8`` at ``node_accumulator[s // 8, token]``. Production
    treats the logical prefix as BF16 while preserving FP32-sized ABI capacity.
    Stage-1
    writes the exact number of contributors into ``node_expected``. The last
    route increments ``node_done`` to that value; the last of 28 completed
    hidden tiles release-publishes ``node_token_ready``. This is the
    direct-accumulator form of the MORI InterNodeV1 combine hierarchy.

    Payloads are parity buffered.  Absolute-generation readiness protects all
    consumer-visible data, so stale payload bytes are never read and payload
    clearing is unnecessary.
    """

    hidden: int
    topk: int
    max_tokens: int
    world_size: int
    gpus_per_node: int
    source_nodes: int
    tile_n: int
    parity_depth: int
    num_qp: int
    records_per_group: int
    rail_quant_type: str
    rail_scale_dim: int
    ready_granularity: str
    ready_group_tiles: int
    include_route_slots: bool
    include_rank_partials: bool
    include_staged_reduce: bool
    # Bounded per-route ring + dedicated reducer experiment.  This is kept
    # append-only after the legacy staged_reduce regions so existing offsets
    # remain stable when the opt-in layout is disabled.
    include_staged_ring: bool
    regions: tuple[Stage2ArenaRegion, ...]
    total_bytes: int

    @property
    def hidden_tiles(self) -> int:
        return self.hidden // self.tile_n

    @property
    def return_groups(self) -> int:
        return (self.max_tokens + self.records_per_group - 1) // self.records_per_group

    @property
    def ready_group_count(self) -> int:
        return (self.hidden_tiles + self.ready_group_tiles - 1) // self.ready_group_tiles

    @classmethod
    def create(
        cls,
        *,
        hidden: int = 7168,
        topk: int = 16,
        max_tokens: int = 128,
        world_size: int = 16,
        gpus_per_node: int = 8,
        source_nodes: int = 2,
        tile_n: int = 256,
        parity_depth: int = 2,
        num_qp: int = 4,
        records_per_group: int = 4,
        rail_quant_type: str = "none",
        rail_scale_dim: int | None = None,
        ready_granularity: str = "token",
        ready_group_tiles: int = 2,
        include_route_slots: bool = False,
        include_rank_partials: bool = False,
        include_staged_reduce: bool = False,
        include_staged_ring: bool = False,
    ) -> "Stage2ArenaLayout":
        values = {
            "hidden": hidden,
            "topk": topk,
            "max_tokens": max_tokens,
            "world_size": world_size,
            "gpus_per_node": gpus_per_node,
            "source_nodes": source_nodes,
            "tile_n": tile_n,
            "parity_depth": parity_depth,
            "num_qp": num_qp,
            "records_per_group": records_per_group,
        }
        if not 1 <= int(max_tokens) <= MAX_FUSED_TOKENS_PER_RANK:
            raise ValueError(
                "max_tokens must be in [1, 4096] for the fused Stage-2 ABI"
            )
        if any(int(v) <= 0 for v in values.values()):
            raise ValueError("all Stage-2 geometry values must be positive")
        if int(world_size) * int(max_tokens) > MAX_PACKED_SOURCE_CAPACITY:
            raise ValueError(
                "world_size * max_tokens exceeds the 24-bit packed source capacity"
            )
        if int(hidden) % int(tile_n):
            raise ValueError("hidden must divide tile_n")
        if int(topk) > 32:
            raise ValueError("node_route_mask is a u32 and supports topk <= 32")
        if int(source_nodes) != 2:
            raise ValueError("the first fused CCO Stage-2 supports exactly two nodes")
        if int(world_size) != int(source_nodes) * int(gpus_per_node):
            raise ValueError("world_size must equal source_nodes * gpus_per_node")
        if int(num_qp) not in (1, 2, 4, 8):
            raise ValueError("num_qp must be one of 1,2,4,8")
        rail_quant_type = str(rail_quant_type)
        if rail_quant_type not in ("none", "fp8_blockwise"):
            raise ValueError("rail_quant_type must be 'none' or 'fp8_blockwise'")
        ready_granularity = str(ready_granularity)
        if ready_granularity not in ("token", "tile"):
            raise ValueError("ready_granularity must be 'token' or 'tile'")
        ready_group_tiles = int(ready_group_tiles)
        if ready_group_tiles <= 0:
            raise ValueError("ready_group_tiles must be positive")
        expected_scale_dim = int(hidden) // 128
        rail_scale_dim = (
            (expected_scale_dim if rail_quant_type == "fp8_blockwise" else 0)
            if rail_scale_dim is None
            else int(rail_scale_dim)
        )
        if rail_quant_type == "none" and rail_scale_dim != 0:
            raise ValueError("rail_scale_dim must be zero when rail quantization is disabled")
        if rail_quant_type == "fp8_blockwise" and rail_scale_dim != expected_scale_dim:
            raise ValueError(
                "fp8_blockwise rail_scale_dim must equal hidden // 128"
            )
        if include_staged_reduce and not include_rank_partials:
            raise ValueError("include_staged_reduce requires include_rank_partials")
        if include_staged_ring and not include_rank_partials:
            raise ValueError("include_staged_ring requires include_rank_partials")
        if include_staged_ring and include_staged_reduce:
            raise ValueError("include_staged_ring and include_staged_reduce are mutually exclusive")
        if int(max_tokens) > 128 and include_staged_reduce:
            raise ValueError("staged_reduce currently supports max_tokens <= 128")
        if int(max_tokens) > 128 and include_staged_ring:
            raise ValueError("staged_ring currently supports max_tokens <= 128")
        wire = Stage2NodePartialWire(int(hidden), int(records_per_group))
        groups = wire.group_count(max_tokens)
        ntiles = int(hidden) // int(tile_n)
        ready_groups = (ntiles + ready_group_tiles - 1) // ready_group_tiles
        if ready_granularity == "tile" and ready_groups > 64:
            raise ValueError("ready_group_count exceeds the 64-bit node_ready_mask")

        # name, shape, dtype, alignment. node_expected/masks are Stage-1
        # outputs. Stage-2 clears accumulator/done in its resident all-CTA
        # initialization phase before opening the compute gate.
        specs: list[tuple[str, tuple[int, ...], torch.dtype, int]] = [
            (
                "node_dest_rank_mask",
                (parity_depth, source_nodes, max_tokens),
                torch.int32,
                64,
            ),
            (
                "source_token_count",
                (parity_depth, source_nodes),
                torch.int32,
                64,
            ),
            (
                "node_expected",
                (parity_depth, source_nodes, max_tokens, ntiles),
                torch.int32,
                64,
            ),
            (
                "node_accumulator",
                (parity_depth, source_nodes, max_tokens, hidden),
                torch.float32,
                256,
            ),
            (
                "node_done",
                (parity_depth, source_nodes, max_tokens, ntiles),
                torch.int32,
                64,
            ),
            # Second-level completion for whole-token readiness.
            # The last route for each hidden tile increments
            # node_token_done; the 28th completed tile publishes one token flag.
            (
                "node_token_done",
                # One 64-byte cache line per token avoids false sharing among
                # tile-completion atomics on MI355. Only lane zero is logical.
                (parity_depth, source_nodes, max_tokens, 16),
                torch.int32,
                64,
            ),
            (
                "node_token_ready",
                (parity_depth, source_nodes, max_tokens),
                torch.int64,
                64,
            ),
            (
                "remote_node_tx",
                (parity_depth, max_tokens, hidden),
                torch.bfloat16,
                256,
            ),
            # The peer RAIL rank writes its node partial for this rank's local
            # source tokens directly here, in token order.
            (
                "remote_partial_rx",
                (parity_depth, max_tokens, hidden),
                torch.bfloat16,
                256,
            ),
            (
                "return_group_ready",
                (parity_depth, groups),
                torch.int64,
                64,
            ),
            # Per-QP producer/coordinator epochs for the diagnostic
            # independent-return schedule. They are unused by lockstep.
            ("return_count", (parity_depth, num_qp), torch.int64, 64),
            ("return_count_ready", (parity_depth, num_qp), torch.int64, 64),
            ("return_consumed", (parity_depth,), torch.int64, 64),
            # Stage-1 completion and Stage-2 init are LSA-visible node gates.
            ("stage1_done", (parity_depth,), torch.int64, 64),
            ("stage2_init", (parity_depth,), torch.int64, 64),
            # Device wall-clock samples used only by timeline-instrumented
            # diagnostic builds. Production kernels compile all stores out.
            (
                "timeline",
                (parity_depth, len(STAGE2_TIMELINE_FIELDS)),
                torch.int64,
                64,
            ),
            (
                "timeline_gmm_worker_done",
                (parity_depth, 256),
                torch.int64,
                64,
            ),
            # Persistent launch state. Each work-head shard owns a separate
            # 64-byte line. grid_barrier is monotonic and therefore needs no
            # hot-path memset when the resident grid geometry is unchanged.
            ("grid_barrier", (1,), torch.int64, 64),
            ("gemm_work_head", (8, 16), torch.int32, 64),
            ("final_work_head", (16,), torch.int32, 64),
            ("final_done", (1,), torch.int32, 64),
            # Stage-1 may already have published errors into its own region;
            # Stage-2 keeps a separate counter so its in-kernel reset cannot
            # erase dispatch diagnostics.
            ("stage2_error_count", (1,), torch.int32, 64),
        ]

        # Optional store/reduce ABIs. Keeping these regions at the end makes
        # the default direct-atomic layout byte-for-byte identical and also
        # preserves every existing region offset when the experiment is on.
        # Within one token, a reducer can load all route slots for one N tile
        # contiguously before advancing to the next tile.
        if include_route_slots:
            specs.append(
                (
                    "route_slots",
                    (
                        parity_depth,
                        source_nodes,
                        max_tokens,
                        ntiles,
                        topk,
                        tile_n,
                    ),
                    torch.bfloat16,
                    256,
                )
            )

        # The rank-local experiment first combines all routes owned by one EP
        # rank in local memory. Its token-ready publication then gates the
        # aligned source proxy's node-local peer pull. The logical source
        # dimension is the complete EP world because every expert rank can
        # receive routes from every source rank.
        if include_rank_partials:
            specs.extend(
                (
                    (
                        "rank_accumulator",
                        (parity_depth, world_size, max_tokens, hidden),
                        torch.bfloat16,
                        256,
                    ),
                    (
                        "rank_token_pending",
                        (parity_depth, world_size, max_tokens, 16),
                        torch.int32,
                        64,
                    ),
                    (
                        "rank_token_ready",
                        (parity_depth, world_size, max_tokens),
                        torch.int64,
                        64,
                    ),
                    (
                        "rank_return_tx_slot",
                        (parity_depth, max_tokens),
                        torch.int32,
                        64,
                    ),
                    (
                        "rank_return_rx_slot",
                        (parity_depth, max_tokens),
                        torch.int32,
                        64,
                    ),
                    (
                        "rank_return_count",
                        # lane 0 = compact TX rows, lane 1 = compact RX rows,
                        # lane 2 = active local+remote plane-token reducers.
                        (parity_depth, 16),
                        torch.int32,
                        64,
                    ),
                    (
                        "rank_reduce_queue",
                        (parity_depth, source_nodes * max_tokens),
                        torch.int32,
                        64,
                    ),
                    (
                        "rank_reduce_queue_ready",
                        (parity_depth, source_nodes * max_tokens),
                        torch.int64,
                        64,
                    ),
                    (
                        "rank_reduce_queue_tail",
                        # One logical counter padded to a cache line.
                        (parity_depth, 16),
                        torch.int32,
                        64,
                    ),
                )
            )

        # Both deferred-reduction experiments converge on the same source-proxy
        # completion protocol. Keep these shared regions single-instanced even
        # if a diagnostic layout requests both optional payload families.
        if include_route_slots or include_rank_partials:
            specs.extend(
                (
                    (
                        "node_partial_done",
                        # One cache line per token avoids false sharing among
                        # independent tile/rank arrivals.
                        (parity_depth, source_nodes, max_tokens, 16),
                        torch.int32,
                        64,
                    ),
                    (
                        "node_partial_ready",
                        (parity_depth, source_nodes, max_tokens),
                        torch.int64,
                        64,
                    ),
                )
            )

        # Keep the dynamic reducer cursor last.  In particular, append it after
        # the shared completion regions above so enabling the new scheduler
        # does not move any pre-existing rank-partial ABI offset.
        if include_rank_partials:
            specs.append(
                (
                    "rank_reduce_queue_head",
                    # One parity-local logical counter padded to a cache line.
                    (parity_depth, 16),
                    torch.int32,
                    64,
                )
            )
            if include_staged_reduce:
                # Optional staged rank reduction payload.  These regions are
                # appended after the existing rank-local ABI so atomic mode
                # and every pre-existing offset remain unchanged.
                specs.extend(
                    (
                        (
                            "rank_stage_values",
                            (
                                parity_depth,
                                world_size * max_tokens,
                                topk,
                                hidden // (2 * tile_n),
                                2 * tile_n,
                            ),
                            torch.bfloat16,
                            256,
                        ),
                        (
                            "rank_stage_slot_generation",
                            (
                                parity_depth,
                                world_size * max_tokens,
                                topk,
                                hidden // tile_n,
                            ),
                            torch.int64,
                            64,
                        ),
                        (
                            "rank_stage_group_pending",
                            (parity_depth, world_size * max_tokens, hidden // (2 * tile_n)),
                            torch.int32,
                            64,
                        ),
                        (
                            "rank_stage_tile_pending",
                            (parity_depth, world_size * max_tokens, hidden // tile_n),
                            torch.int32,
                            64,
                        ),
                        (
                            "rank_stage_tile_done",
                            (parity_depth, world_size * max_tokens, hidden // tile_n),
                            torch.int32,
                            64,
                        ),
                    )
                )

            if include_staged_ring:
                # A compact 4-MiB/parity bounded MPMC ring.  Each slot carries one
                # route x BN=256 tile (512 B BF16 payload); metadata and
                # sequence words are separate cache-line padded arrays.
                # Compact 4-MiB/parity payload ring.  The reducer drains while
                # producers run; this keeps the registered CCO window below
                # the ~300-MiB allocation ceiling on MI355.
                ring_slots = (4 * 1024 * 1024) // (2 * tile_n)
                ring_groups = hidden // (2 * tile_n)
                specs.extend(
                    (
                        (
                            "rank_stage_ring_payload",
                            (parity_depth, ring_slots, tile_n),
                            torch.bfloat16,
                            256,
                        ),
                        (
                            "rank_stage_ring_source",
                            (parity_depth, ring_slots),
                            torch.int32,
                            64,
                        ),
                        (
                            "rank_stage_ring_slot",
                            (parity_depth, ring_slots),
                            torch.int16,
                            64,
                        ),
                        (
                            "rank_stage_ring_tile",
                            (parity_depth, ring_slots),
                            torch.int16,
                            64,
                        ),
                        (
                            "rank_stage_ring_sequence",
                            (parity_depth, ring_slots),
                            torch.int64,
                            64,
                        ),
                        (
                            "rank_stage_ring_head",
                            (parity_depth, 16),
                            torch.int64,
                            64,
                        ),
                        (
                            "rank_stage_ring_tail",
                            (parity_depth, 16),
                            torch.int64,
                            64,
                        ),
                        (
                            # Consumer claim cursor is distinct from the
                            # release/free cursor above.  Producers use tail
                            # for occupancy; the reducer advances claim before
                            # reading and tail only after payload completion.
                            "rank_stage_ring_claim",
                            (parity_depth, 16),
                            torch.int64,
                            64,
                        ),
                        (
                            "rank_stage_ring_reserve_lock",
                            (parity_depth, 16),
                            torch.int32,
                            64,
                        ),
                        (
                            "rank_stage_ring_producer_done",
                            (parity_depth, 16),
                            torch.int32,
                            64,
                        ),
                        (
                            "rank_stage_ring_reducer_done",
                            (parity_depth, 16),
                            torch.int32,
                            64,
                        ),
                        (
                            "rank_stage_ring_scratch",
                            (
                                parity_depth,
                                world_size * max_tokens,
                                ring_groups,
                                2,
                                tile_n,
                            ),
                            # BF16 scratch halves the registered-window
                            # footprint. Reducer widens each element to FP32
                            # for the add, then rounds back to BF16.
                            torch.bfloat16,
                            256,
                        ),
                        (
                            "rank_stage_ring_seen",
                            (
                                parity_depth,
                                world_size * max_tokens,
                                ring_groups,
                                2,
                            ),
                            torch.int32,
                            64,
                        ),
                    )
                )

        # MORI's Stage2 fp8_blockwise combine uses one FP32 scale per 128
        # hidden values: H7168 therefore has scale_dim=56.  Append after every
        # optional legacy ABI so enabling quantization changes no old offset.
        if rail_quant_type == "fp8_blockwise":
            specs.extend(
                (
                    (
                        "rail_fp8_tx_payload",
                        (parity_depth, max_tokens, hidden),
                        torch.uint8,
                        # Start beyond the complete legacy arena, including
                        # its final 4-KiB rounding, to keep append-only ABI.
                        4096,
                    ),
                    (
                        "rail_fp8_rx_payload",
                        (parity_depth, max_tokens, hidden),
                        torch.uint8,
                        256,
                    ),
                    (
                        "rail_fp8_tx_scale",
                        (parity_depth, max_tokens, rail_scale_dim),
                        torch.float32,
                        256,
                    ),
                    (
                        "rail_fp8_rx_scale",
                        (parity_depth, max_tokens, rail_scale_dim),
                        torch.float32,
                        256,
                    ),
                )
            )

        if ready_granularity == "tile":
            tile_entries = source_nodes * max_tokens * ready_groups
            specs.extend(
                (
                    (
                        "rank_tile_pending",
                        (parity_depth, world_size * max_tokens, ready_groups),
                        torch.int32,
                        64,
                    ),
                    (
                        "rank_tile_ready",
                        (parity_depth, world_size * max_tokens, ready_groups),
                        torch.int64,
                        64,
                    ),
                    (
                        "node_tile_arrived",
                        (parity_depth, source_nodes * max_tokens, ready_groups),
                        torch.int32,
                        64,
                    ),
                    (
                        "node_tile_ready",
                        (parity_depth, source_nodes * max_tokens, ready_groups),
                        torch.int64,
                        64,
                    ),
                    (
                        "node_ready_mask",
                        (parity_depth, source_nodes * max_tokens),
                        torch.int64,
                        64,
                    ),
                    (
                        "tile_reduce_queue",
                        (parity_depth, tile_entries),
                        torch.int32,
                        64,
                    ),
                    (
                        "tile_reduce_queue_ready",
                        (parity_depth, tile_entries),
                        torch.int64,
                        64,
                    ),
                    (
                        "tile_reduce_queue_tail",
                        (parity_depth, 16),
                        torch.int32,
                        64,
                    ),
                    (
                        "tile_reduce_queue_head",
                        (parity_depth, 16),
                        torch.int32,
                        64,
                    ),
                )
            )

        offset = 0
        regions: list[Stage2ArenaRegion] = []
        for name, shape, dtype, alignment in specs:
            offset = _align_up(offset, alignment)
            numel = 1
            for dim in shape:
                numel *= int(dim)
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            regions.append(
                Stage2ArenaRegion(
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
            hidden=int(hidden),
            topk=int(topk),
            max_tokens=int(max_tokens),
            world_size=int(world_size),
            gpus_per_node=int(gpus_per_node),
            source_nodes=int(source_nodes),
            tile_n=int(tile_n),
            parity_depth=int(parity_depth),
            num_qp=int(num_qp),
            records_per_group=int(records_per_group),
            rail_quant_type=rail_quant_type,
            rail_scale_dim=rail_scale_dim,
            ready_granularity=ready_granularity,
            ready_group_tiles=ready_group_tiles,
            include_route_slots=bool(include_route_slots),
            include_rank_partials=bool(include_rank_partials),
            include_staged_reduce=bool(include_staged_reduce),
            include_staged_ring=bool(include_staged_ring),
            regions=tuple(regions),
            total_bytes=_align_up(offset, 4096),
        )

    def region(self, name: str) -> Stage2ArenaRegion:
        for item in self.regions:
            if item.name == name:
                return item
        raise KeyError(name)

    def offset(self, name: str, *, parity: int | None = None) -> int:
        item = self.region(name)
        off = item.offset
        if parity is not None:
            if not 0 <= int(parity) < self.parity_depth:
                raise ValueError("parity is outside parity_depth")
            if not item.shape or item.shape[0] != self.parity_depth:
                raise ValueError(f"{name} is not parity indexed")
            off += int(parity) * (item.nbytes // self.parity_depth)
        return off

    def allocate_local(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(self.total_bytes, dtype=torch.uint8, device=device)


def validate_public_stage2_contract(
    routing_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    hidden: int = 7168,
    topk: int = 16,
    max_tokens: int = 128,
) -> int:
    """Validate the public-facing portion of a fused Stage-2 call.

    Source-rank, source-token, top-k-slot, route masks, and node partials are
    deliberately absent: they are internal products of fused Stage-1.
    """

    if routing_weights.ndim != 2 or routing_weights.shape[1] != topk:
        raise ValueError("routing_weights must be [local_tokens, topk]")
    local_tokens = int(routing_weights.shape[0])
    if not 0 <= local_tokens <= int(max_tokens):
        raise ValueError("local token count exceeds max_tokens")
    if routing_weights.dtype != torch.float32 or not routing_weights.is_contiguous():
        raise ValueError("routing_weights must be contiguous float32")
    if topk_ids.shape != routing_weights.shape:
        raise ValueError("topk_ids shape must match routing_weights")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous int32")
    if output.shape != (local_tokens, int(hidden)):
        raise ValueError("output must be [local_tokens, hidden]")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("output must be contiguous bfloat16")
    return local_tokens


__all__ = [
    "STAGE2_TIMELINE_FIELDS",
    "STAGE2_TIMELINE_INDEX",
    "Stage2ArenaLayout",
    "Stage2ArenaRegion",
    "Stage2NodePartialWire",
    "validate_public_stage2_contract",
]
