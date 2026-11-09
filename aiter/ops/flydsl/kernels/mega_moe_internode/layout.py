# SPDX-License-Identifier: MIT
"""Registered-window layout of MegaMoEInternode: the Stage1 arena
(``Stage1ArenaLayout``), the Stage2 regions embedded in it (``Stage2ArenaLayout``)
and the two composed into one window (``TwoKernelArenaLayout``).

Stage1:

The public operator deliberately exposes none of this state.  One Stage-1
launch consumes local BF16 tokens plus routing metadata and leaves expert-tile
major A4 output for Stage-2 in this registered CCO window.

The dispatch inbox follows MORI InterNodeV1 ownership rules: a token is copied
at most once to a selected rank, and at most once across the network to the
aligned proxy of a selected node.  The destination rank expands its locally
owned expert routes only after receiving the rank-deduplicated record.

Stage2:

Stage-1 writes the per-token top-k slot bitmap (``node_dest_slot_mask``)
here.  Stage-2 kernel1
(GEMM2 + node push) writes weighted BF16 rows into the node-local
``plane_slot_inbox`` of the proxy GPU and publishes ``plane_slot_arrived``;
kernel2 reduces them into ``node_accumulator``, ships the remote plane over
the rail into the peer's ``remote_partial_rx`` and combines both planes into
the source-order output.  Every region is double buffered by generation
parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SPARSE_QP_TOKEN_BITS = 32
SPARSE_QP_GENERATION_SHIFT = 32
MAX_FUSED_TOKENS_PER_RANK = 16384
MAX_PACKED_SOURCE_CAPACITY = (1 << 24) - 1
MAX_BUFFER_ADDRESSABLE_BYTES = 1 << 32


def _align_up(value: int, alignment: int) -> int:
    value, alignment = int(value), int(alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def validate_fused_weight_addressing(
    *, hidden: int, inter: int, experts: int, world_size: int
) -> None:
    """Reject model shapes whose per-rank weight buffers exceed i32 offsets."""

    hidden = int(hidden)
    inter = int(inter)
    experts = int(experts)
    world_size = int(world_size)
    if world_size <= 0 or experts <= 0 or experts % world_size:
        raise ValueError("experts must be positive and divisible by world_size")
    local_experts = experts // world_size
    # Sparse Stage1 packs local_expert in eight bits and its finisher assigns
    # one expert to each thread of a 256-thread CTA.
    if local_experts > 256:
        raise ValueError("experts/world_size must be <= 256")
    spans = {
        "w1": local_experts * inter * hidden,
        "w1_scale": local_experts * (2 * inter) * (hidden // 32),
        "w2": local_experts * hidden * inter // 2,
        "w2_scale": local_experts * hidden * (inter // 32),
    }
    oversized = {
        name: nbytes
        for name, nbytes in spans.items()
        if nbytes >= MAX_BUFFER_ADDRESSABLE_BYTES
    }
    if oversized:
        detail = ", ".join(f"{name}={nbytes}" for name, nbytes in oversized.items())
        raise ValueError(
            "per-rank weight buffers must each be smaller than 4 GiB for the "
            f"32-bit buffer-offset ABI: {detail}"
        )


@dataclass(frozen=True)
class Stage1DispatchWire:
    """One node-deduplicated token record.

    The default H7168/TopK16 EP16 shape uses exactly 4096 bytes. Each record
    carries an MXFP4 activation row, raw per-1x32 E8M0 scales, complete top-k
    IDs/weights, source-token identity, a top-k slot mask, and one 16-bit
    top-k-slot bitmap for every EP rank. Complete top-k metadata is intentional:
    the aligned proxy needs it to select destination ranks, and each destination
    rank expands every set slot in its bitmap into an independent expert route.
    The bitmap popcount is the per-rank route multiplicity, so the record does
    not need a redundant count or expert list.
    """

    hidden: int = 7168
    topk: int = 16
    record_alignment: int = 256
    world_size: int = 16
    num_qp: int = 4
    max_tokens: int = 0
    activation_dtype: str = "fp4"

    def __post_init__(self) -> None:
        if self.activation_dtype not in ("fp4", "fp8"):
            raise ValueError("activation_dtype must be fp4 or fp8")
        if self.hidden <= 0 or self.hidden % 128:
            raise ValueError("hidden must be positive and divisible by 128")
        if not 0 < self.topk <= 16:
            raise ValueError("topk must be in [1, 16] for the u16 rank-slot ABI")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.num_qp not in (1, 2, 4, 8):
            raise ValueError("num_qp must be one of 1,2,4,8")
        if self.record_bytes > 64 * 1024:
            raise ValueError("one dispatch record must fit in the 64-KiB CCO group")
        if (64 * 1024) // self.record_bytes < self.num_qp:
            raise ValueError(
                "one dispatch group must contain at least one record per QP"
            )

    @property
    def payload_bytes(self) -> int:
        return self.hidden // (1 if self.activation_dtype == "fp8" else 2)

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
        # One u16 per global rank maps
        # every top-k slot to its owner while preserving duplicate-rank routes.
        return self.world_size * 2

    @property
    def raw_bytes(self) -> int:
        return self.rank_slot_masks_offset + self.rank_slot_masks_bytes

    @property
    def record_bytes(self) -> int:
        return _align_up(self.raw_bytes, self.record_alignment)

    @property
    def records_per_chunk(self) -> int:
        # chunk controls signaling granularity; staging/rx register the full max_tokens buffer.
        # One chunk per rank minimizes WQEs and doorbells (k1 latency reduced by 9%).
        if self.max_tokens > 0:
            capacity = max(self.num_qp, self.max_tokens)
        else:
            capacity = (64 * 1024) // self.record_bytes
        return capacity - capacity % self.num_qp


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

    The layout is specialized at compile time for one EP16 model geometry. A
    fixed rank inbox slot ``source_rank * max_tokens + source_token`` removes a
    device allocator from the rank-deduplicated dispatch path.  Expert routes
    are compacted independently into dynamically allocated BM tiles; physical
    tiles may interleave experts, but every tile is single-expert and therefore
    directly consumable by the existing A4W4 GMM bodies.  ``tile_dst_of_src``
    and the ``*_sorted`` regions carry an expert-major view of that same tile
    set for Stage2, whose GEMM2 wants one expert's weights resident across
    consecutive m-blocks.
    """

    hidden: int
    inter: int
    experts: int
    world_size: int
    gpus_per_node: int
    topk: int
    max_tokens: int
    max_routes_per_token_per_rank: int
    block_m: int
    block_n: int
    parity_depth: int
    num_qp: int
    # GMM1 treats tile_group physically adjacent block_m-row tiles as one m_block.
    # Arena tiles remain block_m rows, as Stage2 / MegaMoEv2 SBM requires.
    # Only GMM1 M tiling changes to block_m*tile_group.
    tile_group: int
    # Precompute placement at the destination. Send counts over the rail first;
    # only topk_ids are needed, so this can start immediately. Each destination
    # stores row bases for (source rank, local expert) in its arena. Intra-node
    # producers and forwarders read them through LSA, avoiding cross-device atomics
    # during payload transfer.
    regions: tuple[Stage1ArenaRegion, ...]
    total_bytes: int
    activation_dtype: str = "fp4"

    @property
    def local_experts(self) -> int:
        return self.experts // self.world_size

    @property
    def source_capacity(self) -> int:
        return self.world_size * self.max_tokens

    @property
    def route_capacity(self) -> int:
        # The compact contract bounds how many routes from one source token
        # may target any one EP rank.  The default equals topk and therefore
        # preserves the original arbitrary-routing capacity.
        return self.source_capacity * self.max_routes_per_token_per_rank

    @property
    def max_tiles_per_expert(self) -> int:
        # Also tolerate repeated exact expert IDs. Normal top-k output is unique,
        # but sizing the small map for full route capacity avoids an unchecked
        # memory-safety assumption. Paired claims allocate whole tile_group groups,
        # so round each expert's map slots up to a multiple of tile_group to prevent
        # the last group from overwriting the next expert's slots.
        n = (self.route_capacity + self.block_m - 1) // self.block_m
        g = self.tile_group
        return ((n + g - 1) // g) * g

    @property
    def max_route_tiles(self) -> int:
        # Splitting route_capacity rows over E_local experts introduces at most
        # E_local-1 additional partially occupied tiles.  Keep one extra tile
        # as the existing conservative bound does.
        return (
            self.route_capacity + self.block_m - 1
        ) // self.block_m + 2 * self.local_experts * self.tile_group  # Pad local and remote segments separately to G.

    @property
    def max_route_rows(self) -> int:
        return self.max_route_tiles * self.block_m

    @property
    def h1_n_blocks(self) -> int:
        return (2 * self.inter) // self.block_n

    @property
    def dispatch_chunks(self) -> int:
        return (
            self.max_tokens + self.wire.records_per_chunk - 1
        ) // self.wire.records_per_chunk

    @property
    def wire(self) -> Stage1DispatchWire:
        return Stage1DispatchWire(
            hidden=self.hidden,
            topk=self.topk,
            world_size=self.world_size,
            num_qp=self.num_qp,
            max_tokens=self.max_tokens,
            activation_dtype=self.activation_dtype,
        )

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
        max_routes_per_token_per_rank: int | None = None,
        block_m: int = 32,
        block_n: int = 256,
        tile_group: int = 1,
        parity_depth: int = 2,
        num_qp: int = 4,
        embed_stage2_bytes: int = 0,
        send_queue_capacity: int = 0,
        activation_dtype: str = "fp4",
    ) -> Stage1ArenaLayout:
        if activation_dtype not in ("fp4", "fp8"):
            raise ValueError("activation_dtype must be fp4 or fp8")
        a_pack = 1 if activation_dtype == "fp8" else 2
        hidden = int(hidden)
        inter = int(inter)
        experts = int(experts)
        world_size = int(world_size)
        gpus_per_node = int(gpus_per_node)
        topk = int(topk)
        max_tokens = int(max_tokens)
        max_routes_per_token_per_rank = (
            topk
            if max_routes_per_token_per_rank is None
            else int(max_routes_per_token_per_rank)
        )
        block_m = int(block_m)
        block_n = int(block_n)
        tile_group = int(tile_group)
        parity_depth = int(parity_depth)
        num_qp = int(num_qp)
        if hidden < 1024 or hidden % 512:
            raise ValueError("hidden must be >= 1024 and divisible by 512")
        if hidden > 8192:
            raise ValueError("hidden must be <= 8192 for the 256-thread quantizer")
        if inter <= 0 or inter % 256:
            raise ValueError("inter must be positive and divisible by 256")
        if world_size != 16 or gpus_per_node != 8:
            raise ValueError("the current transport requires EP16 on two 8-GPU nodes")
        if experts <= 0 or experts % world_size:
            raise ValueError("experts must be positive and divisible by world_size")
        validate_fused_weight_addressing(
            hidden=hidden,
            inter=inter,
            experts=experts,
            world_size=world_size,
        )
        if not 1 <= topk <= 16:
            raise ValueError("topk must be in [1, 16] for the rank-slot ABI")
        if tile_group not in (1, 2, 4):
            raise ValueError("tile_group must be 1, 2 or 4")
        if (block_n, parity_depth, num_qp) != (256, 2, 4) or block_m not in (
            32,
            64,
            128,
        ):
            raise ValueError(
                "the current fused implementation requires "
                "block_m in (32, 64, 128), block_n=256, "
                "parity_depth=2 and num_qp=4"
            )
        if not 1 <= max_tokens <= MAX_FUSED_TOKENS_PER_RANK:
            raise ValueError(
                f"max_tokens must be in [1, {MAX_FUSED_TOKENS_PER_RANK}] for the fused Stage-1 ABI"
            )
        if not 1 <= max_routes_per_token_per_rank <= topk:
            raise ValueError("max_routes_per_token_per_rank must be in [1, topk]")
        local_experts = experts // world_size
        source_capacity = world_size * max_tokens
        if source_capacity > MAX_PACKED_SOURCE_CAPACITY:
            raise ValueError(
                "world_size * max_tokens exceeds the 24-bit packed source capacity"
            )
        route_capacity = source_capacity * max_routes_per_token_per_rank
        _base_tiles = (route_capacity + block_m - 1) // block_m
        max_route_tiles = _base_tiles + 2 * local_experts * tile_group
        max_route_rows = max_route_tiles * block_m
        max_tiles_per_expert = (
            (_base_tiles + tile_group - 1) // tile_group
        ) * tile_group
        # Stage1's grouped payload writers form byte indices in signed i32.
        # Keep each parity-local payload below that boundary instead of
        # allowing a large token/inter configuration to wrap to an earlier row.
        # h1_output_q is exempt: past 2 GiB GMM1 moves each m-block's row
        # offset into its 64-bit base (gemm1 aqout_row_base); one m-block
        # still indexes in i32.
        signed_i32_payloads = {
            "grouped_input_q": source_capacity * (hidden // a_pack),
            "grouped_input_scale": source_capacity * (hidden // 32),
            "h1_output_q (one m-block)": block_m * tile_group * (inter // a_pack),
            "h1_output_scale": max_route_rows * (inter // 32),
        }
        oversized_payloads = {
            name: nbytes
            for name, nbytes in signed_i32_payloads.items()
            if nbytes >= 2**31
        }
        if oversized_payloads:
            detail = ", ".join(
                f"{name}={nbytes}" for name, nbytes in oversized_payloads.items()
            )
            raise ValueError(
                "Stage-1 parity-local payloads must each be smaller than 2 GiB "
                f"for signed 32-bit indexing: {detail}"
            )
        wire = Stage1DispatchWire(
            hidden=hidden,
            topk=topk,
            world_size=world_size,
            num_qp=num_qp,
            max_tokens=max_tokens,
            activation_dtype=activation_dtype,
        )
        chunks = (max_tokens + wire.records_per_chunk - 1) // wire.records_per_chunk
        input_scale_bytes = hidden // 32
        output_scale_bytes = inter // 32

        # All payload/state that is remotely addressed through CCO or LSA lives
        # in this single registered window.  Generation arrays are absolute and
        # parity buffered; payload clearing between hot forwards is unnecessary.
        specs: list[tuple[str, tuple[int, ...], torch.dtype, int]] = [
            (
                "dispatch_staging",
                (parity_depth, max_tokens, wire.record_bytes),
                torch.uint8,
                256,
            ),
            (
                "remote_dispatch_rx",
                (parity_depth, max_tokens, wire.record_bytes),
                torch.uint8,
                256,
            ),
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
            # destination rank. One EOS slot covers the aligned pair of source
            # ranks (local node + remote node), hence exactly eight EOS values.
            # [0,8): comm_eos, both local and remote halves are complete.
            # [8,16): local_eos in split_local, this source rank's fan1 local half is complete.
            ("comm_eos", (parity_depth, 2 * gpus_per_node), torch.int64, 64),
            ("launch_ready", (parity_depth,), torch.int64, 64),
            # Per-destination shard completion counts. The last shard publishes comm_eos
            # and the credit atomic; unused when shards==1. In split_local, [8,16) counts
            # completed fan1 shards for publishing local_eos.
            ("fanout_shard_done", (parity_depth, 2 * gpus_per_node), torch.int32, 64),
            # split_local: [0,LE) counts local-source rows; [LE,2LE) counts remote-source rows.
            # Each segment forms its own groups.
            ("expert_count", (parity_depth, 2 * local_experts), torch.int32, 64),
            (
                "expert_tile_map",
                (parity_depth, 2 * local_experts, max_tiles_per_expert),
                torch.int32,
                64,
            ),
            (
                "expert_tile_map_ready",
                (parity_depth, 2 * local_experts, max_tiles_per_expert),
                torch.int64,
                64,
            ),
            # word1: tile count of the split_local local segment, written after phase 1 sealing.
            ("tile_alloc", (parity_depth, 2), torch.int32, 64),
            ("tile_row_done", (parity_depth, max_route_tiles), torch.int32, 64),
            ("tile_expert", (parity_depth, max_route_tiles), torch.int32, 64),
            # GEMM2 keeps sorted IDs separate from physical expert IDs.
            ("tile_expert_sorted", (parity_depth, max_route_tiles), torch.int32, 64),
            ("num_valid", (parity_depth,), torch.int32, 64),
            # Quantized A is stored once per global source token on each
            # selected destination rank.  tile_row_input maps every grouped
            # expert route back to that shared source row; scales remain in
            # source-major layout; GMM1 gathers and shuffles scales into LDS.
            ("tile_row_input", (parity_depth, max_route_rows), torch.int32, 64),
            ("tile_row_source", (parity_depth, max_route_rows), torch.int32, 64),
            ("tile_row_weight", (parity_depth, max_route_rows), torch.float32, 64),
            (
                "grouped_input_q",
                (parity_depth, source_capacity, hidden // a_pack),
                torch.uint8,
                256,
            ),
            (
                "grouped_input_scale",
                (parity_depth, source_capacity * input_scale_bytes),
                torch.uint8,
                256,
            ),
            (
                "h1_output_q",
                (parity_depth, max_route_rows, inter // a_pack),
                torch.uint8,
                256,
            ),
            (
                "h1_output_scale",
                (parity_depth, max_route_rows * output_scale_bytes),
                torch.uint8,
                256,
            ),
            # In tile-pipeline mode each contiguous n_blocks batch uses its
            # first generation word as the release-published ready marker.
            # Eight independent consumer heads occupy separate cache lines.
            ("h1_queue_eos", (parity_depth,), torch.int64, 64),
            # Persistent launch state is intentionally not parity indexed.
            ("entry_count", (1,), torch.int64, 64),
            ("epoch_gate", (1,), torch.int64, 64),
            ("error_count", (1,), torch.int32, 64),
            # GEMM2 reads expert-major IDs and maps them to physical activation
            # tiles, so sorting never copies the quantized activation payload.
            # h1_phys is the inverse of tile_dst_of_src: the physical tile for sorted tile t.
            # Stage2 uses it to read A.
            ("tile_src_of_dst", (parity_depth, max_route_tiles), torch.int32, 64),
            # Local sealing must precede final metadata publication. Job heads
            # also contain reservation counters; their cache-line spacing is ABI.
            ("h1_local_eos", (parity_depth,), torch.int64, 64),
            ("gmm1_job_head", (parity_depth, 32), torch.int32, 64),
        ]

        if tile_group > 1:
            # GMM1 uses groups as m_blocks, so expert-major destinations must also use groups.
            # Append this only when enabled, preserving the default arena offsets byte for byte.
            for _gname in ("tile_expert_group",):
                specs.append(
                    (
                        _gname,
                        (parity_depth, max_route_tiles // tile_group + 1),
                        torch.int32,
                        64,
                    )
                )

        # Single fused kernel: per physical tile group, the GMM1 jobs done
        # (words [0, n)) and the generation whose h1 is complete (words
        # [n, 2n)); GEMM2 waits on the latter instead of a grid barrier.
        specs.append(
            (
                "h1_grp_sync",
                (parity_depth, 2 * (max_route_tiles // tile_group + 1)),
                torch.int32,
                64,
            )
        )

        # Split-path GMM1 ready queue: (generation << 32) | physical group ID.
        # generation marks validity, avoiding per-generation clearing. An atomic on
        # gmm1_job_head word 8 allocates each queue write position.
        specs.append(
            (
                "gmm1_ready_q",
                (parity_depth, max_route_tiles // tile_group + 1),
                torch.int64,
                64,
            )
        )

        if int(send_queue_capacity):
            if int(send_queue_capacity) < 0:
                raise ValueError("send_queue_capacity must be nonnegative")
            # Local metadata-to-copy handoff. Eight-wave metadata batches use
            # at most this many records; 64 extra ready slots cover terminal
            # claims by all supported copy roles, without reading out of bounds.
            send_nb = min(512 // topk, block_m * tile_group) // 8 * 8
            send_dw = (8 + send_nb + 4 * local_experts + 15) // 16 * 16
            specs.extend(
                [
                    ("send_job_state", (parity_depth, 32), torch.int32, 64),
                    ("send_job_eos", (parity_depth,), torch.int64, 64),
                    (
                        "send_job_ready",
                        (parity_depth, int(send_queue_capacity) + 64),
                        torch.int64,
                        64,
                    ),
                    (
                        "send_job_desc",
                        (parity_depth, int(send_queue_capacity), send_dw),
                        torch.int32,
                        64,
                    ),
                    ("send_join", (parity_depth, 2 * gpus_per_node), torch.int32, 64),
                ]
            )

        # Place RDMA (rail GDA) regions first in the window, reserve embed_stage2_bytes,
        # then place other LSA/local regions, including those scaled by max_route_rows.
        # CCO GDA registers the whole window as one MR with iova=0 and window-relative
        # raddr. Ionic's single-MR range is limited: writes past roughly 1.07GiB did
        # not land, causing the October 8 TPR1024 random hang. This ordering makes
        # RDMA offsets scale with tokens rather than route capacity. Host and kernels
        # resolve offsets by name, so regions can be reordered; Stage1 still starts at 0.
        _rdma_first = (
            "dispatch_staging",
            "remote_dispatch_rx",
            "remote_chunk_ready",
            "remote_chunk_credit",
        )
        _head = [sp for sp in specs if sp[0] in _rdma_first]
        _tail = [sp for sp in specs if sp[0] not in _rdma_first]
        if int(embed_stage2_bytes) > 0:
            _head.append(
                ("stage2_embed", (int(embed_stage2_bytes),), torch.uint8, 4096)
            )
        specs = _head + _tail

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
            max_routes_per_token_per_rank=max_routes_per_token_per_rank,
            block_m=block_m,
            block_n=block_n,
            parity_depth=parity_depth,
            num_qp=num_qp,
            tile_group=tile_group,
            regions=tuple(regions),
            total_bytes=_align_up(offset, 4096),
            activation_dtype=activation_dtype,
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
    ) -> TwoKernelArenaLayout:
        # Embed Stage2 after Stage1's RDMA prefix; see Stage1ArenaLayout.create embed_stage2_bytes.
        slot = stage1.region("stage2_embed")
        if slot.nbytes < int(stage2.total_bytes) or slot.offset % alignment:
            raise ValueError("stage2_embed slot is smaller than stage2 or misaligned")
        return cls(
            stage1, stage2, slot.offset, _align_up(stage1.total_bytes, alignment)
        )


@dataclass(frozen=True)
class Stage2NodePartialWire:
    """Fixed-order CCO return format.

    The RAIL connection is rank aligned.  Consequently record ``token`` is
    already destined for the same local rank on the peer node; no source-rank
    or token header is needed in lockstep mode. The default four-row group is
    28 KiB at H3584 and 56 KiB at H7168; every accepted shape is checked
    against the same 64-KiB group constraint.
    It is distinct from Stage2's return_chunk_tokens=8 default: that path
    posts a contiguous eight-row payload and indexes readiness by chunk.
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

    Kernel1 pushes weighted rows into the proxy GPU's ``plane_slot_inbox``;
    kernel2 reduces them into ``node_accumulator[plane, token]``.

    Absolute generations guard the two parity slots; fully overwritten
    payloads can retain old bytes until release-published.  RAIL's consumed
    credit guards remote buffer reuse.
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
    regions: tuple[Stage2ArenaRegion, ...]
    total_bytes: int

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
        rail_record_bytes: int | None = None,
    ) -> Stage2ArenaLayout:
        """Stage2 window layout: the regions kernel1 / kernel2 use.

        Above 4096 tokens the rail regions are sized by the rail row width.
        """
        ready_group_tiles = 2
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
        # wide_rail_rows: above 4096 tokens the two rail regions are sized by the
        # rail row width (bf16 2*hidden, or fp8 hidden + hidden/32) rather than
        # an FP32 capacity, so the NIC-registered prefix stays small. <= 4096
        # keeps the original layout byte for byte.
        wide_rail_rows = int(max_tokens) > 4096
        if wide_rail_rows:
            if rail_record_bytes is None:
                raise ValueError("max_tokens > 4096 needs rail_record_bytes")
            rail_record_bytes = int(rail_record_bytes)
            if (
                rail_record_bytes <= 0
                or rail_record_bytes % 8
                or rail_record_bytes > 2 * int(hidden)
            ):
                raise ValueError(f"bad rail_record_bytes {rail_record_bytes}")
        if not 1 <= int(max_tokens) <= MAX_FUSED_TOKENS_PER_RANK:
            raise ValueError(
                f"max_tokens must be in [1, {MAX_FUSED_TOKENS_PER_RANK}] for the fused Stage-2 ABI"
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
        wire = Stage2NodePartialWire(int(hidden), int(records_per_group))
        groups = wire.group_count(max_tokens)
        ntiles = int(hidden) // int(tile_n)
        ready_groups = (ntiles + ready_group_tiles - 1) // ready_group_tiles
        if ready_groups > 64:
            raise ValueError("ready_group_count exceeds the 64-bit node_ready_mask")

        # name, shape, dtype, alignment.  Only what the kernels read or write.
        specs: list[tuple[str, tuple[int, ...], torch.dtype, int]] = [
            (
                "node_accumulator",
                # Capacity is FP32; BF16 kernels address a packed prefix using
                # 2-byte elements, while parity stride stays the full allocation.
                # Above 4096 tokens it is the rail row width exactly (the NIC
                # registration must stay below ~1 GiB; see wide_rail_rows).
                (
                    (parity_depth, source_nodes, max_tokens, rail_record_bytes)
                    if wide_rail_rows
                    else (parity_depth, source_nodes, max_tokens, hidden)
                ),
                torch.uint8 if wide_rail_rows else torch.float32,
                256,
            ),
            # The peer RAIL rank writes its node partial for this rank's local
            # source tokens directly here, in token order.
            (
                "remote_partial_rx",
                (
                    (parity_depth, max_tokens, rail_record_bytes)
                    if wide_rail_rows
                    else (parity_depth, max_tokens, hidden)
                ),
                torch.uint8 if wide_rail_rows else torch.bfloat16,
                256,
            ),
            ("return_group_ready", (parity_depth, groups), torch.int64, 64),
            # Token-ready rail: per parity [send ids | received ids] of the
            # remote plane, batch position -> token (rides the rail with rows).
            ("rail_idx", (parity_depth, 2, max_tokens), torch.int32, 256),
            ("return_consumed", (parity_depth,), torch.int64, 64),
            (
                "node_partial_ready",
                (parity_depth, source_nodes, max_tokens),
                torch.int64,
                64,
            ),
            # Stage1 writes, per (node plane, token), the bitmap of top-k slots
            # whose expert lives on this node.
            (
                "node_dest_slot_mask",
                (parity_depth, source_nodes, max_tokens),
                torch.int32,
                64,
            ),
            # One word per (token, top-k slot): the last of the num_n_blocks
            # contributors stores the generation (i64, polled ">= expected", so
            # no pass ever clears it).
            (
                "plane_slot_arrived",
                (parity_depth, source_nodes * max_tokens, topk),
                torch.int64,
                64,
            ),
            # Contributor payload, keyed by global top-k slot.
            (
                "plane_slot_inbox",
                (
                    parity_depth,
                    source_nodes * max_tokens,
                    ready_groups,
                    topk,
                    ready_group_tiles * tile_n,
                ),
                torch.bfloat16,
                256,
            ),
        ]

        # Place rail RDMA regions first in Stage2: source node_accumulator and targets
        # remote_partial_rx, return_group_ready, and return_consumed. See Stage1ArenaLayout.create.
        _rdma_first = (
            "node_accumulator",
            "remote_partial_rx",
            "return_group_ready",
            "return_consumed",
            "rail_idx",
        )
        specs = [sp for sp in specs if sp[0] in _rdma_first] + [
            sp for sp in specs if sp[0] not in _rdma_first
        ]

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


__all__ = [
    "MAX_FUSED_TOKENS_PER_RANK",
    "MAX_PACKED_SOURCE_CAPACITY",
    "SPARSE_QP_GENERATION_SHIFT",
    "SPARSE_QP_TOKEN_BITS",
    "Stage1ArenaLayout",
    "Stage1ArenaRegion",
    "Stage1DispatchWire",
    "Stage2ArenaLayout",
    "Stage2ArenaRegion",
    "Stage2NodePartialWire",
    "TwoKernelArenaLayout",
]
