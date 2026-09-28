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
MAX_FUSED_TOKENS_PER_RANK = 4096
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

    def __post_init__(self) -> None:
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
            raise ValueError("one dispatch group must contain at least one record per QP")

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
        capacity = (64 * 1024) // self.record_bytes
        # 诊断:把每 chunk 的记录数压小,使 dispatch_chunks 变成合数,
        # 从而能用 cco_chunks_per_flush 做「门铃次数」的单变量扫描。
        # H3584 下 record_bytes=2304 ⇒ capacity=28 ⇒ dispatch_chunks=19(质数),
        # cco_chunks_per_flush 除了 1 无合法值。cap=16 ⇒ chunks=32。
        import os as _os_rpc
        # 64KB 不是任何缓冲区的上限 —— dispatch_staging / remote_dispatch_rx
        # 都按 MAX_TOKENS 条记录整块注册,chunk 只是**信令粒度**。把它调大
        # 就是直接减少 WQE 数与门铃数(chunks=19 ⇒ 88 WQE/76 门铃;chunks=1
        # ⇒ 8 WQE/4 门铃),所以这个环境变量既能压小也能放大。
        _cap = int(_os_rpc.environ.get("MEGAMOE_TK_S1_RECORDS_PER_CHUNK", "0") or 0)
        if _cap > 0:
            capacity = _cap
        # Every CCO QP owns the same contiguous number of records. Round the
        # group down instead of relying on the two historical record sizes to
        # happen to divide evenly across four QPs.
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
    # GMM1 把 tile_group 个物理相邻的 block_m 行 tile 当作一个 m_block。
    # arena 的 tile 粒度仍是 block_m(Stage2 / MegaMoEv2 的 SBM 依赖它),
    # 只有 GMM1 的 M 方向 tiling 变成 block_m*tile_group。
    tile_group: int
    # 目的端预计算放置:计数先过 rail(只依赖 topk_ids,入口就有),目的端
    # 算出每个 (源 rank, 本地 expert) 的行基址并留在自己 arena 里,node 内
    # 的推送方/转发方直接 LSA 读。payload 阶段因此没有任何跨设备原子。
    dispatch_plan: bool
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
        # The compact contract bounds how many routes from one source token
        # may target any one EP rank.  The default equals topk and therefore
        # preserves the original arbitrary-routing capacity.
        return self.source_capacity * self.max_routes_per_token_per_rank

    @property
    def max_tiles_per_expert(self) -> int:
        # Also tolerate repeated exact expert IDs.  Normal top-k output is
        # unique, but sizing the small map for the full route capacity avoids
        # turning that assumption into an unchecked memory-safety contract.
        # 成对认领按 tile_group 整组分配,所以每个 expert 的 map 槽位数必须
        # 向上取整到 tile_group 的倍数,否则最后一组会越界写进下一个 expert。
        n = (self.route_capacity + self.block_m - 1) // self.block_m
        g = self.tile_group
        return ((n + g - 1) // g) * g

    @property
    def max_route_tiles(self) -> int:
        # Splitting route_capacity rows over E_local experts introduces at most
        # E_local-1 additional partially occupied tiles.  Keep one extra tile
        # as the existing conservative bound does.
        return (
            (self.route_capacity + self.block_m - 1) // self.block_m
            + 2 * self.local_experts * self.tile_group  # 本地/远端各自按 G 补齐
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
        return Stage1DispatchWire(
            hidden=self.hidden,
            topk=self.topk,
            world_size=self.world_size,
            num_qp=self.num_qp,
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
        dispatch_plan: bool = False,
        parity_depth: int = 2,
        num_qp: int = 4,
    ) -> "Stage1ArenaLayout":
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
        dispatch_plan = bool(dispatch_plan)
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
                "max_tokens must be in [1, 4096] for the fused Stage-1 ABI"
            )
        if not 1 <= max_routes_per_token_per_rank <= topk:
            raise ValueError(
                "max_routes_per_token_per_rank must be in [1, topk]"
            )
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
        n_blocks = (2 * inter) // block_n
        # Stage1's grouped payload writers form byte indices in signed i32.
        # Keep each parity-local payload below that boundary instead of
        # allowing a large token/inter configuration to wrap to an earlier row.
        signed_i32_payloads = {
            "grouped_input_q": source_capacity * (hidden // 2),
            "grouped_input_scale": max_route_rows * (hidden // 32),
            "h1_output_q": max_route_rows * (inter // 2),
            "h1_output_scale": max_route_rows * (inter // 32),
        }
        oversized_payloads = {
            name: nbytes
            for name, nbytes in signed_i32_payloads.items()
            if nbytes >= 2**31
        }
        if oversized_payloads:
            detail = ", ".join(
                f"{name}={nbytes}"
                for name, nbytes in oversized_payloads.items()
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
        )
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
            # [0,8) = comm_eos(本地+远端两半都推完);[8,16) = local_eos(split_local:
            # 该源 rank 的 fan1 已推完,只覆盖本地来源那一半)。
            ("comm_eos", (parity_depth, 2 * gpus_per_node), torch.int64, 64),
            ("launch_ready", (parity_depth,), torch.int64, 64),
            # fanout 分片后,每个目的 rank 的分片完成计数;最后一个分片
            # 负责发 comm_eos 和信用原子。shards==1 时不使用。
            # [8,16) = split_local 下 fan1 完成的分片计数(发 local_eos)。
            ("fanout_shard_done", (parity_depth, 2 * gpus_per_node), torch.int32, 64),
            # split_local:[0,LE) 本地来源行计数,[LE,2LE) 远端来源行计数(各自成组)。
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
            # word1 = split_local 下本地段的 tile 数(段 1 封尾后写)。
            ("tile_alloc", (parity_depth, 2), torch.int32, 64),
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
            # 自旋超时标记:每个自旋点一个槽,挂死时由限时自旋写 generation。
            ("plan_debug", (parity_depth, 128), torch.int64, 64),
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
            # Expert-major Stage2 view.  Physical tiles stay in arrival order
            # (that is what keeps dispatch single-pass and compact); these
            # regions carry the permutation to expert-major order plus the two
            # per-row metadata arrays Stage2 indexes by absolute row.  GMM1
            # writes h1_output straight to the permuted slot, so no activation
            # is ever copied.  Appended so every existing region keeps its
            # byte offset unchanged.
            ("tile_dst_of_src", (parity_depth, max_route_tiles), torch.int32, 64),
            ("tile_expert_sorted", (parity_depth, max_route_tiles), torch.int32, 64),
            # h1_phys:tile_dst_of_src 的逆(排序后第 t 个 tile 的物理 tile);stage2 按它读 A。
            ("tile_src_of_dst", (parity_depth, max_route_tiles), torch.int32, 64),
            (
                "tile_row_source_sorted",
                (parity_depth, max_route_rows),
                torch.int32,
                64,
            ),
            (
                "tile_row_weight_sorted",
                (parity_depth, max_route_rows),
                torch.float32,
                64,
            ),
            # early_local_gmm(F5):本地组封好即发布的 generation 字;GMM1 job 领取头
            # (word0 本地/word16 远端,各占一行;每次 launch 由初始化者清零);按「先本地组后远端组」排的物理组号表。
            ("h1_local_eos", (parity_depth,), torch.int64, 64),
            ("gmm1_job_head", (parity_depth, 32), torch.int32, 64),
            ("gmm1_group_list", (parity_depth, max_route_tiles), torch.int32, 64),
        ]

        if tile_group > 1:
            # GMM1 以「组」为 m_block,expert-major 的目的地也必须按组给出。
            # 只在开启时追加,默认配置的 arena 偏移逐字节不变。
            for _gname in ("tile_group_perm", "tile_expert_group"):
                specs.append(
                    (
                        _gname,
                        (parity_depth, max_route_tiles // tile_group + 1),
                        torch.int32,
                        64,
                    )
                )

        if dispatch_plan:
            gpn = gpus_per_node
            specs.extend([
                # 每个源 rank 报给本 rank 每个本地 expert 的行数。
                ("plan_count", (parity_depth, world_size, local_experts),
                 torch.int32, 64),
                ("plan_count_ready", (parity_depth, world_size), torch.int64, 64),
                # 目的端算完后留在自己 arena:node 内的推送方/转发方来读。
                ("plan_row_base", (parity_depth, world_size, local_experts),
                 torch.int32, 64),
                ("plan_expert_rows", (parity_depth, local_experts), torch.int32, 64),
                ("plan_ready", (parity_depth,), torch.int64, 64),
                # 本 rank 自己的全局 expert 直方图(算计数用)。
                ("plan_local_hist", (parity_depth, experts), torch.int32, 64),
                # rail 伙伴送来的计数:它那 8 个本 node 目的端各一段。
                ("rail_count_inbox", (parity_depth, gpn, local_experts),
                 torch.int32, 64),
                ("rail_count_inbox_ready", (parity_depth,), torch.int64, 64),
            ])

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
            dispatch_plan=dispatch_plan,
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
    "MAX_FUSED_TOKENS_PER_RANK",
    "MAX_PACKED_SOURCE_CAPACITY",
    "SPARSE_QP_GENERATION_SHIFT",
    "SPARSE_QP_TOKEN_BITS",
    "Stage1ArenaLayout",
    "Stage1ArenaRegion",
    "Stage1DispatchWire",
    "TwoKernelArenaLayout",
    "validate_public_stage1_contract",
]
