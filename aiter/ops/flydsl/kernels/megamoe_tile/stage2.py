# SPDX-License-Identifier: MIT
"""EP16 A4W4 GEMM2, rank reduction/push, node reduction and RAIL return.

The retained protocol stores weighted BF16 route rows locally, reduces actual
TopK contributors in original slot order, and pushes token/N-group tiles into
per-rank peer inboxes. Node reducers read only local inboxes. Full mode returns
one BF16 node partial per token over RAIL, then adds the two node partials.

Logical CTA ranges are RAIL / rank-push / node-reduce / final-add / GEMM.
Budgets and publication batch size are tuning parameters; block IDs do not pin
physical CUs. Arbitrary routing, duplicate expert slots, absent ranks/nodes,
and generation-buffer reuse are part of the contract, independent of EPLB.

Historical atomic, staged-ring, watermark and compact-return implementations
are archived in trace_data source snapshots. See the current reduce-push design
under scripts/megamoe_tile for publication ordering and validation status.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Float32, Int8, T

from aiter.ops.flydsl.kernels import buffer_ops
from . import comm_ops
from .gemm_common import (
    _udiv,
    global_typed_ptr,
    kStages,
    lds_typed_ptr,
    lds_vec_load,
)
from .gemm2 import (
    gemm2_compute_v2,
    issue_a_load_lds_dt,
)

from .stage1_abi import Stage1ArenaLayout, TwoKernelArenaLayout
from .stage2_abi import (
    STAGE2_TIMELINE_INDEX,
    Stage2ArenaLayout,
    Stage2NodePartialWire,
)

THREADS = 256
CROSS_BLOCK = 0
DEFAULT_COMBINE_BLOCKS = 14
TEAM_RAIL = "rail"

def _wave_uniform_i64(value):
    """Move a wave-uniform 64-bit value to SGPRs without an i64 waterfall."""

    value_u64 = fx.Uint64(value)
    lo = rocdl.readfirstlane(T.i32, fx.Uint32(value_u64))
    hi = rocdl.readfirstlane(T.i32, fx.Uint32(value_u64 >> 32))
    return (fx.Uint64(hi) << 32) | fx.Uint64(lo)

def _rank_push_poll_sleep():
    """Yield one short scalar sleep only after a failed rank-push arrival poll."""

    _llvm_d.inline_asm(
        None, [], "s_sleep 1", "", has_side_effects=True
    )


def _parity_parts(region, parity_depth: int) -> tuple[int, int]:
    if not region.shape or region.shape[0] != parity_depth:
        raise ValueError(f"region {region.name!r} is not parity indexed")
    return int(region.offset), int(region.nbytes // parity_depth)

def _resolve_layouts(arena_layout, stage2_layout, stage2_window_offset):
    if isinstance(arena_layout, TwoKernelArenaLayout):
        if stage2_layout is not None or stage2_window_offset is not None:
            raise ValueError("split Stage-2 layout args must be omitted for Composite")
        return arena_layout.stage1, arena_layout.stage2, int(arena_layout.stage2_offset)
    if not isinstance(arena_layout, Stage1ArenaLayout):
        raise TypeError("arena_layout must be Stage1ArenaLayout or TwoKernelArenaLayout")
    if not isinstance(stage2_layout, Stage2ArenaLayout):
        raise TypeError("standalone Stage1ArenaLayout requires stage2_layout")
    if stage2_window_offset is None or int(stage2_window_offset) < 0:
        raise ValueError("stage2_window_offset must be non-negative")
    return arena_layout, stage2_layout, int(stage2_window_offset)

from .rank_push_layout import (
    RankPushWorkspace,
    node_reduce_token_owner_mask_prefetch_supported,
    node_reduce_token_owner_supported,
)

def _validate_rank_epilogue_barrier(barrier, node_accumulation, rank_accumulation):
    """Guard the experiment by its publication protocol, never by route balance."""
    if barrier not in ("per_row", "per_tile"):
        raise ValueError("rank_epilogue_barrier must be per_row or per_tile")
    if barrier == "per_tile" and not (
        node_accumulation == "rank_local" and rank_accumulation == "reduce_push"
    ):
        raise ValueError("per_tile epilogue barrier requires rank_local/reduce_push")

def compile_megamoe_tile_ep16_stage2_a4w4(
    arena_layout,
    *,
    rank: int,
    stage2_layout: Stage2ArenaLayout | None = None,
    stage2_window_offset: int | None = None,
    BM: int = 32,
    BN: int = 256,
    BK: int = 256,
    WORK_SHARDS: int = 8,
    waves_per_eu_hint: int = 2,
    team: str = "rail",
    diagnostic_mode: str = "full",
    accumulator_dtype: str = "bf16",
    final_combine_blocks: int = DEFAULT_COMBINE_BLOCKS,
    gmm_schedule: str = "persistent_queue",
    gemm_use_nt: bool = False,
    rank_push_use_nt: bool = False,
    gmm_work_swizzle: str = "token_major",
    window_n_groups: int = 2,
    gmm_work_unsigned_window_division: bool = False,
    gmm_direct_tile_coords: bool = False,
    gmm_queue_ticket_peer_slab: bool = False,
    gmm_queue_head_barrier_elision: bool = False,
    return_chunk_tokens: int = 8,
    bf16_atomic_kind: str = "buffer",
    rail_return_schedule: str = "lockstep",
    rail_quant_type: str = "none",
    ready_granularity: str = "group",
    group_batch: bool = False,
    epilogue_schedule: str = "lane32_meta",
    n_tile_group: int = 2,
    group_pipeline_schedule: str = "a_double_buffer",
    node_accumulation_mode: str = "rank_local",
    rank_accumulation_mode: str = "reduce_push",
    rank_reduce_blocks: int = 8,
    rank_push_batch_size: int = 1,
    rank_push_batch_invariants: bool = False,
    rank_push_count_prefetch: bool = False,
    rank_push_count_wave_broadcast: bool = False,
    rank_push_map_prefetch: bool = False,
    rank_push_count_from_slot_map: bool = False,
    rank_push_tile_index_recurrence: bool = False,
    rank_push_poll_backoff: bool = False,
    rank_push_acquire_cohort: int = 1,
    rank_push_single_row_fastpath: bool = False,
    rank_push_publication: str = "per_tile_counter",
    node_reduce_blocks: int = 32,
    node_reduce_token_owner_fastpath: bool = False,
    node_reduce_token_owner_mask_prefetch: bool = False,
    node_reduce_expected_popcount: bool = False,
    node_reduce_vec_bytes: int = 16,
    node_reduce_schedule: str = "token",
    node_reduce_load_schedule: str = "load_first",
    node_reduce_work_schedule: str = "static_strided",
    node_reduce_rejoin_blocks: int = 0,
    rank_epilogue_lds_addressing: str = "expanded",
    rank_epilogue_barrier: str = "per_row",
    scoreboard_schedule: str = "wave0",
    atomic_issue_schedule: str = "interleaved",
    timeline_instrument: bool = False,
    device_generation: bool = False,
    kernel_name_override: str | None = None,
):
    """Compile the current reduce-push protocol and its tuning variants.

    Tile geometry32/256/256 and two-N-tile publication match the current H1
    arena ABI. Q/R/F budgets divide a resident grid, not hardwired physical CUs.
    ``gemm_use_nt`` controls B/scales cache policy; ``rank_push_use_nt`` controls
    peer stores, allowing the ordinary GEMM2 and MegaMoEv2 policies to be tested.
    ``rank_push_batch_size`` amortizes system publication across ready tiles.
    ``gmm_work_unsigned_window_division`` changes only the proven-nonnegative
    persistent work-ID division used by the W1 N-major mapping.
    ``gmm_direct_tile_coords`` packs each scheduled M/N tile into one
    BM-aligned carrier, avoiding the general flattened-work decode.
    ``gmm_queue_ticket_peer_slab`` keeps the persistent queue ticket in the
    peer-pointer LDS slab, isolating it from the first GEMM A slab while
    retaining the existing queue barriers and work assignment.
    ``gmm_queue_head_barrier_elision`` removes the persistent loop's leading
    CTA barrier. The preceding work already joins before publication, and the
    retained ticket-publication barrier joins every wave before ticket use.
    ``rank_push_batch_invariants`` reuses the N group, source rank, peer
    descriptor and token-plane base shared by a complete source-token batch.
    ``rank_push_count_prefetch`` has the wave load one immutable local route
    count per batch lane, then selects each tile's count with ``readlane``.
    ``rank_push_count_wave_broadcast`` has every lane load the same immutable
    per-source route count and retains ``readfirstlane`` for scalar consumers,
    avoiding the lane-zero EXEC region around the ordinary count load.
    ``rank_push_map_prefetch`` issues each active tile's immutable slot-map
    load before arrival polling, while retaining the payload acquire and the
    original-slot reduction order.
    ``rank_push_count_from_slot_map`` loads that map once, derives the local
    arrival target with ``ctpop``, and reuses the same map for ordered
    reduction. Count storage and validation remain part of the protocol ABI.
    ``rank_push_tile_index_recurrence`` advances the common peer inbox/node
    arrival tile index once per slot in a complete batch instead of rebuilding
    ``token_index * ready_groups + n_group`` for each active tile.
    ``rank_push_poll_backoff`` inserts one ``s_sleep 1`` only after a failed
    producer-arrival observation before the next relaxed load. The first poll,
    success path, acquire, payload order and publication remain unchanged.
    ``rank_push_acquire_cohort=2`` observes two adjacent nonempty tiles with
    relaxed loads before one wave-wide acquire, then consumes both tiles in
    their original order. It changes no payload or publication address.
    ``rank_push_single_row_fastpath`` uses the padding beside each immutable
    route count to cache the sole row when a source has local fan-in one. This
    bypasses the TopK map scan while preserving the ordered map path for every
    source with multiple local contributions.
    The experimental ``batch_bitmap`` publication packs eight tokens' rank
    masks into two disjoint-bit i32 words, reducing remote RMWs without changing
    route counts, arithmetic, inbox ownership, or N readiness.
    ``packed_tile_bitmap`` uses the same four-token word packing while retaining
    per-token waits, so a B64 payload batch needs sixteen publication atomics
    without adding a four- or eight-token consumer readiness barrier.
    ``node_reduce_token_owner_fastpath`` uses the static task mapping only when
    one wave owns every N group for each token.  Its last group publishes the
    token directly and avoids the per-group local ready-mask RMW chain.
    ``node_reduce_token_owner_mask_prefetch`` additionally loads each owner
    wave's immutable token rank masks once and reuses them across N groups.
    ``node_reduce_expected_popcount`` lowers the per-group contributor count
    through LLVM ctpop instead of expanding all eight mask bits separately.

    Legacy keyword names remain only to reject retired configurations clearly
    for saved experiment commands. No historical device implementation remains.
    Full, init_only and node_partial_only are the supported execution modes.
    Node-only retains logical EP16/source16/TopK16 with RAIL/final disabled and
    requires the harness to protect generations across all eight local peers.
    """
    retained = {
        "node_accumulation_mode": (node_accumulation_mode, "rank_local"),
        "rank_accumulation_mode": (rank_accumulation_mode, "reduce_push"),
        "ready_granularity": (ready_granularity, "group"),
        "rail_return_schedule": (rail_return_schedule, "lockstep"),
        "rail_quant_type": (rail_quant_type, "none"),
        "accumulator_dtype": (accumulator_dtype, "bf16"),
        "group_batch": (group_batch, False),
        "node_reduce_rejoin_blocks": (node_reduce_rejoin_blocks, 0),
        "node_reduce_work_schedule": (node_reduce_work_schedule, "static_strided"),
        "node_reduce_schedule": (node_reduce_schedule, "token"),
        "node_reduce_load_schedule": (node_reduce_load_schedule, "load_first"),
        "node_reduce_vec_bytes": (node_reduce_vec_bytes, 16),
        "epilogue_schedule": (epilogue_schedule, "lane32_meta"),
        "n_tile_group": (n_tile_group, 2),
        "group_pipeline_schedule": (group_pipeline_schedule, "a_double_buffer"),
        "rank_epilogue_lds_addressing": (rank_epilogue_lds_addressing, "expanded"),
        "scoreboard_schedule": (scoreboard_schedule, "wave0"),
        "atomic_issue_schedule": (atomic_issue_schedule, "interleaved"),
        "bf16_atomic_kind": (bf16_atomic_kind, "buffer"),
    }
    for name, (actual, required) in retained.items():
        if actual != required:
            raise ValueError(f"retired Stage2 configuration {name}={actual!r}; current protocol requires {required!r}")
    if diagnostic_mode not in ("full", "init_only", "node_partial_only"):
        raise ValueError("current Stage2 supports full, init_only or node_partial_only")
    gemm_use_nt, rank_push_use_nt = bool(gemm_use_nt), bool(rank_push_use_nt)

    from .gda_rail import checked as _rail

    s1, s2, s2_window_off = _resolve_layouts(
        arena_layout, stage2_layout, stage2_window_offset
    )
    # Shape values are compile-time geometry supplied by the arena. Keeping
    # them local makes the generated code and cache identity shape-specific
    # without tying correctness to the K3 production values.
    MAX_TOKENS = int(s1.max_tokens)
    HIDDEN = int(s1.hidden)
    INTER = int(s1.inter)
    EXPERTS = int(s1.experts)
    WORLD = int(s1.world_size)
    GPUS_PER_NODE = int(s1.gpus_per_node)
    TOPK = int(s1.topk)
    SOURCE_CAPACITY = WORLD * MAX_TOKENS
    if WORLD != 16 or GPUS_PER_NODE != 8:
        raise ValueError("Stage-2 transport requires EP16 on two 8-GPU nodes")
    if HIDDEN < 1024 or HIDDEN % (2 * BN):
        raise ValueError(
            f"Stage-2 hidden must be >= 1024 and divisible by {2 * BN}"
        )
    if HIDDEN > 8192:
        raise ValueError(
            "Stage-2 hidden must be <= 8192 for the four-row 64-KiB return group"
        )
    if INTER <= 0 or INTER % BK:
        raise ValueError(f"Stage-2 inter must be positive and divisible by {BK}")
    if EXPERTS <= 0 or EXPERTS % WORLD:
        raise ValueError("Stage-2 experts must be positive and divisible by EP size")
    if EXPERTS // WORLD > 256:
        raise ValueError("Stage-2 requires experts/world_size <= 256")
    if not 1 <= TOPK <= 16:
        raise ValueError("Stage-2 topk must be in [1,16]")
    if not 1 <= MAX_TOKENS <= 4096:
        raise ValueError("Stage-2 max_tokens must be in [1, 4096]")
    if int(s2.max_tokens) != MAX_TOKENS:
        raise ValueError("Stage-1 and Stage-2 max_tokens must match")
    if (
        s2.hidden,
        s2.world_size,
        s2.gpus_per_node,
        s2.topk,
        s2.max_tokens,
    ) != (HIDDEN, WORLD, GPUS_PER_NODE, TOPK, MAX_TOKENS):
        raise ValueError("Stage-1 and Stage-2 arena shapes must match")
    if s1.parity_depth != 2 or s2.parity_depth != 2:
        raise ValueError("the fused pipeline requires parity_depth=2")
    if not 0 <= int(rank) < WORLD:
        raise ValueError("rank must be in [0,16)")
    if (BM, BN, BK) != (32, 256, 256):
        raise ValueError("the direct Stage-2 requires BM/BN/BK=32/256/256")
    if (int(s1.block_m), int(s1.block_n), int(s2.tile_n)) != (BM, BN, BN):
        raise ValueError("Stage-1/Stage-2 tile geometry must match BM/BN=32/256")
    if WORK_SHARDS != 8:
        raise ValueError("the first direct Stage-2 requires 8 work shards")
    if team != TEAM_RAIL:
        raise ValueError("production Stage-2 requires the CCO RAIL team")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")
    # This hint constrains compiler resource allocation; it neither reserves
    # CUs nor changes the four wave64 waves in each 256-thread CTA.
    final_combine_blocks = int(final_combine_blocks)
    if not 1 <= final_combine_blocks <= 56:
        raise ValueError("final_combine_blocks must be in [1,56]")
    if gmm_schedule not in ("persistent_queue", "static_strided"):
        raise ValueError("gmm_schedule must be persistent_queue or static_strided")
    if gmm_work_swizzle not in ("token_major", "n_major_window"):
        raise ValueError("gmm_work_swizzle must be token_major or n_major_window")
    window_n_groups = int(window_n_groups)
    if not 1 <= window_n_groups <= 64:
        raise ValueError("window_n_groups must be in [1,64]")
    gmm_work_unsigned_window_division = bool(
        gmm_work_unsigned_window_division
    )
    if gmm_work_unsigned_window_division and not (
        gmm_work_swizzle == "n_major_window" and window_n_groups == 1
    ):
        raise ValueError(
            "gmm_work_unsigned_window_division requires "
            "n_major_window with window_n_groups=1"
        )
    gmm_direct_tile_coords = bool(gmm_direct_tile_coords)
    if gmm_direct_tile_coords and not (
        gmm_work_swizzle == "n_major_window" and window_n_groups == 1
    ):
        raise ValueError(
            "gmm_direct_tile_coords requires "
            "n_major_window with window_n_groups=1"
        )
    if gmm_direct_tile_coords and HIDDEN // BN > BM:
        raise ValueError(
            "gmm_direct_tile_coords requires hidden/BN <= BM for packed coordinates"
        )
    gmm_queue_ticket_peer_slab = bool(gmm_queue_ticket_peer_slab)
    if gmm_queue_ticket_peer_slab and gmm_schedule != "persistent_queue":
        raise ValueError(
            "gmm_queue_ticket_peer_slab requires persistent_queue scheduling"
        )
    gmm_queue_head_barrier_elision = bool(gmm_queue_head_barrier_elision)
    if gmm_queue_head_barrier_elision and not gmm_queue_ticket_peer_slab:
        raise ValueError(
            "gmm_queue_head_barrier_elision requires "
            "gmm_queue_ticket_peer_slab"
        )
    return_chunk_tokens = int(return_chunk_tokens)
    if return_chunk_tokens not in (4, 8, 16):
        raise ValueError("return_chunk_tokens must be 4, 8, or 16")
    # Lockstep assigns one contiguous chunk to each of four QPs per batch.
    # Capacity divisibility makes every chunk full (no tail transport here).
    # The runtime chunk size is distinct from the ABI's 4-row group.
    if MAX_TOKENS % (s2.num_qp * return_chunk_tokens) != 0:
        raise ValueError(
            "max_tokens must be divisible by num_qp * return_chunk_tokens"
        )
    if getattr(s2, "rail_quant_type", "none") != rail_quant_type:
        raise ValueError("Stage-2 layout rail_quant_type does not match compile option")
    if getattr(s2, "ready_granularity", "token") != ready_granularity:
        raise ValueError("Stage-2 layout ready_granularity does not match compile option")
    # Batched reducers own fixed (N group, token shard) ranges and bypass the
    # token queue. Dynamic queue claims/rejoin do not implement this split.
    n_tile_group = int(n_tile_group)
    rank_reduce_blocks = int(rank_reduce_blocks)
    rank_push_batch_size = int(rank_push_batch_size)
    rank_push_batch_invariants = bool(rank_push_batch_invariants)
    rank_push_count_prefetch = bool(rank_push_count_prefetch)
    rank_push_count_wave_broadcast = bool(rank_push_count_wave_broadcast)
    rank_push_map_prefetch = bool(rank_push_map_prefetch)
    rank_push_count_from_slot_map = bool(rank_push_count_from_slot_map)
    rank_push_tile_index_recurrence = bool(rank_push_tile_index_recurrence)
    rank_push_poll_backoff = bool(rank_push_poll_backoff)
    rank_push_acquire_cohort = int(rank_push_acquire_cohort)
    if rank_push_batch_size not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("rank_push_batch_size must be 1, 2, 4, 8, 16, 32 or 64")
    if rank_push_acquire_cohort not in (1, 2):
        raise ValueError("rank_push_acquire_cohort must be 1 or 2")
    if rank_push_publication not in (
        "per_tile_counter", "batch_bitmap", "packed_tile_bitmap"
    ):
        raise ValueError(
            "rank_push_publication must be per_tile_counter, batch_bitmap or "
            "packed_tile_bitmap"
        )
    if rank_push_publication == "batch_bitmap" and rank_push_batch_size != 8:
        raise ValueError("batch_bitmap publication requires rank_push_batch_size=8")
    if rank_push_publication == "batch_bitmap" and MAX_TOKENS % rank_push_batch_size:
        raise ValueError("batch_bitmap publication requires complete source-token batches")
    if rank_push_publication == "packed_tile_bitmap":
        if rank_push_batch_size < 4 or rank_push_batch_size % 4:
            raise ValueError(
                "packed_tile_bitmap publication requires rank_push_batch_size "
                "divisible by 4"
            )
        if MAX_TOKENS % rank_push_batch_size:
            raise ValueError(
                "packed_tile_bitmap publication requires complete source-token batches"
            )
    if rank_push_batch_invariants:
        if rank_push_publication != "per_tile_counter":
            raise ValueError(
                "rank-push batch invariants require per_tile_counter publication"
            )
        if MAX_TOKENS % rank_push_batch_size:
            raise ValueError(
                "rank-push batch invariants require complete source-token batches"
            )
    if rank_push_count_prefetch:
        if rank_push_publication != "per_tile_counter":
            raise ValueError(
                "rank-push count prefetch requires per_tile_counter publication"
            )
        if MAX_TOKENS % rank_push_batch_size:
            raise ValueError(
                "rank-push count prefetch requires complete source-token batches"
            )
    if rank_push_count_wave_broadcast:
        if rank_push_publication != "per_tile_counter":
            raise ValueError(
                "rank-push count wave broadcast requires per_tile_counter publication"
            )
        incompatible = []
        if rank_push_count_prefetch:
            incompatible.append("count prefetch")
        if rank_push_count_from_slot_map:
            incompatible.append("map-derived count")
        if rank_push_tile_index_recurrence:
            incompatible.append("tile-index recurrence")
        if incompatible:
            raise ValueError(
                "rank-push count wave broadcast does not support "
                + ", ".join(incompatible)
            )
    if rank_push_map_prefetch:
        if rank_push_publication != "per_tile_counter":
            raise ValueError(
                "rank-push map prefetch requires per_tile_counter publication"
            )
        if rank_push_single_row_fastpath:
            raise ValueError(
                "rank-push map prefetch does not support the single-row fast path"
            )
    if rank_push_count_from_slot_map:
        if rank_push_publication != "per_tile_counter":
            raise ValueError(
                "rank-push map-derived count requires per_tile_counter publication"
            )
        incompatible = []
        if rank_push_count_prefetch:
            incompatible.append("count prefetch")
        if rank_push_map_prefetch:
            incompatible.append("map prefetch")
        if rank_push_single_row_fastpath:
            incompatible.append("single-row fast path")
        if incompatible:
            raise ValueError(
                "rank-push map-derived count does not support "
                + ", ".join(incompatible)
            )
    if rank_push_tile_index_recurrence:
        if not rank_push_batch_invariants:
            raise ValueError(
                "rank-push tile-index recurrence requires batch invariants"
            )
        incompatible = []
        if rank_push_count_prefetch:
            incompatible.append("count prefetch")
        if rank_push_map_prefetch:
            incompatible.append("map prefetch")
        if rank_push_count_from_slot_map:
            incompatible.append("map-derived count")
        if rank_push_count_wave_broadcast:
            incompatible.append("count wave broadcast")
        if rank_push_single_row_fastpath:
            incompatible.append("single-row fast path")
        if incompatible:
            raise ValueError(
                "rank-push tile-index recurrence does not support "
                + ", ".join(incompatible)
            )
    if rank_push_acquire_cohort == 2:
        if not rank_push_batch_invariants:
            raise ValueError(
                "rank-push acquire cohort2 requires batch invariants"
            )
        if rank_push_publication != "per_tile_counter":
            raise ValueError(
                "rank-push acquire cohort2 requires per_tile_counter publication"
            )
        if rank_push_batch_size < 2 or rank_push_batch_size % 2:
            raise ValueError(
                "rank-push acquire cohort2 requires an even batch size >=2"
            )
        incompatible = []
        if rank_push_count_prefetch:
            incompatible.append("count prefetch")
        if rank_push_count_wave_broadcast:
            incompatible.append("count wave broadcast")
        if rank_push_map_prefetch:
            incompatible.append("map prefetch")
        if rank_push_count_from_slot_map:
            incompatible.append("map-derived count")
        if rank_push_tile_index_recurrence:
            incompatible.append("tile-index recurrence")
        if rank_push_single_row_fastpath:
            incompatible.append("single-row fast path")
        if incompatible:
            raise ValueError(
                "rank-push acquire cohort2 does not support "
                + ", ".join(incompatible)
            )
    # Role budgets are tuning choices independent of H/I/E/K/T and routing.
    # Keep the validated set explicit until every intermediate allocation has
    # its resident-grid and work-coverage gates.
    if rank_reduce_blocks not in (8, 16, 32, 48, 56, 64):
        raise ValueError(
            "rank_reduce_blocks must be 8, 16, 32, 48, 56 or 64"
        )
    if bool(getattr(s2, "include_rank_push", False)) != (rank_accumulation_mode == "reduce_push"):
        raise ValueError("rank push layout does not match accumulation mode")
    _validate_rank_epilogue_barrier(
        rank_epilogue_barrier, node_accumulation_mode, rank_accumulation_mode
    )
    node_reduce_blocks = int(node_reduce_blocks)
    if node_reduce_blocks not in (8, 16, 32, 56):
        raise ValueError("node_reduce_blocks must be one of 8,16,32,56")
    node_reduce_token_owner_fastpath = bool(node_reduce_token_owner_fastpath)
    node_reduce_token_owner_mask_prefetch = bool(
        node_reduce_token_owner_mask_prefetch
    )
    node_reduce_expected_popcount = bool(node_reduce_expected_popcount)
    if node_reduce_token_owner_fastpath and rank_push_publication == "batch_bitmap":
        raise ValueError("node-reduce token-owner fast path does not support batch_bitmap")
    if node_reduce_token_owner_fastpath and not node_reduce_token_owner_supported(
        max_tokens=MAX_TOKENS, node_reduce_blocks=node_reduce_blocks
    ):
        raise ValueError(
            "node-reduce token-owner fast path requires 2*max_tokens divisible "
            "by 4*node_reduce_blocks"
        )
    if node_reduce_token_owner_mask_prefetch:
        if not node_reduce_token_owner_fastpath:
            raise ValueError(
                "node-reduce token-owner mask prefetch requires the token-owner fast path"
            )
        if not node_reduce_token_owner_mask_prefetch_supported(
            max_tokens=MAX_TOKENS, node_reduce_blocks=node_reduce_blocks
        ):
            raise ValueError(
                "node-reduce token-owner mask prefetch requires all owned token "
                "masks to fit in one wave"
            )
    if node_reduce_expected_popcount and rank_push_publication != "per_tile_counter":
        raise ValueError(
            "node-reduce expected popcount requires per_tile_counter publication"
        )
    node_reduce_vec_bytes = int(node_reduce_vec_bytes)
    if not getattr(s2, "include_rank_partials", False):
        raise ValueError("reduce_push requires a rank-partial Stage2 arena")

    rank = int(rank)
    node = rank // GPUS_PER_NODE
    local_rank = rank % GPUS_PER_NODE
    remote_node = 1 - node
    local_plane = node
    remote_plane = remote_node
    hidden_tiles = HIDDEN // BN
    ready_groups = (hidden_tiles + n_tile_group - 1) // n_tile_group
    if ready_groups != int(s2.ready_group_count):
        raise ValueError(
            "Stage-2 layout ready-group geometry does not match the kernel: "
            f"layout={s2.ready_group_count}, kernel={ready_groups}"
        )
    # Fixed roles must remain resident. More service CTAs subtract from the
    # GEMM pool, so tune Q/R/F jointly with publication batching, by workload.
    rank_push_blocks = rank_reduce_blocks
    reduce_first = 1 + rank_push_blocks
    reduce_blocks = node_reduce_blocks
    combine_first = reduce_first + reduce_blocks
    gmm_first = combine_first + final_combine_blocks
    final_work_items = MAX_TOKENS
    max_m_blocks = s1.max_route_tiles
    wire = Stage2NodePartialWire(HIDDEN, s2.records_per_group)
    return_groups = wire.group_count(MAX_TOKENS)
    if s2.num_qp != 4 or wire.records_per_group != 4:
        raise ValueError("direct Stage-2 requires 4 QPs and 4 records/group")

    # Stage-1 compute outputs and row metadata.
    h1_q_off, h1_q_stride = _parity_parts(s1.region("h1_output_q"), 2)
    h1_scale_off, h1_scale_stride = _parity_parts(s1.region("h1_output_scale"), 2)
    expert_off, expert_stride = _parity_parts(s1.region("tile_expert"), 2)
    nvalid_off, nvalid_stride = _parity_parts(s1.region("num_valid"), 2)
    source_off, source_stride = _parity_parts(s1.region("tile_row_source"), 2)
    weight_off, weight_stride = _parity_parts(s1.region("tile_row_weight"), 2)

    # Stage-2 scoreboard/payload regions, relative to the logical Stage-2 base.
    dest_rank_mask_off, dest_rank_mask_stride = _parity_parts(
        s2.region("node_dest_rank_mask"), 2
    )
    accumulator_off, accumulator_stride = _parity_parts(
        s2.region("node_accumulator"), 2
    )
    rank_pending_off, rank_pending_stride = _parity_parts(
        s2.region("rank_token_pending"), 2
    )
    # Node completion joins groups for each token. Rank readiness lives in the
    # private per-source/group arrival table below; there is no rank watermark.
    node_ready_mask_off, node_ready_mask_stride = _parity_parts(
        s2.region("node_ready_mask"), 2
    )
    partial_ready_off, partial_ready_stride = _parity_parts(
        s2.region("node_partial_ready"), 2
    )
    rx_off, rx_stride = _parity_parts(s2.region("remote_partial_rx"), 2)
    return_ready_off, return_ready_stride = _parity_parts(
        s2.region("return_group_ready"), 2
    )
    consumed_off, consumed_stride = _parity_parts(s2.region("return_consumed"), 2)
    stage2_phase_off, stage2_phase_stride = _parity_parts(s2.region("stage2_init"), 2)
    timeline_off, timeline_stride = _parity_parts(s2.region("timeline"), 2)
    timeline_gmm_done_off, timeline_gmm_done_stride = _parity_parts(
        s2.region("timeline_gmm_worker_done"), 2
    )
    timeline_history_depth = s2.timeline_history_depth
    if timeline_history_depth:
        if not timeline_instrument:
            raise ValueError("timeline history requires timeline instrumentation")
        timeline_off, timeline_stride = _parity_parts(s2.region("timeline_history"), timeline_history_depth)
        timeline_gmm_done_off, timeline_gmm_done_stride = _parity_parts(
            s2.region("timeline_history_gmm_worker_done"), timeline_history_depth
        )
        timeline_generation_off = s2.region("timeline_history_generation").offset
    else:
        timeline_generation_off = 0
    # Route payload/map/arrivals are private to this GPU and share the output
    # backing allocation. Only the reduced contributor inbox is peer-visible.
    push_workspace = RankPushWorkspace.create(
        max_tokens=MAX_TOKENS, hidden=HIDDEN, topk=TOPK,
        max_route_rows=s1.max_route_rows, world_size=WORLD,
    )
    push_workspace_off, push_workspace_stride = push_workspace.workspace_offset, push_workspace.parity_stride
    push_payload_bytes = push_workspace.payload_bytes
    push_map_off, push_map_bytes = push_workspace.row_map_offset, push_workspace.row_map_bytes
    push_arrival_off, push_arrival_bytes = push_workspace.arrival_offset, push_workspace.arrival_bytes
    push_inbox_off, push_inbox_stride = _parity_parts(s2.region("rank_push_inbox"), 2)
    push_node_arrival_off, push_node_arrival_stride = _parity_parts(s2.region("rank_push_arrived"), 2)
    grid_barrier_off = s2.region("grid_barrier").offset
    gemm_head_off = s2.region("gemm_work_head").offset
    final_head_off = s2.region("final_work_head").offset
    final_done_off = s2.region("final_done").offset
    error_off = s2.region("stage2_error_count").offset
    stage1_epoch_off = s1.region("epoch_gate").offset

    # BF16 C-shuffle slab, disjoint next-tile A slab, and eight peer base pointers.
    # Preparing A before the first tile's route stores avoids a later A wait
    # immediately draining those stores before the second tile reaches MFMA.
    # With kStages=2, a_stages=3 provides rotating A storage. The selected
    # rank-local path uses 16,384 + 12,288 + 64 + 256 = 28,992 LDS bytes/CTA
    # without timeline instrumentation; this is an overlap/resource tradeoff.
    a_stages = kStages + 1
    kh_tile_a = BK // 2
    c_shuffle_bytes = BM * BN * 2
    compute_lds_bytes = max(c_shuffle_bytes, a_stages * BM * kh_tile_a)
    a_double_buffer_bytes = a_stages * BM * kh_tile_a
    a_double_buffer_off = compute_lds_bytes
    peer_table_off = compute_lds_bytes + a_double_buffer_bytes
    timeline_scratch_off = peer_table_off + GPUS_PER_NODE * 8
    timeline_scratch_slots = 14
    timeline_scratch_bytes = timeline_scratch_slots * 8 if timeline_instrument else 0
    epilogue_meta_off = timeline_scratch_off + timeline_scratch_bytes
    epilogue_weight_off = epilogue_meta_off + BM * 4
    lds_bytes = epilogue_meta_off + BM * 8

    @fx.struct
    class SharedStorage:
        raw: fx.Array[Int8, lds_bytes, 16]

    kernel_name = (
        "megamoe_tile_ep16_stage2_reduce_push_"
        f"r{rank}_h{HIDDEN}_i{INTER}_bm{BM}_bn{BN}_bk{BK}_mt{MAX_TOKENS}"
        f"_gs{gmm_schedule}_gws{gmm_work_swizzle}{window_n_groups}"
        + (f"_e{EXPERTS}" if EXPERTS != 896 else "")
        + (f"_k{TOPK}" if TOPK != 16 else "")
        + ("_gwudiv" if gmm_work_unsigned_window_division else "")
        + ("_gmdirect" if gmm_direct_tile_coords else "")
        + ("_gmticketpeer" if gmm_queue_ticket_peer_slab else "")
        + ("_gmheadelide" if gmm_queue_head_barrier_elision else "")
        + f"_nt{int(gemm_use_nt)}_pnt{int(rank_push_use_nt)}"
        + f"_nr{node_reduce_blocks}_fc{final_combine_blocks}_rt{return_chunk_tokens}"
        + f"_rpq{rank_reduce_blocks}_rprows{s1.max_route_rows}_rpb{rank_push_batch_size}"
        + ("_rpbinv" if rank_push_batch_invariants else "")
        + ("_rpcprefetch" if rank_push_count_prefetch else "")
        + ("_rpcwave" if rank_push_count_wave_broadcast else "")
        + ("_rpmapprefetch" if rank_push_map_prefetch else "")
        + ("_rpcmap" if rank_push_count_from_slot_map else "")
        + ("_rptileidx" if rank_push_tile_index_recurrence else "")
        + ("_rppsleep1" if rank_push_poll_backoff else "")
        + ("_rpacq2" if rank_push_acquire_cohort == 2 else "")
        + ("_rpsingle" if rank_push_single_row_fastpath else "")
        + ("_rppbitmap" if rank_push_publication == "batch_bitmap" else "")
        + ("_rpppacked" if rank_push_publication == "packed_tile_bitmap" else "")
        + ("_nrtokenowner" if node_reduce_token_owner_fastpath else "")
        + ("_nrmaskprefetch" if node_reduce_token_owner_mask_prefetch else "")
        + ("_nrexpectpopcount" if node_reduce_expected_popcount else "")
        + ("_rebper_tile" if rank_epilogue_barrier == "per_tile" else "")
        + ("_timeline" if timeline_instrument else "")
        + (f"_history{timeline_history_depth}" if timeline_history_depth else "")
        + ("_devgen" if device_generation else "")
        + ("" if diagnostic_mode == "full" else f"_{diagnostic_mode}")
    )
    if kernel_name_override is not None:
        kernel_name = str(kernel_name_override)

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        generation: fx.Int64,
        local_tokens: fx.Int32,
        arg_output_bf16: fx.Int64,
    ):
        arena_window = cco.Window(arena_win)
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        grid = fx.Int32(gpu.grid_dim.x)
        global_tx = bx * fx.Int32(THREADS) + tx
        grid_threads = grid * fx.Int32(THREADS)
        if const_expr(device_generation):
            # Stage1 precedes Stage2 on the same stream and publishes its
            # device epoch here. This scalar load needs no extra GPU launch
            # and observes the current replay, including its parity.
            # Same-stream Stage1/epoch kernel completion supplies ordering for
            # this local immutable scalar, just as for H1 and route metadata.
            generation = fx.ptr_load(global_typed_ptr(
                arena_ptr + fx.Int64(stage1_epoch_off), T.i64, align=8))
        parity = generation & fx.Int64(1)
        s2_base = arena_ptr + fx.Int64(s2_window_off)
        runtime_return_batches = (
            local_tokens
            + fx.Int32(s2.num_qp * return_chunk_tokens - 1)
        ) // fx.Int32(s2.num_qp * return_chunk_tokens)

        lds_raw = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_base = fx.Int32(fx.ptrtoint(lds_raw))
        work_ptr = lds_typed_ptr(lds_base, T.i32, align=4)
        timeline_scratch = lds_typed_ptr(
            lds_base + fx.Int32(timeline_scratch_off), T.i64, align=8
        )

        def s1_ptr(offset, stride):
            return arena_ptr + fx.Int64(offset) + parity * fx.Int64(stride)

        def s2_ptr(offset, stride):
            return s2_base + fx.Int64(offset) + parity * fx.Int64(stride)

        push_private_ptr = arg_output_bf16 + fx.Int64(push_workspace_off) + parity * fx.Int64(push_workspace_stride)
        push_map_ptr = push_private_ptr + fx.Int64(push_map_off)
        push_arrival_ptr = push_private_ptr + fx.Int64(push_arrival_off)
        push_inbox_ptr = s2_ptr(push_inbox_off, push_inbox_stride)
        push_node_arrival_ptr = s2_ptr(push_node_arrival_off, push_node_arrival_stride)
        arg_aq = s1_ptr(h1_q_off, h1_q_stride)
        arg_ascale = s1_ptr(h1_scale_off, h1_scale_stride)
        arg_eids = s1_ptr(expert_off, expert_stride)
        arg_nvalid = s1_ptr(nvalid_off, nvalid_stride)
        arg_stids = s1_ptr(source_off, source_stride)
        arg_sweights = s1_ptr(weight_off, weight_stride)
        dest_rank_mask_ptr = s2_ptr(
            dest_rank_mask_off, dest_rank_mask_stride
        )
        accumulator_ptr = s2_ptr(accumulator_off, accumulator_stride)
        rank_pending_ptr = s2_ptr(rank_pending_off, rank_pending_stride)
        node_ready_mask_ptr = s2_ptr(node_ready_mask_off, node_ready_mask_stride)
        partial_ready_ptr = s2_ptr(partial_ready_off, partial_ready_stride)
        rail_payload_ready_ptr = (
            (partial_ready_ptr)
        )
        rx_ptr = s2_ptr(rx_off, rx_stride)
        return_ready_ptr = s2_ptr(return_ready_off, return_ready_stride)
        consumed_ptr = s2_ptr(consumed_off, consumed_stride)
        stage2_phase_ptr = s2_ptr(stage2_phase_off, stage2_phase_stride)
        timeline_ptr = s2_ptr(timeline_off, timeline_stride)
        timeline_gmm_done_ptr = s2_ptr(
            timeline_gmm_done_off, timeline_gmm_done_stride
        )
        if const_expr(timeline_history_depth > 0):
            timeline_slot = generation & fx.Int64(timeline_history_depth - 1)
            timeline_ptr = s2_base + fx.Int64(timeline_off) + timeline_slot * fx.Int64(timeline_stride)
            timeline_gmm_done_ptr = s2_base + fx.Int64(timeline_gmm_done_off) + timeline_slot * fx.Int64(timeline_gmm_done_stride)
        grid_barrier_ptr = s2_base + fx.Int64(grid_barrier_off)
        gemm_head_ptr = s2_base + fx.Int64(gemm_head_off)
        final_head_ptr = s2_base + fx.Int64(final_head_off)
        final_done_ptr = s2_base + fx.Int64(final_done_off)
        error_ptr = s2_base + fx.Int64(error_off)

        def add_error():
            comm_ops.atomic_add_system(error_ptr, fx.Int32(1))

        def grid_sync():
            gpu.barrier()
            rocdl.s_waitcnt(0)
            # Every wave must first drain its own vector-memory operations,
            # then reconverge before thread zero releases the CTA into the
            # agent-scope ticket RMW sequence.  The acquire below and final CTA
            # barrier make all participating CTAs' writes visible afterwards.
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_agent_release()
                ticket = fx.Int64(
                    comm_ops.atomic_add_agent(grid_barrier_ptr, fx.Int64(1))
                )
                grid64 = fx.Int64(grid)
                # A resident grid is a tunable resource budget. For any
                # power-of-two budget, rounding this ticket up to the next
                # grid boundary is exactly OR(mask)+1. This avoids the long
                # runtime i64 division sequence on gfx950 (twice per call for
                # every CTA); non-power-of-two budgets retain the same formula.
                # No generation/route assumption enters the barrier target.
                target = (ticket | (grid64 - fx.Int64(1))) + fx.Int64(1)
                if (grid & (grid - fx.Int32(1))) != fx.Int32(0):
                    target = ((ticket // grid64) + fx.Int64(1)) * grid64
                seen = fx.Int64(
                    comm_ops.load_i64_global_agent_relaxed(grid_barrier_ptr)
                )
                while seen < target:
                    seen = fx.Int64(
                        comm_ops.load_i64_global_agent_relaxed(grid_barrier_ptr)
                    )
                comm_ops.fence_agent_acquire()
            gpu.barrier()

        if const_expr(timeline_instrument):
            if bx == fx.Int32(CROSS_BLOCK) and tx == fx.Int32(0):
                if const_expr(timeline_history_depth > 0):
                    comm_ops.store_i64_global_relaxed(
                        s2_base + fx.Int64(timeline_generation_off) + timeline_slot * fx.Int64(8),
                        generation,
                    )
                comm_ops.store_i64_global_relaxed(
                    timeline_ptr
                    + fx.Int64(STAGE2_TIMELINE_INDEX["stage2_entry"] * 8),
                    fx.Int64(comm_ops.read_wall_clock()),
                )

        # The preceding Stage1 kernel on this stream supplies local inputs.
        # Peer inbox initialization is acquired lazily by each push wave.
        if const_expr(timeline_instrument):
            if bx == fx.Int32(CROSS_BLOCK) and tx == fx.Int32(0):
                comm_ops.store_i64_global_relaxed(
                    timeline_ptr
                    + fx.Int64(
                        STAGE2_TIMELINE_INDEX["stage2_stage1_gate_done"] * 8
                    ),
                    fx.Int64(comm_ops.read_wall_clock()),
                )

        if bx == fx.Int32(0):
            if tx < fx.Int32(WORK_SHARDS):
                buffer_ops.buffer_store(
                    fx.Int32(0), buffer_ops.create_buffer_resource_from_addr(gemm_head_ptr),
                    tx * fx.Int32(16),
                )
            if tx == fx.Int32(0):
                # Only this GPU consumes these control words. The grid's
                # release/acquire gates order the stores before consumers;
                # peer writers are gated separately by stage2_init below.
                # Per-word system releases would redundantly flush L2 here.
                buffer_ops.buffer_store(
                    fx.Int32(0), buffer_ops.create_buffer_resource_from_addr(final_head_ptr), fx.Int32(0))
                buffer_ops.buffer_store(
                    fx.Int32(0), buffer_ops.create_buffer_resource_from_addr(final_done_ptr), fx.Int32(0))
                buffer_ops.buffer_store(
                    fx.Int32(0), buffer_ops.create_buffer_resource_from_addr(error_ptr), fx.Int32(0))

        # Every output element is overwritten by its owner, including absent
        # node zeros. Clear only generation-local metadata, never payloads.
        node_ready_mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
            node_ready_mask_ptr
        )
        for item in range(
            global_tx, fx.Int32(2 * MAX_TOKENS), grid_threads
        ):
            buffer_ops.buffer_store(
                fx.Int64(0), node_ready_mask_rsrc, item
            )
        # Map zero means absent slot, while a valid sorted row is row+1.
        # Only this generation's counters/map are cleared; neither
        # private payload nor the peer inbox needs a full zero pass.
        push_map_rsrc = buffer_ops.create_buffer_resource_from_addr(push_map_ptr)
        push_arrival_rsrc = buffer_ops.create_buffer_resource_from_addr(push_arrival_ptr)
        push_node_arrival_rsrc = buffer_ops.create_buffer_resource_from_addr(push_node_arrival_ptr)
        for item in range(global_tx, fx.Int32(push_map_bytes // 4), grid_threads):
            buffer_ops.buffer_store(fx.Int32(0), push_map_rsrc, item)
        for item in range(global_tx, fx.Int32(push_arrival_bytes // 4), grid_threads):
            buffer_ops.buffer_store(fx.Int32(0), push_arrival_rsrc, item)
        for item in range(global_tx, fx.Int32(2 * MAX_TOKENS * ready_groups), grid_threads):
            buffer_ops.buffer_store(fx.Int32(0), push_node_arrival_rsrc, item)
        rank_pending_rsrc = buffer_ops.create_buffer_resource_from_addr(
            rank_pending_ptr
        )
        for item in range(global_tx, fx.Int32(SOURCE_CAPACITY), grid_threads):
            buffer_ops.buffer_store(fx.Int32(0), rank_pending_rsrc, item * fx.Int32(16))
        # Clear -> count/map: a late zero must never erase a route increment.
        grid_sync()
        if const_expr(timeline_instrument):
            if bx == fx.Int32(0) and tx == fx.Int32(0):
                comm_ops.store_i64_global_relaxed(
                    timeline_ptr + fx.Int64(STAGE2_TIMELINE_INDEX["stage2_init_clear_done"] * 8),
                    fx.Int64(comm_ops.read_wall_clock()))
        rank_meta_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_stids
        )
        rank_num_valid = (
            (fx.Int32(global_typed_ptr(arg_nvalid, T.i32)[0]))
        )
        # Partition rows across the entire grid with global_tx/grid_threads.
        # Using tx/THREADS repeats every row on every CTA, multiplying pending
        # by grid size and leaving reducers waiting for nonexistent contributors.
        for row in range(
            global_tx, rank_num_valid, grid_threads
        ):
            packed = buffer_ops.buffer_load(
                rank_meta_rsrc, row, vec_width=1, dtype=T.i32
            )
            source = packed & fx.Int32(0x00FFFFFF)
            if source < fx.Int32(SOURCE_CAPACITY):
                prior_count = fx.Int32(comm_ops.atomic_add_agent(
                    rank_pending_ptr
                    + fx.Int64(source) * fx.Int64(64),
                    fx.Int32(
                        (1)
                    ),
                ))
                if const_expr(rank_push_single_row_fastpath):
                    # rank_token_pending reserves a full 64-byte line per
                    # source. Cache the first row in its second dword. A
                    # source with count one has exactly one writer, while
                    # count>1 continues through the original-slot map below.
                    if prior_count == fx.Int32(0):
                        buffer_ops.buffer_store(
                            row + fx.Int32(1), rank_pending_rsrc,
                            source * fx.Int32(16) + fx.Int32(1))
                slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
                if slot < fx.Int32(TOPK):
                    buffer_ops.buffer_store(row + fx.Int32(1), push_map_rsrc,
                                            source * fx.Int32(TOPK) + slot)
                else:
                    add_error()

        if bx == fx.Int32(0) and tx == fx.Int32(0):
            if local_tokens != fx.Int32(MAX_TOKENS):
                add_error()
        # Count/map -> consumers: acquire all producers before either GEMM or
        # rank reducers read metadata. Absent nodes are handled by node reducers;
        # no rank-wide watermark publication or additional grid gate is needed.
        grid_sync()
        if const_expr(timeline_instrument):
            if bx == fx.Int32(0) and tx == fx.Int32(0):
                comm_ops.store_i64_global_relaxed(
                    timeline_ptr + fx.Int64(STAGE2_TIMELINE_INDEX["stage2_init_count_done"] * 8),
                    fx.Int64(comm_ops.read_wall_clock()))

        # Publish completion of local inbox/map/count initialization. Every
        # push wave acquires its target before publishing a tile arrival.
        init_phase = (generation << fx.Int64(2)) + fx.Int64(1)
        if bx == fx.Int32(0) and tx == fx.Int32(0):
            comm_ops.store_i64_global_system(
                stage2_phase_ptr, init_phase
            )
        if const_expr(timeline_instrument):
            if bx == fx.Int32(CROSS_BLOCK) and tx == fx.Int32(0):
                comm_ops.store_i64_global_relaxed(
                    timeline_ptr
                    + fx.Int64(
                        STAGE2_TIMELINE_INDEX["stage2_init_gate_done"] * 8
                    ),
                    fx.Int64(comm_ops.read_wall_clock()),
                )

        # Each CTA caches all local LSA Stage-2 bases. The epilogue indexes this
        # table dynamically using source_rank % 8.
        if tx < fx.Int32(GPUS_PER_NODE):
            peer_base = fx.Int64(
                arena_window.lsa_ptr(tx, fx.Int64(s2_window_off))
            )
            fx.ptr_store(
                peer_base,
                lds_typed_ptr(
                    lds_base + fx.Int32(peer_table_off) + tx * fx.Int32(8),
                    T.i64,
                    align=8,
                ),
            )
        gpu.barrier()

        role_enabled = fx.Int32(1 if diagnostic_mode == "full" else 0)
        compute_enabled = fx.Int32(0 if diagnostic_mode == "init_only" else 1)
        reduce_enabled = compute_enabled

        # ---- RAIL: return complete node tokens in tunable contiguous chunks ----
        if (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)) & (
            fx.Int32(1 if return_chunk_tokens > 4 else 0) == fx.Int32(1)
        ):
            qp = wave
            for batch in range(
                fx.Int32(0),
                runtime_return_batches,
                fx.Int32(1),
            ):
                chunk = batch * fx.Int32(s2.num_qp) + qp
                first_token = chunk * fx.Int32(return_chunk_tokens)
                if lane < fx.Int32(return_chunk_tokens):
                    token = first_token + lane
                    token_ready_index = (
                        fx.Int32(remote_plane * MAX_TOKENS) + token
                    )
                    comm_ops.spin_until_ge_i64_system(
                        rail_payload_ready_ptr
                        + fx.Int64(token_ready_index) * fx.Int64(8),
                        generation,
                    )
                if (fx.Int32(1 if timeline_instrument else 0) == fx.Int32(1)) & (
                    batch == fx.Int32(0)
                ):
                    if lane == fx.Int32(0):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + wave,
                        )
                gpu.barrier()
                if (fx.Int32(1 if timeline_instrument else 0) == fx.Int32(1)) & (
                    batch == fx.Int32(0)
                ):
                    if tx == fx.Int32(0):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + fx.Int32(4),
                        )
                comm_ops.fence_system_acquire()
                src_rel = (
                    fx.Int64(accumulator_off)
                    + parity * fx.Int64(accumulator_stride)
                    + fx.Int64(remote_plane * MAX_TOKENS + first_token)
                    * fx.Int64(wire.record_bytes)
                )
                dst_rel = (
                    fx.Int64(rx_off)
                    + parity * fx.Int64(rx_stride)
                    + fx.Int64(first_token) * fx.Int64(wire.record_bytes)
                )
                _rail.put(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                    arena_win,
                    fx.Int64(s2_window_off) + dst_rel,
                    arena_win,
                    fx.Int64(s2_window_off) + src_rel,
                    fx.Int64(return_chunk_tokens * wire.record_bytes),
                    aggregate=True,
                )
                if (fx.Int32(1 if timeline_instrument else 0) == fx.Int32(1)) & (
                    batch == fx.Int32(0)
                ):
                    if lane == fx.Int32(0):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + fx.Int32(5) + wave,
                        )
                gpu.barrier()
                if (fx.Int32(1 if timeline_instrument else 0) == fx.Int32(1)) & (
                    batch == fx.Int32(0)
                ):
                    if tx == fx.Int32(0):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + fx.Int32(9),
                        )
                if wave == fx.Int32(0):
                    for stream_qp in range_constexpr(s2.num_qp):
                        ready_slot = batch * s2.num_qp + stream_qp
                        ready_rel = (
                            fx.Int64(return_ready_off)
                            + parity * fx.Int64(return_ready_stride)
                            + fx.Int64(ready_slot * 8)
                        )
                        _rail.put_value(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            arena_win,
                            fx.Int64(s2_window_off) + ready_rel,
                            generation,
                            aggregate=True,
                        )
                        if (
                            fx.Int32(
                                1
                                if timeline_instrument and stream_qp == 0
                                else 0
                            )
                            == fx.Int32(1)
                        ) & (batch == fx.Int32(0)):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(10),
                                )
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(11),
                                )
                        request = _rail.flush_async(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                        )
                        if (
                            fx.Int32(
                                1
                                if timeline_instrument and stream_qp == 0
                                else 0
                            )
                            == fx.Int32(1)
                        ) & (batch == fx.Int32(0)):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(12),
                                )
                        _rail.wait(
                            dev_comm,
                            fx.Int32(stream_qp),
                            request,
                        )
                        if (
                            fx.Int32(
                                1
                                if timeline_instrument and stream_qp == 0
                                else 0
                            )
                            == fx.Int32(1)
                        ) & (batch == fx.Int32(0)):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(13),
                                )
                                for marker_qp in range_constexpr(s2.num_qp):
                                    comm_ops.store_i64_global_relaxed(
                                        timeline_ptr
                                        + fx.Int64(
                                            (
                                                STAGE2_TIMELINE_INDEX[
                                                    "stage2_qp0_tokens_ready"
                                                ]
                                                + marker_qp
                                            )
                                            * 8
                                        ),
                                        fx.ptr_load(
                                            timeline_scratch
                                            + fx.Int32(marker_qp)
                                        ),
                                    )
                                    comm_ops.store_i64_global_relaxed(
                                        timeline_ptr
                                        + fx.Int64(
                                            (
                                                STAGE2_TIMELINE_INDEX[
                                                    "stage2_qp0_payload_posted"
                                                ]
                                                + marker_qp
                                            )
                                            * 8
                                        ),
                                        fx.ptr_load(
                                            timeline_scratch
                                            + fx.Int32(5 + marker_qp)
                                        ),
                                    )
                                comm_ops.store_i64_global_relaxed(
                                    timeline_ptr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage2_first_batch_ready"
                                        ]
                                        * 8
                                    ),
                                    fx.ptr_load(
                                        timeline_scratch + fx.Int32(4)
                                    ),
                                )
                                comm_ops.store_i64_global_relaxed(
                                    timeline_ptr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage2_first_batch_payloads_posted"
                                        ]
                                        * 8
                                    ),
                                    fx.ptr_load(
                                        timeline_scratch + fx.Int32(9)
                                    ),
                                )
                                comm_ops.store_i64_global_relaxed(
                                    timeline_ptr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage2_return_terminal_posted"
                                        ]
                                        * 8
                                    ),
                                    fx.ptr_load(
                                        timeline_scratch + fx.Int32(10)
                                    ),
                                )
                                comm_ops.store_i64_global_relaxed(
                                    timeline_ptr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage2_return_flush_pre"
                                        ]
                                        * 8
                                    ),
                                    fx.ptr_load(
                                        timeline_scratch + fx.Int32(11)
                                    ),
                                )
                                comm_ops.store_i64_global_relaxed(
                                    timeline_ptr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage2_return_flush_post"
                                        ]
                                        * 8
                                    ),
                                    fx.ptr_load(
                                        timeline_scratch + fx.Int32(12)
                                    ),
                                )
                                comm_ops.store_i64_global_relaxed(
                                    timeline_ptr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage2_return_request_done"
                                        ]
                                        * 8
                                    ),
                                    fx.ptr_load(
                                        timeline_scratch + fx.Int32(13)
                                    ),
                                )
                gpu.barrier()
                if lane == fx.Int32(0):
                    comm_ops.spin_until_ge_i64_system(
                        return_ready_ptr + fx.Int64(chunk) * fx.Int64(8),
                        generation,
                    )
                gpu.barrier()

            if tx == fx.Int32(0):
                completed = fx.Int32(
                    comm_ops.load_i32_global_system(final_done_ptr)
                )
                while completed < fx.Int32(final_work_items):
                    completed = fx.Int32(
                        comm_ops.load_i32_global_system(final_done_ptr)
                    )
            gpu.barrier()
            if wave == fx.Int32(0):
                remote_consumed = (
                    fx.Int64(s2_window_off + consumed_off)
                    + parity * fx.Int64(consumed_stride)
                )
                _rail.put_value(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    arena_win,
                    remote_consumed,
                    generation,
                    aggregate=True,
                )
                request = _rail.flush_async(
                    dev_comm, wave, fx.Int32(remote_node)
                )
                _rail.wait(dev_comm, wave, request)
                if lane == fx.Int32(0):
                    comm_ops.spin_until_ge_i64_system(consumed_ptr, generation)

        # ---------------------- Role 0: reference return ----------------------
        elif (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)):
            qp = wave
            for group in range(qp, fx.Int32(return_groups), fx.Int32(s2.num_qp)):
                if lane == fx.Int32(0):
                    for record in range_constexpr(wire.records_per_group):
                        token = group * fx.Int32(wire.records_per_group) + fx.Int32(record)
                        token_ready_index = (
                            fx.Int32(remote_plane * MAX_TOKENS) + token
                        )
                        comm_ops.spin_until_ge_i64_system(
                            rail_payload_ready_ptr
                            + fx.Int64(token_ready_index) * fx.Int64(8),
                            generation,
                        )
                gpu.barrier()
                comm_ops.fence_system_acquire()

                for record in range_constexpr(wire.records_per_group):
                    token = group * fx.Int32(wire.records_per_group) + fx.Int32(record)
                    src_rel = (
                        fx.Int64(accumulator_off)
                        + parity * fx.Int64(accumulator_stride)
                        + fx.Int64(remote_plane * MAX_TOKENS + token)
                        * fx.Int64(wire.record_bytes)
                    )
                    dst_rel = (
                        fx.Int64(rx_off)
                        + parity * fx.Int64(rx_stride)
                        + fx.Int64(token) * fx.Int64(wire.record_bytes)
                    )
                    _rail.put(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        fx.Int64(s2_window_off) + dst_rel,
                        arena_win,
                        fx.Int64(s2_window_off) + src_rel,
                        fx.Int64(wire.record_bytes),
                        aggregate=True,
                    )
                ready_rel = (
                    fx.Int64(return_ready_off)
                    + parity * fx.Int64(return_ready_stride)
                    + fx.Int64(group) * fx.Int64(8)
                )
                _rail.put_value(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                    arena_win,
                    fx.Int64(s2_window_off) + ready_rel,
                    generation,
                    aggregate=True,
                )
                request = _rail.flush_async(
                    dev_comm, qp, fx.Int32(remote_node)
                )
                _rail.wait(dev_comm, qp, request)
                if lane == fx.Int32(0):
                    comm_ops.spin_until_ge_i64_system(
                        return_ready_ptr + fx.Int64(group) * fx.Int64(8), generation
                    )
                gpu.barrier()

            # Do not credit remote_rx reuse until all final roles consumed it.
            if tx == fx.Int32(0):
                completed = fx.Int32(comm_ops.load_i32_global_system(final_done_ptr))
                while completed < fx.Int32(final_work_items):
                    completed = fx.Int32(
                        comm_ops.load_i32_global_system(final_done_ptr)
                    )
            gpu.barrier()
            if wave == fx.Int32(0):
                remote_consumed = (
                    fx.Int64(s2_window_off + consumed_off)
                    + parity * fx.Int64(consumed_stride)
                )
                _rail.put_value(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    arena_win,
                    remote_consumed,
                    generation,
                    aggregate=True,
                )
                request = _rail.flush_async(
                    dev_comm, wave, fx.Int32(remote_node)
                )
                _rail.wait(dev_comm, wave, request)
                if lane == fx.Int32(0):
                    comm_ops.spin_until_ge_i64_system(consumed_ptr, generation)

        # ---- Per-token/N-group rank reduction and producer-side peer push ----
        elif (bx >= fx.Int32(1)) & (bx < fx.Int32(1 + rank_push_blocks)) & (
            reduce_enabled == fx.Int32(1)
        ):
            push_values_rsrc = buffer_ops.create_buffer_resource_from_addr(
                push_private_ptr, num_records_bytes=max(4, push_payload_bytes))
            push_rows_rsrc = buffer_ops.create_buffer_resource_from_addr(
                push_map_ptr, num_records_bytes=max(4, push_map_bytes))
            push_counts_rsrc = buffer_ops.create_buffer_resource_from_addr(rank_pending_ptr)
            if const_expr(rank_push_count_prefetch):
                # Masked buffer loads use a large sentinel offset for disabled
                # lanes.  Bound this separate descriptor to the current parity
                # so those lanes return zero instead of addressing the arena.
                push_counts_prefetch_rsrc = (
                    buffer_ops.create_buffer_resource_from_addr(
                        rank_pending_ptr,
                        num_records_bytes=max(4, rank_pending_stride),
                    )
                )
            initialized_peers = fx.Int32(0)
            # A wave computes a bounded batch of independent token/N-group
            # tiles. Payload addresses remain unique; the batch amortizes the
            # system release, without waiting for all N tiles or for EOS.
            # Per-tile publication retains one arrival address in each signal
            # lane. The experimental B8 bitmap instead packs four token masks
            # per i32 word, so each producer issues two disjoint-bit RMWs for
            # eight tiles. The final batch is predicated rather than
            # shape-specialized.
            for batch_base in range(
                ((bx - fx.Int32(1)) * fx.Int32(4) + wave) * fx.Int32(rank_push_batch_size),
                fx.Int32(SOURCE_CAPACITY * ready_groups),
                fx.Int32(max(1, rank_push_blocks) * 4 * rank_push_batch_size),
            ):
                signal_address = fx.Int64(0)
                signal_valid = fx.Int32(0)
                batch_bits_lo = fx.Int32(0)
                batch_bits_hi = fx.Int32(0)
                packed_word_bits = fx.Int32(0)
                if const_expr(
                    rank_push_batch_invariants
                    or rank_push_count_prefetch
                    or rank_push_publication in (
                        "batch_bitmap", "packed_tile_bitmap"
                    )
                ):
                    batch_n_group = batch_base // fx.Int32(SOURCE_CAPACITY)
                    batch_source = batch_base - batch_n_group * fx.Int32(SOURCE_CAPACITY)
                    batch_source_rank = batch_source // fx.Int32(MAX_TOKENS)
                    batch_source_local = batch_source_rank & fx.Int32(7)
                    batch_source_plane = batch_source_rank >> fx.Int32(3)
                    batch_token = batch_source - batch_source_rank * fx.Int32(MAX_TOKENS)
                    batch_token_index = batch_source_plane * fx.Int32(MAX_TOKENS) + batch_token
                    batch_peer_s2 = _wave_uniform_i64(fx.ptr_load(lds_typed_ptr(
                        lds_base + fx.Int32(peer_table_off) + batch_source_local * fx.Int32(8),
                        T.i64, align=8)))
                    batch_signal_lo = (
                        batch_peer_s2 + fx.Int64(push_node_arrival_off)
                        + parity * fx.Int64(push_node_arrival_stride)
                        + fx.Int64(batch_token_index * ready_groups + batch_n_group) * fx.Int64(4))
                    batch_signal_hi = (
                        batch_peer_s2 + fx.Int64(push_node_arrival_off)
                        + parity * fx.Int64(push_node_arrival_stride)
                        + fx.Int64((batch_token_index + fx.Int32(4)) * ready_groups
                                   + batch_n_group) * fx.Int64(4))
                prefetched_count_lane = fx.Int32(0)
                if const_expr(rank_push_count_prefetch):
                    prefetch_task = batch_base + lane
                    prefetch_valid = (
                        (lane < fx.Int32(rank_push_batch_size))
                        & (prefetch_task < fx.Int32(SOURCE_CAPACITY * ready_groups))
                    )
                    prefetched_count_lane = fx.Int32(buffer_ops.buffer_load(
                        push_counts_prefetch_rsrc,
                        (batch_source + lane) * fx.Int32(16),
                        vec_width=1,
                        dtype=T.i32,
                        mask=prefetch_valid,
                    ))
                # Define the loop carrier in the enclosing batch scope before
                # FlyDSL outlines nested route guards into helper regions.
                push_tile_index = fx.Int32(0)
                if const_expr(rank_push_tile_index_recurrence):
                    # Complete batches stay within one source-rank/token plane
                    # and N group. Advance this index even across empty routes.
                    push_tile_index = (
                        batch_token_index * fx.Int32(ready_groups) + batch_n_group
                    )
                # Cohort2 carries the second tile's immutable count and the
                # fact that its arrival was covered by the leader's acquire.
                # Both values are wave-uniform. They are cleared after every
                # odd slot, so cohorts never cross a source-token batch.
                cohort_next_count = fx.Int32(0)
                cohort_next_count_valid = fx.Int32(0)
                cohort_next_acquired = fx.Int32(0)
                for batch_slot in range(fx.Int32(0), fx.Int32(rank_push_batch_size), fx.Int32(1)):
                    push_task = batch_base + batch_slot
                    if push_task < fx.Int32(SOURCE_CAPACITY * ready_groups):
                        if const_expr(rank_push_batch_invariants):
                            n_group = batch_n_group
                            source = batch_source + batch_slot
                        else:
                            n_group = push_task // fx.Int32(SOURCE_CAPACITY)
                            source = push_task - n_group * fx.Int32(SOURCE_CAPACITY)
                        slot_row = fx.Int32(0)
                        slot_mask = fx.Int32(0)
                        if const_expr(rank_push_acquire_cohort == 2):
                            cohort_position = batch_slot & fx.Int32(1)
                            use_cohort_count = cohort_position * cohort_next_count_valid
                            local_count_lane = fx.Int32(0)
                            if (lane == fx.Int32(0)) & (
                                use_cohort_count == fx.Int32(0)
                            ):
                                local_count_lane = fx.Int32(buffer_ops.buffer_load(
                                    push_counts_rsrc,
                                    source * fx.Int32(16),
                                    vec_width=1,
                                    dtype=T.i32,
                                ))
                            loaded_local_count = fx.Int32(rocdl.readfirstlane(
                                T.i32, local_count_lane.ir_value()))
                            local_count = (
                                use_cohort_count != fx.Int32(0)
                            ).select(cohort_next_count, loaded_local_count)
                        elif const_expr(rank_push_count_from_slot_map):
                            # The map is immutable after the grid-init gate.
                            # Its occupied original slots are exactly the local
                            # payload publishers for every valid route input.
                            slot_row = fx.Int32(buffer_ops.buffer_load(
                                push_rows_rsrc,
                                source * fx.Int32(TOPK) + lane,
                                vec_width=1,
                                dtype=T.i32,
                                mask=lane < fx.Int32(TOPK),
                            ))
                            slot_mask = fx.Int32(rocdl.ballot(
                                T.i64,
                                (lane < fx.Int32(TOPK))
                                & (slot_row > fx.Int32(0)),
                            ))
                            local_count = fx.Int32(_llvm_d.call_intrinsic(
                                T.i32,
                                "llvm.ctpop.i32",
                                [slot_mask.ir_value()],
                                [],
                                [],
                            ))
                        elif const_expr(rank_push_count_prefetch):
                            local_count = fx.Int32(rocdl.readlane(
                                T.i32,
                                prefetched_count_lane,
                                batch_slot,
                            ))
                        elif const_expr(rank_push_count_wave_broadcast):
                            # ``source`` is wave-uniform. Counts are immutable
                            # after the acquired grid-init gate, so each lane
                            # may issue the same valid load. Keep readfirstlane
                            # to preserve a scalar count for polling/control.
                            local_count_lane = fx.Int32(buffer_ops.buffer_load(
                                push_counts_rsrc,
                                source * fx.Int32(16),
                                vec_width=1,
                                dtype=T.i32,
                            ))
                            local_count = fx.Int32(rocdl.readfirstlane(
                                T.i32, local_count_lane.ir_value()))
                        else:
                            local_count_lane = fx.Int32(0)
                            if lane == fx.Int32(0):
                                # Counts are immutable after the acquired grid init gate.
                                # Reading local metadata needs no per-task system acquire.
                                local_count_lane = fx.Int32(buffer_ops.buffer_load(
                                    push_counts_rsrc, source * fx.Int32(16), vec_width=1, dtype=T.i32))
                            local_count = fx.Int32(rocdl.readfirstlane(
                                T.i32, local_count_lane.ir_value()))
                        if local_count > fx.Int32(0):
                            if const_expr(rank_push_map_prefetch):
                                # The map is immutable after the grid init gate.
                                # Issue its VMEM load before waiting for payload
                                # arrivals, but consume it only after the acquire.
                                slot_row = fx.Int32(buffer_ops.buffer_load(
                                    push_rows_rsrc,
                                    source * fx.Int32(TOPK) + lane,
                                    vec_width=1,
                                    dtype=T.i32,
                                    mask=lane < fx.Int32(TOPK),
                                ))
                            if const_expr(rank_push_acquire_cohort == 2):
                                acquire_covered = (
                                    cohort_position * cohort_next_acquired
                                )
                                if acquire_covered == fx.Int32(0):
                                    next_local_count_lane = fx.Int32(0)
                                    next_task = push_task + fx.Int32(1)
                                    next_task_valid = next_task < fx.Int32(
                                        SOURCE_CAPACITY * ready_groups
                                    )
                                    if lane == fx.Int32(0):
                                        arrival_addr = (
                                            push_arrival_ptr
                                            + fx.Int64(
                                                source * ready_groups + n_group
                                            )
                                            * fx.Int64(4)
                                        )
                                        arrived = fx.Int32(
                                            comm_ops.load_i32_global_agent_relaxed(
                                                arrival_addr
                                            )
                                        )
                                        while arrived < local_count:
                                            if const_expr(rank_push_poll_backoff):
                                                _rank_push_poll_sleep()
                                            arrived = fx.Int32(
                                                comm_ops.load_i32_global_agent_relaxed(
                                                    arrival_addr
                                                )
                                            )
                                        if arrived != local_count:
                                            add_error()
                                        if (
                                            cohort_position == fx.Int32(0)
                                        ) & next_task_valid:
                                            # Batch invariants and the even
                                            # cohort size guarantee source+1
                                            # stays in this source-rank/token
                                            # plane and shares n_group.
                                            next_source = source + fx.Int32(1)
                                            next_local_count_lane = fx.Int32(
                                                buffer_ops.buffer_load(
                                                    push_counts_rsrc,
                                                    next_source * fx.Int32(16),
                                                    vec_width=1,
                                                    dtype=T.i32,
                                                )
                                            )
                                    if cohort_position == fx.Int32(0):
                                        next_local_count = fx.Int32(
                                            rocdl.readfirstlane(
                                                T.i32,
                                                next_local_count_lane.ir_value(),
                                            )
                                        )
                                        cohort_next_count = next_local_count
                                        cohort_next_count_valid = (
                                            next_task_valid
                                        ).select(fx.Int32(1), fx.Int32(0))
                                        if lane == fx.Int32(0):
                                            if next_task_valid & (
                                                next_local_count > fx.Int32(0)
                                            ):
                                                next_arrival_addr = (
                                                    push_arrival_ptr
                                                    + fx.Int64(
                                                        (source + fx.Int32(1))
                                                        * ready_groups
                                                        + n_group
                                                    )
                                                    * fx.Int64(4)
                                                )
                                                next_arrived = fx.Int32(
                                                    comm_ops.load_i32_global_agent_relaxed(
                                                        next_arrival_addr
                                                    )
                                                )
                                                while next_arrived < next_local_count:
                                                    if const_expr(rank_push_poll_backoff):
                                                        _rank_push_poll_sleep()
                                                    next_arrived = fx.Int32(
                                                        comm_ops.load_i32_global_agent_relaxed(
                                                            next_arrival_addr
                                                        )
                                                    )
                                                if next_arrived != next_local_count:
                                                    add_error()
                                        cohort_next_acquired = (
                                            next_local_count > fx.Int32(0)
                                        ).select(fx.Int32(1), fx.Int32(0))
                                    # This must remain wave-wide: the lane-zero
                                    # relaxed observations dominate all map and
                                    # payload loads for both covered tiles.
                                    comm_ops.fence_agent_acquire()
                            else:
                                if lane == fx.Int32(0):
                                    arrival_addr = push_arrival_ptr + fx.Int64(source * ready_groups + n_group) * fx.Int64(4)
                                    arrived = fx.Int32(comm_ops.load_i32_global_agent_relaxed(arrival_addr))
                                    while arrived < local_count:
                                        if const_expr(rank_push_poll_backoff):
                                            _rank_push_poll_sleep()
                                        arrived = fx.Int32(comm_ops.load_i32_global_agent_relaxed(arrival_addr))
                                    if arrived != local_count:
                                        add_error()
                                comm_ops.fence_agent_acquire()
                            totals = fx.Vector.filled(8, 0.0, Float32)
                            if const_expr(rank_push_single_row_fastpath):
                                if local_count == fx.Int32(1):
                                    direct_row_lane = fx.Int32(0)
                                    if lane == fx.Int32(0):
                                        direct_row_lane = fx.Int32(buffer_ops.buffer_load(
                                            push_counts_rsrc,
                                            source * fx.Int32(16) + fx.Int32(1),
                                            vec_width=1, dtype=T.i32))
                                    row_plus_one = fx.Int32(rocdl.readfirstlane(
                                        T.i32, direct_row_lane.ir_value()))
                                    if row_plus_one <= fx.Int32(0):
                                        add_error()
                                    data_index = ((row_plus_one - fx.Int32(1)) * fx.Int32(HIDDEN)
                                                  + n_group * fx.Int32(2 * BN)
                                                  + lane * fx.Int32(8)) // fx.Int32(2)
                                    words = fx.Vector(buffer_ops.buffer_load(
                                        push_values_rsrc, data_index,
                                        vec_width=4, dtype=T.i32))
                                    totals = totals + words.bitcast(BFloat16).to(Float32)
                                else:
                                    slot_row = fx.Int32(buffer_ops.buffer_load(
                                        push_rows_rsrc, source * fx.Int32(TOPK) + lane,
                                        vec_width=1, dtype=T.i32,
                                        mask=lane < fx.Int32(TOPK)))
                                    slot_mask = fx.Int32(rocdl.ballot(
                                        T.i64, (lane < fx.Int32(TOPK))
                                        & (slot_row > fx.Int32(0))))
                            else:
                                # Read the complete slot map in one coalesced
                                # wave load. Keep original-slot order so
                                # duplicate experts with distinct weights stay
                                # independent contributions.
                                if const_expr(
                                    not rank_push_map_prefetch
                                    and not rank_push_count_from_slot_map
                                ):
                                    slot_row = fx.Int32(buffer_ops.buffer_load(
                                        push_rows_rsrc, source * fx.Int32(TOPK) + lane,
                                        vec_width=1, dtype=T.i32,
                                        mask=lane < fx.Int32(TOPK)))
                                if const_expr(not rank_push_count_from_slot_map):
                                    slot_mask = fx.Int32(rocdl.ballot(
                                        T.i64, (lane < fx.Int32(TOPK))
                                        & (slot_row > fx.Int32(0))))
                            # Visit set bits from low to high: same original-slot sum
                            # order, and work proportional to actual local fan-in. A
                            # runtime loop also avoids unrolling TOPK copies per batch.
                            while slot_mask != fx.Int32(0):
                                slot = fx.Int32(_llvm_d.inline_asm(
                                    T.i32, [arith.unwrap(slot_mask)],
                                    "s_ff1_i32_b32 $0, $1", "=s,s", has_side_effects=False))
                                row_plus_one = fx.Int32(rocdl.readlane(T.i32, slot_row, slot))
                                data_index = ((row_plus_one - fx.Int32(1)) * fx.Int32(HIDDEN)
                                              + n_group * fx.Int32(2 * BN) + lane * fx.Int32(8)) // fx.Int32(2)
                                words = fx.Vector(buffer_ops.buffer_load(
                                    push_values_rsrc, data_index, vec_width=4, dtype=T.i32))
                                totals = totals + words.bitcast(BFloat16).to(Float32)
                                slot_mask = slot_mask & (slot_mask - fx.Int32(1))
                            if const_expr(rank_push_batch_invariants):
                                source_rank = batch_source_rank
                                source_local = batch_source_local
                                source_plane = batch_source_plane
                                token = batch_token + batch_slot
                                token_index = batch_token_index + batch_slot
                            else:
                                source_rank = source // fx.Int32(MAX_TOKENS)
                                source_local = source_rank & fx.Int32(7)
                                source_plane = source_rank >> fx.Int32(3)
                                token = source - source_rank * fx.Int32(MAX_TOKENS)
                                token_index = (
                                    source_plane * fx.Int32(MAX_TOKENS) + token
                                )
                            if const_expr(
                                rank_push_batch_invariants
                                or rank_push_publication in (
                                    "batch_bitmap", "packed_tile_bitmap"
                                )
                            ):
                                peer_s2 = batch_peer_s2
                            else:
                                peer_s2 = _wave_uniform_i64(fx.ptr_load(lds_typed_ptr(
                                    lds_base + fx.Int32(peer_table_off) + source_local * fx.Int32(8),
                                    T.i64, align=8)))
                            peer_bit = fx.Int32(1) << source_local
                            if (initialized_peers & peer_bit) == fx.Int32(0):
                                if lane == fx.Int32(0):
                                    # Initialization stays complete for this generation.
                                    # Acquire each target lazily once per wave, preserving
                                    # the late-peer guard without a P2P read per tile.
                                    comm_ops.spin_until_ge_i64_system(peer_s2 + fx.Int64(stage2_phase_off)
                                               + parity * fx.Int64(stage2_phase_stride), init_phase)
                                initialized_peers = initialized_peers | peer_bit
                            peer_inbox = buffer_ops.create_buffer_resource_from_addr(
                                peer_s2 + fx.Int64(push_inbox_off) + parity * fx.Int64(push_inbox_stride),
                                num_records_bytes=max(4, push_inbox_stride))
                            if const_expr(rank_push_tile_index_recurrence):
                                inbox_index = (
                                    (push_tile_index * fx.Int32(GPUS_PER_NODE)
                                     + fx.Int32(local_rank))
                                    * fx.Int32(2 * BN)
                                )
                            else:
                                inbox_index = (
                                    (token_index * fx.Int32(ready_groups) + n_group)
                                    * fx.Int32(GPUS_PER_NODE) + fx.Int32(local_rank)
                                ) * fx.Int32(2 * BN)
                            packed = totals.to(BFloat16).bitcast(fx.Int32)
                            buffer_ops.buffer_store(packed, peer_inbox,
                                                    (inbox_index + lane * fx.Int32(8)) // fx.Int32(2),
                                                    cache_modifier=2 if rank_push_use_nt else 0)
                            if const_expr(rank_push_publication == "batch_bitmap"):
                                bit = fx.Int32(1) << (
                                    (batch_slot & fx.Int32(3)) * fx.Int32(8)
                                    + fx.Int32(local_rank))
                                batch_bits_lo = (batch_slot < fx.Int32(4)).select(
                                    batch_bits_lo | bit, batch_bits_lo)
                                batch_bits_hi = (batch_slot >= fx.Int32(4)).select(
                                    batch_bits_hi | bit, batch_bits_hi)
                            elif const_expr(
                                rank_push_publication == "packed_tile_bitmap"
                            ):
                                # One lane owns each group of four adjacent
                                # tokens. Producer ranks set disjoint bits in
                                # each byte, so integer add is equivalent to OR.
                                word_lane = batch_slot // fx.Int32(4)
                                bit = fx.Int32(1) << (
                                    (batch_slot & fx.Int32(3)) * fx.Int32(8)
                                    + fx.Int32(local_rank)
                                )
                                packed_word_bits = (lane == word_lane).select(
                                    packed_word_bits | bit, packed_word_bits
                                )
                            else:
                                if const_expr(rank_push_tile_index_recurrence):
                                    arrival_address = (
                                        peer_s2 + fx.Int64(push_node_arrival_off)
                                        + parity * fx.Int64(push_node_arrival_stride)
                                        + fx.Int64(push_tile_index) * fx.Int64(4)
                                    )
                                else:
                                    arrival_address = (
                                        peer_s2 + fx.Int64(push_node_arrival_off)
                                        + parity * fx.Int64(push_node_arrival_stride)
                                        + fx.Int64(token_index * ready_groups + n_group)
                                        * fx.Int64(4)
                                    )
                                signal_address = (lane == batch_slot).select(arrival_address, signal_address)
                                signal_valid = (lane == batch_slot).select(fx.Int32(1), signal_valid)
                    if const_expr(rank_push_acquire_cohort == 2):
                        if (batch_slot & fx.Int32(1)) != fx.Int32(0):
                            cohort_next_count_valid = fx.Int32(0)
                            cohort_next_acquired = fx.Int32(0)
                    if const_expr(rank_push_tile_index_recurrence):
                        push_tile_index = push_tile_index + fx.Int32(ready_groups)
                # Drain every payload-writing lane before the single release.
                # A release fence followed by relaxed RMW publishes all batch
                # payloads to the consumer's acquire load of each arrival word.
                # A lane signals only a tile with an actual local contribution.
                rocdl.s_waitcnt(0)
                comm_ops.fence_system_release()
                if const_expr(rank_push_publication == "batch_bitmap"):
                    if lane == fx.Int32(0):
                        if batch_bits_lo != fx.Int32(0):
                            comm_ops.atomic_add_system(batch_signal_lo, batch_bits_lo)
                        if batch_bits_hi != fx.Int32(0):
                            comm_ops.atomic_add_system(batch_signal_hi, batch_bits_hi)
                elif const_expr(rank_push_publication == "packed_tile_bitmap"):
                    if lane < fx.Int32(rank_push_batch_size // 4):
                        if packed_word_bits != fx.Int32(0):
                            word_token_index = (
                                batch_token_index + lane * fx.Int32(4)
                            )
                            word_address = (
                                batch_peer_s2
                                + fx.Int64(push_node_arrival_off)
                                + parity * fx.Int64(push_node_arrival_stride)
                                + fx.Int64(
                                    word_token_index * ready_groups + batch_n_group
                                )
                                * fx.Int64(4)
                            )
                            comm_ops.atomic_add_system(
                                word_address, packed_word_bits
                            )
                else:
                    if signal_valid != fx.Int32(0):
                        comm_ops.atomic_add_system(signal_address, fx.Int32(1))

        # ---- Node partial: consume producer-pushed tiles from local inbox ----
        elif (bx >= fx.Int32(reduce_first)) & (bx < fx.Int32(combine_first)) & (reduce_enabled == fx.Int32(1)):
            reducer = bx - fx.Int32(reduce_first)
            rank_reduce_acc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                accumulator_ptr,
                num_records_bytes=accumulator_stride,
            )
            rank_mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                dest_rank_mask_ptr,
                num_records_bytes=dest_rank_mask_stride,
            )
            # Payload loads use only this GPU's inbox resource. Peer pointers
            # belong exclusively to the rank-reduce/push producer above.
            inbox_rsrc = buffer_ops.create_buffer_resource_from_addr(
                push_inbox_ptr, num_records_bytes=push_inbox_stride)

            def reduce_node_tile(token_index, n_group, rank_mask):
                totals = fx.Vector.filled(8, 0.0, Float32)
                for peer in range_constexpr(GPUS_PER_NODE):
                    peer_active = (rank_mask & fx.Int32(1 << peer)) != fx.Int32(0)
                    inbox_index = ((token_index * fx.Int32(ready_groups) + n_group)
                                   * fx.Int32(GPUS_PER_NODE) + fx.Int32(peer)) * fx.Int32(2 * BN)
                    words = fx.Vector(buffer_ops.buffer_load(
                        inbox_rsrc, (inbox_index + lane * fx.Int32(8)) // fx.Int32(2),
                        vec_width=4, dtype=T.i32, mask=peer_active))
                    totals = totals + words.bitcast(BFloat16).to(Float32)
                # With mask==0 this overwrites the absent-node partial with
                # zero, preserving lockstep return across routing changes.
                output_index = token_index * fx.Int32(HIDDEN) + n_group * fx.Int32(2 * BN) + lane * fx.Int32(8)
                buffer_ops.buffer_store(totals.to(BFloat16).bitcast(fx.Int32),
                                        rank_reduce_acc_rsrc, output_index // fx.Int32(2))
                rocdl.s_waitcnt(0)
                if lane == fx.Int32(0):
                    if const_expr(node_reduce_token_owner_fastpath):
                        # The validated static stride maps every N group for a
                        # token to this same wave in increasing group order.
                        # The wave-wide wait above drains each group payload;
                        # the last group can therefore publish the complete
                        # token without an inter-wave mask RMW chain. Keep the
                        # diagnostic mask ABI populated for end-of-run checks.
                        if n_group == fx.Int32(ready_groups - 1):
                            comm_ops.store_i64_global_relaxed(
                                node_ready_mask_ptr
                                + fx.Int64(token_index) * fx.Int64(8),
                                fx.Int64((1 << hidden_tiles) - 1),
                            )
                            comm_ops.store_i64_global_system(
                                partial_ready_ptr
                                + fx.Int64(token_index) * fx.Int64(8),
                                generation,
                            )
                    else:
                        # These group partials are consumed on this GPU. The
                        # acq_rel mask chain collects all groups; only its last
                        # contributor publishes system-visible token readiness
                        # below, flushing the complete payload for RAIL/NIC use.
                        group_mask = fx.Int64(3) << fx.Int64(n_group * fx.Int32(2))
                        old_mask = fx.Int64(comm_ops.atomic_add_agent_acq_rel(
                            node_ready_mask_ptr
                            + fx.Int64(token_index) * fx.Int64(8),
                            group_mask,
                        ))
                        if (old_mask & group_mask) != fx.Int64(0):
                            add_error()
                        if (old_mask | group_mask) == fx.Int64(
                            (1 << hidden_tiles) - 1
                        ):
                            comm_ops.store_i64_global_system(
                                partial_ready_ptr
                                + fx.Int64(token_index) * fx.Int64(8),
                                generation,
                            )

            if const_expr(rank_push_publication == "batch_bitmap"):
                # A B8 push batch never crosses a source-rank or source-plane
                # boundary. Two i32 words pack the expected 8-bit rank masks
                # for tokens0..3 and4..7. All contributors update disjoint bit
                # positions, so fetch-add is equivalent to OR without carry.
                token_batches = (2 * MAX_TOKENS) // rank_push_batch_size
                for node_batch in range(
                    reducer * fx.Int32(4) + wave,
                    fx.Int32(token_batches * ready_groups),
                    fx.Int32(max(1, reduce_blocks) * 4),
                ):
                    n_group = node_batch // fx.Int32(token_batches)
                    token_batch = node_batch - n_group * fx.Int32(token_batches)
                    token_base = token_batch * fx.Int32(rank_push_batch_size)
                    # The masks are immutable after the acquired init gate.
                    # Load all eight adjacent tokens once across the wave and
                    # retain the lane values through the readiness wait; the
                    # old path issued eight lane0 loads to build the bitmap and
                    # repeated the same eight loads before reducing payloads.
                    batch_rank_mask_lane = fx.Int32(buffer_ops.buffer_load(
                        rank_mask_rsrc, token_base + lane, vec_width=1,
                        dtype=T.i32, mask=lane < fx.Int32(rank_push_batch_size)))
                    expected_lo = fx.Int32(0)
                    expected_hi = fx.Int32(0)
                    for batch_slot in range(
                        fx.Int32(0), fx.Int32(rank_push_batch_size), fx.Int32(1)
                    ):
                        rank_mask = fx.Int32(rocdl.readlane(
                            T.i32, batch_rank_mask_lane,
                            fx.Int32(batch_slot))) & fx.Int32(0xFF)
                        packed_mask = rank_mask << (
                            (batch_slot & fx.Int32(3)) * fx.Int32(8))
                        expected_lo = (batch_slot < fx.Int32(4)).select(
                            expected_lo | packed_mask, expected_lo)
                        expected_hi = (batch_slot >= fx.Int32(4)).select(
                            expected_hi | packed_mask, expected_hi)
                    if lane == fx.Int32(0):
                        arrival_lo_addr = (
                            push_node_arrival_ptr
                            + fx.Int64(token_base * ready_groups + n_group) * fx.Int64(4))
                        arrival_hi_addr = (
                            push_node_arrival_ptr
                            + fx.Int64((token_base + fx.Int32(4)) * ready_groups
                                       + n_group) * fx.Int64(4))
                        arrived_lo = fx.Int32(
                            comm_ops.load_i32_global_system_relaxed(arrival_lo_addr))
                        arrived_hi = fx.Int32(
                            comm_ops.load_i32_global_system_relaxed(arrival_hi_addr))
                        while ((arrived_lo & expected_lo) != expected_lo) | (
                            (arrived_hi & expected_hi) != expected_hi
                        ):
                            arrived_lo = fx.Int32(
                                comm_ops.load_i32_global_system_relaxed(arrival_lo_addr))
                            arrived_hi = fx.Int32(
                                comm_ops.load_i32_global_system_relaxed(arrival_hi_addr))
                        if (arrived_lo != expected_lo) | (arrived_hi != expected_hi):
                            add_error()
                    comm_ops.fence_system_acquire()
                    for batch_slot in range(
                        fx.Int32(0), fx.Int32(rank_push_batch_size), fx.Int32(1)
                    ):
                        token_index = token_base + fx.Int32(batch_slot)
                        packed_expected = (batch_slot < fx.Int32(4)).select(
                            expected_lo, expected_hi)
                        rank_mask = (packed_expected >> (
                            (batch_slot & fx.Int32(3)) * fx.Int32(8))) & fx.Int32(0xFF)
                        reduce_node_tile(token_index, n_group, rank_mask)
            else:
                owner_stride = max(1, reduce_blocks) * 4
                owner_token_base = reducer * fx.Int32(4) + wave
                owner_token_count = (2 * MAX_TOKENS) // owner_stride
                owner_rank_mask_lane = fx.Int32(0)
                if const_expr(node_reduce_token_owner_mask_prefetch):
                    # The init gate has already acquired these immutable masks.
                    # One lane loads each token owned by this wave and retains
                    # it while the existing group-major task order progresses.
                    owner_token_index = (
                        owner_token_base + lane * fx.Int32(owner_stride)
                    )
                    owner_rank_mask_lane = fx.Int32(buffer_ops.buffer_load(
                        rank_mask_rsrc, owner_token_index, vec_width=1,
                        dtype=T.i32,
                        mask=lane < fx.Int32(owner_token_count),
                    ))
                for node_task in range(
                    owner_token_base,
                    fx.Int32(2 * MAX_TOKENS * ready_groups),
                    fx.Int32(owner_stride),
                ):
                    n_group = node_task // fx.Int32(2 * MAX_TOKENS)
                    token_index = node_task - n_group * fx.Int32(2 * MAX_TOKENS)
                    if const_expr(node_reduce_token_owner_mask_prefetch):
                        owner_token_slot = token_index // fx.Int32(owner_stride)
                        rank_mask = fx.Int32(rocdl.readlane(
                            T.i32, owner_rank_mask_lane,
                            owner_token_slot,
                        )) & fx.Int32(0xFF)
                    else:
                        rank_mask_lane = fx.Int32(0)
                        if lane == fx.Int32(0):
                            rank_mask_lane = fx.Int32(buffer_ops.buffer_load(
                                rank_mask_rsrc, token_index, vec_width=1,
                                dtype=T.i32))
                        rank_mask = fx.Int32(rocdl.readfirstlane(
                            T.i32, rank_mask_lane.ir_value())) & fx.Int32(0xFF)
                    if lane == fx.Int32(0):
                        if const_expr(
                            rank_push_publication == "packed_tile_bitmap"
                        ):
                            word_token_index = token_index & fx.Int32(-4)
                            shift = (token_index & fx.Int32(3)) * fx.Int32(8)
                            arrival_addr = (
                                push_node_arrival_ptr
                                + fx.Int64(
                                    word_token_index * ready_groups + n_group
                                )
                                * fx.Int64(4)
                            )
                            arrived_word = fx.Int32(
                                comm_ops.load_i32_global_system_relaxed(
                                    arrival_addr
                                )
                            )
                            arrived = (arrived_word >> shift) & fx.Int32(0xFF)
                            while (arrived & rank_mask) != rank_mask:
                                arrived_word = fx.Int32(
                                    comm_ops.load_i32_global_system_relaxed(
                                        arrival_addr
                                    )
                                )
                                arrived = (
                                    arrived_word >> shift
                                ) & fx.Int32(0xFF)
                            if arrived != rank_mask:
                                add_error()
                        else:
                            if const_expr(node_reduce_expected_popcount):
                                expected = fx.Int32(_llvm_d.call_intrinsic(
                                    T.i32,
                                    "llvm.ctpop.i32",
                                    [rank_mask.ir_value()],
                                    [],
                                    [],
                                ))
                            else:
                                expected = fx.Int32(0)
                                for peer in range_constexpr(GPUS_PER_NODE):
                                    expected = expected + (
                                        (rank_mask >> fx.Int32(peer)) & fx.Int32(1)
                                    )
                            arrival_addr = (
                                push_node_arrival_ptr
                                + fx.Int64(
                                    token_index * ready_groups + n_group
                                )
                                * fx.Int64(4)
                            )
                            # Peers update this local counter, so polling retains
                            # system scope. Only readiness needs repeated atomic
                            # loads; acquire the payload once after the loop.
                            arrived = fx.Int32(
                                comm_ops.load_i32_global_system_relaxed(
                                    arrival_addr
                                )
                            )
                            while arrived < expected:
                                arrived = fx.Int32(
                                    comm_ops.load_i32_global_system_relaxed(
                                        arrival_addr
                                    )
                                )
                            if arrived != expected:
                                add_error()
                    comm_ops.fence_system_acquire()
                    reduce_node_tile(token_index, n_group, rank_mask)

        # ---- Final add: source-aligned local and RAIL-returned node partials ----
        if (bx >= fx.Int32(combine_first)) & (
            bx < fx.Int32(combine_first + final_combine_blocks)
        ) & (role_enabled == fx.Int32(1)):
            final_local_rsrc = buffer_ops.create_buffer_resource_from_addr(
                accumulator_ptr,
                num_records_bytes=accumulator_stride,
            )
            final_remote_rsrc = buffer_ops.create_buffer_resource_from_addr(
                rx_ptr,
                num_records_bytes=rx_stride,
            )
            final_output_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_output_bf16,
                num_records_bytes=MAX_TOKENS * HIDDEN * 2,
            )
            active = fx.Int32(1) == fx.Int32(1)
            while active:
                gpu.barrier()
                if tx == fx.Int32(0):
                    work = fx.Int32(
                        comm_ops.atomic_add_agent(final_head_ptr, fx.Int32(1))
                    )
                    fx.ptr_store(work, work_ptr)
                gpu.barrier()
                work = fx.Int32(fx.ptr_load(work_ptr))
                has_work = work < fx.Int32(final_work_items)
                if has_work:
                    token = (
                        (work)
                    )
                    if tx == fx.Int32(0):
                        token_ready_index = (
                            fx.Int32(local_plane * MAX_TOKENS) + token
                        )
                        comm_ops.spin_until_ge_i64_system(
                            rail_payload_ready_ptr
                            + fx.Int64(token_ready_index) * fx.Int64(8),
                            generation,
                        )
                        group = token // fx.Int32(return_chunk_tokens)
                        comm_ops.spin_until_ge_i64_system(
                            return_ready_ptr
                            + fx.Int64(group) * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    local_active = fx.Int32(1) == fx.Int32(1)
                    remote_active = fx.Int32(1) == fx.Int32(1)
                    # A wave moves 512 values with 16-byte/lane operations.
                    # Four iterations cover every accepted H<=8192 shape; the
                    # mask excludes unused wave chunks.
                    for chunk_iter in range_constexpr(4):
                        chunk = wave + fx.Int32(chunk_iter * 4)
                        active_chunk = chunk < fx.Int32(HIDDEN // (64 * 8))
                        col = (chunk * fx.Int32(64) + lane) * fx.Int32(8)
                        local_index = (
                            (
                                fx.Int32(local_plane * MAX_TOKENS)
                                + token
                            )
                            * fx.Int32(HIDDEN)
                            + col
                        )
                        token_index = token * fx.Int32(HIDDEN) + col
                        local_packed = buffer_ops.buffer_load(
                            final_local_rsrc,
                            local_index // fx.Int32(2),
                            vec_width=4,
                            dtype=T.i32,
                            mask=active_chunk & local_active,
                        )
                        remote_index = (
                            (token_index)
                        )
                        remote_packed = buffer_ops.buffer_load(
                            final_remote_rsrc,
                            remote_index // fx.Int32(2),
                            vec_width=4,
                            dtype=T.i32,
                            mask=active_chunk & remote_active,
                        )
                        local_values = fx.Vector(local_packed).bitcast(
                            BFloat16
                        ).to(Float32)
                        remote_values = fx.Vector(remote_packed).bitcast(
                            BFloat16
                        ).to(Float32)
                        result = fx.Vector.from_elements(
                            [
                                fx.Float32(local_values[index])
                                + fx.Float32(remote_values[index])
                                for index in range_constexpr(8)
                            ],
                            Float32,
                        ).to(BFloat16).bitcast(fx.Int32)
                        buffer_ops.buffer_store(
                            result,
                            final_output_rsrc,
                            token_index // fx.Int32(2),
                            mask=active_chunk,
                        )
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        comm_ops.fence_system_release()
                        comm_ops.atomic_add_system_acq_rel(
                            final_done_ptr, fx.Int32(1)
                        )
                active = has_work

        # ------------------ Remaining roles: persistent GMM2 ------------------
        # Each rank-local work item computes two N tiles and weighted BF16
        # local accumulation, then publishes token/group completion. The eight
        # work-head shards distribute these items among the remaining CTAs.
        elif (bx >= fx.Int32(gmm_first)) & (compute_enabled == fx.Int32(1)):
            num_valid = fx.Int32(global_typed_ptr(arg_nvalid, T.i32)[0])
            total_m_blocks = num_valid // fx.Int32(BM)
            n_groups = hidden_tiles // n_tile_group
            total_work = total_m_blocks * fx.Int32(n_groups)
            shard = bx & fx.Int32(WORK_SHARDS - 1)
            active = fx.Int32(1) == fx.Int32(1)
            gmm_work_ptr = work_ptr
            if const_expr(gmm_queue_ticket_peer_slab):
                # Rank-push consumes peer pointers in a disjoint CTA role.
                # Reusing the first peer-table word here isolates the queue
                # ticket from GEMM's first A DMA without increasing LDS.
                gmm_work_ptr = lds_typed_ptr(
                    lds_base + fx.Int32(peer_table_off), T.i32, align=4
                )

            def publish_route_group_arrivals(m_row, n_group):
                """Publish actual route completions for this pair of N tiles."""

                # Every payload-writing wave has drained its stores, joined
                # the CTA barrier, and executed an agent release fence before
                # these notification lanes run. A relaxed RMW is sufficient:
                # the rank reducer observes the actual route count with an
                # atomic poll, then acquires before reading private payloads.
                # All increments are RMWs, so the release sequence also covers
                # multiple GEMM CTAs contributing to this token/N group. The
                # publisher neither reads consumer data nor uses the old
                # count; acq_rel here would repeat release work and invalidate
                # caches without adding a needed dependency. Keep the all-wave
                # drain/join/release in run_gmm_work; this is not a last-arriver
                # operation like the node reducer's ready-mask update.
                if tx < fx.Int32(BM):
                    packed = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(arg_stids),
                        m_row + tx, vec_width=1, dtype=T.i32)
                    source = packed & fx.Int32(0x00FFFFFF)
                    if source < fx.Int32(SOURCE_CAPACITY):
                        comm_ops.atomic_add_agent(
                            push_arrival_ptr + fx.Int64(source * ready_groups + n_group) * fx.Int64(4),
                            fx.Int32(1))

            def cache_epilogue_metadata(m_row):
                """Cache route metadata once for every N tile in an M group."""

                meta = buffer_ops.create_buffer_resource_from_addr(arg_stids)
                weights = buffer_ops.create_buffer_resource_from_addr(arg_sweights)
                if tx < fx.Int32(BM):
                    packed_row = buffer_ops.buffer_load(
                        meta, m_row + tx, vec_width=1, dtype=T.i32
                    )
                    weight_row = buffer_ops.buffer_load(
                        weights, m_row + tx, vec_width=1, dtype=T.f32
                    )
                    fx.ptr_store(
                        packed_row,
                        lds_typed_ptr(
                            lds_base
                            + fx.Int32(epilogue_meta_off)
                            + tx * fx.Int32(4),
                            T.i32,
                            align=4,
                        ),
                    )
                    fx.ptr_store(
                        weight_row,
                        lds_typed_ptr(
                            lds_base
                            + fx.Int32(epilogue_weight_off)
                            + tx * fx.Int32(4),
                            T.f32,
                            align=4,
                        ),
                    )

            def store_weighted_route_tile(
                accm, m_row, n_block, cache_metadata, issue_barrier
            ):
                """C-shuffle to BF16 and store each distinct sorted route row."""

                # All original slots keep independent physical rows, including
                # duplicate experts. Rows read immutable C-shuffle LDS and do not
                # publish readiness until both N tiles have drained below.
                lds_bf16 = lds_typed_ptr(lds_base, T.bf16, align=2)
                if const_expr(cache_metadata):
                    cache_epilogue_metadata(m_row)
                lane_div_16 = lane // fx.Int32(16)
                lane_mod_16 = lane % fx.Int32(16)
                wave_n = BN // 4
                num_acc_n = (BN // 4) // 16
                for i in range_constexpr(BM // 16):
                    row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
                    row_weights = [
                        fx.Float32(
                            fx.ptr_load(
                                lds_typed_ptr(
                                    lds_base
                                    + fx.Int32(epilogue_weight_off)
                                    + (row_base + fx.Int32(v)) * fx.Int32(4),
                                    T.f32,
                                    align=4,
                                )
                            )
                        )
                        for v in range_constexpr(4)
                    ]
                    for j in range_constexpr(num_acc_n):
                        col = (
                            wave * fx.Int32(wave_n)
                            + fx.Int32(j * 16)
                            + lane_mod_16
                        )
                        vec = fx.Vector(accm[i][j])
                        for v in range_constexpr(4):
                            lds_bf16[
                                (row_base + fx.Int32(v)) * fx.Int32(BN)
                                + col
                            ] = fx.BFloat16(
                                fx.Float32(vec[v]) * row_weights[v]
                            )
                gpu.barrier()

                # Match the current MegaMoEv2 vec8 C-shuffle traversal: one
                # route row per wave, 32 active lanes and one 16-byte store per
                # active lane. This private route store predicates invalid
                # lanes; the reference peer scatter uses bounded OOB offsets.
                for row_group in range_constexpr(BM // 4):
                    row_in_block = fx.Int32(row_group * 4) + wave
                    packed = fx.Int32(
                        fx.ptr_load(
                            lds_typed_ptr(
                                lds_base
                                + fx.Int32(epilogue_meta_off)
                                + row_in_block * fx.Int32(4),
                                T.i32,
                                align=4,
                            )
                        )
                    )
                    packed = fx.Int32(
                        rocdl.readfirstlane(T.i32, packed.ir_value())
                    )
                    source = packed & fx.Int32(0x00FFFFFF)
                    topk_slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
                    valid = (source < fx.Int32(SOURCE_CAPACITY)) & (
                        topk_slot < fx.Int32(TOPK)
                    )
                    active = lane < fx.Int32(BN // 8)
                    col_start = active.select(
                        lane * fx.Int32(8), fx.Int32(0)
                    )
                    lds_index = row_in_block * fx.Int32(BN) + col_start
                    values = fx.Vector(
                        lds_vec_load(
                            lds_base,
                            lds_index * fx.Int32(2),
                            fx.Vector.make_type(8, BFloat16),
                            BFloat16,
                            align=16,
                        )
                    )
                    if valid & active:
                        # Each physical sorted route row has exactly one
                        # writer per N tile, including duplicate experts
                        # in distinct TopK slots. No payload atomic.
                        push_rsrc = buffer_ops.create_buffer_resource_from_addr(
                            push_private_ptr, num_records_bytes=push_payload_bytes)
                        push_offset = ((m_row + row_in_block) * fx.Int32(HIDDEN)
                                       + n_block * fx.Int32(BN) + col_start) * fx.Int32(2)
                        buffer_ops.buffer_store(values.ir_value(), push_rsrc, push_offset,
                                                offset_is_bytes=True)
                    # The per_tile tune lets waves advance independently over
                    # immutable LDS rows. Tile-exit wait/barrier protects LDS
                    # reuse; the caller's release protects payload publication.
                    if const_expr(rank_epilogue_barrier == "per_row"):
                        gpu.barrier()

                if const_expr(issue_barrier):
                    # The first tile's C-shuffle shares LDS with the next GEMM.
                    # All waves must finish LDS reads before that reuse; local
                    # route stores may remain in flight until the group-end wait.
                    rocdl.s_waitcnt(lgkmcnt=0)
                    gpu.barrier()

            def issue_node_epilogue(
                accm, m_row, n_block, cache_metadata, issue_barrier
            ):
                store_weighted_route_tile(
                    accm,
                    m_row,
                    n_block,
                    cache_metadata,
                    issue_barrier,
                )

            def preload_initial_a(m_block, compute_lds_base):
                for slot in range_constexpr(kStages):
                    issue_a_load_lds_dt(
                        arg_aq,
                        compute_lds_base,
                        slot,
                        slot,
                        m_block * fx.Int32(BM),
                        wave,
                        lane,
                        False,
                        kh_tile_a,
                        fx.Int32(INTER // 2),
                        BM=BM,
                    )

            def compute_gmm_tile(
                m_block,
                n_block,
                preloaded_expert,
                compute_lds_base,
                initial_a_preloaded,
                use_direct_coords,
            ):
                if const_expr(use_direct_coords):
                    # BM is power-of-two and HIDDEN/BN <= BM for the supported
                    # Stage2 shape range. Pack N in the low bits so GEMM2 can
                    # recover both coordinates with a shift/mask while carrying
                    # only one value across its long compute region.
                    work = m_block * fx.Int32(BM) + n_block
                else:
                    work = m_block * fx.Int32(hidden_tiles) + n_block
                if const_expr(not initial_a_preloaded):
                    preload_initial_a(m_block, compute_lds_base)
                rocdl.sched_barrier(0)
                accm, m_row, n_block, _ = gemm2_compute_v2(
                    compute_lds_base,
                    arg_ascale,
                    arg_bq,
                    arg_bscale,
                    arg_eids,
                    arg_aq,
                    fx.Int32(max_m_blocks),
                    work,
                    lane,
                    wave,
                    fx.Int32(INTER),
                    fx.Int32(HIDDEN),
                    fx.Int32(0),
                    fx.Int32(0),
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    use_nt=gemm_use_nt,
                    INTER_MAX=INTER,
                    aStages=a_stages,
                    a_dtype="fp4",
                    has_pad=False,
                    SBM=BM,
                    g2_bhoist=True,
                    g2_ascale_pf=True,
                    expert_offset=0,
                    preloaded_expert=preloaded_expert,
                    packed_tile_coord=use_direct_coords,
                )
                return accm, m_row

            def run_gmm_work(work):
                if const_expr(gmm_work_swizzle == "token_major"):
                    m_block = work // fx.Int32(n_groups)
                    n_group = work - m_block * fx.Int32(n_groups)
                else:
                    # Visit all M blocks within one N-group window before advancing windows;
                    # inside a window, a given M block visits its window's N groups first.
                    # Window=1 prioritizes completing one whole N group. Window=2 balances
                    # that goal with adjacent-N work on the same M rows; it is not a proven
                    # optimum. H3584 has 7 groups, so window=2 needs a final 1-group tail.
                    full_windows = n_groups // window_n_groups
                    tail_groups = n_groups % window_n_groups
                    full_work = (
                        fx.Int32(full_windows * window_n_groups) * total_m_blocks
                    )
                    in_full = work < full_work
                    full_span = total_m_blocks * fx.Int32(window_n_groups)
                    safe_span = (full_span > fx.Int32(0)).select(
                        full_span, fx.Int32(1)
                    )
                    if const_expr(gmm_work_unsigned_window_division):
                        # Persistent work IDs are shard + 8*ticket and this
                        # path executes only under work < total_work. With W1,
                        # total_work = n_groups*total_m_blocks, so any active
                        # work implies work>=0 and safe_span=total_m_blocks>0.
                        # Keep tail_work signed because it is negative for the
                        # selected full-window path and is evaluated before the
                        # final select.
                        window = _udiv(work, safe_span)
                    else:
                        window = work // safe_span
                    within = work - window * safe_span
                    full_m = within // fx.Int32(window_n_groups)
                    full_n = (
                        window * fx.Int32(window_n_groups)
                        + within
                        - full_m * fx.Int32(window_n_groups)
                    )
                    tail_work = work - full_work
                    safe_tail = max(tail_groups, 1)
                    tail_m = tail_work // fx.Int32(safe_tail)
                    tail_n = (
                        fx.Int32(full_windows * window_n_groups)
                        + tail_work
                        - tail_m * fx.Int32(safe_tail)
                    )
                    m_block = in_full.select(full_m, tail_m)
                    n_group = in_full.select(full_n, tail_n)
                n_block0 = n_group * fx.Int32(n_tile_group)
                group_m_row = m_block * fx.Int32(BM)
                preloaded_expert = None
                preloaded_expert = rocdl.readfirstlane(
                    T.i32,
                    global_typed_ptr(arg_eids, T.i32)[m_block],
                )
                cache_epilogue_metadata(group_m_row)

                accm0, m_row = compute_gmm_tile(
                    m_block,
                    n_block0,
                    preloaded_expert,
                    lds_base,
                    False,
                    gmm_direct_tile_coords,
                )
                n_block1 = n_block0 + fx.Int32(1)
                # Complete the next N tile's initial A DMA before
                # storing tile0. Its GEMM can then enter MFMA
                # without a post-store A dependency forcing
                # vmcnt(0); later rotating A loads may still bound
                # how long the overlap lasts.
                a_next = lds_base + fx.Int32(a_double_buffer_off)
                preload_initial_a(m_block, a_next)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                issue_node_epilogue(
                    accm0,
                    m_row,
                    n_block0,
                    group_pipeline_schedule == "baseline",
                    True,
                )
                accm1, _ = compute_gmm_tile(
                    m_block,
                    n_block1,
                    preloaded_expert,
                    (
                        (lds_base + fx.Int32(a_double_buffer_off))
                    ),
                    group_pipeline_schedule == "a_double_buffer",
                    gmm_direct_tile_coords,
                )
                issue_node_epilogue(
                    accm1, m_row, n_block1, False, False
                )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_agent_release()
                publish_route_group_arrivals(m_row, n_group)

            if const_expr(gmm_schedule == "static_strided"):
                worker = bx - fx.Int32(gmm_first)
                worker_count = grid - fx.Int32(gmm_first)
                for work in range(worker, total_work, worker_count):
                    run_gmm_work(work)
            else:
                # A work ID is shard + 8*ticket. Shards have separate 64-byte
                # counter lines to limit contention; each still has to drain.
                # The swizzle changes work order, not the (M,N-group) work set.
                while active:
                    if const_expr(not gmm_queue_head_barrier_elision):
                        gpu.barrier()
                    if tx == fx.Int32(0):
                        local_work = fx.Int32(
                            comm_ops.atomic_add_agent(
                                gemm_head_ptr + fx.Int64(shard) * fx.Int64(64),
                                fx.Int32(1),
                            )
                        )
                        work = shard + local_work * fx.Int32(WORK_SHARDS)
                        fx.ptr_store(work, gmm_work_ptr)
                    gpu.barrier()
                    work = fx.Int32(fx.ptr_load(gmm_work_ptr))
                    has_work = work < total_work
                    if has_work:
                        run_gmm_work(work)
                    active = has_work
        if const_expr(timeline_instrument):
            # Record completion for every CTA role. Legacy consumers select
            # only [gmm_first, grid); node diagnostics also read rank-push and
            # node-reducer ranges. Join the four independent reducer waves so
            # wave0's timestamp cannot precede the CTA's final payload work.
            # All of this, including the barrier, disappears in production.
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.store_i64_global_relaxed(
                    timeline_gmm_done_ptr + fx.Int64(bx) * fx.Int64(8),
                    fx.Int64(comm_ops.read_wall_clock()),
                )

    @flyc.jit
    def launch_megamoe_tile_ep16_stage2(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        generation: fx.Int64,
        local_tokens: fx.Int32,
        worker_blocks: fx.Int32,
        arg_output_bf16: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            dev_comm,
            arena_win,
            arena_ptr,
            arg_bq,
            arg_bscale,
            generation,
            local_tokens,
            arg_output_bf16,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=(worker_blocks, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    launch_megamoe_tile_ep16_stage2.kernel_name = kernel_name
    launch_megamoe_tile_ep16_stage2.device_generation = bool(device_generation)
    launch_megamoe_tile_ep16_stage2.lds_bytes = lds_bytes
    launch_megamoe_tile_ep16_stage2.diagnostic_mode = diagnostic_mode
    launch_megamoe_tile_ep16_stage2.gemm2_contraction = diagnostic_mode != "init_only"
    launch_megamoe_tile_ep16_stage2.communication_roles_enabled = (
        diagnostic_mode == "full"
    )
    launch_megamoe_tile_ep16_stage2.accumulator_dtype = accumulator_dtype
    launch_megamoe_tile_ep16_stage2.final_combine_blocks = final_combine_blocks
    launch_megamoe_tile_ep16_stage2.gemm_use_nt = gemm_use_nt
    launch_megamoe_tile_ep16_stage2.rank_push_use_nt = rank_push_use_nt
    launch_megamoe_tile_ep16_stage2.gmm_schedule = gmm_schedule
    launch_megamoe_tile_ep16_stage2.gmm_work_swizzle = gmm_work_swizzle
    launch_megamoe_tile_ep16_stage2.window_n_groups = window_n_groups
    launch_megamoe_tile_ep16_stage2.gmm_work_unsigned_window_division = (
        gmm_work_unsigned_window_division
    )
    launch_megamoe_tile_ep16_stage2.gmm_direct_tile_coords = (
        gmm_direct_tile_coords
    )
    launch_megamoe_tile_ep16_stage2.gmm_queue_ticket_peer_slab = (
        gmm_queue_ticket_peer_slab
    )
    launch_megamoe_tile_ep16_stage2.gmm_queue_head_barrier_elision = (
        gmm_queue_head_barrier_elision
    )
    launch_megamoe_tile_ep16_stage2.gmm_queue_ticket_lds_offset = (
        peer_table_off if gmm_queue_ticket_peer_slab else 0
    )
    launch_megamoe_tile_ep16_stage2.peer_table_lds_offset = peer_table_off
    launch_megamoe_tile_ep16_stage2.return_chunk_tokens = return_chunk_tokens
    launch_megamoe_tile_ep16_stage2.node_ready_granularity = "token"
    launch_megamoe_tile_ep16_stage2.bf16_atomic_kind = bf16_atomic_kind
    launch_megamoe_tile_ep16_stage2.rail_return_schedule = rail_return_schedule
    launch_megamoe_tile_ep16_stage2.rail_quant_type = rail_quant_type
    launch_megamoe_tile_ep16_stage2.ready_granularity = ready_granularity
    launch_megamoe_tile_ep16_stage2.group_batch = bool(group_batch)
    launch_megamoe_tile_ep16_stage2.epilogue_schedule = epilogue_schedule
    launch_megamoe_tile_ep16_stage2.n_tile_group = n_tile_group
    launch_megamoe_tile_ep16_stage2.group_pipeline_schedule = (
        group_pipeline_schedule
    )
    launch_megamoe_tile_ep16_stage2.node_accumulation_mode = (
        node_accumulation_mode
    )
    launch_megamoe_tile_ep16_stage2.rank_accumulation_mode = rank_accumulation_mode
    launch_megamoe_tile_ep16_stage2.rank_push_batch_size = rank_push_batch_size
    launch_megamoe_tile_ep16_stage2.rank_push_batch_invariants = (
        rank_push_batch_invariants
    )
    launch_megamoe_tile_ep16_stage2.rank_push_count_prefetch = (
        rank_push_count_prefetch
    )
    launch_megamoe_tile_ep16_stage2.rank_push_count_wave_broadcast = (
        rank_push_count_wave_broadcast
    )
    launch_megamoe_tile_ep16_stage2.rank_push_map_prefetch = (
        rank_push_map_prefetch
    )
    launch_megamoe_tile_ep16_stage2.rank_push_count_from_slot_map = (
        rank_push_count_from_slot_map
    )
    launch_megamoe_tile_ep16_stage2.rank_push_tile_index_recurrence = (
        rank_push_tile_index_recurrence
    )
    launch_megamoe_tile_ep16_stage2.rank_push_poll_backoff = (
        rank_push_poll_backoff
    )
    launch_megamoe_tile_ep16_stage2.rank_push_acquire_cohort = (
        rank_push_acquire_cohort
    )
    launch_megamoe_tile_ep16_stage2.rank_push_single_row_fastpath = bool(
        rank_push_single_row_fastpath
    )
    launch_megamoe_tile_ep16_stage2.rank_push_publication = rank_push_publication
    launch_megamoe_tile_ep16_stage2.rank_reduce_blocks = rank_push_blocks
    launch_megamoe_tile_ep16_stage2.gmm_first_block = gmm_first
    launch_megamoe_tile_ep16_stage2.rank_push_workspace = push_workspace
    launch_megamoe_tile_ep16_stage2.node_reduce_blocks = node_reduce_blocks
    launch_megamoe_tile_ep16_stage2.node_reduce_token_owner_fastpath = (
        node_reduce_token_owner_fastpath
    )
    launch_megamoe_tile_ep16_stage2.node_reduce_token_owner_mask_prefetch = (
        node_reduce_token_owner_mask_prefetch
    )
    launch_megamoe_tile_ep16_stage2.node_reduce_expected_popcount = (
        node_reduce_expected_popcount
    )
    launch_megamoe_tile_ep16_stage2.node_reduce_vec_bytes = (
        node_reduce_vec_bytes
    )
    launch_megamoe_tile_ep16_stage2.node_reduce_schedule = node_reduce_schedule
    launch_megamoe_tile_ep16_stage2.node_reduce_load_schedule = (
        node_reduce_load_schedule
    )
    launch_megamoe_tile_ep16_stage2.node_reduce_work_schedule = (
        node_reduce_work_schedule
    )
    launch_megamoe_tile_ep16_stage2.node_reduce_rejoin_blocks = (
        node_reduce_rejoin_blocks
    )
    launch_megamoe_tile_ep16_stage2.rank_epilogue_lds_addressing = (
        rank_epilogue_lds_addressing
    )
    launch_megamoe_tile_ep16_stage2.rank_epilogue_barrier = rank_epilogue_barrier
    launch_megamoe_tile_ep16_stage2.scoreboard_schedule = scoreboard_schedule
    launch_megamoe_tile_ep16_stage2.atomic_issue_schedule = atomic_issue_schedule
    launch_megamoe_tile_ep16_stage2.timeline_instrument = bool(
        timeline_instrument
    )
    launch_megamoe_tile_ep16_stage2.single_gpu_launch = True
    launch_megamoe_tile_ep16_stage2.requires_resident_grid = True
    launch_megamoe_tile_ep16_stage2.fixed_roles = {
        "cross": 1,
        "node_reduce": reduce_blocks,
        "intranode_final": final_combine_blocks,
        "gmm2": "remaining blocks",
    }
    if rank_push_blocks:
        launch_megamoe_tile_ep16_stage2.fixed_roles["rank_reduce_push"] = rank_push_blocks
    launch_megamoe_tile_ep16_stage2.fixed_shape = {
        "tokens_per_rank": MAX_TOKENS,
        "hidden": HIDDEN,
        "inter": INTER,
        "experts": EXPERTS,
        "ep": WORLD,
        "topk": TOPK,
        "quant": "a4w4",
    }
    launch_megamoe_tile_ep16_stage2.route_contract = (
        "arbitrary top-k routes; Stage1 per-node route-count scoreboard; "
        "zero-route nodes publish a cleared zero partial"
    )
    combine_prefix = (
        "GMM2 weighted BF16 private route store -> rank FP32 register reduce "
        "and peer slot push -> node-local inbox FP32 register reduce -> "
    )
    launch_megamoe_tile_ep16_stage2.combine_contract = (
        combine_prefix
        + "direct node accumulator -> "
        "one BF16 RAIL return/token -> source final add"
    )
    launch_megamoe_tile_ep16_stage2.architecture_contract = {
        "epilogue": "route_store_rank_reduce_push_node_local_reduce",
        "node_accumulator_dtype": accumulator_dtype,
        "uses_rank_partial": node_accumulation_mode == "rank_local",
        "uses_node_scan": node_accumulation_mode == "rank_local",
        "uses_external_reduce_kernel": False,
        "uses_external_return_kernel": False,
        "uses_external_final_kernel": False,
        "cross_ctas": 1,
        "intranode_combine_ctas": final_combine_blocks,
        "diagnostic_mode": diagnostic_mode,
        "gmm_schedule": gmm_schedule,
        "gemm_use_nt": gemm_use_nt,
        "rank_push_use_nt": rank_push_use_nt,
        "gmm_work_swizzle": gmm_work_swizzle,
        "window_n_groups": window_n_groups,
        "gmm_work_unsigned_window_division": (
            gmm_work_unsigned_window_division
        ),
        "gmm_direct_tile_coords": gmm_direct_tile_coords,
        "gmm_queue_ticket_peer_slab": gmm_queue_ticket_peer_slab,
        "gmm_queue_head_barrier_elision": gmm_queue_head_barrier_elision,
        "gmm_queue_ticket_lds_offset": (
            peer_table_off if gmm_queue_ticket_peer_slab else 0
        ),
        "peer_table_lds_offset": peer_table_off,
        "return_chunk_tokens": return_chunk_tokens,
        "node_ready_granularity": "token",
        "bf16_atomic_kind": bf16_atomic_kind,
        "rail_return_schedule": rail_return_schedule,
        "rail_quant_type": rail_quant_type,
        "ready_granularity": ready_granularity,
        "epilogue_schedule": epilogue_schedule,
        "n_tile_group": n_tile_group,
        "group_pipeline_schedule": group_pipeline_schedule,
        "node_accumulation_mode": node_accumulation_mode,
        "rank_accumulation_mode": rank_accumulation_mode,
        "rank_push_publication": rank_push_publication,
        "rank_push_batch_invariants": rank_push_batch_invariants,
        "rank_push_count_prefetch": rank_push_count_prefetch,
        "rank_push_count_wave_broadcast": rank_push_count_wave_broadcast,
        "rank_push_map_prefetch": rank_push_map_prefetch,
        "rank_push_count_from_slot_map": rank_push_count_from_slot_map,
        "rank_push_tile_index_recurrence": rank_push_tile_index_recurrence,
        "rank_push_poll_backoff": rank_push_poll_backoff,
        "rank_push_acquire_cohort": rank_push_acquire_cohort,
        "node_reduce_blocks": node_reduce_blocks,
        "node_reduce_token_owner_fastpath": node_reduce_token_owner_fastpath,
        "node_reduce_token_owner_mask_prefetch": (
            node_reduce_token_owner_mask_prefetch
        ),
        "node_reduce_expected_popcount": node_reduce_expected_popcount,
        "node_reduce_vec_bytes": node_reduce_vec_bytes,
        "node_reduce_schedule": node_reduce_schedule,
        "node_reduce_load_schedule": node_reduce_load_schedule,
        "node_reduce_work_schedule": node_reduce_work_schedule,
        "node_reduce_rejoin_blocks": node_reduce_rejoin_blocks,
        "rank_epilogue_lds_addressing": rank_epilogue_lds_addressing,
        "rank_epilogue_barrier": rank_epilogue_barrier,
        "scoreboard_schedule": scoreboard_schedule,
        "atomic_issue_schedule": atomic_issue_schedule,
    }
    launch_megamoe_tile_ep16_stage2.stage2_window_offset = s2_window_off
    launch_megamoe_tile_ep16_stage2.uses_rank_partial = (
        node_accumulation_mode == "rank_local"
    )
    launch_megamoe_tile_ep16_stage2.uses_external_reduce_kernel = False
    launch_megamoe_tile_ep16_stage2.uses_external_return_kernel = False
    launch_megamoe_tile_ep16_stage2.uses_external_final_kernel = False
    if rank_push_blocks:
        launch_megamoe_tile_ep16_stage2.architecture_contract.update({
            "epilogue": "route_store_rank_reduce_push_node_local_reduce",
            "rank_reduce_push_ctas": rank_push_blocks,
            "node_peer_payload_pull": False,
            "rank_push_ready_unit": "source_token_n_group",
            "rank_push_local_load_schedule": (
                "single_row_cached_else_coalesced_map_ordered_set_bits"
                if rank_push_single_row_fastpath
                else "coalesced_map_ordered_set_bits"
            ),
            "rank_push_batch_size": rank_push_batch_size,
            "rank_push_batch_invariants": rank_push_batch_invariants,
            "rank_push_count_prefetch": rank_push_count_prefetch,
            "rank_push_count_wave_broadcast": rank_push_count_wave_broadcast,
            "rank_push_map_prefetch": rank_push_map_prefetch,
            "rank_push_count_from_slot_map": rank_push_count_from_slot_map,
            "rank_push_tile_index_recurrence": rank_push_tile_index_recurrence,
            "rank_push_poll_backoff": rank_push_poll_backoff,
            "rank_push_acquire_cohort": rank_push_acquire_cohort,
            "rank_push_acquire_schedule": (
                "adjacent_nonempty_pair_one_agent_acquire"
                if rank_push_acquire_cohort == 2
                else "per_nonempty_tile_agent_acquire"
            ),
            "rank_push_poll_schedule": (
                "failed_observation_s_sleep_1"
                if rank_push_poll_backoff
                else "tight_relaxed_reload"
            ),
            "rank_push_tile_index_schedule": (
                "batch_scalar_recurrence"
                if rank_push_tile_index_recurrence
                else "per_tile_multiply_add"
            ),
            "rank_push_count_schedule": (
                "slot_map_ballot_ctpop"
                if rank_push_count_from_slot_map
                else (
                    "wave_lane_batch_prefetch"
                    if rank_push_count_prefetch
                    else (
                        "per_tile_wave_uniform_load_readfirstlane"
                        if rank_push_count_wave_broadcast
                        else "per_tile_lane0_load"
                    )
                )
            ),
            "rank_push_map_schedule": (
                "pre_arrival_load_derive_count_reuse"
                if rank_push_count_from_slot_map
                else (
                    "pre_arrival_load_post_acquire_consume"
                    if rank_push_map_prefetch
                    else "post_acquire_load_consume"
                )
            ),
            "rank_push_single_row_fastpath": bool(rank_push_single_row_fastpath),
            "rank_push_publication": rank_push_publication,
            "node_inbox_load_schedule": "fixed_peer_load_add",
            "node_reduce_token_ownership": (
                "static_wave_owner_last_group_publish"
                if node_reduce_token_owner_fastpath
                else "agent_ready_mask_last_arriver"
            ),
            "node_reduce_rank_mask_schedule": (
                "wave_lane_prefetch"
                if node_reduce_token_owner_mask_prefetch
                else "per_group_lane0_load"
            ),
            "node_reduce_expected_count_schedule": (
                "llvm_ctpop_i32"
                if node_reduce_expected_popcount
                else "expanded_eight_bits"
            ),
            "local_workspace_bytes": push_workspace.total_bytes,
        })
    return launch_megamoe_tile_ep16_stage2
