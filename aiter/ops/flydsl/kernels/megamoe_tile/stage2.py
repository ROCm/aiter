# SPDX-License-Identifier: MIT
"""Direct-node-accumulator EP16 A4W4 Stage-2 in one GPU launch.

For arbitrary Top-K routing, a source token may contribute zero or multiple
routes on an EP rank.  The weighted GMM2 epilogue does not materialize rank
partials: every route LSA atomic-adds its contribution directly into the
aligned source proxy on the current expert node. Production uses MI355 packed
BF16 pair atomics; an FP32 reference remains available for diagnostics. A
Stage-1-produced
scoreboard supplies the exact route-slot count (0..16) for every
``(source-node, token, hidden-tile)``.  The last contributor advances a second
token-level counter; the last of 28 hidden tiles publishes one whole-token
ready flag. An expected count of zero publishes an explicitly cleared partial.

Resident CTA roles after a common initialization barrier are::

    block 0      four-wave CCO RAIL BF16 return / receive / credit
    blocks 1..C  final local-node + remote-node combine
    remaining    persistent A4W4 GMM2 work queue

No rank-partial, node-reducer, pack, return-sidecar, or final-combine kernel is
launched.
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


HIDDEN = 7168
INTER = 3072
EXPERTS = 896
WORLD = 16
GPUS_PER_NODE = 8
TOPK = 16
MAX_TOKENS = 128
SOURCE_CAPACITY = WORLD * MAX_TOKENS
THREADS = 256
CROSS_BLOCK = 0
COMBINE_FIRST = 1
DEFAULT_COMBINE_BLOCKS = 14
TEAM_RAIL = "rail"


def _atomic_add_f32_system(address, value):
    """System-scope scalar FP32 atomic add for peer LSA memory."""

    ptr = _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(address)
    ).result
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.fadd,
        ptr,
        arith.unwrap(value),
        _llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def _atomic_add_bf16x2_system(address, value):
    """System-scope packed-BF16 atomic add for a lane-varying peer address."""

    ptr = _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(address)
    ).result
    raw_value = value.ir_value() if hasattr(value, "ir_value") else value
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.fadd,
        ptr,
        raw_value,
        _llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
        alignment=4,
    ).res


def _wave_uniform_i64(value):
    """Move a wave-uniform 64-bit value to SGPRs without an i64 waterfall."""

    value_u64 = fx.Uint64(value)
    lo = rocdl.readfirstlane(T.i32, fx.Uint32(value_u64))
    hi = rocdl.readfirstlane(T.i32, fx.Uint32(value_u64 >> 32))
    return (fx.Uint64(hi) << 32) | fx.Uint64(lo)


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
    return_chunk_tokens: int = 8,
    bf16_atomic_kind: str = "buffer",
    rail_return_schedule: str = "lockstep",
    epilogue_schedule: str = "lane32_meta",
    n_tile_group: int = 2,
    group_pipeline_schedule: str = "a_double_buffer",
    node_accumulation_mode: str = "direct_atomic",
    rank_accumulation_mode: str = "atomic",
    node_reduce_blocks: int = 32,
    node_reduce_vec_bytes: int = 4,
    node_reduce_schedule: str = "token",
    node_reduce_load_schedule: str = "interleaved",
    node_reduce_work_schedule: str = "static_strided",
    node_reduce_rejoin_blocks: int = 0,
    rank_epilogue_lds_addressing: str = "expanded",
    scoreboard_schedule: str = "wave0",
    atomic_issue_schedule: str = "interleaved",
    timeline_instrument: bool = False,
    kernel_name_override: str | None = None,
):
    """Compile the direct-accumulator fixed K3/EP16 Stage-2 launcher."""

    from .cco.ops import (
        flush_async,
        put,
        put_value,
        wait_ready,
        wait_request,
    )

    s1, s2, s2_window_off = _resolve_layouts(
        arena_layout, stage2_layout, stage2_window_offset
    )
    if (
        s1.hidden,
        s1.inter,
        s1.experts,
        s1.world_size,
        s1.gpus_per_node,
        s1.topk,
        s1.max_tokens,
    ) != (HIDDEN, INTER, EXPERTS, WORLD, GPUS_PER_NODE, TOPK, MAX_TOKENS):
        raise ValueError("Stage-1 layout is not the locked K3 EP16 shape")
    if (
        s2.hidden,
        s2.world_size,
        s2.gpus_per_node,
        s2.topk,
        s2.max_tokens,
    ) != (HIDDEN, WORLD, GPUS_PER_NODE, TOPK, MAX_TOKENS):
        raise ValueError("Stage-2 layout is not the locked K3 EP16 shape")
    if s1.parity_depth != 2 or s2.parity_depth != 2:
        raise ValueError("the fused pipeline requires parity_depth=2")
    if not 0 <= int(rank) < WORLD:
        raise ValueError("rank must be in [0,16)")
    if (BM, BN, BK) != (32, 256, 256):
        raise ValueError("the direct Stage-2 requires BM/BN/BK=32/256/256")
    if WORK_SHARDS != 8:
        raise ValueError("the first direct Stage-2 requires 8 work shards")
    if team != TEAM_RAIL:
        raise ValueError("production Stage-2 requires the CCO RAIL team")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")
    if diagnostic_mode not in (
        "full",
        "init_only",
        "atomic_only",
        "gmm2_only",
        "gmm2_atomic_only",
        "route_store_only",
        "return_only",
    ):
        raise ValueError(
            "diagnostic_mode must be full, init_only, atomic_only, "
            "gmm2_only, gmm2_atomic_only, route_store_only, or return_only"
        )
    if accumulator_dtype not in ("fp32", "bf16"):
        raise ValueError("accumulator_dtype must be fp32 or bf16")
    final_combine_blocks = int(final_combine_blocks)
    if not 1 <= final_combine_blocks <= 56:
        raise ValueError("final_combine_blocks must be in [1,56]")
    if gmm_schedule not in ("persistent_queue", "static_strided"):
        raise ValueError("gmm_schedule must be persistent_queue or static_strided")
    return_chunk_tokens = int(return_chunk_tokens)
    if return_chunk_tokens not in (4, 8, 16):
        raise ValueError("return_chunk_tokens must be 4, 8, or 16")
    if return_chunk_tokens > 4 and accumulator_dtype != "bf16":
        raise ValueError("large return chunks require the BF16 accumulator")
    if bf16_atomic_kind not in ("buffer", "global_system"):
        raise ValueError("bf16_atomic_kind must be buffer or global_system")
    if rail_return_schedule not in (
        "lockstep",
        "qp_independent",
        "qp_prepost",
        "compact",
    ):
        raise ValueError(
            "rail_return_schedule must be lockstep, qp_independent, "
            "qp_prepost, or compact"
        )
    if rail_return_schedule != "lockstep" and return_chunk_tokens <= 4:
        raise ValueError("independent QP schedules require return_chunk_tokens > 4")
    if rail_return_schedule == "compact" and node_accumulation_mode != "rank_local":
        raise ValueError("compact return requires rank_local accumulation")
    if rail_return_schedule == "compact" and timeline_instrument:
        raise ValueError("compact return timeline instrumentation is not implemented")
    if epilogue_schedule not in (
        "lane32",
        "lane32_meta",
        "lane32_meta_expected",
        "wave64_meta",
    ):
        raise ValueError(
            "epilogue_schedule must be lane32, lane32_meta, "
            "lane32_meta_expected, or wave64_meta"
        )
    n_tile_group = int(n_tile_group)
    if n_tile_group not in (1, 2):
        raise ValueError("n_tile_group must be 1 or 2")
    if group_pipeline_schedule not in (
        "baseline",
        "expert_meta_hoist",
        "runtime_loop",
        "a_double_buffer",
    ):
        raise ValueError(
            "group_pipeline_schedule must be baseline, expert_meta_hoist, "
            "runtime_loop, or a_double_buffer"
        )
    if group_pipeline_schedule != "baseline" and n_tile_group != 2:
        raise ValueError("group pipeline experiments require n_tile_group=2")
    if node_accumulation_mode not in (
        "direct_atomic",
        "route_store",
        "rank_local",
    ):
        raise ValueError(
            "node_accumulation_mode must be direct_atomic, route_store, "
            "or rank_local"
        )
    if rank_accumulation_mode not in ("atomic", "staged_reduce", "staged_ring"):
        raise ValueError(
            "rank_accumulation_mode must be atomic, staged_reduce, or staged_ring"
        )
    if rank_accumulation_mode == "staged_reduce" and not (
        node_accumulation_mode == "rank_local"
        and node_reduce_vec_bytes == 8
        and node_reduce_load_schedule == "load_first"
        and node_reduce_work_schedule == "static_strided"
        and node_reduce_rejoin_blocks == 0
        and n_tile_group == 2
    ):
        raise ValueError(
            "staged_reduce requires rank_local, vec8/load_first/static_strided, "
            "rejoin_blocks=0, and n_tile_group=2"
        )
    if rank_accumulation_mode == "staged_ring" and not (
        node_accumulation_mode == "rank_local"
        and node_reduce_vec_bytes == 8
        and node_reduce_load_schedule == "load_first"
        and node_reduce_work_schedule == "static_strided"
        and node_reduce_rejoin_blocks == 0
        and n_tile_group == 2
    ):
        raise ValueError(
            "staged_ring requires rank_local, vec8/load_first/static_strided, "
            "rejoin_blocks=0, and n_tile_group=2"
        )
    node_reduce_blocks = int(node_reduce_blocks)
    if node_reduce_blocks not in (8, 16, 32, 56):
        raise ValueError("node_reduce_blocks must be one of 8,16,32,56")
    node_reduce_vec_bytes = int(node_reduce_vec_bytes)
    if node_reduce_vec_bytes not in (4, 8, 16):
        raise ValueError("node_reduce_vec_bytes must be 4, 8, or 16")
    if (
        node_reduce_vec_bytes == 16
        and node_accumulation_mode != "rank_local"
    ):
        raise ValueError(
            "16-byte node reduction requires rank_local accumulation"
        )
    if node_reduce_schedule not in ("token", "group", "tile"):
        raise ValueError("node_reduce_schedule must be token, group, or tile")
    if node_reduce_load_schedule not in ("interleaved", "load_first"):
        raise ValueError(
            "node_reduce_load_schedule must be interleaved or load_first"
        )
    if node_reduce_work_schedule not in ("static_strided", "dynamic_head"):
        raise ValueError(
            "node_reduce_work_schedule must be static_strided or dynamic_head"
        )
    node_reduce_rejoin_blocks = int(node_reduce_rejoin_blocks)
    if node_reduce_rejoin_blocks not in (0, 8, 16, 32):
        raise ValueError("node_reduce_rejoin_blocks must be one of 0,8,16,32")
    if rank_epilogue_lds_addressing not in ("expanded", "dynamic_base"):
        raise ValueError(
            "rank_epilogue_lds_addressing must be expanded or dynamic_base"
        )
    if rank_epilogue_lds_addressing == "dynamic_base" and not (
        node_accumulation_mode == "rank_local"
        and node_reduce_vec_bytes == 8
        and node_reduce_load_schedule == "load_first"
        and node_reduce_work_schedule == "static_strided"
        and node_reduce_rejoin_blocks == 0
    ):
        raise ValueError(
            "dynamic_base LDS addressing requires rank_local, vec8, "
            "load_first, static_strided reduction, and rejoin_blocks=0"
        )
    if (
        node_reduce_work_schedule == "dynamic_head"
        and node_accumulation_mode != "rank_local"
    ):
        raise ValueError("dynamic_head node reduction requires rank_local")
    if node_reduce_rejoin_blocks > 0 and not (
        node_accumulation_mode == "rank_local"
        and rail_return_schedule == "compact"
        and gmm_schedule == "persistent_queue"
        and node_reduce_work_schedule == "dynamic_head"
    ):
        raise ValueError(
            "node reducer rejoin requires rank_local, compact return, "
            "persistent_queue GMM, and dynamic_head reduction"
        )
    if node_reduce_rejoin_blocks > 0 and diagnostic_mode not in (
        "full",
        "atomic_only",
        "gmm2_atomic_only",
    ):
        raise ValueError(
            "node reducer rejoin requires a queue-producing, reducer-enabled "
            "diagnostic mode (full, atomic_only, or gmm2_atomic_only)"
        )
    if node_accumulation_mode == "route_store":
        if not getattr(s2, "include_route_slots", False):
            raise ValueError("route_store requires a route-slot Stage2 arena")
        if accumulator_dtype != "bf16":
            raise ValueError("route_store requires the BF16 node accumulator")
        if epilogue_schedule != "lane32_meta":
            raise ValueError("route_store requires lane32_meta metadata caching")
        if n_tile_group != 2 or group_pipeline_schedule != "a_double_buffer":
            raise ValueError(
                "route_store requires n_tile_group=2 and a_double_buffer"
            )
    if node_accumulation_mode == "rank_local":
        if not getattr(s2, "include_rank_partials", False):
            raise ValueError("rank_local requires a rank-partial Stage2 arena")
        if accumulator_dtype != "bf16" or bf16_atomic_kind != "buffer":
            raise ValueError(
                "rank_local requires the BF16 buffer-atomic path"
            )
        if epilogue_schedule != "lane32_meta":
            raise ValueError("rank_local requires lane32_meta metadata caching")
        if n_tile_group != 2 or group_pipeline_schedule != "a_double_buffer":
            raise ValueError(
                "rank_local requires n_tile_group=2 and a_double_buffer"
            )
        if node_reduce_schedule != "token":
            raise ValueError("rank_local requires the token reducer schedule")
        if rank_accumulation_mode == "staged_reduce" and not getattr(
            s2, "include_staged_reduce", False
        ):
            raise ValueError(
                "staged_reduce requires a Stage-2 arena with staged rank regions"
            )
        if rank_accumulation_mode == "staged_ring" and not getattr(
            s2, "include_staged_ring", False
        ):
            raise ValueError(
                "staged_ring requires a Stage-2 arena with ring rank regions"
            )
    if scoreboard_schedule not in ("wave0", "four_wave"):
        raise ValueError("scoreboard_schedule must be wave0 or four_wave")
    if atomic_issue_schedule not in ("interleaved", "preload_pairs"):
        raise ValueError(
            "atomic_issue_schedule must be interleaved or preload_pairs"
        )
    if atomic_issue_schedule == "preload_pairs" and (
        accumulator_dtype != "bf16" or bf16_atomic_kind != "buffer"
    ):
        raise ValueError("preload_pairs requires the BF16 buffer-atomic path")

    rank = int(rank)
    node = rank // GPUS_PER_NODE
    local_rank = rank % GPUS_PER_NODE
    remote_node = 1 - node
    local_plane = node
    remote_plane = remote_node
    hidden_tiles = HIDDEN // BN
    # staged_ring reserves bx=1 for the dedicated local stage reducer.  All
    # existing rank reducers/final/GMM roles are shifted by one CTA only for
    # this opt-in mode; atomic and legacy staged_reduce retain byte-for-byte
    # geometry.
    stage_ring_reducer_blocks = 1 if rank_accumulation_mode == "staged_ring" else 0
    reduce_first = 1 + stage_ring_reducer_blocks
    reduce_blocks = (
        node_reduce_blocks
        if node_accumulation_mode in ("route_store", "rank_local")
        else 0
    )
    combine_first = reduce_first + reduce_blocks
    gmm_first = combine_first + final_combine_blocks
    final_work_items = (
        MAX_TOKENS
        if node_accumulation_mode == "rank_local"
        else MAX_TOKENS * hidden_tiles
    )
    max_m_blocks = s1.max_route_tiles
    wire = Stage2NodePartialWire(HIDDEN, s2.records_per_group)
    return_groups = wire.group_count(MAX_TOKENS)
    if s2.num_qp != 4 or wire.records_per_group != 4 or return_groups != 32:
        raise ValueError("direct Stage-2 requires 4 QPs, 4 records/group, 32 groups")

    # Stage-1 compute outputs and row metadata.
    h1_q_off, h1_q_stride = _parity_parts(s1.region("h1_output_q"), 2)
    h1_scale_off, h1_scale_stride = _parity_parts(s1.region("h1_output_scale"), 2)
    expert_off, expert_stride = _parity_parts(s1.region("tile_expert"), 2)
    nvalid_off, nvalid_stride = _parity_parts(s1.region("num_valid"), 2)
    source_off, source_stride = _parity_parts(s1.region("tile_row_source"), 2)
    weight_off, weight_stride = _parity_parts(s1.region("tile_row_weight"), 2)
    dispatch_staging_off, dispatch_staging_stride = _parity_parts(
        s1.region("dispatch_staging"), 2
    )
    remote_dispatch_off, remote_dispatch_stride = _parity_parts(
        s1.region("remote_dispatch_rx"), 2
    )
    dispatch_record_bytes = s1.wire.record_bytes
    rank_slot_masks_offset = s1.wire.rank_slot_masks_offset

    # Stage-2 scoreboard/payload regions, relative to the logical Stage-2 base.
    dest_rank_mask_off, dest_rank_mask_stride = _parity_parts(
        s2.region("node_dest_rank_mask"), 2
    )
    expected_off, expected_stride = _parity_parts(s2.region("node_expected"), 2)
    accumulator_off, accumulator_stride = _parity_parts(
        s2.region("node_accumulator"), 2
    )
    done_off, done_stride = _parity_parts(s2.region("node_done"), 2)
    token_done_off, token_done_stride = _parity_parts(
        s2.region("node_token_done"), 2
    )
    token_ready_off, token_ready_stride = _parity_parts(
        s2.region("node_token_ready"), 2
    )
    if node_accumulation_mode == "route_store":
        route_slots_off, route_slots_stride = _parity_parts(
            s2.region("route_slots"), 2
        )
    else:
        route_slots_off = route_slots_stride = 0
    if node_accumulation_mode == "rank_local":
        rank_accumulator_off, rank_accumulator_stride = _parity_parts(
            s2.region("rank_accumulator"), 2
        )
        rank_pending_off, rank_pending_stride = _parity_parts(
            s2.region("rank_token_pending"), 2
        )
        rank_ready_off, rank_ready_stride = _parity_parts(
            s2.region("rank_token_ready"), 2
        )
        rank_tx_slot_off, rank_tx_slot_stride = _parity_parts(
            s2.region("rank_return_tx_slot"), 2
        )
        rank_rx_slot_off, rank_rx_slot_stride = _parity_parts(
            s2.region("rank_return_rx_slot"), 2
        )
        rank_return_count_off, rank_return_count_stride = _parity_parts(
            s2.region("rank_return_count"), 2
        )
        rank_reduce_queue_off, rank_reduce_queue_stride = _parity_parts(
            s2.region("rank_reduce_queue"), 2
        )
        rank_reduce_queue_ready_off, rank_reduce_queue_ready_stride = (
            _parity_parts(s2.region("rank_reduce_queue_ready"), 2)
        )
        rank_reduce_queue_tail_off, rank_reduce_queue_tail_stride = (
            _parity_parts(s2.region("rank_reduce_queue_tail"), 2)
        )
        rank_reduce_queue_head_off, rank_reduce_queue_head_stride = (
            _parity_parts(s2.region("rank_reduce_queue_head"), 2)
        )
        if rank_accumulation_mode == "staged_reduce":
            rank_stage_values_off, rank_stage_values_stride = _parity_parts(
                s2.region("rank_stage_values"), 2
            )
            rank_stage_slot_generation_off, rank_stage_slot_generation_stride = (
                _parity_parts(s2.region("rank_stage_slot_generation"), 2)
            )
            rank_stage_group_pending_off, rank_stage_group_pending_stride = (
                _parity_parts(s2.region("rank_stage_group_pending"), 2)
            )
            rank_stage_tile_pending_off, rank_stage_tile_pending_stride = (
                _parity_parts(s2.region("rank_stage_tile_pending"), 2)
            )
            rank_stage_tile_done_off, rank_stage_tile_done_stride = _parity_parts(
                s2.region("rank_stage_tile_done"), 2
            )
        else:
            rank_stage_values_off = rank_stage_values_stride = 0
            rank_stage_slot_generation_off = rank_stage_slot_generation_stride = 0
            rank_stage_group_pending_off = rank_stage_group_pending_stride = 0
            rank_stage_tile_pending_off = rank_stage_tile_pending_stride = 0
            rank_stage_tile_done_off = rank_stage_tile_done_stride = 0
        if rank_accumulation_mode == "staged_ring":
            rank_stage_ring_payload_off, rank_stage_ring_payload_stride = _parity_parts(
                s2.region("rank_stage_ring_payload"), 2
            )
            rank_stage_ring_source_off, rank_stage_ring_source_stride = _parity_parts(
                s2.region("rank_stage_ring_source"), 2
            )
            rank_stage_ring_slot_off, rank_stage_ring_slot_stride = _parity_parts(
                s2.region("rank_stage_ring_slot"), 2
            )
            rank_stage_ring_tile_off, rank_stage_ring_tile_stride = _parity_parts(
                s2.region("rank_stage_ring_tile"), 2
            )
            rank_stage_ring_sequence_off, rank_stage_ring_sequence_stride = _parity_parts(
                s2.region("rank_stage_ring_sequence"), 2
            )
            rank_stage_ring_head_off, rank_stage_ring_head_stride = _parity_parts(
                s2.region("rank_stage_ring_head"), 2
            )
            rank_stage_ring_tail_off, rank_stage_ring_tail_stride = _parity_parts(
                s2.region("rank_stage_ring_tail"), 2
            )
            rank_stage_ring_claim_off, rank_stage_ring_claim_stride = _parity_parts(
                s2.region("rank_stage_ring_claim"), 2
            )
            rank_stage_ring_reserve_lock_off, rank_stage_ring_reserve_lock_stride = _parity_parts(
                s2.region("rank_stage_ring_reserve_lock"), 2
            )
            rank_stage_ring_producer_done_off, rank_stage_ring_producer_done_stride = _parity_parts(
                s2.region("rank_stage_ring_producer_done"), 2
            )
            rank_stage_ring_reducer_done_off, rank_stage_ring_reducer_done_stride = _parity_parts(
                s2.region("rank_stage_ring_reducer_done"), 2
            )
            rank_stage_ring_scratch_off, rank_stage_ring_scratch_stride = _parity_parts(
                s2.region("rank_stage_ring_scratch"), 2
            )
            rank_stage_ring_seen_off, rank_stage_ring_seen_stride = _parity_parts(
                s2.region("rank_stage_ring_seen"), 2
            )
        else:
            rank_stage_ring_payload_off = rank_stage_ring_payload_stride = 0
            rank_stage_ring_source_off = rank_stage_ring_source_stride = 0
            rank_stage_ring_slot_off = rank_stage_ring_slot_stride = 0
            rank_stage_ring_tile_off = rank_stage_ring_tile_stride = 0
            rank_stage_ring_sequence_off = rank_stage_ring_sequence_stride = 0
            rank_stage_ring_head_off = rank_stage_ring_head_stride = 0
            rank_stage_ring_tail_off = rank_stage_ring_tail_stride = 0
            rank_stage_ring_claim_off = rank_stage_ring_claim_stride = 0
            rank_stage_ring_reserve_lock_off = rank_stage_ring_reserve_lock_stride = 0
            rank_stage_ring_producer_done_off = rank_stage_ring_producer_done_stride = 0
            rank_stage_ring_reducer_done_off = rank_stage_ring_reducer_done_stride = 0
            rank_stage_ring_scratch_off = rank_stage_ring_scratch_stride = 0
            rank_stage_ring_seen_off = rank_stage_ring_seen_stride = 0
    else:
        rank_accumulator_off = rank_accumulator_stride = 0
        rank_pending_off = rank_pending_stride = 0
        rank_ready_off = rank_ready_stride = 0
        rank_tx_slot_off = rank_tx_slot_stride = 0
        rank_rx_slot_off = rank_rx_slot_stride = 0
        rank_return_count_off = rank_return_count_stride = 0
        rank_reduce_queue_off = rank_reduce_queue_stride = 0
        rank_reduce_queue_ready_off = rank_reduce_queue_ready_stride = 0
        rank_reduce_queue_tail_off = rank_reduce_queue_tail_stride = 0
        rank_reduce_queue_head_off = rank_reduce_queue_head_stride = 0
        rank_stage_values_off = rank_stage_values_stride = 0
        rank_stage_slot_generation_off = rank_stage_slot_generation_stride = 0
        rank_stage_group_pending_off = rank_stage_group_pending_stride = 0
        rank_stage_tile_pending_off = rank_stage_tile_pending_stride = 0
        rank_stage_tile_done_off = rank_stage_tile_done_stride = 0
        rank_stage_ring_payload_off = rank_stage_ring_payload_stride = 0
        rank_stage_ring_source_off = rank_stage_ring_source_stride = 0
        rank_stage_ring_slot_off = rank_stage_ring_slot_stride = 0
        rank_stage_ring_tile_off = rank_stage_ring_tile_stride = 0
        rank_stage_ring_sequence_off = rank_stage_ring_sequence_stride = 0
        rank_stage_ring_head_off = rank_stage_ring_head_stride = 0
        rank_stage_ring_tail_off = rank_stage_ring_tail_stride = 0
        rank_stage_ring_claim_off = rank_stage_ring_claim_stride = 0
        rank_stage_ring_reserve_lock_off = rank_stage_ring_reserve_lock_stride = 0
        rank_stage_ring_producer_done_off = rank_stage_ring_producer_done_stride = 0
        rank_stage_ring_reducer_done_off = rank_stage_ring_reducer_done_stride = 0
        rank_stage_ring_scratch_off = rank_stage_ring_scratch_stride = 0
        rank_stage_ring_seen_off = rank_stage_ring_seen_stride = 0
    if node_accumulation_mode in ("route_store", "rank_local"):
        partial_done_off, partial_done_stride = _parity_parts(
            s2.region("node_partial_done"), 2
        )
        partial_ready_off, partial_ready_stride = _parity_parts(
            s2.region("node_partial_ready"), 2
        )
    else:
        partial_done_off = partial_done_stride = 0
        partial_ready_off = partial_ready_stride = 0
    tx_off, tx_stride = _parity_parts(s2.region("remote_node_tx"), 2)
    rx_off, rx_stride = _parity_parts(s2.region("remote_partial_rx"), 2)
    return_ready_off, return_ready_stride = _parity_parts(
        s2.region("return_group_ready"), 2
    )
    return_posted_off, return_posted_stride = _parity_parts(
        s2.region("return_count"), 2
    )
    return_reuse_off, return_reuse_stride = _parity_parts(
        s2.region("return_count_ready"), 2
    )
    consumed_off, consumed_stride = _parity_parts(s2.region("return_consumed"), 2)
    stage1_done_off, stage1_done_stride = _parity_parts(s2.region("stage1_done"), 2)
    stage2_phase_off, stage2_phase_stride = _parity_parts(s2.region("stage2_init"), 2)
    timeline_off, timeline_stride = _parity_parts(s2.region("timeline"), 2)
    timeline_gmm_done_off, timeline_gmm_done_stride = _parity_parts(
        s2.region("timeline_gmm_worker_done"), 2
    )
    grid_barrier_off = s2.region("grid_barrier").offset
    gemm_head_off = s2.region("gemm_work_head").offset
    final_head_off = s2.region("final_work_head").offset
    final_done_off = s2.region("final_done").offset
    error_off = s2.region("stage2_error_count").offset

    # Custom FP32 C-shuffle slab plus an optional disjoint A slab and eight
    # peer Stage-2 base pointers.  The disjoint slab lets the grouped pipeline
    # finish the next tile's required A DMA before atomic0 is issued; otherwise
    # the later A dependency forces vmcnt(0) and drains every outstanding
    # atomic before GEMM1 reaches its first MFMA.
    a_stages = kStages + 1
    kh_tile_a = BK // 2
    c_shuffle_bytes = BM * BN * (
        2
        if node_accumulation_mode in ("route_store", "rank_local")
        else 4
    )
    compute_lds_bytes = max(c_shuffle_bytes, a_stages * BM * kh_tile_a)
    a_double_buffer_bytes = (
        a_stages * BM * kh_tile_a
        if group_pipeline_schedule == "a_double_buffer"
        else 0
    )
    a_double_buffer_off = compute_lds_bytes
    peer_table_off = compute_lds_bytes + a_double_buffer_bytes
    timeline_scratch_off = peer_table_off + GPUS_PER_NODE * 8
    timeline_scratch_slots = 14
    timeline_scratch_bytes = timeline_scratch_slots * 8 if timeline_instrument else 0
    epilogue_meta_off = timeline_scratch_off + timeline_scratch_bytes
    epilogue_weight_off = epilogue_meta_off + BM * 4
    epilogue_expected_off = epilogue_weight_off + BM * 4
    lds_bytes = epilogue_meta_off + (
        (
            BM * 12
            if epilogue_schedule == "lane32_meta_expected"
            else BM * 8
        )
        if epilogue_schedule != "lane32"
        else 0
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[Int8, lds_bytes, 16]

    kernel_name = (
        "megamoe_tile_ep16_stage2_direct_node_a4w4_"
        f"r{rank}_h{HIDDEN}_i{INTER}_e{EXPERTS}_bm{BM}_bn{BN}_bk{BK}_q4_direct_v6_gridrelease"
        f"_acc{accumulator_dtype}_fc{final_combine_blocks}"
        f"_gs{gmm_schedule}"
        f"_rt{return_chunk_tokens}"
        "_nrtoken"
        f"_ba{bf16_atomic_kind}"
        f"_rr{rail_return_schedule}"
        f"_epi{epilogue_schedule}"
        f"_ng{n_tile_group}"
        f"_gp{group_pipeline_schedule}"
        f"_na{node_accumulation_mode}"
        f"_ram{rank_accumulation_mode}"
        f"_nr{node_reduce_blocks}v{node_reduce_vec_bytes}{node_reduce_schedule}"
        f"{node_reduce_load_schedule}"
        f"_nrw{node_reduce_work_schedule}"
        f"_nrr{node_reduce_rejoin_blocks}"
        f"_rla{rank_epilogue_lds_addressing}"
        f"_sb{scoreboard_schedule}"
        f"_ai{atomic_issue_schedule}"
        + ("_timeline" if timeline_instrument else "")
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
        parity = generation & fx.Int64(1)
        s2_base = arena_ptr + fx.Int64(s2_window_off)

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

        arg_aq = s1_ptr(h1_q_off, h1_q_stride)
        arg_ascale = s1_ptr(h1_scale_off, h1_scale_stride)
        arg_eids = s1_ptr(expert_off, expert_stride)
        arg_nvalid = s1_ptr(nvalid_off, nvalid_stride)
        arg_stids = s1_ptr(source_off, source_stride)
        arg_sweights = s1_ptr(weight_off, weight_stride)
        local_dispatch_records = s1_ptr(
            dispatch_staging_off, dispatch_staging_stride
        )
        remote_dispatch_records = s1_ptr(
            remote_dispatch_off, remote_dispatch_stride
        )
        dest_rank_mask_ptr = s2_ptr(
            dest_rank_mask_off, dest_rank_mask_stride
        )
        expected_ptr = s2_ptr(expected_off, expected_stride)
        accumulator_ptr = s2_ptr(accumulator_off, accumulator_stride)
        done_ptr = s2_ptr(done_off, done_stride)
        token_done_ptr = s2_ptr(token_done_off, token_done_stride)
        token_ready_ptr = s2_ptr(token_ready_off, token_ready_stride)
        route_slots_ptr = s2_ptr(route_slots_off, route_slots_stride)
        rank_accumulator_ptr = s2_ptr(
            rank_accumulator_off, rank_accumulator_stride
        )
        rank_pending_ptr = s2_ptr(rank_pending_off, rank_pending_stride)
        rank_ready_ptr = s2_ptr(rank_ready_off, rank_ready_stride)
        rank_tx_slot_ptr = s2_ptr(rank_tx_slot_off, rank_tx_slot_stride)
        rank_rx_slot_ptr = s2_ptr(rank_rx_slot_off, rank_rx_slot_stride)
        rank_return_count_ptr = s2_ptr(
            rank_return_count_off, rank_return_count_stride
        )
        rank_reduce_queue_ptr = s2_ptr(
            rank_reduce_queue_off, rank_reduce_queue_stride
        )
        rank_reduce_queue_ready_ptr = s2_ptr(
            rank_reduce_queue_ready_off, rank_reduce_queue_ready_stride
        )
        rank_reduce_queue_tail_ptr = s2_ptr(
            rank_reduce_queue_tail_off, rank_reduce_queue_tail_stride
        )
        rank_reduce_queue_head_ptr = s2_ptr(
            rank_reduce_queue_head_off, rank_reduce_queue_head_stride
        )
        rank_stage_values_ptr = s2_ptr(
            rank_stage_values_off, rank_stage_values_stride
        )
        rank_stage_slot_generation_ptr = s2_ptr(
            rank_stage_slot_generation_off, rank_stage_slot_generation_stride
        )
        rank_stage_group_pending_ptr = s2_ptr(
            rank_stage_group_pending_off, rank_stage_group_pending_stride
        )
        rank_stage_tile_pending_ptr = s2_ptr(
            rank_stage_tile_pending_off, rank_stage_tile_pending_stride
        )
        rank_stage_tile_done_ptr = s2_ptr(
            rank_stage_tile_done_off, rank_stage_tile_done_stride
        )
        rank_stage_ring_payload_ptr = s2_ptr(
            rank_stage_ring_payload_off, rank_stage_ring_payload_stride
        )
        rank_stage_ring_source_ptr = s2_ptr(
            rank_stage_ring_source_off, rank_stage_ring_source_stride
        )
        rank_stage_ring_slot_ptr = s2_ptr(
            rank_stage_ring_slot_off, rank_stage_ring_slot_stride
        )
        rank_stage_ring_tile_ptr = s2_ptr(
            rank_stage_ring_tile_off, rank_stage_ring_tile_stride
        )
        rank_stage_ring_sequence_ptr = s2_ptr(
            rank_stage_ring_sequence_off, rank_stage_ring_sequence_stride
        )
        rank_stage_ring_head_ptr = s2_ptr(
            rank_stage_ring_head_off, rank_stage_ring_head_stride
        )
        rank_stage_ring_tail_ptr = s2_ptr(
            rank_stage_ring_tail_off, rank_stage_ring_tail_stride
        )
        rank_stage_ring_claim_ptr = s2_ptr(
            rank_stage_ring_claim_off, rank_stage_ring_claim_stride
        )
        rank_stage_ring_reserve_lock_ptr = s2_ptr(
            rank_stage_ring_reserve_lock_off,
            rank_stage_ring_reserve_lock_stride,
        )
        rank_stage_ring_producer_done_ptr = s2_ptr(
            rank_stage_ring_producer_done_off,
            rank_stage_ring_producer_done_stride,
        )
        rank_stage_ring_reducer_done_ptr = s2_ptr(
            rank_stage_ring_reducer_done_off,
            rank_stage_ring_reducer_done_stride,
        )
        rank_stage_ring_scratch_ptr = s2_ptr(
            rank_stage_ring_scratch_off, rank_stage_ring_scratch_stride
        )
        rank_stage_ring_seen_ptr = s2_ptr(
            rank_stage_ring_seen_off, rank_stage_ring_seen_stride
        )
        partial_done_ptr = s2_ptr(partial_done_off, partial_done_stride)
        partial_ready_ptr = s2_ptr(partial_ready_off, partial_ready_stride)
        rail_payload_ready_ptr = (
            partial_ready_ptr
            if node_accumulation_mode in ("route_store", "rank_local")
            else token_ready_ptr
        )
        tx_ptr = s2_ptr(tx_off, tx_stride)
        rx_ptr = s2_ptr(rx_off, rx_stride)
        return_ready_ptr = s2_ptr(return_ready_off, return_ready_stride)
        return_posted_ptr = s2_ptr(return_posted_off, return_posted_stride)
        return_reuse_ptr = s2_ptr(return_reuse_off, return_reuse_stride)
        consumed_ptr = s2_ptr(consumed_off, consumed_stride)
        stage1_done_ptr = s2_ptr(stage1_done_off, stage1_done_stride)
        stage2_phase_ptr = s2_ptr(stage2_phase_off, stage2_phase_stride)
        timeline_ptr = s2_ptr(timeline_off, timeline_stride)
        timeline_gmm_done_ptr = s2_ptr(
            timeline_gmm_done_off, timeline_gmm_done_stride
        )
        grid_barrier_ptr = s2_base + fx.Int64(grid_barrier_off)
        gemm_head_ptr = s2_base + fx.Int64(gemm_head_off)
        final_head_ptr = s2_base + fx.Int64(final_head_off)
        final_done_ptr = s2_base + fx.Int64(final_done_off)
        error_ptr = s2_base + fx.Int64(error_off)

        def add_error():
            comm_ops.atomic_add_system(error_ptr, fx.Int32(1))

        def store_timeline_scratch(marker_name, scratch_index):
            comm_ops.store_i64_global_relaxed(
                timeline_ptr
                + fx.Int64(STAGE2_TIMELINE_INDEX[marker_name] * 8),
                fx.ptr_load(timeline_scratch + fx.Int32(scratch_index)),
            )

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
                target = ((ticket // grid64) + fx.Int64(1)) * grid64
                seen = fx.Int64(
                    comm_ops.atomic_add_agent(grid_barrier_ptr, fx.Int64(0))
                )
                while seen < target:
                    seen = fx.Int64(
                        comm_ops.atomic_add_agent(grid_barrier_ptr, fx.Int64(0))
                    )
                comm_ops.fence_agent_acquire()
            gpu.barrier()

        if const_expr(timeline_instrument):
            if bx == fx.Int32(CROSS_BLOCK) and tx == fx.Int32(0):
                comm_ops.store_i64_global_relaxed(
                    timeline_ptr
                    + fx.Int64(STAGE2_TIMELINE_INDEX["stage2_entry"] * 8),
                    fx.Int64(comm_ops.read_wall_clock()),
                )

        # Stage-1 completion from every local rank closes all scoreboards and
        # H1 rows before Stage-2 clears its accumulator parity.
        if const_expr(node_accumulation_mode != "rank_local"):
            if bx == fx.Int32(0) and tx == fx.Int32(0):
                for peer in range_constexpr(GPUS_PER_NODE):
                    peer_s2 = fx.Int64(
                        arena_window.lsa_ptr(
                            fx.Int32(peer), fx.Int64(s2_window_off)
                        )
                    )
                    wait_ready(
                        peer_s2
                        + fx.Int64(stage1_done_off)
                        + parity * fx.Int64(stage1_done_stride),
                        generation,
                    )
            grid_sync()
            comm_ops.fence_system_acquire()
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
                comm_ops.store_i32_system(
                    gemm_head_ptr, tx * fx.Int32(16), fx.Int32(0)
                )
            if tx == fx.Int32(0):
                comm_ops.store_i32_system(final_head_ptr, fx.Int32(0), fx.Int32(0))
                comm_ops.store_i32_system(final_done_ptr, fx.Int32(0), fx.Int32(0))
                comm_ops.store_i32_system(error_ptr, fx.Int32(0), fx.Int32(0))
        if const_expr(node_accumulation_mode != "rank_local"):
            grid_sync()

        # The node accumulator is the only large clear. The arena retains FP32
        # capacity for the diagnostic reference; production uses its compact
        # BF16 logical prefix.
        acc_rsrc = buffer_ops.create_buffer_resource_from_addr(accumulator_ptr)
        if const_expr(accumulator_dtype == "bf16"):
            zero4 = fx.Vector.filled(4, 0, fx.Int32)
            acc_vec4 = fx.Int32(
                (2 * MAX_TOKENS * HIDDEN * 2) // 16
                if node_accumulation_mode == "direct_atomic"
                or diagnostic_mode == "return_only"
                else 0
            )
            for item in range(global_tx, acc_vec4, grid_threads):
                buffer_ops.buffer_store(zero4, acc_rsrc, item * fx.Int32(4))
        else:
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            acc_vec4 = fx.Int32(
                (2 * MAX_TOKENS * HIDDEN) // 4
                if node_accumulation_mode == "direct_atomic"
                or diagnostic_mode == "return_only"
                else 0
            )
            for item in range(global_tx, acc_vec4, grid_threads):
                buffer_ops.buffer_store(zero4, acc_rsrc, item * fx.Int32(4))

        if const_expr(node_accumulation_mode == "rank_local"):
            # The local rank accumulator covers every global source token.
            # It is intentionally local: only the source-proxy reducer reads it
            # over LSA after this rank has completed the token.
            rank_acc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                rank_accumulator_ptr
            )
            rank_zero4 = fx.Vector.filled(4, 0, fx.Int32)
            rank_acc_vec4 = fx.Int32(
                (SOURCE_CAPACITY * HIDDEN * 2) // 16
            )
            for item in range(global_tx, rank_acc_vec4, grid_threads):
                buffer_ops.buffer_store(
                    rank_zero4, rank_acc_rsrc, item * fx.Int32(4)
                )
            if const_expr(rank_accumulation_mode == "staged_reduce"):
                stage_values_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_values_ptr
                )
                stage_slot_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_slot_generation_ptr
                )
                stage_group_pending_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_group_pending_ptr
                )
                stage_tile_pending_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_tile_pending_ptr
                )
                stage_tile_done_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_tile_done_ptr
                )
                stage_values_vec4 = fx.Int32(
                    (SOURCE_CAPACITY * TOPK * hidden_tiles * (2 * BN) * 2) // 16
                )
                for item in range(global_tx, stage_values_vec4, grid_threads):
                    buffer_ops.buffer_store(
                        rank_zero4, stage_values_rsrc, item * fx.Int32(4)
                    )
                stage_slots_vec2 = fx.Int32(
                    (SOURCE_CAPACITY * TOPK * 8) // 8
                )
                zero2 = fx.Vector.filled(2, 0, fx.Int32)
                for item in range(global_tx, stage_slots_vec2, grid_threads):
                    buffer_ops.buffer_store(
                        zero2, stage_slot_rsrc, item * fx.Int32(2)
                    )
                stage_counts_vec4 = fx.Int32(
                    (SOURCE_CAPACITY * hidden_tiles * 3) // 4
                )
                for item in range(global_tx, stage_counts_vec4, grid_threads):
                    buffer_ops.buffer_store(
                        rank_zero4, stage_group_pending_rsrc, item * fx.Int32(4)
                    )
                    buffer_ops.buffer_store(
                        rank_zero4, stage_tile_pending_rsrc, item * fx.Int32(4)
                    )
                    buffer_ops.buffer_store(
                        rank_zero4, stage_tile_done_rsrc, item * fx.Int32(4)
                    )
            if const_expr(rank_accumulation_mode == "staged_ring"):
                ring_scratch_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_scratch_ptr
                )
                zero_bf16x8 = fx.Vector.filled(8, 0, fx.Int16)
                ring_scratch_vec8 = fx.Int32(
                    (SOURCE_CAPACITY * HIDDEN) // 8
                )
                for item in range(global_tx, ring_scratch_vec8, grid_threads):
                    buffer_ops.buffer_store(
                        zero_bf16x8, ring_scratch_rsrc, item * fx.Int32(8)
                    )
                ring_seq_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_sequence_ptr
                )
                ring_seq_items = fx.Int32(8192)
                for item in range(global_tx, ring_seq_items, grid_threads):
                    buffer_ops.buffer_store(
                        fx.Int64(0), ring_seq_rsrc, item
                    )
                ring_head_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_head_ptr
                )
                ring_tail_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_tail_ptr
                )
                ring_prod_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_producer_done_ptr
                )
                ring_red_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_reducer_done_ptr
                )
                ring_claim_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_claim_ptr
                )
                ring_lock_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_reserve_lock_ptr
                )
                ring_seen_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_stage_ring_seen_ptr
                )
                ring_seen_items = fx.Int32(SOURCE_CAPACITY * hidden_tiles)
                for item in range(global_tx, ring_seen_items, grid_threads):
                    buffer_ops.buffer_store(
                        fx.Int32(0), ring_seen_rsrc, item
                    )
                if tx == fx.Int32(0):
                    buffer_ops.buffer_store(fx.Int64(0), ring_head_rsrc, 0)
                    buffer_ops.buffer_store(fx.Int64(0), ring_tail_rsrc, 0)
                    buffer_ops.buffer_store(fx.Int64(0), ring_claim_rsrc, 0)
                    buffer_ops.buffer_store(fx.Int32(0), ring_lock_rsrc, 0)
                    buffer_ops.buffer_store(fx.Int32(0), ring_prod_rsrc, 0)
                    buffer_ops.buffer_store(fx.Int32(0), ring_red_rsrc, 0)
            rank_pending_rsrc = buffer_ops.create_buffer_resource_from_addr(
                rank_pending_ptr
            )
            rank_partial_done_rsrc = (
                buffer_ops.create_buffer_resource_from_addr(partial_done_ptr)
            )
            rank_reduce_queue_rsrc = (
                buffer_ops.create_buffer_resource_from_addr(
                    rank_reduce_queue_ptr
                )
            )
            rank_reduce_tail_rsrc = (
                buffer_ops.create_buffer_resource_from_addr(
                    rank_reduce_queue_tail_ptr
                )
            )
            if const_expr(node_reduce_work_schedule == "dynamic_head"):
                rank_reduce_head_rsrc = (
                    buffer_ops.create_buffer_resource_from_addr(
                        rank_reduce_queue_head_ptr
                    )
                )
            rank_dest_mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                dest_rank_mask_ptr
            )
            if bx == fx.Int32(0):
                for item in range(
                    tx, fx.Int32(SOURCE_CAPACITY), fx.Int32(THREADS)
                ):
                    buffer_ops.buffer_store(
                        fx.Int32(0), rank_pending_rsrc, item * fx.Int32(16)
                    )
                if tx < fx.Int32(2 * MAX_TOKENS):
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        rank_partial_done_rsrc,
                        tx * fx.Int32(16),
                    )
                if tx == fx.Int32(0):
                    buffer_ops.buffer_store(
                        fx.Int32(0), rank_reduce_tail_rsrc, fx.Int32(0)
                    )
                    if const_expr(node_reduce_work_schedule == "dynamic_head"):
                        buffer_ops.buffer_store(
                            fx.Int32(0), rank_reduce_head_rsrc, fx.Int32(0)
                        )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_agent_release()
            # The rank-local path also clears shared ring/scratch state from
            # bx=0.  Gate every role before producers or the dedicated
            # reducer can observe partially initialized parity data.
            # Keep the following metadata initialization active for atomic and
            # staged modes too; only the ring mode needs the extra grid gate,
            # but this wrapper is constexpr so all rank-local variants retain
            # their existing metadata setup.
            if const_expr(True):
                if const_expr(rank_accumulation_mode == "staged_ring"):
                    grid_sync()
                rank_meta_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    arg_stids
                )
                rank_num_valid = fx.Int32(
                    global_typed_ptr(arg_nvalid, T.i32)[0]
                )
                for row in range(
                    tx, rank_num_valid, fx.Int32(THREADS)
                ):
                    packed = buffer_ops.buffer_load(
                        rank_meta_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    source = packed & fx.Int32(0x00FFFFFF)
                    if source < fx.Int32(SOURCE_CAPACITY):
                        comm_ops.atomic_add_agent(
                            rank_pending_ptr
                            + fx.Int64(source) * fx.Int64(64),
                            fx.Int32(hidden_tiles // n_tile_group),
                        )
                        if const_expr(rank_accumulation_mode == "staged_reduce"):
                            for group in range_constexpr(hidden_tiles // n_tile_group):
                                comm_ops.atomic_add_agent(
                                    rank_stage_group_pending_ptr
                                    + fx.Int64(
                                        source * (hidden_tiles // n_tile_group) + group
                                    ) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                                comm_ops.atomic_add_agent(
                                    rank_stage_tile_pending_ptr
                                    + fx.Int64(
                                        source * hidden_tiles + group * 2
                                    ) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                                comm_ops.atomic_add_agent(
                                    rank_stage_tile_pending_ptr
                                    + fx.Int64(
                                        source * hidden_tiles + group * 2 + 1
                                    ) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                if const_expr(rail_return_schedule == "compact"):
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        tx_count = fx.Int32(0)
                        rx_count = fx.Int32(0)
                        active_count = fx.Int32(0)
                        tx_slot_rsrc = (
                            buffer_ops.create_buffer_resource_from_addr(
                                rank_tx_slot_ptr
                            )
                        )
                        rx_slot_rsrc = (
                            buffer_ops.create_buffer_resource_from_addr(
                                rank_rx_slot_ptr
                            )
                        )
                        count_rsrc = (
                            buffer_ops.create_buffer_resource_from_addr(
                                rank_return_count_ptr
                            )
                        )
                        rank_mask_rsrc = (
                            buffer_ops.create_buffer_resource_from_addr(
                                dest_rank_mask_ptr
                            )
                        )
                        for token in range(
                            fx.Int32(0),
                            fx.Int32(MAX_TOKENS),
                            fx.Int32(1),
                        ):
                            local_mask = fx.Int32(
                                buffer_ops.buffer_load(
                                    rank_mask_rsrc,
                                    fx.Int32(local_plane * MAX_TOKENS)
                                    + token,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            ) & fx.Int32(0xFF)
                            local_active = local_mask != fx.Int32(0)
                            active_count = local_active.select(
                                active_count + fx.Int32(1), active_count
                            )
                            tx_mask = fx.Int32(
                                buffer_ops.buffer_load(
                                    rank_mask_rsrc,
                                    fx.Int32(remote_plane * MAX_TOKENS)
                                    + token,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            ) & fx.Int32(0xFF)
                            tx_active = tx_mask != fx.Int32(0)
                            buffer_ops.buffer_store(
                                tx_active.select(tx_count, fx.Int32(-1)),
                                tx_slot_rsrc,
                                token,
                            )
                            tx_count = tx_active.select(
                                tx_count + fx.Int32(1), tx_count
                            )
                            active_count = tx_active.select(
                                active_count + fx.Int32(1), active_count
                            )

                            record_addr = (
                                fx.Int64(local_dispatch_records)
                                + fx.Int64(token)
                                * fx.Int64(dispatch_record_bytes)
                                + fx.Int64(rank_slot_masks_offset)
                                + fx.Int64(
                                    remote_node * GPUS_PER_NODE * 2
                                )
                            )
                            remote_mask_rsrc = (
                                buffer_ops.create_buffer_resource_from_addr(
                                    record_addr,
                                    num_records_bytes=GPUS_PER_NODE * 2,
                                )
                            )
                            rx_mask = fx.Int32(0)
                            for mask_pair in range_constexpr(
                                GPUS_PER_NODE // 2
                            ):
                                packed_masks = fx.Int32(
                                    buffer_ops.buffer_load(
                                        remote_mask_rsrc,
                                        fx.Int32(mask_pair),
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                                rx_mask = rx_mask | (
                                    packed_masks & fx.Int32(0xFFFF)
                                )
                                rx_mask = rx_mask | (
                                    (packed_masks >> fx.Int32(16))
                                    & fx.Int32(0xFFFF)
                                )
                            rx_active = rx_mask != fx.Int32(0)
                            buffer_ops.buffer_store(
                                rx_active.select(rx_count, fx.Int32(-1)),
                                rx_slot_rsrc,
                                token,
                            )
                            rx_count = rx_active.select(
                                rx_count + fx.Int32(1), rx_count
                            )
                        buffer_ops.buffer_store(
                            tx_count, count_rsrc, fx.Int32(0)
                        )
                        buffer_ops.buffer_store(
                            rx_count, count_rsrc, fx.Int32(1)
                        )
                        buffer_ops.buffer_store(
                            active_count, count_rsrc, fx.Int32(2)
                        )
                if const_expr(rail_return_schedule != "compact"):
                    if tx < fx.Int32(2 * MAX_TOKENS):
                        rank_mask = fx.Int32(
                            buffer_ops.buffer_load(
                                rank_dest_mask_rsrc,
                                tx,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        ) & fx.Int32(0xFF)
                        if rank_mask == fx.Int32(0):
                            queue_slot = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    rank_reduce_queue_tail_ptr,
                                    fx.Int32(1),
                                )
                            )
                            comm_ops.store_i32_system(
                                rank_reduce_queue_ptr, queue_slot, tx
                            )
                            comm_ops.store_i64_global_system(
                                rank_reduce_queue_ready_ptr
                                + fx.Int64(queue_slot) * fx.Int64(8),
                                generation,
                            )

        # staged_ring metadata, scratch, and queue cursors are initialized by
        # the rank-local init CTA; close the resident-grid init phase before
        # any producer can reserve a ring slot.
        if const_expr(node_accumulation_mode == "rank_local"):
            grid_sync()

        done_rsrc = buffer_ops.create_buffer_resource_from_addr(done_ptr)
        scoreboard_items = fx.Int32(2 * MAX_TOKENS * hidden_tiles)
        if const_expr(node_accumulation_mode != "rank_local"):
            for item in range(global_tx, scoreboard_items, grid_threads):
                buffer_ops.buffer_store(fx.Int32(0), done_rsrc, item)
        token_items = fx.Int32(2 * MAX_TOKENS)
        token_done_rsrc = buffer_ops.create_buffer_resource_from_addr(
            token_done_ptr
        )
        if const_expr(node_accumulation_mode != "rank_local"):
            for item in range(global_tx, token_items, grid_threads):
                buffer_ops.buffer_store(
                    fx.Int32(0), token_done_rsrc, item * fx.Int32(16)
                )
        if const_expr(node_accumulation_mode == "route_store"):
            partial_done_rsrc = buffer_ops.create_buffer_resource_from_addr(
                partial_done_ptr
            )
            for item in range(global_tx, token_items, grid_threads):
                buffer_ops.buffer_store(
                    fx.Int32(0), partial_done_rsrc, item * fx.Int32(16)
                )
        expected_rsrc = buffer_ops.create_buffer_resource_from_addr(expected_ptr)
        if const_expr(node_accumulation_mode != "rank_local"):
            for item in range(global_tx, scoreboard_items, grid_threads):
                expected = buffer_ops.buffer_load(
                    expected_rsrc, item, vec_width=1, dtype=T.i32
                )
                invalid_expected = (expected < fx.Int32(0)) | (
                    expected > fx.Int32(TOPK)
                )
                if invalid_expected:
                    add_error()
        if bx == fx.Int32(0) and tx == fx.Int32(0):
            if local_tokens != fx.Int32(MAX_TOKENS):
                add_error()
        grid_sync()

        # A token whose routes all belong to the other node has no GMM2
        # producer. The cleared accumulator is already the correct zero node
        # partial, so publish whole-token readiness after initialization.
        if const_expr(node_accumulation_mode != "rank_local"):
            comm_ops.fence_system_release()
            for item in range(global_tx, token_items, grid_threads):
                expected = buffer_ops.buffer_load(
                    expected_rsrc,
                    item * fx.Int32(hidden_tiles),
                    vec_width=1,
                    dtype=T.i32,
                )
                if expected == fx.Int32(0):
                    buffer_ops.buffer_store(
                        fx.Int32(hidden_tiles),
                        token_done_rsrc,
                        item * fx.Int32(16),
                    )
                    comm_ops.store_i64_global_system(
                        token_ready_ptr + fx.Int64(item) * fx.Int64(8),
                        generation,
                    )
        if const_expr(node_accumulation_mode != "rank_local"):
            grid_sync()

        # Every target proxy must clear before any peer issues LSA atomics.
        init_phase = (generation << fx.Int64(2)) + fx.Int64(1)
        if const_expr(node_accumulation_mode == "rank_local"):
            if bx == fx.Int32(0) and tx == fx.Int32(0):
                comm_ops.store_i64_global_system(
                    stage2_phase_ptr, init_phase
                )
        else:
            if bx == fx.Int32(0) and tx == fx.Int32(0):
                comm_ops.store_i64_global_system(stage2_phase_ptr, init_phase)
            if bx == fx.Int32(0):
                gpu.barrier()
                if tx == fx.Int32(0):
                    for peer in range_constexpr(GPUS_PER_NODE):
                        peer_s2 = fx.Int64(
                            arena_window.lsa_ptr(
                                fx.Int32(peer), fx.Int64(s2_window_off)
                            )
                        )
                        wait_ready(
                            peer_s2
                            + fx.Int64(stage2_phase_off)
                            + parity * fx.Int64(stage2_phase_stride),
                            init_phase,
                        )
            grid_sync()
            comm_ops.fence_system_acquire()
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

        # Diagnostic return-only mode synthesizes the post-atomic readiness
        # boundary after the same accumulator clear and all-rank init fence as
        # production.  Both source-node planes are ready because the cleared
        # accumulator is the partial payload under test.
        if const_expr(diagnostic_mode == "return_only"):
            for item in range(global_tx, scoreboard_items, grid_threads):
                buffer_ops.buffer_store(
                    fx.Int32(GPUS_PER_NODE), done_rsrc, item
                )
            rocdl.s_waitcnt(0)
            grid_sync()
            comm_ops.fence_system_release()
            for item in range(global_tx, token_items, grid_threads):
                buffer_ops.buffer_store(
                    fx.Int32(hidden_tiles),
                    token_done_rsrc,
                    item * fx.Int32(16),
                )
                comm_ops.store_i64_global_system(
                    token_ready_ptr + fx.Int64(item) * fx.Int64(8),
                    generation,
                )
                if const_expr(
                    node_accumulation_mode in ("route_store", "rank_local")
                ):
                    comm_ops.store_i64_global_system(
                        partial_ready_ptr + fx.Int64(item) * fx.Int64(8),
                        generation,
                    )
            grid_sync()
            comm_ops.fence_system_acquire()

        role_enabled = fx.Int32(
            1 if diagnostic_mode in ("full", "return_only") else 0
        )
        compute_enabled = fx.Int32(
            1
            if diagnostic_mode
            in (
                "full",
                "atomic_only",
                "gmm2_only",
                "gmm2_atomic_only",
                "route_store_only",
            )
            else 0
        )
        reduce_enabled = fx.Int32(
            1
            if node_accumulation_mode in ("route_store", "rank_local")
            and diagnostic_mode
            in (
                "full",
                "atomic_only",
                "gmm2_atomic_only",
            )
            else 0
        )

        def load_rank_reduce_work_items():
            if const_expr(rail_return_schedule == "compact"):
                if tx == fx.Int32(0):
                    reduce_work_items = fx.Int32(
                        comm_ops.load_i32_global_system(
                            rank_return_count_ptr + fx.Int64(2 * 4)
                        )
                    )
                    fx.ptr_store(reduce_work_items, work_ptr)
                gpu.barrier()
                reduce_work_items = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32, fx.ptr_load(work_ptr).ir_value()
                    )
                )
            else:
                reduce_work_items = fx.Int32(2 * MAX_TOKENS)
            return reduce_work_items

        def reduce_rank_queue_slot(
            queue_slot,
            rank_reduce_acc_rsrc,
            rank_reduce_tx_rsrc,
            rank_mask_rsrc,
            rank_tx_slot_rsrc,
        ):
            """Pull and publish one ready rank-local reduction queue slot."""

            reduce_elems = node_reduce_vec_bytes // 2
            reduce_i32s = node_reduce_vec_bytes // 4
            tiles_per_wave = hidden_tiles // 4
            if const_expr(node_reduce_vec_bytes == 16):
                # A 16-byte load carries eight BF16 columns, so one half-wave
                # covers the complete 256-column tile.  Keep the upper lanes
                # masked and on a valid address because masked buffer ops may
                # still retain their address arithmetic during lowering.
                col_iterations = BN // (32 * reduce_elems)
            else:
                col_iterations = BN // (64 * reduce_elems)
            if tx == fx.Int32(0):
                wait_ready(
                    rank_reduce_queue_ready_ptr
                    + fx.Int64(queue_slot) * fx.Int64(8),
                    generation,
                )
                token_index = fx.Int32(
                    comm_ops.load_i32_global_system(
                        rank_reduce_queue_ptr
                        + fx.Int64(queue_slot) * fx.Int64(4)
                    )
                )
                fx.ptr_store(token_index, work_ptr)
            gpu.barrier()
            token_index = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, fx.ptr_load(work_ptr).ir_value()
                )
            )
            token = token_index % fx.Int32(MAX_TOKENS)
            source_plane = token_index // fx.Int32(MAX_TOKENS)
            global_source = (
                source_plane * fx.Int32(GPUS_PER_NODE)
                + fx.Int32(local_rank)
            ) * fx.Int32(MAX_TOKENS) + token
            if tx == fx.Int32(0):
                rank_mask = fx.Int32(
                    buffer_ops.buffer_load(
                        rank_mask_rsrc,
                        token_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
                ) & fx.Int32(0xFF)
                fx.ptr_store(rank_mask, work_ptr + fx.Int32(1))
                tx_slot = fx.Int32(
                    buffer_ops.buffer_load(
                        rank_tx_slot_rsrc,
                        token,
                        vec_width=1,
                        dtype=T.i32,
                        mask=source_plane == fx.Int32(remote_plane),
                    )
                )
                fx.ptr_store(tx_slot, work_ptr + fx.Int32(2))
                if (rank_mask & fx.Int32(~0xFF)) != fx.Int32(0):
                    add_error()
            gpu.barrier()
            rank_mask = fx.Int32(
                rocdl.readfirstlane(
                    T.i32,
                    fx.ptr_load(work_ptr + fx.Int32(1)).ir_value(),
                )
            )
            tx_slot = fx.Int32(
                rocdl.readfirstlane(
                    T.i32,
                    fx.ptr_load(work_ptr + fx.Int32(2)).ir_value(),
                )
            )
            comm_ops.fence_system_acquire()

            for tile_iter in range_constexpr(tiles_per_wave):
                n_block = wave * fx.Int32(tiles_per_wave) + fx.Int32(
                    tile_iter
                )
                for col_iter in range_constexpr(col_iterations):
                    if const_expr(node_reduce_vec_bytes == 16):
                        lane_active = lane < fx.Int32(32)
                        col_in_tile = lane_active.select(
                            lane * fx.Int32(reduce_elems)
                            + fx.Int32(col_iter * 32 * reduce_elems),
                            fx.Int32(0),
                        )
                    else:
                        col_in_tile = (
                            lane * fx.Int32(reduce_elems)
                            + fx.Int32(col_iter * 64 * reduce_elems)
                        )
                    accum_values = [
                        fx.Float32(0.0)
                        for _ in range_constexpr(reduce_elems)
                    ]
                    rank_row_offset_bytes = fx.Int32(
                        rocdl.readfirstlane(
                            T.i32,
                            (
                                (
                                    global_source * fx.Int32(HIDDEN)
                                    + n_block * fx.Int32(BN)
                                )
                                * fx.Int32(2)
                            ).ir_value(),
                        )
                    )
                    if const_expr(
                        node_reduce_load_schedule == "load_first"
                    ):
                        packed_peers = []
                        for source_peer in range_constexpr(
                            GPUS_PER_NODE
                        ):
                            peer_active = (
                                rank_mask
                                & (
                                    fx.Int32(1)
                                    << fx.Int32(source_peer)
                                )
                            ) != fx.Int32(0)
                            peer_s2 = _wave_uniform_i64(
                                fx.ptr_load(
                                    lds_typed_ptr(
                                        lds_base
                                        + fx.Int32(peer_table_off)
                                        + fx.Int32(source_peer * 8),
                                        T.i64,
                                        align=8,
                                    )
                                )
                            )
                            peer_rank_rsrc = buffer_ops.create_buffer_resource_from_addr(
                                peer_s2
                                + fx.Int64(rank_accumulator_off)
                                + parity
                                * fx.Int64(rank_accumulator_stride),
                                num_records_bytes=rank_accumulator_stride,
                            )
                            if const_expr(node_reduce_vec_bytes == 16):
                                packed = buffer_ops.buffer_load(
                                    peer_rank_rsrc,
                                    col_in_tile // fx.Int32(2),
                                    vec_width=reduce_i32s,
                                    dtype=T.i32,
                                    mask=peer_active & lane_active,
                                    soffset_bytes=rank_row_offset_bytes,
                                )
                            else:
                                packed = buffer_ops.buffer_load(
                                    peer_rank_rsrc,
                                    col_in_tile // fx.Int32(2),
                                    vec_width=reduce_i32s,
                                    dtype=T.i32,
                                    mask=peer_active,
                                    soffset_bytes=rank_row_offset_bytes,
                                )
                            if const_expr(reduce_i32s == 1):
                                packed_peers.append(
                                    fx.Vector.from_elements(
                                        [packed], fx.Int32
                                    )
                                )
                            else:
                                packed_peers.append(fx.Vector(packed))
                        for source_peer in range_constexpr(
                            GPUS_PER_NODE
                        ):
                            values = packed_peers[source_peer].bitcast(
                                BFloat16
                            ).to(Float32)
                            for value_index in range_constexpr(
                                reduce_elems
                            ):
                                accum_values[value_index] = (
                                    accum_values[value_index]
                                    + fx.Float32(values[value_index])
                                )
                    else:
                        for source_peer in range_constexpr(
                            GPUS_PER_NODE
                        ):
                            peer_active = (
                                rank_mask
                                & (
                                    fx.Int32(1)
                                    << fx.Int32(source_peer)
                                )
                            ) != fx.Int32(0)
                            peer_s2 = _wave_uniform_i64(
                                fx.ptr_load(
                                    lds_typed_ptr(
                                        lds_base
                                        + fx.Int32(peer_table_off)
                                        + fx.Int32(source_peer * 8),
                                        T.i64,
                                        align=8,
                                    )
                                )
                            )
                            peer_rank_rsrc = buffer_ops.create_buffer_resource_from_addr(
                                peer_s2
                                + fx.Int64(rank_accumulator_off)
                                + parity
                                * fx.Int64(rank_accumulator_stride),
                                num_records_bytes=rank_accumulator_stride,
                            )
                            if const_expr(node_reduce_vec_bytes == 16):
                                packed = buffer_ops.buffer_load(
                                    peer_rank_rsrc,
                                    col_in_tile // fx.Int32(2),
                                    vec_width=reduce_i32s,
                                    dtype=T.i32,
                                    mask=peer_active & lane_active,
                                    soffset_bytes=rank_row_offset_bytes,
                                )
                            else:
                                packed = buffer_ops.buffer_load(
                                    peer_rank_rsrc,
                                    col_in_tile // fx.Int32(2),
                                    vec_width=reduce_i32s,
                                    dtype=T.i32,
                                    mask=peer_active,
                                    soffset_bytes=rank_row_offset_bytes,
                                )
                            if const_expr(reduce_i32s == 1):
                                packed_vec = fx.Vector.from_elements(
                                    [packed], fx.Int32
                                )
                            else:
                                packed_vec = fx.Vector(packed)
                            values = packed_vec.bitcast(BFloat16).to(
                                Float32
                            )
                            for value_index in range_constexpr(
                                reduce_elems
                            ):
                                accum_values[value_index] = (
                                    accum_values[value_index]
                                    + fx.Float32(values[value_index])
                                )
                    output_packed = (
                        fx.Vector.from_elements(accum_values, Float32)
                        .to(BFloat16)
                        .bitcast(fx.Int32)
                    )
                    output_bf16_index = (
                        token_index * fx.Int32(HIDDEN)
                        + n_block * fx.Int32(BN)
                        + col_in_tile
                    )
                    if const_expr(rail_return_schedule == "compact"):
                        if source_plane == fx.Int32(remote_plane):
                            tx_bf16_index = (
                                tx_slot * fx.Int32(HIDDEN)
                                + n_block * fx.Int32(BN)
                                + col_in_tile
                            )
                            if const_expr(node_reduce_vec_bytes == 16):
                                buffer_ops.buffer_store(
                                    output_packed,
                                    rank_reduce_tx_rsrc,
                                    tx_bf16_index // fx.Int32(2),
                                    mask=(rank_mask != fx.Int32(0))
                                    & lane_active,
                                )
                            elif const_expr(reduce_i32s == 1):
                                buffer_ops.buffer_store(
                                    output_packed[0],
                                    rank_reduce_tx_rsrc,
                                    tx_bf16_index // fx.Int32(2),
                                    mask=rank_mask != fx.Int32(0),
                                )
                            else:
                                buffer_ops.buffer_store(
                                    output_packed,
                                    rank_reduce_tx_rsrc,
                                    tx_bf16_index // fx.Int32(2),
                                    mask=rank_mask != fx.Int32(0),
                                )
                        elif const_expr(node_reduce_vec_bytes == 16):
                            buffer_ops.buffer_store(
                                output_packed,
                                rank_reduce_acc_rsrc,
                                output_bf16_index // fx.Int32(2),
                                mask=(rank_mask != fx.Int32(0)) & lane_active,
                            )
                        elif const_expr(reduce_i32s == 1):
                            buffer_ops.buffer_store(
                                output_packed[0],
                                rank_reduce_acc_rsrc,
                                output_bf16_index // fx.Int32(2),
                                mask=rank_mask != fx.Int32(0),
                            )
                        else:
                            buffer_ops.buffer_store(
                                output_packed,
                                rank_reduce_acc_rsrc,
                                output_bf16_index // fx.Int32(2),
                                mask=rank_mask != fx.Int32(0),
                            )
                    elif const_expr(node_reduce_vec_bytes == 16):
                        buffer_ops.buffer_store(
                            output_packed,
                            rank_reduce_acc_rsrc,
                            output_bf16_index // fx.Int32(2),
                            mask=lane_active,
                        )
                    elif const_expr(reduce_i32s == 1):
                        buffer_ops.buffer_store(
                            output_packed[0],
                            rank_reduce_acc_rsrc,
                            output_bf16_index // fx.Int32(2),
                        )
                    else:
                        buffer_ops.buffer_store(
                            output_packed,
                            rank_reduce_acc_rsrc,
                            output_bf16_index // fx.Int32(2),
                        )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            publish_partial = (
                rank_mask != fx.Int32(0)
                if const_expr(rail_return_schedule == "compact")
                else fx.Int32(1) == fx.Int32(1)
            )
            if (tx == fx.Int32(0)) & publish_partial:
                comm_ops.fence_system_release()
                ready_index = token_index
                if const_expr(rail_return_schedule == "compact"):
                    ready_index = (
                        source_plane == fx.Int32(remote_plane)
                    ).select(
                        fx.Int32(remote_plane * MAX_TOKENS) + tx_slot,
                        token_index,
                    )
                comm_ops.store_i64_global_system(
                    partial_ready_ptr
                    + fx.Int64(ready_index) * fx.Int64(8),
                    generation,
                )
            gpu.barrier()


        def consume_dynamic_rank_reduce(
            reduce_work_items,
            rank_reduce_acc_rsrc,
            rank_reduce_tx_rsrc,
            rank_mask_rsrc,
            rank_tx_slot_rsrc,
            wait_for_complete,
        ):
            """Drain unclaimed slots from the parity-local shared queue head."""

            if const_expr(wait_for_complete):
                if tx == fx.Int32(0):
                    queued = fx.Int32(
                        comm_ops.load_i32_global_system(
                            rank_reduce_queue_tail_ptr
                        )
                    )
                    while queued < reduce_work_items:
                        queued = fx.Int32(
                            comm_ops.load_i32_global_system(
                                rank_reduce_queue_tail_ptr
                            )
                        )
                    if queued > reduce_work_items:
                        add_error()
                gpu.barrier()
                comm_ops.fence_system_acquire()

            reduce_active = fx.Int32(1) == fx.Int32(1)
            while reduce_active:
                gpu.barrier()
                if tx == fx.Int32(0):
                    queue_slot = fx.Int32(
                        comm_ops.atomic_add_agent(
                            rank_reduce_queue_head_ptr, fx.Int32(1)
                        )
                    )
                    fx.ptr_store(queue_slot, work_ptr)
                gpu.barrier()
                queue_slot = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32, fx.ptr_load(work_ptr).ir_value()
                    )
                )
                has_reduce_work = queue_slot < reduce_work_items
                if has_reduce_work:
                    reduce_rank_queue_slot(
                        queue_slot,
                        rank_reduce_acc_rsrc,
                        rank_reduce_tx_rsrc,
                        rank_mask_rsrc,
                        rank_tx_slot_rsrc,
                    )
                reduce_active = has_reduce_work

        # ---- Compact active-token return for rank-local accumulation ----
        if (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)) & (
            fx.Int32(1 if rail_return_schedule == "compact" else 0)
            == fx.Int32(1)
        ):
            qp = wave
            count_rsrc = buffer_ops.create_buffer_resource_from_addr(
                rank_return_count_ptr
            )
            tx_count = fx.Int32(
                buffer_ops.buffer_load(
                    count_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )
            for batch in range_constexpr(
                MAX_TOKENS // (s2.num_qp * return_chunk_tokens)
            ):
                chunk = fx.Int32(batch * s2.num_qp) + qp
                first_slot = chunk * fx.Int32(return_chunk_tokens)
                remaining = tx_count - first_slot
                chunk_active = remaining > fx.Int32(0)
                chunk_count = (
                    remaining > fx.Int32(return_chunk_tokens)
                ).select(fx.Int32(return_chunk_tokens), remaining)
                chunk_count = chunk_active.select(chunk_count, fx.Int32(0))
                if lane < chunk_count:
                    ready_index = (
                        fx.Int32(remote_plane * MAX_TOKENS)
                        + first_slot
                        + lane
                    )
                    wait_ready(
                        partial_ready_ptr
                        + fx.Int64(ready_index) * fx.Int64(8),
                        generation,
                    )
                gpu.barrier()
                comm_ops.fence_system_acquire()
                if chunk_active:
                    src_rel = (
                        fx.Int64(tx_off)
                        + parity * fx.Int64(tx_stride)
                        + fx.Int64(first_slot) * fx.Int64(wire.record_bytes)
                    )
                    dst_rel = (
                        fx.Int64(rx_off)
                        + parity * fx.Int64(rx_stride)
                        + fx.Int64(first_slot) * fx.Int64(wire.record_bytes)
                    )
                    ready_rel = (
                        fx.Int64(return_ready_off)
                        + parity * fx.Int64(return_ready_stride)
                        + fx.Int64(chunk) * fx.Int64(8)
                    )
                    put(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        fx.Int64(s2_window_off) + dst_rel,
                        arena_win,
                        fx.Int64(s2_window_off) + src_rel,
                        fx.Int64(chunk_count) * fx.Int64(wire.record_bytes),
                        aggregate=True,
                        scope="warp",
                        team=team,
                    )
                    put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        fx.Int64(s2_window_off) + ready_rel,
                        generation,
                        aggregate=True,
                        scope="warp",
                        team=team,
                    )
                return_phase = (
                    generation << fx.Int64(3)
                ) + fx.Int64(batch + 1)
                if lane == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        return_posted_ptr + fx.Int64(qp) * fx.Int64(8),
                        return_phase,
                    )
                if wave == fx.Int32(0):
                    for stream_qp in range_constexpr(s2.num_qp):
                        if lane == fx.Int32(0):
                            wait_ready(
                                return_posted_ptr
                                + fx.Int64(stream_qp * 8),
                                return_phase,
                            )
                        comm_ops.fence_system_acquire()
                        stream_first_slot = fx.Int32(
                            (batch * s2.num_qp + stream_qp)
                            * return_chunk_tokens
                        )
                        if stream_first_slot < tx_count:
                            request = flush_async(
                                dev_comm,
                                fx.Int32(stream_qp),
                                fx.Int32(remote_node),
                                scope="warp",
                                team=team,
                            )
                            wait_request(
                                dev_comm,
                                fx.Int32(stream_qp),
                                request,
                                scope="warp",
                            )
                        if lane == fx.Int32(0):
                            comm_ops.store_i64_global_system(
                                return_reuse_ptr
                                + fx.Int64(stream_qp * 8),
                                return_phase,
                            )
                if lane == fx.Int32(0):
                    wait_ready(
                        return_reuse_ptr + fx.Int64(qp) * fx.Int64(8),
                        return_phase,
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
                put_value(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    arena_win,
                    remote_consumed,
                    generation,
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
                request = flush_async(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    scope="warp",
                    team=team,
                )
                wait_request(dev_comm, wave, request, scope="warp")
                if lane == fx.Int32(0):
                    wait_ready(consumed_ptr, generation)

        # ---- Independent QP progress; wave 0 remains the only doorbell owner ----
        elif (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)) & (
            fx.Int32(
                1
                if rail_return_schedule in ("qp_independent", "qp_prepost")
                else 0
            )
            == fx.Int32(1)
        ):
            qp = wave
            for batch in range_constexpr(
                MAX_TOKENS // (s2.num_qp * return_chunk_tokens)
            ):
                chunk = fx.Int32(batch * s2.num_qp) + qp
                first_token = chunk * fx.Int32(return_chunk_tokens)
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
                ready_rel = (
                    fx.Int64(return_ready_off)
                    + parity * fx.Int64(return_ready_stride)
                    + fx.Int64(chunk) * fx.Int64(8)
                )

                def append_return_wqes():
                    put(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        fx.Int64(s2_window_off) + dst_rel,
                        arena_win,
                        fx.Int64(s2_window_off) + src_rel,
                        fx.Int64(return_chunk_tokens * wire.record_bytes),
                        aggregate=True,
                        scope="warp",
                        team=team,
                    )
                    if const_expr(timeline_instrument) and const_expr(batch == 0):
                        if (wave == fx.Int32(0)) & (lane == fx.Int32(0)):
                            fx.ptr_store(
                                fx.Int64(comm_ops.read_wall_clock()),
                                timeline_scratch + fx.Int32(5),
                            )
                    put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        fx.Int64(s2_window_off) + ready_rel,
                        generation,
                        aggregate=True,
                        scope="warp",
                        team=team,
                    )

                if const_expr(rail_return_schedule == "qp_prepost"):
                    # Aggregate only reserves/fills the WQEs; the NIC cannot
                    # consume their source until wave0 rings this QP's
                    # doorbell.  Build descriptors while GMM2 is still making
                    # the first token group ready, then acquire and flush.
                    append_return_wqes()
                if lane < fx.Int32(return_chunk_tokens):
                    token = first_token + lane
                    token_ready_index = (
                        fx.Int32(remote_plane * MAX_TOKENS) + token
                    )
                    wait_ready(
                        rail_payload_ready_ptr
                        + fx.Int64(token_ready_index) * fx.Int64(8),
                        generation,
                    )
                if const_expr(timeline_instrument) and const_expr(batch == 0):
                    if (wave == fx.Int32(0)) & (lane == fx.Int32(0)):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + fx.Int32(0),
                        )
                comm_ops.fence_system_acquire()
                if const_expr(rail_return_schedule == "qp_independent"):
                    append_return_wqes()
                return_phase = (
                    generation << fx.Int64(3)
                ) + fx.Int64(batch + 1)
                if lane == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        return_posted_ptr + fx.Int64(qp) * fx.Int64(8),
                        return_phase,
                    )

                if wave == fx.Int32(0):
                    for stream_qp in range_constexpr(s2.num_qp):
                        if lane == fx.Int32(0):
                            wait_ready(
                                return_posted_ptr
                                + fx.Int64(stream_qp * 8),
                                return_phase,
                            )
                        comm_ops.fence_system_acquire()
                        if const_expr(timeline_instrument) and const_expr(
                            batch == 0
                        ) and const_expr(stream_qp == 0):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(10),
                                )
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(11),
                                )
                        request = flush_async(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            scope="warp",
                            team=team,
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            batch == 0
                        ) and const_expr(stream_qp == 0):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(12),
                                )
                        wait_request(
                            dev_comm,
                            fx.Int32(stream_qp),
                            request,
                            scope="warp",
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            batch == 0
                        ) and const_expr(stream_qp == 0):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(13),
                                )
                                store_timeline_scratch(
                                    "stage2_qp0_tokens_ready", 0
                                )
                                store_timeline_scratch(
                                    "stage2_first_batch_ready", 0
                                )
                                store_timeline_scratch(
                                    "stage2_qp0_payload_posted", 5
                                )
                                store_timeline_scratch(
                                    "stage2_first_batch_payloads_posted", 5
                                )
                                store_timeline_scratch(
                                    "stage2_return_terminal_posted", 10
                                )
                                store_timeline_scratch(
                                    "stage2_return_flush_pre", 11
                                )
                                store_timeline_scratch(
                                    "stage2_return_flush_post", 12
                                )
                                store_timeline_scratch(
                                    "stage2_return_request_done", 13
                                )
                        if lane == fx.Int32(0):
                            comm_ops.store_i64_global_system(
                                return_reuse_ptr
                                + fx.Int64(stream_qp * 8),
                                return_phase,
                            )

                if lane == fx.Int32(0):
                    wait_ready(
                        return_reuse_ptr + fx.Int64(qp) * fx.Int64(8),
                        return_phase,
                    )
                    wait_ready(
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
                put_value(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    arena_win,
                    remote_consumed,
                    generation,
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
                request = flush_async(
                    dev_comm, wave, fx.Int32(remote_node), scope="warp", team=team
                )
                wait_request(dev_comm, wave, request, scope="warp")
                if lane == fx.Int32(0):
                    wait_ready(consumed_ptr, generation)

        # -------- Lockstep baseline, four QPs, serial doorbells --------
        elif (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)) & (
            fx.Int32(1 if return_chunk_tokens > 4 else 0) == fx.Int32(1)
        ):
            qp = wave
            for batch in range_constexpr(
                MAX_TOKENS // (s2.num_qp * return_chunk_tokens)
            ):
                chunk = fx.Int32(batch * s2.num_qp) + qp
                first_token = chunk * fx.Int32(return_chunk_tokens)
                if lane < fx.Int32(return_chunk_tokens):
                    token = first_token + lane
                    token_ready_index = (
                        fx.Int32(remote_plane * MAX_TOKENS) + token
                    )
                    wait_ready(
                        rail_payload_ready_ptr
                        + fx.Int64(token_ready_index) * fx.Int64(8),
                        generation,
                    )
                if const_expr(timeline_instrument) and const_expr(batch == 0):
                    if lane == fx.Int32(0):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + wave,
                        )
                gpu.barrier()
                if const_expr(timeline_instrument) and const_expr(batch == 0):
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
                put(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                    arena_win,
                    fx.Int64(s2_window_off) + dst_rel,
                    arena_win,
                    fx.Int64(s2_window_off) + src_rel,
                    fx.Int64(return_chunk_tokens * wire.record_bytes),
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
                if const_expr(timeline_instrument) and const_expr(batch == 0):
                    if lane == fx.Int32(0):
                        fx.ptr_store(
                            fx.Int64(comm_ops.read_wall_clock()),
                            timeline_scratch + fx.Int32(5) + wave,
                        )
                gpu.barrier()
                if const_expr(timeline_instrument) and const_expr(batch == 0):
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
                        put_value(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            arena_win,
                            fx.Int64(s2_window_off) + ready_rel,
                            generation,
                            aggregate=True,
                            scope="warp",
                            team=team,
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            batch == 0
                        ) and const_expr(stream_qp == 0):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(10),
                                )
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(11),
                                )
                        request = flush_async(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            scope="warp",
                            team=team,
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            batch == 0
                        ) and const_expr(stream_qp == 0):
                            if lane == fx.Int32(0):
                                fx.ptr_store(
                                    fx.Int64(comm_ops.read_wall_clock()),
                                    timeline_scratch + fx.Int32(12),
                                )
                        wait_request(
                            dev_comm,
                            fx.Int32(stream_qp),
                            request,
                            scope="warp",
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            batch == 0
                        ) and const_expr(stream_qp == 0):
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
                    wait_ready(
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
                put_value(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    arena_win,
                    remote_consumed,
                    generation,
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
                request = flush_async(
                    dev_comm, wave, fx.Int32(remote_node), scope="warp", team=team
                )
                wait_request(dev_comm, wave, request, scope="warp")
                if lane == fx.Int32(0):
                    wait_ready(consumed_ptr, generation)

        # ---------------------- Role 0: reference return ----------------------
        elif (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)):
            qp = wave
            if const_expr(accumulator_dtype == "fp32"):
                node_acc = global_typed_ptr(accumulator_ptr, T.f32, align=4)
                tx_bf16 = global_typed_ptr(tx_ptr, T.bf16, align=2)
            for group in range(qp, fx.Int32(return_groups), fx.Int32(s2.num_qp)):
                if lane == fx.Int32(0):
                    for record in range_constexpr(wire.records_per_group):
                        token = group * fx.Int32(wire.records_per_group) + fx.Int32(record)
                        token_ready_index = (
                            fx.Int32(remote_plane * MAX_TOKENS) + token
                        )
                        wait_ready(
                            rail_payload_ready_ptr
                            + fx.Int64(token_ready_index) * fx.Int64(8),
                            generation,
                        )
                gpu.barrier()
                comm_ops.fence_system_acquire()

                if const_expr(accumulator_dtype == "fp32"):
                    group_elems = fx.Int32(wire.records_per_group * HIDDEN)
                    for elem in range(lane, group_elems, fx.Int32(64)):
                        record = elem // fx.Int32(HIDDEN)
                        col = elem - record * fx.Int32(HIDDEN)
                        token = group * fx.Int32(wire.records_per_group) + record
                        src_index = (
                            (fx.Int32(remote_plane * MAX_TOKENS) + token)
                            * fx.Int32(HIDDEN)
                            + col
                        )
                        dst_index = token * fx.Int32(HIDDEN) + col
                        value = fx.ptr_load(node_acc + fx.Int64(src_index))
                        fx.ptr_store(
                            fx.BFloat16(value), tx_bf16 + fx.Int64(dst_index)
                        )
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    comm_ops.fence_system_release()

                for record in range_constexpr(wire.records_per_group):
                    token = group * fx.Int32(wire.records_per_group) + fx.Int32(record)
                    if const_expr(accumulator_dtype == "bf16"):
                        src_rel = (
                            fx.Int64(accumulator_off)
                            + parity * fx.Int64(accumulator_stride)
                            + fx.Int64(remote_plane * MAX_TOKENS + token)
                            * fx.Int64(wire.record_bytes)
                        )
                    else:
                        src_rel = (
                            fx.Int64(tx_off)
                            + parity * fx.Int64(tx_stride)
                            + fx.Int64(token) * fx.Int64(wire.record_bytes)
                        )
                    dst_rel = (
                        fx.Int64(rx_off)
                        + parity * fx.Int64(rx_stride)
                        + fx.Int64(token) * fx.Int64(wire.record_bytes)
                    )
                    put(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        fx.Int64(s2_window_off) + dst_rel,
                        arena_win,
                        fx.Int64(s2_window_off) + src_rel,
                        fx.Int64(wire.record_bytes),
                        aggregate=True,
                        scope="warp",
                        team=team,
                    )
                ready_rel = (
                    fx.Int64(return_ready_off)
                    + parity * fx.Int64(return_ready_stride)
                    + fx.Int64(group) * fx.Int64(8)
                )
                put_value(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                    arena_win,
                    fx.Int64(s2_window_off) + ready_rel,
                    generation,
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
                request = flush_async(
                    dev_comm, qp, fx.Int32(remote_node), scope="warp", team=team
                )
                wait_request(dev_comm, qp, request, scope="warp")
                if lane == fx.Int32(0):
                    wait_ready(
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
                put_value(
                    dev_comm,
                    wave,
                    fx.Int32(remote_node),
                    arena_win,
                    remote_consumed,
                    generation,
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
                request = flush_async(
                    dev_comm, wave, fx.Int32(remote_node), scope="warp", team=team
                )
                wait_request(dev_comm, wave, request, scope="warp")
                if lane == fx.Int32(0):
                    wait_ready(consumed_ptr, generation)

        # ---- Dedicated staged-ring local reducer (opt-in) ----
        # bx=1 is reserved for this CTA.  The bounded ring metadata/payload
        # is append-only in the arena ABI; the reducer drains entries before
        # normal rank reducers (bx>=2) consume rank_reduce_queue.  The first
        # landing keeps the reducer role explicit and fenced even when the
        # producer ring is empty (e.g. diagnostic init-only launches).
        elif (bx == fx.Int32(1)) & (
            fx.Int32(1 if rank_accumulation_mode == "staged_ring" else 0)
            == fx.Int32(1)
        ) & (reduce_enabled == fx.Int32(1)):
            ring_slots = 8192
            ring_head_addr = rank_stage_ring_head_ptr
            ring_tail_addr = rank_stage_ring_tail_ptr
            ring_claim_addr = rank_stage_ring_claim_ptr
            ring_done_addr = rank_stage_ring_reducer_done_ptr
            ring_prod_addr = rank_stage_ring_producer_done_ptr
            ring_source = global_typed_ptr(
                rank_stage_ring_source_ptr, T.i32, align=4
            )
            ring_slot = global_typed_ptr(rank_stage_ring_slot_ptr, T.i16, align=2)
            ring_tile = global_typed_ptr(rank_stage_ring_tile_ptr, T.i16, align=2)
            ring_seq_ptr = global_typed_ptr(
                rank_stage_ring_sequence_ptr, T.i64, align=8
            )
            ring_payload = global_typed_ptr(
                rank_stage_ring_payload_ptr, T.bf16, align=4
            )
            ring_scratch = global_typed_ptr(
                rank_stage_ring_scratch_ptr, T.bf16, align=2
            )
            ring_seen = global_typed_ptr(
                rank_stage_ring_seen_ptr, T.i32, align=4
            )

            def wait_ring_sequence(seq_addr, expected):
                """Acquire one committed payload sequence (strict equality)."""
                ready = fx.Int64(comm_ops.load_i64_global_system(seq_addr))
                while ready != expected:
                    ready = fx.Int64(comm_ops.load_i64_global_system(seq_addr))
                return ready

            ring_target = grid - fx.Int32(gmm_first)
            ring_active = fx.Int32(1) == fx.Int32(1)
            while ring_active:
                # CTA-wide claim/has-slot broadcast through LDS. readfirstlane
                # is wave-local and cannot safely feed the payload barrier.
                if tx == fx.Int32(0):
                    head = fx.Int64(comm_ops.load_i64_global_system(ring_head_addr))
                    tail = fx.Int64(comm_ops.load_i64_global_system(ring_tail_addr))
                    if tail < head:
                        claimed = fx.Int64(
                            comm_ops.atomic_add_system_acq_rel(
                                ring_claim_addr, fx.Int64(1)
                            )
                        )
                        fx.ptr_store(claimed.to(fx.Int32), work_ptr)
                        fx.ptr_store(fx.Int32(1), work_ptr + fx.Int32(1))
                        fx.ptr_store(fx.Int32(1), work_ptr + fx.Int32(2))
                    else:
                        produced = fx.Int32(comm_ops.load_i32_global_system(ring_prod_addr))
                        fx.ptr_store(fx.Int32(0), work_ptr + fx.Int32(1))
                        fx.ptr_store(
                            (produced < ring_target).select(fx.Int32(1), fx.Int32(0)),
                            work_ptr + fx.Int32(2),
                        )
                gpu.barrier()
                claimed = fx.Int64(fx.ptr_load(work_ptr))
                has_slot = fx.Int32(fx.ptr_load(work_ptr + fx.Int32(1)))
                active_flag = fx.Int32(fx.ptr_load(work_ptr + fx.Int32(2)))
                ring_active = active_flag == fx.Int32(1)
                if has_slot == fx.Int32(1):
                    slot_index = claimed & fx.Int64(ring_slots - 1)
                    seq_addr = rank_stage_ring_sequence_ptr + slot_index * fx.Int64(8)
                    wait_ring_sequence(seq_addr, claimed + fx.Int64(1))
                    source = fx.Int32(fx.ptr_load(ring_source + slot_index))
                    slot = fx.Int32(fx.ptr_load(ring_slot + slot_index))
                    tile = fx.Int32(fx.ptr_load(ring_tile + slot_index))
                    seen_index = ((source * fx.Int32(hidden_tiles // 2) + tile // fx.Int32(2)) * fx.Int32(2) + tile % fx.Int32(2))
                    if tx == fx.Int32(0):
                        comm_ops.atomic_add_system_acq_rel(
                            rank_stage_ring_seen_ptr + fx.Int64(seen_index) * fx.Int64(4),
                            fx.Int32(1),
                        )
                    if tx < fx.Int32(BN):
                        payload_index = slot_index * fx.Int64(BN) + fx.Int64(tx)
                        value = fx.Float32(fx.ptr_load(ring_payload + payload_index))
                        group = tile // fx.Int32(2)
                        sub_tile = tile - group * fx.Int32(2)
                        scratch_index = ((source * fx.Int32(hidden_tiles // 2) + group) * fx.Int32(2 * BN) + sub_tile * fx.Int32(BN) + tx)
                        old = fx.Float32(fx.ptr_load(ring_scratch + scratch_index))
                        fx.ptr_store(
                            fx.BFloat16(value + old), ring_scratch + scratch_index
                        )
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        comm_ops.atomic_add_system_acq_rel(ring_tail_addr, fx.Int64(1))
            gpu.barrier()
            # Materialize the complete source partial after EOS.  One CTA owns
            # this pass, so scratch accumulation is race-free and no output
            # element atomic is required.
            scratch_items = fx.Int32(SOURCE_CAPACITY * HIDDEN)
            rank_acc = global_typed_ptr(rank_accumulator_ptr, T.bf16, align=4)
            for item in range(tx, scratch_items, fx.Int32(THREADS)):
                source = item // fx.Int32(HIDDEN)
                col = item - source * fx.Int32(HIDDEN)
                tile = col // fx.Int32(BN)
                group = tile // fx.Int32(2)
                sub_tile = tile - group * fx.Int32(2)
                tile_col = col - tile * fx.Int32(BN)
                scratch_index = (
                    (source * fx.Int32(hidden_tiles // 2) + group)
                    * fx.Int32(2 * BN)
                    + sub_tile * fx.Int32(BN)
                    + tile_col
                )
                fx.ptr_store(
                    fx.BFloat16(fx.ptr_load(ring_scratch + scratch_index)),
                    rank_acc + item,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                for source in range_constexpr(SOURCE_CAPACITY):
                    source_has_routes = fx.Int32(0)
                    for group in range_constexpr(hidden_tiles // 2):
                        source_has_routes = source_has_routes + fx.Int32(
                            fx.ptr_load(
                                ring_seen
                                + fx.Int64(
                                    (source * (hidden_tiles // 2) + group) * 2
                                )
                            )
                        )
                        source_has_routes = source_has_routes + fx.Int32(
                            fx.ptr_load(
                                ring_seen
                                + fx.Int64(
                                    (source * (hidden_tiles // 2) + group) * 2
                                    + 1
                                )
                            )
                        )
                    if source_has_routes > fx.Int32(0):
                        comm_ops.store_i64_global_system(
                            rank_ready_ptr + fx.Int64(source) * fx.Int64(8),
                            generation,
                        )
                    source_rank = fx.Int32(source) >> fx.Int32(7)
                    source_plane = source_rank >> fx.Int32(3)
                    source_local = source_rank & fx.Int32(7)
                    token = fx.Int32(source) & fx.Int32(127)
                    peer_s2 = fx.Int64(
                        fx.ptr_load(
                            lds_typed_ptr(
                                lds_base
                                + fx.Int32(peer_table_off)
                                + source_local * fx.Int32(8),
                                T.i64,
                                align=8,
                            )
                        )
                    )
                    token_index = source_plane * fx.Int32(MAX_TOKENS) + token
                    rank_mask = fx.Int32(
                        comm_ops.load_i32_global_system(
                            peer_s2
                            + fx.Int64(dest_rank_mask_off)
                            + parity * fx.Int64(dest_rank_mask_stride)
                            + fx.Int64(token_index) * fx.Int64(4)
                        )
                    ) & fx.Int32(0xFF)
                    expected_ranks = fx.Int32(0)
                    for source_peer in range_constexpr(GPUS_PER_NODE):
                        expected_ranks = expected_ranks + (
                            (rank_mask >> fx.Int32(source_peer)) & fx.Int32(1)
                        )
                    old_rank = fx.Int32(0)
                    if source_has_routes > fx.Int32(0):
                        old_rank = fx.Int32(
                            comm_ops.atomic_add_system_acq_rel(
                                peer_s2
                                + fx.Int64(partial_done_off)
                                + parity * fx.Int64(partial_done_stride)
                                + fx.Int64(token_index) * fx.Int64(64),
                                fx.Int32(1),
                            )
                        )
                    if (source_has_routes > fx.Int32(0)) & (
                        expected_ranks > fx.Int32(0)
                    ) & (old_rank + fx.Int32(1) == expected_ranks):
                        queue_slot = fx.Int32(
                            comm_ops.atomic_add_system_acq_rel(
                                peer_s2
                                + fx.Int64(rank_reduce_queue_tail_off)
                                + parity * fx.Int64(rank_reduce_queue_tail_stride),
                                fx.Int32(1),
                            )
                        )
                        comm_ops.store_i32_system(
                            peer_s2
                            + fx.Int64(rank_reduce_queue_off)
                            + parity * fx.Int64(rank_reduce_queue_stride),
                            queue_slot,
                            token_index,
                        )
                        comm_ops.store_i64_global_system(
                            peer_s2
                            + fx.Int64(rank_reduce_queue_ready_off)
                            + parity * fx.Int64(rank_reduce_queue_ready_stride)
                            + fx.Int64(queue_slot) * fx.Int64(8),
                            generation,
                        )
                comm_ops.store_i32_system(
                    ring_done_addr, fx.Int32(0), fx.Int32(1)
                )

        # ---- Rank-local partial pull/reduction on each source proxy ----
        elif (bx >= fx.Int32(reduce_first)) & (
            bx < fx.Int32(combine_first)
        ) & (reduce_enabled == fx.Int32(1)) & (
            fx.Int32(1 if node_accumulation_mode == "rank_local" else 0)
            == fx.Int32(1)
        ):
            reducer = bx - fx.Int32(reduce_first)
            rank_reduce_acc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                accumulator_ptr,
                num_records_bytes=accumulator_stride,
            )
            rank_reduce_tx_rsrc = buffer_ops.create_buffer_resource_from_addr(
                tx_ptr,
                num_records_bytes=tx_stride,
            )
            rank_mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                dest_rank_mask_ptr,
                num_records_bytes=dest_rank_mask_stride,
            )
            rank_tx_slot_rsrc = buffer_ops.create_buffer_resource_from_addr(
                rank_tx_slot_ptr,
                num_records_bytes=rank_tx_slot_stride,
            )
            reduce_work_items = load_rank_reduce_work_items()
            if const_expr(node_reduce_work_schedule == "static_strided"):
                for queue_slot in range(
                    reducer,
                    reduce_work_items,
                    fx.Int32(reduce_blocks),
                ):
                    reduce_rank_queue_slot(
                        queue_slot,
                        rank_reduce_acc_rsrc,
                        rank_reduce_tx_rsrc,
                        rank_mask_rsrc,
                        rank_tx_slot_rsrc,
                    )
            else:
                consume_dynamic_rank_reduce(
                    reduce_work_items,
                    rank_reduce_acc_rsrc,
                    rank_reduce_tx_rsrc,
                    rank_mask_rsrc,
                    rank_tx_slot_rsrc,
                    False,
                )

        # ---- Group-pipelined source-aligned reduction for route-store mode ----
        elif (bx >= fx.Int32(reduce_first)) & (
            bx < fx.Int32(combine_first)
        ) & (reduce_enabled == fx.Int32(1)) & (
            fx.Int32(
                1
                if node_accumulation_mode == "route_store"
                and node_reduce_schedule == "group"
                else 0
            )
            == fx.Int32(1)
        ):
            reducer = bx - fx.Int32(reduce_first)
            group_reduce_route_rsrc = buffer_ops.create_buffer_resource_from_addr(
                route_slots_ptr,
                num_records_bytes=route_slots_stride,
            )
            group_reduce_acc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                accumulator_ptr,
                num_records_bytes=accumulator_stride,
            )
            reduce_elems = node_reduce_vec_bytes // 2
            reduce_i32s = node_reduce_vec_bytes // 4
            for reduce_work in range(
                reducer,
                fx.Int32(2 * MAX_TOKENS),
                fx.Int32(reduce_blocks),
            ):
                token = reduce_work // fx.Int32(2)
                source_plane = reduce_work - token * fx.Int32(2)
                token_index = source_plane * fx.Int32(MAX_TOKENS) + token

                if tx == fx.Int32(0):
                    expected = buffer_ops.buffer_load(
                        expected_rsrc,
                        token_index * fx.Int32(hidden_tiles),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    has_routes = expected > fx.Int32(0)
                    record_base = (
                        source_plane == fx.Int32(node)
                    ).select(local_dispatch_records, remote_dispatch_records)
                    record_addr = (
                        fx.Int64(record_base)
                        + fx.Int64(token) * fx.Int64(dispatch_record_bytes)
                        + fx.Int64(rank_slot_masks_offset)
                        + fx.Int64(node * GPUS_PER_NODE * 2)
                    )
                    group_reduce_record_rsrc = (
                        buffer_ops.create_buffer_resource_from_addr(
                            record_addr,
                            num_records_bytes=GPUS_PER_NODE * 2,
                        )
                    )
                    slot_mask = fx.Int32(0)
                    if const_expr(diagnostic_mode != "atomic_only"):
                        for mask_pair in range_constexpr(GPUS_PER_NODE // 2):
                            packed_masks = fx.Int32(
                                buffer_ops.buffer_load(
                                    group_reduce_record_rsrc,
                                    fx.Int32(mask_pair),
                                    vec_width=1,
                                    dtype=T.i32,
                                    mask=has_routes,
                                )
                            )
                            slot_mask = slot_mask | (
                                packed_masks & fx.Int32(0xFFFF)
                            )
                            slot_mask = slot_mask | (
                                (packed_masks >> fx.Int32(16))
                                & fx.Int32(0xFFFF)
                            )
                    fx.ptr_store(slot_mask, work_ptr)
                    fx.ptr_store(expected, work_ptr + fx.Int32(1))
                gpu.barrier()
                slot_mask = fx.Int32(fx.ptr_load(work_ptr))
                expected = fx.Int32(fx.ptr_load(work_ptr + fx.Int32(1)))

                for n_group in range_constexpr(hidden_tiles // 2):
                    n_block0 = fx.Int32(n_group * 2)
                    n_block1 = n_block0 + fx.Int32(1)
                    if tx == fx.Int32(0):
                        done0_index = (
                            token_index * fx.Int32(hidden_tiles) + n_block0
                        )
                        done1_index = done0_index + fx.Int32(1)
                        completed0 = fx.Int32(
                            comm_ops.load_i32_global_system(
                                done_ptr + fx.Int64(done0_index) * fx.Int64(4)
                            )
                        )
                        while completed0 < expected:
                            completed0 = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    done_ptr
                                    + fx.Int64(done0_index) * fx.Int64(4)
                                )
                            )
                        completed1 = fx.Int32(
                            comm_ops.load_i32_global_system(
                                done_ptr + fx.Int64(done1_index) * fx.Int64(4)
                            )
                        )
                        while completed1 < expected:
                            completed1 = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    done_ptr
                                    + fx.Int64(done1_index) * fx.Int64(4)
                                )
                            )
                        if (completed0 != expected) | (completed1 != expected):
                            add_error()
                    gpu.barrier()
                    comm_ops.fence_system_acquire()

                    if const_expr(node_reduce_vec_bytes == 4):
                        reduce_n_block = n_block0 + wave // fx.Int32(2)
                        col_in_tile = (
                            (wave % fx.Int32(2)) * fx.Int32(BN // 2)
                            + lane * fx.Int32(2)
                        )
                        accum_values = [fx.Float32(0.0), fx.Float32(0.0)]
                        for slot in range_constexpr(TOPK):
                            slot_active = (
                                slot_mask & (fx.Int32(1) << fx.Int32(slot))
                            ) != fx.Int32(0)
                            route_bf16_index = (
                                (
                                    (
                                        token_index * fx.Int32(hidden_tiles)
                                        + reduce_n_block
                                    )
                                    * fx.Int32(TOPK)
                                    + fx.Int32(slot)
                                )
                                * fx.Int32(BN)
                                + col_in_tile
                            )
                            packed = buffer_ops.buffer_load(
                                group_reduce_route_rsrc,
                                route_bf16_index // fx.Int32(2),
                                vec_width=1,
                                dtype=T.i32,
                                mask=slot_active,
                            )
                            values = fx.Vector.from_elements(
                                [packed], fx.Int32
                            ).bitcast(BFloat16).to(Float32)
                            accum_values[0] = accum_values[0] + fx.Float32(
                                values[0]
                            )
                            accum_values[1] = accum_values[1] + fx.Float32(
                                values[1]
                            )
                        output_packed = (
                            fx.Vector.from_elements(accum_values, Float32)
                            .to(BFloat16)
                            .bitcast(fx.Int32)[0]
                        )
                        output_bf16_index = (
                            token_index * fx.Int32(HIDDEN)
                            + reduce_n_block * fx.Int32(BN)
                            + col_in_tile
                        )
                        buffer_ops.buffer_store(
                            output_packed,
                            group_reduce_acc_rsrc,
                            output_bf16_index // fx.Int32(2),
                        )
                    else:
                        active_wave = wave < fx.Int32(2)
                        safe_wave = active_wave.select(wave, fx.Int32(0))
                        reduce_n_block = n_block0 + safe_wave
                        col_in_tile = lane * fx.Int32(4)
                        accum_values = [
                            fx.Float32(0.0)
                            for _ in range_constexpr(4)
                        ]
                        for slot in range_constexpr(TOPK):
                            slot_active = active_wave & (
                                (
                                    slot_mask
                                    & (fx.Int32(1) << fx.Int32(slot))
                                )
                                != fx.Int32(0)
                            )
                            route_bf16_index = (
                                (
                                    (
                                        token_index * fx.Int32(hidden_tiles)
                                        + reduce_n_block
                                    )
                                    * fx.Int32(TOPK)
                                    + fx.Int32(slot)
                                )
                                * fx.Int32(BN)
                                + col_in_tile
                            )
                            packed = buffer_ops.buffer_load(
                                group_reduce_route_rsrc,
                                route_bf16_index // fx.Int32(2),
                                vec_width=2,
                                dtype=T.i32,
                                mask=slot_active,
                            )
                            values = fx.Vector(packed).bitcast(BFloat16).to(
                                Float32
                            )
                            for value_index in range_constexpr(4):
                                accum_values[value_index] = (
                                    accum_values[value_index]
                                    + fx.Float32(values[value_index])
                                )
                        output_packed = (
                            fx.Vector.from_elements(accum_values, Float32)
                            .to(BFloat16)
                            .bitcast(fx.Int32)
                        )
                        output_bf16_index = (
                            token_index * fx.Int32(HIDDEN)
                            + reduce_n_block * fx.Int32(BN)
                            + col_in_tile
                        )
                        buffer_ops.buffer_store(
                            output_packed,
                            group_reduce_acc_rsrc,
                            output_bf16_index // fx.Int32(2),
                            mask=active_wave,
                        )
                    rocdl.s_waitcnt(0)
                    gpu.barrier()

                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        partial_ready_ptr
                        + fx.Int64(token_index) * fx.Int64(8),
                        generation,
                    )
                gpu.barrier()

        # ---- Tile-granular source-aligned reduction for route-store mode ----
        elif (bx >= fx.Int32(reduce_first)) & (
            bx < fx.Int32(combine_first)
        ) & (reduce_enabled == fx.Int32(1)) & (
            fx.Int32(
                1
                if node_accumulation_mode == "route_store"
                and node_reduce_schedule == "tile"
                else 0
            )
            == fx.Int32(1)
        ):
            reducer = bx - fx.Int32(reduce_first)
            tile_reduce_route_rsrc = buffer_ops.create_buffer_resource_from_addr(
                route_slots_ptr,
                num_records_bytes=route_slots_stride,
            )
            tile_reduce_acc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                accumulator_ptr,
                num_records_bytes=accumulator_stride,
            )
            reduce_elems = node_reduce_vec_bytes // 2
            reduce_i32s = node_reduce_vec_bytes // 4
            col_iterations = BN // (64 * reduce_elems)
            for tile_work in range(
                reducer,
                fx.Int32(2 * hidden_tiles),
                fx.Int32(reduce_blocks),
            ):
                source_plane = tile_work // fx.Int32(hidden_tiles)
                n_block = tile_work - source_plane * fx.Int32(hidden_tiles)
                for token_base in range(
                    fx.Int32(0), fx.Int32(MAX_TOKENS), fx.Int32(4)
                ):
                    token = token_base + wave
                    token_index = source_plane * fx.Int32(MAX_TOKENS) + token
                    expected_index = (
                        token_index * fx.Int32(hidden_tiles) + n_block
                    )
                    expected = buffer_ops.buffer_load(
                        expected_rsrc,
                        expected_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    if lane == fx.Int32(0):
                        completed = fx.Int32(
                            comm_ops.load_i32_global_system(
                                done_ptr
                                + fx.Int64(expected_index) * fx.Int64(4)
                            )
                        )
                        while completed < expected:
                            completed = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    done_ptr
                                    + fx.Int64(expected_index) * fx.Int64(4)
                                )
                            )
                        if completed != expected:
                            add_error()
                    comm_ops.fence_system_acquire()

                    has_routes = expected > fx.Int32(0)
                    record_base = (
                        source_plane == fx.Int32(node)
                    ).select(local_dispatch_records, remote_dispatch_records)
                    record_addr = (
                        fx.Int64(record_base)
                        + fx.Int64(token) * fx.Int64(dispatch_record_bytes)
                        + fx.Int64(rank_slot_masks_offset)
                        + fx.Int64(node * GPUS_PER_NODE * 2)
                    )
                    tile_reduce_record_rsrc = buffer_ops.create_buffer_resource_from_addr(
                        record_addr,
                        num_records_bytes=GPUS_PER_NODE * 2,
                    )
                    slot_mask = fx.Int32(0)
                    for mask_pair in range_constexpr(GPUS_PER_NODE // 2):
                        packed_masks = fx.Int32(
                            buffer_ops.buffer_load(
                                tile_reduce_record_rsrc,
                                fx.Int32(mask_pair),
                                vec_width=1,
                                dtype=T.i32,
                                mask=has_routes,
                            )
                        )
                        slot_mask = slot_mask | (
                            packed_masks & fx.Int32(0xFFFF)
                        )
                        slot_mask = slot_mask | (
                            (packed_masks >> fx.Int32(16))
                            & fx.Int32(0xFFFF)
                        )

                    for col_iter in range_constexpr(col_iterations):
                        col_in_tile = (
                            lane * fx.Int32(reduce_elems)
                            + fx.Int32(col_iter * 64 * reduce_elems)
                        )
                        accum_values = [
                            fx.Float32(0.0)
                            for _ in range_constexpr(reduce_elems)
                        ]
                        for slot in range_constexpr(TOPK):
                            slot_active = (
                                slot_mask & (fx.Int32(1) << fx.Int32(slot))
                            ) != fx.Int32(0)
                            route_bf16_index = (
                                (
                                    (
                                        token_index * fx.Int32(hidden_tiles)
                                        + n_block
                                    )
                                    * fx.Int32(TOPK)
                                    + fx.Int32(slot)
                                )
                                * fx.Int32(BN)
                                + col_in_tile
                            )
                            packed = buffer_ops.buffer_load(
                                tile_reduce_route_rsrc,
                                route_bf16_index // fx.Int32(2),
                                vec_width=reduce_i32s,
                                dtype=T.i32,
                                mask=slot_active,
                            )
                            if const_expr(reduce_i32s == 1):
                                packed_vec = fx.Vector.from_elements(
                                    [packed], fx.Int32
                                )
                            else:
                                packed_vec = fx.Vector(packed)
                            values = packed_vec.bitcast(BFloat16).to(Float32)
                            for value_index in range_constexpr(reduce_elems):
                                accum_values[value_index] = (
                                    accum_values[value_index]
                                    + fx.Float32(values[value_index])
                                )
                        output_packed = (
                            fx.Vector.from_elements(accum_values, Float32)
                            .to(BFloat16)
                            .bitcast(fx.Int32)
                        )
                        output_bf16_index = (
                            token_index * fx.Int32(HIDDEN)
                            + n_block * fx.Int32(BN)
                            + col_in_tile
                        )
                        if const_expr(reduce_i32s == 1):
                            buffer_ops.buffer_store(
                                output_packed[0],
                                tile_reduce_acc_rsrc,
                                output_bf16_index // fx.Int32(2),
                            )
                        else:
                            buffer_ops.buffer_store(
                                output_packed,
                                tile_reduce_acc_rsrc,
                                output_bf16_index // fx.Int32(2),
                            )
                    rocdl.s_waitcnt(0)
                    if lane == fx.Int32(0):
                        comm_ops.fence_system_release()
                        old_partial = fx.Int32(
                            comm_ops.atomic_add_system_acq_rel(
                                partial_done_ptr
                                + fx.Int64(token_index) * fx.Int64(64),
                                fx.Int32(1),
                            )
                        )
                        if old_partial >= fx.Int32(hidden_tiles):
                            add_error()
                        if old_partial + fx.Int32(1) == fx.Int32(hidden_tiles):
                            comm_ops.store_i64_global_system(
                                partial_ready_ptr
                                + fx.Int64(token_index) * fx.Int64(8),
                                generation,
                            )

        # -------- Token-granular node reduction for route-store mode --------
        elif (bx >= fx.Int32(reduce_first)) & (
            bx < fx.Int32(combine_first)
        ) & (reduce_enabled == fx.Int32(1)) & (
            fx.Int32(1 if node_accumulation_mode == "route_store" else 0)
            == fx.Int32(1)
        ):
            reducer = bx - fx.Int32(reduce_first)
            token_reduce_route_rsrc = buffer_ops.create_buffer_resource_from_addr(
                route_slots_ptr,
                num_records_bytes=route_slots_stride,
            )
            token_reduce_acc_rsrc = buffer_ops.create_buffer_resource_from_addr(
                accumulator_ptr,
                num_records_bytes=accumulator_stride,
            )
            reduce_elems = node_reduce_vec_bytes // 2
            reduce_i32s = node_reduce_vec_bytes // 4
            tiles_per_wave = hidden_tiles // 4
            col_iterations = BN // (64 * reduce_elems)
            for reduce_work in range(
                reducer,
                fx.Int32(2 * MAX_TOKENS),
                fx.Int32(reduce_blocks),
            ):
                token = reduce_work // fx.Int32(2)
                source_plane = reduce_work - token * fx.Int32(2)
                token_index = source_plane * fx.Int32(MAX_TOKENS) + token
                if tx == fx.Int32(0):
                    wait_ready(
                        token_ready_ptr
                        + fx.Int64(token_index) * fx.Int64(8),
                        generation,
                    )
                gpu.barrier()
                comm_ops.fence_system_acquire()

                if tx == fx.Int32(0):
                    expected = buffer_ops.buffer_load(
                        expected_rsrc,
                        token_index * fx.Int32(hidden_tiles),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    has_routes = expected > fx.Int32(0)
                    slot_mask = fx.Int32(0)
                    if const_expr(diagnostic_mode != "atomic_only"):
                        record_base = (
                            source_plane == fx.Int32(node)
                        ).select(local_dispatch_records, remote_dispatch_records)
                        record_addr = (
                            fx.Int64(record_base)
                            + fx.Int64(token) * fx.Int64(dispatch_record_bytes)
                            + fx.Int64(rank_slot_masks_offset)
                            + fx.Int64(node * GPUS_PER_NODE * 2)
                        )
                        token_reduce_record_rsrc = buffer_ops.create_buffer_resource_from_addr(
                            record_addr,
                            num_records_bytes=GPUS_PER_NODE * 2,
                        )
                        for mask_pair in range_constexpr(GPUS_PER_NODE // 2):
                            packed_masks = fx.Int32(
                                buffer_ops.buffer_load(
                                    token_reduce_record_rsrc,
                                    fx.Int32(mask_pair),
                                    vec_width=1,
                                    dtype=T.i32,
                                    mask=has_routes,
                                )
                            )
                            slot_mask = slot_mask | (
                                packed_masks & fx.Int32(0xFFFF)
                            )
                            slot_mask = slot_mask | (
                                (packed_masks >> fx.Int32(16))
                                & fx.Int32(0xFFFF)
                            )
                    fx.ptr_store(slot_mask, work_ptr)
                gpu.barrier()
                slot_mask = fx.Int32(fx.ptr_load(work_ptr))

                if slot_mask != fx.Int32(0):
                    # Four waves split the 28 hidden tiles into seven
                    # contiguous tiles each. Every lane accumulates either 2
                    # or 4 BF16 columns in FP32 registers.
                    for tile_iter in range_constexpr(tiles_per_wave):
                        n_block = wave * fx.Int32(tiles_per_wave) + fx.Int32(
                            tile_iter
                        )
                        for col_iter in range_constexpr(col_iterations):
                            col_in_tile = (
                                lane * fx.Int32(reduce_elems)
                                + fx.Int32(col_iter * 64 * reduce_elems)
                            )
                            accum_values = [
                                fx.Float32(0.0)
                                for _ in range_constexpr(reduce_elems)
                            ]
                            if const_expr(
                                node_reduce_load_schedule == "load_first"
                            ):
                                packed_slots = []
                                for slot in range_constexpr(TOPK):
                                    slot_active = (
                                        slot_mask
                                        & (fx.Int32(1) << fx.Int32(slot))
                                    ) != fx.Int32(0)
                                    route_bf16_index = (
                                        (
                                            (
                                                token_index
                                                * fx.Int32(hidden_tiles)
                                                + n_block
                                            )
                                            * fx.Int32(TOPK)
                                            + fx.Int32(slot)
                                        )
                                        * fx.Int32(BN)
                                        + col_in_tile
                                    )
                                    packed = buffer_ops.buffer_load(
                                        token_reduce_route_rsrc,
                                        route_bf16_index // fx.Int32(2),
                                        vec_width=reduce_i32s,
                                        dtype=T.i32,
                                        mask=slot_active,
                                    )
                                    if const_expr(reduce_i32s == 1):
                                        packed_slots.append(
                                            fx.Vector.from_elements(
                                                [packed], fx.Int32
                                            )
                                        )
                                    else:
                                        packed_slots.append(fx.Vector(packed))
                                for slot in range_constexpr(TOPK):
                                    values = packed_slots[slot].bitcast(
                                        BFloat16
                                    ).to(Float32)
                                    for value_index in range_constexpr(
                                        reduce_elems
                                    ):
                                        accum_values[value_index] = (
                                            accum_values[value_index]
                                            + fx.Float32(values[value_index])
                                        )
                            else:
                                for slot in range_constexpr(TOPK):
                                    slot_active = (
                                        slot_mask
                                        & (fx.Int32(1) << fx.Int32(slot))
                                    ) != fx.Int32(0)
                                    route_bf16_index = (
                                        (
                                            (
                                                token_index
                                                * fx.Int32(hidden_tiles)
                                                + n_block
                                            )
                                            * fx.Int32(TOPK)
                                            + fx.Int32(slot)
                                        )
                                        * fx.Int32(BN)
                                        + col_in_tile
                                    )
                                    packed = buffer_ops.buffer_load(
                                        token_reduce_route_rsrc,
                                        route_bf16_index // fx.Int32(2),
                                        vec_width=reduce_i32s,
                                        dtype=T.i32,
                                        mask=slot_active,
                                    )
                                    if const_expr(reduce_i32s == 1):
                                        packed_vec = fx.Vector.from_elements(
                                            [packed], fx.Int32
                                        )
                                    else:
                                        packed_vec = fx.Vector(packed)
                                    values = packed_vec.bitcast(BFloat16).to(
                                        Float32
                                    )
                                    for value_index in range_constexpr(
                                        reduce_elems
                                    ):
                                        accum_values[value_index] = (
                                            accum_values[value_index]
                                            + fx.Float32(values[value_index])
                                        )
                            output_packed = (
                                fx.Vector.from_elements(accum_values, Float32)
                                .to(BFloat16)
                                .bitcast(fx.Int32)
                            )
                            output_bf16_index = (
                                token_index * fx.Int32(HIDDEN)
                                + n_block * fx.Int32(BN)
                                + col_in_tile
                            )
                            if const_expr(reduce_i32s == 1):
                                buffer_ops.buffer_store(
                                    output_packed[0],
                                    token_reduce_acc_rsrc,
                                    output_bf16_index // fx.Int32(2),
                                )
                            else:
                                buffer_ops.buffer_store(
                                    output_packed,
                                    token_reduce_acc_rsrc,
                                    output_bf16_index // fx.Int32(2),
                                )
                else:
                    zero_packed = (
                        fx.Vector.filled(reduce_elems, 0.0, Float32)
                        .to(BFloat16)
                        .bitcast(fx.Int32)
                    )
                    for tile_iter in range_constexpr(tiles_per_wave):
                        n_block = wave * fx.Int32(tiles_per_wave) + fx.Int32(
                            tile_iter
                        )
                        for col_iter in range_constexpr(col_iterations):
                            col_in_tile = (
                                lane * fx.Int32(reduce_elems)
                                + fx.Int32(col_iter * 64 * reduce_elems)
                            )
                            output_bf16_index = (
                                token_index * fx.Int32(HIDDEN)
                                + n_block * fx.Int32(BN)
                                + col_in_tile
                            )
                            if const_expr(reduce_i32s == 1):
                                buffer_ops.buffer_store(
                                    zero_packed[0],
                                    token_reduce_acc_rsrc,
                                    output_bf16_index // fx.Int32(2),
                                )
                            else:
                                buffer_ops.buffer_store(
                                    zero_packed,
                                    token_reduce_acc_rsrc,
                                    output_bf16_index // fx.Int32(2),
                                )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        partial_ready_ptr
                        + fx.Int64(token_index) * fx.Int64(8),
                        generation,
                    )
                gpu.barrier()

        # ---------------- Source final-combine roles ----------------
        elif (bx >= fx.Int32(combine_first)) & (
            bx < fx.Int32(combine_first + final_combine_blocks)
        ) & (role_enabled == fx.Int32(1)):
            if const_expr(accumulator_dtype == "bf16"):
                local_acc_bf16 = global_typed_ptr(
                    accumulator_ptr, T.bf16, align=2
                )
            else:
                local_acc_f32 = global_typed_ptr(
                    accumulator_ptr, T.f32, align=4
                )
            remote_in = global_typed_ptr(rx_ptr, T.bf16, align=2)
            output = global_typed_ptr(arg_output_bf16, T.bf16, align=2)
            if const_expr(node_accumulation_mode == "rank_local"):
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
                final_mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    dest_rank_mask_ptr,
                    num_records_bytes=dest_rank_mask_stride,
                )
                final_rx_slot_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    rank_rx_slot_ptr,
                    num_records_bytes=rank_rx_slot_stride,
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
                        work
                        if const_expr(node_accumulation_mode == "rank_local")
                        else work // fx.Int32(hidden_tiles)
                    )
                    if tx == fx.Int32(0):
                        if const_expr(rail_return_schedule == "compact"):
                            local_mask = fx.Int32(
                                buffer_ops.buffer_load(
                                    final_mask_rsrc,
                                    fx.Int32(local_plane * MAX_TOKENS)
                                    + token,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            ) & fx.Int32(0xFF)
                            rx_slot = fx.Int32(
                                buffer_ops.buffer_load(
                                    final_rx_slot_rsrc,
                                    token,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            local_active = local_mask != fx.Int32(0)
                            remote_active = rx_slot >= fx.Int32(0)
                            fx.ptr_store(local_active.select(fx.Int32(1), fx.Int32(0)), work_ptr + fx.Int32(1))
                            fx.ptr_store(remote_active.select(fx.Int32(1), fx.Int32(0)), work_ptr + fx.Int32(2))
                            fx.ptr_store(rx_slot, work_ptr + fx.Int32(3))
                            if local_active:
                                wait_ready(
                                    partial_ready_ptr
                                    + fx.Int64(local_plane * MAX_TOKENS)
                                    * fx.Int64(8)
                                    + fx.Int64(token) * fx.Int64(8),
                                    generation,
                                )
                            if remote_active:
                                group = rx_slot // fx.Int32(return_chunk_tokens)
                                wait_ready(
                                    return_ready_ptr
                                    + fx.Int64(group) * fx.Int64(8),
                                    generation,
                                )
                        else:
                            token_ready_index = (
                                fx.Int32(local_plane * MAX_TOKENS) + token
                            )
                            wait_ready(
                                rail_payload_ready_ptr
                                + fx.Int64(token_ready_index) * fx.Int64(8),
                                generation,
                            )
                            group = token // fx.Int32(return_chunk_tokens)
                            wait_ready(
                                return_ready_ptr
                                + fx.Int64(group) * fx.Int64(8),
                                generation,
                            )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    if const_expr(node_accumulation_mode == "rank_local"):
                        if const_expr(rail_return_schedule == "compact"):
                            local_active = fx.Int32(
                                fx.ptr_load(work_ptr + fx.Int32(1))
                            ) != fx.Int32(0)
                            remote_active = fx.Int32(
                                fx.ptr_load(work_ptr + fx.Int32(2))
                            ) != fx.Int32(0)
                            rx_slot = fx.Int32(
                                fx.ptr_load(work_ptr + fx.Int32(3))
                            )
                        else:
                            local_active = fx.Int32(1) == fx.Int32(1)
                            remote_active = fx.Int32(1) == fx.Int32(1)
                            rx_slot = token
                        # Four waves cover the H7168 row as fourteen 512-value
                        # chunks. Every active lane moves eight BF16 values in
                        # one 16-byte operation.
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
                                remote_active.select(rx_slot, fx.Int32(0))
                                * fx.Int32(HIDDEN)
                                + col
                                if const_expr(rail_return_schedule == "compact")
                                else token_index
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
                    else:
                        ntile = work - token * fx.Int32(hidden_tiles)
                        col = ntile * fx.Int32(BN) + tx
                        local_index = (
                            (fx.Int32(local_plane * MAX_TOKENS) + token)
                            * fx.Int32(HIDDEN)
                            + col
                        )
                        token_index = token * fx.Int32(HIDDEN) + col
                        if const_expr(accumulator_dtype == "bf16"):
                            local_bf16 = fx.ptr_load(
                                local_acc_bf16 + fx.Int64(local_index)
                            )
                        else:
                            local_f32 = fx.ptr_load(
                                local_acc_f32 + fx.Int64(local_index)
                            )
                            local_bf16 = fx.BFloat16(local_f32)
                        remote_bf16 = fx.ptr_load(
                            remote_in + fx.Int64(token_index)
                        )
                        result = fx.Float32(local_bf16) + fx.Float32(
                            remote_bf16
                        )
                        fx.ptr_store(
                            fx.BFloat16(result),
                            output + fx.Int64(token_index),
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
        elif (bx >= fx.Int32(gmm_first)) & (compute_enabled == fx.Int32(1)):
            num_valid = fx.Int32(global_typed_ptr(arg_nvalid, T.i32)[0])
            total_m_blocks = num_valid // fx.Int32(BM)
            n_groups = hidden_tiles // n_tile_group
            total_work = total_m_blocks * fx.Int32(n_groups)
            shard = bx & fx.Int32(WORK_SHARDS - 1)
            active = fx.Int32(1) == fx.Int32(1)

            def publish_node_tile(m_row, n_block):
                meta = buffer_ops.create_buffer_resource_from_addr(arg_stids)
                score_active = (
                    lane < fx.Int32(BM // 4)
                    if scoreboard_schedule == "four_wave"
                    else tx < fx.Int32(BM)
                )
                score_row = (
                    wave * fx.Int32(BM // 4) + lane
                    if scoreboard_schedule == "four_wave"
                    else tx
                )
                if score_active:
                    if const_expr(epilogue_schedule != "lane32"):
                        packed = fx.Int32(
                            fx.ptr_load(
                                lds_typed_ptr(
                                    lds_base
                                    + fx.Int32(epilogue_meta_off)
                                    + score_row * fx.Int32(4),
                                    T.i32,
                                    align=4,
                                )
                            )
                        )
                    else:
                        packed = buffer_ops.buffer_load(
                            meta, m_row + score_row, vec_width=1, dtype=T.i32
                        )
                    source = packed & fx.Int32(0x00FFFFFF)
                    if source < fx.Int32(SOURCE_CAPACITY):
                        source_rank = source >> fx.Int32(7)
                        source_node = source_rank >> fx.Int32(3)
                        source_local = source_rank & fx.Int32(7)
                        token = source & fx.Int32(127)
                        peer_base = fx.Int64(
                            fx.ptr_load(
                                lds_typed_ptr(
                                    lds_base
                                    + fx.Int32(peer_table_off)
                                    + source_local * fx.Int32(8),
                                    T.i64,
                                    align=8,
                                )
                            )
                        )
                        score_index = (
                            (source_node * fx.Int32(MAX_TOKENS) + token)
                            * fx.Int32(hidden_tiles)
                            + n_block
                        )
                        if const_expr(
                            epilogue_schedule == "lane32_meta_expected"
                        ):
                            expected = fx.Int32(
                                fx.ptr_load(
                                    lds_typed_ptr(
                                        lds_base
                                        + fx.Int32(epilogue_expected_off)
                                        + score_row * fx.Int32(4),
                                        T.i32,
                                        align=4,
                                    )
                                )
                            )
                        else:
                            expected = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    peer_base
                                    + fx.Int64(expected_off)
                                    + parity * fx.Int64(expected_stride)
                                    + fx.Int64(score_index) * fx.Int64(4)
                                )
                            )
                        valid_expected = (expected > fx.Int32(0)) & (
                            expected <= fx.Int32(TOPK)
                        )
                        if not valid_expected:
                            add_error()
                        safe_expected = valid_expected.select(expected, fx.Int32(1))
                        old = fx.Int32(
                            comm_ops.atomic_add_system_acq_rel(
                                peer_base
                                + fx.Int64(done_off)
                                + parity * fx.Int64(done_stride)
                                + fx.Int64(score_index) * fx.Int64(4),
                                fx.Int32(1),
                            )
                        )
                        if old + fx.Int32(1) == safe_expected:
                            token_index = (
                                source_node * fx.Int32(MAX_TOKENS) + token
                            )
                            old_token = fx.Int32(
                                comm_ops.atomic_add_system_acq_rel(
                                    peer_base
                                    + fx.Int64(token_done_off)
                                    + parity * fx.Int64(token_done_stride)
                                    + fx.Int64(token_index) * fx.Int64(64),
                                    fx.Int32(1),
                                )
                            )
                            if old_token >= fx.Int32(hidden_tiles):
                                add_error()
                            if old_token + fx.Int32(1) == fx.Int32(hidden_tiles):
                                comm_ops.store_i64_global_system(
                                    peer_base
                                    + fx.Int64(token_ready_off)
                                    + parity * fx.Int64(token_ready_stride)
                                    + fx.Int64(token_index) * fx.Int64(8),
                                    generation,
                                )

            def publish_rank_group(m_row, n_group):
                """Publish one locally accumulated two-N-tile route group."""

                if tx < fx.Int32(BM):
                    packed = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(arg_stids),
                        m_row + tx,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    source = packed & fx.Int32(0x00FFFFFF)
                    if source < fx.Int32(SOURCE_CAPACITY):
                        old_pending = fx.Int32(
                            comm_ops.atomic_add_agent_acq_rel(
                                rank_pending_ptr
                                + fx.Int64(source) * fx.Int64(64),
                                fx.Int32(-1),
                            )
                        )
                        if old_pending <= fx.Int32(0):
                            add_error()
                        if const_expr(rank_accumulation_mode == "staged_reduce"):
                            old_group_pending = fx.Int32(
                                comm_ops.atomic_add_agent_acq_rel(
                                    rank_stage_group_pending_ptr
                                    + fx.Int64(
                                        source * (hidden_tiles // n_tile_group)
                                        + n_group
                                    ) * fx.Int64(4),
                                    fx.Int32(-1),
                                )
                            )
                            if old_group_pending <= fx.Int32(0):
                                add_error()
                            if old_group_pending == fx.Int32(1):
                                stage_values = global_typed_ptr(
                                    rank_stage_values_ptr, T.bf16, align=4
                                )
                                rank_acc = global_typed_ptr(
                                    rank_accumulator_ptr, T.bf16, align=4
                                )
                                for tile in range_constexpr(2):
                                    for col in range_constexpr(BN):
                                        total = fx.Float32(0.0)
                                        for slot in range_constexpr(TOPK):
                                            idx = (
                                                (source * fx.Int32(TOPK) + fx.Int32(slot))
                                                * fx.Int32(hidden_tiles)
                                                + n_group * fx.Int32(2)
                                                + fx.Int32(tile)
                                            ) * fx.Int32(BN) + fx.Int32(col)
                                            total = total + fx.Float32(
                                                fx.ptr_load(
                                                    stage_values + fx.Int64(idx)
                                                )
                                            )
                                        out_idx = (
                                            source * fx.Int32(HIDDEN)
                                            + n_group * fx.Int32(2 * BN)
                                            + fx.Int32(tile * BN + col)
                                        )
                                        fx.ptr_store(
                                            fx.BFloat16(total),
                                            rank_acc + fx.Int64(out_idx),
                                        )
                                comm_ops.atomic_add_agent_acq_rel(
                                    rank_stage_tile_done_ptr
                                    + fx.Int64(
                                        source * hidden_tiles + n_group * 2
                                    ) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                                comm_ops.atomic_add_agent_acq_rel(
                                    rank_stage_tile_pending_ptr
                                    + fx.Int64(source * hidden_tiles + n_group * 2)
                                    * fx.Int64(4),
                                    fx.Int32(-1),
                                )
                                comm_ops.atomic_add_agent_acq_rel(
                                    rank_stage_tile_pending_ptr
                                    + fx.Int64(source * hidden_tiles + n_group * 2 + 1)
                                    * fx.Int64(4),
                                    fx.Int32(-1),
                                )
                                comm_ops.atomic_add_agent_acq_rel(
                                    rank_stage_tile_done_ptr
                                    + fx.Int64(
                                        source * hidden_tiles + n_group * 2 + 1
                                    ) * fx.Int64(4),
                                    fx.Int32(1),
                                )
                            # Publish the plain rank-accumulator stores before
                            # the final source-level ready generation becomes
                            # visible to peer reducers.
                            comm_ops.fence_system_release()
                        if const_expr(rank_accumulation_mode != "staged_ring") & (
                            old_pending == fx.Int32(1)
                        ):
                            comm_ops.store_i64_global_system(
                                rank_ready_ptr
                                + fx.Int64(source) * fx.Int64(8),
                                generation,
                            )
                            source_rank = source >> fx.Int32(7)
                            source_plane = source_rank >> fx.Int32(3)
                            source_local = source_rank & fx.Int32(7)
                            token = source & fx.Int32(127)
                            token_index = (
                                source_plane * fx.Int32(MAX_TOKENS) + token
                            )
                            peer_s2 = fx.Int64(
                                fx.ptr_load(
                                    lds_typed_ptr(
                                        lds_base
                                        + fx.Int32(peer_table_off)
                                        + source_local * fx.Int32(8),
                                        T.i64,
                                        align=8,
                                    )
                                )
                            )
                            wait_ready(
                                peer_s2
                                + fx.Int64(stage2_phase_off)
                                + parity * fx.Int64(stage2_phase_stride),
                                init_phase,
                            )
                            rank_mask = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    peer_s2
                                    + fx.Int64(dest_rank_mask_off)
                                    + parity
                                    * fx.Int64(dest_rank_mask_stride)
                                    + fx.Int64(token_index) * fx.Int64(4)
                                )
                            ) & fx.Int32(0xFF)
                            expected_ranks = fx.Int32(0)
                            for source_peer in range_constexpr(GPUS_PER_NODE):
                                expected_ranks = expected_ranks + (
                                    (
                                        rank_mask
                                        >> fx.Int32(source_peer)
                                    )
                                    & fx.Int32(1)
                                )
                            if (expected_ranks <= fx.Int32(0)) | (
                                expected_ranks > fx.Int32(GPUS_PER_NODE)
                            ):
                                add_error()
                            old_rank = fx.Int32(
                                comm_ops.atomic_add_system_acq_rel(
                                    peer_s2
                                    + fx.Int64(partial_done_off)
                                    + parity
                                    * fx.Int64(partial_done_stride)
                                    + fx.Int64(token_index) * fx.Int64(64),
                                    fx.Int32(1),
                                )
                            )
                            if old_rank >= expected_ranks:
                                add_error()
                            if old_rank + fx.Int32(1) == expected_ranks:
                                queue_slot = fx.Int32(
                                    comm_ops.atomic_add_system_acq_rel(
                                        peer_s2
                                        + fx.Int64(rank_reduce_queue_tail_off)
                                        + parity
                                        * fx.Int64(rank_reduce_queue_tail_stride),
                                        fx.Int32(1),
                                    )
                                )
                                if queue_slot >= fx.Int32(2 * MAX_TOKENS):
                                    add_error()
                                comm_ops.store_i32_system(
                                    peer_s2
                                    + fx.Int64(rank_reduce_queue_off)
                                    + parity
                                    * fx.Int64(rank_reduce_queue_stride),
                                    queue_slot,
                                    token_index,
                                )
                                comm_ops.store_i64_global_system(
                                    peer_s2
                                    + fx.Int64(rank_reduce_queue_ready_off)
                                    + parity
                                    * fx.Int64(
                                        rank_reduce_queue_ready_stride
                                    )
                                    + fx.Int64(queue_slot) * fx.Int64(8),
                                    generation,
                                )

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
                    if const_expr(epilogue_schedule == "lane32_meta_expected"):
                        source_row = fx.Int32(packed_row) & fx.Int32(0x00FFFFFF)
                        expected_row = fx.Int32(1)
                        if source_row < fx.Int32(SOURCE_CAPACITY):
                            source_rank_row = source_row >> fx.Int32(7)
                            source_node_row = source_rank_row >> fx.Int32(3)
                            source_local_row = source_rank_row & fx.Int32(7)
                            token_row = source_row & fx.Int32(127)
                            peer_base_row = fx.Int64(
                                fx.ptr_load(
                                    lds_typed_ptr(
                                        lds_base
                                        + fx.Int32(peer_table_off)
                                        + source_local_row * fx.Int32(8),
                                        T.i64,
                                        align=8,
                                    )
                                )
                            )
                            expected_index_row = (
                                source_node_row * fx.Int32(MAX_TOKENS) + token_row
                            ) * fx.Int32(hidden_tiles)
                            expected_row = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    peer_base_row
                                    + fx.Int64(expected_off)
                                    + parity * fx.Int64(expected_stride)
                                    + fx.Int64(expected_index_row) * fx.Int64(4)
                                )
                            )
                            valid_expected_row = (expected_row > fx.Int32(0)) & (
                                expected_row <= fx.Int32(TOPK)
                            )
                            if not valid_expected_row:
                                add_error()
                            expected_row = valid_expected_row.select(
                                expected_row, fx.Int32(1)
                            )
                        fx.ptr_store(
                            expected_row,
                            lds_typed_ptr(
                                lds_base
                                + fx.Int32(epilogue_expected_off)
                                + tx * fx.Int32(4),
                                T.i32,
                                align=4,
                            ),
                        )

            def direct_node_atomic_issue(
                accm, m_row, n_block, cache_metadata, issue_barrier
            ):
                lds_f32 = lds_typed_ptr(lds_base, T.f32, align=4)
                meta = buffer_ops.create_buffer_resource_from_addr(arg_stids)
                weights = buffer_ops.create_buffer_resource_from_addr(arg_sweights)
                if const_expr(epilogue_schedule != "lane32" and cache_metadata):
                    cache_epilogue_metadata(m_row)
                lane_div_16 = lane // fx.Int32(16)
                lane_mod_16 = lane % fx.Int32(16)
                wave_n = BN // 4
                num_acc_n = (BN // 4) // 16
                for i in range_constexpr(BM // 16):
                    row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
                    for j in range_constexpr(num_acc_n):
                        col = wave * fx.Int32(wave_n) + fx.Int32(j * 16) + lane_mod_16
                        vec = fx.Vector(accm[i][j])
                        for v in range_constexpr(4):
                            lds_f32[(row_base + fx.Int32(v)) * fx.Int32(BN) + col] = fx.Float32(
                                vec[v]
                            )
                if const_expr(
                    epilogue_schedule == "lane32_meta_expected"
                    and cache_metadata
                ):
                    rocdl.s_waitcnt(0)
                gpu.barrier()

                atomic_bf16x2 = fx.make_copy_atom(
                    fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16
                )
                epi_lanes = 64 if epilogue_schedule == "wave64_meta" else 32
                epi_rows = THREADS // epi_lanes
                m_lane = (
                    wave
                    if epilogue_schedule == "wave64_meta"
                    else tx // fx.Int32(epi_lanes)
                )
                n_lane = (
                    lane
                    if epilogue_schedule == "wave64_meta"
                    else tx % fx.Int32(epi_lanes)
                )
                col_start = n_lane * fx.Int32(2)
                for mr in range_constexpr(BM // epi_rows):
                    row_in_block = fx.Int32(mr * epi_rows) + m_lane
                    sorted_pos = m_row + row_in_block
                    if const_expr(epilogue_schedule != "lane32"):
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
                        weight = fx.Float32(
                            fx.ptr_load(
                                lds_typed_ptr(
                                    lds_base
                                    + fx.Int32(epilogue_weight_off)
                                    + row_in_block * fx.Int32(4),
                                    T.f32,
                                    align=4,
                                )
                            )
                        )
                        if const_expr(epilogue_schedule == "wave64_meta"):
                            packed = fx.Int32(
                                rocdl.readfirstlane(T.i32, packed.ir_value())
                            )
                            weight = fx.Float32(
                                rocdl.readfirstlane(T.f32, weight.ir_value())
                            )
                    else:
                        packed = buffer_ops.buffer_load(
                            meta, sorted_pos, vec_width=1, dtype=T.i32
                        )
                        weight = buffer_ops.buffer_load(
                            weights, sorted_pos, vec_width=1, dtype=T.f32
                        )
                    source = packed & fx.Int32(0x00FFFFFF)
                    if source < fx.Int32(SOURCE_CAPACITY):
                        source_rank = source >> fx.Int32(7)
                        source_node = source_rank >> fx.Int32(3)
                        source_local = source_rank & fx.Int32(7)
                        token = source & fx.Int32(127)
                        if const_expr(node_accumulation_mode == "rank_local"):
                            row_base = (
                                source * fx.Int32(HIDDEN)
                                + n_block * fx.Int32(BN)
                                + col_start
                            )
                            peer_acc_addr = rank_accumulator_ptr
                        else:
                            peer_base = fx.Int64(
                                fx.ptr_load(
                                    lds_typed_ptr(
                                        lds_base
                                        + fx.Int32(peer_table_off)
                                        + source_local * fx.Int32(8),
                                        T.i64,
                                        align=8,
                                    )
                                )
                            )
                            if const_expr(epilogue_schedule == "wave64_meta"):
                                peer_base = fx.Int64(
                                    rocdl.readfirstlane(
                                        T.i64, peer_base.ir_value()
                                    )
                                )
                            row_base = (
                                (
                                    source_node * fx.Int32(MAX_TOKENS)
                                    + token
                                )
                                * fx.Int32(HIDDEN)
                                + n_block * fx.Int32(BN)
                                + col_start
                            )
                            peer_acc_addr = (
                                peer_base
                                + fx.Int64(accumulator_off)
                                + parity * fx.Int64(accumulator_stride)
                            )
                        if const_expr(
                            accumulator_dtype == "bf16"
                            and bf16_atomic_kind == "buffer"
                        ):
                            peer_acc = global_typed_ptr(
                                peer_acc_addr, T.bf16, align=4
                            )
                            peer_view = fx.Tensor(
                                fx.make_view(
                                    peer_acc, fx.make_layout((1, 1), (1, 1))
                                )
                            )
                            peer_out = fx.rocdl.make_buffer_tensor(
                                peer_view, max_size=True
                            )
                        if const_expr(atomic_issue_schedule == "preload_pairs"):
                            atomic_pairs = []
                            for s in range_constexpr(BN // (epi_lanes * 2)):
                                idx = (
                                    row_in_block * fx.Int32(BN)
                                    + col_start
                                    + fx.Int32(s * epi_lanes * 2)
                                )
                                values = fx.Vector(
                                    lds_vec_load(
                                        lds_base,
                                        idx * fx.Int32(4),
                                        fx.Vector.make_type(2, Float32),
                                        Float32,
                                        align=8,
                                    )
                                )
                                pair = fx.Vector.from_elements(
                                    [values[0] * weight, values[1] * weight],
                                    Float32,
                                ).to(BFloat16)
                                atomic_pairs.append(pair)
                            for s in range_constexpr(BN // (epi_lanes * 2)):
                                out_col = row_base + fx.Int32(
                                    s * epi_lanes * 2
                                )
                                out_frag = fx.make_rmem_tensor(2, BFloat16)
                                out_frag.store(atomic_pairs[s])
                                fx.copy(
                                    atomic_bf16x2,
                                    out_frag,
                                    peer_out[None, fx.Int64(out_col)],
                                )
                        else:
                            for s in range_constexpr(BN // (epi_lanes * 2)):
                                idx = (
                                    row_in_block * fx.Int32(BN)
                                    + col_start
                                    + fx.Int32(s * epi_lanes * 2)
                                )
                                values = fx.Vector(
                                    lds_vec_load(
                                        lds_base,
                                        idx * fx.Int32(4),
                                        fx.Vector.make_type(2, Float32),
                                        Float32,
                                        align=8,
                                    )
                                )
                                out_col = row_base + fx.Int32(
                                    s * epi_lanes * 2
                                )
                                if const_expr(accumulator_dtype == "bf16"):
                                    pair = fx.Vector.from_elements(
                                        [values[0] * weight, values[1] * weight],
                                        Float32,
                                    ).to(BFloat16)
                                    if const_expr(
                                        bf16_atomic_kind == "global_system"
                                    ):
                                        _atomic_add_bf16x2_system(
                                            peer_acc_addr
                                            + fx.Int64(out_col) * fx.Int64(2),
                                            pair,
                                        )
                                    else:
                                        out_frag = fx.make_rmem_tensor(2, BFloat16)
                                        out_frag.store(pair)
                                        fx.copy(
                                            atomic_bf16x2,
                                            out_frag,
                                            peer_out[None, fx.Int64(out_col)],
                                        )
                                else:
                                    _atomic_add_f32_system(
                                        peer_acc_addr
                                        + fx.Int64(out_col) * fx.Int64(4),
                                        values[0] * weight,
                                    )
                                    _atomic_add_f32_system(
                                        peer_acc_addr
                                        + fx.Int64(out_col + fx.Int32(1))
                                        * fx.Int64(4),
                                        values[1] * weight,
                                    )

                if const_expr(issue_barrier):
                    # The next GEMM may reuse the C-shuffle LDS as soon as all
                    # waves have issued their atomics; completion is deferred.
                    rocdl.s_waitcnt(lgkmcnt=0)
                    gpu.barrier()

            def direct_node_route_store_issue(
                accm, m_row, n_block, cache_metadata, issue_barrier
            ):
                """Materialize weighted BF16, then store or locally accumulate."""

                lds_bf16 = lds_typed_ptr(lds_base, T.bf16, align=2)
                if const_expr(node_accumulation_mode == "rank_local"):
                    if const_expr(rank_accumulation_mode != "staged_ring"):
                        rank_atomic_bf16x2 = fx.make_copy_atom(
                            fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16
                        )
                        rank_acc = global_typed_ptr(
                            rank_accumulator_ptr, T.bf16, align=4
                        )
                        rank_view = fx.Tensor(
                            fx.make_view(
                                rank_acc, fx.make_layout((1, 1), (1, 1))
                            )
                        )
                        rank_out = fx.rocdl.make_buffer_tensor(
                            rank_view, max_size=True
                        )
                    if const_expr(rank_accumulation_mode == "staged_reduce"):
                        stage_values = global_typed_ptr(
                            rank_stage_values_ptr, T.bf16, align=4
                        )
                    if const_expr(rank_accumulation_mode == "staged_ring"):
                        ring_payload = global_typed_ptr(
                            rank_stage_ring_payload_ptr, T.bf16, align=4
                        )
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
                        if const_expr(
                            rank_epilogue_lds_addressing == "dynamic_base"
                        ):
                            # Keep one dynamic C-shuffle row/column base live;
                            # element offsets are compile-time constants.
                            lds_row_base = row_base * fx.Int32(BN) + col
                            for v in range_constexpr(4):
                                lds_bf16[
                                    lds_row_base + fx.Int32(v * BN)
                                ] = fx.BFloat16(
                                    fx.Float32(vec[v]) * row_weights[v]
                                )
                        else:
                            for v in range_constexpr(4):
                                lds_bf16[
                                    (row_base + fx.Int32(v)) * fx.Int32(BN)
                                    + col
                                ] = fx.BFloat16(
                                    fx.Float32(vec[v]) * row_weights[v]
                                )
                gpu.barrier()

                # Match fused_moe's proven p2p_scatter_epilog exactly: one
                # route row per wave, 32 active lanes and one 16-byte store per
                # active lane. Invalid lanes use the bounded OOB offset rather
                # than entering divergent control flow.
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
                    if const_expr(node_accumulation_mode == "rank_local"):
                        packed = fx.Int32(
                            rocdl.readfirstlane(T.i32, packed.ir_value())
                        )
                    source = packed & fx.Int32(0x00FFFFFF)
                    topk_slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
                    valid = (source < fx.Int32(SOURCE_CAPACITY)) & (
                        topk_slot < fx.Int32(TOPK)
                    )
                    safe_source = valid.select(source, fx.Int32(0))
                    safe_slot = valid.select(topk_slot, fx.Int32(0))
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
                    # Keep these SSA values defined on the invalid/inactive
                    # path as well.  The metadata publication below is
                    # predicated, but FlyDSL still requires variables captured
                    # across that control-flow join to have a dominating
                    # definition.  Initialize unconditionally because the
                    # compiler still visits the other constexpr branches while
                    # emitting the fused kernel.
                    ring_seq = fx.Int64(0)
                    ring_index = fx.Int64(0)
                    if const_expr(node_accumulation_mode == "rank_local"):
                        if valid & active:
                            if const_expr(rank_accumulation_mode == "staged_reduce"):
                                stage_index = (
                                    (safe_source * fx.Int32(TOPK) + safe_slot)
                                    * fx.Int32(hidden_tiles)
                                    + n_block
                                ) * fx.Int32(BN) + col_start
                                for pair_index in range_constexpr(4):
                                    fx.ptr_store(
                                        values[pair_index * 2],
                                        stage_values
                                        + fx.Int64(stage_index + fx.Int32(pair_index * 2)),
                                    )
                                    fx.ptr_store(
                                        values[pair_index * 2 + 1],
                                        stage_values
                                        + fx.Int64(
                                            stage_index + fx.Int32(pair_index * 2 + 1)
                                        ),
                                    )
                                marker_index = (
                                    (safe_source * fx.Int32(TOPK) + safe_slot)
                                    * fx.Int32(hidden_tiles)
                                    + n_block
                                )
                                marker_old = comm_ops.atomic_add_system_acq_rel(
                                    rank_stage_slot_generation_ptr
                                    + fx.Int64(marker_index) * fx.Int64(8),
                                    fx.Int64(1),
                                )
                                if marker_old != fx.Int64(0):
                                    add_error()
                            if const_expr(rank_accumulation_mode == "staged_ring"):
                                # Reserve one bounded ring slot per route/tile
                                # payload.  The dedicated bx=1 reducer consumes
                                # these entries; sequence metadata is published
                                # only after the 512-byte BF16 payload store.
                                ring_slots = 8192
                                def reserve_ring_slot():
                                    """Reserve one bounded slot without changing the
                                    producer's payload/sequence protocol.

                                    Keeping this dynamic admission loop in one
                                    helper gives FlyDSL a single AST region to
                                    lower instead of rebuilding it around every
                                    route-group epilogue call.
                                    """
                                    ring_seq_local = fx.Int64(0)
                                    if lane == fx.Int32(0):
                                        admitted = fx.Int32(0)
                                        while admitted == fx.Int32(0):
                                            lock_old = fx.Int32(
                                                comm_ops.atomic_add_system_acq_rel(
                                                    rank_stage_ring_reserve_lock_ptr,
                                                    fx.Int32(1),
                                                )
                                            )
                                            if lock_old == fx.Int32(0):
                                                head_now = fx.Int64(
                                                    comm_ops.load_i64_global_system(
                                                        rank_stage_ring_head_ptr
                                                    )
                                                )
                                                tail_now = fx.Int64(
                                                    comm_ops.load_i64_global_system(
                                                        rank_stage_ring_tail_ptr
                                                    )
                                                )
                                                if head_now - tail_now < fx.Int64(
                                                    ring_slots
                                                ):
                                                    ring_seq_local = head_now
                                                    comm_ops.store_i64_global_system(
                                                        rank_stage_ring_head_ptr,
                                                        head_now + fx.Int64(1),
                                                    )
                                                    admitted = fx.Int32(1)
                                                comm_ops.store_i32_system(
                                                    rank_stage_ring_reserve_lock_ptr,
                                                    fx.Int32(0),
                                                    fx.Int32(0),
                                                )
                                            else:
                                                comm_ops.atomic_add_system_acq_rel(
                                                    rank_stage_ring_reserve_lock_ptr,
                                                    fx.Int32(-1),
                                                )
                                    return fx.Int64(
                                        rocdl.readfirstlane(
                                            T.i64, ring_seq_local.ir_value()
                                        )
                                    )

                                ring_seq = reserve_ring_slot()
                                ring_index = ring_seq & fx.Int64(ring_slots - 1)
                                ring_base = ring_index * fx.Int64(BN)
                                for pair_index in range_constexpr(4):
                                    fx.ptr_store(
                                        values[pair_index * 2],
                                        ring_payload + ring_base
                                        + fx.Int64(col_start + pair_index * 2),
                                    )
                                    fx.ptr_store(
                                        values[pair_index * 2 + 1],
                                        ring_payload + ring_base
                                        + fx.Int64(col_start + pair_index * 2 + 1),
                                    )
                                comm_ops.fence_agent_release()
                                if lane == fx.Int32(0):
                                    fx.ptr_store(
                                        safe_source,
                                        global_typed_ptr(
                                            rank_stage_ring_source_ptr,
                                            T.i32,
                                            align=4,
                                        )
                                        + ring_index,
                                    )
                                    fx.ptr_store(
                                        safe_slot.to(fx.Int16),
                                        global_typed_ptr(
                                            rank_stage_ring_slot_ptr,
                                            T.i16,
                                            align=2,
                                        )
                                        + ring_index,
                                    )
                                    fx.ptr_store(
                                        n_block.to(fx.Int16),
                                        global_typed_ptr(
                                            rank_stage_ring_tile_ptr,
                                            T.i16,
                                            align=2,
                                        )
                                        + ring_index,
                                    )
                            if const_expr(rank_accumulation_mode != "staged_ring"):
                                rank_row_base = (
                                    safe_source * fx.Int32(HIDDEN)
                                    + n_block * fx.Int32(BN)
                                    + col_start
                                )
                                for pair_index in range_constexpr(4):
                                    pair = fx.Vector.from_elements(
                                        [
                                            values[pair_index * 2],
                                            values[pair_index * 2 + 1],
                                        ],
                                        BFloat16,
                                    )
                                    out_frag = fx.make_rmem_tensor(2, BFloat16)
                                    out_frag.store(pair)
                                    fx.copy(
                                        rank_atomic_bf16x2,
                                        out_frag,
                                        rank_out[
                                            None,
                                            fx.Int64(
                                                rank_row_base
                                                + fx.Int32(pair_index * 2)
                                            ),
                                        ],
                                    )
                    # Ensure all route lanes finish the payload before lane 0
                    # publishes metadata/sequence.  This barrier is outside
                    # the valid/active predicate to avoid divergence deadlock.
                    gpu.barrier()
                    if const_expr(rank_accumulation_mode == "staged_ring"):
                        if (valid & active) & (lane == fx.Int32(0)):
                            comm_ops.store_i64_global_system(
                                rank_stage_ring_sequence_ptr
                                + ring_index * fx.Int64(8),
                                ring_seq + fx.Int64(1),
                            )
                        source_rank = safe_source >> fx.Int32(7)
                        source_node = source_rank >> fx.Int32(3)
                        source_local = source_rank & fx.Int32(7)
                        token = safe_source & fx.Int32(127)
                        peer_base = fx.Int64(
                            fx.ptr_load(
                                lds_typed_ptr(
                                    lds_base
                                    + fx.Int32(peer_table_off)
                                    + source_local * fx.Int32(8),
                                    T.i64,
                                    align=8,
                                )
                            )
                        )
                        peer_base = fx.Int64(
                            rocdl.readfirstlane(T.i64, peer_base.ir_value())
                        )
                        route_tile_index = (
                            (
                                (
                                    source_node * fx.Int32(MAX_TOKENS) + token
                                )
                                * fx.Int32(hidden_tiles)
                                + n_block
                            )
                            * fx.Int32(TOPK)
                            + safe_slot
                        )
                        route_tile_addr = (
                            peer_base
                            + fx.Int64(route_slots_off)
                            + parity * fx.Int64(route_slots_stride)
                            + fx.Int64(route_tile_index) * fx.Int64(BN * 2)
                        )
                        route_rsrc = (
                            buffer_ops.create_buffer_resource_from_addr(
                                route_tile_addr,
                                num_records_bytes=BN * 2,
                            )
                        )
                        store_offset = (valid & active).select(
                            col_start * fx.Int32(2), fx.Int32(BN * 2)
                        )
                        buffer_ops.buffer_store(
                            values.ir_value(),
                            route_rsrc,
                            store_offset,
                            offset_is_bytes=True,
                            cache_modifier=2,
                        )

                if const_expr(issue_barrier):
                    rocdl.s_waitcnt(lgkmcnt=0)
                    gpu.barrier()

            def issue_node_epilogue(
                accm, m_row, n_block, cache_metadata, issue_barrier
            ):
                if const_expr(
                    node_accumulation_mode in ("route_store", "rank_local")
                ):
                    direct_node_route_store_issue(
                        accm,
                        m_row,
                        n_block,
                        cache_metadata,
                        issue_barrier,
                    )
                else:
                    direct_node_atomic_issue(
                        accm,
                        m_row,
                        n_block,
                        cache_metadata,
                        issue_barrier,
                    )

            def gmm2_sink_epilog(accm, work):
                checksum = fx.Float32(0.0)
                for i in range_constexpr(BM // 16):
                    for j in range_constexpr((BN // 4) // 16):
                        vec = fx.Vector(accm[i][j])
                        for v in range_constexpr(4):
                            checksum = checksum + fx.Float32(vec[v])
                sink = buffer_ops.create_buffer_resource_from_addr(accumulator_ptr)
                sink_index = (work * fx.Int32(THREADS) + tx) % fx.Int32(
                    2 * MAX_TOKENS * HIDDEN
                )
                buffer_ops.buffer_store(checksum, sink, sink_index)

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
            ):
                work = m_block * fx.Int32(hidden_tiles) + n_block
                if const_expr(diagnostic_mode == "atomic_only"):
                    m_row = m_block * fx.Int32(BM)
                    accm = [
                        [
                            fx.Vector.filled(4, 0.0, fx.Float32)
                            for _ in range((BN // 4) // 16)
                        ]
                        for _ in range(BM // 16)
                    ]
                else:
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
                        use_nt=False,
                        INTER_MAX=INTER,
                        aStages=a_stages,
                        a_dtype="fp4",
                        has_pad=False,
                        SBM=BM,
                        g2_bhoist=True,
                        g2_ascale_pf=True,
                        expert_offset=0,
                        preloaded_expert=preloaded_expert,
                    )
                return accm, m_row

            def run_gmm_work(work):
                m_block = work // fx.Int32(n_groups)
                n_group = work - m_block * fx.Int32(n_groups)
                n_block0 = n_group * fx.Int32(n_tile_group)
                group_m_row = m_block * fx.Int32(BM)
                preloaded_expert = None
                if const_expr(group_pipeline_schedule != "baseline"):
                    if const_expr(diagnostic_mode != "atomic_only"):
                        preloaded_expert = rocdl.readfirstlane(
                            T.i32,
                            global_typed_ptr(arg_eids, T.i32)[m_block],
                        )
                    if const_expr(
                        epilogue_schedule != "lane32"
                        and diagnostic_mode != "gmm2_only"
                    ):
                        cache_epilogue_metadata(group_m_row)

                if const_expr(group_pipeline_schedule == "runtime_loop"):
                    # A runtime two-iteration loop keeps a single GEMM body in
                    # the ISA instead of statically cloning it for both N tiles.
                    for sub in range(fx.Int32(0), fx.Int32(2), fx.Int32(1)):
                        n_block = n_block0 + sub
                        accm, m_row = compute_gmm_tile(
                            m_block,
                            n_block,
                            preloaded_expert,
                            lds_base,
                            False,
                        )
                        if const_expr(diagnostic_mode == "gmm2_only"):
                            gmm2_sink_epilog(
                                accm,
                                m_block * fx.Int32(hidden_tiles) + n_block,
                            )
                            rocdl.s_waitcnt(lgkmcnt=0)
                            gpu.barrier()
                        else:
                            issue_node_epilogue(
                                accm, m_row, n_block, False, True
                            )
                    if const_expr(diagnostic_mode != "gmm2_only"):
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        if const_expr(node_accumulation_mode == "rank_local"):
                            comm_ops.fence_agent_release()
                        else:
                            comm_ops.fence_system_release()
                        if const_expr(node_accumulation_mode == "rank_local"):
                            publish_rank_group(group_m_row, n_group)
                        else:
                            publish_node_tile(group_m_row, n_block0)
                            publish_node_tile(
                                group_m_row, n_block0 + fx.Int32(1)
                            )
                else:
                    accm0, m_row = compute_gmm_tile(
                        m_block,
                        n_block0,
                        preloaded_expert,
                        lds_base,
                        False,
                    )
                    if const_expr(diagnostic_mode == "gmm2_only"):
                        gmm2_sink_epilog(
                            accm0,
                            m_block * fx.Int32(hidden_tiles) + n_block0,
                        )
                        if const_expr(n_tile_group == 2):
                            n_block1 = n_block0 + fx.Int32(1)
                            if const_expr(
                                group_pipeline_schedule == "a_double_buffer"
                            ):
                                gmm_only_a_next = lds_base + fx.Int32(
                                    a_double_buffer_off
                                )
                                preload_initial_a(m_block, gmm_only_a_next)
                                rocdl.s_waitcnt(0)
                                gpu.barrier()
                            else:
                                rocdl.s_waitcnt(lgkmcnt=0)
                                gpu.barrier()
                            accm1, _ = compute_gmm_tile(
                                m_block,
                                n_block1,
                                preloaded_expert,
                                (
                                    lds_base + fx.Int32(a_double_buffer_off)
                                    if group_pipeline_schedule == "a_double_buffer"
                                    else lds_base
                                ),
                                group_pipeline_schedule == "a_double_buffer",
                            )
                            gmm2_sink_epilog(
                                accm1,
                                m_block * fx.Int32(hidden_tiles) + n_block1,
                            )
                    elif const_expr(n_tile_group == 1):
                        issue_node_epilogue(
                            accm0, m_row, n_block0, True, False
                        )
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        if const_expr(node_accumulation_mode == "rank_local"):
                            comm_ops.fence_agent_release()
                        else:
                            comm_ops.fence_system_release()
                        if const_expr(node_accumulation_mode == "rank_local"):
                            publish_rank_group(m_row, n_group)
                        else:
                            publish_node_tile(m_row, n_block0)
                    else:
                        n_block1 = n_block0 + fx.Int32(1)
                        if const_expr(
                            group_pipeline_schedule == "a_double_buffer"
                        ):
                            # Complete the next N tile's initial A DMA before
                            # issuing atomic0.  Its GEMM can then enter MFMA
                            # without a post-atomic A dependency forcing
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
                                lds_base + fx.Int32(a_double_buffer_off)
                                if group_pipeline_schedule == "a_double_buffer"
                                else lds_base
                            ),
                            group_pipeline_schedule == "a_double_buffer",
                        )
                        issue_node_epilogue(
                            accm1, m_row, n_block1, False, False
                        )
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        if const_expr(node_accumulation_mode == "rank_local"):
                            comm_ops.fence_agent_release()
                        else:
                            comm_ops.fence_system_release()
                        if const_expr(node_accumulation_mode == "rank_local"):
                            publish_rank_group(m_row, n_group)
                        else:
                            publish_node_tile(m_row, n_block0)
                            publish_node_tile(m_row, n_block1)

            if const_expr(gmm_schedule == "static_strided"):
                worker = bx - fx.Int32(gmm_first)
                worker_count = grid - fx.Int32(gmm_first)
                for work in range(worker, total_work, worker_count):
                    run_gmm_work(work)
            else:
                while active:
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        local_work = fx.Int32(
                            comm_ops.atomic_add_agent(
                                gemm_head_ptr + fx.Int64(shard) * fx.Int64(64),
                                fx.Int32(1),
                            )
                        )
                        work = shard + local_work * fx.Int32(WORK_SHARDS)
                        fx.ptr_store(work, work_ptr)
                    gpu.barrier()
                    work = fx.Int32(fx.ptr_load(work_ptr))
                    has_work = work < total_work
                    if has_work:
                        run_gmm_work(work)
                    active = has_work
            if const_expr(node_reduce_rejoin_blocks > 0):
                gmm_worker = bx - fx.Int32(gmm_first)
                if gmm_worker < fx.Int32(node_reduce_rejoin_blocks):
                    # Joining happens only after this CTA has exhausted its GMM
                    # shard.  Waiting for all queue reservations avoids having
                    # helpers claim not-yet-produced slots; each slot still has
                    # its own generation-ready acquire in the shared helper.
                    reduce_work_items = load_rank_reduce_work_items()
                    rank_reduce_acc_rsrc = (
                        buffer_ops.create_buffer_resource_from_addr(
                            accumulator_ptr,
                            num_records_bytes=accumulator_stride,
                        )
                    )
                    rank_reduce_tx_rsrc = (
                        buffer_ops.create_buffer_resource_from_addr(
                            tx_ptr,
                            num_records_bytes=tx_stride,
                        )
                    )
                    rank_mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                        dest_rank_mask_ptr,
                        num_records_bytes=dest_rank_mask_stride,
                    )
                    rank_tx_slot_rsrc = (
                        buffer_ops.create_buffer_resource_from_addr(
                            rank_tx_slot_ptr,
                            num_records_bytes=rank_tx_slot_stride,
                        )
                    )
                    consume_dynamic_rank_reduce(
                        reduce_work_items,
                        rank_reduce_acc_rsrc,
                        rank_reduce_tx_rsrc,
                        rank_mask_rsrc,
                        rank_tx_slot_rsrc,
                        True,
                    )
            if const_expr(rank_accumulation_mode == "staged_ring"):
                # EOS from each resident GMM producer lets bx=1 drain the
                # bounded ring after the final payload publication.
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.atomic_add_system_acq_rel(
                        rank_stage_ring_producer_done_ptr, fx.Int32(1)
                    )
            if const_expr(timeline_instrument):
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
    launch_megamoe_tile_ep16_stage2.lds_bytes = lds_bytes
    launch_megamoe_tile_ep16_stage2.diagnostic_mode = diagnostic_mode
    launch_megamoe_tile_ep16_stage2.gemm2_contraction = diagnostic_mode in (
        "full",
        "gmm2_only",
        "gmm2_atomic_only",
        "route_store_only",
    )
    launch_megamoe_tile_ep16_stage2.communication_roles_enabled = (
        diagnostic_mode in ("full", "return_only")
    )
    launch_megamoe_tile_ep16_stage2.accumulator_dtype = accumulator_dtype
    launch_megamoe_tile_ep16_stage2.final_combine_blocks = final_combine_blocks
    launch_megamoe_tile_ep16_stage2.gmm_schedule = gmm_schedule
    launch_megamoe_tile_ep16_stage2.return_chunk_tokens = return_chunk_tokens
    launch_megamoe_tile_ep16_stage2.node_ready_granularity = "token"
    launch_megamoe_tile_ep16_stage2.bf16_atomic_kind = bf16_atomic_kind
    launch_megamoe_tile_ep16_stage2.rail_return_schedule = rail_return_schedule
    launch_megamoe_tile_ep16_stage2.epilogue_schedule = epilogue_schedule
    launch_megamoe_tile_ep16_stage2.n_tile_group = n_tile_group
    launch_megamoe_tile_ep16_stage2.group_pipeline_schedule = (
        group_pipeline_schedule
    )
    launch_megamoe_tile_ep16_stage2.node_accumulation_mode = (
        node_accumulation_mode
    )
    launch_megamoe_tile_ep16_stage2.rank_accumulation_mode = rank_accumulation_mode
    launch_megamoe_tile_ep16_stage2.node_reduce_blocks = node_reduce_blocks
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
    launch_megamoe_tile_ep16_stage2.scoreboard_schedule = scoreboard_schedule
    launch_megamoe_tile_ep16_stage2.atomic_issue_schedule = atomic_issue_schedule
    launch_megamoe_tile_ep16_stage2.timeline_instrument = bool(
        timeline_instrument
    )
    launch_megamoe_tile_ep16_stage2.single_gpu_launch = True
    launch_megamoe_tile_ep16_stage2.requires_resident_grid = True
    launch_megamoe_tile_ep16_stage2.fixed_roles = {
        "cross": 1,
        "stage_ring_reduce": stage_ring_reducer_blocks,
        "node_reduce": reduce_blocks,
        "intranode_final": final_combine_blocks,
        "gmm2": "remaining blocks",
    }
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
    if node_accumulation_mode == "route_store":
        combine_prefix = (
            f"GMM2 weighted {accumulator_dtype.upper()} route-slot store -> "
            "node-local FP32 register reduce -> "
        )
    elif node_accumulation_mode == "rank_local":
        if rank_accumulation_mode == "staged_ring":
            combine_prefix = (
                "GMM2 weighted BF16 ring payload copy -> dedicated stage-ring "
                "CTA FP32 source/group reduce -> source-proxy peer-pull FP32 "
                "register reduce -> "
            )
        elif rank_accumulation_mode == "staged_reduce":
            combine_prefix = (
                f"GMM2 weighted {accumulator_dtype.upper()} staged route copy -> "
                "last-contributor FP32 source/group reduce -> "
                "source-proxy peer-pull FP32 register reduce -> "
            )
        else:
            combine_prefix = (
                f"GMM2 weighted {accumulator_dtype.upper()} local rank atomic -> "
                "source-proxy peer-pull FP32 register reduce -> "
            )
    else:
        combine_prefix = (
            f"GMM2 weighted {accumulator_dtype.upper()} LSA atomic -> "
        )
    launch_megamoe_tile_ep16_stage2.combine_contract = (
        combine_prefix
        + "direct node accumulator -> "
        "one BF16 RAIL return/token -> source final add"
    )
    launch_megamoe_tile_ep16_stage2.architecture_contract = {
        "epilogue": (
            "source_aligned_route_slot_store_then_register_reduce"
            if node_accumulation_mode == "route_store"
            else (
                "rank_local_atomic_then_peer_pull_node_reduce"
                if node_accumulation_mode == "rank_local"
                else "direct_lsa_atomic_source_aligned_node_accumulator"
            )
        ),
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
        "return_chunk_tokens": return_chunk_tokens,
        "node_ready_granularity": "token",
        "bf16_atomic_kind": bf16_atomic_kind,
        "rail_return_schedule": rail_return_schedule,
        "epilogue_schedule": epilogue_schedule,
        "n_tile_group": n_tile_group,
        "group_pipeline_schedule": group_pipeline_schedule,
        "node_accumulation_mode": node_accumulation_mode,
        "rank_accumulation_mode": rank_accumulation_mode,
        "node_reduce_blocks": node_reduce_blocks,
        "node_reduce_vec_bytes": node_reduce_vec_bytes,
        "node_reduce_schedule": node_reduce_schedule,
        "node_reduce_load_schedule": node_reduce_load_schedule,
        "node_reduce_work_schedule": node_reduce_work_schedule,
        "node_reduce_rejoin_blocks": node_reduce_rejoin_blocks,
        "rank_epilogue_lds_addressing": rank_epilogue_lds_addressing,
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
    return launch_megamoe_tile_ep16_stage2


__all__ = ["compile_megamoe_tile_ep16_stage2_a4w4"]
