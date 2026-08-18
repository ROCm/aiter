# SPDX-License-Identifier: MIT
"""One-launch EP16 dispatch + A4W4 GMM1 + SiLU Stage-1.

This is the first production-shape specialization.  It deliberately keeps the
MORI InterNodeV1 semantic boundary:

* BF16 is quantized once per source token;
* one activation row is copied per selected destination rank on the source node;
* one record crosses RAIL per source token / remote node;
* the aligned proxy copies its activation once per selected destination rank;
* each destination rank expands every locally owned top-k slot into an expert
  route that gathers the shared activation row.

The 4096-byte record carries a 16-bit top-k-slot bitmap per EP rank.  This keeps
node/rank payload deduplication while preserving duplicate-rank routes and
their independent expert IDs and weights.

For the sparse production path, a full BM32 expert tile is release-published
to an in-kernel ready queue by its last route arrival.  Finished communication
CTAs immediately rejoin as GMM1 consumers, overlapping remaining fanout with
GMM1, SiLU and A4 requant.  Partial expert tails are padded and published only
after the eight communication EOS signals are acquired.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from . import comm_ops
from .gemm_common import MXFP4_SCALE_LAYOUT_TAG, k_tiles_total_for
from .gemm1 import (
    _bm_constants,
    _gemm1_body,
)

from .cco.ops import (
    flush_async,
    put,
    put_value,
    wait_ready,
    wait_request,
)
from .stage1_abi import (
    SPARSE_QP_GENERATION_SHIFT,
    SPARSE_QP_TOKEN_BITS,
    Stage1ArenaLayout,
)


# K3 production constants.  Keeping them visible makes trace/resource audits
# independent of host-side config objects.
HIDDEN = 7168
INTER = 3072
EXPERTS = 896
WORLD = 16
GPUS_PER_NODE = 8
LOCAL_EXPERTS = 56
TOPK = 16
MAX_TOKENS = 128
BM = 32
BN = 256
BK = 256
THREADS = 256
COMM_CTAS = GPUS_PER_NODE
PRODUCER_CTAS = MAX_TOKENS
CCO_TICKET = 0
PRODUCER_FIRST = COMM_CTAS
ESSENTIAL_CTAS = PRODUCER_FIRST + PRODUCER_CTAS
INVALID_SOURCE = WORLD * MAX_TOKENS

_FP4_INV_MAX_POS_BITS = 0x3E2AAAAB
TEAM_RAIL = "rail"

# Diagnostic setup and primary kernels can share one non-parity epoch gate in
# consecutive launches.  Reserve a full three-bit phase field so every phase
# remains distinct across adjacent generations.
DIAGNOSTIC_PHASE_IDS = {
    "full": 0,
    "quant_pack_only": 1,
    "transport_only": 2,
    "fanout_only": 3,
    "quant_core_only": 4,
    "dispatch_only": 5,
}
DIAGNOSTIC_CONTROL_GENERATION_BITS = 3


def _plane_bytes(layout: Stage1ArenaLayout, name: str) -> int:
    region = layout.region(name)
    if not region.shape or region.shape[0] != layout.parity_depth:
        raise ValueError(f"{name} is not parity indexed")
    return region.nbytes // layout.parity_depth


def compile_megamoe_tile_ep16_stage1(
    layout: Stage1ArenaLayout,
    stage2_layout,
    *,
    rank: int,
    stage2_window_offset: int = 0,
    worker_blocks: int = 512,
    work_shards: int = 8,
    waves_per_eu_hint: int = 2,
    enable_cco: bool = True,
    diagnostic_comm_only: bool = False,
    diagnostic_split_fanout: bool = False,
    diagnostic_wave_fanout: bool = False,
    diagnostic_no_arrival_rmw: bool = False,
    cco_chunks_per_flush: int = 1,
    cco_geometry: str = "chunked",
    diagnostic_phase: str = "full",
    quant_two_cta_per_token: bool = False,
    prequant_input: bool = False,
    tile_pipeline: bool = False,
    tile_pipeline_instrument: bool = False,
    tile_pipeline_fanout_shards: int = 16,
):
    """Compile the fixed K3 EP16 Stage-1 persistent kernel.

    ``arena_ptr`` is the base of the one registered two-kernel window.
    ``stage2_window_offset`` locates the logical :class:`Stage2ArenaLayout`
    inside that same window.  The launcher performs exactly one GPU launch.
    """

    if not isinstance(layout, Stage1ArenaLayout):
        raise TypeError("layout must be Stage1ArenaLayout")
    expected = (HIDDEN, INTER, EXPERTS, WORLD, GPUS_PER_NODE, TOPK, MAX_TOKENS, BM, BN)
    actual = (
        layout.hidden,
        layout.inter,
        layout.experts,
        layout.world_size,
        layout.gpus_per_node,
        layout.topk,
        layout.max_tokens,
        layout.block_m,
        layout.block_n,
    )
    if actual != expected:
        raise ValueError(f"Stage-1 layout is not the fixed K3 specialization: {actual}")
    if not 0 <= int(rank) < WORLD:
        raise ValueError("rank must be in [0, 16)")
    if worker_blocks < ESSENTIAL_CTAS:
        raise ValueError(
            f"worker_blocks must be >= {ESSENTIAL_CTAS} so all progress roles are resident"
        )
    if work_shards not in (1, 2, 4, 8):
        raise ValueError("work_shards must be one of 1,2,4,8")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")
    if not enable_cco:
        raise ValueError("strict EP16 Stage-1 requires real CCO transport")
    if diagnostic_split_fanout:
        if worker_blocks != 256:
            raise ValueError(
                "internodev1 split fanout requires worker_blocks=256"
            )
    if diagnostic_wave_fanout:
        if not diagnostic_split_fanout:
            raise ValueError(
                "diagnostic_wave_fanout requires split fanout mode"
            )
        if quant_two_cta_per_token:
            raise ValueError(
                "diagnostic_wave_fanout does not support two-CTA quant"
            )
    if diagnostic_no_arrival_rmw:
        if not diagnostic_comm_only or not diagnostic_split_fanout:
            raise ValueError(
                "diagnostic_no_arrival_rmw requires split comm-only mode"
            )
    if cco_chunks_per_flush not in (1, 2, 4, 8):
        raise ValueError("cco_chunks_per_flush must be one of 1,2,4,8")
    if cco_geometry not in ("chunked", "mori64x2", "sparse_wqe"):
        raise ValueError(
            "cco_geometry must be chunked, mori64x2, or sparse_wqe"
        )
    if diagnostic_phase not in (
        "full",
        "quant_core_only",
        "quant_pack_only",
        "transport_only",
        "fanout_only",
        "dispatch_only",
    ):
        raise ValueError(
            "diagnostic_phase must be full, quant_core_only, quant_pack_only, "
            "transport_only, fanout_only, or dispatch_only"
        )
    if diagnostic_phase != "full":
        if not diagnostic_comm_only or not diagnostic_split_fanout:
            raise ValueError(
                "non-full diagnostic_phase requires split comm-only mode"
            )
    if cco_geometry == "mori64x2":
        if not diagnostic_split_fanout:
            raise ValueError("mori64x2 geometry requires split fanout")
    if cco_geometry == "sparse_wqe":
        if not diagnostic_split_fanout or diagnostic_phase != "full":
            raise ValueError(
                "sparse_wqe requires split fanout with full phase"
            )
        if quant_two_cta_per_token:
            raise ValueError("sparse_wqe does not support two-CTA quant")
        if diagnostic_wave_fanout:
            raise ValueError(
                "sparse_wqe initially requires the CTA fanout path"
            )
    if quant_two_cta_per_token:
        if not diagnostic_split_fanout or worker_blocks != 256:
            raise ValueError(
                "quant_two_cta_per_token requires split 256-CTA diagnostic"
            )
    if prequant_input:
        if diagnostic_phase != "full":
            raise ValueError("prequant_input currently requires full Stage1")
        if quant_two_cta_per_token:
            raise ValueError("prequant_input does not use two-CTA quant")
    if tile_pipeline:
        if (
            diagnostic_comm_only
            or not diagnostic_split_fanout
            or diagnostic_wave_fanout
            or cco_geometry != "sparse_wqe"
            or diagnostic_phase != "full"
            or worker_blocks != 256
        ):
            raise ValueError(
                "tile_pipeline requires the full real-GMM1 256-CTA "
                "split sparse_wqe path"
            )
        if work_shards != 8:
            raise ValueError("tile_pipeline requires work_shards=8")
    if tile_pipeline_instrument and not tile_pipeline:
        raise ValueError("tile_pipeline_instrument requires tile_pipeline")
    if int(tile_pipeline_fanout_shards) not in (8, 12, 16):
        raise ValueError("tile_pipeline_fanout_shards must be 8, 12, or 16")

    rank = int(rank)
    local_rank = rank % GPUS_PER_NODE
    node = rank // GPUS_PER_NODE
    remote_node = 1 - node
    remote_source_rank = remote_node * GPUS_PER_NODE + local_rank
    stage2_window_offset = int(stage2_window_offset)
    if stage2_window_offset < 0 or stage2_window_offset % 4096:
        raise ValueError("stage2_window_offset must be non-negative and 4096-byte aligned")

    wire = layout.wire
    record_bytes = wire.record_bytes
    record_dwords = record_bytes // 4
    payload_dwords = wire.payload_bytes // 4
    scale_bytes = wire.scale_bytes
    scale_dwords = scale_bytes // 4
    records_per_chunk = wire.records_per_chunk
    dispatch_chunks = layout.dispatch_chunks
    if dispatch_chunks % cco_chunks_per_flush:
        raise ValueError("cco_chunks_per_flush must divide dispatch_chunks")
    records_per_qp = records_per_chunk // layout.num_qp
    qp_bytes = records_per_qp * record_bytes
    max_route_tiles = layout.max_route_tiles
    max_route_rows = layout.max_route_rows
    max_tiles_per_expert = layout.max_tiles_per_expert
    h1_n_blocks = layout.h1_n_blocks
    max_jobs = max_route_tiles * h1_n_blocks
    split_fanout_shards = (
        int(tile_pipeline_fanout_shards) if tile_pipeline else 16
    )
    split_fanout_ctas = GPUS_PER_NODE * split_fanout_shards
    dedicated_compute_ctas = worker_blocks - 2 * split_fanout_ctas

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(HIDDEN, BK)
    _, _, _, lds_bytes = _bm_constants(BM, BN, kh_tile, k_tiles_total)

    # Compile-time region helpers.  CCO offsets are relative to the physical
    # window; local addresses add arena_ptr at runtime.
    def off(name):
        return int(layout.region(name).offset)

    def plane(name):
        return _plane_bytes(layout, name)

    s2_parity_depth = int(stage2_layout.parity_depth)
    if s2_parity_depth != 2:
        raise ValueError("Stage-2 metadata must be double buffered")

    def s2_off(name):
        return int(stage2_layout.region(name).offset)

    def s2_plane(name):
        region = stage2_layout.region(name)
        if not region.shape or region.shape[0] != s2_parity_depth:
            raise ValueError(f"Stage-2 region {name} is not parity indexed")
        return region.nbytes // s2_parity_depth

    # Resolve the shared metadata contract at compile time, not in the hot path.
    for required in (
        "node_dest_rank_mask",
        "source_token_count",
        "node_expected",
        "stage1_done",
    ):
        stage2_layout.region(required)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    transport_tag = "cco" if enable_cco else "stub"
    kernel_name = (
        f"megamoe_tile_ep16_stage1_k3_a4w4_silu_r{rank}_"
        f"wb{worker_blocks}_ws{work_shards}_{transport_tag}_{MXFP4_SCALE_LAYOUT_TAG}_"
        "scoreboard_v14_tilequeue_rankslots_payload2_qpballot4_no_send_atomic_inputscale_abi3"
        + ("_diagnostic_comm_only" if diagnostic_comm_only else "")
        + (
            (
                "_internodev1_split128x2_grid256"
                if diagnostic_comm_only
                else (
                    f"_split{split_fanout_ctas}x2_tilepipe{worker_blocks}r"
                    if tile_pipeline
                    else "_split128x2_rejoin256_posteos"
                )
            )
            if diagnostic_split_fanout
            else ""
        )
        + ("_wave64x1_fullgrid" if diagnostic_wave_fanout else "")
        + ("_no_arrival_rmw" if diagnostic_no_arrival_rmw else "")
        + (
            f"_cco_flushb{cco_chunks_per_flush}"
            if cco_chunks_per_flush != 1 and cco_geometry == "chunked"
            else ""
        )
        + ("_cco_mori64x2" if cco_geometry == "mori64x2" else "")
        + ("_cco_sparse_wqe" if cco_geometry == "sparse_wqe" else "")
        + (
            f"_phase_{diagnostic_phase}"
            if diagnostic_phase != "full"
            else ""
        )
        + ("_quant2cta" if quant_two_cta_per_token else "")
        + ("_prequant_input" if prequant_input else "")
        + ("_overlapstats" if tile_pipeline_instrument else "")
        + (
            "_qpstream_intergate"
            if cco_geometry == "sparse_wqe"
            else ""
        )
    )

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        x_bf16: fx.Int64,
        input_scale: fx.Int64,
        route_weights: fx.Int64,
        topk_ids: fx.Int64,
        w1q: fx.Int64,
        w1scale: fx.Int64,
        ntokens: fx.Int32,
        generation: fx.Int64,
    ):
        lds_raw = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        arena_window = cco.Window(arena_win)
        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx & fx.Int32(63)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        parity = fx.Int64(generation & fx.Int64(1))
        phase_id = DIAGNOSTIC_PHASE_IDS[diagnostic_phase]
        control_generation = (
            generation
            if const_expr(diagnostic_phase == "full")
            else (
                (
                    generation
                    << fx.Int64(DIAGNOSTIC_CONTROL_GENERATION_BITS)
                )
                + fx.Int64(phase_id)
            )
        )

        def local_addr(name):
            return arena_ptr + fx.Int64(off(name)) + parity * fx.Int64(plane(name))

        def window_off(name):
            return fx.Int64(off(name)) + parity * fx.Int64(plane(name))

        def stage2_addr(name):
            return (
                arena_ptr
                + fx.Int64(stage2_window_offset + s2_off(name))
                + parity * fx.Int64(s2_plane(name))
            )

        error_addr = arena_ptr + fx.Int64(off("error_count"))

        # Arrival tickets, rather than block IDs, ensure every progress role is
        # among the first resident CTAs on an oversubscribed persistent grid.
        ticket_ptr = fx.recast_iter(fx.Int64, lds_raw)
        ticket_view = fx.make_view(ticket_ptr, fx.make_layout(1, 1))
        if tx == fx.Int32(0):
            ticket_lane = fx.Int64(
                comm_ops.atomic_add_agent(
                    arena_ptr + fx.Int64(off("entry_count")), fx.Int64(1)
                )
            )
            fx.ptr_store(Vec.from_elements([ticket_lane], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        ticket = fx.Int32(ticket64 % fx.Int64(worker_blocks))
        is_initializer = ticket == fx.Int32(0)
        transport_enabled = diagnostic_phase in (
            "full",
            "transport_only",
            "dispatch_only",
        )
        quant_enabled = diagnostic_phase in (
            "full",
            "quant_core_only",
            "quant_pack_only",
        )
        fanout_enabled = diagnostic_phase in (
            "full",
            "fanout_only",
            "dispatch_only",
        )
        is_cco = (ticket == fx.Int32(CCO_TICKET)) & (
            fx.Int32(1 if transport_enabled else 0) == fx.Int32(1)
        )
        if const_expr(diagnostic_split_fanout):
            # The tile pipeline can tune 8/12/16 shards per destination.
            # With fewer than 16, the unused fanout tickets wait directly on
            # the ready queue; with 16, all roles finish their own short
            # fanout slice and immediately rejoin as GMM1 consumers.  The
            # post-EOS reference retains the original 128+128 mapping.
            inter_ticket = ticket < fx.Int32(split_fanout_ctas)
            intra_ticket = (ticket >= fx.Int32(128)) & (
                ticket < fx.Int32(128 + split_fanout_ctas)
            )
            quant_ticket = ticket < fx.Int32(MAX_TOKENS)
            phase_fanout = fx.Int32(1 if fanout_enabled else 0)
            fanout_pred = phase_fanout == fx.Int32(1)
            is_inter_fanout = inter_ticket & fanout_pred
            is_intra_fanout = intra_ticket & fanout_pred
            is_comm = (inter_ticket | intra_ticket) & fanout_pred
            quant_pred = (
                fx.Int32(1 if quant_enabled else 0) == fx.Int32(1)
            )
            is_producer = (
                (inter_ticket | intra_ticket)
                if const_expr(quant_two_cta_per_token)
                else quant_ticket
            ) & quant_pred
        else:
            is_inter_fanout = fx.Int32(0) == fx.Int32(1)
            is_intra_fanout = fx.Int32(0) == fx.Int32(1)
            is_comm = ticket < fx.Int32(COMM_CTAS)
            is_producer = (ticket >= fx.Int32(PRODUCER_FIRST)) & (
                ticket < fx.Int32(PRODUCER_FIRST + PRODUCER_CTAS)
            )

        # Owner initializes only counters/metadata.  Payload and generation
        # arrays are overwrite-before-publish and never need a hot-path memset.
        if is_initializer:
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            queue_heads = buffer_ops.create_buffer_resource_from_addr(
                local_addr("h1_queue_head")
            )
            for item in range(tx, LOCAL_EXPERTS, THREADS):
                buffer_ops.buffer_store(fx.Int32(0), expert_count, item)
            consumed = buffer_ops.create_buffer_resource_from_addr(
                local_addr("remote_chunk_consumed")
            )
            for item in range(
                tx,
                fx.Int32(dispatch_chunks * layout.num_qp),
                fx.Int32(THREADS),
            ):
                buffer_ops.buffer_store(fx.Int32(0), consumed, item)
            if tx < fx.Int32(work_shards):
                buffer_ops.buffer_store(fx.Int32(0), queue_heads, tx * fx.Int32(16))
            if tx == fx.Int32(0):
                for name in (
                    "tile_alloc",
                    "h1_queue_tail",
                    "h1_compute_done",
                    "h1_early_full_tiles",
                    "h1_gmm_started_before_all_comm_eos",
                    "h1_gmm_completed_before_all_comm_eos",
                ):
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(local_addr(name)),
                        fx.Int32(0),
                    )
                if const_expr(cco_geometry == "sparse_wqe"):
                    for name in (
                        "sparse_remote_consumed",
                        "sparse_remote_send_count",
                    ):
                        buffer_ops.buffer_store(
                            fx.Int32(0),
                            buffer_ops.create_buffer_resource_from_addr(
                                local_addr(name)
                            ),
                            fx.Int32(0),
                        )
                if ntokens != fx.Int32(MAX_TOKENS):
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    arena_ptr + fx.Int64(off("epoch_gate")),
                    control_generation,
                )
                comm_ops.store_i64_global_system(
                    local_addr("launch_ready"), control_generation
                )
        else:
            if tx == fx.Int32(0):
                wait_ready(
                    arena_ptr + fx.Int64(off("epoch_gate")),
                    control_generation,
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()

        # ------------------------------------------------------------------
        # One producer CTA owns one local token.  Each thread quantizes one
        # 1x32 group, so BF16->A4/E8M0 is part of this Stage-1 launch.
        # ------------------------------------------------------------------
        if is_producer:
            if const_expr(diagnostic_split_fanout):
                token = (
                    ticket & fx.Int32(127)
                    if const_expr(quant_two_cta_per_token)
                    else ticket
                )
                quant_half = (
                    ticket // fx.Int32(128)
                    if const_expr(quant_two_cta_per_token)
                    else fx.Int32(0)
                )
            else:
                token = ticket - fx.Int32(PRODUCER_FIRST)
                quant_half = fx.Int32(0)
            record_addr = local_addr("dispatch_staging") + fx.Int64(token) * fx.Int64(
                record_bytes
            )
            record_rsrc = buffer_ops.create_buffer_resource_from_addr(
                record_addr, num_records_bytes=record_bytes
            )
            x_rsrc = buffer_ops.create_buffer_resource_from_addr(x_bf16)
            group = (
                tx + quant_half * fx.Int32((HIDDEN // 32) // 2)
                if const_expr(quant_two_cta_per_token)
                else tx
            )
            group_active = (
                tx < fx.Int32((HIDDEN // 32) // 2)
                if const_expr(quant_two_cta_per_token)
                else group < fx.Int32(HIDDEN // 32)
            )
            owns_metadata = quant_half == fx.Int32(0)
            internal_quant = (
                fx.Int32(0 if prequant_input else 1) == fx.Int32(1)
            )
            external_quant = (
                fx.Int32(1 if prequant_input else 0) == fx.Int32(1)
            )
            if group_active & internal_quant:
                in_dw = token * fx.Int32(HIDDEN // 2) + group * fx.Int32(16)
                values = []
                local_max = fx.Float32(1e-10)
                for chunk in range_constexpr(4):
                    raw = buffer_ops.buffer_load(
                        x_rsrc,
                        in_dw + fx.Int32(chunk * 4),
                        vec_width=4,
                        dtype=T.i32,
                    )
                    vals = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                    local_max = local_max.maximumf(
                        fmath.absf(vals).reduce(ReductionOp.MAX)
                    )
                    for elem in range_constexpr(8):
                        values.append(vals[elem])
                working = (
                    local_max * fx.Int32(_FP4_INV_MAX_POS_BITS).bitcast(fx.Float32)
                ).bitcast(fx.Int32)
                mantissa = working & fx.Int32(0x7FFFFF)
                biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
                e8m0 = (mantissa != fx.Int32(0)).select(
                    biased_exp + fx.Int32(1), biased_exp
                )
                e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
                qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                words = []
                for word in range_constexpr(4):
                    packed = fx.Int32(0)
                    for pair in range_constexpr(4):
                        idx = word * 8 + pair * 2
                        packed = rocdl.cvt_scalef32_pk_fp4_f32(
                            T.i32,
                            packed,
                            values[idx],
                            values[idx + 1],
                            qscale,
                            pair,
                        )
                    words.append(packed)
                buffer_ops.buffer_store(
                    fx.Vector.from_elements(words, fx.Int32),
                    record_rsrc,
                    group * fx.Int32(4),
                )
                buffer_ops.buffer_store(
                    e8m0.to(fx.Int8),
                    record_rsrc,
                    fx.Int32(wire.payload_bytes) + group,
                    offset_is_bytes=True,
                )
            if group_active & external_quant:
                input_q_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    x_bf16
                )
                input_scale_rsrc = (
                    buffer_ops.create_buffer_resource_from_addr(input_scale)
                )
                input_q_dw = (
                    token * fx.Int32(HIDDEN // 8) + group * fx.Int32(4)
                )
                packed = buffer_ops.buffer_load(
                    input_q_rsrc,
                    input_q_dw,
                    vec_width=4,
                    dtype=T.i32,
                )
                buffer_ops.buffer_store(
                    packed,
                    record_rsrc,
                    group * fx.Int32(4),
                )
                input_e8m0 = buffer_ops.buffer_load(
                    input_scale_rsrc,
                    token * fx.Int32(HIDDEN // 32) + group,
                    vec_width=1,
                    dtype=T.i8,
                )
                buffer_ops.buffer_store(
                    input_e8m0,
                    record_rsrc,
                    fx.Int32(wire.payload_bytes) + group,
                    offset_is_bytes=True,
                )

            ids_rsrc = buffer_ops.create_buffer_resource_from_addr(topk_ids)
            weights_rsrc = buffer_ops.create_buffer_resource_from_addr(route_weights)
            metadata_enabled = (
                fx.Int32(
                    1 if diagnostic_phase != "quant_core_only" else 0
                )
                == fx.Int32(1)
            )
            if metadata_enabled & owns_metadata & (tx < fx.Int32(TOPK)):
                route = token * fx.Int32(TOPK) + tx
                expert = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
                weight = buffer_ops.buffer_load(
                    weights_rsrc, route, vec_width=1, dtype=T.f32
                )
                buffer_ops.buffer_store(
                    expert,
                    record_rsrc,
                    fx.Int32(wire.ids_offset // 4) + tx,
                )
                buffer_ops.buffer_store(
                    weight,
                    record_rsrc,
                    fx.Int32(wire.weights_offset // 4) + tx,
                )

            # Thread zero builds the node/rank masks used by Stage-2 and by the
            # sparse cross-node sender.  Duplicate destination ranks are legal:
            # the per-rank slot bitmaps below preserve every expert route while
            # the rank mask continues to deduplicate payload placement.
            rank_mask_scratch = fx.recast_iter(fx.Int32, lds_raw)
            rank_mask_view = fx.make_view(
                rank_mask_scratch, fx.make_layout(1, 1)
            )
            rank_mask_lane = fx.Int32(0)
            route_mask_lane = fx.Int64(0)
            route_error = fx.Int32(0)
            local_expected_lane = fx.Int32(0)
            if metadata_enabled & owns_metadata & (tx == fx.Int32(0)):
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        ids_rsrc,
                        token * fx.Int32(TOPK) + fx.Int32(slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(EXPERTS))
                    owner = valid.select(expert // fx.Int32(LOCAL_EXPERTS), fx.Int32(0))
                    bit = fx.Int32(1) << owner
                    rank_mask_lane = valid.select(rank_mask_lane | bit, rank_mask_lane)
                    route_mask_lane = valid.select(
                        route_mask_lane | (fx.Int64(1) << fx.Int64(slot)), route_mask_lane
                    )
                    owner_is_local = valid & (
                        (owner // fx.Int32(GPUS_PER_NODE)) == node
                    )
                    local_expected_lane = owner_is_local.select(
                        local_expected_lane + fx.Int32(1),
                        local_expected_lane,
                    )
                    if const_expr(cco_geometry == "sparse_wqe"):
                        invalid_non_padding = (expert < fx.Int32(-1)) | (
                            expert >= fx.Int32(EXPERTS)
                        )
                        route_error = invalid_non_padding.select(
                            route_error + fx.Int32(1), route_error
                        )
                    else:
                        route_error = valid.select(
                            route_error, route_error + fx.Int32(1)
                        )
                if route_error != fx.Int32(0):
                    comm_ops.atomic_add_system(error_addr, route_error)
                buffer_ops.buffer_store(
                    fx.Int32(rank * MAX_TOKENS) + token,
                    record_rsrc,
                    fx.Int32(wire.source_offset // 4),
                )
                buffer_ops.buffer_store(
                    fx.Int32(0),
                    record_rsrc,
                    fx.Int32(wire.source_offset // 4 + 1),
                )
                buffer_ops.buffer_store(
                    fx.Vector.from_elements([route_mask_lane], fx.Int64).bitcast(fx.Int32),
                    record_rsrc,
                    fx.Int32(wire.route_mask_offset // 4),
                )
                # Zero the record tail so validation is deterministic.
                for dword in range_constexpr(wire.raw_bytes // 4, record_dwords):
                    buffer_ops.buffer_store(fx.Int32(0), record_rsrc, fx.Int32(dword))
                mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    stage2_addr("node_dest_rank_mask")
                )
                local_mask = (rank_mask_lane >> fx.Int32(node * GPUS_PER_NODE)) & fx.Int32(0xFF)
                # This record belongs to the local aligned source rank.  The
                # reciprocal CCO receive, not this producer, owns the remote
                # source plane on this rank.
                buffer_ops.buffer_store(
                    local_mask,
                    mask_rsrc,
                    fx.Int32(node * MAX_TOKENS) + token,
                )
                if token == fx.Int32(0):
                    buffer_ops.buffer_store(
                        ntokens,
                        buffer_ops.create_buffer_resource_from_addr(
                            stage2_addr("source_token_count")
                        ),
                        fx.Int32(node),
                    )
                expected_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    stage2_addr("node_expected")
                )
                for ntile in range_constexpr(HIDDEN // 256):
                    buffer_ops.buffer_store(
                        local_expected_lane,
                        expected_rsrc,
                        fx.Int32(node * MAX_TOKENS * (HIDDEN // 256))
                        + token * fx.Int32(HIDDEN // 256)
                        + fx.Int32(ntile),
                    )
                # Broadcast rank_mask through the first LDS word after quant.
                fx.ptr_store(
                    Vec.from_elements([rank_mask_lane], fx.Int32),
                    rank_mask_scratch,
                )
            # Eight lanes each pack the 16-bit slot masks for two EP ranks.
            # A destination rank obtains its multiplicity with popcount(mask)
            # and resolves every set bit through the existing topk ID/weight
            # arrays.  This occupies bytes [3952, 3984) of the 4096-B record.
            if metadata_enabled & owns_metadata & (tx < fx.Int32(WORLD // 2)):
                rank0 = tx * fx.Int32(2)
                rank1 = rank0 + fx.Int32(1)
                slots0 = fx.Int32(0)
                slots1 = fx.Int32(0)
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        ids_rsrc,
                        token * fx.Int32(TOPK) + fx.Int32(slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    valid = (expert >= fx.Int32(0)) & (
                        expert < fx.Int32(EXPERTS)
                    )
                    owner = valid.select(
                        expert // fx.Int32(LOCAL_EXPERTS), fx.Int32(0)
                    )
                    slot_bit = fx.Int32(1 << slot)
                    slots0 = (valid & (owner == rank0)).select(
                        slots0 | slot_bit, slots0
                    )
                    slots1 = (valid & (owner == rank1)).select(
                        slots1 | slot_bit, slots1
                    )
                buffer_ops.buffer_store(
                    slots0 | (slots1 << fx.Int32(16)),
                    record_rsrc,
                    fx.Int32(wire.rank_slot_masks_offset // 4) + tx,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(cco_geometry == "sparse_wqe"):
                rank_mask = Vec(rank_mask_view.load())[0]
                remote_rank_mask = (
                    rank_mask >> fx.Int32(remote_node * GPUS_PER_NODE)
                ) & fx.Int32(0xFF)
                if (wave == fx.Int32(0)) & (
                    remote_rank_mask != fx.Int32(0)
                ):
                    qp_id = token % fx.Int32(layout.num_qp)
                    record_offset = (
                        window_off("dispatch_staging")
                        + fx.Int64(token) * fx.Int64(record_bytes)
                    )
                    put(
                        dev_comm,
                        qp_id,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_dispatch_rx")
                        + fx.Int64(token) * fx.Int64(record_bytes),
                        arena_win,
                        record_offset,
                        fx.Int64(record_bytes),
                        aggregate=True,
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                if tx == fx.Int32(0):
                    comm_ops.store_i64_global_system(
                        local_addr("sparse_remote_token_ready")
                        + fx.Int64(token) * fx.Int64(8),
                        (remote_rank_mask != fx.Int32(0)).select(
                            fx.Int64(1), fx.Int64(0)
                        ),
                    )
                # dispatch_staging_ready doubles as the cross-CTA WQE-posted
                # completion. The CCO coordinator observes it only after this
                # converged point, so postIdx reservation and descriptor stores
                # are both complete before any doorbell is rung.
                gpu.barrier()
            if const_expr(diagnostic_phase != "quant_core_only"):
                if const_expr(quant_two_cta_per_token):
                    if quant_half == fx.Int32(1):
                        if tx == fx.Int32(0):
                            comm_ops.fence_system_release()
                            comm_ops.store_i64_global_system(
                                local_addr("quant_half_done")
                                + fx.Int64(token) * fx.Int64(8),
                                generation,
                            )
                    else:
                        if tx == fx.Int32(0):
                            wait_ready(
                                local_addr("quant_half_done")
                                + fx.Int64(token) * fx.Int64(8),
                                generation,
                            )
                            comm_ops.fence_system_acquire()
                            comm_ops.fence_system_release()
                            comm_ops.store_i64_global_system(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(token) * fx.Int64(8),
                                generation,
                            )
                else:
                    comm_ops.fence_system_release()
                    if tx == fx.Int32(0):
                        comm_ops.store_i64_global_system(
                            local_addr("dispatch_staging_ready")
                            + fx.Int64(token) * fx.Int64(8),
                            generation,
                        )

        # ------------------------------------------------------------------
        # Four waves own four QPs.  Each chunk is one aggregate PUT per QP plus
        # a trailing ready value and one flush/doorbell.  The same CTA receives
        # the reciprocal chunk and performs selected-rank proxy fan-out.
        # ------------------------------------------------------------------
        if is_cco:
            qp = wave
            if const_expr(cco_geometry == "sparse_wqe"):
                if wave == fx.Int32(0):
                    # One wave serially owns all four doorbells. Producer CTAs
                    # may reserve/fill WQEs concurrently, but Ionic QPs share a
                    # doorbell mapping and must not be flushed concurrently.
                    for stream_qp in range_constexpr(layout.num_qp):
                        # The first 32 lanes wait for one producer each, then
                        # ballot their local send decisions into the terminal
                        # bitmap.  Publish and flush this QP immediately so its
                        # destination fanout can start while the following QPs
                        # are still waiting for producers.
                        token_flag = fx.Int32(0)
                        if lane < fx.Int32(MAX_TOKENS // layout.num_qp):
                            source_token = (
                                fx.Int32(stream_qp)
                                + lane * fx.Int32(layout.num_qp)
                            )
                            wait_ready(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(source_token) * fx.Int64(8),
                                generation,
                            )
                            token_flag = fx.Int32(
                                comm_ops.load_i64_global_system(
                                    local_addr("sparse_remote_token_ready")
                                    + fx.Int64(source_token) * fx.Int64(8)
                                )
                            )
                        comm_ops.fence_system_acquire()
                        token_mask = rocdl.ballot(
                            T.i64,
                            (lane < fx.Int32(MAX_TOKENS // layout.num_qp))
                            & (token_flag != fx.Int32(0)),
                        )
                        terminal_ready = (
                            generation
                            << fx.Int64(SPARSE_QP_GENERATION_SHIFT)
                        ) | (
                            token_mask
                            & fx.Int64(
                                (1 << SPARSE_QP_TOKEN_BITS) - 1
                            )
                        )
                        put_value(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            arena_win,
                            window_off("sparse_remote_qp_ready")
                            + fx.Int64(stream_qp * 8),
                            terminal_ready,
                            aggregate=True,
                            scope="warp",
                            team=TEAM_RAIL,
                        )
                        request = flush_async(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            scope="warp",
                            team=TEAM_RAIL,
                        )
                        if lane == fx.Int32(0):
                            comm_ops.store_i64_global_system(
                                local_addr("sparse_remote_request")
                                + fx.Int64(stream_qp * 8),
                                request,
                            )
                    for ready_qp in range_constexpr(layout.num_qp):
                        if lane == fx.Int32(0):
                            wait_ready(
                                local_addr("sparse_remote_qp_ready")
                                + fx.Int64(ready_qp * 8),
                                generation
                                << fx.Int64(SPARSE_QP_GENERATION_SHIFT),
                            )
                        comm_ops.fence_system_acquire()
                gpu.barrier()
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        local_addr("sparse_remote_batch_ready"), generation
                    )
                gpu.barrier()
            if const_expr(cco_geometry == "mori64x2"):
                active_rail = wave < fx.Int32(2)
                if active_rail:
                    half = wave
                    # One lane owns one token-ready word; wave reconvergence
                    # proves all 64 contiguous records are packed.
                    token = half * fx.Int32(64) + lane
                    wait_ready(
                        local_addr("dispatch_staging_ready")
                        + fx.Int64(token) * fx.Int64(8),
                        generation,
                    )
                    comm_ops.fence_system_acquire()
                    src_byte = (
                        window_off("dispatch_staging")
                        + fx.Int64(half) * fx.Int64(64 * record_bytes)
                    )
                    dst_byte = (
                        window_off("remote_dispatch_rx")
                        + fx.Int64(half) * fx.Int64(64 * record_bytes)
                    )
                    put(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        dst_byte,
                        arena_win,
                        src_byte,
                        fx.Int64(64 * record_bytes),
                        aggregate=True,
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_chunk_ready")
                        + fx.Int64(half) * fx.Int64(8),
                        generation,
                        aggregate=True,
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    request = flush_async(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    if lane == fx.Int32(0):
                        comm_ops.store_i64_global_system(
                            local_addr("remote_chunk_request")
                            + fx.Int64(half) * fx.Int64(8),
                            request,
                        )
                        wait_ready(
                            local_addr("remote_chunk_ready")
                            + fx.Int64(half) * fx.Int64(8),
                            generation,
                        )
                    comm_ops.fence_system_acquire()
                # wave2/3 are intentionally idle; this is the sole CTA-wide
                # rendezvous after both independent half-record transfers.
                gpu.barrier()

            cco_batch_count = (
                dispatch_chunks // cco_chunks_per_flush
                if cco_geometry == "chunked"
                else 0
            )
            for batch in range_constexpr(
                cco_batch_count
            ):
                batch_first = batch * cco_chunks_per_flush
                for batch_item in range_constexpr(cco_chunks_per_flush):
                    chunk = batch_first + batch_item
                    first_token = fx.Int32(chunk * records_per_chunk)
                    # All four waves wait for their four source records.
                    for item in range_constexpr(records_per_qp):
                        token = (
                            first_token
                            + qp * fx.Int32(records_per_qp)
                            + fx.Int32(item)
                        )
                        if lane == fx.Int32(0):
                            wait_ready(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(token) * fx.Int64(8),
                                generation,
                            )
                    gpu.barrier()
                    src_byte = (
                        window_off("dispatch_staging")
                        + fx.Int64(
                            first_token + qp * fx.Int32(records_per_qp)
                        )
                        * fx.Int64(record_bytes)
                    )
                    dst_byte = (
                        window_off("remote_dispatch_rx")
                        + fx.Int64(
                            first_token + qp * fx.Int32(records_per_qp)
                        )
                        * fx.Int64(record_bytes)
                    )
                    if const_expr(enable_cco):
                        put(
                            dev_comm,
                            qp,
                            fx.Int32(remote_node),
                            arena_win,
                            dst_byte,
                            arena_win,
                            src_byte,
                            fx.Int64(qp_bytes),
                            aggregate=True,
                            scope="warp",
                            team=TEAM_RAIL,
                        )
                        ready_byte = (
                            window_off("remote_chunk_ready")
                            + (
                                fx.Int64(chunk * layout.num_qp)
                                + fx.Int64(qp)
                            )
                            * fx.Int64(8)
                        )
                        put_value(
                            dev_comm,
                            qp,
                            fx.Int32(remote_node),
                            arena_win,
                            ready_byte,
                            generation,
                            aggregate=True,
                            scope="warp",
                            team=TEAM_RAIL,
                        )
                    else:
                        # EP8 single-node bring-up: treat this rank's staging
                        # slab as the aligned remote source.
                        src_local = buffer_ops.create_buffer_resource_from_addr(
                            local_addr("dispatch_staging")
                        )
                        dst_local = buffer_ops.create_buffer_resource_from_addr(
                            local_addr("remote_dispatch_rx")
                        )
                        base_dw = (
                            first_token + qp * fx.Int32(records_per_qp)
                        ) * fx.Int32(record_dwords)
                        for dword in range(
                            base_dw + lane * fx.Int32(4),
                            base_dw
                            + fx.Int32(records_per_qp * record_dwords),
                            fx.Int32(64 * 4),
                        ):
                            value = buffer_ops.buffer_load(
                                src_local,
                                dword,
                                vec_width=4,
                                dtype=T.i32,
                            )
                            buffer_ops.buffer_store(value, dst_local, dword)
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        if lane == fx.Int32(0):
                            comm_ops.store_i64_global_system(
                                local_addr("remote_chunk_ready")
                                + (
                                    fx.Int64(chunk * layout.num_qp)
                                    + fx.Int64(qp)
                                )
                                * fx.Int64(8),
                                generation,
                            )

                if const_expr(enable_cco):
                    request = flush_async(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    if lane == fx.Int32(0):
                        request_index = (
                            fx.Int64(batch_first * layout.num_qp)
                            + fx.Int64(qp)
                        )
                        comm_ops.store_i64_global_system(
                            local_addr("remote_chunk_request")
                            + request_index * fx.Int64(8),
                            request,
                        )

                # The batch is visible remotely after one doorbell; acquire
                # every reciprocal ready word before proxy fanout consumes it.
                for batch_item in range_constexpr(cco_chunks_per_flush):
                    chunk = batch_first + batch_item
                    if lane == fx.Int32(0):
                        wait_ready(
                            local_addr("remote_chunk_ready")
                            + (
                                fx.Int64(chunk * layout.num_qp)
                                + fx.Int64(qp)
                            )
                            * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    gpu.barrier()

            if lane == fx.Int32(0) and qp == fx.Int32(0):
                buffer_ops.buffer_store(
                    ntokens,
                    buffer_ops.create_buffer_resource_from_addr(
                        stage2_addr("source_token_count")
                    ),
                    fx.Int32(remote_node),
                )
            remote_expected = buffer_ops.create_buffer_resource_from_addr(
                stage2_addr("node_expected")
            )
            remote_masks = buffer_ops.create_buffer_resource_from_addr(
                stage2_addr("node_dest_rank_mask")
            )
            if tx < fx.Int32(MAX_TOKENS):
                remote_record_available = fx.Int32(1)
                if const_expr(cco_geometry == "sparse_wqe"):
                    ready_qp = tx % fx.Int32(layout.num_qp)
                    ready_bit = tx // fx.Int32(layout.num_qp)
                    terminal_ready = fx.Int64(
                        comm_ops.load_i64_global_system(
                            local_addr("sparse_remote_qp_ready")
                            + fx.Int64(ready_qp) * fx.Int64(8)
                        )
                    )
                    mask_bit_set = (
                        (
                            (
                                terminal_ready & fx.Int64(0xFFFFFFFF)
                            )
                            >> fx.Int64(ready_bit)
                        )
                        & fx.Int64(1)
                    ) != fx.Int64(0)
                    token_ready = (
                        (
                            terminal_ready
                            >> fx.Int64(SPARSE_QP_GENERATION_SHIFT)
                        )
                        >= generation
                    ) & mask_bit_set
                    remote_record_available = token_ready.select(
                        fx.Int32(1), fx.Int32(0)
                    )
                remote_mask = fx.Int32(0)
                remote_route_count = fx.Int32(0)
                if remote_record_available != fx.Int32(0):
                    remote_record = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("remote_dispatch_rx")
                        + fx.Int64(tx) * fx.Int64(record_bytes),
                        num_records_bytes=record_bytes,
                    )
                    for slot in range_constexpr(TOPK):
                        expert = buffer_ops.buffer_load(
                            remote_record,
                            fx.Int32(wire.ids_offset // 4 + slot),
                            vec_width=1,
                            dtype=T.i32,
                        )
                        valid = (expert >= fx.Int32(0)) & (
                            expert < fx.Int32(EXPERTS)
                        )
                        owner = valid.select(
                            expert // fx.Int32(LOCAL_EXPERTS),
                            fx.Int32(0),
                        )
                        on_node = valid & (
                            (owner // fx.Int32(GPUS_PER_NODE))
                            == fx.Int32(node)
                        )
                        remote_mask = on_node.select(
                            remote_mask
                            | (
                                fx.Int32(1)
                                << (owner % fx.Int32(GPUS_PER_NODE))
                            ),
                            remote_mask,
                        )
                        remote_route_count = on_node.select(
                            remote_route_count + fx.Int32(1),
                            remote_route_count,
                        )
                buffer_ops.buffer_store(
                    remote_mask,
                    remote_masks,
                    fx.Int32(remote_node * MAX_TOKENS) + tx,
                )
                for ntile in range_constexpr(HIDDEN // 256):
                    buffer_ops.buffer_store(
                        remote_route_count,
                        remote_expected,
                        fx.Int32(
                            remote_node * MAX_TOKENS * (HIDDEN // 256)
                        )
                        + tx * fx.Int32(HIDDEN // 256)
                        + fx.Int32(ntile),
                    )

        # ------------------------------------------------------------------
        # Split fanout CTAs cover the eight node-local destinations. Each writes
        # both its local source rank and the aligned remote source rank directly
        # into destination expert tiles; there is no rank inbox/sort.
        # ------------------------------------------------------------------
        if const_expr(diagnostic_split_fanout):
            split_active = is_inter_fanout | is_intra_fanout
            split_worker_raw = is_inter_fanout.select(
                ticket, ticket - fx.Int32(128)
            )
            split_worker = split_active.select(
                split_worker_raw, fx.Int32(0)
            )
            split_dest = split_worker % fx.Int32(GPUS_PER_NODE)
            split_group = split_worker // fx.Int32(GPUS_PER_NODE)
            split_shard = split_active.select(
                split_group, fx.Int32(MAX_TOKENS)
            )
            is_finisher = (
                is_intra_fanout
                & (split_dest == fx.Int32(local_rank))
                & (split_group == fx.Int32(0))
            )
        else:
            split_worker = fx.Int32(0)
            split_dest = fx.Int32(0)
            split_group = fx.Int32(0)
            split_shard = fx.Int32(0)
            is_finisher = is_comm & (ticket == fx.Int32(local_rank))

        def _peer_addr(dest, name):
            return fx.Int64(arena_window.lsa_ptr(dest, window_off(name)))

        def _enqueue_tile_jobs(dest, physical, early_full_tile):
            """Append one ready BM32 tile as one contiguous 24-job batch.

            This helper is called by exactly one thread: either the unique
            ``tile_row_done`` last-arriver for a full tile or one finisher
            thread for an EOS-sealed partial tile.  ``h1_queue_tail`` reserves
            space only; the generation word at the first slot of each batch is
            the release publication consumed by compute CTAs.
            """

            base = fx.Int32(
                comm_ops.atomic_add_system_acq_rel(
                    _peer_addr(dest, "h1_queue_tail"),
                    fx.Int32(h1_n_blocks),
                )
            )
            in_bounds = base <= fx.Int32(max_jobs - h1_n_blocks)
            if in_bounds:
                queue = buffer_ops.create_buffer_resource_from_addr(
                    _peer_addr(dest, "h1_ready_queue")
                )
                for nblock in range_constexpr(h1_n_blocks):
                    job = physical * fx.Int32(h1_n_blocks) + fx.Int32(nblock)
                    buffer_ops.buffer_store(
                        job,
                        queue,
                        base + fx.Int32(nblock),
                    )
                rocdl.s_waitcnt(0)
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    _peer_addr(dest, "h1_ready_queue_generation")
                    + fx.Int64(base) * fx.Int64(8),
                    generation,
                )
                if const_expr(tile_pipeline_instrument):
                    if early_full_tile:
                        comm_ops.atomic_add_system_acq_rel(
                            _peer_addr(dest, "h1_early_full_tiles"),
                            fx.Int32(1),
                        )
            else:
                comm_ops.atomic_add_system(error_addr, fx.Int32(1))

        def _all_comm_eos_seen():
            all_ready = fx.Int32(1)
            for peer in range_constexpr(GPUS_PER_NODE):
                observed = fx.Int64(
                    comm_ops.load_i64_global_system(
                        local_addr("comm_eos") + fx.Int64(peer * 8)
                    )
                )
                all_ready = (observed >= generation).select(
                    all_ready, fx.Int32(0)
                )
            return all_ready

        def _record_dest_slot_mask(record, dest):
            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            predicate_scratch = fx.recast_iter(fx.Int32, lds_raw)
            predicate_view = fx.make_view(
                predicate_scratch, fx.make_layout(1, 1)
            )
            if tx == fx.Int32(0):
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                packed_masks = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.rank_slot_masks_offset // 4)
                    + dest_global // fx.Int32(2),
                    vec_width=1,
                    dtype=T.i32,
                )
                shift = (dest_global & fx.Int32(1)) * fx.Int32(16)
                slots = (packed_masks >> shift) & fx.Int32(0xFFFF)
                fx.ptr_store(
                    Vec.from_elements([slots], fx.Int32),
                    predicate_scratch,
                )
            gpu.barrier()
            return Vec(predicate_view.load())[0]

        def _record_targets_dest(record, dest):
            return _record_dest_slot_mask(record, dest) != fx.Int32(0)

        def _dispatch_route(record, dest, source_index, topk_slot):
            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            scratch = fx.recast_iter(fx.Int32, lds_raw)
            scratch_view = fx.make_view(scratch, fx.make_layout(1, 1))
            if tx == fx.Int32(0):
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                expert = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.ids_offset // 4) + topk_slot,
                    vec_width=1,
                    dtype=T.i32,
                )
                valid = (expert >= dest_global * fx.Int32(LOCAL_EXPERTS)) & (
                    expert
                    < (dest_global + fx.Int32(1)) * fx.Int32(LOCAL_EXPERTS)
                )
                invalid = (expert < dest_global * fx.Int32(LOCAL_EXPERTS)) | (
                    expert
                    >= (dest_global + fx.Int32(1))
                    * fx.Int32(LOCAL_EXPERTS)
                )
                if invalid:
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                fx.ptr_store(
                    Vec.from_elements(
                        [
                            valid.select(
                                expert
                                - dest_global * fx.Int32(LOCAL_EXPERTS),
                                fx.Int32(0),
                            )
                        ],
                        fx.Int32,
                    ),
                    scratch,
                )
            gpu.barrier()
            local_expert = Vec(scratch_view.load())[0]
            expert_count_addr = _peer_addr(dest, "expert_count")
            if tx == fx.Int32(0):
                row_slot_lane = fx.Int32(
                    comm_ops.atomic_add_system(
                        expert_count_addr + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
                fx.ptr_store(Vec.from_elements([row_slot_lane], fx.Int32), scratch)
            gpu.barrier()
            row_slot = Vec(scratch_view.load())[0]
            logical_tile = row_slot // fx.Int32(BM)
            row_in_tile = row_slot % fx.Int32(BM)
            map_index = local_expert * fx.Int32(max_tiles_per_expert) + logical_tile
            map_ready = _peer_addr(dest, "expert_tile_map_ready") + fx.Int64(map_index) * 8
            if row_in_tile == fx.Int32(0):
                if tx == fx.Int32(0):
                    physical = fx.Int32(
                        comm_ops.atomic_add_system(
                            _peer_addr(dest, "tile_alloc"), fx.Int32(1)
                        )
                    )
                    if physical >= fx.Int32(max_route_tiles):
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                        physical = fx.Int32(0)
                    buffer_ops.buffer_store(
                        physical,
                        buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "expert_tile_map")),
                        map_index,
                    )
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_row_done")),
                        physical,
                    )
                    buffer_ops.buffer_store(
                        local_expert,
                        buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_expert")),
                        physical,
                    )
                    buffer_ops.buffer_store(
                        physical * fx.Int32(BM),
                        buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_row_base")),
                        physical,
                    )
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(map_ready, generation)
                    fx.ptr_store(Vec.from_elements([physical], fx.Int32), scratch)
            else:
                if tx == fx.Int32(0):
                    wait_ready(map_ready, generation)
                    physical = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "expert_tile_map")),
                        map_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    fx.ptr_store(Vec.from_elements([physical], fx.Int32), scratch)
            gpu.barrier()
            physical = Vec(scratch_view.load())[0]
            grouped_row = physical * fx.Int32(BM) + row_in_tile
            if tx < fx.Int32(scale_bytes):
                scale = buffer_ops.buffer_load(
                    rec, fx.Int32(wire.payload_bytes) + tx, vec_width=1, dtype=T.i8
                )
                # Exact BM32 A-scale preshuffle consumed by
                # package-local gemm1.issue_a_scale_load():
                # (ku, ikxdl, k_lane, n_lane, im_a).
                ku = tx // fx.Int32(8)
                ikxdl = (tx % fx.Int32(8)) // fx.Int32(4)
                k_lane = tx % fx.Int32(4)
                im_a = row_in_tile // fx.Int32(16)
                n_lane = row_in_tile % fx.Int32(16)
                dst_dword = (
                    physical * fx.Int32(scale_dwords * BM)
                    + ku * fx.Int32(64)
                    + k_lane * fx.Int32(16)
                    + n_lane
                )
                dst_byte = dst_dword * fx.Int32(4) + ikxdl * fx.Int32(2) + im_a
                buffer_ops.buffer_store(
                    scale,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "grouped_input_scale")
                    ),
                    dst_byte,
                    offset_is_bytes=True,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                weight = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.weights_offset // 4) + topk_slot,
                    vec_width=1,
                    dtype=T.f32,
                )
                source_encoding = source_index | (topk_slot << fx.Int32(24))
                buffer_ops.buffer_store(
                    source_index,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_input")
                    ),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    source_encoding,
                    buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_row_source")),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    weight,
                    buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_row_weight")),
                    grouped_row,
                )
                comm_ops.fence_system_release()
                if const_expr(not diagnostic_no_arrival_rmw):
                    completed = fx.Int32(
                        comm_ops.atomic_add_system_acq_rel(
                            _peer_addr(dest, "tile_row_done")
                            + fx.Int64(physical) * 4,
                            fx.Int32(1),
                        )
                    )
                    if const_expr(tile_pipeline):
                        # The acq_rel RMW chain makes the unique last arriver
                        # observe all 32 row payload/scale/metadata releases.
                        # It can therefore publish this full tile immediately,
                        # before the eight communication roles reach EOS.
                        if completed == fx.Int32(BM - 1):
                            _enqueue_tile_jobs(dest, physical, True)
                    else:
                        # The post-EOS reference publishes every physical tile
                        # only after all eight communication roles complete.
                        _ = completed
            gpu.barrier()

        def _dispatch_record(record, dest, source_index):
            # The record is deduplicated at node/rank granularity, but every
            # matching top-k slot remains a distinct expert contribution.
            # The mask is uniform across the CTA, so barriers in
            # _dispatch_route are reached by all threads for every set bit.
            route_slots = _record_dest_slot_mask(record, dest)
            if route_slots != fx.Int32(0):
                # Second-level payload deduplication: one quantized activation
                # copy per (source token, destination rank), irrespective of
                # how many local experts that rank owns in the token's Top-K.
                # Route rows below point back to this fixed source-indexed row.
                rec = buffer_ops.create_buffer_resource_from_addr(
                    record, num_records_bytes=record_bytes
                )
                dst_payload = buffer_ops.create_buffer_resource_from_addr(
                    _peer_addr(dest, "grouped_input_q")
                    + fx.Int64(source_index) * fx.Int64(wire.payload_bytes),
                    num_records_bytes=wire.payload_bytes,
                )
                for dword in range(
                    tx * fx.Int32(4),
                    payload_dwords,
                    fx.Int32(THREADS * 4),
                ):
                    value = buffer_ops.buffer_load(
                        rec, dword, vec_width=4, dtype=T.i32
                    )
                    buffer_ops.buffer_store(value, dst_payload, dword)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                for topk_slot in range(
                    fx.Int32(0), fx.Int32(TOPK), fx.Int32(1)
                ):
                    if (
                        (route_slots >> topk_slot) & fx.Int32(1)
                    ) != fx.Int32(0):
                        _dispatch_route(
                            record, dest, source_index, topk_slot
                        )

        def _dispatch_record_wave(record, dest, source_index):
            """Place one record with one independent 64-lane wave.

            The legacy split probe assigns a whole 256-thread CTA to a token.
            This diagnostic body keeps the exact destination ABI and scoreboard
            protocol while replacing CTA scratch/barriers with lane-0 values
            broadcast through ``readlane``.  Four waves can consequently place
            four unrelated records at the same time.
            """

            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            packed_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                found = fx.Int32(0)
                found_slot = fx.Int32(0)
                found_expert = fx.Int32(0)
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        rec,
                        fx.Int32(wire.ids_offset // 4 + slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    valid = (
                        expert >= dest_global * fx.Int32(LOCAL_EXPERTS)
                    ) & (
                        expert
                        < (dest_global + fx.Int32(1))
                        * fx.Int32(LOCAL_EXPERTS)
                    )
                    if valid & (found != fx.Int32(0)):
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                    take = valid & (found == fx.Int32(0))
                    found_slot = take.select(fx.Int32(slot), found_slot)
                    found_expert = take.select(
                        expert - dest_global * fx.Int32(LOCAL_EXPERTS),
                        found_expert,
                    )
                    found = valid.select(fx.Int32(1), found)
                if found == fx.Int32(0):
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                packed_lane0 = found_expert | (found_slot << fx.Int32(8))
            packed = fx.Int32(rocdl.readlane(T.i32, packed_lane0, 0))
            local_expert = packed & fx.Int32(0xFF)
            topk_slot = (packed >> fx.Int32(8)) & fx.Int32(0xFF)

            row_slot_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                row_slot_lane0 = fx.Int32(
                    comm_ops.atomic_add_system(
                        _peer_addr(dest, "expert_count")
                        + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
            row_slot = fx.Int32(rocdl.readlane(T.i32, row_slot_lane0, 0))
            logical_tile = row_slot // fx.Int32(BM)
            row_in_tile = row_slot % fx.Int32(BM)
            map_index = (
                local_expert * fx.Int32(max_tiles_per_expert) + logical_tile
            )
            map_ready = (
                _peer_addr(dest, "expert_tile_map_ready")
                + fx.Int64(map_index) * 8
            )

            physical_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                if row_in_tile == fx.Int32(0):
                    physical_lane0 = fx.Int32(
                        comm_ops.atomic_add_system(
                            _peer_addr(dest, "tile_alloc"), fx.Int32(1)
                        )
                    )
                    if physical_lane0 >= fx.Int32(max_route_tiles):
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                        physical_lane0 = fx.Int32(0)
                    buffer_ops.buffer_store(
                        physical_lane0,
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "expert_tile_map")
                        ),
                        map_index,
                    )
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_done")
                        ),
                        physical_lane0,
                    )
                    buffer_ops.buffer_store(
                        local_expert,
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_expert")
                        ),
                        physical_lane0,
                    )
                    buffer_ops.buffer_store(
                        physical_lane0 * fx.Int32(BM),
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_base")
                        ),
                        physical_lane0,
                    )
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(map_ready, generation)
                else:
                    wait_ready(map_ready, generation)
                    physical_lane0 = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "expert_tile_map")
                        ),
                        map_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
            physical = fx.Int32(
                rocdl.readlane(T.i32, physical_lane0, 0)
            )
            grouped_row = physical * fx.Int32(BM) + row_in_tile

            dst_payload = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "grouped_input_q")
                + fx.Int64(source_index) * fx.Int64(wire.payload_bytes),
                num_records_bytes=wire.payload_bytes,
            )
            for dword in range(
                lane * fx.Int32(4),
                payload_dwords,
                fx.Int32(64 * 4),
            ):
                value = buffer_ops.buffer_load(
                    rec, dword, vec_width=4, dtype=T.i32
                )
                buffer_ops.buffer_store(value, dst_payload, dword)

            for scale_index in range(
                lane, fx.Int32(scale_bytes), fx.Int32(64)
            ):
                scale = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.payload_bytes) + scale_index,
                    vec_width=1,
                    dtype=T.i8,
                )
                ku = scale_index // fx.Int32(8)
                ikxdl = (scale_index % fx.Int32(8)) // fx.Int32(4)
                k_lane = scale_index % fx.Int32(4)
                im_a = row_in_tile // fx.Int32(16)
                n_lane = row_in_tile % fx.Int32(16)
                dst_dword = (
                    physical * fx.Int32(scale_dwords * BM)
                    + ku * fx.Int32(64)
                    + k_lane * fx.Int32(16)
                    + n_lane
                )
                dst_byte = (
                    dst_dword * fx.Int32(4)
                    + ikxdl * fx.Int32(2)
                    + im_a
                )
                buffer_ops.buffer_store(
                    scale,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "grouped_input_scale")
                    ),
                    dst_byte,
                    offset_is_bytes=True,
                )
            rocdl.s_waitcnt(0)

            if lane == fx.Int32(0):
                weight = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.weights_offset // 4) + topk_slot,
                    vec_width=1,
                    dtype=T.f32,
                )
                source_encoding = source_index | (
                    topk_slot << fx.Int32(24)
                )
                buffer_ops.buffer_store(
                    source_index,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_input")
                    ),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    source_encoding,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_source")
                    ),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    weight,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_weight")
                    ),
                    grouped_row,
                )
                comm_ops.fence_system_release()
                if const_expr(not diagnostic_no_arrival_rmw):
                    completed = fx.Int32(
                        comm_ops.atomic_add_system_acq_rel(
                            _peer_addr(dest, "tile_row_done")
                            + fx.Int64(physical) * 4,
                            fx.Int32(1),
                        )
                    )
                    _ = completed

        dest = fx.Int32(0)
        shard = fx.Int32(0)
        if const_expr(diagnostic_split_fanout and fanout_enabled):
            dest = split_dest
            shard = split_shard
            if is_comm:
                if tx == fx.Int32(0):
                    wait_ready(
                        _peer_addr(dest, "launch_ready"), control_generation
                    )
            gpu.barrier()
            sparse_qp_token_mask = fx.Int64(0)
            if const_expr(cco_geometry == "sparse_wqe"):
                if is_inter_fanout:
                    qp_ready_scratch = fx.recast_iter(fx.Int64, lds_raw)
                    qp_ready_view = fx.make_view(
                        qp_ready_scratch, fx.make_layout(1, 1)
                    )
                    if tx == fx.Int32(0):
                        ready_qp = shard % fx.Int32(layout.num_qp)
                        observed = fx.Int64(
                            wait_ready(
                                local_addr("sparse_remote_qp_ready")
                                + fx.Int64(ready_qp) * fx.Int64(8),
                                generation
                                << fx.Int64(SPARSE_QP_GENERATION_SHIFT),
                            )
                        )
                        fx.ptr_store(
                            Vec.from_elements([observed], fx.Int64),
                            qp_ready_scratch,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    sparse_qp_token_mask = (
                        Vec(qp_ready_view.load())[0]
                        & fx.Int64(0xFFFFFFFF)
                    )
            completion_slot = is_inter_fanout.select(
                shard, fx.Int32(split_fanout_shards) + shard
            )

            if is_inter_fanout:
                if const_expr(diagnostic_wave_fanout):
                    if wave == fx.Int32(0):
                        for token in range(
                            shard,
                            fx.Int32(MAX_TOKENS),
                            fx.Int32(split_fanout_shards),
                        ):
                            if const_expr(cco_geometry == "mori64x2"):
                                ready_index = token // fx.Int32(64)
                            else:
                                chunk = token // records_per_chunk
                                qp_id = (
                                    token % records_per_chunk
                                ) // records_per_qp
                                ready_index = (
                                    chunk * fx.Int32(layout.num_qp) + qp_id
                                )
                            if lane == fx.Int32(0):
                                wait_ready(
                                    local_addr("remote_chunk_ready")
                                    + fx.Int64(ready_index) * fx.Int64(8),
                                    generation,
                                )
                            comm_ops.fence_system_acquire()
                            _dispatch_record_wave(
                                local_addr("remote_dispatch_rx")
                                + fx.Int64(token * record_bytes),
                                dest,
                                fx.Int32(
                                    remote_source_rank * MAX_TOKENS + token
                                ),
                            )
                else:
                    for token in range(
                        shard,
                        fx.Int32(MAX_TOKENS),
                        fx.Int32(split_fanout_shards),
                    ):
                        remote_record = (
                            local_addr("remote_dispatch_rx")
                            + fx.Int64(token * record_bytes)
                        )
                        if const_expr(cco_geometry == "sparse_wqe"):
                            ready_bit = token // fx.Int32(layout.num_qp)
                            if (
                                (sparse_qp_token_mask >> fx.Int64(ready_bit))
                                & fx.Int64(1)
                            ) != fx.Int64(0):
                                if _record_targets_dest(remote_record, dest):
                                    _dispatch_record(
                                        remote_record,
                                        dest,
                                        fx.Int32(
                                            remote_source_rank * MAX_TOKENS
                                            + token
                                        ),
                                    )
                        else:
                            if const_expr(cco_geometry == "mori64x2"):
                                ready_index = token // fx.Int32(64)
                            else:
                                chunk = token // records_per_chunk
                                qp_id = (
                                    token % records_per_chunk
                                ) // records_per_qp
                                ready_index = (
                                    chunk * fx.Int32(layout.num_qp) + qp_id
                                )
                            if tx == fx.Int32(0):
                                wait_ready(
                                    local_addr("remote_chunk_ready")
                                    + fx.Int64(ready_index) * fx.Int64(8),
                                    generation,
                                )
                            gpu.barrier()
                            _dispatch_record(
                                remote_record,
                                dest,
                                fx.Int32(
                                    remote_source_rank * MAX_TOKENS + token
                                ),
                            )
            if is_intra_fanout:
                if const_expr(diagnostic_wave_fanout):
                    if wave == fx.Int32(0):
                        for token in range(
                            shard,
                            fx.Int32(MAX_TOKENS),
                            fx.Int32(split_fanout_shards),
                        ):
                            if lane == fx.Int32(0):
                                wait_ready(
                                    local_addr("dispatch_staging_ready")
                                    + fx.Int64(token) * 8,
                                    generation,
                                )
                            comm_ops.fence_system_acquire()
                            _dispatch_record_wave(
                                local_addr("dispatch_staging")
                                + fx.Int64(token * record_bytes),
                                dest,
                                fx.Int32(rank * MAX_TOKENS + token),
                            )
                else:
                    for token in range(
                        shard,
                        fx.Int32(MAX_TOKENS),
                        fx.Int32(split_fanout_shards),
                    ):
                        if tx == fx.Int32(0):
                            wait_ready(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(token) * 8,
                                generation,
                            )
                        gpu.barrier()
                        local_record = (
                            local_addr("dispatch_staging")
                            + fx.Int64(token * record_bytes)
                        )
                        if const_expr(cco_geometry == "sparse_wqe"):
                            if _record_targets_dest(local_record, dest):
                                _dispatch_record(
                                    local_record,
                                    dest,
                                    fx.Int32(rank * MAX_TOKENS + token),
                                )
                        else:
                            _dispatch_record(
                                local_record,
                                dest,
                                fx.Int32(rank * MAX_TOKENS + token),
                            )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(diagnostic_wave_fanout):
                if is_comm & (tx == fx.Int32(0)):
                    comm_ops.fence_system_release()
                    flag_index = dest * fx.Int32(32) + completion_slot
                    comm_ops.store_i64_global_system(
                        local_addr("fanout_done")
                        + fx.Int64(flag_index) * fx.Int64(8),
                        generation,
                    )
            else:
                if is_comm & (tx == fx.Int32(0)):
                    comm_ops.fence_system_release()
                    flag_index = dest * fx.Int32(32) + completion_slot
                    comm_ops.store_i64_global_system(
                        local_addr("fanout_done")
                        + fx.Int64(flag_index) * fx.Int64(8),
                        generation,
                    )
            gpu.barrier()

            # One local-domain coordinator per destination turns all unique
            # producer flags into the legacy one-EOS/one-consumed contribution.
            is_dest_coordinator = is_intra_fanout & (
                split_group == fx.Int32(0)
            )
            if is_dest_coordinator:
                if tx == fx.Int32(0):
                    for producer in range_constexpr(
                        2 * split_fanout_shards
                    ):
                        flag_index = dest * fx.Int32(32) + fx.Int32(producer)
                        wait_ready(
                            local_addr("fanout_done")
                            + fx.Int64(flag_index) * fx.Int64(8),
                            generation,
                        )
                    comm_ops.fence_system_acquire()
                    if const_expr(cco_geometry == "sparse_wqe"):
                        comm_ops.atomic_add_system_acq_rel(
                            local_addr("sparse_remote_consumed"),
                            fx.Int32(1),
                        )
                    else:
                        consumed_words = (
                            2
                            if cco_geometry == "mori64x2"
                            else dispatch_chunks * layout.num_qp
                        )
                        for consume_index in range_constexpr(
                            consumed_words
                        ):
                            comm_ops.atomic_add_system_acq_rel(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(consume_index * 4),
                                fx.Int32(1),
                            )
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        _peer_addr(dest, "comm_eos")
                        + fx.Int64(local_rank) * 8,
                        generation,
                    )
                gpu.barrier()
        else:
            if is_comm:
                dest = ticket
                if tx == fx.Int32(0):
                    wait_ready(
                        _peer_addr(dest, "launch_ready"), control_generation
                    )
                gpu.barrier()
                for token in range(fx.Int32(0), fx.Int32(MAX_TOKENS), 1):
                    if tx == fx.Int32(0):
                        wait_ready(
                            local_addr("dispatch_staging_ready")
                            + fx.Int64(token) * 8,
                            generation,
                        )
                    gpu.barrier()
                    _dispatch_record(
                        local_addr("dispatch_staging")
                        + fx.Int64(token * record_bytes),
                        dest,
                        fx.Int32(rank * MAX_TOKENS + token),
                    )
                for token in range(fx.Int32(0), fx.Int32(MAX_TOKENS), 1):
                    chunk = token // records_per_chunk
                    qp_id = (token % records_per_chunk) // records_per_qp
                    if tx == fx.Int32(0):
                        wait_ready(
                            local_addr("remote_chunk_ready")
                            + fx.Int64((chunk * layout.num_qp + qp_id) * 8),
                            generation,
                        )
                    gpu.barrier()
                    _dispatch_record(
                        local_addr("remote_dispatch_rx")
                        + fx.Int64(token * record_bytes),
                        dest,
                        fx.Int32(remote_source_rank * MAX_TOKENS + token),
                    )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    for consume_index in range_constexpr(
                        dispatch_chunks * layout.num_qp
                    ):
                        comm_ops.atomic_add_system_acq_rel(
                            local_addr("remote_chunk_consumed")
                            + fx.Int64(consume_index * 4),
                            fx.Int32(1),
                        )
                    comm_ops.store_i64_global_system(
                        _peer_addr(dest, "comm_eos")
                        + fx.Int64(local_rank) * 8,
                        generation,
                    )

        # The CCO CTA is also destination role zero, so delayed-credit progress
        # must run only after the common is_comm path above has contributed its
        # own consumed count. Otherwise the scoreboard can reach only seven.
        if const_expr(diagnostic_phase == "transport_only"):
            if is_cco:
                consumed = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("remote_chunk_consumed")
                )
                consumed_words = (
                    2
                    if cco_geometry == "mori64x2"
                    else dispatch_chunks * layout.num_qp
                )
                if tx < fx.Int32(consumed_words):
                    buffer_ops.buffer_store(
                        fx.Int32(GPUS_PER_NODE), consumed, tx
                    )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_system_release()

        if is_cco:
            qp = wave
            if const_expr(cco_geometry == "sparse_wqe"):
                if wave == fx.Int32(0):
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("sparse_remote_consumed")
                            )
                        )
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("sparse_remote_consumed")
                                )
                            )
                    comm_ops.fence_system_acquire()
                    put_value(
                        dev_comm,
                        fx.Int32(0),
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("sparse_remote_credit"),
                        generation,
                        aggregate=True,
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    credit_request = flush_async(
                        dev_comm,
                        fx.Int32(0),
                        fx.Int32(remote_node),
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    wait_request(
                        dev_comm,
                        fx.Int32(0),
                        credit_request,
                        scope="warp",
                    )
                    if lane == fx.Int32(0):
                        wait_ready(
                            local_addr("sparse_remote_credit"), generation
                        )
                    comm_ops.fence_system_acquire()
                    for request_qp in range_constexpr(layout.num_qp):
                        request = fx.Int64(
                            comm_ops.load_i64_global(
                                local_addr("sparse_remote_request")
                                + fx.Int64(request_qp * 8)
                            )
                        )
                        wait_request(
                            dev_comm,
                            fx.Int32(request_qp),
                            request,
                            scope="warp",
                        )
                gpu.barrier()
            if const_expr(cco_geometry == "mori64x2"):
                active_rail = wave < fx.Int32(2)
                if active_rail:
                    half = wave
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(half) * fx.Int64(4)
                            )
                        )
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("remote_chunk_consumed")
                                    + fx.Int64(half) * fx.Int64(4)
                                )
                            )
                    comm_ops.fence_system_acquire()
                    put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_chunk_credit")
                        + fx.Int64(half) * fx.Int64(8),
                        generation,
                        aggregate=True,
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    credit_req = flush_async(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        scope="warp",
                        team=TEAM_RAIL,
                    )
                    wait_request(dev_comm, qp, credit_req, scope="warp")
                    if lane == fx.Int32(0):
                        wait_ready(
                            local_addr("remote_chunk_credit")
                            + fx.Int64(half) * fx.Int64(8),
                            generation,
                        )
                    original_request = fx.Int64(
                        comm_ops.load_i64_global(
                            local_addr("remote_chunk_request")
                            + fx.Int64(half) * fx.Int64(8)
                        )
                    )
                    wait_request(
                        dev_comm, qp, original_request, scope="warp"
                    )
                gpu.barrier()

            credit_batch_count = (
                dispatch_chunks // cco_chunks_per_flush
                if cco_geometry == "chunked"
                else 0
            )
            for batch in range_constexpr(
                credit_batch_count
            ):
                batch_first = batch * cco_chunks_per_flush
                # Do not credit any chunk in the batch until every destination
                # fanout role has consumed all B reciprocal payloads.
                for batch_item in range_constexpr(cco_chunks_per_flush):
                    chunk = batch_first + batch_item
                    consume_index = fx.Int32(chunk * layout.num_qp) + qp
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(consume_index) * fx.Int64(4)
                            )
                        )
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("remote_chunk_consumed")
                                    + fx.Int64(consume_index) * fx.Int64(4)
                                )
                            )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    credit_byte = (
                        window_off("remote_chunk_credit")
                        + fx.Int64(consume_index) * fx.Int64(8)
                    )
                    put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        credit_byte,
                        generation,
                        aggregate=True,
                        scope="warp",
                        team=TEAM_RAIL,
                    )

                credit_req = flush_async(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                    scope="warp",
                    team=TEAM_RAIL,
                )
                wait_request(dev_comm, qp, credit_req, scope="warp")

                for batch_item in range_constexpr(cco_chunks_per_flush):
                    chunk = batch_first + batch_item
                    consume_index = fx.Int32(chunk * layout.num_qp) + qp
                    if lane == fx.Int32(0):
                        wait_ready(
                            local_addr("remote_chunk_credit")
                            + fx.Int64(consume_index) * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()

                # One retained data request belongs to the whole batch and is
                # reclaimed exactly once after all B reciprocal credits arrive.
                request_index = (
                    fx.Int64(batch_first * layout.num_qp) + fx.Int64(qp)
                )
                original_request = fx.Int64(
                    comm_ops.load_i64_global(
                        local_addr("remote_chunk_request")
                        + request_index * fx.Int64(8)
                    )
                )
                wait_request(dev_comm, qp, original_request, scope="warp")

        # The role targeting this rank observes exactly eight EOS values; each
        # one covers a local source rank and its aligned remote source rank.
        if is_finisher:
            if tx == fx.Int32(0):
                # Independent lane acquires followed by a CTA barrier do not
                # merge into one happens-before chain. The publishing thread
                # itself must acquire all eight communication-role releases.
                for peer in range_constexpr(GPUS_PER_NODE):
                    wait_ready(
                        local_addr("comm_eos") + fx.Int64(peer * 8), generation
                    )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            if tx < fx.Int32(LOCAL_EXPERTS):
                count = buffer_ops.buffer_load(
                    expert_count, tx, vec_width=1, dtype=T.i32
                )
                if const_expr(diagnostic_no_arrival_rmw):
                    tile_count = (
                        count + fx.Int32(BM - 1)
                    ) // fx.Int32(BM)
                    tile_map = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("expert_tile_map")
                    )
                    row_done = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_done")
                    )
                    for logical_tile in range(
                        fx.Int32(0), tile_count, fx.Int32(1)
                    ):
                        map_index = (
                            tx * fx.Int32(max_tiles_per_expert)
                            + logical_tile
                        )
                        wait_ready(
                            local_addr("expert_tile_map_ready")
                            + fx.Int64(map_index) * fx.Int64(8),
                            generation,
                        )
                        physical = buffer_ops.buffer_load(
                            tile_map,
                            map_index,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        remaining = count - logical_tile * fx.Int32(BM)
                        valid_rows = (
                            remaining < fx.Int32(BM)
                        ).select(remaining, fx.Int32(BM))
                        buffer_ops.buffer_store(
                            valid_rows, row_done, physical
                        )
                remainder = count % fx.Int32(BM)
                if remainder != fx.Int32(0):
                    logical_tile = count // fx.Int32(BM)
                    map_index = tx * fx.Int32(max_tiles_per_expert) + logical_tile
                    wait_ready(
                        local_addr("expert_tile_map_ready")
                        + fx.Int64(map_index) * fx.Int64(8),
                        generation,
                    )
                    physical = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("expert_tile_map")
                        ),
                        map_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    sources = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_source")
                    )
                    inputs = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_input")
                    )
                    weights = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_weight")
                    )
                    fallback_input = buffer_ops.buffer_load(
                        inputs,
                        physical * fx.Int32(BM),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    for row in range(remainder, BM, 1):
                        dst = physical * fx.Int32(BM) + row
                        buffer_ops.buffer_store(fallback_input, inputs, dst)
                        buffer_ops.buffer_store(fx.Int32(INVALID_SOURCE), sources, dst)
                        buffer_ops.buffer_store(fx.Float32(0.0), weights, dst)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            finish_scratch = fx.recast_iter(fx.Int32, lds_raw)
            if tx == fx.Int32(0):
                tiles = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
                fx.ptr_store(Vec.from_elements([tiles], fx.Int32), finish_scratch)
            gpu.barrier()
            tiles = Vec(
                fx.make_view(finish_scratch, fx.make_layout(1, 1)).load()
            )[0]
            total_jobs = tiles * fx.Int32(h1_n_blocks)
            if const_expr(tile_pipeline):
                # Full tiles were already appended by their unique row-32
                # last-arriver.  After EOS, append exactly the partial expert
                # tails that were padded above.  Different threads may reserve
                # batches concurrently; per-batch generation is the publication
                # point, so reservation order need not equal ready order.
                row_done = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_row_done")
                )
                for physical in range(tx, tiles, fx.Int32(THREADS)):
                    completed_rows = buffer_ops.buffer_load(
                        row_done, physical, vec_width=1, dtype=T.i32
                    )
                    if completed_rows < fx.Int32(BM):
                        _enqueue_tile_jobs(
                            fx.Int32(local_rank), physical, False
                        )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    final_tail = fx.Int32(
                        comm_ops.load_i32_global_system(
                            local_addr("h1_queue_tail")
                        )
                    )
                    if final_tail != total_jobs:
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                    buffer_ops.buffer_store(
                        tiles * fx.Int32(BM),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("num_valid")
                        ),
                        fx.Int32(0),
                    )
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        local_addr("h1_queue_eos"), generation
                    )
            else:
                queue = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("h1_ready_queue")
                )
                for job_index in range(tx, total_jobs, fx.Int32(THREADS)):
                    buffer_ops.buffer_store(job_index, queue, job_index)
                    comm_ops.store_i64_global_system(
                        local_addr("h1_ready_queue_generation")
                        + fx.Int64(job_index) * fx.Int64(8),
                        generation,
                    )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    buffer_ops.buffer_store(
                        total_jobs,
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("h1_queue_tail")
                        ),
                        fx.Int32(0),
                    )
                    buffer_ops.buffer_store(
                        tiles * fx.Int32(BM),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("num_valid")
                        ),
                        fx.Int32(0),
                    )
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        local_addr("h1_queue_eos"), generation
                    )

        def _run_gemm1_job(job):
            _gemm1_body(
                lds_raw,
                local_addr("grouped_input_q"),
                local_addr("grouped_input_scale"),
                w1q,
                w1scale,
                local_addr("tile_expert"),
                local_addr("tile_row_input"),
                local_addr("h1_output_q"),
                local_addr("h1_output_scale"),
                x_bf16,
                job,
                lane,
                wave,
                True,
                fx.Int32(layout.source_capacity),
                fx.Int32(max_route_tiles),
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=False,
                K=HIDDEN,
                N_OUT=2 * INTER,
                NE=LOCAL_EXPERTS,
                interleave=False,
                act="silu",
                swiglu_limit=None,
                situ_beta=4.0,
                situ_linear_beta=25.0,
            )

        # ------------------------------------------------------------------
        # GMM1 scheduler.  sparse_wqe production uses a ready-order queue:
        # full BM32 tiles are published by their last route arrival, so CTAs
        # that have completed their communication duty can overlap GMM1 with
        # the remaining fanout.  Partial tiles are sealed and appended at EOS.
        # Other geometries retain the post-EOS static reference scheduler.
        # ------------------------------------------------------------------
        if const_expr(not diagnostic_comm_only):
            is_compute = (
                fx.Int32(1) == fx.Int32(1)
                if const_expr(diagnostic_split_fanout)
                else ticket >= fx.Int32(ESSENTIAL_CTAS)
            )
            if is_compute:
                if const_expr(tile_pipeline and cco_geometry == "sparse_wqe"):
                    if is_inter_fanout:
                        # Per-QP receive lets this CTA place remote routes
                        # early.  Keep it out of the GMM queue until all four
                        # QPs have arrived, leaving the already-finished intra
                        # CTAs to consume early tiles without an inter-CTA
                        # queue-head/polling burst competing with transport.
                        if tx == fx.Int32(0):
                            wait_ready(
                                local_addr("sparse_remote_batch_ready"),
                                generation,
                            )
                        gpu.barrier()
                        comm_ops.fence_system_acquire()
                if const_expr(diagnostic_split_fanout):
                    # Each role drains its own communication stores before it
                    # joins either the streaming or post-EOS GMM scheduler.
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                if const_expr(tile_pipeline):
                    work_scratch = fx.recast_iter(fx.Int32, lds_raw)
                    work_view = fx.make_view(
                        work_scratch, fx.make_layout(1, 1)
                    )
                    queue = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("h1_ready_queue")
                    )
                    work_shard = ticket & fx.Int32(work_shards - 1)
                    consumer_active = fx.Int32(1) == fx.Int32(1)
                    while consumer_active:
                        gpu.barrier()
                        if tx == fx.Int32(0):
                            sequence = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    local_addr("h1_queue_head")
                                    + fx.Int64(work_shard * fx.Int32(16 * 4)),
                                    fx.Int32(1),
                                )
                            )
                            qidx = (
                                work_shard
                                + sequence * fx.Int32(work_shards)
                            )
                            job = fx.Int32(-1)
                            if qidx < fx.Int32(max_jobs):
                                batch = (
                                    qidx // fx.Int32(h1_n_blocks)
                                ) * fx.Int32(h1_n_blocks)
                                ready = fx.Int64(
                                    comm_ops.load_i64_global_system(
                                        local_addr(
                                            "h1_ready_queue_generation"
                                        )
                                        + fx.Int64(batch) * fx.Int64(8)
                                    )
                                )
                                eos = fx.Int64(
                                    comm_ops.load_i64_global_system(
                                        local_addr("h1_queue_eos")
                                    )
                                )
                                while (ready < generation) & (
                                    eos < generation
                                ):
                                    ready = fx.Int64(
                                        comm_ops.load_i64_global_system(
                                            local_addr(
                                                "h1_ready_queue_generation"
                                            )
                                            + fx.Int64(batch) * fx.Int64(8)
                                        )
                                    )
                                    eos = fx.Int64(
                                        comm_ops.load_i64_global_system(
                                            local_addr("h1_queue_eos")
                                        )
                                    )
                                if ready < generation:
                                    # EOS release-orders every final enqueue.
                                    # A claimed slot below final_tail must have
                                    # a matching generation even if this CTA
                                    # observed EOS before reloading the marker.
                                    comm_ops.fence_system_acquire()
                                    final_tail = fx.Int32(
                                        comm_ops.load_i32_global_system(
                                            local_addr("h1_queue_tail")
                                        )
                                    )
                                    if qidx < final_tail:
                                        wait_ready(
                                            local_addr(
                                                "h1_ready_queue_generation"
                                            )
                                            + fx.Int64(batch) * fx.Int64(8),
                                            generation,
                                        )
                                        ready = generation
                                if ready >= generation:
                                    comm_ops.fence_system_acquire()
                                    job = buffer_ops.buffer_load(
                                        queue,
                                        qidx,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                    if const_expr(tile_pipeline_instrument):
                                        if _all_comm_eos_seen() == fx.Int32(0):
                                            comm_ops.atomic_add_system_acq_rel(
                                                local_addr(
                                                    "h1_gmm_started_before_all_comm_eos"
                                                ),
                                                fx.Int32(1),
                                            )
                            fx.ptr_store(
                                Vec.from_elements([job], fx.Int32),
                                work_scratch,
                            )
                        gpu.barrier()
                        job = fx.Int32(Vec(work_view.load())[0])
                        has_work = job >= fx.Int32(0)
                        if has_work:
                            _run_gemm1_job(job)
                            rocdl.s_waitcnt(0)
                            gpu.barrier()
                            if tx == fx.Int32(0):
                                if const_expr(tile_pipeline_instrument):
                                    if _all_comm_eos_seen() == fx.Int32(0):
                                        comm_ops.atomic_add_system_acq_rel(
                                            local_addr(
                                                "h1_gmm_completed_before_all_comm_eos"
                                            ),
                                            fx.Int32(1),
                                        )
                                comm_ops.fence_system_release()
                                comm_ops.atomic_add_system_acq_rel(
                                    local_addr("h1_compute_done"), fx.Int32(1)
                                )
                        consumer_active = has_work
                else:
                    if tx == fx.Int32(0):
                        wait_ready(local_addr("h1_queue_eos"), generation)
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    tiles = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("tile_alloc")
                        ),
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    total_jobs = tiles * fx.Int32(h1_n_blocks)
                    consumer_index = (
                        ticket
                        if const_expr(diagnostic_split_fanout)
                        else ticket - fx.Int32(ESSENTIAL_CTAS)
                    )
                    consumer_count = (
                        fx.Int32(256)
                        if const_expr(diagnostic_split_fanout)
                        else fx.Int32(worker_blocks - ESSENTIAL_CTAS)
                    )
                    for job in range(
                        consumer_index, total_jobs, consumer_count
                    ):
                        _run_gemm1_job(job)
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        if tx == fx.Int32(0):
                            comm_ops.fence_system_release()
                            comm_ops.atomic_add_system_acq_rel(
                                local_addr("h1_compute_done"), fx.Int32(1)
                            )

        if is_finisher:
            if tx == fx.Int32(0):
                if const_expr(not diagnostic_comm_only):
                    tiles = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    expected_jobs = tiles * fx.Int32(h1_n_blocks)
                    completed = fx.Int32(0)
                    while completed < expected_jobs:
                        completed = fx.Int32(
                            comm_ops.load_i32_global_system(local_addr("h1_compute_done"))
                        )
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    stage2_addr("stage1_done"), generation
                )

    @flyc.jit
    def launch_megamoe_tile_ep16_stage1(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        x_bf16: fx.Int64,
        input_scale: fx.Int64,
        route_weights: fx.Int64,
        topk_ids: fx.Int64,
        w1q: fx.Int64,
        w1scale: fx.Int64,
        ntokens: fx.Int32,
        generation: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            dev_comm,
            arena_win,
            arena_ptr,
            x_bf16,
            input_scale,
            route_weights,
            topk_ids,
            w1q,
            w1scale,
            ntokens,
            generation,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=(worker_blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch_megamoe_tile_ep16_stage1.kernel_name = kernel_name
    launch_megamoe_tile_ep16_stage1.layout = layout
    launch_megamoe_tile_ep16_stage1.stage2_layout = stage2_layout
    launch_megamoe_tile_ep16_stage1.stage2_window_offset = stage2_window_offset
    launch_megamoe_tile_ep16_stage1.worker_blocks = worker_blocks
    launch_megamoe_tile_ep16_stage1.lds_bytes = lds_bytes
    launch_megamoe_tile_ep16_stage1.essential_ctas = (
        256 if diagnostic_split_fanout else ESSENTIAL_CTAS
    )
    launch_megamoe_tile_ep16_stage1.enable_cco = bool(enable_cco)
    launch_megamoe_tile_ep16_stage1.diagnostic_comm_only = bool(
        diagnostic_comm_only
    )
    launch_megamoe_tile_ep16_stage1.diagnostic_split_fanout = bool(
        diagnostic_split_fanout
    )
    launch_megamoe_tile_ep16_stage1.diagnostic_wave_fanout = bool(
        diagnostic_wave_fanout
    )
    launch_megamoe_tile_ep16_stage1.diagnostic_no_arrival_rmw = bool(
        diagnostic_no_arrival_rmw
    )
    launch_megamoe_tile_ep16_stage1.cco_chunks_per_flush = int(
        cco_chunks_per_flush
    )
    launch_megamoe_tile_ep16_stage1.cco_geometry = cco_geometry
    launch_megamoe_tile_ep16_stage1.quant_two_cta_per_token = bool(
        quant_two_cta_per_token
    )
    launch_megamoe_tile_ep16_stage1.prequant_input = bool(prequant_input)
    launch_megamoe_tile_ep16_stage1.gemm1_contraction = bool(
        not diagnostic_comm_only
    )
    launch_megamoe_tile_ep16_stage1.full_stage1_fusion = bool(
        not diagnostic_comm_only
    )
    launch_megamoe_tile_ep16_stage1.tile_pipeline = bool(tile_pipeline)
    launch_megamoe_tile_ep16_stage1.tile_pipeline_instrument = bool(
        tile_pipeline_instrument
    )
    launch_megamoe_tile_ep16_stage1.tile_pipeline_fanout_shards = int(
        split_fanout_shards
    )
    launch_megamoe_tile_ep16_stage1.cco_logical_doorbells = (
        5
        if cco_geometry == "sparse_wqe"
        else (
            4
            if cco_geometry == "mori64x2"
            else 2
            * layout.num_qp
            * (dispatch_chunks // cco_chunks_per_flush)
        )
    )
    launch_megamoe_tile_ep16_stage1.split_flag_bytes = (
        GPUS_PER_NODE * 32 * 8 if diagnostic_split_fanout else 0
    )
    launch_megamoe_tile_ep16_stage1.single_gpu_launch = True
    launch_megamoe_tile_ep16_stage1.requires_resident_grid = True
    launch_megamoe_tile_ep16_stage1.architecture_contract = {
        "dispatch": "scoreboard_direct_to_expert_tile",
        "receive_comm_roles": 8,
        "cross_node_comm_roles": 1,
        "intra_node_comm_roles": 7,
        "allocation_counter": "alloc_count",
        "arrival_counter": "tile_arrived",
        "eos_tail": True,
        "uses_rank_inbox": False,
        "uses_source_activation_inbox": True,
        "uses_group_sort": False,
        "cross_node_dedup": (
            "skip_token_without_remote_route_one_data_wqe_per_remote_token"
            if cco_geometry == "sparse_wqe"
            else "one_record_per_token_per_node"
        ),
        "destination_rank_payload": "one_source_indexed_activation_row_per_rank",
        "rank_route_encoding": "u16_topk_slot_mask_per_global_rank",
        "sparse_token_readiness": (
            "four_streamed_qp_terminal_words_with_inter_compute_batch_gate"
            if cco_geometry == "sparse_wqe"
            else "not_applicable"
        ),
        "early_full_tile_enqueue": bool(tile_pipeline),
        "queue_publication": (
            "full_tile_last_arrival_plus_partial_tile_post_8_role_eos"
            if tile_pipeline
            else "post_8_role_eos_physical_major"
        ),
        "gmm_scheduler": (
            "bypassed_after_queue_publish"
            if diagnostic_comm_only
            else (
                "concurrent_ready_queue_8_shards_256_all_roles_rejoin"
                if tile_pipeline
                else (
                    "post_eos_static_strided_256_all_roles_rejoin"
                    if diagnostic_split_fanout
                    else "post_eos_static_strided_24_pure_compute_consumers"
                )
            )
        ),
        "input_scale_layout": "bm32_ku_ikxdl_klane_nlane_ima",
        "diagnostic_comm_only": bool(diagnostic_comm_only),
        "diagnostic_split_fanout": bool(diagnostic_split_fanout),
        "diagnostic_wave_fanout": bool(diagnostic_wave_fanout),
        "diagnostic_no_arrival_rmw": bool(diagnostic_no_arrival_rmw),
        "cco_chunks_per_flush": int(cco_chunks_per_flush),
        "cco_geometry": cco_geometry,
        "quant_two_cta_per_token": bool(quant_two_cta_per_token),
        "prequant_input": bool(prequant_input),
        "tile_pipeline": bool(tile_pipeline),
        "tile_pipeline_instrument": bool(tile_pipeline_instrument),
        "tile_pipeline_fanout_shards": int(split_fanout_shards),
        "cco_logical_doorbells": (
            5
            if cco_geometry == "sparse_wqe"
            else (
                4
                if cco_geometry == "mori64x2"
                else 2
                * layout.num_qp
                * (dispatch_chunks // cco_chunks_per_flush)
            )
        ),
        "fanout_mapping": (
            "128_inter_plus_128_intra_ctas_one_active_wave_eight_records"
            if diagnostic_wave_fanout
            else (
                (
                    f"{split_fanout_ctas}_inter_plus_{split_fanout_ctas}_intra_"
                    f"{dedicated_compute_ctas}_dedicated_compute_ctas_"
                    f"dest_mod8_shard_div{split_fanout_shards}"
                    if tile_pipeline
                    else "128_inter_plus_128_intra_dest_mod8_shard_div8"
                )
                if diagnostic_split_fanout
                else "legacy_8_destination_ctas"
            )
        ),
        "fanout_completion": (
            (
                f"8x{2 * split_fanout_shards}_unique_generation_flags_"
                "then_one_eos_and_consumed_per_dest"
                if tile_pipeline
                else "8x32_unique_generation_flags_then_one_eos_and_consumed_per_dest"
            )
            if diagnostic_split_fanout
            else "one_role_one_eos_and_consumed_per_dest"
        ),
        "fanout_flag_storage": (
            "arena_parity_8x32_i64"
            if diagnostic_split_fanout
            else "none"
        ),
    }
    launch_megamoe_tile_ep16_stage1.output_regions = {
        name: layout.region(name).offset
        for name in (
            "h1_output_q",
            "h1_output_scale",
            "tile_expert",
            "tile_row_base",
            "num_valid",
            "tile_row_input",
            "tile_row_source",
            "tile_row_weight",
        )
    }
    return launch_megamoe_tile_ep16_stage1


__all__ = ["compile_megamoe_tile_ep16_stage1"]
