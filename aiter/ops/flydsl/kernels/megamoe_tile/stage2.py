# SPDX-License-Identifier: MIT
"""Direct-node-accumulator EP16 A4W4 Stage-2 in one GPU launch.

For arbitrary Top-K routing, a source token may contribute zero or multiple
routes on an EP rank.  The weighted GMM2 epilogue does not materialize rank
partials: every route LSA atomic-adds its FP32 contribution directly into the
aligned source proxy on the current expert node.  A Stage-1-produced
scoreboard supplies the exact route-slot count (0..16) for every
``(source-node, token, hidden-tile)``.  The last contributor publishes tile
readiness; an expected count of zero publishes an explicitly cleared partial.

Resident CTA roles after a common initialization barrier are::

    block 0      four-wave CCO RAIL BF16 return / receive / credit
    blocks 1..7  final local-node + remote-node combine
    blocks 8..N  persistent A4W4 GMM2 work queue

No rank-partial, node-reducer, pack, return-sidecar, or final-combine kernel is
launched.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float32, Int8, T

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
from .stage2_abi import Stage2ArenaLayout, Stage2NodePartialWire


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
COMBINE_BLOCKS = 7
GMM_FIRST = COMBINE_FIRST + COMBINE_BLOCKS
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
        raise ValueError("the first direct Stage-2 requires BM/BN/BK=32/256/256")
    if WORK_SHARDS != 8:
        raise ValueError("the first direct Stage-2 requires 8 work shards")
    if team != TEAM_RAIL:
        raise ValueError("production Stage-2 requires the CCO RAIL team")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")
    if diagnostic_mode not in ("full", "atomic_only", "return_only"):
        raise ValueError(
            "diagnostic_mode must be full, atomic_only, or return_only"
        )

    rank = int(rank)
    node = rank // GPUS_PER_NODE
    local_rank = rank % GPUS_PER_NODE
    remote_node = 1 - node
    local_plane = node
    remote_plane = remote_node
    hidden_tiles = HIDDEN // BN
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

    # Stage-2 scoreboard/payload regions, relative to the logical Stage-2 base.
    expected_off, expected_stride = _parity_parts(s2.region("node_expected"), 2)
    accumulator_off, accumulator_stride = _parity_parts(
        s2.region("node_accumulator"), 2
    )
    done_off, done_stride = _parity_parts(s2.region("node_done"), 2)
    ready_off, ready_stride = _parity_parts(s2.region("node_tile_ready"), 2)
    tx_off, tx_stride = _parity_parts(s2.region("remote_node_tx"), 2)
    rx_off, rx_stride = _parity_parts(s2.region("remote_partial_rx"), 2)
    return_ready_off, return_ready_stride = _parity_parts(
        s2.region("return_group_ready"), 2
    )
    consumed_off, consumed_stride = _parity_parts(s2.region("return_consumed"), 2)
    stage1_done_off, stage1_done_stride = _parity_parts(s2.region("stage1_done"), 2)
    stage2_phase_off, stage2_phase_stride = _parity_parts(s2.region("stage2_init"), 2)
    grid_barrier_off = s2.region("grid_barrier").offset
    gemm_head_off = s2.region("gemm_work_head").offset
    final_head_off = s2.region("final_work_head").offset
    final_done_off = s2.region("final_done").offset
    error_off = s2.region("stage2_error_count").offset

    # Custom FP32 C-shuffle slab plus eight peer Stage-2 base pointers.
    a_stages = kStages + 1
    kh_tile_a = BK // 2
    compute_lds_bytes = max(BM * BN * 4, a_stages * BM * kh_tile_a)
    peer_table_off = compute_lds_bytes
    lds_bytes = peer_table_off + GPUS_PER_NODE * 8

    @fx.struct
    class SharedStorage:
        raw: fx.Array[Int8, lds_bytes, 16]

    kernel_name = (
        "megamoe_tile_ep16_stage2_direct_node_a4w4_"
        f"r{rank}_h{HIDDEN}_i{INTER}_e{EXPERTS}_bm{BM}_bn{BN}_bk{BK}_q4_direct_v6_gridrelease"
        + ("" if diagnostic_mode == "full" else f"_{diagnostic_mode}")
    )

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
        expected_ptr = s2_ptr(expected_off, expected_stride)
        accumulator_ptr = s2_ptr(accumulator_off, accumulator_stride)
        done_ptr = s2_ptr(done_off, done_stride)
        ready_ptr = s2_ptr(ready_off, ready_stride)
        tx_ptr = s2_ptr(tx_off, tx_stride)
        rx_ptr = s2_ptr(rx_off, rx_stride)
        return_ready_ptr = s2_ptr(return_ready_off, return_ready_stride)
        consumed_ptr = s2_ptr(consumed_off, consumed_stride)
        stage1_done_ptr = s2_ptr(stage1_done_off, stage1_done_stride)
        stage2_phase_ptr = s2_ptr(stage2_phase_off, stage2_phase_stride)
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

        # Stage-1 completion from every local rank closes all scoreboards and
        # H1 rows before Stage-2 clears its accumulator parity.
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

        if bx == fx.Int32(0):
            if tx < fx.Int32(WORK_SHARDS):
                comm_ops.store_i32_system(
                    gemm_head_ptr, tx * fx.Int32(16), fx.Int32(0)
                )
            if tx == fx.Int32(0):
                comm_ops.store_i32_system(final_head_ptr, fx.Int32(0), fx.Int32(0))
                comm_ops.store_i32_system(final_done_ptr, fx.Int32(0), fx.Int32(0))
                comm_ops.store_i32_system(error_ptr, fx.Int32(0), fx.Int32(0))
        grid_sync()

        # FP32 accumulator is the only large clear. Stale ready generations do
        # not match this generation; done must start at zero for last-arriver.
        acc_rsrc = buffer_ops.create_buffer_resource_from_addr(accumulator_ptr)
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        acc_vec4 = fx.Int32((2 * MAX_TOKENS * HIDDEN) // 4)
        for item in range(global_tx, acc_vec4, grid_threads):
            buffer_ops.buffer_store(zero4, acc_rsrc, item * fx.Int32(4))

        done_rsrc = buffer_ops.create_buffer_resource_from_addr(done_ptr)
        scoreboard_items = fx.Int32(2 * MAX_TOKENS * hidden_tiles)
        for item in range(global_tx, scoreboard_items, grid_threads):
            buffer_ops.buffer_store(fx.Int32(0), done_rsrc, item)
        expected_rsrc = buffer_ops.create_buffer_resource_from_addr(expected_ptr)
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

        # A token/tile whose routes all belong to the other node has no GMM2
        # producer that could become the last arriver.  The cleared FP32
        # accumulator is already the correct zero partial, so publish it here
        # after the grid-wide clear/done initialization barrier.
        comm_ops.fence_system_release()
        for item in range(global_tx, scoreboard_items, grid_threads):
            expected = buffer_ops.buffer_load(
                expected_rsrc, item, vec_width=1, dtype=T.i32
            )
            if expected == fx.Int32(0):
                comm_ops.store_i64_global_system(
                    ready_ptr + fx.Int64(item) * fx.Int64(8), generation
                )
        grid_sync()

        # Every target proxy must clear before any peer issues LSA atomics.
        init_phase = (generation << fx.Int64(2)) + fx.Int64(1)
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
        # FP32 accumulator is the partial payload under test.
        if const_expr(diagnostic_mode == "return_only"):
            for item in range(global_tx, scoreboard_items, grid_threads):
                buffer_ops.buffer_store(
                    fx.Int32(GPUS_PER_NODE), done_rsrc, item
                )
            rocdl.s_waitcnt(0)
            grid_sync()
            comm_ops.fence_system_release()
            for item in range(global_tx, scoreboard_items, grid_threads):
                comm_ops.store_i64_global_system(
                    ready_ptr + fx.Int64(item) * fx.Int64(8), generation
                )
            grid_sync()
            comm_ops.fence_system_acquire()

        role_enabled = fx.Int32(0 if diagnostic_mode == "atomic_only" else 1)
        compute_enabled = fx.Int32(0 if diagnostic_mode == "return_only" else 1)

        # ---------------------- Role 0: cross-node return ----------------------
        if (bx == fx.Int32(CROSS_BLOCK)) & (role_enabled == fx.Int32(1)):
            qp = wave
            node_acc = global_typed_ptr(accumulator_ptr, T.f32, align=4)
            tx_bf16 = global_typed_ptr(tx_ptr, T.bf16, align=2)
            for group in range(qp, fx.Int32(return_groups), fx.Int32(s2.num_qp)):
                if lane == fx.Int32(0):
                    for record in range_constexpr(wire.records_per_group):
                        token = group * fx.Int32(wire.records_per_group) + fx.Int32(record)
                        for ntile in range_constexpr(hidden_tiles):
                            ready_index = (
                                (fx.Int32(remote_plane * MAX_TOKENS) + token)
                                * fx.Int32(hidden_tiles)
                                + fx.Int32(ntile)
                            )
                            wait_ready(
                                ready_ptr + fx.Int64(ready_index) * fx.Int64(8),
                                generation,
                            )
                gpu.barrier()
                comm_ops.fence_system_acquire()

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
                    fx.ptr_store(fx.BFloat16(value), tx_bf16 + fx.Int64(dst_index))
                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_system_release()

                for record in range_constexpr(wire.records_per_group):
                    token = group * fx.Int32(wire.records_per_group) + fx.Int32(record)
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

            # Do not credit remote_rx reuse until all seven final roles consumed it.
            if tx == fx.Int32(0):
                completed = fx.Int32(comm_ops.load_i32_global_system(final_done_ptr))
                while completed < fx.Int32(MAX_TOKENS * hidden_tiles):
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

        # ---------------- Roles 1..7: source final combine ----------------
        elif (bx >= fx.Int32(COMBINE_FIRST)) & (
            bx < fx.Int32(COMBINE_FIRST + COMBINE_BLOCKS)
        ) & (role_enabled == fx.Int32(1)):
            local_acc = global_typed_ptr(accumulator_ptr, T.f32, align=4)
            remote_in = global_typed_ptr(rx_ptr, T.bf16, align=2)
            output = global_typed_ptr(arg_output_bf16, T.bf16, align=2)
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
                has_work = work < fx.Int32(MAX_TOKENS * hidden_tiles)
                if has_work:
                    token = work // fx.Int32(hidden_tiles)
                    ntile = work - token * fx.Int32(hidden_tiles)
                    local_ready_index = (
                        (fx.Int32(local_plane * MAX_TOKENS) + token)
                        * fx.Int32(hidden_tiles)
                        + ntile
                    )
                    if tx == fx.Int32(0):
                        wait_ready(
                            ready_ptr + fx.Int64(local_ready_index) * fx.Int64(8),
                            generation,
                        )
                        group = token // fx.Int32(wire.records_per_group)
                        wait_ready(
                            return_ready_ptr + fx.Int64(group) * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    col = ntile * fx.Int32(BN) + tx
                    local_index = (
                        (fx.Int32(local_plane * MAX_TOKENS) + token)
                        * fx.Int32(HIDDEN)
                        + col
                    )
                    token_index = token * fx.Int32(HIDDEN) + col
                    local_f32 = fx.ptr_load(local_acc + fx.Int64(local_index))
                    # Match the BF16 node-partial wire rounding on both nodes.
                    local_bf16 = fx.BFloat16(local_f32)
                    remote_bf16 = fx.ptr_load(remote_in + fx.Int64(token_index))
                    result = fx.Float32(local_bf16) + fx.Float32(remote_bf16)
                    fx.ptr_store(fx.BFloat16(result), output + fx.Int64(token_index))
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        comm_ops.fence_system_release()
                        comm_ops.atomic_add_system_acq_rel(
                            final_done_ptr, fx.Int32(1)
                        )
                active = has_work

        # ------------------ Remaining roles: persistent GMM2 ------------------
        elif (bx >= fx.Int32(GMM_FIRST)) & (compute_enabled == fx.Int32(1)):
            num_valid = fx.Int32(global_typed_ptr(arg_nvalid, T.i32)[0])
            total_m_blocks = num_valid // fx.Int32(BM)
            total_work = total_m_blocks * fx.Int32(hidden_tiles)
            shard = bx & fx.Int32(WORK_SHARDS - 1)
            active = fx.Int32(1) == fx.Int32(1)

            def direct_node_epilog(accm, m_row, n_block):
                lds_f32 = lds_typed_ptr(lds_base, T.f32, align=4)
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
                gpu.barrier()

                meta = buffer_ops.create_buffer_resource_from_addr(arg_stids)
                weights = buffer_ops.create_buffer_resource_from_addr(arg_sweights)
                epi_lanes = 32
                epi_rows = THREADS // epi_lanes
                m_lane = tx // fx.Int32(epi_lanes)
                n_lane = tx % fx.Int32(epi_lanes)
                col_start = n_lane * fx.Int32(2)
                for mr in range_constexpr(BM // epi_rows):
                    sorted_pos = m_row + fx.Int32(mr * epi_rows) + m_lane
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
                        row_base = (
                            (source_node * fx.Int32(MAX_TOKENS) + token)
                            * fx.Int32(HIDDEN)
                            + n_block * fx.Int32(BN)
                            + col_start
                        )
                        row_in_block = fx.Int32(mr * epi_rows) + m_lane
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
                            out_col = row_base + fx.Int32(s * epi_lanes * 2)
                            _atomic_add_f32_system(
                                peer_base
                                + fx.Int64(accumulator_off)
                                + parity * fx.Int64(accumulator_stride)
                                + fx.Int64(out_col) * fx.Int64(4),
                                values[0] * weight,
                            )
                            _atomic_add_f32_system(
                                peer_base
                                + fx.Int64(accumulator_off)
                                + parity * fx.Int64(accumulator_stride)
                                + fx.Int64(out_col + fx.Int32(1)) * fx.Int64(4),
                                values[1] * weight,
                            )

                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_system_release()
                if tx < fx.Int32(BM):
                    packed = buffer_ops.buffer_load(
                        meta, m_row + tx, vec_width=1, dtype=T.i32
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
                            # Each producer release-fences its FP32 atomics
                            # before joining the acq_rel done RMW release
                            # sequence. The last arriver acquires that sequence
                            # before publishing ready, so a ready acquire makes
                            # every route contribution visible.
                            comm_ops.store_i64_global_system(
                                peer_base
                                + fx.Int64(ready_off)
                                + parity * fx.Int64(ready_stride)
                                + fx.Int64(score_index) * fx.Int64(8),
                                generation,
                            )

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
                    m_block = work // fx.Int32(hidden_tiles)
                    if const_expr(diagnostic_mode == "atomic_only"):
                        n_block = work - m_block * fx.Int32(hidden_tiles)
                        m_row = m_block * fx.Int32(BM)
                        accm = [
                            [
                                fx.Vector.filled(4, 0.0, fx.Float32)
                                for _ in range((BN // 4) // 16)
                            ]
                            for _ in range(BM // 16)
                        ]
                    else:
                        for slot in range_constexpr(kStages):
                            issue_a_load_lds_dt(
                                arg_aq,
                                lds_base,
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
                        rocdl.sched_barrier(0)
                        accm, m_row, n_block, _ = gemm2_compute_v2(
                            lds_base,
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
                        )
                    direct_node_epilog(accm, m_row, n_block)
                active = has_work

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
    launch_megamoe_tile_ep16_stage2.single_gpu_launch = True
    launch_megamoe_tile_ep16_stage2.requires_resident_grid = True
    launch_megamoe_tile_ep16_stage2.fixed_roles = {
        "cross": 1,
        "intranode_final": COMBINE_BLOCKS,
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
    launch_megamoe_tile_ep16_stage2.combine_contract = (
        "GMM2 weighted FP32 LSA atomic -> direct node accumulator -> "
        "one BF16 RAIL return/token -> source final add"
    )
    launch_megamoe_tile_ep16_stage2.architecture_contract = {
        "epilogue": "direct_lsa_atomic_source_aligned_node_accumulator",
        "node_accumulator_dtype": "fp32",
        "uses_rank_partial": False,
        "uses_node_scan": False,
        "uses_external_reduce_kernel": False,
        "uses_external_return_kernel": False,
        "uses_external_final_kernel": False,
        "cross_ctas": 1,
        "intranode_combine_ctas": COMBINE_BLOCKS,
        "diagnostic_mode": diagnostic_mode,
    }
    launch_megamoe_tile_ep16_stage2.stage2_window_offset = s2_window_off
    launch_megamoe_tile_ep16_stage2.uses_rank_partial = False
    launch_megamoe_tile_ep16_stage2.uses_external_reduce_kernel = False
    launch_megamoe_tile_ep16_stage2.uses_external_return_kernel = False
    launch_megamoe_tile_ep16_stage2.uses_external_final_kernel = False
    return launch_megamoe_tile_ep16_stage2


__all__ = ["compile_megamoe_tile_ep16_stage2_a4w4"]
