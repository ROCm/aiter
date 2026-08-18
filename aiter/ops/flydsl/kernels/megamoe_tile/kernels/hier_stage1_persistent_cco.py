# SPDX-License-Identifier: MIT
"""Persistent CCO dispatch + preloaded A4W4 H1 ticket candidate.

One resident ticket owns a 256-thread CCO role (four waves/four QPs), submits a
64-KiB aggregate dispatch and waits for the reciprocal local payload.  It then
publishes the precomputed input plan/readiness and rejoins the same sharded
flat-GMM work loop as every compute ticket.  Request/credit retirement remains
an external sidecar responsibility; this kernel never waits a local request or
remote credit.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops
from ..gemm1 import (
    MXFP4_SCALE_LAYOUT_TAG,
    _bm_constants,
    _gemm1_body,
    _global_i32_at,
    k_tiles_total_for,
    n_out_for,
    num_n_blocks_for,
)

from ..activation import normalize_activation, validate_activation_parameters
from ..runtime import HierCcoArenaLayout
from .hier_sync import wait_i64_at_least_system


def _tag(value: float) -> str:
    return (
        str(float(value))
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def compile_hier_stage1_persistent_cco_a4w4(
    layout: HierCcoArenaLayout,
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    WORK_SHARDS: int = 8,
    waves_per_eu_hint: int = 2,
    BM: int = 32,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = True,
    team: str = "rail",
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    """Compile the real-CCO single-kernel ticket candidate.

    The compute tensors and expert-major metadata are preloaded.  The reciprocal
    CCO dispatch is nevertheless real and gates publication of
    ``h1_input_expected``, ``h1_input_ready`` and ``plan_ready``.  This seam is
    explicit until record unpack/sort/fanout is moved into the communication
    role.
    """

    # Keep MORI/CCO optional for ordinary megamoeTile kernel imports.
    from ..cco.ops import (
        TEAM_RAIL,
        TEAM_WORLD,
        flush_async,
        put,
        put_value,
        wait_ready,
    )

    activation = normalize_activation(activation)
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if team not in (TEAM_WORLD, TEAM_RAIL):
        raise ValueError("team must be world or rail")
    if layout.num_qp != 4 or layout.chunk_bytes != 64 * 1024:
        raise ValueError("persistent CCO H1 requires four QPs and 64-KiB chunks")
    if BN != 256 or BK != 256:
        raise ValueError("persistent CCO H1 currently requires BN=BK=256")
    if (BM, use_nt) not in {
        (32, True),
        (32, False),
        (64, False),
        (128, False),
    }:
        raise ValueError(f"unsupported CCO H1 variant BM={BM}, use_nt={use_nt}")
    if D_HIDDEN % BK or (2 * D_INTER) % BN:
        raise ValueError("persistent CCO H1 dimensions must divide BK/BN")
    if WORK_SHARDS not in (1, 2, 4, 8):
        raise ValueError("WORK_SHARDS must be one of 1,2,4,8")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")

    batch_per_qp = 8
    segment_bytes = 2048
    if layout.num_qp * batch_per_qp * segment_bytes != layout.chunk_bytes:
        raise ValueError("CCO H1 dispatch geometry must fill one ring chunk")

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(D_HIDDEN, BK)
    _, _, _, lds_bytes = _bm_constants(BM, BN, kh_tile, k_tiles_total)
    num_n_blocks = num_n_blocks_for(n_out_for(D_INTER), BN)

    tx_base = layout.region("dispatch_tx").offset
    rx_base = layout.region("dispatch_rx").offset
    ready_base = layout.region("dispatch_ready").offset
    request_base = layout.region("dispatch_request").offset
    plan_base = layout.region("plan_ready").offset
    expected_base = layout.region("h1_input_expected").offset
    expected_parity_bytes = layout.region("h1_input_expected").nbytes // 2
    input_ready_base = layout.region("h1_input_ready").offset
    input_ready_parity_bytes = layout.region("h1_input_ready").nbytes // 2

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    act_tag = activation
    if activation == "swiglu":
        limit = 7.0 if swiglu_limit is None else float(swiglu_limit)
        act_tag += f"_l{_tag(limit)}"
    elif activation == "situv2":
        act_tag += f"_b{_tag(situ_beta)}_lb{_tag(situ_linear_beta)}"
    elif swiglu_limit is not None:
        act_tag += f"_l{_tag(swiglu_limit)}"
    name = (
        f"megamoe_tile_h1_persistent_cco_a4w4_{act_tag}_"
        f"h{D_HIDDEN}_i{D_INTER}_e{NE}_k{TOPK}_bm{BM}_"
        f"ws{WORK_SHARDS}_wpe{waves_per_eu_hint}_{team}_"
        f"{MXFP4_SCALE_LAYOUT_TAG}"
    )
    if not use_nt:
        name += "_nt0"

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        peer: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int64,
        active_m_tiles: fx.Int32,
        expected_per_tile: fx.Int32,
        arg_entry_count: fx.Int64,
        arg_epoch_gate: fx.Int64,
        arg_work_head: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        qp = tx // fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, qp)

        # Arrival tickets guarantee that owner and CCO roles are resident.
        entry_scratch = fx.recast_iter(fx.Int64, lds_raw_ptr)
        entry_view = fx.make_view(entry_scratch, fx.make_layout(1, 1))
        if tx == fx.Int32(0):
            ticket_lane = fx.Int64(
                comm_ops.atomic_add_agent(arg_entry_count, fx.Int64(1))
            )
            fx.ptr_store(Vec.from_elements([ticket_lane], fx.Int64), entry_scratch)
        gpu.barrier()
        ticket64 = Vec(entry_view.load())[0]
        launch_grid = fx.Int64(gpu.grid_dim.x)
        epoch = ticket64 // launch_grid
        ticket = fx.Int32(ticket64 - epoch * launch_grid)
        gate_epoch = fx.Int32(epoch + fx.Int64(1))
        is_owner = ticket == fx.Int32(0)
        is_comm = ticket == fx.Int32(1)

        if is_owner:
            if tx < fx.Int32(WORK_SHARDS):
                comm_ops.store_i32_system(
                    arg_work_head, tx * fx.Int32(16), fx.Int32(0)
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.store_i32_system(arg_epoch_gate, fx.Int32(0), gate_epoch)
        else:
            if tx == fx.Int32(0):
                observed = fx.Int32(
                    comm_ops.atomic_add_agent(arg_epoch_gate, fx.Int32(0))
                )
                while observed < gate_epoch:
                    observed = fx.Int32(
                        comm_ops.atomic_add_agent(arg_epoch_gate, fx.Int32(0))
                    )
                comm_ops.fence_agent_acquire()
        gpu.barrier()

        if is_comm:
            slot_byte = fx.Int64(slot) * fx.Int64(layout.chunk_bytes)
            for item in range_constexpr(batch_per_qp):
                segment = qp * fx.Int32(batch_per_qp) + fx.Int32(item)
                payload_byte = fx.Int64(segment) * fx.Int64(segment_bytes)
                put(
                    dev_comm,
                    qp,
                    peer,
                    arena_win,
                    fx.Int64(rx_base) + slot_byte + payload_byte,
                    arena_win,
                    fx.Int64(tx_base) + slot_byte + payload_byte,
                    fx.Int64(segment_bytes),
                    aggregate=True,
                    scope="warp",
                    team=team,
                )
            ready_byte = (
                fx.Int64(ready_base)
                + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp))
                * fx.Int64(8)
            )
            put_value(
                dev_comm,
                qp,
                peer,
                arena_win,
                ready_byte,
                generation,
                aggregate=True,
                scope="warp",
                team=team,
            )
            request = flush_async(
                dev_comm, qp, peer, scope="warp", team=team
            )
            if lane == fx.Int32(0):
                request_addr = (
                    arena_ptr
                    + fx.Int64(request_base)
                    + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp))
                    * fx.Int64(8)
                )
                comm_ops.store_i64_global_system(request_addr, request)

            gpu.barrier()
            if lane == fx.Int32(0):
                wait_ready(arena_ptr + ready_byte, generation)
            gpu.barrier()
            comm_ops.fence_system_acquire()

            parity = generation & fx.Int64(1)
            expected_ptr = (
                arena_ptr
                + fx.Int64(expected_base)
                + parity * fx.Int64(expected_parity_bytes)
            )
            input_ready_ptr = (
                arena_ptr
                + fx.Int64(input_ready_base)
                + parity * fx.Int64(input_ready_parity_bytes)
            )
            expected_rsrc = buffer_ops.create_buffer_resource_from_addr(
                expected_ptr
            )
            input_ready_rsrc = buffer_ops.create_buffer_resource_from_addr(
                input_ready_ptr
            )
            for tile in range(tx, active_m_tiles, fx.Int32(256)):
                buffer_ops.buffer_store(expected_per_tile, expected_rsrc, tile)
                buffer_ops.buffer_store(expected_per_tile, input_ready_rsrc, tile)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    arena_ptr + fx.Int64(plan_base) + parity * fx.Int64(8),
                    generation,
                )

        # All tickets observe the reciprocal dispatch before claiming work.
        parity = generation & fx.Int64(1)
        plan_ptr = arena_ptr + fx.Int64(plan_base) + parity * fx.Int64(8)
        if tx == fx.Int32(0):
            wait_i64_at_least_system(plan_ptr, generation)
        gpu.barrier()
        comm_ops.fence_system_acquire()

        total_m_blocks = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
        total_work = total_m_blocks * fx.Int32(num_n_blocks)
        work_shard = bx & fx.Int32(WORK_SHARDS - 1)
        work_scratch = fx.recast_iter(fx.Int32, lds_raw_ptr)
        work_view = fx.make_view(work_scratch, fx.make_layout(1, 1))

        consumer_active = fx.Int32(1) == fx.Int32(1)
        while consumer_active:
            gpu.barrier()
            if tx == fx.Int32(0):
                local_work = fx.Int32(
                    comm_ops.atomic_add_agent(
                        arg_work_head + fx.Int64(work_shard) * fx.Int64(64),
                        fx.Int32(1),
                    )
                )
                work = work_shard + local_work * fx.Int32(WORK_SHARDS)
                fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            gpu.barrier()
            work = Vec(work_view.load())[0]
            if tx == fx.Int32(0):
                has_work = (work < total_work).select(fx.Int32(1), fx.Int32(0))
                fx.ptr_store(
                    Vec.from_elements([has_work], fx.Int32), work_scratch
                )
            gpu.barrier()
            has_work = Vec(work_view.load())[0]
            if has_work != fx.Int32(0):
                _gemm1_body(
                    lds_raw_ptr,
                    arg_aq,
                    arg_ascale,
                    arg_bq,
                    arg_bscale,
                    arg_eids,
                    arg_mind,
                    arg_aqout,
                    arg_ascaleout,
                    arg_hidden,
                    work,
                    lane,
                    wave,
                    use_nt,
                    i32_ntok,
                    total_m_blocks,
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    inline_quant=False,
                    K=D_HIDDEN,
                    N_OUT=2 * D_INTER,
                    NE=NE,
                    interleave=False,
                    act=activation,
                    swiglu_limit=swiglu_limit,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
            consumer_active = has_work != fx.Int32(0)

    @flyc.jit
    def launch_persistent_cco_v1(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        peer: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int64,
        active_m_tiles: fx.Int32,
        expected_per_tile: fx.Int32,
        arg_entry_count: fx.Int64,
        arg_epoch_gate: fx.Int64,
        arg_work_head: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        worker_blocks: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            dev_comm,
            arena_win,
            arena_ptr,
            peer,
            slot,
            generation,
            active_m_tiles,
            expected_per_tile,
            arg_entry_count,
            arg_epoch_gate,
            arg_work_head,
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=(worker_blocks, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_persistent_cco_v1.kernel_name = name
    launch_persistent_cco_v1.lds_bytes = lds_bytes
    launch_persistent_cco_v1.num_n_blocks = num_n_blocks
    launch_persistent_cco_v1.work_shards = WORK_SHARDS
    launch_persistent_cco_v1.preloaded_compute_seam = True
    launch_persistent_cco_v1.request_reclaim_external = True
    launch_persistent_cco_v1.team = team
    return launch_persistent_cco_v1


__all__ = ["compile_hier_stage1_persistent_cco_a4w4"]
