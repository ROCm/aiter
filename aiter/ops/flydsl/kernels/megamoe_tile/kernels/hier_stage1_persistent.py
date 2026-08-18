# SPDX-License-Identifier: MIT
"""Persistent H1: communication roles join a sharded per-tile work queue."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
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


def _float_tag(value: float) -> str:
    return (
        str(float(value))
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def compile_hier_stage1_a4w4_persistent(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    COMM_BLOCKS: int = 1,
    WORK_SHARDS: int = 8,
    scheduler: str = "ticket",
    waves_per_eu_hint: int = 3,
    BM: int = 32,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = True,
    enable_copy: bool = True,
    enable_signal: bool = True,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    """Compile a persistent, dynamically scheduled H1 kernel.

    An entry ticket assigns the per-launch owner/communication roles and derives
    an epoch. The owner resets ``WORK_SHARDS`` int32 counters (each on its own
    64-byte cache line) and publishes ``arg_epoch_gate``. Every CTA then claims
    one GEMM tile at a time until the runtime tile bound is exhausted. The
    current copy stub supports exactly one communication ticket, which enters
    the same work loop after publishing its signal.

    ``arg_entry_count`` is initialized once and then remains monotonic. Repeated
    launches with the same workspace require the same worker-grid size; use a
    separate workspace slot for each supported grid geometry or concurrent
    stream.
    """

    activation = normalize_activation(activation)
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if BN != 256 or BK != 256:
        raise ValueError(
            f"persistent H1 A4W4 requires BN=BK=256, got BN={BN}, BK={BK}"
        )
    if (BM, use_nt) not in {
        (32, True),
        (32, False),
        (64, False),
        (128, False),
    }:
        raise ValueError(
            f"unsupported persistent H1 variant BM={BM}, use_nt={use_nt}"
        )
    if D_HIDDEN % BK or (2 * D_INTER) % BN:
        raise ValueError("persistent H1 dimensions must be divisible by BK/BN")
    if COMM_BLOCKS != 1:
        raise ValueError(
            "copy-stub persistent H1 currently requires COMM_BLOCKS=1"
        )
    if WORK_SHARDS not in (1, 2, 4, 8):
        raise ValueError("WORK_SHARDS must be one of 1, 2, 4, 8")
    if scheduler not in ("ticket", "strided"):
        raise ValueError("scheduler must be 'ticket' or 'strided'")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1, 2, 3, 4")

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(D_HIDDEN, BK)
    _, _, _, lds_bytes = _bm_constants(BM, BN, kh_tile, k_tiles_total)
    num_n_blocks = num_n_blocks_for(n_out_for(D_INTER), BN)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    activation_tag = activation
    if activation == "swiglu":
        limit = 7.0 if swiglu_limit is None else float(swiglu_limit)
        activation_tag += f"_l{_float_tag(limit)}"
    elif activation == "situv2":
        activation_tag += (
            f"_b{_float_tag(situ_beta)}_lb{_float_tag(situ_linear_beta)}"
        )
    elif swiglu_limit is not None:
        activation_tag += f"_l{_float_tag(swiglu_limit)}"

    name = (
        f"megamoe_tile_h1_persistent_a4w4_{activation_tag}_"
        f"h{D_HIDDEN}_i{D_INTER}_e{NE}_k{TOPK}_bm{BM}_"
        f"cb{COMM_BLOCKS}_ws{WORK_SHARDS}_wpe{waves_per_eu_hint}_"
        f"cp{int(enable_copy)}_sig{int(enable_signal)}_"
        f"{MXFP4_SCALE_LAYOUT_TAG}"
    )
    if not use_nt:
        name += "_nt0"
    if scheduler == "strided":
        name += "_strided"

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def h1_persistent_kernel(
        copy_src: fx.Int64,
        copy_dst: fx.Int64,
        copy_signal: fx.Int64,
        copy_nbytes: fx.Int32,
        generation: fx.Int64,
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
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))

        # Arrival-order tickets guarantee that the owner and communication role
        # are resident even when a future grid oversubscribes physical CUs.
        entry_scratch = fx.recast_iter(fx.Int64, lds_raw_ptr)
        entry_scratch_view = fx.make_view(entry_scratch, fx.make_layout(1, 1))
        if tx == fx.Int32(0):
            ticket64_lane = fx.Int64(
                comm_ops.atomic_add_agent(arg_entry_count, fx.Int64(1))
            )
            fx.ptr_store(
                Vec.from_elements([ticket64_lane], fx.Int64), entry_scratch
            )
        gpu.barrier()
        ticket64 = Vec(entry_scratch_view.load())[0]
        launch_grid = fx.Int64(gpu.grid_dim.x)
        epoch = ticket64 // launch_grid
        ticket = fx.Int32(ticket64 - epoch * launch_grid)
        gate_epoch = fx.Int32(epoch + fx.Int64(1))
        is_owner = ticket == fx.Int32(0)
        is_comm = (ticket > fx.Int32(0)) & (
            ticket <= fx.Int32(COMM_BLOCKS)
        )

        if is_owner:
            if tx < fx.Int32(WORK_SHARDS):
                comm_ops.store_i32_system(
                    arg_work_head,
                    tx * fx.Int32(16),
                    fx.Int32(0),
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.store_i32_system(
                    arg_epoch_gate, fx.Int32(0), gate_epoch
                )
        else:
            if tx == fx.Int32(0):
                observed = fx.Int32(
                    comm_ops.atomic_add_agent(arg_epoch_gate, fx.Int32(0))
                )
                while observed < gate_epoch:
                    observed = fx.Int32(
                        comm_ops.atomic_add_agent(
                            arg_epoch_gate, fx.Int32(0)
                        )
                    )
                comm_ops.fence_agent_acquire()
        gpu.barrier()

        # Completely erase communication selection in the compute-only
        # specialization; an empty dynamic role branch perturbs MFMA scheduling.
        if const_expr(enable_copy or enable_signal):
            if is_comm:
                if const_expr(enable_copy):
                    src = buffer_ops.create_buffer_resource_from_addr(copy_src)
                    dst = buffer_ops.create_buffer_resource_from_addr(copy_dst)
                    units = fx.Int32(copy_nbytes) // fx.Int32(16)
                    for unit in range(tx, units, fx.Int32(256)):
                        elem = unit * fx.Int32(4)
                        value = buffer_ops.buffer_load(
                            src, elem, vec_width=4, dtype=T.i32
                        )
                        buffer_ops.buffer_store(value, dst, elem)
                    tail = units * fx.Int32(16) + tx
                    if tail < fx.Int32(copy_nbytes):
                        value = buffer_ops.buffer_load(
                            src, tail, vec_width=1, dtype=T.i8
                        )
                        buffer_ops.buffer_store(value, dst, tail)
                    # A CTA barrier alone does not drain global stores. The
                    # following system-release signal must not overtake data.
                    rocdl.s_waitcnt(0)
                if const_expr(enable_signal):
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        comm_ops.store_i64_global_system(copy_signal, generation)
        # Keep every communication wave out of GMM/LDS reuse until lane 0 has
        # published the release signal. This is CTA-local, not a grid barrier.
        gpu.barrier()

        total_m_blocks = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
        total_work = total_m_blocks * fx.Int32(num_n_blocks)
        # Ticket is only a control-role identity. Use block ID for compute
        # assignment so 64-bit entry/epoch SSA values die before the MFMA loop.
        work_shard = bx & fx.Int32(WORK_SHARDS - 1)
        work_scratch = fx.recast_iter(fx.Int32, lds_raw_ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))

        def run_gemm(tile):
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
                tile,
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

        if const_expr(scheduler == "strided"):
            # Static persistent schedule keeps the MFMA loop free of atomics.
            # The communication ticket executes its own tile after publishing.
            if bx < total_work:
                run_gemm(bx)
            for iv in range(
                bx + fx.Int32(gpu.grid_dim.x),
                total_work,
                gpu.grid_dim.x,
            ):
                gpu.barrier()
                run_gemm(fx.Int32(iv))
        else:
            consumer_active = fx.Int32(1) == fx.Int32(1)
            while consumer_active:
                # Reuse the GMM LDS as a one-word CTA broadcast only after every
                # wave has completed the previous tile's epilogue.
                gpu.barrier()
                if tx == fx.Int32(0):
                    local_work = fx.Int32(
                        comm_ops.atomic_add_agent(
                            arg_work_head
                            + fx.Int64(work_shard) * fx.Int64(64),
                            fx.Int32(1),
                        )
                    )
                    work = work_shard + local_work * fx.Int32(WORK_SHARDS)
                    fx.ptr_store(
                        Vec.from_elements([work], fx.Int32), work_scratch
                    )
                gpu.barrier()
                work = Vec(work_scratch_view.load())[0]
                if tx == fx.Int32(0):
                    has_work = (work < total_work).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    fx.ptr_store(
                        Vec.from_elements([has_work], fx.Int32), work_scratch
                    )
                gpu.barrier()
                has_work = Vec(work_scratch_view.load())[0]
                if has_work != fx.Int32(0):
                    run_gemm(work)
                consumer_active = has_work != fx.Int32(0)

    @flyc.jit
    def launch_h1_persistent_sc2(
        copy_src: fx.Int64,
        copy_dst: fx.Int64,
        copy_signal: fx.Int64,
        copy_nbytes: fx.Int32,
        generation: fx.Int64,
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
        i32_worker_blocks: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        h1_persistent_kernel(
            copy_src,
            copy_dst,
            copy_signal,
            copy_nbytes,
            generation,
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
            grid=(i32_worker_blocks, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_h1_persistent_sc2.kernel_name = name
    launch_h1_persistent_sc2.lds_bytes = lds_bytes
    launch_h1_persistent_sc2.num_n_blocks = num_n_blocks
    launch_h1_persistent_sc2.work_shards = WORK_SHARDS
    launch_h1_persistent_sc2.min_worker_blocks = WORK_SHARDS
    launch_h1_persistent_sc2.scheduler = scheduler
    launch_h1_persistent_sc2.entry_count_int64 = 1
    launch_h1_persistent_sc2.epoch_gate_int32 = 1
    launch_h1_persistent_sc2.work_head_int32 = WORK_SHARDS * 16

    def checked_launch(
        workspace,
        copy_src,
        copy_dst,
        copy_signal,
        copy_nbytes,
        generation,
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
        *,
        stream,
    ):
        """Checked host entry point; normal callers should use this method."""

        workspace.validate_launch(
            launch_h1_persistent_sc2, generation=generation, stream=stream
        )
        return launch_h1_persistent_sc2(
            copy_src,
            copy_dst,
            copy_signal,
            copy_nbytes,
            generation,
            workspace.entry_count.data_ptr(),
            workspace.epoch_gate.data_ptr(),
            workspace.work_head.data_ptr(),
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            workspace.worker_blocks,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
            stream=stream,
        )

    launch_h1_persistent_sc2.checked = checked_launch
    return launch_h1_persistent_sc2
