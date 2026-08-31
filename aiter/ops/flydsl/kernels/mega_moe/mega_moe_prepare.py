# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Low-resource compact-dispatch prepare kernel for MegaMoE Stage1."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .dispatch import DispatchSlot, emit_dispatch_group, emit_dispatch_plan
from .gemm_util import _buffer_load, _buffer_store, _make_buffer_from_addr
from .quant import emit_per_1x32_mx_fp8_group


@functools.cache
def compile_mega_moe_prepare(
    *,
    rank: int,
    experts_per_rank: int,
    fuse_npes: int,
    fuse_topk: int,
    fuse_mtpr: int,
    sort_block_m: int,
    num_dispatch_cu: int,
    num_prepare_cu: int,
    num_quant_cu: int,
    quant_cu_capacity: int,
    model_dim: int,
    payload_chunk_rows: int,
    payload_tile_ready: bool,
    tile_state_stride: int,
    fanout_masks: tuple[int, ...],
    runtime_fanout: bool = False,
    dynamic_fanout: bool = False,
):
    """Compile compact count/group/plan without the GEMM1 shared footprint."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(
            f"MegaMoE prepare requires CDNA4 (gfx95x), got {arch or 'unknown'}"
        )
    npes = int(fuse_npes)
    epr = int(experts_per_rank)
    topk = int(fuse_topk)
    mtpr = int(fuse_mtpr)
    rank = int(rank)
    tile_m = int(sort_block_m)
    dispatch_blocks = int(num_dispatch_cu)
    prepare_blocks = int(num_prepare_cu)
    quant_blocks = int(num_quant_cu)
    quant_cu_capacity = int(quant_cu_capacity)
    model_dim = int(model_dim)
    num_waves = 8
    chunk_rows = int(payload_chunk_rows)
    tile_state_stride = int(tile_state_stride)
    total_experts = npes * epr
    fanout_enabled = bool(fanout_masks) or runtime_fanout
    total_segments = total_experts + (npes if fanout_enabled else 0)
    block_threads = num_waves * 64
    assert prepare_blocks >= 1
    launch_grid = prepare_blocks + quant_blocks + 1
    assert dispatch_blocks % npes == 0
    assert 0 <= quant_blocks <= quant_cu_capacity
    assert model_dim % 32 == 0
    assert chunk_rows > 0 and payload_tile_ready
    assert tile_state_stride > 0

    @fx.struct
    class SharedStorage:
        ticket: fx.Array[fx.Int64, 1, 8]
        count_scratch: fx.Array[fx.Int32, total_segments, 16]

    kernel_name = (
        f"megamoe_prepare_compact_m{tile_m}_dcu{dispatch_blocks}_pcu{prepare_blocks}_pc{chunk_rows}"
        f"_qcu{quant_blocks}qcap{quant_cu_capacity}"
        f"_fov{int(fanout_enabled)}r{int(runtime_fanout)}"
        f"_dyn{int(dynamic_fanout)}"
        f"_tss{tile_state_stride}_v13"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def kernel(
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_idx: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_quant_in: fx.Int64,
        addr_quant_out: fx.Int64,
        addr_quant_scale: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_scratch = fx.recast_iter(fx.Int64, lds.ticket.ptr)
        count_scratch = lds.count_scratch.ptr
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)

        def disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        entry_slot = prepare_blocks * (quant_cu_capacity + 1) + quant_blocks
        entry_count = disp_ptr(DispatchSlot.PREP_ENTRY_COUNT) + fx.Int64(entry_slot * 8)
        epoch_gate = disp_ptr(DispatchSlot.PREP_EPOCH_GATE) + fx.Int64(entry_slot * 4)
        work_head = disp_ptr(DispatchSlot.WORK_HEAD)
        work_tail = disp_ptr(DispatchSlot.WORK_TAIL)
        ready_tile_tail = disp_ptr(DispatchSlot.READY_TILE_TAIL)
        payload_ready_rows = disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)
        local_hist = disp_ptr(DispatchSlot.LOCAL_HIST)

        if tid == fx.Int32(0):
            ticket64 = fx.Int64(comm_ops.atomic_add_agent(entry_count, fx.Int64(1)))
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid))
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        owner = ticket == fx.Int32(0)
        producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(prepare_blocks))
        producer_slot = ticket - fx.Int32(1)
        quant_producer = ticket > fx.Int32(prepare_blocks)
        quant_slot = ticket - fx.Int32(prepare_blocks + 1)

        if const_expr(quant_blocks > 0):  # noqa: SIM102 - preserve DSL staging
            if quant_producer:
                quant_in = buffer_ops.create_buffer_resource_from_addr(addr_quant_in)
                quant_out = buffer_ops.create_buffer_resource_from_addr(addr_quant_out)
                quant_scale = buffer_ops.create_buffer_resource_from_addr(
                    addr_quant_scale
                )
                scale_dim = model_dim // 32
                total_groups = i32_cur_tok * fx.Int32(scale_dim)
                group_stride = fx.Int32(quant_blocks * block_threads)
                group0 = quant_slot * fx.Int32(block_threads) + tid
                for group_id in range(group0, total_groups, group_stride):
                    emit_per_1x32_mx_fp8_group(
                        quant_in, quant_out, quant_scale, group_id
                    )

        # Quant CTAs are independent specialists.  They must retire as soon as
        # their quant groups drain instead of entering prepare's epoch/barrier
        # protocol.  The queued quant CTAs can then immediately occupy CUs
        # released by owner/group CTAs.
        if not quant_producer:
            if owner:
                next_parity_lane = fx.Int32(0)
                if tid == fx.Int32(0):
                    old_parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
                    next_parity_lane = old_parity ^ fx.Int32(1)
                    previous_expected = comm_ops.load_i32_system(
                        addr_expected,
                        next_parity_lane,
                    )
                    next_expected = previous_expected + fx.Int32(npes)
                    comm_ops.store_i32_system(
                        addr_expected,
                        next_parity_lane,
                        next_expected,
                    )
                next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(
                        payload_ready_rows, fx.Int32(0), fx.Int32(tile_m)
                    )
                    comm_ops.fence_system_release()
                fx.barrier()
                if tid == fx.Int32(0):
                    work_head_rsrc = _make_buffer_from_addr(work_head, fx.Int32)
                    for work_shard in range(8):
                        _buffer_store(
                            work_head_rsrc,
                            fx.Int32(work_shard * 16),
                            fx.Int32(0),
                            fx.Int32,
                        )
                    comm_ops.store_i32_system(work_tail, fx.Int32(0), fx.Int32(0))
                    # Payload publishers reserve queue slots with a system-scope
                    # atomic.  Reset the tail in that same coherence domain so a
                    # layout transition cannot append after the previous epoch.
                    comm_ops.store_i32_system(ready_tile_tail, next_parity, fx.Int32(0))
                hist_rsrc = _make_buffer_from_addr(local_hist, fx.Int32)
                for segment in range(tid, total_segments, block_threads):
                    _buffer_store(hist_rsrc, segment, fx.Int32(0), fx.Int32)
                fx.rocdl.s_waitcnt(0)
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(
                        addr_parity,
                        fx.Int32(0),
                        next_parity,
                    )
                    comm_ops.store_i32_system(epoch_gate, fx.Int32(0), gate_epoch)
            else:
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(epoch_gate, gate_epoch)
                    comm_ops.fence_agent_acquire()
                fx.barrier()

            # The owner publishes parity from lane 0.  Converge the complete
            # CTA before any lane consumes it; otherwise the owner lanes can
            # classify and derive offsets with different epochs.
            fx.barrier()
            comm_ops.fence_system_acquire()
            parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
            expected = comm_ops.load_i32_system(addr_expected, parity)
            group_phase_base = fx.Int32(generation) * fx.Int32(prepare_blocks * 2)
            if owner:
                emit_dispatch_plan(
                    num_waves=num_waves,
                    fz_npes=npes,
                    fz_epr=epr,
                    fz_k=topk,
                    fz_mtpr=mtpr,
                    fz_rank=rank,
                    fz_tile_m=tile_m,
                    fz_total_experts=total_experts,
                    addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok,
                    addr_in_idx=addr_in_idx,
                    parity=parity,
                    expected=expected,
                    dispatch_blocks=dispatch_blocks,
                    group_blocks=prepare_blocks,
                    group_done_slot=entry_slot,
                    group_phase_base=group_phase_base,
                    payload_chunk_rows=chunk_rows,
                    payload_tile_ready=True,
                    tile_state_stride=tile_state_stride,
                    fanout_masks=fanout_masks,
                    runtime_fanout=runtime_fanout,
                    dynamic_fanout=dynamic_fanout,
                )
            if producer:
                emit_dispatch_group(
                    num_waves=num_waves,
                    fz_npes=npes,
                    fz_k=topk,
                    fz_epr=epr,
                    fz_total_experts=total_experts,
                    addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok,
                    addr_in_idx=addr_in_idx,
                    dispatch_blocks=prepare_blocks,
                    group_done_slot=entry_slot,
                    producer_slot=producer_slot,
                    parity=parity,
                    expected=expected,
                    fanout_masks=fanout_masks,
                    runtime_fanout=runtime_fanout,
                    count_scratch=count_scratch,
                )

    @flyc.jit
    def launch(
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_idx: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_quant_in: fx.Int64,
        addr_quant_out: fx.Int64,
        addr_quant_scale: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            addr_disp,
            i32_cur_tok,
            addr_in_idx,
            addr_parity,
            addr_expected,
            addr_quant_in,
            addr_quant_out,
            addr_quant_scale,
        ).launch(
            grid=(launch_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def run_mega_moe_prepare(
    addr_disp,
    i32_cur_tok,
    addr_in_idx,
    addr_parity,
    addr_expected,
    addr_quant_in,
    addr_quant_out,
    addr_quant_scale,
    stream,
    **kwargs,
):
    launch = compile_mega_moe_prepare(**kwargs)
    _run_compiled(
        launch,
        addr_disp,
        i32_cur_tok,
        addr_in_idx,
        addr_parity,
        addr_expected,
        addr_quant_in,
        addr_quant_out,
        addr_quant_scale,
        stream,
    )


def preload_mega_moe_prepare(
    addr_disp,
    i32_cur_tok,
    addr_in_idx,
    addr_parity,
    addr_expected,
    addr_quant_in,
    addr_quant_out,
    addr_quant_scale,
    stream,
    **kwargs,
):
    """Compile and load one prepare variant without dispatching it."""
    launch = compile_mega_moe_prepare(**kwargs)
    return launch.preload(
        addr_disp,
        i32_cur_tok,
        addr_in_idx,
        addr_parity,
        addr_expected,
        addr_quant_in,
        addr_quant_out,
        addr_quant_scale,
        stream,
    )
